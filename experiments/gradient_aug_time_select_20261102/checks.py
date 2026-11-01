"""Focused formula checks and real-batch CUDA acceptance checks."""
import copy
from pathlib import Path
import numpy as np
import torch
from experiments.gradient_aug_momentum_20261101.checks import Toy
from experiments.gradient_aug_20261031_v2.views import make_model
from experiments.gradient_aug_20261031_v2.objectives import losses as source_losses
from .protocol import (protocol,config,lambda_for,optimizer,setup,cpu_copy,equal,rng_state,state_digest,
    snapshot,restore,write_json,parameter_groups)
from .objectives import rank_probabilities,project_task,losses,train_step
from .shadows import ShadowController,replay_sequence
from .analyze import compare


def formula_checks():
    records=[]
    for dtype in (torch.float32,torch.float64):
        g=[torch.tensor([1.,0.],dtype=dtype),torch.tensor([0.,2.],dtype=dtype),None]
        d=[torch.tensor([-1.,2.],dtype=dtype),torch.tensor([1.,1.],dtype=dtype),None]
        kept,rows=project_task(g,d,['E','P','U'])
        torch.testing.assert_close(kept[0],torch.tensor([0.,2.],dtype=dtype),atol=2e-12,rtol=1e-5)
        assert torch.equal(kept[1],d[1]) and kept[2] is None
        halfway,_=project_task(g,d,['E','P','U'],.5)
        torch.testing.assert_close(halfway[0],torch.tensor([-.5,2.],dtype=dtype))
        zero,_=project_task([torch.zeros(2,dtype=dtype)],[d[0]],['E']);assert torch.equal(zero[0],d[0])
        records.append(dict(dtype=str(dtype),case='task_group_projection',projected_dot=rows[0]['projected_dot']))
    p=rank_probabilities(torch.tensor([3.,1.,1.]));torch.testing.assert_close(p,torch.tensor([.05,.725,.725]))
    assert rank_probabilities(torch.tensor([7.])).item()==.5
    ctl=protocol();off=ctl['random_off_epochs']
    for arm in ('A1','B3','R3'):
        assert abs(sum(lambda_for(config(arm,'movielens'),e,off) for e in range(1,51))-25)<1e-12
    assert len(off)==5 and len(set(off))==5
    setup(2022);model=Toy();c=config('B1','movielens');opt=optimizer(model,c)
    x=torch.tensor([[1.,2.,3.,4.],[0.,1.,2.,0.]]);y=torch.tensor([[0.,1.],[1.,0.],[1.,1.],[0.,0.]])
    before=cpu_copy(rng_state());xc=x.clone();yc=y.clone()
    _,info=losses(model,x,y,0,c,'movielens',2022,1,.5)
    assert equal(before,rng_state()) and torch.equal(x,xc) and torch.equal(y,yc)
    _,random_info=losses(model,x,y,0,config('R1','movielens'),'movielens',2022,1,.5,info['mask_record'])
    assert random_info['mask_record']['counts']==info['mask_record']['counts']
    np.testing.assert_allclose(random_info['mask_record']['energies'],info['mask_record']['energies'],rtol=2e-6,atol=1e-12)
    # Task sum/B followed by one task mean equals the original B*2 target delta.
    parameters=tuple(model.parameters());real=info['real'];aug=info['augmented'][0];target=info['mask']
    original=torch.autograd.grad((aug.sum()-real[target].sum())/real.numel(),parameters,retain_graph=True,allow_unused=True)
    taskwise=[torch.autograd.grad((aug[:,t].sum()-real[target,t].sum())/len(y),parameters,
        retain_graph=t==0,allow_unused=True) for t in range(2)]
    for param,old,a,b in zip(parameters,original,*taskwise):
        if old is not None:torch.testing.assert_close(old,((torch.zeros_like(param) if a is None else a)+(torch.zeros_like(param) if b is None else b))/2)
    before=cpu_copy(rng_state());_,empty=losses(model,x,y,99,c,'movielens',2022,1,.5)
    assert not empty['views'] and equal(before,rng_state())
    c=config('A1','movielens');ctl=ShadowController(model,c,replay=[.9]*200)
    assert len(ctl.shadows)==3 and len({id(v) for v in ctl.shadows.values()})==3
    before=state_digest(model.state_dict());rng=cpu_copy(rng_state())
    for step in range(1,201):ctl.step(model,float(step%7+1),step,1)
    assert before==state_digest(model.state_dict()) and equal(rng,rng_state())
    assert ctl.rhos[98]==c['rho_fixed'] and c['rho_min']<=ctl.rhos[99]<=c['rho_max']
    assert sorted(replay_sequence(ctl.rhos))==sorted(ctl.rhos)
    delayed=ShadowController(model,config('A0','movielens'))
    delayed.epoch_end(model,5);assert equal(delayed.shadows['delayed5'].state_dict(),model.state_dict())
    assert 'swa20' not in delayed.valid(30) and 'delayed5' not in delayed.valid(5)
    with torch.no_grad():
        for p in model.parameters():p.add_(1.)
    first=cpu_copy(model.state_dict());delayed.epoch_end(model,31)
    with torch.no_grad():
        for p in model.parameters():p.add_(2.)
    delayed.epoch_end(model,32)
    for name,p in delayed.shadows['swa20'].named_parameters():torch.testing.assert_close(p,first[name]+1.)
    # Full controller/Adam/RNG/loader RAM restoration is checked on this small source model.
    c=config('A1','movielens');setup(2022);model=Toy();opt=optimizer(model,c);ctl=ShadowController(model,c)
    loaders={'train':torch.Generator().manual_seed(13022)}
    _,_,_,counts=train_step(model,opt,x,y,0,c,'movielens',2022,1,.5);ctl.step(model,counts['gradnorm'],1,1)
    saved=snapshot(model,opt,ctl.shadows,ctl,loaders,1,1,{}, {}, {},c,'cpu',{})
    torch.randperm(4,generator=loaders['train']);_,_,_,counts=train_step(model,opt,x,y,0,c,'movielens',2022,2,.5)
    ctl.step(model,counts['gradnorm'],2,1)
    expected=cpu_copy((model.state_dict(),opt.state_dict(),{k:s.state_dict() for k,s in ctl.shadows.items()},ctl.state(),rng_state(),loaders['train'].get_state()))
    restore(model,opt,ctl.shadows,ctl,loaders,saved,c,'cpu');torch.randperm(4,generator=loaders['train'])
    _,_,_,counts=train_step(model,opt,x,y,0,c,'movielens',2022,2,.5);ctl.step(model,counts['gradnorm'],2,1)
    assert equal(expected,cpu_copy((model.state_dict(),opt.state_dict(),{k:s.state_dict() for k,s in ctl.shadows.items()},ctl.state(),rng_state(),loaders['train'].get_state())))
    def row(auc=.7,ll=.5):return dict(macro_auc=auc,macro_logloss=ll,worst_cell_auc=.6,d2t0_auc=.7,d2t1_auc=.7)
    base={s:row() for s in (2022,2024)}
    assert compare({s:row(.702) for s in base},base,'movielens')['passed']
    assert not compare({s:row(.702,.503) for s in base},base,'movielens')['passed']
    assert not compare({2022:row(.704),2024:row(.699)},base,'movielens')['passed']
    return dict(status='PASS',records=records,rank_ties_singleton=True,mask_count_energy_match=True,
        mask_RNG_clean_unchanged=True,empty_domain_no_view=True,task_B_then_mean_normalization=True,
        shadow_nonfeedback=True,adaptive_replay_values=True,
        delayed_and_SWA_validity=True,RAM_restore_next_step_exact=True,lambda_budget=True,decision_gates=True)


def real_checks(bundle,dataset,p,device,out):
    conf=p['datasets'][dataset];target=conf['augmented_domain'];setup(2022,device)
    model=make_model(bundle,conf).to(device);initial=cpu_copy(model.state_dict())
    x=torch.as_tensor(bundle.train_x[:40960],device=device).T;y=torch.as_tensor(bundle.train_y[:40960],device=device)
    xc=x.clone();yc=y.clone();torch.cuda.reset_peak_memory_stats(device);checks=[];updates=0
    for arm in ('A0','A1'):
        c=config(arm,dataset);model.load_state_dict(initial);opt=optimizer(model,c)
        reference=copy.deepcopy(model);refopt=optimizer(reference,c)
        lam=lambda_for(c,1,p['random_off_epochs'])
        scalar,source=source_losses(reference,x,y,target,dict(num_views=int(c['augmentation_enabled']),view_family='gaussian',pairing='iid'),
            dataset,2022,1,mix_lambda=lam,kappa=c['noise_scale'])
        scalar.backward();refopt.step();_,info,_,counts=train_step(model,opt,x,y,target,c,dataset,2022,1,lam);updates+=2
        assert equal(model.state_dict(),reference.state_dict()) and equal(opt.state_dict(),refopt.state_dict())
        checks.append(dict(case=arm+'_source_Adam_exact',status='PASS',gradient_calls=counts['gradient_calls']))
        del reference,refopt,info,source,scalar,opt
    model.load_state_dict(initial);c=config('B1',dataset);before=cpu_copy(rng_state(device))
    _,info=losses(model,x,y,target,c,dataset,2022,1,.5)
    assert equal(before,rng_state(device)) and torch.equal(x,xc) and torch.equal(y,yc)
    record=info['mask_record'];assert len(record['counts'])==len(model.perturb_slices)
    _,other=losses(model,x,y,target,config('R1',dataset),dataset,2022,1,.5,record)
    assert record['counts']==other['mask_record']['counts']
    np.testing.assert_allclose(record['energies'],other['mask_record']['energies'],rtol=2e-6,atol=1e-12)
    checks.append(dict(case='B1_R1_real_count_energy_and_RNG',status='PASS',source_counts=record['counts'],
        source_energies=record['energies'],control_energies=other['mask_record']['energies']))
    del info,other
    model.load_state_dict(initial);c=config('B2',dataset);opt=optimizer(model,c);ctl=ShadowController(model,c)
    loaders={k:torch.Generator().manual_seed(2022+v) for k,v in p['rng']['loader_offsets'].items()}
    _,info,projection,counts=train_step(model,opt,x,y,target,c,dataset,2022,1,.5);updates+=1
    assert counts['gradient_calls']==5 and len(projection)==4
    for row in projection:
        assert row['projected_dot']>=-1e-8-max(row['clean_norm']*row['delta_norm'],1e-8)*1e-5
    ctl.step(model,counts['gradnorm'],1,1);del info
    saved=snapshot(model,opt,ctl.shadows,ctl,loaders,1,1,{}, {}, {},c,device,{})
    order=torch.randperm(len(bundle.train_y),generator=loaders['train']).numpy();ids=order[:40960]
    nx=torch.as_tensor(bundle.train_x[ids],device=device).T;ny=torch.as_tensor(bundle.train_y[ids],device=device)
    _,info,_,counts=train_step(model,opt,nx,ny,target,c,dataset,2022,2,.5);ctl.step(model,counts['gradnorm'],2,1);updates+=1
    vh=info['views'][0]['hash'];del info
    expected=cpu_copy((model.state_dict(),opt.state_dict(),{k:s.state_dict() for k,s in ctl.shadows.items()},ctl.state(),rng_state(device),
        {k:g.get_state() for k,g in loaders.items()}))
    restore(model,opt,ctl.shadows,ctl,loaders,saved,c,device)
    again=torch.randperm(len(bundle.train_y),generator=loaders['train']).numpy();assert np.array_equal(order,again)
    _,info,_,counts=train_step(model,opt,nx,ny,target,c,dataset,2022,2,.5);ctl.step(model,counts['gradnorm'],2,1);updates+=1
    assert vh==info['views'][0]['hash']
    assert equal(expected,cpu_copy((model.state_dict(),opt.state_dict(),{k:s.state_dict() for k,s in ctl.shadows.items()},ctl.state(),rng_state(device),
        {k:g.get_state() for k,g in loaders.items()})))
    checks.append(dict(case='B2_real_task_gradients_U_projection_and_RAM_restore',status='PASS',projection=projection,
        gradient_calls=5,U_delta_exactly_zero=True,full_next_step_equal=True))
    del info,opt,ctl,nx,ny,expected,saved
    # Disabling each algorithm returns the source A1 path without extra backward/RNG.
    for arm,flag in [('B1','difficulty_mask_enabled'),('B2','task_conflict_projection_enabled'),('B3','delayed_augmentation_enabled')]:
        c=config(arm,dataset,**{flag:False});model.load_state_dict(initial);opt=optimizer(model,c)
        reference=copy.deepcopy(model);refopt=optimizer(reference,config('A1',dataset))
        _,info,_,counts=train_step(model,opt,x,y,target,c,dataset,2022,1,.5)
        _,other,_,_=train_step(reference,refopt,x,y,target,config('A1',dataset),dataset,2022,1,.5);updates+=2
        assert equal(model.state_dict(),reference.state_dict()) and counts['gradient_calls']==1
        assert info['views'][0]['hash']==other['views'][0]['hash']
        checks.append(dict(case=arm+'_disabled_exact_A1',status='PASS'))
        del reference,refopt,opt,info,other
    for arm in ('A0','A1'):
        c=config(arm,dataset);ctl=ShadowController(model,c,replay=None if arm=='A0' else [.99]*100)
        before=cpu_copy(model.state_dict());rng=cpu_copy(rng_state(device));ctl.step(model,.5,1,6);ctl.epoch_end(model,31)
        assert equal(before,model.state_dict()) and equal(rng,rng_state(device)) and len(ctl.shadows)==3
    checks.append(dict(case='three_independent_shadows_nonfeedback_and_lambda_budget',status='PASS'))
    answer=dict(dataset=dataset,status='PASS',batch_size=40960,checks=checks,actual_optimizer_updates=updates,
        peak_memory_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),test_evaluated=False)
    write_json(Path(out)/f'{dataset}_stage0.json',answer)
    del model,x,y,ctl;gc_collect_cuda(device)
    return answer


def gc_collect_cuda(device):
    import gc
    gc.collect();torch.cuda.empty_cache()


def run_checks(bundles,p,device,out):
    assert str(device).startswith('cuda') and torch.cuda.is_available()
    answer=formula_checks();write_json(Path(out)/'formula_checks.json',answer)
    rows=[real_checks(bundle,dataset,p,device,out) for dataset,bundle in bundles.items()]
    return dict(status='PASS',datasets=rows,test_evaluated=False,storage='RAM_ONLY')
