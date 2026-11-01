"""Small formula oracles and the reviewed real-model CUDA Stage0."""
import copy
import math
from pathlib import Path
import numpy as np
import torch
from experiments.gradient_aug_momentum_20261101.checks import (
    Toy, close, formula_checks as historical_formula_checks, semantic_checks as historical_semantic_checks,
    run_checks as historical_cuda_checks)
from experiments.gradient_aug_20261031_v2.views import make_model
from experiments.gradient_aug_momentum_20261101.objectives import split_gradients
from experiments.gradient_aug_momentum_20261101.optimizer import SADMomentum
from .optimizer import AIGMomentum, parameter_groups
from .protocol import (protocol, candidates, make_optimizer, make_shadows, update_shadows,
    snapshot, restore, setup, cpu_copy, equal, rng_state, state_digest, write_json, digest)
from .objectives import losses, train_step
from .analyze import compare, choose


def formula_checks():
    records=[]
    for dtype in (torch.float32,torch.float64):
        theta=torch.tensor([.2,-.4],dtype=dtype);param=torch.nn.Parameter(theta.clone())
        opt=AIGMomentum([param],['E'],weight_decay=.03,lambda_max=.5,thresholds={'E':.2,'P':.4})
        m=torch.zeros_like(theta);q=m.clone();v=m.clone();error=0.
        for step in range(1,5):
            g=torch.tensor([.1*step,-.07],dtype=dtype)
            d=torch.tensor([-.03,.02*step],dtype=dtype) if step!=3 else torch.zeros_like(g)
            u=float((d-q).double().norm())/(float(q.double().norm())+float(g.double().norm())+1e-12)
            lam=.5/(1+math.exp(-4*(u-.2)))
            r=g+.03*theta;m=.9*m+.1*r;q=.95*q+.05*d;v=.999*v+.001*(r+lam*d).square()
            theta=theta-.01*(m/(1-.9**step)+lam*q/(1-.95**step))/(torch.sqrt(v/(1-.999**step))+1e-8)
            diagnostic=opt.step([g],[d],diagnose=True)
            error=max(error,close(param,theta,dtype),abs(diagnostic[0]['lambda_t']-lam))
            close(opt.state[param]['q_delta'],q,dtype)
        records.append(dict(dtype=str(dtype),case='dynamic_gate_coupled_decay_hand_oracle',max_abs_error=error))
        left=[torch.nn.Parameter(torch.tensor([.2,-.4],dtype=dtype)) for _ in range(4)]
        right=[torch.nn.Parameter(p.detach().clone()) for p in left]
        opt=AIGMomentum(left,['E','P','U','P'],weight_decay=.03,lambda_min=.5,lambda_max=.5)
        ref=SADMomentum(right,weight_decay=.03,beta_delta=.95,lambda_aug=.5)
        for step in range(3):
            gr=[torch.tensor([.1,-.07],dtype=dtype),None,torch.zeros(2,dtype=dtype),None]
            dg=[torch.tensor([-.02,.04],dtype=dtype) if step!=1 else None,
                torch.tensor([.03,-.01],dtype=dtype),None,None]
            opt.step(gr,dg);ref.step(gr,dg)
            for a,b in zip(left,right):close(a,b,dtype)
        assert 'q_delta' not in opt.state[left[2]] and not opt.state[left[3]]
        # U receives standard Adam even while the separated target path is active.
        refp=torch.nn.Parameter(left[2].detach().clone());ordinary=torch.optim.Adam([refp],lr=.01,weight_decay=.03,foreach=False,fused=False)
        up=torch.nn.Parameter(refp.detach().clone());uo=AIGMomentum([up],['U'],weight_decay=.03)
        for step in range(3):
            grad=torch.tensor([.03*step,-.04],dtype=dtype);refp.grad=grad.clone();ordinary.step();uo.step([grad],[None]);assert torch.equal(up,refp)
        records.append(dict(dtype=str(dtype),case='constant_SAD_None_zero_U_Adam',steps=3))
    p=protocol();c=candidates('S4','movielens',p,thresholds={'q50':{'E':0.,'P':0.},'q75':{'E':0.,'P':0.}})['M1']
    model=Toy();x=torch.tensor([[1.,2.,3.,4.],[0.,1.,2.,0.]]);y=torch.tensor([[0.,1.],[1.,0.],[1.,1.],[0.,0.]])
    for change in ({'lambda_max':0.},{'use_augmentation':False,'use_aig_m':False},{'noise_scale':0.}):
        config=dict(c,**change);before=rng_state();_,info=losses(model,x,y,0,config,'movielens',2022,1)
        assert not info['views'] and equal(before,rng_state())
        assert type(make_optimizer(model,p,config,'movielens')) is torch.optim.Adam
    base=candidates('S3','movielens',p)['S95'];base['beta_delta']=0.
    assert type(make_optimizer(model,p,base,'movielens')) is torch.optim.Adam
    assert not make_shadows(model,dict(c,use_shadow_ema=False))
    return dict(status='PASS',records=records,disabled_components_no_view=True,beta0_sentinel=True)


def decision_checks():
    def row(auc=.7,ll=.5):
        return dict(macro_auc=auc,macro_logloss=ll,worst_cell_auc=.6,d2t0_auc=.7,d2t1_auc=.7,
                    d1t0_auc=.7,d1t1_auc=.7,pooled_task_mean_auc=.7,sample_weighted_cell_auc=.7)
    baseline={s:row() for s in (2022,2024)};candidate={s:row(.702) for s in baseline}
    assert compare(candidate,baseline,'movielens')['passed']
    assert not compare({s:row(.702,.503) for s in baseline},baseline,'movielens')['passed']
    assert not compare({2022:row(.704),2024:row(.699)},baseline,'movielens')['passed']
    results=[]
    for trial,auc in [('A',.7004),('B',.699),('C',.7),('D',.699)]:
        for seed in baseline:results.append(dict(dataset='movielens',trial=trial,seed=seed,
            selected_validation={k:row(auc) for k in ('ema995','H2','H8','H20')}))
    chosen=choose('S1',results,'movielens');assert chosen['trial']=='C' and chosen['kind']=='ema995'
    return dict(status='PASS',two_seed_gate=True,guards=True,frozen_near_tie_order=True)


def trajectory(bundle,dataset,p,device,enabled,real=False):
    setup(2022,device)
    thresholds={'q50':{'E':.1,'P':.2},'q75':{'E':.1,'P':.2}}
    c=candidates('S4',dataset,p,thresholds=thresholds)['M2'];c['use_shadow_ema']=enabled
    model=make_model(bundle,p['datasets'][dataset]).to(device) if real else Toy().to(device)
    optimizer=(make_optimizer(model,p,c,dataset) if real else
               AIGMomentum(model.parameters(),['E','P','U'],weight_decay=c['weight_decay'],thresholds=c['thresholds']))
    shadows=make_shadows(model,c)
    loaders={k:torch.Generator().manual_seed(2022+v) for k,v in p['rng']['loader_offsets'].items()}
    target=p['datasets'][dataset]['augmented_domain'] if real else 0
    saved=None;last_view=None;shadow_error=0.
    for step in range(1,4):
        order=torch.randperm(len(bundle.train_y) if real else 4,generator=loaders['train']).numpy()
        ids=order[:40960]
        if real:
            x=torch.as_tensor(bundle.train_x[ids],device=device).T;y=torch.as_tensor(bundle.train_y[ids],device=device)
        else:
            x=torch.tensor([[1.,2.,3.,4.],[0.,1.,2.,0.]],device=device)[:,ids]
            y=torch.tensor([[0.,1.],[1.,0.],[1.,1.],[0.,0.]],device=device)[ids]
        old={k:cpu_copy(v.state_dict()) for k,v in shadows.items()} if step==1 else {}
        _,info,_,_=train_step(model,optimizer,x,y,target,c,dataset,2022,step)
        update_shadows(shadows,model,c);last_view=info['views'][0]['hash']
        if step==1:
            for key,shadow in shadows.items():
                for name,value in shadow.named_parameters():
                    rho=c['shadow_rates'][key]
                    shadow_error=max(shadow_error,close(value,old[key][name].to(device)*rho+model.state_dict()[name]*(1-rho),torch.float32))
                for name,value in shadow.named_buffers():assert torch.equal(value,model.state_dict()[name])
        if step==2:
            saved=snapshot(model,optimizer,shadows,loaders,1,2,{}, {}, {}, c,device,{'optimizer_steps':2})
    final=cpu_copy((model.state_dict(),optimizer.state_dict(),rng_state(device),{k:g.get_state() for k,g in loaders.items()}))
    if enabled:
        expected_shadows=cpu_copy({k:v.state_dict() for k,v in shadows.items()})
        restore(model,optimizer,shadows,loaders,saved,c,device)
        restored_order=torch.randperm(len(bundle.train_y) if real else 4,generator=loaders['train']).numpy()
        assert np.array_equal(restored_order,order)
        if real:
            ids=restored_order[:40960];x=torch.as_tensor(bundle.train_x[ids],device=device).T;y=torch.as_tensor(bundle.train_y[ids],device=device)
        _,info,_,_=train_step(model,optimizer,x,y,target,c,dataset,2022,3);update_shadows(shadows,model,c)
        assert info['views'][0]['hash']==last_view
        assert equal(final,cpu_copy((model.state_dict(),optimizer.state_dict(),rng_state(device),{k:g.get_state() for k,g in loaders.items()})))
        assert equal(expected_shadows,{k:v.state_dict() for k,v in shadows.items()})
    return final,dict(shadow_formula_max_abs_error=shadow_error,peak_memory_bytes=torch.cuda.max_memory_allocated(device) if real else None)


def restoration_checks():
    p=protocol();left,_=trajectory(None,'movielens',p,'cpu',False);right,_=trajectory(None,'movielens',p,'cpu',True)
    assert equal(left,right)
    return dict(status='PASS',four_shadows_noninterference=True,full_RAM_restore_next_step_equal=True)


def run_checks(bundles,p,device,out):
    out=Path(out)
    write_json(out/'formula_checks.json',formula_checks());write_json(out/'decision_checks.json',decision_checks())
    write_json(out/'checkpoint_restore_checks.json',restoration_checks())
    # Reuse the frozen source real-model/gradient/Adam/SAD checks rather than duplicate them.
    from experiments.gradient_aug_momentum_20261101.protocol import protocol as old_protocol
    source=historical_cuda_checks(bundles,old_protocol(),device,out/'source_checks')
    records=[]
    for dataset,bundle in bundles.items():
        setup(2022,device);conf=p['datasets'][dataset];model=make_model(bundle,conf).to(device)
        c=candidates('S3',dataset,p)['S95'];x=torch.as_tensor(bundle.train_x[:40960],device=device).T
        y=torch.as_tensor(bundle.train_y[:40960],device=device)
        _,info=losses(model,x,y,conf['augmented_domain'],c,dataset,2022,1)
        real,delta,_=split_gradients(info,tuple(model.parameters()))
        groups=parameter_groups(model,conf['augmented_domain']);unrelated=[]
        for (name,_),group,d in zip(model.named_parameters(),groups,delta):
            if group=='U':assert d is None or bool((d==0).all()),name
            if group=='U':unrelated.append(name)
        write_json(out/f'{dataset}_parameter_groups.json',dict(zip((n for n,_ in model.named_parameters()),groups)))
        del model,info,real,delta,x,y
        torch.cuda.reset_peak_memory_stats(device)
        without,_=trajectory(bundle,dataset,p,device,False,real=True)
        with_shadow,details=trajectory(bundle,dataset,p,device,True,real=True)
        assert equal(without,with_shadow)
        records.append(dict(dataset=dataset,status='PASS',U_delta_exactly_zero=True,U_parameters=len(unrelated),
            four_shadow_noninterference=True,RAM_restore_next_step_equal=True,actual_batch_size=40960,
            actual_optimizer_updates_including_source=19,test_evaluated=False,**details))
        write_json(out/'cuda_check_progress.json',dict(datasets=records))
        del without,with_shadow
    return dict(status='PASS',datasets=records,source_checks=source,test_evaluated=False,storage='RAM_ONLY')
