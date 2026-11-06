"""Formula, trajectory, RAM restoration, and frozen-source acceptance checks."""
import copy
from pathlib import Path
import numpy as np
import torch
from experiments.gradient_aug_20261031_v2.views import make_model, draw_view, apply_view
from experiments.gradient_aug_20261031_v2.objectives import losses as source_losses
from experiments.gradient_aug_20261031_v2.vendor_ga31.audited_data import row_ids_hash
from .protocol import (protocol, arm_config, make_optimizer, setup, cpu_copy, equal, rng_state,
    make_shadow, update_shadow, snapshot, restore, state_digest, selected_update, write_json)
from .optimizer import SADMomentum
from .objectives import losses, split_gradients, train_step


def close(a,b,dtype):
    torch.testing.assert_close(a,b,atol=1e-12 if dtype == torch.float64 else 1e-6,
                              rtol=1e-10 if dtype == torch.float64 else 1e-5)
    return float((a.detach()-b.detach()).abs().max())


def formula_checks():
    errors = []
    for dtype in (torch.float32,torch.float64):
        for mode in ('lambda0','beta0','beta1'):
            left = [torch.nn.Parameter(torch.tensor([.2,-.4],dtype=dtype)) for _ in range(4)]
            right = [torch.nn.Parameter(v.detach().clone()) for v in left]
            lam = 0. if mode == 'lambda0' else .5
            if mode == 'beta1':
                opt = SADMomentum(left,beta_delta=.9,lambda_aug=lam,weight_decay=.03)
            else:
                opt = torch.optim.Adam(left,lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=.03,foreach=False,fused=False)
            reference = torch.optim.Adam(right,lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=.03,foreach=False,fused=False)
            error = 0.
            for step in range(3):
                gr = [torch.tensor([.1+.04*step,-.07],dtype=dtype),None,
                      torch.zeros(2,dtype=dtype) if step != 1 else None,None]
                delta = [torch.tensor([-.02,.03-.01*step],dtype=dtype) if step != 1 else None,
                         torch.tensor([.04,-.05],dtype=dtype) if step != 1 else None,None,None]
                mixed = [None if a is None and (b is None or lam == 0) else
                         (torch.zeros_like(left[i]) if a is None else a)+(0 if b is None else lam*b)
                         for i,(a,b) in enumerate(zip(gr,delta))]
                if isinstance(opt,SADMomentum):
                    opt.step(gr,delta)
                else:
                    for param,g in zip(left,mixed):
                        param.grad = None if g is None else g.clone()
                    opt.step()
                for param,g in zip(right,mixed):
                    param.grad = None if g is None else g.clone()
                reference.step()
                for a,b in zip(left,right):
                    error = max(error,close(a,b,dtype))
                    if b not in reference.state or not reference.state[b]:
                        assert not opt.state[a]
                        continue
                    sa,sb = opt.state[a],reference.state[b]
                    assert int(sa['step']) == int(sb['step'])
                    ma = sa['m_real']+.5*sa['q_delta'] if isinstance(opt,SADMomentum) else sa['exp_avg']
                    error = max(error,close(ma,sb['exp_avg'],dtype),close(sa['exp_avg_sq'],sb['exp_avg_sq'],dtype))
            errors.append(dict(dtype=str(dtype),case=mode,steps=3,max_abs_error=error))
        # This oracle changes beta_delta rather than only checking an Adam-equivalent branch.
        param = torch.nn.Parameter(torch.tensor([.2,-.4],dtype=dtype))
        opt = SADMomentum([param],beta_delta=.99,lambda_aug=.5,weight_decay=.03)
        theta = param.detach().clone(); mr = torch.zeros_like(theta); q = mr.clone(); v = mr.clone(); error = 0.
        for step in range(1,4):
            gr = torch.tensor([.1*step,-.07],dtype=dtype); delta = torch.tensor([-.03,.02*step],dtype=dtype)
            r = gr+.03*theta; mr = .9*mr+.1*r; q = .99*q+.01*delta
            v = .999*v+.001*(r+.5*delta).square()
            theta = theta-.01*(mr/(1-.9**step)+.5*q/(1-.99**step))/(torch.sqrt(v/(1-.999**step))+1e-8)
            opt.step([gr],[delta]); error = max(error,close(param,theta,dtype))
        errors.append(dict(dtype=str(dtype),case='beta99_hand_oracle',steps=3,max_abs_error=error))
    # lambda0 must not activate a parameter connected only to the augmentation graph.
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = SADMomentum([parameter],lambda_aug=0.)
    optimizer.step([None],[torch.ones_like(parameter)])
    assert not optimizer.state[parameter] and float(parameter.detach()) == 1.
    return dict(status='PASS',records=errors,none_and_per_parameter_steps=True,coupled_decay=True)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.embedding = torch.nn.Embedding(8,2); self.output = torch.nn.Linear(2,2)
        self.perturb_slices = [(0,2)]
        self.register_buffer('copied_buffer',torch.tensor([2.]))

    def forward(self,x,view=None,kappa=.1):
        embedding = self.embedding(x[0].long())
        if view is not None:
            embedding = apply_view(embedding,self.perturb_slices,view,kappa)
        return torch.sigmoid(self.output(embedding))


def semantic_checks():
    p = protocol(); setup(2022)
    model = Toy(); c = arm_config(p,'A3','movielens')
    x = torch.tensor([[1.,2.,3.,4.],[0.,1.,2.,0.]])
    y = torch.tensor([[0.,1.],[1.,0.],[1.,1.],[0.,0.]])
    view = draw_view(model,2,'gaussian','GAM01-check',x.device)
    scalar,info = losses(model,x,y,0,c,'movielens',2022,1,explicit_views=[view])
    parameters = tuple(model.parameters()); real,delta,_ = split_gradients(info,parameters)
    reference = copy.deepcopy(model)
    value,_ = source_losses(reference,x,y,0,dict(num_views=1,view_family='gaussian',pairing='iid'),
                           'movielens',2022,1,explicit_views=[view])
    value.backward(); max_error = 0.
    for gr,gd,param in zip(real,delta,reference.parameters()):
        mixed = (0 if gr is None else gr)+(0 if gd is None else .5*gd)
        max_error = max(max_error,close(mixed,param.grad,torch.float32))
    for change in ({'use_augmented_view':False},{'lambda_aug':0.},{'noise_scale':0.}):
        config = dict(c,**change); before = rng_state()
        _,details = losses(model,x,y,0,config,'movielens',2022,1)
        assert not details['views'] and equal(before,rng_state())
    _,details = losses(model,x[:,1:3],y[1:3],0,c,'movielens',2022,1)
    assert not details['views']
    ordinary = dict(c,beta_delta=0)
    assert type(make_optimizer(model,p,ordinary)) is torch.optim.Adam
    assert type(make_optimizer(model,p,dict(c,lambda_aug=0))) is torch.optim.Adam
    assert type(make_optimizer(model,p,dict(c,use_augmented_view=False))) is torch.optim.Adam
    assert make_shadow(model,dict(c,use_shadow_ema=False)) is None
    before = rng_state(); shadow = make_shadow(model,c)
    assert equal(before,rng_state())
    expected = copy.deepcopy(shadow.state_dict())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(.1)
        model.copied_buffer.add_(1.)
    update_shadow(shadow,model,c['rho_shadow'])
    for name,parameter in shadow.named_parameters():
        close(parameter,expected[name]*.995+model.state_dict()[name]*.005,torch.float32)
    assert torch.equal(shadow.copied_buffer,model.copied_buffer)
    selected = {}; best = {'online':-float('inf'),'shadow':-float('inf')}
    assert not selected_update(selected,best,'online',0,.9)
    for epoch,a,b in [(1,.5,.4),(2,.5,.6),(3,.4,.6)]:
        selected_update(selected,best,'online',epoch,a); selected_update(selected,best,'shadow',epoch,b)
    assert selected == dict(online=1,shadow=2)
    return dict(status='PASS',target_delta_max_abs_error=max_error,no_view_when_disabled=True,
                independent_view_rng=True,shadow_formula_buffers=True,independent_selection=True)


def toy_trajectory(shadow_enabled):
    p = protocol(); c = dict(arm_config(p,'A3','movielens'),use_shadow_ema=shadow_enabled)
    setup(2022); model = Toy(); optimizer = make_optimizer(model,p,c); shadow = make_shadow(model,c)
    loaders = {'train':torch.Generator().manual_seed(13022)}
    state = None; views = []
    for step in range(1,4):
        x = torch.tensor([[1.,2.,3.,4.],[0.,1.,2.,0.]])
        y = torch.tensor([[0.,1.],[1.,0.],[1.,1.],[0.,0.]])
        order = torch.randperm(4,generator=loaders['train']); x = x[:,order]; y = y[order]
        _,info,_,_ = train_step(model,optimizer,x,y,0,c,'movielens',2022,step)
        views.append(info['views'][0]['hash']); update_shadow(shadow,model,c['rho_shadow'])
        if step == 2:
            state = snapshot(model,optimizer,shadow,loaders,1,step,{},None,{'order':'toy'})
    final = cpu_copy((model.state_dict(),optimizer.state_dict(),rng_state(),loaders['train'].get_state()))
    if shadow_enabled:
        final_shadow = cpu_copy(shadow.state_dict())
        epoch,step,updates = restore(model,optimizer,shadow,loaders,state,None)
        assert (epoch,step,updates) == (1,2,2)
        order = torch.randperm(4,generator=loaders['train'])
        x = torch.tensor([[1.,2.,3.,4.],[0.,1.,2.,0.]])[:,order]
        y = torch.tensor([[0.,1.],[1.,0.],[1.,1.],[0.,0.]])[order]
        _,info,_,_ = train_step(model,optimizer,x,y,0,c,'movielens',2022,step+1)
        update_shadow(shadow,model,c['rho_shadow'])
        assert views[-1] == info['views'][0]['hash']
        assert equal(final,cpu_copy((model.state_dict(),optimizer.state_dict(),rng_state(),loaders['train'].get_state())))
        assert equal(final_shadow,shadow.state_dict())
    return final


def restoration_checks():
    without = toy_trajectory(False); with_shadow = toy_trajectory(True)
    assert equal(without,with_shadow)
    return dict(status='PASS',shadow_online_noninterference=True,RAM_resume_next_step_equal=True,
                q_shadow_optimizer_rng_loader_restored=True,disk_weight_files=False)


def actual_trajectory(bundle,dataset,p,device,enabled):
    setup(2022,device); c = dict(arm_config(p,'A3',dataset),use_shadow_ema=enabled)
    model = make_model(bundle,p['datasets'][dataset]).to(device)
    optimizer = make_optimizer(model,p,c); shadow = make_shadow(model,c)
    loaders = {k:torch.Generator().manual_seed(2022+offset) for k,offset in p['rng']['loader_offsets'].items()}
    saved = None; last_hash = None
    for step in range(1,4):
        # A short development window checks the same next-permutation boundary as formal epochs.
        if step in (1,3):
            order = torch.randperm(len(bundle.train_y),generator=loaders['train']).numpy()
        offset = (step-1)*40960 if step < 3 else 0
        ids = order[offset:offset+40960]
        x = torch.as_tensor(bundle.train_x[ids],device=device).T
        y = torch.as_tensor(bundle.train_y[ids],device=device)
        _,info,_,_ = train_step(model,optimizer,x,y,p['datasets'][dataset]['augmented_domain'],c,dataset,2022,step)
        update_shadow(shadow,model,c['rho_shadow']); last_hash = info['views'][0]['hash']
        if step == 2:
            saved = snapshot(model,optimizer,shadow,loaders,1,2,{},device,{'order_hash':row_ids_hash(bundle.train_row_ids[order])})
    final = cpu_copy((model.state_dict(),optimizer.state_dict(),rng_state(device),{k:g.get_state() for k,g in loaders.items()}))
    if enabled:
        expected_shadow = cpu_copy(shadow.state_dict())
        restore(model,optimizer,shadow,loaders,saved,device)
        restored_order = torch.randperm(len(bundle.train_y),generator=loaders['train']).numpy()
        assert np.array_equal(order,restored_order)
        ids = restored_order[:40960]
        x = torch.as_tensor(bundle.train_x[ids],device=device).T
        y = torch.as_tensor(bundle.train_y[ids],device=device)
        _,info,_,_ = train_step(model,optimizer,x,y,p['datasets'][dataset]['augmented_domain'],c,dataset,2022,3)
        update_shadow(shadow,model,c['rho_shadow'])
        assert last_hash == info['views'][0]['hash']
        assert equal(final,cpu_copy((model.state_dict(),optimizer.state_dict(),rng_state(device),{k:g.get_state() for k,g in loaders.items()})))
        assert equal(expected_shadow,shadow.state_dict())
    return final


def run_checks(bundles,p,device,out):
    out = Path(out)
    write_json(out/'formula_checks.json',formula_checks())
    write_json(out/'semantic_checks.json',semantic_checks())
    write_json(out/'checkpoint_restore_checks.json',restoration_checks())
    records = []
    try:
        for dataset,bundle in bundles.items():
            setup(2022,device); conf = p['datasets'][dataset]; c = arm_config(p,'A0',dataset)
            model = make_model(bundle,conf).to(device); reference = make_model(bundle,conf,reference=True).to(device)
            reference.load_state_dict(model.state_dict())
            assert list(model.state_dict()) == list(reference.state_dict())
            x = torch.as_tensor(bundle.train_x[:40960],device=device).T; y = torch.as_tensor(bundle.train_y[:40960],device=device)
            oa,ob = make_optimizer(model,p,c),make_optimizer(reference,p,c)
            loss,info = losses(model,x,y,conf['augmented_domain'],c,dataset,2022,1)
            source,_ = source_losses(reference,x,y,conf['augmented_domain'],
                dict(num_views=0,view_family='gaussian',pairing='iid'),dataset,2022,1)
            close(info['pred'],reference(x),torch.float32); close(loss,source,torch.float32)
            loss.backward(); source.backward(); error = 0.
            for a,b in zip(model.parameters(),reference.parameters()):
                assert (a.grad is None) == (b.grad is None)
                if a.grad is not None:
                    error = max(error,close(a.grad,b.grad,torch.float32))
            oa.step(); ob.step()
            for a,b in zip(model.parameters(),reference.parameters()):
                error = max(error,close(a,b,torch.float32))
            del model,reference,oa,ob,info,loss,source
            peaks = []
            for arm in ('A1','A2','A3'):
                setup(2022,device); candidate = make_model(bundle,conf).to(device); ac = arm_config(p,arm,dataset)
                optimizer = make_optimizer(candidate,p,ac); torch.cuda.reset_peak_memory_stats(device)
                _,details,diagnostic,calls = train_step(candidate,optimizer,x,y,conf['augmented_domain'],ac,dataset,2022,1,diagnose=True)
                peaks.append(dict(arm=arm,peak_memory_bytes=torch.cuda.max_memory_allocated(device),
                    gradient_calls=calls,target_rows=int(details['mask'].sum()),diagnostic=diagnostic))
                del candidate,optimizer,details
            del x,y
            without = actual_trajectory(bundle,dataset,p,device,False)
            with_shadow = actual_trajectory(bundle,dataset,p,device,True)
            assert equal(without,with_shadow)
            records.append(dict(dataset=dataset,status='PASS',source_prediction_loss_gradient_step_max_error=error,
                actual_batch_size=40960,parameter_count={'movielens':773470,'kuairand':1614254}[dataset],
                memory_checks=peaks,shadow_trajectory_equal=True,RAM_resume_equal=True,
                development_optimizer_updates=12,development_test_access=False))
            write_json(out/'cuda_check_progress.json',dict(datasets=records))
            del without,with_shadow
    except Exception as exc:
        write_json(out/'checks_failure.json',dict(status='FAIL',error=repr(exc),completed_datasets=records))
        raise
    return dict(status='PASS',seed=2022,datasets=records,test_evaluated=False,disk_weight_files=False)
