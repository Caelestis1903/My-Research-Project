"""Offline, one-step source-gradient probes on joint model/Adam copies."""
import copy
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import make_model
from .objectives import losses
from .audited_data import row_ids_hash
from .protocol import (optimizer,restore,cpu_copy,equal,rng_state,restore_rng,append_csv,arm_config)


def groups(model):
    names=[n for n,_ in model.named_parameters()]
    result={'all':list(range(len(names))),'shared':[],'slot':[],'domain_expert':[],
            'task_expert':[],'gate':[],'tower':[]}
    for i,name in enumerate(names):
        if name.startswith(('embedding.','shared_weight','shared_bias','star_mlp.','skip_conn.','expert.')):
            result['shared'].append(i)
        elif name.startswith(('slot_weight','slot_bias')): result['slot'].append(i)
        else: result[name.split('.')[0]].append(i)
    assert sorted(i for group,ids in result.items() if group!='all' for i in ids)==list(range(len(names)))
    return result


def vector(values,parameters,indices):
    return torch.cat([(torch.zeros_like(parameters[i]) if values[i] is None else values[i]).detach().reshape(-1).cpu()
                      for i in indices]).double()


def geometry(real,aug):
    rn=float(real.norm()); an=float(aug.norm()); delta=aug-real; dn=float(delta.norm())
    dot=float(torch.dot(real,aug))
    orth=delta-real*(float(torch.dot(delta,real))/(rn*rn)) if rn else delta
    return dict(real_norm=rn,aug_norm=an,cosine=dot/(rn*an) if rn and an else None,
                cosine_reason=None if rn and an else 'zero_norm',norm_ratio=an/rn if rn else None,
                delta_norm=dn,delta_orthogonal_norm=float(orth.norm()))


def combine(a,b,weight):
    return tuple(None if x is None and y is None else
                 ((1-weight)*x if y is None else weight*y if x is None else (1-weight)*x+weight*y)
                 for x,y in zip(a,b))


def run_probes(bundle,dataset,p,arm,seed,states,selected_epoch,device,out):
    start=time.perf_counter(); saved_rng=rng_state(device); unchanged=cpu_copy(states)
    conf=p['datasets'][dataset]; scale=arm_config(p,arm)['loss_scale']
    train_indices=np.random.default_rng(p['rng']['probe_train_row_seed']).permutation(len(bundle.train_y))[:2048]
    validation_indices=np.random.default_rng(p['rng']['probe_validation_row_seed']).permutation(len(bundle.val_y))[:4096]
    vx=torch.as_tensor(bundle.val_x[validation_indices],device=device).T
    vy=torch.as_tensor(bundle.val_y[validation_indices],device=device)
    count=0; backwards=0; forwards=0
    try:
        for epoch in dict.fromkeys([1,10,selected_epoch]):
            state=states[epoch]
            model=make_model(bundle,conf).to(device); opt=optimizer(model,p)
            loaders={k:torch.Generator() for k in state['loaders']}
            restore(model,opt,loaders,state,device,restore_random=False); model.eval()
            params=list(model.parameters()); block_groups=groups(model)
            with torch.no_grad():before=float(F.binary_cross_entropy(model(vx),vy)); forwards+=1
            for batch_index in (1,2):
                ids=train_indices[(batch_index-1)*1024:batch_index*1024]
                x=torch.as_tensor(bundle.train_x[ids],device=device).T
                y=torch.as_tensor(bundle.train_y[ids],device=device)
                noise_seed=seed*1000000+100000+2*state['global_step']+batch_index
                _,info=losses(model,x,y,conf['augmented_domain'],noise_seed,enabled=True,mix_lambda=.5)
                real_loss=info['real'].mean(); mask=info['mask']; denom=info['real'].numel()
                if info['augmented'] is None:
                    append_csv(Path(out)/'gradient_probe_unavailable.csv',dict(epoch=epoch,batch=batch_index,reason='no_target'))
                    continue
                target_real=info['real'][mask].sum()/denom
                target_aug=info['augmented'].sum()/denom
                extended_loss=real_loss+target_aug-target_real
                gr=torch.autograd.grad(real_loss,params,retain_graph=True,allow_unused=True)
                ga=torch.autograd.grad(extended_loss,params,retain_graph=True,allow_unused=True)
                gt_r=torch.autograd.grad(target_real,params,retain_graph=True,allow_unused=True)
                gt_a=torch.autograd.grad(target_aug,params,allow_unused=True)
                gm=combine(gr,ga,.5); backwards+=4; forwards+=2
                branch_stats={}; updates={}; moment_deltas={}
                for branch,gradient in [('real',gr),('extended_aug',ga),('mix',gm)]:
                    clone,clone_opt=copy.deepcopy((model,opt))
                    clone_params=list(clone.parameters())
                    assert all(cp is op for cp,op in zip(clone_params,clone_opt.param_groups[0]['params']))
                    before_m=[cpu_copy(clone_opt.state[cp].get('exp_avg',torch.zeros_like(cp))) for cp in clone_params]
                    before_v=[cpu_copy(clone_opt.state[cp].get('exp_avg_sq',torch.zeros_like(cp))) for cp in clone_params]
                    for cp,g in zip(clone_params,gradient):cp.grad=None if g is None else (scale*g).clone()
                    clone_opt.step()
                    updates[branch]=[cp.detach()-original.detach() for cp,original in zip(clone_params,params)]
                    moment_deltas[branch]=(
                        [clone_opt.state[cp].get('exp_avg',torch.zeros_like(cp)).detach().cpu()-m for cp,m in zip(clone_params,before_m)],
                        [clone_opt.state[cp].get('exp_avg_sq',torch.zeros_like(cp)).detach().cpu()-v for cp,v in zip(clone_params,before_v)])
                    with torch.no_grad():after=float(F.binary_cross_entropy(clone(vx),vy)); forwards+=1
                    branch_stats[branch]=dict(validation_before=before,validation_after=after,validation_delta=after-before)
                    del clone,clone_opt
                for block,indices in block_groups.items():
                    raw=geometry(vector(gr,params,indices),vector(ga,params,indices))
                    target=geometry(vector(gt_r,params,indices),vector(gt_a,params,indices))
                    base=dict(dataset=dataset,arm=arm,seed=seed,epoch=epoch,global_step=state['global_step'],
                              batch=batch_index,block=block,noise_seed=noise_seed,
                              train_row_hash=row_ids_hash(bundle.train_row_ids[ids]),
                              validation_row_hash=row_ids_hash(bundle.val_row_ids[validation_indices]),
                              target_rows=int(mask.sum()),source_immutable=True)
                    for key,value in target.items():raw['target_'+key]=value
                    for branch in ('real','extended_aug','mix'):
                        update=vector(updates[branch],params,indices)
                        real_update=vector(updates['real'],params,indices)
                        u=geometry(real_update,update)
                        raw[branch+'_update_norm']=u['aug_norm'];raw[branch+'_update_cosine_to_real']=u['cosine']
                        raw[branch+'_m_delta_norm']=float(vector(moment_deltas[branch][0],params,indices).norm())
                        raw[branch+'_v_delta_norm']=float(vector(moment_deltas[branch][1],params,indices).norm())
                        for key,value in branch_stats[branch].items():raw[branch+'_'+key]=value
                    append_csv(Path(out)/'gradient_probe.csv',dict(**base,**raw))
                    m=[opt.state[param].get('exp_avg',torch.zeros_like(param)) for param in params]
                    v=[opt.state[param].get('exp_avg_sq',torch.zeros_like(param)) for param in params]
                    append_csv(Path(out)/'optimizer_state_stats.csv',dict(**base,
                        m_norm=float(vector(m,params,indices).norm()),v_norm=float(vector(v,params,indices).norm()),
                        min_step=min(float(opt.state[params[i]].get('step',0)) for i in indices),
                        max_step=max(float(opt.state[params[i]].get('step',0)) for i in indices)))
                count+=1
            del model,opt
    finally:
        restore_rng(saved_rng,device)
    assert equal(states,unchanged), 'Probe mutated the formal source snapshot'
    assert equal(saved_rng,rng_state(device)), 'Probe changed global RNG'
    return dict(batches=count,backwards=backwards,forwards=forwards,seconds=time.perf_counter()-start,
                source_immutable=True,rng_restored=True)
