"""Bounded read-only temporal and same-state multiview measurements."""
import copy
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .views import make_model,draw_view
from .protocol import (cpu_copy,equal,rng_state,restore_rng,restore,optimizer_for,
                       append_csv,view_key)
from .vendor_ga31.audited_data import row_ids_hash


def vector(gradients,parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1).cpu()
                      for p,g in zip(parameters,gradients)]).double()


def linear(*terms):
    return tuple(None if all(gs[i] is None for weight,gs in terms) else
                 sum(weight*gs[i] for weight,gs in terms if gs[i] is not None)
                 for i in range(len(terms[0][1])))


def raw_grad(loss,model,retain=False):
    return torch.autograd.grad(loss,list(model.parameters()),retain_graph=retain,allow_unused=True)


def fixed_rows(bundle):
    return np.random.default_rng(314159).permutation(len(bundle.train_y))[:2048]


def temporal_observation(model,opt,loaders,bundle,dataset,seed,epoch,batch,step,device,out,previous=None):
    started=time.perf_counter();saved_rng=rng_state(device)
    before=cpu_copy((model.state_dict(),opt.state_dict(),[p.grad for p in model.parameters()],
                     {k:g.get_state() for k,g in loaders.items()},model.training))
    ids=fixed_rows(bundle)[:1024]
    x=torch.as_tensor(bundle.train_x[ids],device=device).T;y=torch.as_tensor(bundle.train_y[ids],device=device)
    target=0 if dataset=='movielens' else 2;mask=x[-1]==target
    params=list(model.parameters());vectors=[];metadata=[]
    if not bool(mask.any()):raise RuntimeError('Fixed temporal reference lacks target rows')
    clean=F.binary_cross_entropy(model(x[:,mask]),y[mask],reduction='sum')/y.numel()
    gr=raw_grad(clean,model)
    for v in (1,2):
        key=view_key(dataset,seed,'temporal',epoch,'gaussian',v,batch=batch)
        view=draw_view(model,int(mask.sum()),'gaussian',key,device)
        value=F.binary_cross_entropy(model(x[:,mask],view=view),y[mask],reduction='sum')/y.numel()
        ga=raw_grad(value,model);vectors.append(vector(linear((1,ga),(-1,gr)),params))
        metadata.append(dict(key=key,seed=view['seed'],hash=view['hash']))
    d1,d2=vectors;mean=(d1+d2)/2;signal=float(torch.dot(d1,d2))
    cross=float(torch.dot(previous['mean'],mean)) if previous is not None else None
    reason=None if previous is not None and signal>0 and previous['signal']>0 else 'first_step_or_nonpositive_signal'
    normalized=cross/np.sqrt(signal*previous['signal']) if reason is None else None
    row=dict(dataset=dataset,seed=seed,epoch=epoch,batch=batch,global_step=step,target_rows=int(mask.sum()),
        reference_rows=len(ids),row_hash=row_ids_hash(bundle.train_row_ids[ids]),delta1_norm=float(d1.norm()),
        delta2_norm=float(d2.norm()),mean_delta_norm=float(mean.norm()),view_variance=float((d1-d2).square().sum()/2),
        S=signal,C_previous=cross,signal_normalized_C=normalized,normalization_reason=reason,
        key1=metadata[0]['key'],key2=metadata[1]['key'],noise_hash1=metadata[0]['hash'],noise_hash2=metadata[1]['hash'],
        source_immutable=True,extra_gradients=3,extra_optimizer_steps=0,seconds=time.perf_counter()-started)
    after=(model.state_dict(),opt.state_dict(),[p.grad for p in model.parameters()],
           {k:g.get_state() for k,g in loaders.items()},model.training)
    assert equal(before,after) and equal(saved_rng,rng_state(device)), 'Temporal probe mutated training state'
    if out:append_csv(Path(out)/'temporal_probe.csv',row)
    return dict(mean=mean,signal=signal,row=row)


def sample_statistics(draws):
    matrix=torch.stack(draws);mean=matrix.mean(0)
    variance=float((matrix-mean).square().sum()/(len(draws)-1))
    signal=float(mean.square().sum())-variance/len(draws)
    half1=matrix[:2].mean(0);half2=matrix[2:].mean(0)
    denominator=float(half1.norm()*half2.norm())
    return dict(mean_delta_norm=float(mean.norm()),variance=variance,signal=signal,
                half_cosine=float(torch.dot(half1,half2))/denominator if denominator else None,
                half_cosine_reason=None if denominator else 'zero_norm'),mean


def variance_pair(plus,minus):
    plus=torch.stack(plus);minus=torch.stack(minus)
    pc=plus-plus.mean(0);mc=minus-minus.mean(0)
    vp=float(pc.square().sum()/3);vm=float(mc.square().sum()/3);cross=float((pc*mc).sum()/3)
    va=(vp+vm+2*cross)/4;vi=(vp+vm)/4
    pairs=(plus+minus)/2;direct=float((pairs-pairs.mean(0)).square().sum()/3)
    assert va>=-1e-12 and vi>=-1e-12
    assert abs(va-direct)<=1e-10*max(1.,vp,vm)
    return dict(V_plus=vp,V_minus=vm,cross_covariance=cross,V_anti=max(0.,va),V_iid=max(0.,vi),
                paired_direct_variance=direct,variance_rounding_clipped=va<0 or vi<0)


def validation_panels(bundle,device):
    domains=bundle.val_x[:,-1].astype(int);panels=[[],[]]
    shares=[float((domains==d).mean()) for d in range(3)]
    for d in range(3):
        rows=np.flatnonzero(domains==d)
        rows=rows[np.random.default_rng(271828+100*d).permutation(len(rows))]
        n=min(1024,len(rows)//2)
        for panel in range(2):panels[panel].extend(rows[panel*n:(panel+1)*n].tolist())
    assert not set(panels[0]) & set(panels[1])
    result=[]
    for ids in panels:
        ids=np.asarray(ids,dtype=int)
        result.append((torch.as_tensor(bundle.val_x[ids],device=device).T,
                       torch.as_tensor(bundle.val_y[ids],device=device),row_ids_hash(bundle.val_row_ids[ids])))
    return result,shares


def panel_bce(model,x,y,shares):
    with torch.no_grad():loss=F.binary_cross_entropy(model(x),y,reduction='none')
    record={f'd{d}t{t}_bce':float(loss[x[-1]==d,t].mean()) for d in range(3) for t in range(2)}
    record['macro_bce']=float(np.mean(list(record.values())))
    record['population_bce']=sum(shares[d]*record[f'd{d}t{t}_bce']/2 for d in range(3) for t in range(2))
    return record


def run_offline(bundle,dataset,p,result,device,out):
    started=time.perf_counter();saved_rng=rng_state(device);out=Path(out)
    conf=p['datasets'][dataset];config=result['config'];observations=[];batches=0;gradients=0;steps=0;forwards=0
    panels,shares=validation_panels(bundle,device);rows=fixed_rows(bundle)
    try:
        for epoch in dict.fromkeys([1,10,result['selected_epoch']]):
            state=result['states'][epoch];original=cpu_copy(state)
            model=make_model(bundle,conf).to(device);opt=optimizer_for(model,p,config)
            loaders={k:torch.Generator() for k in state['loaders']}
            restore(model,opt,loaders,state,device,restore_random=False);model.eval();params=list(model.parameters())
            before=[panel_bce(model,x,y,shares) for x,y,identity in panels];forwards+=2
            for batch in (1,2):
                ids=rows[(batch-1)*1024:batch*1024];x=torch.as_tensor(bundle.train_x[ids],device=device).T
                y=torch.as_tensor(bundle.train_y[ids],device=device);mask=x[-1]==conf['augmented_domain'];denom=y.numel()
                if not bool(mask.any()):
                    append_csv(out/'probe_unavailable.csv',dict(epoch=epoch,batch=batch,reason='no_target'));continue
                clean_pred=model(x);real=F.binary_cross_entropy(clean_pred,y,reduction='none');forwards+=1
                gr=raw_grad(real.sum()/denom,model,retain=True);gt=raw_grad(real[mask].sum()/denom,model)
                gradients+=2;all_grad=[];all_delta=[];plus=[];minus=[];masked=[]
                base=dict(dataset=dataset,arm=result['arm'],seed=result['seed'],epoch=epoch,global_step=state['global_step'],
                    batch=batch,gate_anchor=epoch in (10,result['selected_epoch']),target_rows=int(mask.sum()),
                    row_hash=row_ids_hash(bundle.train_row_ids[ids]),source_immutable=True)
                plus_views=[]
                for family,sign in [('gaussian',1),('gaussian',-1),('fieldmask',1)]:
                    family_grads=[];delta_vectors=[]
                    for v in range(1,5):
                        key=view_key(dataset,result['seed'],'probe',state['global_step'],family,v,batch=batch)
                        view=draw_view(model,int(mask.sum()),family,key,device,negative_of=plus_views[v-1] if sign==-1 else None)
                        if family=='gaussian' and sign==1:plus_views.append(view)
                        pred=model(x[:,mask],view=view);ll=F.binary_cross_entropy(pred,y[mask],reduction='none')
                        ga=raw_grad(ll.sum()/denom,model);gradients+=1;forwards+=1
                        delta=linear((1,ga),(-1,gt));dv=vector(delta,params)
                        family_grads.append(ga);delta_vectors.append(dv)
                        append_csv(out/'view_probe.csv',dict(**base,family=family,sign=sign,view=v,key=view['key'],noise_seed=view['seed'],
                            view_hash=view['hash'],clean_task0_bce=float(real[mask,0].detach().mean()),
                            clean_task1_bce=float(real[mask,1].detach().mean()),aug_task0_bce=float(ll[:,0].detach().mean()),
                            aug_task1_bce=float(ll[:,1].detach().mean()),prediction_abs_change0=float((pred[:,0]-clean_pred[mask,0]).abs().detach().mean()),
                            prediction_abs_change1=float((pred[:,1]-clean_pred[mask,1]).abs().detach().mean()),delta_norm=float(dv.norm())))
                    all_grad.append(family_grads);all_delta.append(delta_vectors)
                plus,minus,masked=all_delta
                gaussian_stats,_=sample_statistics(plus);mask_stats,_=sample_statistics(masked)
                variance=variance_pair(plus,minus);real_norm_sq=float(vector(gr,params).square().sum())
                valid=variance['V_iid']>1e-24*max(1.,real_norm_sq)
                record=dict(**base,**variance,valid_variance=valid,variance_reason=None if valid else 'numerical_floor',
                    **{'gaussian_'+k:v for k,v in gaussian_stats.items()},**{'mask_'+k:v for k,v in mask_stats.items()})
                append_csv(out/'increment_probe.csv',record);observations.append(record)
                deltas=[linear((1,g),(-1,gt)) for g in [all_grad[0][0],all_grad[0][1],all_grad[1][0],all_grad[2][0]]]
                gp1,gp2,gm1,mask1=deltas
                branches=[('real',gr),('G-single',linear((1,gr),(.5,gp1))),
                          ('G-IID2',linear((1,gr),(.25,gp1),(.25,gp2))),
                          ('G-ANTI2',linear((1,gr),(.25,gp1),(.25,gm1))),
                          ('M-single',linear((1,gr),(.5,mask1)))]
                real_update=None
                for branch,gradient in branches:
                    clone,clone_opt=copy.deepcopy((model,opt));cp=list(clone.parameters())
                    assert all(a is b for a,b in zip(cp,clone_opt.param_groups[0]['params']))
                    prior_m=[v.detach().clone() for v in [clone_opt.state[a].get('exp_avg',torch.zeros_like(a)) for a in cp]]
                    prior_v=[v.detach().clone() for v in [clone_opt.state[a].get('exp_avg_sq',torch.zeros_like(a)) for a in cp]]
                    for a,g in zip(cp,gradient):a.grad=None if g is None else (config['loss_scale']*g).detach().clone()
                    clone_opt.step();steps+=1
                    update=vector([a.detach()-b.detach() for a,b in zip(cp,params)],cp)
                    if real_update is None:real_update=update
                    unorm=float(update.norm());rnorm=float(real_update.norm())
                    mdelta=vector([clone_opt.state[a].get('exp_avg',torch.zeros_like(a))-m for a,m in zip(cp,prior_m)],cp)
                    vdelta=vector([clone_opt.state[a].get('exp_avg_sq',torch.zeros_like(a))-v for a,v in zip(cp,prior_v)],cp)
                    common=dict(**base,branch=branch,update_norm=unorm,update_difference_norm=float((update-real_update).norm()),
                        update_cosine_to_real=float(torch.dot(update,real_update))/(unorm*rnorm) if unorm and rnorm else None,
                        m_change_norm=float(mdelta.norm()),v_change_norm=float(vdelta.norm()),
                        parameter_step_min=min(float(clone_opt.state[a].get('step',0)) for a in cp),
                        parameter_step_max=max(float(clone_opt.state[a].get('step',0)) for a in cp))
                    for panel,(vx,vy,identity) in enumerate(panels,1):
                        after=panel_bce(clone,vx,vy,shares);forwards+=1
                        append_csv(out/'counterfactual_update.csv',dict(**common,panel=panel,panel_row_hash=identity,
                            **{'before_'+k:v for k,v in before[panel-1].items()},**{'after_'+k:v for k,v in after.items()},
                            **{'delta_'+k:after[k]-v for k,v in before[panel-1].items()}))
                    del clone,clone_opt
                batches+=1
            assert equal(state,original),'Offline probe mutated source RAM state'
    finally:restore_rng(saved_rng,device)
    assert equal(saved_rng,rng_state(device))
    return dict(dataset=dataset,arm=result['arm'],seed=result['seed'],observations=observations,batches=batches,
                gradients=gradients,counterfactual_steps=steps,forwards=forwards,seconds=time.perf_counter()-started,
                source_immutable=True,rng_restored=True)
