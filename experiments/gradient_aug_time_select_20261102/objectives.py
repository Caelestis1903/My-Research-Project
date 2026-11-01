"""Sample-preserving Gaussian masks and task-conditioned delta projection."""
import math
import torch
from torch.nn import functional as F
from experiments.gradient_aug_20261031_v2.objectives import losses as source_losses
from experiments.gradient_aug_20261031_v2.protocol import derive_seed,view_key
from experiments.gradient_aug_20261031_v2.views import draw_view,digest_tensor
from .protocol import parameter_groups


def rank_probabilities(difficulty):
    n=difficulty.numel()
    if n<=1:return torch.full_like(difficulty,.5)
    _,inverse,counts=torch.unique(difficulty.detach(),sorted=True,return_inverse=True,return_counts=True)
    end=counts.cumsum(0);start=end-counts
    ranks=(start+end-1).to(difficulty.dtype)/2
    return (.05+.90*(1-ranks[inverse]/(n-1))).clamp(.05,.95)


def masked_losses(model,x,y,target,c,dataset,seed,step,lam,control=None):
    pred=model(x);real=F.binary_cross_entropy(pred,y,reduction='none');mask=x[-1]==target
    info=dict(pred=pred,real=real,mask=mask,augmented=[],views=[],extra_views=0,forwards=1,mask_record=None)
    if lam==0 or c['noise_scale']==0 or not bool(mask.any()):return real.mean(1).mean(),info
    n=int(mask.sum());fields=len(model.perturb_slices)
    key=f'GATS02|{dataset}|{seed}|{step}|'+('random-mask' if c['random_mask_enabled'] else 'difficulty-mask')
    gen=torch.Generator(device=x.device).manual_seed(derive_seed(key))
    probabilities=rank_probabilities(real[mask].detach().mean(1)) if c['difficulty_mask_enabled'] else None
    if probabilities is not None:
        selected=torch.rand((n,fields),device=x.device,generator=gen)<probabilities[:,None]
    else:
        assert control is not None and len(control['counts'])==fields
        selected=torch.zeros((n,fields),device=x.device,dtype=torch.bool)
        for f,count in enumerate(control['counts']):
            assert 0<=count<=n
            if count:selected[torch.randperm(n,generator=gen,device=x.device)[:count],f]=True
    vkey=view_key(dataset,seed,'train',step,'gaussian',1)
    view=draw_view(model,n,'gaussian',vkey,x.device);energies=[];rms=[];scales=[]
    def apply(module,inputs,embedding):
        output=embedding.clone()
        for f,(lo,hi) in enumerate(model.perturb_slices):
            clean=embedding[:,lo:hi];scale=clean.square().mean(1,keepdim=True).sqrt().detach()
            noise=c['noise_scale']*scale*view['noise'][:,f,:]*selected[:,f,None]
            energy=noise.detach().double().square().sum();factor=1.
            if control is not None:
                wanted=control['energies'][f]
                if wanted==0:factor=0.
                elif float(energy)==0:raise RuntimeError('R1 nonzero source energy with zero control energy')
                else:factor=math.sqrt(wanted/float(energy))
                noise=noise*factor
            output[:,lo:hi]=clean+noise
            energies.append(float(noise.detach().double().square().sum()));rms.append(float(scale.double().square().mean().sqrt()));scales.append(factor)
        return output
    handle=model.embedding.register_forward_hook(apply)
    try:aug=F.binary_cross_entropy(model(x[:,mask]),y[mask],reduction='none')
    finally:handle.remove()
    counts=selected.sum(0).cpu().tolist()
    record=dict(key=key,seed=derive_seed(key),hash=digest_tensor(selected),counts=counts,energies=energies,
        embedding_rms=rms,energy_scale=scales,target_rows=n,field_rate=float(selected.float().mean()),
        p_quantiles=None if probabilities is None else torch.quantile(probabilities,torch.tensor([0.,.25,.5,.75,1.],device=x.device)).cpu().tolist())
    info.update(augmented=[aug],views=[view],extra_views=n,forwards=2,mask_record=record,
        probabilities=probabilities,field_mask=selected)
    return (real.sum()+lam*(aug.sum()-real[mask].sum()))/real.numel(),info


def losses(model,x,y,target,c,dataset,seed,step,lam,control=None):
    if c['difficulty_mask_enabled'] or c['random_mask_enabled']:
        return masked_losses(model,x,y,target,c,dataset,seed,step,lam,control)
    return source_losses(model,x,y,target,dict(num_views=int(c['augmentation_enabled']),view_family='gaussian',pairing='iid'),
        dataset,seed,step,mix_lambda=lam,kappa=c['noise_scale'])


def group_norm(values,groups,group):
    parts=[v.detach().double().square().sum() for v,g in zip(values,groups) if g==group and v is not None]
    return math.sqrt(float(torch.stack(parts).sum())) if parts else 0.


def project_task(clean,delta,groups,eta=1.):
    result=list(delta);records=[]
    for group in ('E','P'):
        pairs=[(g,d) for g,d,tag in zip(clean,delta,groups) if tag==group]
        dots=[(g.detach().double()*d.detach().double()).sum() for g,d in pairs if g is not None and d is not None]
        dot=float(torch.stack(dots).sum()) if dots else 0.
        gn=group_norm(clean,groups,group);dn=group_norm(delta,groups,group)
        coefficient=dot/(gn*gn+1e-12);removed=eta*min(coefficient,0.)
        for i,(g,d,tag) in enumerate(zip(clean,delta,groups)):
            if tag!=group or g is None:continue
            if removed!=0:result[i]=(torch.zeros_like(g) if d is None else d)-removed*g
        kept=group_norm(result,groups,group)
        records.append(dict(group=group,clean_norm=gn,delta_norm=dn,retained_norm=kept,dot=dot,
            cosine=dot/(gn*dn+1e-12),coefficient=coefficient,removed_norm=abs(removed)*gn,
            projected_dot=dot-removed*gn*gn))
    for i,(d,tag) in enumerate(zip(delta,groups)):
        if tag=='U':
            assert d is None or bool((d==0).all()),'Target delta nonzero in U; revise grouping before training'
            result[i]=None if d is None else torch.zeros_like(d)
    return result,records


def combine(real,delta,lam,parameters):
    return [None if g is None and d is None else
        (torch.zeros_like(p) if g is None else g)+(0 if d is None else lam*d)
        for p,g,d in zip(parameters,real,delta)]


def input_gradnorm(parameters):
    values=[p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
    return math.sqrt(float(torch.stack(values).sum())) if values else 0.


def train_step(model,opt,x,y,target,c,dataset,seed,step,lam,control=None):
    opt.zero_grad(set_to_none=True)
    scalar,info=losses(model,x,y,target,c,dataset,seed,step,lam,control)
    if not bool(torch.isfinite(scalar)):raise FloatingPointError(f'Nonfinite objective step{step}')
    params=tuple(model.parameters());records=[];active=bool(info['augmented'])
    if active and (c['task_conflict_projection_enabled'] or c['fixed_shrink_enabled']):
        real=torch.autograd.grad(info['real'].mean(1).mean(),params,retain_graph=True,allow_unused=True)
        groups=parameter_groups(model,target)
        if c['task_conflict_projection_enabled']:
            projected=[]
            for task in range(2):
                cleanloss=info['real'][info['mask'],task].sum()/len(y)
                deltaloss=(info['augmented'][0][:,task].sum()-info['real'][info['mask'],task].sum())/len(y)
                clean=torch.autograd.grad(cleanloss,params,retain_graph=True,allow_unused=True)
                delta=torch.autograd.grad(deltaloss,params,retain_graph=task==0,allow_unused=True)
                kept,rows=project_task(clean,delta,groups,c['eta']);projected.append(kept)
                records.extend(dict(task=task,**r) for r in rows)
            delta=[None if a is None and b is None else
                ((torch.zeros_like(p) if a is None else a)+(torch.zeros_like(p) if b is None else b))/2
                for p,a,b in zip(params,*projected)]
            calls=5
        else:
            deltaloss=(info['augmented'][0].sum()-info['real'][info['mask']].sum())/info['real'].numel()
            raw=torch.autograd.grad(deltaloss,params,allow_unused=True)
            for d,tag in zip(raw,groups):
                if tag=='U':assert d is None or bool((d==0).all())
            delta=[None if d is None else d*c['shrink'].get(tag,0.) for d,tag in zip(raw,groups)];calls=2
        for param,grad in zip(params,combine(real,delta,lam,params)):param.grad=grad
    else:scalar.backward();calls=1
    gradnorm=input_gradnorm(params);opt.step()
    return scalar.detach(),info,records,dict(gradient_calls=calls,optimizer_steps=1,gradnorm=gradnorm)
