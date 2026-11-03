"""Frozen real/mixed objectives and one optimizer/EMA transition."""
import torch
from torch.nn import functional as F
from .mixing import build_mix_batches,forward_mixed,model_forward,generator
from .protocol import update_ema
from .optimizers import shared_name,norm
from .vendor.v3.vendor.ema02.objectives import train_step as source_step

def real_loss(model,x,y,functional_state=None):
    return F.binary_cross_entropy(model_forward(model,x,functional_state),y,reduction='sum')/y.numel()

def mixed_objective(model,x,y,batches,lambda_mix,functional_state=None):
    real=real_loss(model,x,y,functional_state)
    if lambda_mix==0 or not batches or batches[0].tau==0:return real
    auxiliary=real.new_zeros(())
    for mb in batches:
        prediction=forward_mixed(model,x,mb,functional_state)
        labels=y[mb.anchor_idx.to(y.device),mb.target_task]
        weight=(x[-1]==mb.domain).sum()/y.numel()
        auxiliary=auxiliary+weight*F.binary_cross_entropy(prediction[:,mb.target_task],labels)
    return (1-lambda_mix)*real+lambda_mix*auxiliary

def collect_history_probes(model,x,y,context,batches,mode):
    params=[p for n,p in model.named_parameters() if shared_name(n)];pairs=[]
    for mb in batches:
        idx=mb.anchor_idx.to(x.device)
        loss=F.binary_cross_entropy(model(x[:,idx])[:,mb.target_task],y[idx,mb.target_task])
        a=[g.detach() for g in torch.autograd.grad(loss,params)]
        if mode=='real_second':
            candidates=torch.where(x[-1].detach().cpu()==mb.domain)[0]
            second=candidates[torch.randperm(len(candidates),generator=generator(context,'real_second',mb.domain,mb.target_task))[:len(idx)]].to(x.device)
            loss_b=F.binary_cross_entropy(model(x[:,second])[:,mb.target_task],y[second,mb.target_task])
        else:
            loss_b=F.binary_cross_entropy(forward_mixed(model,x,mb)[:,mb.target_task],y[idx,mb.target_task])
        b=[g.detach() for g in torch.autograd.grad(loss_b,params)]
        pairs.append((a,b))
    return pairs

def train_step(model,optimizer,ema,x,y,row_ids,config,context,diagnostic=False):
    c=config;arm=c['arm'];b=len(y)
    assert x.shape[1]==b and y.shape[1]==2
    if arm=='A1':
        return source_step(model,optimizer,ema,x,y,row_ids,c,dict(context,study='EMA02V2',split_id='original'),diagnostic=diagnostic)
    optimizer.zero_grad();batches=[]
    mixing=c['goal'] in ('conditioned_mix','same_target_mix') and c.get('lambda_mix',0)!=0 and c.get('tau',0)!=0
    mode='unconditional' if arm in ('U_MIX','T_U','H_U') else 'conditioned'
    if mixing:batches=build_mix_batches(x,y,row_ids,context,mode,c['tau'])
    is_h=optimizer.mode in ('history_repair','history_norm_only','current_gradient_repair')
    if is_h:
        start=(context['global_step']-1)%3
        domain=next(d for d in ((start+i)%3 for i in range(3)) if bool((x[-1]==d).any()))
        batches=build_mix_batches(x,y,row_ids,context,mode,c.get('tau',0),domains=[domain])
    work=dict(original_rows=b,training_forwards=0,training_forward_rows=0,probe_forwards=0,
              probe_forward_rows=0,old_point_forwards=0,old_point_forward_rows=0,
              parameter_backward_calls=1,shared_autograd_calls=0,optimizer_steps=1,ema_updates=1)
    phase='training'
    def count(module,args):
        work[phase+'_forwards']+=1;work[phase+'_forward_rows']+=args[0].shape[1]
    handle=model.register_forward_pre_hook(count)
    try:
        prediction=model(x);real=F.binary_cross_entropy(prediction,y,reduction='sum')/y.numel()
        loss=real;info={}
        if mixing:
            auxiliary=real.new_zeros(());weights={}
            for mb in batches:
                out=forward_mixed(model,x,mb);weight=float((x[-1]==mb.domain).sum())/y.numel()
                auxiliary=auxiliary+weight*F.binary_cross_entropy(out[:,mb.target_task],y[mb.anchor_idx.to(y.device),mb.target_task])
                weights[f'd{mb.domain}t{mb.target_task}']=weight
            loss=(1-c['lambda_mix'])*real+c['lambda_mix']*auxiliary
            info.update(mix_loss=float(auxiliary.detach()),cell_weights=weights)
        elif arm=='SCALE':loss=real*(1-c['lambda_scale'])
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite supervised objective')
        loss.backward();pairs=None;old_grads=None;old_parameters=None;preupdate=None
        if is_h:
            phase='probe';pairs=collect_history_probes(model,x,y,context,batches,c['probe'])
            work['shared_autograd_calls']+=2*len(pairs)
        if optimizer.mode.startswith('transport'):
            previous=optimizer.previous_state
            if previous is not None:
                phase='old_point'
                old_parameters={n:p.detach().requires_grad_(shared_name(n)) for n,p in previous[0].items()}
                state=(old_parameters,previous[1])
                old_loss=mixed_objective(model,x,y,batches,c.get('lambda_mix',0),state) if mixing else real_loss(model,x,y,state)
                old_grads=[g.detach() for g in torch.autograd.grad(old_loss,[old_parameters[n] for n,_ in optimizer.shared])]
                work['shared_autograd_calls']+=1
                if diagnostic:info['parameter_distance']=float(norm([p.detach()-old_parameters[n] for n,p in model.named_parameters()]))
            else:info['parameter_distance']=0.
            preupdate=({n:p.detach().clone() for n,p in model.named_parameters()},
                       {n:v.detach().clone() for n,v in model.named_buffers()})
        if not bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()):
            raise FloatingPointError('Nonfinite current gradient')
        for gradients in ([] if pairs is None else [q for pair in pairs for q in pair])+([] if old_grads is None else [old_grads]):
            if not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):raise FloatingPointError('Nonfinite probe/old gradient')
        info.update(optimizer.step(probes=pairs,old_point_data_grads=old_grads,old_parameters=old_parameters))
        if preupdate is not None:optimizer.previous_state=preupdate
        update_ema(ema,model,c['rho'])
        if batches:
            levels=torch.cat([mb.support_level for mb in batches])
            info.update(support_rates={str(k):float((levels==k).float().mean()) for k in (1,2,3)},
                        task_support_rates={str(t):{str(k):float(torch.cat([mb.support_level for mb in batches if mb.target_task==t]).eq(k).float().mean())
                            for k in (1,2,3)} for t in (0,1)},
                        mix_index_hashes=[mb.index_hash for mb in batches])
        return dict(loss=float(loss.detach()),real_loss=float(real.detach()),pred=prediction.detach(),diagnostics=info,counts=work)
    finally:handle.remove()
