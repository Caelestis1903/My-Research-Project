"""NDP-A: two task-specific full-support views with natural convex weights."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from ..core import forward, sensitivity
from ..protocol import state_digest
from experiments.conditional_information_momentum_20261102_v3.protocol import update_ema


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def summary(value):
    value = value.detach().float()
    quantiles = torch.quantile(value,torch.tensor([.5,.9,.99],device=value.device))
    return dict(count=value.numel(),mean=float(value.mean()),median=float(quantiles[0]),
                p90=float(quantiles[1]),p99=float(quantiles[2]),maximum=float(value.max()))


def make_views(model,x,y,alpha,force_zero=False):
    synchronize(x.device); started = time.perf_counter()
    embedding,logits,gradient = sensitivity(model,x)
    synchronize(x.device); probe_seconds = time.perf_counter()-started; started = time.perf_counter()
    rms = embedding.square().mean(1).sqrt(); deltas = []; diagnostics = {}
    for task in range(2):
        direction = (logits[:,task].sigmoid()-y[:,task])[:,None]*gradient[:,task]
        norm = direction.norm(dim=1); valid = (norm > 1e-10)&(rms > 1e-12)
        delta = alpha*rms[:,None]*direction/(norm[:,None]+1e-12)
        delta = torch.where(valid[:,None],delta,0.).detach()
        if force_zero:
            delta = torch.zeros_like(delta)
        nonzero = delta.norm(dim=1) > 0
        info = dict(nonzero_count=int(nonzero.sum()),zero_count=int((~nonzero).sum()),
            zero_direction_count=int((norm <= 1e-10).sum()),
            zero_rms_count=int(((norm > 1e-10)&(rms <= 1e-12)).sum()),
            forced_zero=bool(force_zero),delta_norm=summary(delta.norm(dim=1)),cells={})
        for domain in range(3):
            mask = x[-1] == domain; count = int(mask.sum())
            info['cells'][f'd{domain}t{task}'] = dict(count=count,zero_count=int((mask&~nonzero).sum()),
                zero_fraction=float((mask&~nonzero).sum())/count if count else None)
        deltas.append(delta); diagnostics[f'task{task}'] = info
    synchronize(x.device)
    return deltas,logits,diagnostics,probe_seconds,time.perf_counter()-started


def losses(model,x,y,deltas,mixture_lambda):
    probabilities = model(x); clean = F.binary_cross_entropy(probabilities,y)
    canonical = all(not torch.count_nonzero(delta) for delta in deltas)
    if canonical:
        return clean,clean,clean,probabilities,None,True
    views = [forward(model,x,delta) for delta in deltas]
    view_loss = .5*(F.binary_cross_entropy(views[0][0][:,0],y[:,0])+
                    F.binary_cross_entropy(views[1][0][:,1],y[:,1]))
    total = (1-mixture_lambda)*clean+mixture_lambda*view_loss
    return total,clean,view_loss,probabilities,views,False


def weight_ledger(x,y,probabilities,views,mixture_lambda):
    B = len(y); denominator = 2*B
    clean = F.binary_cross_entropy(probabilities.detach(),y,reduction='none')
    augmented = (clean if views is None else torch.stack([
        F.binary_cross_entropy(views[t][0].detach()[:,t],y[:,t],reduction='none') for t in range(2)],dim=1))
    cells = {}
    for domain in range(3):
        for task in range(2):
            for label in (0,1):
                mask = (x[-1] == domain)&(y[:,task] == label); count = int(mask.sum())
                natural = count/denominator; clean_sum = float(clean[mask,task].sum())
                view_sum = float(augmented[mask,task].sum())
                cells[f'd{domain}t{task}y{label}'] = dict(count=count,clean_count=count,view_count=count,
                    natural_weight_mass=natural,clean_weight_mass=(1-mixture_lambda)*natural,
                    view_weight_mass=mixture_lambda*natural,total_weight_mass=natural,
                    clean_numerator=clean_sum,view_numerator=view_sum,
                    clean_contribution=(1-mixture_lambda)*clean_sum/denominator,
                    view_contribution=mixture_lambda*view_sum/denominator,
                    total_contribution=((1-mixture_lambda)*clean_sum+mixture_lambda*view_sum)/denominator)
    assert sum(v['count'] for v in cells.values()) == denominator
    assert abs(sum(v['total_weight_mass'] for v in cells.values())-1.) < 1e-12
    for task in range(2):
        assert abs(sum(v['total_weight_mass'] for k,v in cells.items() if k[3] == str(task))-.5) < 1e-12
    return dict(B=B,entries=denominator,clean_numerator=float(clean.sum()),view_numerator=float(augmented.sum()),
        total_numerator=float(((1-mixture_lambda)*clean+mixture_lambda*augmented).sum()),
        denominator=denominator,clean_entry_coefficient=(1-mixture_lambda)/denominator,
        view_entry_coefficient=mixture_lambda/denominator,total_entry_coefficient=1/denominator,
        clean_weight_mass=1-mixture_lambda,view_weight_mass=mixture_lambda,total_weight_mass=1.,cells=cells)


def train_step(model,optimizer,ema,x,y,config,context,row_ids=None,force_zero=False):
    optimizer.zero_grad(set_to_none=True)
    deltas,clean_logits,diagnostics,probe_seconds,construction_seconds = make_views(model,x,y,config['alpha'],force_zero)
    synchronize(x.device); started = time.perf_counter()
    total,clean,view_loss,probabilities,views,canonical = losses(model,x,y,deltas,config['mixture_lambda'])
    assert torch.isfinite(total)
    ledger = weight_ledger(x,y,probabilities,views,config['mixture_lambda'])
    for task in range(2):
        change = torch.zeros_like(clean_logits) if views is None else views[task][1].detach()-clean_logits
        diagnostics[f'task{task}'].update(target_logit_change=summary(change[:,task]),
            non_target_absolute_drift=summary(change[:,1-task].abs()))
    total.backward(); optimizer.step(); update_ema(ema,model,config['rho'])
    synchronize(x.device)
    ids = np.arange(len(y),dtype=np.int64) if row_ids is None else np.asarray(row_ids,dtype=np.int64)
    return dict(loss=float(total.detach()),clean_loss=float(clean.detach()),view_loss=float(view_loss.detach()),
        zero_delta_canonical=canonical,diagnostics=diagnostics,ledger=ledger,row_order_hash=state_digest(ids),
        optimizer_updates=1,ema_updates=1,probe_seconds=probe_seconds,construction_seconds=construction_seconds,
        training_seconds=time.perf_counter()-started)
