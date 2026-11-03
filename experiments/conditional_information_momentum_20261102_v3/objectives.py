"""One original forward, genuine tower pair gradients and one optimizer/EMA step."""
import torch
from torch.nn import functional as F
from .boundary_sampler import sample, forward_logits
from .protocol import update_ema
from .vendor.ema02.objectives import train_step as source_step, counts


def train_step(model,optimizer,ema,x,y,row_ids,config,context,diagnostic=False):
    if config['arm'] in ('A0','A1') or config['lambda_pair']==0:
        c=dict(config)
        if c['arm'] not in ('A0','A1'):c['arm']='A0'
        return source_step(model,optimizer,ema,x,y,row_ids,c,context,diagnostic=diagnostic)
    optimizer.zero_grad()
    probability,logits=forward_logits(model,x)
    real=F.binary_cross_entropy(probability,y,reduction='mean')
    pair=sample(x[-1].detach().cpu().numpy(),y.detach().cpu().numpy(),optimizer.demand,
                config['dynamic'],context,row_ids)
    loss=real; r=None; error=None
    if pair is not None:
        pos=torch.as_tensor(pair['pos'],device=x.device);neg=torch.as_tensor(pair['neg'],device=x.device)
        margin=logits[pos,pair['task']]-logits[neg,pair['task']]
        auxiliary=F.softplus(-margin).mean();error=float(torch.sigmoid(-margin).mean().detach())
        loss=real+config['lambda_pair']*float(pair['w'][pair['index']]/pair['q'][pair['index']])*auxiliary
        if optimizer.mode!='adam':
            r=[g.detach() for g in torch.autograd.grad(auxiliary,[p for _,p in optimizer.shared],retain_graph=True)]
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite supervised loss')
    loss.backward()
    if not bool(torch.stack([torch.isfinite(p.grad).all() for p in optimizer.parameters if p.grad is not None]).all()):
        raise FloatingPointError('Nonfinite parameter gradient')
    before=[p.detach().clone() for p in optimizer.parameters] if diagnostic else None
    optimizer.step(pair,r);update_ema(ema,model,config['rho'])
    if pair is not None and config['dynamic']:
        s=pair['selected'];optimizer.demand[s]=.9*optimizer.demand.get(s,.5)+.1*error
    work=counts(len(y),0);work['pair_rows']=4096 if pair else 0
    work['shared_autograd_calls']=int(pair is not None and optimizer.mode!='adam')
    info=dict(pair_count=2048 if pair else 0,active_clocks=dict(optimizer.active),selected_counts=dict(optimizer.selected),
              pair_clock=optimizer.pair_clock,demand=dict(optimizer.demand))
    if pair is not None:
        info.update({k:pair[k] for k in ('active','selected','task','unique_pair_rows','pair_index_hash')})
        info.update(w=pair['w'].tolist(),q=pair['q'].tolist(),w_over_q=(pair['w']/pair['q']).tolist(),
                    alpha_over_q=(.01/pair['q']).tolist(),pair_loss=float(auxiliary.detach()),ranking_error=error)
    if diagnostic:
        info['update_norm']=float(sum((p-old).double().square().sum() for p,old in zip(optimizer.parameters,before)).sqrt())
        info['history_norm']={str(i):float(sum(h.double().square().sum() for h in hs).sqrt()) for i,hs in optimizer.history.items()}
    return dict(loss=float(loss.detach()),real_loss=float(real.detach()),pred=probability.detach(),diagnostics=info,counts=work)
