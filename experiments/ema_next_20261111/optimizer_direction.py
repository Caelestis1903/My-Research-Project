"""Source Adam followed by a coordinate-exact EMBED direction replacement."""
import copy
import torch


def capture_embed_inputs(model, optimizer, layout, mask):
    groups={id(p):g for g in optimizer.param_groups for p in g['params']}
    rows=[]
    for p,(start,end) in zip(model.parameters(),layout.offsets):
        take=mask[start:end].to(p.device).reshape(p.shape)
        if p.grad is None or not bool(take.any()): continue
        if p.grad.is_sparse: raise ValueError('Sparse EMBED gradient is outside source protocol')
        g=groups[id(p)]
        rows.append(dict(parameter=p,mask=take,before=p.detach()[take].double().clone(),
                         gradient=p.grad.detach()[take].double().clone(),group=g))
    return rows


@torch.no_grad()
def apply_matched_direction(pre, optimizer, direction_alpha):
    if not 0<=direction_alpha<=1: raise ValueError('Invalid direction_alpha')
    if not pre: return dict(native_norm=0.,applied_norm=0.,relative_error=0.,cosine=0.,fallback='EMPTY_GROUP')
    actual=[];current=[]
    for row in pre:
        p,g=row['parameter'],row['group'];st=optimizer.state[p];step=int(st['step'])
        q=row['gradient']+g['weight_decay']*row['before']
        vhat=st['exp_avg_sq'][row['mask']].double()/(1-g['betas'][1]**step)
        actual.append(p[row['mask']].double()-row['before'])
        current.append(-g['lr']*q/(vhat.sqrt()+g['eps']))
    a=torch.cat(actual);c=torch.cat(current);r=a.norm();cn=c.norm();fallback=None
    if not bool(torch.isfinite(a).all() and torch.isfinite(c).all()): raise FloatingPointError('Nonfinite direction')
    if direction_alpha==0 or r==0: d=a
    elif cn==0: d=a;fallback='ZERO_CURRENT'
    else:
        w=(1-direction_alpha)*a+direction_alpha*(r/cn)*c
        if w.norm()<=1e-12*r: d=a;fallback='CANCELLATION'
        else: d=w*(r/w.norm())
    if direction_alpha!=0 and r!=0 and fallback is None:
        offset=0
        for row in pre:
            n=row['before'].numel();p=row['parameter']
            p[row['mask']]=(row['before']+d[offset:offset+n]).to(p.dtype);offset+=n
    written=torch.cat([row['parameter'][row['mask']].double()-row['before'] for row in pre])
    error=float((written.norm()-r).abs()/r.clamp_min(1e-12))
    if error>1e-4: raise FloatingPointError(f'FP32 applied direction radius error {error}')
    return dict(native_norm=float(r),applied_norm=float(written.norm()),relative_error=error,
                cosine=float(torch.dot(a,written)/(r*written.norm())) if r and written.norm() else 0.,
                current_cosine=float(torch.dot(a,c)/(r*cn)) if r and cn else 0.,fallback=fallback)


def set_embed_beta1_zero(optimizer, layout, mask, model):
    flags={id(p):mask[a:b] for p,(a,b) in zip(model.parameters(),layout.offsets)}
    result=[]
    for group in optimizer.param_groups:
        embed=[];other=[]
        for p in group['params']:
            m=flags[id(p)]
            if bool(m.any()) and not bool(m.all()):
                raise ValueError('A_RAW requires tensor-aligned EMBED support')
            (embed if bool(m.any()) else other).append(p)
        for params,beta1 in ((other,group['betas'][0]),(embed,0.)):
            if params:
                new={k:copy.deepcopy(v) for k,v in group.items() if k!='params'}
                new.update(params=params,betas=(beta1,group['betas'][1]));result.append(new)
    optimizer.param_groups=result
