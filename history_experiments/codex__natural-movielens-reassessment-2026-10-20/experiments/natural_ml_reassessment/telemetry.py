"""Read-only full-coordinate optimizer observations and aggregate cell metrics."""
import csv
import math
from pathlib import Path
import numpy as np
import torch

def append(path, rows):
    if not rows: return
    path=Path(path);exists=path.exists()
    with path.open('a',newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists: writer.writeheader()
        writer.writerows(rows)

def observe(named, names, groups, optimizer, before, epoch, step, kind):
    sums={g:dict(raw=0.,wd=0.,effective=0.,m=0.,v=0.,delta=0.,none=0,step_min=None,step_max=None,
                 bias1_min=None,bias1_max=None,bias2_min=None,bias2_max=None) for g in set(groups.values())}
    lookup={id(p):pg for pg in optimizer.param_groups for p in pg['params']}
    for name in names:
        p=named[name];g=sums[groups[name]];pg=lookup[id(p)];state=optimizer.state.get(p,{})
        if p.grad is None: g['none']+=1;continue
        raw=p.grad.detach().double();wd=before[name].double()*pg['weight_decay']
        g['raw']+=float(raw.square().sum());g['wd']+=float(wd.square().sum())
        g['effective']+=float((raw+wd).square().sum())
        g['delta']+=float((p.detach().double()-before[name].double()).square().sum())
        for key,field in [('m','exp_avg'),('v','exp_avg_sq')]:
            if field in state: g[key]+=float(state[field].double().square().sum())
        count=int(state.get('step',0))
        b1,b2=pg.get('betas',(0.,0.))
        for key,value in [('step',count),('bias1',1-b1**count),('bias2',1-b2**count)]:
            g[key+'_min']=value if g[key+'_min'] is None else min(g[key+'_min'],value)
            g[key+'_max']=value if g[key+'_max'] is None else max(g[key+'_max'],value)
    rows=[]
    for group,value in sorted(sums.items()):
        if not any(groups[n]==group for n in names):continue
        rows.append(dict(epoch=epoch,batch=step,kind=kind,partition=group,
                         raw_gradient_norm=math.sqrt(value.pop('raw')),
                         weight_decay_gradient_norm=math.sqrt(value.pop('wd')),
                         effective_gradient_norm=math.sqrt(value.pop('effective')),
                         persistent_m_norm=math.sqrt(value.pop('m')),
                         persistent_v_norm=math.sqrt(value.pop('v')),
                         applied_delta_norm=math.sqrt(value.pop('delta')),**value))
    return rows

def aggregates(records, bundle, split, epoch, checkpoint, main_steps, arch_steps):
    pp=np.array([bundle.train_counts[d] for d in range(3)],dtype=float);pp/=pp.sum()
    xx=getattr(bundle,'val_x' if split=='validation' else split+'_x')
    ep=np.bincount(xx[:,-1].astype(int),minlength=3).astype(float);ep/=ep.sum()
    rows=[]
    for task in (-1,0,1):
        for label,weights in [('macro',np.ones(3)/3),('train_p',pp),('eval_p',ep),('target_d2',np.array([0.,0.,1.])),('non_target_macro',np.array([.5,.5,0.]))]:
            tasks=(0,1) if task==-1 else (task,)
            row=dict(split=split,epoch=epoch,checkpoint=checkpoint,task=task,aggregate=label,
                     main_steps=main_steps,arch_steps=arch_steps)
            for field in ('auc','logloss'):
                row[field]=sum(weights[d]*float(records[t*3+d][field])/len(tasks) for t in tasks for d in range(3))
            rows.append(row)
    return rows
