"""Immediate scalar reductions of innovation and bounded pulse probes."""
import numpy as np
import torch
from torch.nn import functional as F
from experiments.aug_history_20261105.group_registry import GROUPS


def reduce_groups(values,groups):
    # Fixed group and coordinate order avoids CUDA atomic accumulation.
    return torch.stack([values[groups==b].to(torch.float64).sum() for b in range(12)])


@torch.no_grad()
def observe_innovation(theta,phi_f,groups):
    r=theta.to(torch.float64)-phi_f.to(torch.float64);indices=groups.indices(r.device)
    energy=reduce_groups(r.square(),indices);counts=torch.bincount(indices,minlength=12)
    rms=(energy/counts).sqrt();total=energy.sum()
    shares=energy/total if total>0 else torch.full_like(energy,float('nan'))
    return [dict(group_id=name,l2_squared=float(energy[i]),rms=float(rms[i]),
        energy_share=float(shares[i]) if total>0 else None,actual_count=int(counts[i])) for i,name in enumerate(GROUPS)]


def rank_direction_scores(scratch,phi_f,psi_fast,psi_slow,panel_a,layout,groups):
    device=phi_f.device;rank=np.asarray(panel_a['rank_x'])
    x=torch.as_tensor(np.array(rank.reshape(-1,rank.shape[-1])),device=device).T
    indices=groups.indices(device);counts=torch.bincount(indices,minlength=12)
    pairs=torch.as_tensor(panel_a['rank_pairs'],device=device,dtype=torch.long)
    df=psi_fast.to(torch.float64)-phi_f.to(torch.float64);ds=psi_slow.to(torch.float64)-phi_f.to(torch.float64);result={}
    for semantic,full in (('full',True),('source_detach',False)):
        flat=phi_f.detach().clone().requires_grad_(True)
        params={name:flat[left:right].view(shape) for name,shape,(left,right) in zip(layout.names,layout.shapes,layout.offsets)}
        logits=scratch.forward_logits(x,params=params,probe_input_full_graph=full).reshape(6,rank.shape[1],2)
        for c in range(6):
            z=logits[c,:,c%2];pair=pairs[c]
            loss=F.softplus(-(z[pair[:,0]]-z[pair[:,1]])).mean()
            g=torch.autograd.grad(loss,flat,retain_graph=c<5)[0].to(torch.float64)
            fast=-reduce_groups(g*df,indices);slow=-reduce_groups(g*ds,indices)
            rms=(reduce_groups(g.square(),indices)/counts).sqrt()
            for b in range(12):
                row=result.setdefault((c,b),dict(domain=c//2,task=c%2,group_id=GROUPS[b]))
                row[f'P_A_gamma_{semantic}_fastF']=float(fast[b]);row[f'P_A_gamma_{semantic}_slowF']=float(slow[b])
                row[f'P_A_gamma_{semantic}_fastslow']=float(fast[b]-slow[b])
                row[f'P_A_{semantic}_rank_gradient_group_rms']=float(rms[b])
            del g
        del flat,params,logits
    return result


def observe_pulse(theta,phi_f,psi_fast,psi_slow,groups,predictions,panels,readout_rows,m2):
    indices=groups.indices(theta.device);counts=torch.bincount(indices,minlength=12)
    gap=theta.to(torch.float64)-phi_f.to(torch.float64);delta=psi_fast.to(torch.float64)-psi_slow.to(torch.float64)
    e_gap=reduce_groups(gap.square(),indices);e_delta=reduce_groups(delta.square(),indices)
    dots=reduce_groups(gap*delta,indices);rms_gap=(e_gap/counts).sqrt();rms_delta=(e_delta/counts).sqrt()
    lookup={(r['panel_id'],r['readout_id'],r['domain'],r['task']):r for r in readout_rows};result=[]
    global_pref={}
    for panel_id in ('P_A','P_B'):
        global_pref[panel_id]={}
        for variant in ('all_fast','all_slow'):
            global_pref[panel_id][variant]=float(np.mean([lookup[panel_id,variant,d,t]['auc']-lookup[panel_id,'F',d,t]['auc'] for d in range(3) for t in range(2)]))
    for c in range(6):
        d,t=c//2,c%2
        for b,name in enumerate(GROUPS):
            zero=bool(e_gap[b]==0 or e_delta[b]==0)
            row=dict(domain=d,task=t,group_id=name,panel_id='P_A',functional_support=name in ('S',f'D{d}',f'T{t}',f'DT{d}{t}'),
                gap_rms=float(rms_gap[b]),pulse_fastslow_rms=float(rms_delta[b]),
                gap_pulse_cos=None if zero else float(dots[b]/torch.sqrt(e_gap[b]*e_delta[b])),
                cos_missing_flag=int(zero),gap_zero_flag=int(e_gap[b]==0),pulse_zero_flag=int(e_delta[b]==0),
                actual_count=int(counts[b]),**{k:v for k,v in m2[c,b].items() if k not in ('domain','task','group_id')})
            for panel_id in ('P_A','P_B'):
                f=predictions[panel_id]['F'];fast=predictions[panel_id]['fast_'+name];slow=predictions[panel_id]['slow_'+name]
                for contrast,left,right in (('fastF',fast,f),('slowF',slow,f),('fastslow',fast,slow)):
                    diff=left['natural_z'][d,:,t].astype('float64')-right['natural_z'][d,:,t].astype('float64')
                    row[f'{panel_id}_natural_logit_rms_{contrast}']=float(np.sqrt(np.mean(diff**2)))
                base=lookup[panel_id,'F',d,t]
                for speed in ('fast','slow'):
                    value=lookup[panel_id,speed+'_'+name,d,t]
                    row[f'{panel_id}_rank_exact_auc_{speed}F']=value['auc']-base['auc']
                    row[f'{panel_id}_rank_risk_reduction_{speed}F']=base['rank_risk']-value['rank_risk']
                row[f'{panel_id}_rank_six_cell_macro_auc_allfastF']=global_pref[panel_id]['all_fast']
                row[f'{panel_id}_rank_six_cell_macro_auc_allslowF']=global_pref[panel_id]['all_slow']
            result.append(row)
    return result
