"""P2-A same-theta projected shadow-state interventions; no live theta update."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd
import torch
from v2_20261016 import OUT,sha,dump


def update(g,m,v,step,lr=.01):
    mm=.9*m+.1*g;vv=.999*v+.001*g*g
    delta=-lr*(mm/(1-.9**step))/(np.sqrt(vv/(1-.999**step))+1e-8)
    return delta,mm,vv


def verify():
    p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=torch.float64));opt=torch.optim.Adam([p],lr=.01)
    m=np.array([.02,-.04]);v=np.array([.003,.001]);g=np.array([-.1,.3]);step=7
    opt.state[p]=dict(step=torch.tensor(float(step-1)),exp_avg=torch.tensor(m),exp_avg_sq=torch.tensor(v))
    original=p.detach().numpy().copy();p.grad=torch.tensor(g);opt.step();expected,_,_=update(g,m,v,step)
    error=float(np.max(np.abs(p.detach().numpy()-original-expected)));assert error<1e-12
    dump(OUT/'shadow_engineering.json',dict(status='PASS',nonzero_state_torch_adam_error=error,source_sha256=sha(__file__)))


def main(run):
    c=json.loads((run/'config.json').read_text());assert c['optimizer']=='adam' and c['arm']=='SN'
    seed=c['seed'];probe=OUT/'probes'/c['dataset']/'adam'/f'seed{seed}'
    target=OUT/'analysis'/'shadow'/c['dataset']/f'seed{seed}';target.mkdir(parents=True,exist_ok=False)
    coordinates=np.load(probe/'projection_coordinates.npz',allow_pickle=False)
    alpha=np.asarray(c['alpha']);q_b=np.ones(3)/3;records=[];raw={}
    for key,filename in [('initial','epoch00.pt'),('selected','selected_state.pt'),('final','epoch50.pt')]:
        state=torch.load(run/filename,map_location='cpu',weights_only=False);params=state['optimizer_parameter_names'];index={n:i for i,n in enumerate(params)}
        with np.load(probe/(key+'_gradient_pool.npz'),allow_pickle=False) as z:pools={g:z[g] for g in ('C','P','R','E')}
        rng=np.random.default_rng(seed*1000+1811);uniforms=rng.random((129,64,2));current={}
        for block,names in c['partitions'].items():
            selected=coordinates[block];theta=np.concatenate([state['model'][n].numpy().reshape(-1) for n in names])[selected].astype(float)
            moment={}
            for field in ('exp_avg','exp_avg_sq'):
                if state['epoch']==0:moment[field]=np.zeros(len(selected))
                else:moment[field]=np.concatenate([state['optimizer']['state'][index[n]][field].numpy().reshape(-1) for n in names])[selected].astype(float)
            m_n=np.tile(moment['exp_avg'],(64,1));m_b=m_n.copy();v_n=np.tile(moment['exp_avg_sq'],(64,1));v_b=v_n.copy()
            step0=state['steps'];pool=pools[block]
            for t in range(129):
                u=uniforms[t];j=np.floor(u[:,1]*8).astype(int)
                d_n=np.searchsorted(np.cumsum(alpha),u[:,0],side='right');d_b=np.searchsorted(np.cumsum(q_b),u[:,0],side='right')
                g_n=pool[d_n,j]+1e-5*theta;g_b=pool[d_b,j]*(alpha[d_b]/q_b[d_b])[:,None]+1e-5*theta
                step=step0+t+1
                if t<128:
                    _,m_n,v_n=update(g_n,m_n,v_n,step,c['lr']);_,m_b,v_b=update(g_b,m_b,v_b,step,c['lr']);continue
                delta_n,post_mn,post_vn=update(g_n,m_n,v_n,step,c['lr']);delta_b,post_mb,post_vb=update(g_b,m_b,v_b,step,c['lr'])
                common_m=(post_mn+post_mb)/2;common_v=(post_vn+post_vb)/2
                def from_moments(m,v):return -c['lr']*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
                shared_full_n,_,_=update(g_n,(m_n+m_b)/2,(v_n+v_b)/2,step,c['lr'])
                shared_full_b,_,_=update(g_b,(m_n+m_b)/2,(v_n+v_b)/2,step,c['lr'])
                conditions={'current_gradient_only':(-c['lr']*g_n,-c['lr']*g_b),
                    'shared_m':(from_moments(common_m,post_vn),from_moments(common_m,post_vb)),
                    'shared_v':(from_moments(post_mn,common_v),from_moments(post_mb,common_v)),
                    'shared_full_prior':(shared_full_n,shared_full_b),'full_shadow_history':(delta_n,delta_b)}
                reference=float(np.linalg.norm((delta_b-delta_n).mean(0)))
                for name,(one,two) in conditions.items():
                    difference=two-one;dist=np.linalg.norm(difference,axis=1);mean_norm=float(np.linalg.norm(difference.mean(0)))
                    records.append(dict(snapshot=key,epoch=state['epoch'],block=block,condition=name,
                        mean_update_difference_norm=mean_norm,mean_paired_distance=float(dist.mean()),median_paired_distance=float(np.median(dist)),
                        relative_to_full_history=mean_norm/reference if reference>0 else None,replicates=64,history_steps=128,source_step=step0))
                    raw[key+'_'+block+'_'+name]=difference
            assert np.min(v_n)>=0 and np.min(v_b)>=0
        dump(target/(key+'_state_receipt.json'),dict(state_sha256=sha(run/filename),theta_source=str(run/filename),
            foreign_trajectory_state_loaded=False,theta_updated=False,projection_sha256=sha(probe/'projection_coordinates.npz')))
    pd.DataFrame(records).to_csv(target/'counterfactual_updates.csv',index=False);np.savez_compressed(target/'paired_update_vectors.npz',**raw)
    dump(target/'receipt.json',dict(status='COMPLETE',seed=seed,source_sha256=sha(__file__),same_theta=True,
        scope='finite pool coordinate projections and diagnostic single-domain32-row draw, not production full-batch estimator or AUC intervention',
        mediator_definition='shared_m/v fix realized post-current components to paired mean; shared_full_prior fixes past state only',
        causal_performance_identified=False))
    print('Same-theta shadow readouts complete',seed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--check',action='store_true');args=p.parse_args()
    if args.check:verify()
    else:main(args.run)
