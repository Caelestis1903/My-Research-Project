"""P4 direct four-measurement additive estimator versus six-cell measurement."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from n18_common import OUT,bundle,model_and_state,flat_grad,timer,dump
from n18_probe import cell_gradients

def marginals(model,b,rows,device):
    x=torch.tensor(b.train_x[rows],device=device,dtype=torch.float32).T
    y=torch.tensor(b.train_y[rows],device=device,dtype=torch.float32)
    loss=F.binary_cross_entropy(model(x),y,reduction='none')
    terms=[loss[x[-1]==d].mean() for d in range(3)]
    terms.append((loss[:,1]-loss[:,0]).mean())
    values=[flat_grad(v,model,i<3) for i,v in enumerate(terms)]
    return torch.stack([values[d]+sign*.5*values[3] for d in range(3) for sign in [-1,1]])

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--device',required=True);a=p.parse_args()
    torch.set_num_threads(2);device=torch.device(a.device);torch.cuda.set_device(device)
    b=bundle();model,_,_=model_and_state(b,a.seed,device)
    source=OUT/'measurements_functional'/f'seed{a.seed}';raw=np.load(source/'P4_gram.npz')
    lookup={int(v):i for i,v in enumerate(b.train_row_ids)}
    rows=[np.array([lookup[int(v)] for v in raw[f'{key}_row_ids']]) for key in ['A','B']]
    # Warm up one forward, then a single paired timing; no benchmark grid.
    with torch.no_grad():model(torch.tensor(b.train_x[rows[0][:4]],device=device,dtype=torch.float32).T)
    full=[];reduced=[];full_s=0.;reduced_s=0.
    # Alternate order by seed to reduce systematic warmup/order bias.
    for r in rows:
        current={}
        for method in (['full','marginal'] if a.seed%2==0 else ['marginal','full']):
            start=timer(device)
            current[method]=cell_gradients(model,b,r,'train',device) if method=='full' else marginals(model,b,r,device)
            dt=timer(device)-start
            if method=='full':full_s+=dt
            else:reduced_s+=dt
        full.append(current['full']);reduced.append(current['marginal'])
    from n18_readout import cell_operators
    P,_=cell_operators();projection=torch.tensor(P,device=device,dtype=torch.float32)
    residual=max(float((projection@f-g).norm()/f.norm()) for f,g in zip(full,reduced))
    assert residual<1e-4,residual
    dump(source/'P4_direct_cost.json',dict(seed=a.seed,status='COMPLETE',full_training_gradient_calls=12,
        marginal_training_gradient_calls=8,full_seconds=full_s,marginal_seconds=reduced_s,
        time_ratio=reduced_s/full_s,relative_projection_identity_error=residual,
        estimator='3 domain averages + 1 global task contrast; 4 independent measurements per batch',
        interaction_measurements=0,timing_repeats=1,kind='exploratory single paired timing; no general throughput claim'))
    print('COMPLETE',a.seed,flush=True)

if __name__=='__main__':main()
