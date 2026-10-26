"""V1 deterministic fixed-subset raw experiment."""
import argparse
import copy
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import v3_audit_20261016 as audit
base=audit.base

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();a.run_dir.mkdir(parents=True,exist_ok=False);device=torch.device(a.device);torch.set_num_threads(2)
    if device.type=='cuda':torch.cuda.set_device(device)
    b=audit.bundle('kuairand',a.seed);p=np.array([b.train_counts[d] for d in range(3)],dtype=float);p/=p.sum()
    pools=[np.flatnonzero(b.train_x[:,-1]==d) for d in range(3)]
    rng=np.random.default_rng(a.seed+20261016);subset=np.concatenate([rng.choice(pool,96,replace=False) for pool in pools])
    b.train_x=b.train_x[subset];b.train_y=b.train_y[subset];b.train_row_ids=b.train_row_ids[subset]
    original=base.fixed_model(b,a.seed).to(device);models={name:copy.deepcopy(original) for name in ('EXACT_SN','EXACT_SB','SN','SB')}
    optimizers={name:torch.optim.SGD(model.parameters(),lr=.1,weight_decay=1e-5) for name,model in models.items()}
    samplers={name:base.Sampler(b,p if name=='SN' else np.ones(3)/3,a.seed+1000000,'iid') for name in ('SN','SB')}
    base.dump(a.run_dir/'config.json',dict(seed=a.seed,p=p.tolist(),subset_row_ids=b.train_row_ids.tolist(),batch_size=96,steps=100,lr=.1,
        objective='sum_d p_d mean_subset_d mean_task BCE',exact_weight=(3*p).tolist(),code_sha=base.sha(__file__),
        init_hash=base.initial_state_hash(original),scope='fixed96rows/domain diagnostic; not population generalization'))
    started=time.time();max_residual=0.
    for step in range(1,101):
        for name,model in models.items():
            exact=name.startswith('EXACT');rows=np.arange(288) if exact else samplers[name].sample(96)
            q=np.ones(3)/3 if exact or name=='SB' else p
            opt=optimizers[name];opt.zero_grad(set_to_none=True)
            g,loss=audit.grad(model,b,rows,torch.tensor(p/q,device=device,dtype=torch.float32),device);opt.step()
            base.csv_append(a.run_dir/'trajectory.csv',[dict(step=step,condition=name,objective_estimate=loss,gradient_norm=float(g.double().norm()))])
        exact_residual=max(float((x-y).abs().max()) for x,y in zip(models['EXACT_SN'].parameters(),models['EXACT_SB'].parameters()))
        max_residual=max(max_residual,exact_residual);assert exact_residual<=1e-6
    distances={}
    for name in models:
        distances[name]=float(torch.cat([(x-y).detach().flatten() for x,y in zip(models[name].parameters(),models['EXACT_SN'].parameters())]).double().norm())
    stochastic=float(torch.cat([(x-y).detach().flatten() for x,y in zip(models['SN'].parameters(),models['SB'].parameters())]).double().norm())
    base.dump(a.run_dir/'receipt.json',dict(status='PASS',max_exact_pair_absolute_difference=max_residual,
        stochastic_pair_distance=stochastic,distance_to_exact=distances,elapsed_seconds=time.time()-started,
        artifact_hashes={x.name:base.sha(x) for x in a.run_dir.iterdir() if x.is_file()}))

if __name__=='__main__':main()
