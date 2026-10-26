"""Six-cell graph-gradient support audit for the actual CPRE partition."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import v3_audit_20261016 as audit
base=audit.base

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    a.run_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    b=audit.bundle('kuairand',a.seed);model=base.fixed_model(b,a.seed);named=dict(model.named_parameters())
    chosen=np.concatenate([np.flatnonzero(b.train_x[:,-1]==d)[:16] for d in range(3)])
    x=torch.tensor(b.train_x[chosen]).T;y=torch.tensor(b.train_y[chosen]);domains=x[-1].long()
    rows=[];support={n:[] for n in named}
    for task in range(2):
        for domain in range(3):
            cell=task*3+domain
            loss=F.binary_cross_entropy(model(x)[domains==domain,task],y[domains==domain,task])
            gradients=torch.autograd.grad(loss,list(named.values()),allow_unused=True)
            for (name,param),gradient in zip(named.items(),gradients):
                norm=0. if gradient is None else float(gradient.double().norm());active=norm>0
                if active:support[name].append(cell)
                rows.append(dict(seed=a.seed,domain=domain,task=task,cell=cell,parameter=name,block=base.group(name),
                    gradient_norm=norm,nonzero_coordinates=0 if gradient is None else int(torch.count_nonzero(gradient)),coordinates=param.numel()))
                if name.startswith(('gate.','tower.')):
                    own=int(name.split('.')[1]);assert not active or own==cell
                if name.startswith(('slot_weight.','slot_bias.')):
                    own=int(name.split('.')[1]);assert not active or own==domain
    base.csv_append(a.run_dir/'parameter_cell_support.csv',rows)
    base.dump(a.run_dir/'receipt.json',dict(status='PASS',seed=a.seed,router_parameters_cell_private=True,
        tower_parameters_cell_private=True,slot_parameters_domain_private_task_shared=True,
        expert_block='fixed nonzero mixing across cells; inspect per-parameter measured support',
        parameter_tensor_support=support,coordinate_sparsity_not_equivalent_to_tensor_sharing=True,
        source_hashes={'support':base.sha(__file__),'model':base.sha(audit.ROOT/'experiments/stage_S1_models.py')},
        artifact_hashes={p.name:base.sha(p) for p in a.run_dir.iterdir() if p.is_file()}))

if __name__=='__main__':main()
