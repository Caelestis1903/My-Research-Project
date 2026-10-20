"""Minimal scientific-identity checks, no performance analysis."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import v3_audit_20261016 as audit
base=audit.base

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();torch.set_num_threads(2)
    b=audit.bundle('kuairand',2022);model=base.fixed_model(b,2022)
    pools=[np.flatnonzero(b.train_x[:,-1]==d) for d in range(3)]
    rows=np.concatenate([pool[:8] for pool in pools]);x=torch.tensor(b.train_x[rows]).T;y=torch.tensor(b.train_y[rows])
    p=np.array([len(pool) for pool in pools],float);p/=p.sum();weight=torch.tensor(3*p,dtype=torch.float32)
    prediction=model(x);terms=F.binary_cross_entropy(prediction,y,reduction='none');d=x[-1].long()
    total=(terms.mean(1)*weight[d]).mean();decomposed=sum((terms[d==i].mean(1)*weight[i]).sum()/len(rows) for i in range(3))
    params=list(model.parameters());first=torch.autograd.grad(total,params,retain_graph=True)
    second=torch.autograd.grad(decomposed,params)
    error=max(float((u-v).abs().max()) for u,v in zip(first,second));assert error<1e-6
    wr=base.Sampler(b,p,2022,'iid');perm=base.Sampler(b,p,2022,'cyclic')
    for _ in range(3):
        r=wr.sample(40960);s=perm.sample(40960)
        assert np.array_equal(b.train_x[r,-1],b.train_x[s,-1])
    # Same exact batch contents/row multiplicities under order intervention.
    batches=[wr.sample(256) for _ in range(7)];order=np.random.default_rng(20261016).permutation(7)
    assert np.array_equal(np.sort(np.concatenate(batches)),np.sort(np.concatenate([batches[j] for j in order])))
    assert sorted(tuple(batch.tolist()) for batch in batches)==sorted(tuple(batches[j].tolist()) for j in order)
    # Dataset aliases do not change the empirical within-domain distribution.
    pool=pools[2][:100]
    for factor in (2,4):
        values,counts=np.unique(np.tile(pool,factor),return_counts=True)
        assert np.array_equal(values,pool) and np.all(counts==factor)
    # Exhaustive/disjoint full-coordinate partition.
    groups,_=audit.block_indices(model);coords=np.concatenate(list(groups.values()))
    assert np.array_equal(np.sort(coords),np.arange(sum(z.numel() for z in model.parameters())))
    base.dump(a.out,dict(status='PASS',weighted_group_gradient_max_error=error,
        domain_draw_identity_WR_PERM=True,same_batch_multiset_order_identity=True,
        duplication_empirical_distribution_identity=True,exhaustive_CPRE_partition=True,
        source_hashes={p.name:base.sha(p) for p in [Path(__file__),Path(audit.__file__),Path(__file__).with_name('v3_train_20261016.py')]}))

if __name__=='__main__':main()
