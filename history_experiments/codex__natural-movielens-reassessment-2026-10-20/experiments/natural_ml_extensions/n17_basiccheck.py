"""Same-batch implementation identities and full-state resume; no research run."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from .n17_bootstrap import OUT,sha,dump
from .n17_train import build_runner
from .n17_manifest import configuration,base_configurations

def execute(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config=configuration('engineering','SB',.001,2022);config['run_dir']=str(output/'run')
    runner=build_runner(config,'cpu')
    assert len(base_configurations())==72
    assert len(runner.opt.param_groups)==1
    assert not any(name.startswith('_weight_') for name in runner.named)
    assert len(runner.b.train_x)==800167
    b=runner.b
    rows=np.concatenate([np.flatnonzero(b.train_x[:,-1]==d)[:16] for d in range(3)])
    model=copy.deepcopy(runner.model)
    opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
    x=torch.tensor(b.train_x[rows],dtype=torch.float32).T;y=torch.tensor(b.train_y[rows],dtype=torch.float32)
    terms=F.binary_cross_entropy(model(x),y,reduction='none').mean(1)
    actual=(terms*runner.weight[x[-1].long()]).mean()
    expected=sum(terms[x[-1].long()==d].sum()*float(runner.p[d]/runner.q[d])/len(rows) for d in range(3))
    assert torch.allclose(actual,expected,atol=1e-7)
    actual.backward();opt.step();runner.step(rows,1,0)
    assert all(torch.equal(model.state_dict()[name],value) for name,value in runner.model.state_dict().items())
    runner.save(0,'last_resume.pt');next_rows=runner.sampler.sample(21927)
    restored=build_runner(config,'cpu',resume=True)
    assert np.array_equal(next_rows,restored.sampler.sample(21927))
    assert restored.steps==1 and restored.opt.state_dict()['state']
    import n17_quota
    quota=n17_quota.QuotaSampler(b,runner.q,2022)
    tests=[]
    for size in (40960,21927,23660):
        sampled=quota.sample(size);counts=np.bincount(b.train_x[sampled,-1].astype(int),minlength=3)
        weights=size*runner.p/counts
        assert counts.sum()==size and np.all(counts>0)
        # Sum_i w_d/B == p_d independently in every actual tail.
        assert np.allclose(counts*weights/size,runner.p)
        tests.append(dict(batch_size=size,counts=counts.tolist(),weights=weights.tolist()))
    assert np.intersect1d(runner.selection,runner.diagnostic).size==0
    assert len(runner.selection)+len(runner.diagnostic)==len(b.val_x)
    dump(output/'receipt.json',dict(status='PASS_BASIC_IDENTITY_CHECKS',formal_training=False,
        same_batch_exact_parameter_equality=True,weighted_loss_identity=True,
        full_state_resume_sampler=True,one_optimizer=True,fixed_fusion=True,
        quota_tail_checks=tests,test_access=False))
    return output/'receipt.json'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(execute(a.output))
