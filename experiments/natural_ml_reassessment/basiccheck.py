"""Authorized same-batch algorithmic checks. Never calls fit or trains epochs."""
import copy
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from .bootstrap import ROOT, dump, sha, source_hashes
from .manifest import configurations
from .stage2d import Runner, original
from .mechanism import risk_clone

def main():
    output=ROOT/'basicchecks';output.mkdir(parents=True,exist_ok=True)
    checkroot=Path(tempfile.mkdtemp(prefix='same_batch_',dir=output))
    reports=[]
    for arm in ('SN','SB','EN','EB'):
        cfg=next(c.copy() for c in configurations() if c['stage']=='AC-2D' and c['arm']==arm and c['population']=='movielens_natural')
        cfg['run_dir']=str(checkroot/arm);r=Runner(cfg,'cpu');b=r.bundle
        rows=np.concatenate([pool[:7] for pool in r.main_sampler.pools])
        x=torch.tensor(b.train_x[rows]).T;y=torch.tensor(b.train_y[rows])
        reference=copy.deepcopy(r.model);named=dict(reference.named_parameters())
        opt=torch.optim.Adam([named[n] for n in r.main_names],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
        pred=reference(x);bce=F.binary_cross_entropy(pred,y,reduction='none').mean(1)
        # Explicit sum over domains with ACTUAL fixed batch denominator.
        expected=sum(float(r.alpha[d]/r.q[d])*bce[x[-1].long()==d].sum() for d in range(3))/len(rows)
        expected.backward();gradients={n:p.grad.clone() if p.grad is not None else None for n,p in named.items()}
        opt.step();actual,_=r.step(rows,'main',0,0,True)
        loss_error=abs(actual-float(expected.detach()))
        grad_error=max(float((r.named[n].grad-gradients[n]).abs().max()) for n in named if gradients[n] is not None)
        delta_error=max(float((r.named[n]-named[n]).abs().max()) for n in named)
        assert loss_error<1e-6 and grad_error<1e-6 and delta_error<1e-7
        assert not set(r.main_names)&set(r.arch_names)
        clone=risk_clone(r.model)
        with torch.no_grad():assert torch.equal(r.model(x),clone(x))
        for qa in (r.p,np.ones(3)/3):
            a=original.Sampler(b,qa,2022,17);c=original.Sampler(b,qa,2022,17)
            assert np.array_equal(a.sample(40960),c.sample(40960))
        reports.append(dict(arm=arm,fixed_batch_rows=len(rows),loss_max_abs_error=loss_error,
                            raw_gradient_max_abs_error=grad_error,actual_parameter_max_abs_error=delta_error,
                            train_counts=b.train_counts,init_sha=r.init,
                            original_step_source_sha256=sha(original.__file__),full_parameter_partition=r.partition))
    dump(output/'ac2d_same_batch.json',dict(status='PASS',formal_training_epochs_run=0,checks=reports,source_hashes=source_hashes()))
    print(str(output/'ac2d_same_batch.json'))

if __name__=='__main__':main()
