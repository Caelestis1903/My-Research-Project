"""Authorized one-batch fixed-network/SGD oracle, without formal training."""
import copy
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from .bootstrap import ROOT,dump,source_hashes
from .manifest import configurations
from .v9 import Runner

def main():
    root=ROOT/'basicchecks';root.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='v9_same_batch_',dir=root));reports=[]
    for arm in ('SN','SB'):
        cfg=next(c.copy() for c in configurations() if c['stage']=='AC-V9' and c['arm']==arm)
        cfg['run_dir']=str(directory/arm);r=Runner(cfg,'cpu');b=r.b
        rows=np.concatenate([np.flatnonzero(b.train_x[:,-1]==d)[:7] for d in range(3)])
        x=torch.tensor(b.train_x[rows]).T;y=torch.tensor(b.train_y[rows]);reference=copy.deepcopy(r.model)
        opt=torch.optim.SGD(reference.parameters(),lr=.3,momentum=0,weight_decay=1e-5)
        bce=F.binary_cross_entropy(reference(x),y,reduction='none').mean(1)
        loss=sum(float(r.p[d]/r.q[d])*bce[x[-1].long()==d].sum() for d in range(3))/len(rows)
        loss.backward();gradients={n:p.grad.detach().clone() for n,p in reference.named_parameters() if p.grad is not None};opt.step()
        expected_loss=float(loss.detach());weighted,_=r.step(rows,1,0)
        parameter_error=max(float((p.detach()-dict(reference.named_parameters())[n].detach()).abs().max()) for n,p in r.model.named_parameters())
        gradient_error=max(float((r.named[n].grad-gradients[n]).abs().max()) for n in gradients)
        assert parameter_error<1e-7 and gradient_error<1e-6 and abs(weighted/len(rows)-expected_loss)<1e-6
        assert r.opt.param_groups[0]['lr']==.3 and r.opt.param_groups[0]['momentum']==0.
        assert not any(n.startswith('_weight_') for n,p in r.model.named_parameters())
        reports.append(dict(arm=arm,loss_error=abs(weighted/len(rows)-expected_loss),gradient_error=gradient_error,
            actual_sgd_parameter_error=parameter_error,lr=.3,momentum=0.,darts=False,train_counts=b.train_counts,
            validation_selection_rows=len(r.selection),validation_diagnostic_rows=len(r.diagnostic)))
    dump(root/'v9_same_batch.json',dict(status='PASS',formal_training_epochs_run=0,checks=reports,source_hashes=source_hashes(('v9.py',))))
    print(str(root/'v9_same_batch.json'))

if __name__=='__main__':main()
