"""Minimal numerical and pipeline checks tied to N1 comparisons."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import n17_common as c
from n17_train import Runner

def main():
    results=[]
    for opt,lr in [('sgd',.1),('adam',.001)]:
        directory=c.OUT/'preflight'/('smoke_'+opt)
        a=argparse.Namespace(run_dir=directory,device='cpu',seed=2022,optimizer=opt,lr=lr,arm='SB',phase='engineering',epochs=1,batch_size=64,resume=False,smoke=True)
        r=Runner(a);initial=r.config['init_sha256'];other=c.base.fixed_model(r.b,2022)
        assert c.base.initial_state_hash(other)==initial
        before=copy.deepcopy(r.model);otheropt=(torch.optim.Adam if opt=='adam' else torch.optim.SGD)(before.parameters(),lr=lr,weight_decay=1e-5)
        rows=np.concatenate([pool[:8] for pool in r.sampler.pools]);x=torch.tensor(r.b.train_x[rows],dtype=torch.float32).T;y=torch.tensor(r.b.train_y[rows],dtype=torch.float32)
        terms=F.binary_cross_entropy(before(x),y,reduction='none').mean(1);loss=(terms*r.weight[x[-1].long()]).mean()
        expected=sum(terms[x[-1].long()==d].sum()*float(r.p[d]/r.q[d])/len(rows) for d in range(3));assert torch.allclose(loss,expected,atol=1e-7)
        loss.backward();otheropt.step();r.step(rows,0,0)
        residual=max(float((p-q).abs().max()) for p,q in zip(before.parameters(),r.model.parameters()));assert residual==0
        # Two independently constructed samplers share draws, never initialization RNG state.
        sampler2=c.base.Sampler(r.b,r.q,2022,'iid');assert np.array_equal(r.sampler.sample(1000),sampler2.sample(1000))
        ix=np.concatenate([pool[:64] for pool in r.sampler.pools])
        for suffix in ('x','y','row_ids'):setattr(r.b,'train_'+suffix,getattr(r.b,'train_'+suffix)[ix])
        chosen=[]
        for d in range(3):
            ds=np.flatnonzero(r.b.val_x[:,-1]==d);chosen.extend(ds[:16])
            for task in range(2):
                for label in (0,1):chosen.append(ds[np.flatnonzero(r.b.val_y[ds,task]==label)[0]])
        chosen=np.unique(chosen)
        for suffix in ('x','y','row_ids'):setattr(r.b,'val_'+suffix,getattr(r.b,'val_'+suffix)[chosen])
        r.selection=np.arange(len(chosen));r.diagnostic=r.selection;r.sampler=c.base.Sampler(r.b,r.q,2022,'iid')
        r.steps=0;r.fit();state=torch.load(directory/'last_resume.pt',map_location='cpu',weights_only=False)
        assert state['steps']==3 and state['optimizer']['state']==r.opt.state_dict()['state'] if opt=='sgd' else state['steps']==3 and len(state['optimizer']['state'])>0
        clone=c.base.fixed_model(r.b,2022);clone.load_state_dict(state['model']);assert c.base.initial_state_hash(clone)==c.base.initial_state_hash(r.model)
        s=c.base.Sampler(r.b,r.q,2022,'iid');s.rng.bit_generator.state=state['sampler_rng'];assert np.array_equal(s.sample(64),r.sampler.sample(64))
        results.append(dict(optimizer=opt,status='PASS',loss_oracle_residual=float(abs(loss.detach()-expected.detach())),native_step_max_residual=residual,steps=3,complete_state=True,pairing=True))
    c.dump(c.OUT/'preflight/smoke_receipt.json',dict(status='PASS',checks=results,source_hashes=c.sources(),test_read=False))

if __name__=='__main__':main()
