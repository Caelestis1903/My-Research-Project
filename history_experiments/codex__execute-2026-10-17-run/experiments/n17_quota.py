"""Conditional domain-quota training under exactly the frozen Sample objective."""
import argparse
import json
from pathlib import Path
import traceback
import numpy as np
import torch
import n17_common as c
from n17_capture import Runner as IIDRunner

class QuotaSampler:
    def __init__(self,bundle,q,seed):
        self.q=np.asarray(q,dtype=float);self.pools=[np.flatnonzero(bundle.train_x[:,-1]==d) for d in range(3)]
        self.rng=np.random.default_rng(seed*1000+17)
    def sample(self,n):
        cumulative=np.r_[0.,np.cumsum(self.q)];cumulative[-1]=1.
        u=self.rng.random();counts=np.diff(np.floor(n*cumulative+u)).astype(int)
        assert counts.sum()==n and np.all(counts>0),'positive domain quotas required'
        rows=np.concatenate([pool[self.rng.integers(0,len(pool),size=count)] for pool,count in zip(self.pools,counts)])
        # No order-dependent layers; randomize positions without changing quotas.
        self.rng.shuffle(rows)
        return rows

class Runner(IIDRunner):
    def __init__(self,a):
        super().__init__(a);state=self.sampler.rng.bit_generator.state if a.resume else None
        self.sampler=QuotaSampler(self.b,self.q,a.seed)
        if state is not None:self.sampler.rng.bit_generator.state=state
        self.quota_sha=c.sha(__file__)
        if a.resume:
            assert self.config['quota_source_sha256']==self.quota_sha and self.config['sampling']=='systematic_quota_WR'
        else:
            self.config.update(sampling='systematic_quota_WR',actual_weight='B_t*p_d/n_d per batch',loss='sum_d p_d * mean_{i in batch domain d} mean_task BCE',quota_source_sha256=self.quota_sha,importance_weight_is_iid_reference=True)
            c.dump(self.dir/'config.json',self.config)
    def step(self,rows,epoch,index):
        counts=np.bincount(self.b.train_x[rows,-1].astype(int),minlength=3);assert np.all(counts>0)
        self.weight=torch.tensor(len(rows)*self.p/counts,device=self.device,dtype=torch.float32)
        result=super().step(rows,epoch,index)
        c.base.csv_append(self.dir/'quota_weights.csv',[dict(epoch=epoch,batch=index,n=len(rows),**{f'n{d}':int(counts[d]) for d in range(3)},**{f'w{d}':float(len(rows)*self.p[d]/counts[d]) for d in range(3)})])
        return result
    def fit(self):
        super().fit();assert c.sha(__file__)==self.quota_sha
        receipt=json.loads((self.dir/'receipt.json').read_text());receipt['quota_source_sha256']=self.quota_sha;c.dump(self.dir/'receipt.json',receipt)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=['SN','SB'],required=True)
    p.add_argument('--optimizer',choices=['adam','sgd'],required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--phase',default='N2_quota');p.add_argument('--batch-size',type=int,default=40960);p.add_argument('--device',default='cuda:0')
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:Runner(a).fit()
    except Exception:
        c.dump(a.run_dir/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
