"""Only the quota identities needed for the new experimental intervention."""
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import n17_common as c
from n17_quota import QuotaSampler
from n17_geometry import frozen_map

def main():
    torch.set_num_threads(2);b=c.bundle();p=np.array([b.train_counts[d] for d in range(3)],dtype=float);p/=p.sum();model=c.base.fixed_model(b,2022);rows=[]
    for arm,q in [('SN',p),('SB',np.ones(3)/3)]:
        for size in (64,40960,40052):
            bounds=np.r_[0.,np.cumsum(q)];bounds[-1]=1.
            cuts=np.unique(np.r_[0.,1.,np.mod(-size*bounds[1:-1],1.)]);expected=np.zeros(3)
            for left,right in zip(cuts[:-1],cuts[1:]):
                u=(left+right)/2;counts=np.diff(np.floor(size*bounds+u)).astype(int);assert counts.sum()==size and np.all(counts>0)
                expected+=(right-left)*counts
            assert np.allclose(expected,size*q,atol=1e-10,rtol=0)
            sampler=QuotaSampler(b,q,2022);state=copy.deepcopy(sampler.rng.bit_generator.state);ix=sampler.sample(size);clone=QuotaSampler(b,q,2022);clone.rng.bit_generator.state=state;assert np.array_equal(ix,clone.sample(size))
            counts=np.bincount(b.train_x[ix,-1].astype(int),minlength=3);assert np.allclose(counts*(size*p/counts)/size,p,atol=1e-15,rtol=0)
            if size==64:
                x=torch.tensor(b.train_x[ix],dtype=torch.float32).T;y=torch.tensor(b.train_y[ix],dtype=torch.float32);losses=F.binary_cross_entropy(model(x),y,reduction='none').mean(1);d=x[-1].long()
                actual=(losses*torch.tensor(size*p/counts,dtype=torch.float32)[d]).mean();reference=sum(float(p[k])*losses[d==k].mean() for k in range(3))
                params=list(model.parameters());g1=torch.autograd.grad(actual,params,retain_graph=True,allow_unused=True);g2=torch.autograd.grad(reference,params,allow_unused=True)
                residual=max(float((a-bb).abs().max()) for a,bb in zip(g1,g2) if a is not None and bb is not None);assert residual<1e-6
            else:residual=None
            rows.append(dict(arm=arm,batch_size=size,expected_count_error=float(np.max(abs(expected-size*q))),gradient_residual=residual,RNG_restoration=True,loss_identity=True))
    theta=torch.linspace(-1,1,2048);g=torch.sin(theta);m=.1*torch.cos(theta);v=torch.ones_like(theta)*.03;step=87
    param=torch.nn.Parameter(theta.clone());opt=torch.optim.Adam([param],lr=.003,weight_decay=1e-5);opt.state[param]=dict(step=torch.tensor(float(step)),exp_avg=m.clone(),exp_avg_sq=v.clone());param.grad=g.clone();opt.step()
    mapped=frozen_map(g,theta,dict(m=m,v=v,step=step),'adam',.003);rounding=(param.detach()-theta)-mapped
    assert float(rounding.abs().max())<1e-7
    c.dump(c.OUT/'preflight/quota_receipt.json',dict(status='PASS',checks=rows,adam_mapping_parameter_rounding_max=float(rounding.abs().max()),quota_sha=c.sha(Path(__file__).with_name('n17_quota.py')),geometry_sha=c.sha(Path(__file__).with_name('n17_geometry.py')),scope='necessary numerical identities; no N2 research run',test_read=False))

if __name__=='__main__':main()
