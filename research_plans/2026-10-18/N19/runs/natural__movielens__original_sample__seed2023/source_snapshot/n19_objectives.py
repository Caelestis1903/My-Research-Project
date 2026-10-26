"""Declared observation likelihoods and task-mask policies, no research decisions."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class NoiseChannel(nn.Module):
    def __init__(self,axis=False):
        super().__init__();self.axis=axis
        if axis:
            self.mu=nn.Parameter(torch.full((2,1,1),math.log(.1/.25)))
            self.row=nn.Parameter(torch.zeros(2,3,1));self.column=nn.Parameter(torch.zeros(2,1,2))
        else:self.raw=nn.Parameter(torch.full((2,3,2),math.log(.1/.25)))
    def rates(self):
        v=self.mu+self.row+self.column if self.axis else self.raw
        rates=.35*torch.sigmoid(v)
        return rates[0],rates[1]
    def loss(self,clean_p,noisy,clean_audit,audit,domains,joint=False):
        p=clean_p.clamp(1e-6,1-1e-6);aa,bb=self.rates();a=aa[domains];b=bb[domains]
        if joint:
            states=torch.tensor([[0.,0.],[1.,0.],[1.,1.]],device=p.device)
            mass=torch.stack([1-p[:,0],p[:,0]-p[:,1],p[:,1]],dim=1).clamp_min(1e-8)
            eta=a[:,None,:]+(1-a-b)[:,None,:]*states[None,:,:]
            obsprob=torch.where(noisy[:,None,:]>0,eta,1-eta).prod(2)
            marginal=(mass*obsprob).sum(1).clamp_min(1e-8)
            rowloss=-marginal.log()/2
            if audit.any():
                state_index=clean_audit[audit].sum(1).long()
                cleanloss=-mass[audit, state_index].log()/2
                channel=a[audit]+(1-a[audit]-b[audit])*clean_audit[audit]
                rowloss[audit]=cleanloss+F.binary_cross_entropy(channel,noisy[audit],reduction='none').mean(1)
        else:
            eta=a+(1-a-b)*p
            rowloss=F.binary_cross_entropy(eta,noisy,reduction='none').mean(1)
            if audit.any():
                channel=a[audit]+(1-a[audit]-b[audit])*clean_audit[audit]
                rowloss[audit]=F.binary_cross_entropy(p[audit],clean_audit[audit],reduction='none').mean(1)+F.binary_cross_entropy(channel,noisy[audit],reduction='none').mean(1)
        return rowloss.mean()

def clean_predictions(prob,method):
    if method in ['nested_ind','joint_axis']:
        return torch.stack([prob[:,0],prob[:,0]*prob[:,1]],dim=1)
    return prob

def task_masks(method,domains,generator,copula):
    pi=.55
    if method in ['mask_positive']:r=torch.full_like(domains,pi,dtype=torch.float32)
    elif method in ['mask_antithetic']:r=torch.full_like(domains,.1,dtype=torch.float32)
    elif method in ['mask_copula','mask_copula_cv']:r=torch.where(copula[domains]>=0,.1,.55)
    else:r=torch.full_like(domains,pi*pi,dtype=torch.float32)
    u=torch.rand(len(domains),device=domains.device,generator=generator)
    both=u<r;t0=both|((u>=r)&(u<pi));t1=both|((u>=pi)&(u<2*pi-r))
    return torch.stack([t0,t1],dim=1),r
