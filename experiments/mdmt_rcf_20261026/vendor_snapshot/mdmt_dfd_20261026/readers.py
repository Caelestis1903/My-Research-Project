"""Frozen-input ordinary readers and accurately solved convex controls."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize

from .functions import ResidualMLP


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_array(cls,x):
        x=np.asarray(x,dtype=np.float64)
        return cls(x.mean(0),np.maximum(x.std(0,ddof=0),1e-3))

    def transform(self,x):
        if torch.is_tensor(x):
            mean=torch.as_tensor(self.mean,device=x.device,dtype=x.dtype)
            std=torch.as_tensor(self.std,device=x.device,dtype=x.dtype)
            return ((x-mean)/std).clamp(-10,10)
        return np.clip((np.asarray(x)-self.mean)/self.std,-10,10).astype(np.float32)

    def clipping(self,x):
        x=np.asarray(x,dtype=np.float64)
        clipped=np.abs((x-self.mean)/self.std)>10
        return {'per_dimension':clipped.mean(0).tolist(),'overall':float(clipped.mean())}

    def state_dict(self): return {'mean':self.mean,'std':self.std}

    @classmethod
    def from_state_dict(cls,state):return cls(np.asarray(state['mean']),np.asarray(state['std']))


class ProbabilityHead(ResidualMLP):
    def __init__(self,input_dim,positive_rate=.5,seed=0):
        rate=float(np.clip(positive_rate,1e-4,1-1e-4))
        super().__init__(input_dim=input_dim,seed=seed,output_bias=[np.log(rate/(1-rate))])


class MixtureHead(ResidualMLP):
    def __init__(self,input_dim=263,components=5,seed=0):
        prior=np.full(components,.1/(components-1),dtype=np.float64);prior[0]=.9
        super().__init__(input_dim=input_dim,output_dim=components,seed=seed,output_bias=np.log(prior))


def stable_mixture_bce(candidate_logits,gate_logits,y,reduction='mean'):
    # FP64 protects the mixed risk while gradient storage stays FP32 in the heads.
    a=candidate_logits.detach().double()
    log_alpha=F.log_softmax(gate_logits.double(),dim=-1)
    logp=torch.logsumexp(log_alpha+F.logsigmoid(a),dim=-1)
    logn=torch.logsumexp(log_alpha+F.logsigmoid(-a),dim=-1)
    loss=-y.detach().double()*logp-(1-y.detach().double())*logn
    if reduction=='none':return loss
    if reduction=='sum':return loss.sum()
    if reduction=='mean':return loss.mean()
    raise ValueError(reduction)


def mixture_probability(candidate_logits,gate_logits):
    return (gate_logits.double().softmax(-1)*candidate_logits.detach().double().sigmoid()).sum(-1)


def solve_simplex(probs,y):
    """Three fixed SLSQP starts; accept only a feasible, near-optimal solution."""
    p=np.clip(np.asarray(probs,dtype=np.float64),1e-15,1-1e-15)
    y=np.asarray(y,dtype=np.float64).reshape(-1)
    if p.ndim!=2 or p.shape[0]!=len(y) or p.shape[1]<2 or not len(y):
        raise ValueError('Expected nonempty [N,K] probabilities and binary [N] labels')
    k=p.shape[1]
    def objective(alpha):
        v=p@alpha
        return float(np.mean(-y*np.log(v)-(1-y)*np.log1p(-v)))
    def gradient(alpha):
        v=p@alpha
        return p.T@((v-y)/(v*(1-v)))/len(y)
    starts=[np.eye(k)[0],np.ones(k)/k,np.r_[.9,np.full(k-1,.1/(k-1))]]
    results=[]
    for start in starts:
        r=minimize(objective,start,jac=gradient,method='SLSQP',bounds=[(0.,1.)]*k,
                   constraints=[{'type':'eq','fun':lambda a:a.sum()-1.,'jac':lambda a:np.ones(k)}],
                   options={'maxiter':500,'ftol':1e-12,'disp':False})
        alpha=np.asarray(r.x,dtype=np.float64)
        # SLSQP tolerates tiny constraint errors. Project only numerical negatives
        # (at machine precision), then explicitly remeasure objective and FW gap.
        alpha[np.abs(alpha)<1e-15]=0.
        alpha=alpha/alpha.sum()
        g=gradient(alpha)
        results.append({'alpha':alpha,'loss':objective(alpha),'success':bool(r.success),
                        'message':str(r.message),'iterations':int(r.nit),
                        'sum_error':float(abs(alpha.sum()-1)),
                        'bounds_error':float(max(0.,-alpha.min(),alpha.max()-1)),
                        'fw_gap':float(g@alpha-g.min())})
    best=min(results,key=lambda r:(r['loss'],float(r['alpha'][1:].sum()),tuple(r['alpha'])))
    receipt={**best,'alpha':best['alpha'].tolist(),'component_clip':1e-15,
             'starts':[{**r,'alpha':r['alpha'].tolist()} for r in results]}
    receipt['accepted']=bool(best['sum_error']<=1e-8 and best['bounds_error']<=1e-10 and best['fw_gap']<=1e-7 and np.isfinite(best['loss']))
    if not receipt['accepted']:
        raise RuntimeError('Simplex solver failed feasibility/near-optimality: '+str(receipt))
    return best['alpha'],receipt


class ReaderGroup(nn.Module):
    def __init__(self,seed_fn=None):
        super().__init__()
        self.heads=nn.ModuleDict({f't{t}_q{q}':MixtureHead(seed=int(seed_fn(t,q,0)) if seed_fn else 3*q+t)
                                  for q in range(2) for t in range(3)})
        self.normalizers={}
        self.off=False

    def head(self,target,task):return self.heads[f't{int(target)}_q{int(task)}']

    @staticmethod
    def features(h,z,candidates,task):
        return torch.cat((h.detach(),z.detach(),candidates[:,task].detach().sigmoid()),dim=-1)

    def predict(self,h,z,domain,candidates):
        with torch.no_grad():
            probabilities=candidates.double().sigmoid()
            if self.off:
                alpha=torch.zeros_like(probabilities);alpha[...,0]=1
                return probabilities[...,0],alpha
            out=torch.empty((len(h),2),device=h.device,dtype=torch.float64)
            alpha=torch.empty_like(probabilities)
            for t in range(3):
                mask=domain==t
                if not bool(mask.any()):continue
                for q in range(2):
                    x=self.features(h[mask],z[mask],candidates[mask],q)
                    g=self.head(t,q)(self.normalizers[t,q].transform(x))
                    alpha[mask,q]=g.double().softmax(-1)
                    out[mask,q]=(alpha[mask,q]*probabilities[mask,q]).sum(-1)
            return out,alpha

    def extra_state(self):
        return {'normalizers':{k:v.state_dict() for k,v in self.normalizers.items()},'off':self.off}

    def load_extra_state(self,state):
        self.normalizers={k:Normalizer.from_state_dict(v) for k,v in state['normalizers'].items()}
        self.off=bool(state['off'])


def select_whole_policy(candidates):
    """Candidates are ordered OFF, pass0, pass1..10; one scalar selects all heads."""
    if not candidates:raise ValueError('No complete policies')
    finite=[c for c in candidates if np.isfinite(c['macro_logloss'])]
    if len(finite)!=len(candidates):raise ValueError('Nonfinite whole-policy risk')
    return min(enumerate(candidates),key=lambda item:(item[1]['macro_logloss'],item[0]))[1]
