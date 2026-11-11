"""Trainable N19 architectures; historical modules remain unchanged."""
import math
import numpy as np
import torch
from torch import nn
from stage_S1_models import build_model

class ConstantFusion(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('value',torch.sigmoid(torch.ones(1)))
    def forward(self):return self.value

class Model(nn.Module):
    def __init__(self,features,seed,method):
        super().__init__();self.method=method
        self.base=build_model(features,3,2,method if method in ['mmoe','ple'] else 'm3oe',seed=seed)
        self.is_m3oe=method not in ['mmoe','ple']
        if not self.is_m3oe:return
        if method not in ['original','original_sample']:
            for name in ['_weight_exp_d','_weight_exp_t','_weight_bal_d','_weight_bal_t']:setattr(self.base,name,ConstantFusion())
        self.rank_mode=method in ['rank_uniform','rank_protected']
        self.codebook=method in ['cell_codebook','domain_codebook']
        if self.rank_mode or self.codebook:
            dims=self.base.shared_weight.shape;self.slot_dimensions=tuple(dims)
            del self.base.slot_weight
            if self.rank_mode:
                self.left=nn.Parameter(torch.randn(3,dims[0],32)/math.sqrt(32))
                self.right=nn.Parameter(torch.randn(3,32,dims[1])/math.sqrt(dims[0]+dims[1]))
                self.register_buffer('rank_mask',torch.ones(3,32));self.register_buffer('rank_frozen',torch.tensor(False))
            else:
                R=47;self.left=nn.Parameter(torch.randn(dims[0],R)/math.sqrt(R))
                self.right=nn.Parameter(torch.randn(R,dims[1])/math.sqrt(dims[0]+dims[1]))
                self.code=nn.Parameter(torch.randn(3,2 if method=='cell_codebook' else 1,R))
        self.mix_mode=method in ['interaction','domain_mix','full_mix','regret_router','utility_router','coalition_router','reliability_interaction']
        if self.mix_mode:self.mix_logits=nn.Parameter(torch.zeros(3,2,3))
        self.register_buffer('interaction_confidence',torch.ones(3,2,3))

    def router_logits(self):
        v=self.mix_logits
        if self.method in ['interaction','reliability_interaction']:
            v=v-v.mean(0,keepdim=True)-v.mean(1,keepdim=True)+v.mean((0,1),keepdim=True)
            if self.method=='reliability_interaction':v=v*self.interaction_confidence
        elif self.method=='domain_mix':v=v.mean(1,keepdim=True).expand(-1,2,-1)
        return v

    def slots(self,masks=None):
        if self.rank_mode:
            mask=self.rank_mask if masks is None else masks
            return torch.bmm(self.left*mask[:,None,:],self.right)
        if self.codebook:
            return torch.einsum('ir,dtr,rj->dtij',self.left,self.code,self.right)
        return self.base.slot_weight

    def forward(self,x,details=False,rank_masks=None):
        if not self.is_m3oe:
            p=self.base(x);return (p,{}) if details else p
        m=self.base;domains=x[-1].long().detach();e=m.embedding(x,m.features,squeeze_dim=True)
        skip=m.skip_conn(e);slots=self.slots(rank_masks);allstar=[]
        for task in range(2 if self.codebook and self.method=='cell_codebook' else 1):
            value=torch.zeros((len(domains),m.star_dim[1]),device=x.device,dtype=e.dtype)
            for d in range(3):
                slot=slots[d,task if self.method=='cell_codebook' else 0] if self.codebook else slots[d]
                out=e@(slot*m.shared_weight)+m.slot_bias[d]+m.shared_bias
                value=torch.where((domains==d)[:,None],out,value)
            allstar.append(m.star_mlp(value)+skip)
        out=torch.zeros((len(domains),2),device=x.device);source_probs=[];router_weights=[];embeddings=[]
        for task in range(2):
            emb=allstar[task] if len(allstar)==2 else allstar[0]
            embeddings.append(emb)
            # Sharing the single STAR/expert forward for ordinary two-task models
            # is mathematically identical and avoids computing it twice.
            if task==0 or len(allstar)==2:
                shared=torch.stack([f(emb) for f in m.expert],dim=1)
                dex=torch.stack([f(emb) for f in m.domain_expert],dim=1)
                tex=torch.stack([f(emb) for f in m.task_expert],dim=1)
            for d in range(3):
                ix=task*3+d;where=domains==d;gate=m.gate[ix](emb.detach())
                sh=(gate[:,:,None]*shared).sum(1)
                default=torch.ones(3,device=x.device)*(1-m._weight_bal_d())/2
                default=default+torch.nn.functional.one_hot(torch.tensor(d,device=x.device),3)*(m._weight_bal_d()-(1-m._weight_bal_d())/2)
                weights=torch.softmax(default.log()+self.router_logits()[d,task],0) if self.mix_mode else default
                dm=(weights[None,:,None]*dex).sum(1)
                tw=m._weight_bal_t()*tex[:,task]+(1-m._weight_bal_t())*tex[:,1-task]
                fused=sh+m._weight_exp_d()*dm+m._weight_exp_t()*tw
                pred=torch.sigmoid(m.tower[ix](fused[where]).squeeze(1));out[where,task]=pred
                if details and self.mix_mode:
                    possible=[]
                    for k in range(3):
                        if self.method=='coalition_router':
                            alt=(dm-weights[k]*dex[:,k])/(1-weights[k]).clamp_min(1e-5)[:,None] if weights.ndim>1 else (dm-weights[k]*dex[:,k])/(1-weights[k]).clamp_min(1e-5)
                        else:alt=dex[:,k]
                        pp=torch.sigmoid(m.tower[ix]((sh+m._weight_exp_d()*alt+m._weight_exp_t()*tw)[where]).squeeze(1))
                        possible.append(pp)
                    source_probs.append((d,task,where,torch.stack(possible,dim=1)));router_weights.append((d,task,weights))
        info=dict(emb=embeddings[0],source_probs=source_probs,router_weights=router_weights)
        return (out,info) if details else out

    def deployed_parameters(self):
        total=sum(p.numel() for p in self.parameters())
        if self.is_m3oe and self.rank_mode:
            total-=self.left.numel()+self.right.numel()
            total+=int(self.rank_mask.sum())*sum(self.slot_dimensions)
        return total
