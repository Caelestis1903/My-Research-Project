"""Ordinary Adam plus explicit dense-source dispatch; each parameter updates once."""
import copy
import torch


def shared_name(name):
    return name in ('shared_weight','shared_bias') or name.startswith(('skip_conn.','star_mlp.','expert.'))

class PairOptimizer:
    def __init__(self,model,config):
        self.config=config;self.named=list(model.named_parameters())
        self.shared=[(n,p) for n,p in self.named if shared_name(n)]
        self.mode=config['mode'] if config['lambda_pair'] else 'adam'
        self.parameters=[p for _,p in self.named]; self.shared_ids={id(p) for _,p in self.shared}
        ordinary=[p for _,p in self.named if self.mode in ('adam','cv') or id(p) not in self.shared_ids]
        self.adam=torch.optim.Adam(ordinary,lr=config['lr'],betas=(.9,.999),eps=1e-8,
            weight_decay=config['weight_decay'],foreach=False,fused=False,amsgrad=False)
        self.history={};self.active={};self.selected={};self.pair_clock=0;self.demand={};self.moments={}
        self.parameter_manifest=[dict(name=n,shape=list(p.shape),history=shared_name(n)) for n,p in self.named]
    def zero_grad(self,set_to_none=True):
        for p in self.parameters:p.grad=None if set_to_none else torch.zeros_like(p)
    def _table(self,i):
        if i not in self.history:self.history[i]=[torch.zeros_like(p) for _,p in self.shared]
        return self.history[i]
    @torch.no_grad()
    def step(self,pair=None,r=None):
        lam=self.config['lambda_pair'];mode=self.mode
        if pair is not None:
            sid=pair['selected'];q=float(pair['q'][pair['index']]);ratio=float(pair['w'][pair['index']]/q)
            self.selected[sid]=self.selected.get(sid,0)+1
        if mode=='cv' and pair is not None:
            old=self._table(sid)
            correction=[torch.zeros_like(p) for _,p in self.shared]
            for i,w in zip(pair['active'],pair['w']):
                for a,h in zip(correction,self._table(i)):a.add_(h,alpha=float(w))
            for (_,p),a,h in zip(self.shared,correction,old):
                assert p.grad is not None
                p.grad.add_(a,alpha=lam).add_(h,alpha=-lam*ratio)
        if mode in ('eccm','collapsed'):
            if pair is not None:
                if mode=='eccm':
                    for i in pair['active']:self.active[i]=self.active.get(i,0)+1
                    for h,g in zip(self._table(sid),r):h.lerp_(g,.01/q)
                else:
                    self.pair_clock+=1
                    for h,g in zip(self._table(-1),r):h.mul_(.99).add_(g,alpha=.01*ratio)
            for k,(name,p) in enumerate(self.shared):
                assert p.grad is not None, ('missing shared gradient',name)
                if name not in self.moments:self.moments[name]=dict(step=0,m=torch.zeros_like(p),v=torch.zeros_like(p))
                s=self.moments[name];s['step']+=1
                mixed=p.grad.add(p,alpha=self.config['weight_decay'])
                real=mixed if pair is None else mixed-lam*ratio*r[k]
                s['m'].mul_(.9).add_(real,alpha=.1);s['v'].mul_(.999).addcmul_(mixed,mixed,value=.001)
                numerator=s['m']/(1-.9**s['step'])
                if pair is not None:
                    if mode=='eccm':
                        for i,w in zip(pair['active'],pair['w']):
                            numerator.add_(self._table(i)[k],alpha=lam*float(w)/(1-.99**self.active[i]))
                    else:numerator.add_(self._table(-1)[k],alpha=lam/(1-.99**self.pair_clock))
                p.addcdiv_(numerator,(s['v']/(1-.999**s['step'])).sqrt().add_(1e-8),value=-self.config['lr'])
        self.adam.step()
        if mode=='cv' and pair is not None:
            for h,g in zip(self._table(sid),r):h.mul_(.9).add_(g,alpha=.1)
    def state_dict(self):
        return dict(mode=self.mode,config=self.config,adam=self.adam.state_dict(),history=self.history,active=self.active,
            selected=self.selected,pair_clock=self.pair_clock,demand=self.demand,moments=self.moments,
            parameter_manifest=self.parameter_manifest)
    def load_state_dict(self,state):
        assert state['mode']==self.mode and state['config']==self.config and state['parameter_manifest']==self.parameter_manifest
        self.adam.load_state_dict(state['adam'])
        device=self.parameters[0].device
        def move(v):
            if torch.is_tensor(v):return v.to(device).clone()
            if isinstance(v,dict):return {k:move(x) for k,x in v.items()}
            if isinstance(v,list):return [move(x) for x in v]
            return copy.deepcopy(v)
        for key in ('history','active','selected','pair_clock','demand','moments'):setattr(self,key,move(state[key]))
