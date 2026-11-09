"""PUB updates and externally replayed alpha-gradient Adam on disjoint coordinates."""
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F
from .solver import CCPSolver
from .group_registry import GROUPS

CELLS=tuple((d,t) for d in range(3) for t in range(2))

def losses(model,x,y,topology):
    pred=model(x); bce=F.binary_cross_entropy(pred,y,reduction='none')
    cells=[bce[x[-1]==d,t].sum()/(2*len(y)) for d,t in CELLS]
    terms=[sum(cells[d*2+t] for d in range(3)) for t in range(2)] if topology==0 else cells
    return terms,bce.mean(1).mean(),[i for i,(d,t) in enumerate(CELLS) if bool((x[-1]==d).any())]

def gradients(terms,model):
    params=tuple(model.parameters()); rows=[]
    for i,term in enumerate(terms):
        values=torch.autograd.grad(term,params,retain_graph=i<len(terms)-1,allow_unused=True)
        rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,values)]))
    return torch.stack(rows)

def moments(z,m,v,step,betas=(.9,.999)):
    b1,b2=betas
    return m.expand_as(z).clone().lerp_(z,1-b1),v.expand_as(z).clone().mul_(b2).addcmul_(z,z,value=1-b2)

def candidate(z,m,v,step,betas=(.9,.999),eps=1e-8):
    mn,vn=moments(z,m,v,step,betas)
    return (mn/(1-betas[0]**step))/(vn.sqrt()/math.sqrt(1-betas[1]**step)+eps)

class PUBAdam:
    def __init__(self,model,registry,topology=1,mode='PUB',lr=.01,weight_decay=0.,frequency=10,solver=None):
        self.model,self.registry,self.topology,self.mode=model,registry,topology,mode
        self.lr,self.wd,self.frequency=lr,weight_decay,frequency
        self.layout=registry.layout; self.solver=solver or CCPSolver()
        self.m=torch.zeros_like(self.layout.flatten(model.parameters())); self.v=self.m.clone(); self.step=0; self.cached={}
        coords=registry.indices(self.m.device)
        self.indices={g:torch.where(coords==i)[0] for i,g in enumerate(GROUPS)}

    def state_dict(self): return copy.deepcopy(dict(m=self.m,v=self.v,step=self.step,cached=self.cached))
    def load_state_dict(self,value):
        self.m=value['m'].to(self.m.device).clone(); self.v=value['v'].to(self.v.device).clone(); self.step=value['step']; self.cached=copy.deepcopy(value['cached'])

    def members(self,group,active):
        if group.startswith('DT') or self.mode=='ADAM' or (self.topology==0 and group.startswith('T')): return []
        if self.topology==0: return [0,1] if group=='S' or any(CELLS[c][0]==int(group[1:]) for c in active) else []
        return [i for i in active if group=='S' or group==f'D{CELLS[i][0]}' or group==f'T{CELLS[i][1]}']

    @torch.no_grad()
    def apply(self,grads,active,replay=None,diagnostics=False,retain_updates=False,task_order=None):
        if task_order is not None:
            grads=grads[[task_order.index(i) for i in range(len(task_order))]]
        active=sorted(active)
        theta=self.layout.flatten(self.model.parameters()); natural=grads.sum(0); t=self.step+1
        newtheta=theta.clone(); newm=self.m.clone(); newv=self.v.clone(); coefficients={}; diags={}; updates={}
        if self.mode=='G' and replay is None: raise ValueError('Gradient control requires PUB coefficient replay')
        for group in GROUPS:
            idx=self.indices[group]; old=theta[idx]; m=self.m[idx]; v=self.v[idx]; plain=natural[idx]
            members=self.members(group,active); balanced=len(members)>1
            row=dict(members=members,fallback=False,alpha=None)
            u=None; G=None; solve_record=None
            if balanced:
                taskg=grads[members][:,idx]
                if self.mode=='G':
                    source=replay[group]
                    if source['members']!=members: raise ValueError('Replay active tasks changed')
                    row=copy.deepcopy(source)
                else:
                    u=candidate(taskg,m,v,t); ud=u.double(); G=(ud@ud.T).cpu().numpy()
                    cached=self.cached.get(group)
                    if self.topology==1 and group!='S':
                        source=coefficients['S']
                        row['fallback']=source['fallback']
                        if not row['fallback']: row['alpha']=[source['alpha'][source['members'].index(i)] for i in members]
                    else:
                        due=(t-1)%self.frequency==0 or cached is None or cached['members']!=members or cached['fallback']
                        if due:
                            warm=cached['alpha'] if cached and cached['members']==members and not cached['fallback'] else None
                            alpha,solve_record=self.solver.solve(G,warm)
                            row['alpha']=alpha.tolist() if alpha is not None else None; row['fallback']=alpha is None
                        else: row=copy.deepcopy(cached)
                    self.cached[group]=copy.deepcopy(row)
            coefficients[group]=row
            if balanced and not row['fallback']:
                alpha=torch.tensor(row['alpha'],dtype=grads.dtype,device=grads.device)
                gbar=(taskg*alpha[:,None]).sum(0); z=gbar.add(old,alpha=self.wd)
                mn,vn=moments(z,m,v,t)
                if self.mode=='G':
                    denom=vn.sqrt()/math.sqrt(1-.999**t)+1e-8
                    newtheta[idx]=old.addcdiv(mn,denom,value=-self.lr/(1-.9**t))
                    direction=(old-newtheta[idx])/self.lr
                else:
                    direction=(u*alpha[:,None]).sum(0)+candidate(z,m,v,t)-candidate(gbar,m,v,t)
                    newtheta[idx]=old-self.lr*direction
            else:
                z=plain.add(old,alpha=self.wd); mn,vn=moments(z,m,v,t)
                denom=vn.sqrt()/math.sqrt(1-.999**t)+1e-8
                newtheta[idx]=old.addcdiv(mn,denom,value=-self.lr/(1-.9**t))
                direction=(old-newtheta[idx])/self.lr
            newm[idx]=mn; newv[idx]=vn
            if retain_updates: updates[group]=direction.clone()
            if balanced and self.mode!='G' and (diagnostics or solve_record is not None):
                natural_u=candidate(plain.add(old,alpha=self.wd),m,v,t); actual_util=(u.double()@direction.double()).cpu().tolist()
                diag=np.sqrt(np.maximum(np.diag(G),0)); cosine=G/np.maximum(diag[:,None]*diag[None,:],1e-30)
                diags[group]=dict(gram=G.tolist(),candidate_norms=diag.tolist(),cosines=cosine.tolist(),alpha=row['alpha'],fallback=row['fallback'],solve=solve_record,actual_utility=actual_util,update_norm=float(direction.norm()),natural_adam_norm=float(natural_u.norm()),update_norm_ratio=float(direction.norm()/natural_u.norm().clamp_min(1e-30)))
                if row['alpha'] is not None: diags[group]['data_utility']=(G@np.asarray(row['alpha'])).tolist()
        self.layout.copy_to(self.model,newtheta); self.m,self.v,self.step=newm,newv,t
        return dict(step=t,active_cells=active,groups=coefficients),diags,updates
