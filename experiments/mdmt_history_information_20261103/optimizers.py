"""One shared moment, preconditioned history projection or same-sample transport."""
import copy
import itertools
import numpy as np
import torch

def shared_name(name):
    return name in ('shared_weight','shared_bias') or name.startswith(('skip_conn.','star_mlp.','expert.'))

def dot(a,b):
    return sum((x.double()*y.double()).sum() for x,y in zip(a,b))

def norm(a):
    return dot(a,a).sqrt()

def project_history(grad_m,preconditioner,probes,kappa,gamma=.2):
    root=[d.sqrt() for d in preconditioner]
    h=[r*m for r,m in zip(root,grad_m)]; normals=[];cosines=[]
    for a,b in probes:
        u=[r*g for r,g in zip(root,a)];v=[r*g for r,g in zip(root,b)]
        nu,nv=norm(u),norm(v)
        cosine=float(dot(u,v)/(nu*nv)) if float(nu)>1e-12 and float(nv)>1e-12 else None
        cosines.append(cosine)
        if cosine is not None and cosine>=gamma:
            normals.extend(([g/nu for g in u],[g/nv for g in v]))
    hn=float(norm(h));info=dict(constraints=len(normals),probe_cosines=cosines,projection_distance=0.,history_norm_ratio=1.)
    if not normals or hn==0 or kappa==0: return [g.clone() for g in grad_m],info
    count=len(normals)
    gram=torch.stack([torch.stack([dot(a,b) for b in normals]) for a in normals]).cpu().numpy()
    q=torch.stack([dot(n,h) for n in normals]).cpu().numpy()
    tol=1e-9*max(1.,hn);best=None;cost=float('inf')
    for size in range(count+1):
        for active in itertools.combinations(range(count),size):
            lam=np.zeros(count)
            if active:
                idx=np.asarray(active);lam[idx]=np.linalg.pinv(gram[np.ix_(idx,idx)],rcond=1e-12)@(-q[idx])
                if np.max(np.abs((q+gram@lam)[idx]))>tol:continue
            if np.min(lam)<-tol or np.min(q+gram@lam)<-tol:continue
            distance=float(lam@gram@lam)
            if distance<cost-1e-14:best=lam;cost=distance
    if best is None:raise ArithmeticError(f'History KKT failed: G={gram.tolist()} q={q.tolist()} tol={tol}')
    corrected=[]
    for i,(m,r) in enumerate(zip(grad_m,root)):
        change=sum(float(lam)*n[i] for lam,n in zip(best,normals))
        corrected.append(m+kappa*change/r)
    info.update(projection_distance=max(cost,0.)**.5,
                history_norm_ratio=float(norm([r*m for r,m in zip(root,corrected)]))/hn)
    return corrected,info

def transport_correction(current_grads,old_point_grads,kappa,parameter_step):
    delta=[g-old for g,old in zip(current_grads,old_point_grads)]
    ng,nd=float(norm(current_grads)),float(norm(delta));factor=min(1.,ng/(nd+1e-12))
    quality=kappa*.9*(1-.9**(parameter_step-1))
    return [d*(factor*quality) for d in delta],dict(delta_over_gradient=nd/(ng+1e-12),
            transport_clip_factor=factor,transport_quality=quality)

class HistoryOptimizer:
    def __init__(self,model,config):
        self.config=config;self.named=list(model.named_parameters());self.parameters=[p for _,p in self.named]
        self.shared=[(n,p) for n,p in self.named if shared_name(n)]
        mode=config['optimizer']
        self.mode='source_adam' if config.get('kappa',1)==0 else mode
        ordinary=[p for n,p in self.named if self.mode=='source_adam' or not shared_name(n)]
        self.adam=torch.optim.Adam(ordinary,lr=config['lr'],betas=(.9,.999),eps=1e-8,
            weight_decay=config['weight_decay'],foreach=False,fused=False,amsgrad=False)
        self.moments={};self.global_step=0;self.previous_state=None
        self.parameter_manifest=[dict(name=n,shape=list(p.shape),shared=shared_name(n)) for n,p in self.named]
    def zero_grad(self,set_to_none=True):
        for p in self.parameters:p.grad=None if set_to_none else torch.zeros_like(p)
    @torch.no_grad()
    def step(self,current_data_grads=None,probes=None,old_point_data_grads=None,old_parameters=None):
        self.global_step+=1
        if current_data_grads is not None:
            for p,g in zip(self.parameters,current_data_grads):p.grad=g
        if self.mode=='source_adam':self.adam.step();return {}
        g=[];old=[];v=[];m=[];pre=[]
        for name,p in self.shared:
            assert p.grad is not None,('missing shared gradient',name)
            if name not in self.moments:self.moments[name]=dict(step=0,m=torch.zeros_like(p),v=torch.zeros_like(p))
            state=self.moments[name];state['step']+=1;assert state['step']==self.global_step
            current=p.grad.add(p,alpha=self.config['weight_decay']);g.append(current)
            state['v'].mul_(.999).addcmul_(current,current,value=.001)
            v.append(state['v']);m.append(state['m'])
            pre.append((state['v']/(1-.999**state['step'])).sqrt().add_(1e-8).reciprocal())
        mode=self.mode;info={}
        if mode in ('history_repair','history_norm_only','current_gradient_repair'):
            target=g if mode=='current_gradient_repair' else m
            repaired,info=project_history(target,pre,probes or [],self.config['kappa'])
            if mode=='history_norm_only':
                denominator=float(norm([d.sqrt()*a for d,a in zip(pre,m)]))
                ratio=1. if denominator<=1e-12 else info['history_norm_ratio']
                info.update(norm_degenerate=int(denominator<=1e-12),applied_history_norm_ratio=ratio)
                repaired=[a*ratio for a in m]
            new=[.9*a+.1*b for a,b in zip(m,repaired)] if mode=='current_gradient_repair' else [.9*a+.1*b for a,b in zip(repaired,g)]
        else:
            base=[.9*a+.1*b for a,b in zip(m,g)]
            if old_point_data_grads is None:
                assert self.global_step==1
                correction=[torch.zeros_like(a) for a in g];info=dict(delta_over_gradient=0.,transport_clip_factor=1.,transport_quality=0.)
            else:
                old=[a.add(old_parameters[n],alpha=self.config['weight_decay'])
                     for (n,_),a in zip(self.shared,old_point_data_grads)]
                correction,info=transport_correction(g,old,self.config['kappa'],self.global_step)
            new=[a+b for a,b in zip(base,correction)]
            denom=float(norm([d.sqrt()*a for d,a in zip(pre,base)]))
            ratio=float(norm([d.sqrt()*a for d,a in zip(pre,new)]))/(denom+1e-30) if denom>1e-12 else 1.
            info.update(transport_moment_ratio=ratio,norm_degenerate=int(denom<=1e-12))
            if mode=='transport_norm_only':new=[a*ratio for a in base]
        for (_,p),state_m,numerator,d in zip(self.shared,m,new,pre):
            state_m.copy_(numerator)
            p.add_(d*numerator,alpha=-self.config['lr']/(1-.9**self.global_step))
        self.adam.step();return info
    def state_dict(self):
        return dict(mode=self.mode,config=self.config,adam=self.adam.state_dict(),moments=self.moments,
                    global_step=self.global_step,previous_state=self.previous_state,parameter_manifest=self.parameter_manifest)
    def load_state_dict(self,state):
        assert state['mode']==self.mode and state['config']==self.config and state['parameter_manifest']==self.parameter_manifest
        self.adam.load_state_dict(state['adam']);device=self.parameters[0].device
        def move(v):
            if torch.is_tensor(v):return v.to(device).clone()
            if isinstance(v,dict):return {k:move(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return type(v)(move(x) for x in v)
            return copy.deepcopy(v)
        self.moments=move(state['moments']);self.global_step=state['global_step'];self.previous_state=move(state['previous_state'])
