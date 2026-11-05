"""One fixed-EMA teacher and its old-state conditional augmentation statistics."""
import copy
import time
import torch
from experiments.overnight_ab_20261103.state import state_digest
from .drift_probe import DriftProbeSampler,input_drift
from .covariance import statistics,covariance_root,fixed_rotations,controlled_statistics


class ConditionalAugState:
    def __init__(self,online,layout,bundle,config,seed,candidate_id):
        self.teacher=copy.deepcopy(online).requires_grad_(False)
        self.layout=layout; self.config=copy.deepcopy(config); self.seed=seed; self.candidate_id=candidate_id
        fields=len(online.perturb_slices)
        self.A=torch.zeros(2,fields,16,16,dtype=torch.float64)
        self.root=torch.eye(16,dtype=torch.float64).expand(2,fields,16,16).clone()
        self.rotation=fixed_rotations(bundle.dataset,seed,fields) if config['control_mode']=='rotated' else None
        self.sampler=DriftProbeSampler(bundle)
        self.last_refresh=0; self.last_valid_step=0; self.updates=0; self.last_probe={}; self.probe_seconds=0.

    @property
    def phi(self):
        return {self.candidate_id:self.layout.flatten(self.teacher.parameters())}

    def before_step(self,online,row_ids,step):
        if self.updates!=step-1:
            raise ValueError('Teacher does not hold the required old state')
        elapsed=0.; refreshed=False
        if (step-1)%self.config['refresh_steps']==0:
            started=time.perf_counter(); self.last_refresh=step
            values,metadata=self.sampler.sample(row_ids,self.seed,step)
            metadata.update(step=step,online_step=step-1,teacher_step=self.updates,
                teacher_hash=state_digest(self.teacher.state_dict()))
            if len(values) and self.config['gamma']!=0:
                device=next(online.parameters()).device
                x=torch.as_tensor(values,device=device).T
                u=input_drift(online,self.teacher,x,len(online.perturb_slices))
                self.A=statistics(u)
                transformed=controlled_statistics(self.A,self.config['control_mode'],self.rotation)
                self.root=covariance_root(transformed,self.config['gamma'])
                self.last_valid_step=step; refreshed=True
                metadata.update(mean_statistic_trace=float(self.A.diagonal(dim1=-2,dim2=-1).sum(-1).mean()),
                    root_hash=state_digest(self.root),mean_drift_norm=float(u.double().norm(dim=-1).mean()))
            self.last_probe=metadata
            elapsed=time.perf_counter()-started; self.probe_seconds+=elapsed
        return dict(probe_seconds=elapsed,refreshed=refreshed,last_refresh=self.last_refresh,
            last_valid_step=self.last_valid_step,teacher_preupdate_step=self.updates,
            online_preupdate_step=step-1,probe=self.last_probe if self.last_refresh==step else None)

    @torch.no_grad()
    def after_step(self,online,step):
        if step!=self.updates+1:
            raise ValueError('Teacher update is not consecutive')
        rho=self.config['rho']
        for target,source in zip(self.teacher.parameters(),online.parameters()):
            target.mul_(rho).add_(source,alpha=1-rho)
        for target,source in zip(self.teacher.buffers(),online.buffers()):
            target.copy_(source)
        self.updates=step

    def state_dict(self):
        return dict(teacher=self.teacher.state_dict(),A=self.A,root=self.root,rotation=self.rotation,
            config=self.config,seed=self.seed,candidate_id=self.candidate_id,last_refresh=self.last_refresh,
            last_valid_step=self.last_valid_step,updates=self.updates,last_probe=self.last_probe,
            probe_seconds=self.probe_seconds)

    def load_state_dict(self,value,device):
        if value['config']!=self.config or value['seed']!=self.seed or value['candidate_id']!=self.candidate_id:
            raise ValueError('Conditional augmentation RAM identity changed')
        self.teacher.load_state_dict(value['teacher'])
        for key in ('A','root','rotation','last_refresh','last_valid_step','updates','last_probe','probe_seconds'):
            setattr(self,key,copy.deepcopy(value[key]))
