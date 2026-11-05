"""Bounded independent path/global EMA controllers; online parameters stay detached."""
import copy
import math
import time
from collections import OrderedDict
from dataclasses import dataclass,field
import torch
from .group_registry import GROUPS
from .ranking_probe import RankProbeSampler,ranking_objective


@torch.no_grad()
def grouped_update(phi,theta,registry,rho):
    if len(rho)!=12:
        raise ValueError('Expected twelve active path rates')
    if len(set(rho))==1 and registry.counts.get('INACTIVE',0)==0:
        phi.mul_(rho[0]).add_(theta,alpha=1-rho[0])
    else:
        rates=torch.tensor([*rho,registry.rho0],dtype=phi.dtype,device=phi.device)
        complements=torch.tensor([*(1-r for r in rho),1-registry.rho0],dtype=phi.dtype,device=phi.device)
        indices=registry.indices(phi.device)
        phi.mul_(rates[indices]).addcmul_(theta,complements[indices])


def virtual_parameters(theta,phi,q,K,registry):
    qpaths=q.expand(12) if q.numel()==1 else q
    if qpaths.shape!=(12,):
        raise ValueError('Controller q must be scalar or twelve-dimensional')
    rates=q.new_tensor(registry.rho0).pow(2.**(-qpaths)*K)
    rates=torch.cat((rates,rates.new_tensor([registry.rho0**K])))
    coefficient=rates[registry.indices(theta.device)]
    virtual=coefficient*phi.detach()+(1-coefficient)*theta.detach()
    return OrderedDict((name,virtual[start:end].view(shape)) for name,shape,(start,end) in
        zip(registry.layout.names,registry.layout.shapes,registry.layout.offsets))


@dataclass
class GroupedEMAState:
    phi: torch.Tensor
    q: torch.Tensor
    eta: float | None
    K: int | None
    next_refresh: int | None
    buffers: dict=field(default_factory=dict)
    controller_updates: int=0
    last_refresh: int=0
    last_probe: dict=field(default_factory=dict)
    cost: dict=field(default_factory=lambda:dict(controller_seconds=0.,ema_seconds=0.,validation_seconds=0.))


def controller_step(state,theta,probe,registry,model):
    before=state.q.detach().clone()
    if not probe.cells:
        state.last_probe=copy.deepcopy(probe.metadata)
        return dict(q_before=before.tolist(),q_after=before.tolist(),grad_max=0.,valid_cells=0)
    q=before.requires_grad_(True)
    params=virtual_parameters(theta,state.phi,q,state.K,registry)
    loss=ranking_objective(params,probe,model)
    gradient=torch.autograd.grad(loss,q,create_graph=False)[0]
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError('Nonfinite ranking-controller gradient')
    norm=gradient.detach().double().abs().max()
    if float(norm)>0:
        state.q=(q-state.eta*gradient/norm.clamp_min(1e-12).to(gradient)).clamp(-1,1).detach()
    state.controller_updates+=1
    state.last_probe=copy.deepcopy(probe.metadata)
    return dict(q_before=before.tolist(),q_after=state.q.tolist(),grad_max=float(norm),
        ranking_loss=float(loss.detach()),valid_cells=len(probe.cells))


class GroupedEMABank:
    def __init__(self,theta0,candidates,rho0,registry,bundle,seed):
        self.registry=registry; registry.rho0=rho0
        self.candidates=candidates; self.rho0=rho0; self.seed=seed
        self.sampler=RankProbeSampler(bundle); self.updates=0; self.last_probe={}
        self.states={}
        for candidate in candidates:
            family=candidate['family']; eta=candidate.get('eta'); K=candidate.get('refresh_steps')
            if family=='U':
                q=theta0.new_tensor([math.log2(candidate['half_life_ratio'])])
            elif family=='A_FIXED':
                profile=dict(zip(('S','D','T','DT'),candidate['half_life_ratios']))
                q=theta0.new_tensor([math.log2(profile['DT' if group.startswith('DT') else group[0]]) for group in GROUPS])
            else:
                q=theta0.new_zeros(1 if family=='A_GLOBAL' else 12)
            self.states[candidate['id']]=GroupedEMAState(theta0.clone(),q,eta,K,K)
        self.phi={key:state.phi for key,state in self.states.items()}

    @staticmethod
    def refresh_due(step,cadence=20):
        return step>0 and step%cadence==0

    def after_step(self,context):
        step,theta=context['step'],context['theta']
        if step!=self.updates+1:
            raise ValueError('A shadow update is not consecutive')
        due=[(cid,state) for cid,state in self.states.items() if state.K and step==state.next_refresh]
        sampled=time.perf_counter()
        probe=self.sampler.sample(context['row_ids'],self.seed,step) if due else None
        sampling_seconds=time.perf_counter()-sampled if due else 0.
        if probe:
            self.last_probe=copy.deepcopy(probe.metadata)
        metadata={}; total=0.
        for candidate in self.candidates:
            cid=candidate['id']; state=self.states[cid]
            cuda=theta.device.type=='cuda'
            if state.K and step==state.next_refresh:
                if cuda: torch.cuda.synchronize(theta.device)
                started=time.perf_counter()
                metadata[cid]=controller_step(state,theta,probe,self.registry,context['model'])
                if cuda: torch.cuda.synchronize(theta.device)
                elapsed=time.perf_counter()-started
                state.cost['controller_seconds']+=elapsed; total+=elapsed
                state.last_refresh=step; state.next_refresh=step+state.K
                metadata[cid].update(probe_hash=probe.metadata['probe_hash'],controller_seconds=elapsed)
            if cuda: torch.cuda.synchronize(theta.device)
            started=time.perf_counter()
            q=state.q.tolist(); q=q*12 if len(q)==1 else q
            rates=[self.rho0**(2.**(-value)) for value in q]
            grouped_update(state.phi,theta,self.registry,rates)
            if cuda: torch.cuda.synchronize(theta.device)
            state.cost['ema_seconds']+=time.perf_counter()-started
            state.buffers={k:v.detach().clone() for k,v in context['model'].named_buffers()}
        self.updates=step
        return dict(controller_seconds=total,sampling_seconds=sampling_seconds,
            probe=probe.metadata if probe else None,controllers=metadata)

    def state_dict(self):
        return dict(group_hash=self.registry.hash,states={cid:vars(state).copy() for cid,state in self.states.items()},
            updates=self.updates,last_probe=self.last_probe,seed=self.seed)

    def load_state_dict(self,value,device):
        if value['group_hash']!=self.registry.hash or value['seed']!=self.seed:
            raise ValueError('A coordinate/RNG identity changed')
        for cid,data in value['states'].items():
            restored=copy.deepcopy(data)
            restored['phi']=restored['phi'].to(device); restored['q']=restored['q'].to(device)
            restored['buffers']={k:v.to(device) for k,v in restored['buffers'].items()}
            self.states[cid]=GroupedEMAState(**restored)
        self.phi={cid:state.phi for cid,state in self.states.items()}
        self.updates=value['updates']; self.last_probe=copy.deepcopy(value['last_probe'])
