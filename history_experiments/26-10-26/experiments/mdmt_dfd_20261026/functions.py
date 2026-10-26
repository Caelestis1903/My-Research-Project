"""Independent frozen-input pilot and source-supervised function banks (DFD-v4)."""
from __future__ import annotations

import copy
from typing import Callable
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualMLP(nn.Module):
    def __init__(self, input_dim=256, output_dim=1, seed=0, output_bias=None):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.hidden = nn.Linear(input_dim, 32)
            self.output = nn.Linear(32, output_dim)
            nn.init.xavier_uniform_(self.hidden.weight)
            nn.init.zeros_(self.hidden.bias)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
            if output_bias is not None:
                self.output.bias.data.copy_(torch.as_tensor(output_bias).reshape_as(self.output.bias))

    def forward(self, x):
        result = self.output(F.relu(self.hidden(x)))
        return result.squeeze(-1) if result.shape[-1] == 1 else result


def _seed(seed_fn, s, q, component=0):
    return int(seed_fn(s, q, component)) if seed_fn else int(1009*s+101*q+component)


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    for p in module.parameters():
        p.grad = None
    return module


class PilotBank(nn.Module):
    """Six separate owners; only active owners enter a training graph."""
    def __init__(self, seed_fn: Callable | None = None):
        super().__init__()
        self.heads = nn.ModuleDict({f's{s}_q{q}': ResidualMLP(seed=_seed(seed_fn,s,q))
                                    for q in range(2) for s in range(3)})
        self.owner_map = [{'key':f's{s}_q{q}','source':s,'task':q,'head_index':3*q+s}
                          for q in range(2) for s in range(3)]

    def head(self, source, task):
        return self.heads[f's{int(source)}_q{int(task)}']

    def residuals(self, h):
        h = h.detach()
        return torch.stack([torch.stack([self.head(s,q)(h) for q in range(2)],-1)
                            for s in range(3)],1)

    def local_logits(self, h, z, domain):
        h, z = h.detach(), z.detach()
        # This inference helper is deliberately separate from pilot_loss.
        values = self.residuals(h)
        own = values[torch.arange(len(h),device=h.device),domain.long()]
        return z + own

    def pilot_loss(self, h, z, y, domain, denominator=None):
        h, z, y = h.detach(), z.detach(), y.detach()
        denominator = len(h) if denominator is None else int(denominator)
        losses=[]
        for s in range(3):
            mask = domain == s
            if bool(mask.any()):
                for q in range(2):
                    losses.append(F.binary_cross_entropy_with_logits(
                        z[mask,q]+self.head(s,q)(h[mask]), y[mask,q], reduction='sum'))
        return torch.stack(losses).sum()/(2*denominator)


ResidualBank = PilotBank


class FunctionBank(nn.Module):
    """Independent source/intended-target/task functions, or strong local copies."""
    def __init__(self, pilot: PilotBank, seed_fn=None, kind='SOURCE', pooled=None):
        super().__init__()
        if kind not in ('SOURCE','LOCAL_CAP','POOL'):
            raise ValueError(kind)
        self.kind=kind
        self.heads=nn.ModuleDict()
        self.owner_map=[]
        for q in range(2):
            for s in range(3):
                targets = list(range(4)) if kind=='LOCAL_CAP' else [t for t in range(3) if t != s]
                for slot,t in enumerate(targets):
                    key=f's{s}_t{t}_q{q}'
                    original=pooled[q] if kind=='POOL' else pilot.head(s,q)
                    head=copy.deepcopy(original).cpu()
                    head.requires_grad_(True)
                    for p in head.parameters(): p.grad=None
                    gen=torch.Generator(device='cpu').manual_seed(_seed(seed_fn,s,q,slot))
                    with torch.no_grad():
                        head.output.weight.add_(torch.randn(head.output.weight.shape,generator=gen)*1e-3)
                    self.heads[key]=head
                    self.owner_map.append({'key':key,'source':s,'intended_target':t,
                                           'slot':slot,'task':q,'kind':kind})

    def head(self, source, intended_target, task):
        return self.heads[f's{int(source)}_t{int(intended_target)}_q{int(task)}']

    def formation_loss(self,h,z,y,domain,weights=None,denominator=None):
        h,z,y=h.detach(),z.detach(),y.detach()
        denominator=len(h) if denominator is None else int(denominator)
        terms=[]
        for owner in self.owner_map:
            s,t,q,slot=(owner[k] for k in ('source','intended_target','task','slot'))
            mask=torch.ones(len(h),device=h.device,dtype=torch.bool) if self.kind=='POOL' else domain==s
            if not bool(mask.any()): continue
            loss=F.binary_cross_entropy_with_logits(z[mask,q]+self.head(s,t,q)(h[mask]),y[mask,q],reduction='none')
            if weights is not None and self.kind=='SOURCE': loss=loss*weights[mask,q,slot].detach()
            terms.append(loss.sum())
        divisor=12 if self.kind=='POOL' else 4
        return torch.stack(terms).sum()/(divisor*denominator)

    def residuals(self,h):
        if self.kind=='LOCAL_CAP':
            raise ValueError('LOCAL_CAP has four local slots, not two source intents')
        h=h.detach()
        return torch.stack([torch.stack([torch.stack([
            self.head(s,t,q)(h) for q in range(2)],-1)
            for t in range(3) if t!=s],1) for s in range(3)],1)

    def candidate_slots(self,target):
        if self.kind=='LOCAL_CAP': return [(int(target),k) for k in range(4)]
        return [(s,t) for s in range(3) if s!=target for t in range(3) if t!=s]

    def candidates(self,h,z,domain,pilot):
        h,z=h.detach(),z.detach()
        result=torch.empty((len(h),2,5),device=h.device,dtype=z.dtype)
        with torch.no_grad():
            for target in range(3):
                mask=domain==target
                if not bool(mask.any()):continue
                for q in range(2):
                    values=[z[mask,q]+pilot.head(target,q)(h[mask])]
                    values.extend(z[mask,q]+self.head(s,t,q)(h[mask]) for s,t in self.candidate_slots(target))
                    result[mask,q]=torch.stack(values,-1)
        return result


def optimizer_owner_receipt(bank,optimizer):
    out=[]
    for owner in bank.owner_map:
        head=bank.heads[owner['key']]
        states=[optimizer.state.get(p,{}) for p in head.parameters()]
        out.append({**owner,'parameter_steps':[int(s.get('step',0)) for s in states],
                    'm_norm':float(sum(s.get('exp_avg',torch.tensor(0.)).double().square().sum().item() for s in states)**.5),
                    'v_norm':float(sum(s.get('exp_avg_sq',torch.tensor(0.)).double().square().sum().item() for s in states)**.5)})
    return out
