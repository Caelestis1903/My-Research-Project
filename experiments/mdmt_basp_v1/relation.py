"""Measured, signed benefit tables and a NULL-preserving reading policy."""
from __future__ import annotations
import torch
from torch import nn


def effective_sample_size(mass):
    return mass.sum(0).square() / mass.square().sum(0).clamp_min(1e-30)


def weighted_low_rank(values, weights, rank=2, iterations=5, ridge=.001):
    """Fixed five-step weighted ALS; missing cells carry zero weight."""
    dtype = values.dtype
    y, w = values.double(), weights.double()
    m, n = y.shape
    r = min(rank, m, n)
    if not r:
        return values.clone()
    # SVD is deterministic and does not turn missing observations into evidence.
    u, s, vh = torch.linalg.svd(torch.where(w > 0, y, 0.), full_matrices=False)
    a = u[:, :r] * s[:r].sqrt()
    b = vh[:r].T * s[:r].sqrt()
    eye = torch.eye(r, device=y.device, dtype=y.dtype)
    for _ in range(iterations):
        for i in range(m):
            a[i] = torch.linalg.solve(b.T @ (w[i, :, None] * b) + ridge*eye,
                                      b.T @ (w[i]*y[i]))
        for j in range(n):
            b[j] = torch.linalg.solve(a.T @ (w[:, j, None] * a) + ridge*eye,
                                      a.T @ (w[:, j]*y[:, j]))
    return (a @ b.T).to(dtype)


class GainTable(nn.Module):
    def __init__(self, tasks, leaves, actions):
        super().__init__()
        shape = (tasks, leaves, actions)
        for name in ('observed_gain', 'policy_gain', 'second_moment'):
            self.register_buffer(name, torch.zeros(shape))
        self.register_buffer('ess', torch.zeros(leaves))
        self.register_buffer('unique_count', torch.zeros(shape, dtype=torch.long))
        self.register_buffer('observed', torch.zeros(shape, dtype=torch.bool))
        self.register_buffer('prior', torch.zeros(shape, dtype=torch.bool))
        self.register_buffer('prior_refreshes', torch.zeros(shape, dtype=torch.long))
        self.version = None

    @torch.no_grad()
    def measure(self, gain, mass, row_ids, minimum_ess=64, low_rank=False, version=None):
        if len(torch.unique(torch.as_tensor(row_ids))) != len(row_ids):
            raise ValueError('Probe rows must be unique within a teacher measurement')
        denom = mass.sum(0)
        self.ess = effective_sample_size(mass)
        self.unique_count = (mass > 0).sum(0)[None, :, None].expand_as(self.unique_count).clone()
        first = torch.einsum('nk,nta->tka', mass, gain) / denom[None, :, None].clamp_min(1e-30)
        second = torch.einsum('nk,nta->tka', mass, gain.square()) / denom[None, :, None].clamp_min(1e-30)
        observed = (self.ess >= minimum_ess)[None, :, None].expand_as(first)
        inherited = self.prior & (self.prior_refreshes < 1) & ~observed
        self.observed_gain = torch.where(observed, first, torch.zeros_like(first))
        self.second_moment = torch.where(observed, second, torch.zeros_like(second))
        self.policy_gain = torch.where(observed, first,
                                       torch.where(inherited, self.policy_gain, 0.))
        self.observed = observed.clone()
        self.prior_refreshes += self.prior.long()
        self.prior = inherited
        if low_rank:
            for task in range(len(first)):
                weight = self.ess[:, None] * observed[task]
                estimate = weighted_low_rank(first[task], weight)
                self.policy_gain[task] = torch.where(observed[task], estimate, self.policy_gain[task])
        self.version = version

    @torch.no_grad()
    def split_need(self, parent):
        for name in ('observed_gain', 'policy_gain', 'second_moment', 'observed', 'prior', 'prior_refreshes', 'unique_count'):
            x = getattr(self, name)
            y = torch.cat((x, x[:, parent:parent+1].clone()), 1)
            if name == 'observed':
                y[:, parent] = False
                y[:, -1] = False
            elif name == 'prior':
                y[:, parent] = True
                y[:, -1] = True
            elif name in ('prior_refreshes', 'unique_count'):
                y[:, parent] = 0
                y[:, -1] = 0
            setattr(self, name, y)
        for name in ('ess',):
            x = getattr(self, name)
            x = torch.cat((x, x[parent:parent+1].clone()))
            x[parent] = 0
            x[-1] = 0
            setattr(self, name, x)

    @torch.no_grad()
    def split_source(self, parent):
        for name in ('observed_gain', 'policy_gain', 'second_moment', 'observed', 'prior', 'prior_refreshes', 'unique_count'):
            x = getattr(self, name)
            y = torch.cat((x, x[..., parent:parent+1].clone()), -1)
            if name == 'observed':
                y[..., parent] = False
                y[..., -1] = False
            elif name == 'prior':
                y[..., parent] = True
                y[..., -1] = True
            elif name in ('prior_refreshes', 'unique_count'):
                y[..., parent] = 0
                y[..., -1] = 0
            setattr(self, name, y)


def reading_policy(gain, prior, scale, uniform=False):
    """gain[T,K,A], prior[B,A] -> weights[B,T,K,A], NULL[B,T,K]."""
    if uniform:
        alpha = prior[:, None, None, :].expand(-1, gain.shape[0], gain.shape[1], -1) / 2
        return alpha, torch.full_like(alpha[..., 0], .5)
    utility = (gain/scale).clamp_min(0)
    numerator = prior[:, None, None, :] * utility[None]
    denominator = 1 + numerator.sum(-1, keepdim=True)
    return numerator / denominator, denominator.squeeze(-1).reciprocal()
