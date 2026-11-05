"""Fifteen independent non-feedback parameter estimators on one online trajectory."""
import hashlib
import torch
from .directed_ema import update


class ShadowBank:
    def __init__(self, theta0, candidates, rho0, dataset, seed):
        self.candidates, self.rho0 = candidates, rho0
        self.phi = {c['id']: theta0.clone() for c in candidates}
        key = f'AH1105|{dataset}|{seed}|0|B_SIGN|0|0'
        value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')%(2**63-1)
        generator = torch.Generator().manual_seed(value)
        self.signs = (2*torch.randint(2, theta0.shape, generator=generator)-1).to(theta0)
        self.q = {k: dict(D=theta0.new_empty((len(theta0), 0)), M=theta0.new_empty((len(theta0), 0))) for k in (10, 20)}
        self.last_refresh = {k: 0 for k in self.q}
        self.updates = 0

    @staticmethod
    def refresh_due(step, cadence=10):
        return step >= 1 and (step-1)%cadence == 0

    @torch.no_grad()
    def after_step(self, context):
        step, theta = context['step'], context['theta']
        if step != self.updates+1:
            raise ValueError('Shadow update is not consecutive')
        response = context.get('response')
        if self.refresh_due(step):
            if response is None:
                raise ValueError('Missing required response refresh')
            for k in self.q:
                if self.refresh_due(step, k):
                    self.q[k] = {family: response[family].detach() for family in ('D', 'M')}
                    self.last_refresh[k] = step
        for candidate in self.candidates:
            phi = self.phi[candidate['id']]
            h, family = candidate['half_life_ratio'], candidate['family']
            if family == 'U':
                rho = self.rho0**(1/h)
                phi.mul_(rho).add_(theta, alpha=1-rho)
            else:
                q = self.q[candidate['refresh_steps']]['M' if family == 'M' else 'D']
                if family == 'S':
                    q = q*self.signs[:, None]
                update(phi, theta, q, self.rho0, h)
        self.updates = step

    def state_dict(self):
        return dict(phi=self.phi, signs=self.signs, q=self.q,
                    last_refresh=self.last_refresh, updates=self.updates)

    def load_state_dict(self, value, device):
        self.phi = {k: v.to(device) for k, v in value['phi'].items()}
        self.signs = value['signs'].to(device)
        self.q = {int(k): {f: q.to(device) for f, q in row.items()} for k, row in value['q'].items()}
        self.last_refresh = value['last_refresh'].copy()
        self.updates = value['updates']
