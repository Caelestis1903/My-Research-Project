"""Static collapsed and long mixed-Adam controls with same-trajectory shadows."""
import torch

from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import CONDITIONS
from experiments.conditional_information_momentum_20261102_v3.optimizers import PairOptimizer
from .protocol import v3


def history_config(arm, dataset, lam=.05):
    if arm not in ('A21', 'A22') or lam not in (0., .05):
        raise ValueError((arm, lam))
    if lam == 0:
        return v3.config('A0', dataset, 0.)
    cfg = v3.config('PCOLL' if arm == 'A21' else 'P00', dataset, lam)
    return dict(cfg, arm=arm, dynamic=False, mode='collapsed' if arm == 'A21' else 's99',
                history_scope='S_dense')


class HistoryOptimizer(PairOptimizer):
    """The v3 step requests raw r when mode != adam. Its native Adam owns S99."""
    def __init__(self, model, config):
        super().__init__(model, config)
        if self.mode == 's99':
            self.adam.add_param_group(dict(params=[p for _, p in self.shared], betas=(.99, .999)))
        self.shadow = {}
        self.on_pre_step = None

    def state_dict(self):
        return dict(super().state_dict(), shadow=self.shadow)

    def load_state_dict(self, state):
        super().load_state_dict(state)
        device = self.parameters[0].device
        self.shadow = {n:{k:t.to(device).clone() for k,t in group.items()}
                       for n,group in state['shadow'].items()}

    @torch.no_grad()
    def step(self, pair=None, r=None):
        if self.on_pre_step is not None:
            self.on_pre_step(pair, r)
        if pair is not None:
            d, task, _, _ = CONDITIONS[pair['selected']]
            cell = 2 * d + task
            idx = pair['index']
            ratio = float(pair['w'][idx] / pair['q'][idx])
        for k, (name, p) in enumerate(self.shared):
            if p.grad is None:
                raise RuntimeError(f'Missing S_dense gradient for {name}')
            if name not in self.shadow:
                cells = lambda: p.new_zeros((6, *p.shape))
                state = dict(M9=cells())
                if self.mode == 'collapsed':
                    state['Q'] = cells()
                else:
                    state.update(M99=cells(), R9=torch.zeros_like(p), R99=torch.zeros_like(p))
                self.shadow[name] = state
            s = self.shadow[name]
            s['M9'].mul_(.9)
            if pair is not None:
                s['M9'][cell].add_(r[k], alpha=.1*self.config['lambda_pair']*ratio)
            if self.mode == 'collapsed':
                if pair is not None:
                    s['Q'].mul_(.99)
                    s['Q'][cell].add_(r[k], alpha=.01*ratio)
            else:
                real = p.grad.add(p, alpha=self.config['weight_decay'])
                s['M99'].mul_(.99)
                if pair is not None:
                    scaled = self.config['lambda_pair'] * ratio
                    s['M99'][cell].add_(r[k], alpha=.01*scaled)
                    real = real - scaled*r[k]
                s['R9'].mul_(.9).add_(real, alpha=.1)
                s['R99'].mul_(.99).add_(real, alpha=.01)
        super().step(pair, r)

    def reconstruction_error(self):
        errors = []
        for k,(name,p) in enumerate(self.shared):
            s = self.shadow[name]
            if self.mode == 'collapsed':
                if not self.pair_clock:
                    continue
                actual, rebuilt = self.history[-1][k], s['Q'].sum(0)
            else:
                actual = self.adam.state[p]['exp_avg']
                rebuilt = s['R99'] + s['M99'].sum(0)
            torch.testing.assert_close(actual, rebuilt, rtol=1e-4, atol=1e-6)
            errors.append((actual-rebuilt).abs().max())
        return float(torch.stack(errors).max()) if errors else 0.


def make_history_optimizer(model, config):
    if config['lambda_pair'] == 0:
        return v3.make_optimizer(model, v3.config('A0', config['dataset'], 0.))
    return HistoryOptimizer(model, config)
