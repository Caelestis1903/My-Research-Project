"""Frozen CP update: repair the real current term, retain native PAIR history."""
from experiments.conditional_information_momentum_20261102_v3.optimizers import PairOptimizer
from experiments.mdmt_history_information_20261103.optimizers import norm, project_history
import torch


class SerialCPOptimizer(PairOptimizer):
    def __init__(self, model, config):
        if config['mode'] != 'eccm' or config['lambda_pair'] != .05 or config['dynamic']:
            raise ValueError('CP requires the frozen static P01 configuration')
        super().__init__(model, config)
        self.global_step = 0

    @torch.no_grad()
    def step(self, pair=None, r=None, probes=None, diagnostic=False):
        self.global_step += 1
        lam = self.config['lambda_pair']
        ratio = 0.
        if pair is not None:
            if r is None or len(r) != len(self.shared):
                raise ValueError('PAIR requires one gradient for every shared parameter')
            sid = pair['selected']
            q = float(pair['q'][pair['index']])
            ratio = float(pair['w'][pair['index']] / q)
            self.selected[sid] = self.selected.get(sid, 0) + 1
            for i in pair['active']:
                self.active[i] = self.active.get(i, 0) + 1
            for h, g in zip(self._table(sid), r):
                h.lerp_(g, .01 / q)

        current, preconditioner, moments = [], [], []
        for k, (name, p) in enumerate(self.shared):
            assert p.grad is not None, ('missing shared gradient', name)
            if name not in self.moments:
                self.moments[name] = dict(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
            state = self.moments[name]
            state['step'] += 1
            assert state['step'] == self.global_step
            mixed = p.grad.add(p, alpha=self.config['weight_decay'])
            real = mixed if pair is None else mixed - lam * ratio * r[k]
            state['v'].mul_(.999).addcmul_(mixed, mixed, value=.001)
            current.append(real)
            preconditioner.append((state['v'] / (1 - .999 ** state['step'])).sqrt().add_(1e-8).reciprocal())
            moments.append(state)
        c_mix = self.config['c_mix']
        repaired, info = project_history(current, preconditioner, probes or [], c_mix['kappa'], c_mix['gamma'])
        history_terms, numerators = [], []
        for k, ((_, p), state, corrected, scale) in enumerate(zip(self.shared, moments, repaired, preconditioner)):
            state['m'].mul_(.9).add_(corrected, alpha=.1)
            numerator = state['m'] / (1 - .9 ** state['step'])
            history = torch.zeros_like(p)
            if pair is not None:
                for i, w in zip(pair['active'], pair['w']):
                    history.add_(self._table(i)[k], alpha=lam * float(w) / (1 - .99 ** self.active[i]))
            numerator.add_(history)
            p.add_(scale * numerator, alpha=-self.config['lr'])
            if diagnostic:
                history_terms.append(history)
                numerators.append(numerator)
        self.adam.step()
        if diagnostic:
            info.update(raw_gradient_norm=float(norm(current)), repaired_gradient_norm=float(norm(repaired)),
                        real_moment_norm=float(norm([s['m'] / (1 - .9 ** s['step']) for s in moments])),
                        pair_history_term_norm=float(norm(history_terms)),
                        second_moment_norm=float(norm([s['v'] for s in moments])),
                        shared_update_norm=self.config['lr'] * float(norm([d * n for d, n in zip(preconditioner, numerators)])))
            info['update_probe_dots'] = [[-self.config['lr'] * float(sum(((d * n).double() * g.double()).sum()
                for d, n, g in zip(preconditioner, numerators, gradients))) for gradients in pair_probe]
                for pair_probe in (probes or [])]
        return info

    def state_dict(self):
        return dict(super().state_dict(), global_step=self.global_step)

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.global_step = state['global_step']
