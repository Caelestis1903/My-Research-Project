"""Detached diagnostics attached to the actual post-backward transaction."""
import math

import numpy as np
import torch

from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import CONDITIONS

GROUPS = ('STAR_SKIP', 'EXPERT')
ANCHOR_EPOCHS = (5, 20, 35, 50)
COMMON_EPOCHS = (10, 25, 50)
S2_ARMS = {'C1_Z050', 'C1_Z100', 'C2_A050', 'C2_A100'}


def selected_work(spec):
    return dict(d1=spec.arm == 'H0' or spec.arm in S2_ARMS, d2=spec.arm == 'H0',
                d4=spec.arm in {'B1', 'H0'} | S2_ARMS)


def pair_record(pair, domains, labels):
    joint = (2*labels[:, 0]+labels[:, 1]).astype(int)
    counts = np.bincount(4*domains.astype(int)+joint, minlength=12)
    row = dict(pair_event=pair is not None, condition_id=None, source_cell=None, joint_boundary=None,
               active_conditions=[], pair_index_hash=None, w=None, q=None, w_over_q=None,
               selected_support=None, joint_counts=counts.tolist(),
               cell_mass=(counts.reshape(3,4).sum(1).repeat(2)/(2*len(labels))).tolist())
    if pair is not None:
        d, t, a, b = CONDITIONS[pair['selected']]
        row.update(condition_id=pair['selected'], source_cell=f'd{d}t{t}', joint_boundary=[a,b],
                   active_conditions=pair['active'], pair_index_hash=pair['pair_index_hash'],
                   w=pair['w'].tolist(), q=pair['q'].tolist(),
                   w_over_q=(pair['w']/pair['q']).tolist(), selected_support=[int(counts[4*d+a]), int(counts[4*d+b])])
    return row


def _inner(left, right):
    return (left.double()*right.double()).sum()


def _l2(value):
    return _inner(value, value).sqrt()


def _cos(left, right):
    radius = _l2(left)*_l2(right)
    return _inner(left, right)/radius


class HistoryObserver:
    """S_DENSE has one native clock and one PAIR clock across both dense groups."""
    def __init__(self, spec, optimizer):
        self.spec = spec
        self.names = dict((group, []) for group in GROUPS)
        self.positions = {}
        self.state = {}
        self.step = self.events = 0
        for index, (name, param) in enumerate(optimizer.shared):
            group = 'EXPERT' if name.startswith('expert.') else 'STAR_SKIP'
            self.positions[name] = (group, len(self.names[group]))
            self.names[group].append((index, name, param))
        self._before = {}
        self.last = None

    def state_dict(self):
        return dict(version=1, step=self.step, events=self.events, groups=self.state)

    def load_state_dict(self, state):
        if state['version'] != 1:
            raise ValueError('Unsupported observer recurrence')
        self.step, self.events = state['step'], state['events']
        self.state = {group: {key: value.to(self.names[group][0][2].device)
                             for key, value in values.items()} for group, values in state['groups'].items()}

    def vector(self, name, field):
        group, index = self.positions[name]
        sizes = [p.numel() for _, _, p in self.names[group]]
        return self.state[group][field].split(sizes)[index].reshape(self.names[group][index][2].shape)

    @torch.no_grad()
    def observe(self, pair, raw):
        self.step += 1
        active = pair is not None
        self.events += int(active)
        ratio = float(pair['w'][pair['index']]/pair['q'][pair['index']]) if active else 0.
        wd, lam, beta, zeta = self.spec.weight_decay, self.spec.lambda_pair, self.spec.beta_pair, self.spec.zeta
        self.last = []
        for group, items in self.names.items():
            if not items or any(p.grad is None or (active and (raw is None or raw[i] is None)) for i, _, p in items):
                raise RuntimeError('S_DENSE observation requires exactly the registered dense gradient support')
            theta = torch.cat([p.detach().flatten() for _, _, p in items])
            self._before[group] = theta
            grad = torch.cat([p.grad.detach().flatten() for _, _, p in items])
            mixed = grad.add(theta, alpha=wd)
            r = torch.cat([raw[i].flatten() for i, _, _ in items]) if active else torch.zeros_like(theta)
            b = mixed-r.mul(lam*ratio) if active else mixed
            h = r.mul(ratio)
            lh = h.mul(lam)
            fields = ('vmix', 'vsep', 'vactive', 'cross_bce', 'cross_wd', 'q', 'u0', 'uactive')
            if group not in self.state:
                self.state[group] = {name: torch.zeros_like(theta) for name in fields}
            s = self.state[group]
            s['vmix'].mul_(.999).addcmul_(mixed, mixed, value=.001)
            sep = b.square().add_(lh.square())
            s['vsep'].mul_(.999)
            if active:
                s['vsep'].add_(sep,alpha=.001)
            else:
                s['vsep'].addcmul_(mixed,mixed,value=.001)
            if zeta:
                s['vactive'].mul_(.999)
                if active:
                    s['vactive'].add_(mixed.square().mul_(1-zeta).add_(sep,alpha=zeta),alpha=.001)
                else:
                    s['vactive'].addcmul_(mixed,mixed,value=.001)
            g_wd = theta.mul(wd)
            s['cross_bce'].mul_(.999).addcmul_(b-g_wd, lh, value=.002)
            s['cross_wd'].mul_(.999).addcmul_(g_wd, lh, value=.002)
            native_denom = 1-.999**self.step
            d0 = (s['vmix']/native_denom).sqrt().add_(1e-8).reciprocal_()
            d_active = (s['vactive']/native_denom).sqrt().add_(1e-8).reciprocal_() if zeta else d0
            if active:
                complement = .01 if beta == .99 else .1
                s['q'].mul_(beta).add_(r, alpha=complement*ratio)
                s['u0'].mul_(beta).addcmul_(d0,h,value=complement)
                s['uactive'].mul_(beta).addcmul_(d_active,h,value=complement)
            bc = 1-beta**self.events if self.events else 1.
            q = s['q']/bc
            dq = q*d_active
            u = s['uactive']/bc
            exchange = dq-u
            cross = s['vmix']-s['vsep']
            norm_cross = cross/(s['vmix']+s['vsep']+1e-30)
            rebuilt = s['cross_bce']+s['cross_wd']
            vals = dict(b_l2=_l2(b), lambda_h_l2=_l2(lh), q_hat_l2=_l2(q), d_q_hat_l2=_l2(dq),
                        u_hat_l2=_l2(u), h_q_cosine=_cos(h,q), cross_signed_mean=cross.mean(dtype=torch.float64),
                        cross_rms=_l2(cross)/math.sqrt(cross.numel()), cross_positive=(cross>0).mean(dtype=torch.float64),
                        cross_negative=(cross<0).mean(dtype=torch.float64), cross_normalized_mean=norm_cross.mean(dtype=torch.float64),
                        cross_normalized_rms=_l2(norm_cross)/math.sqrt(cross.numel()),
                        cross_bce_mean=s['cross_bce'].mean(dtype=torch.float64), cross_bce_rms=_l2(s['cross_bce'])/math.sqrt(cross.numel()),
                        cross_wd_mean=s['cross_wd'].mean(dtype=torch.float64), cross_wd_rms=_l2(s['cross_wd'])/math.sqrt(cross.numel()),
                        cross_reconstruction_max_abs=(cross-rebuilt).abs().max(),
                        exchange_l2=_l2(exchange), exchange_relative_l2=_l2(exchange)/_l2(dq), exchange_cosine=_cos(dq,u),
                        exchange_dq_cosine=_cos(exchange,dq))
            vals = {key: float(value) if math.isfinite(value) else None
                    for key, value in zip(vals, torch.stack(tuple(vals.values())).cpu().tolist())}
            self.last.append(dict(parameter_group=group, tau=self.step, k=self.events, **vals))

    @torch.no_grad()
    def finish(self):
        exchange_sq = sum(r['exchange_l2']**2 for r in self.last)
        for record in self.last:
            group = record['parameter_group']
            theta = torch.cat([p.detach().flatten() for _, _, p in self.names[group]])
            record.update(displacement_l2=float(_l2(theta.double()-self._before[group].double())),
                          exchange_squared_fraction=record['exchange_l2']**2/exchange_sq if exchange_sq else None)
        self._before = {}
        return self.last
