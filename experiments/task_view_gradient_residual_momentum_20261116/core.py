"""Clean torch Adam plus an independently accumulated residual channel."""
import hashlib
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from experiments.auxiliary_objective_attribution_20261115.core import forward
from experiments.auxiliary_objective_attribution_20261115.stage2.core import make_views, synchronize
from experiments.auxiliary_objective_attribution_20261115.protocol import state_digest
from experiments.conditional_information_momentum_20261102_v3.protocol import update_ema


def norm(tensors):
    values = [v.detach().double().square().sum() for v in tensors if v is not None]
    return math.sqrt(float(torch.stack(values).sum())) if values else 0.


def cosine(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    denominator = norm([x for x, _ in pairs])*norm([y for _, y in pairs])
    if not denominator:
        return None
    return float(torch.stack([(x.detach().double()*y.detach().double()).sum()
                              for x, y in pairs]).sum())/denominator


def rademacher(parameter, context, name):
    key = '|'.join(str(context[k]) for k in ('dataset', 'seed', 'epoch', 'batch'))
    key += '|'+name+'|residual_update_rademacher_v1'
    generator = torch.Generator(device=parameter.device)
    generator.manual_seed(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))
    return torch.randint(0, 2, parameter.shape, device=parameter.device,
                         generator=generator, dtype=torch.int8).to(parameter.dtype).mul_(2).sub_(1)


class ResidualChannel:
    """Adam owns clean m/v/counters; q uses the corresponding parameter counter."""
    def __init__(self, model, optimizer, config):
        self.named = list(model.named_parameters())
        self.optimizer = optimizer
        self.config = config
        self.q = {}

    def state_dict(self):
        return {'q': self.q}

    def load_state_dict(self, state):
        by_name = dict(self.named)
        self.q = {name: value.to(by_name[name].device).clone() for name, value in state['q'].items()}

    @torch.no_grad()
    def step(self, clean, view, context, diagnose=False):
        opt = self.optimizer
        assert len(opt.param_groups) == 1
        group = opt.param_groups[0]
        beta1, beta2 = group['betas']
        active = [(name, p, gc, gv) for (name, p), gc, gv in zip(self.named, clean, view) if gc is not None]
        for (_, p), gradient in zip(self.named, clean):
            p.grad = gradient
        # Preserve the source Adam operator order, including coupled WD and None handling.
        opt.step()
        residuals = []; uc = []; ur = []; counters = []
        for name, p, gc, gv in active:
            residual = -gc if gv is None else gv-gc
            state = opt.state[p]; n = int(state['step'].item()); counters.append(n)
            denominator = (state['exp_avg_sq']/(1-beta2**n)).sqrt().add_(group['eps'])
            if self.config['residual_mode'] == 'instant':
                qhat = residual
            else:
                q = self.q.setdefault(name, torch.zeros_like(p))
                q.mul_(beta1).add_(residual, alpha=1-beta1)
                qhat = q/(1-beta1**n)
            residuals.append(residual)
            uc.append((state['exp_avg']/(1-beta1**n))/denominator)
            ur.append(qhat/denominator)
        clean_norm = norm(uc); residual_norm = norm(ur)
        scale = min(1., self.config['gamma']*clean_norm/residual_norm) if residual_norm else 0.
        target = norm([u*scale for u in ur])
        dimension = sum(p.numel() for _, p, _, _ in active)
        if self.config['residual_mode'] == 'shuffle':
            injection = [rademacher(p, context, name).mul_(target/math.sqrt(dimension))
                         if target else torch.zeros_like(p) for name, p, _, _ in active]
        else:
            injection = [u*scale for u in ur]
        injected = norm(injection)
        error = abs(injected-target)/target if target else 0.
        assert error <= 1e-5 and (target != 0. or injected == 0.)
        assert injected <= self.config['gamma']*clean_norm*(1+1e-5)+1e-30
        for (_, p, _, _), value in zip(active, injection):
            p.add_(value, alpha=-group['lr'])
        record = dict(active_parameters=len(active), clean_none_parameters=len(clean)-len(active),
            view_none_active_parameters=sum(gv is None for _, _, _, gv in active),
            view_only_parameters=sum(gc is None and gv is not None for gc, gv in zip(clean, view)),
            clean_update_norm=clean_norm, residual_preconditioned_norm=residual_norm,
            cap_scale=scale, cap_active=bool(residual_norm and scale < 1.),
            residual_injection_norm=injected, residual_parameter_update_norm=group['lr']*injected,
            shuffle_target_norm=target if self.config['residual_mode'] == 'shuffle' else None,
            shuffle_relative_error=error if self.config['residual_mode'] == 'shuffle' else None,
            shuffle_stream=dict(context, stream='residual_update_rademacher_v1')
                if self.config['residual_mode'] == 'shuffle' else None,
            parameter_step_min=min(counters) if counters else None,
            parameter_step_max=max(counters) if counters else None)
        if diagnose:
            states = [opt.state[p] for _, p, _, _ in active]
            record['gradient_diagnostics'] = dict(clean_norm=norm(clean), view_norm=norm(view),
                residual_norm=norm(residuals), q_norm=norm(list(self.q.values())),
                m_norm=norm([s['exp_avg'] for s in states]), v_norm=norm([s['exp_avg_sq'] for s in states]),
                clean_view_cosine=cosine(clean, view),
                clean_residual_cosine=cosine([gc for _, _, gc, _ in active], residuals),
                q_residual_cosine=cosine([self.q.get(name) for name, _, _, _ in active], residuals),
                parameter_groups={})
            for prefix in sorted({name.split('.')[0] for name, _, _, _ in active}):
                indices = [i for i, (name, _, _, _) in enumerate(active) if name.split('.')[0] == prefix]
                record['gradient_diagnostics']['parameter_groups'][prefix] = dict(
                    parameter_names=[active[i][0] for i in indices],
                    coordinates=sum(active[i][1].numel() for i in indices),
                    clean_norm=norm([active[i][2] for i in indices]),
                    view_norm=norm([active[i][3] for i in indices]), residual_norm=norm([residuals[i] for i in indices]),
                    q_norm=norm([self.q.get(active[i][0]) for i in indices]),
                    m_norm=norm([states[i]['exp_avg'] for i in indices]),
                    v_norm=norm([states[i]['exp_avg_sq'] for i in indices]),
                    step_min=min(counters[i] for i in indices), step_max=max(counters[i] for i in indices))
        return record


def view_losses(model, x, y, deltas):
    clean_probability = model(x)
    clean = F.binary_cross_entropy(clean_probability, y)
    views = [forward(model, x, delta)[0] for delta in deltas]
    view = .5*sum(F.binary_cross_entropy(views[t][:, t], y[:, t]) for t in range(2))
    return clean, view


def support_ledger(x, y):
    B = len(y)
    cells = {f'd{d}t{t}y{label}': int(((x[-1] == d)&(y[:, t] == label)).sum())
             for d in range(3) for t in range(2) for label in (0, 1)}
    assert sum(cells.values()) == 2*B
    return dict(B=B, clean_entries=2*B, view_entries=2*B, clean_denominator=2*B,
                view_denominator=2*B, clean_entry_coefficient=1/(2*B),
                view_entry_coefficient=1/(2*B), cells=cells)


def source_clean_step(model, optimizer, ema, x, y, config):
    optimizer.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy(model(x), y)
    assert torch.isfinite(loss)
    loss.backward(); optimizer.step(); update_ema(ema, model, config['rho'])
    return float(loss.detach())


def train_step(model, optimizer, channel, ema, x, y, config, context, row_ids=None, diagnose=False):
    synchronize(x.device); started = time.perf_counter()
    if config['gamma'] == 0.:
        # No view/probe/random work, and no additional source RNG/loader consumption.
        clean_value = source_clean_step(model, optimizer, ema, x, y, config)
        record = dict(clean_loss=clean_value, view_loss=None, probe_seconds=0., construction_seconds=0.)
    else:
        optimizer.zero_grad(set_to_none=True)
        deltas, _, views_info, probe_seconds, construction_seconds = make_views(model, x, y, config['alpha'])
        clean, view = view_losses(model, x, y, deltas)
        assert torch.isfinite(clean) and torch.isfinite(view)
        parameters = [p for _, p in channel.named]
        gc = torch.autograd.grad(clean, parameters, allow_unused=True)
        gv = torch.autograd.grad(view, parameters, allow_unused=True)
        record = channel.step(gc, gv, context, diagnose)
        update_ema(ema, model, config['rho'])
        record.update(clean_loss=float(clean.detach()), view_loss=float(view.detach()),
                      probe_seconds=probe_seconds, construction_seconds=construction_seconds,
                      view_diagnostics=views_info)
    synchronize(x.device)
    ids = np.arange(len(y), dtype=np.int64) if row_ids is None else np.asarray(row_ids, dtype=np.int64)
    record.update(ledger=support_ledger(x, y), row_order_hash=state_digest(ids),
                  optimizer_updates=1, ema_updates=1, seconds=time.perf_counter()-started)
    return record
