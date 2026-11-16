"""Natural pair-completed risks with unchanged source Adam and EMA."""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.auxiliary_objective_attribution_20261115.protocol import state_digest
from experiments.auxiliary_objective_attribution_20261115.stage2.core import synchronize
from experiments.conditional_information_momentum_20261102_v3.protocol import update_ema
from experiments.task_view_gradient_residual_momentum_20261116.core import source_clean_step


ARMS = ('NQR-025', 'BRIER-OPPORTUNITY-025', 'MEAN-GAP-025')
_Q = dict(zip(ARMS, ('Q_R', 'Q_B', 'Q_M')))
_STATISTICS = ('mu0', 'mu1', 'var0', 'var1', 'bbar0', 'bbar1',
               'S', 'Q_R', 'Q_B', 'Q_M', 'risk')


def _validate_support(targets, domains, arm, mixture_lambda):
    if arm not in ARMS or mixture_lambda not in (0., .25):
        raise ValueError('Technical PAUSE: unregistered arm or mixture_lambda')
    if targets.ndim != 2 or targets.shape[1] != 2 or len(targets) == 0:
        raise ValueError('Technical PAUSE: targets must have nonempty shape [B,2]')
    if domains.shape != (len(targets),) or domains.device != targets.device:
        raise ValueError('Technical PAUSE: domains must be colocated [B]')
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError('Technical PAUSE: labels must be binary')
    if not bool(((domains == 0) | (domains == 1) | (domains == 2)).all()):
        raise ValueError('Technical PAUSE: domain values must be integral 0,1,2')


def _support_ledger(targets, domains, mixture_lambda):
    B = len(targets)
    cells = {}
    for d in range(3):
        mask = domains == d
        n = int(mask.sum())
        for t in range(2):
            n1 = int((targets[mask, t] == 1).sum())
            n0 = n - n1
            pi0, pi1 = (n0/n, n1/n) if n else (None, None)
            weight = n/(2*B)
            bce_mass = weight*(1-mixture_lambda+mixture_lambda*(pi0**2+pi1**2)) if n else 0.
            opposite_mass = weight*mixture_lambda*2*pi0*pi1 if n else 0.
            cells[f'd{d}t{t}'] = dict(
                n=n, n0=n0, n1=n1, pi0=pi0, pi1=pi1, weight=weight,
                bce_mass=bce_mass, opposite_mass=opposite_mass,
                bce_coefficient0=(1-mixture_lambda+mixture_lambda*pi0)/(2*B) if n0 else None,
                bce_coefficient1=(1-mixture_lambda+mixture_lambda*pi1)/(2*B) if n1 else None,
                **dict.fromkeys(_STATISTICS))
    opposite_mass = sum(c['opposite_mass'] for c in cells.values())
    return dict(schema_version=1, B=B, entries=2*B, mixture_lambda=mixture_lambda,
                total_mass=1., task_mass={'0': .5, '1': .5},
                bce_mass=1.-opposite_mass, opposite_mass=opposite_mass, cells=cells)


def _branch(ledger):
    if ledger['mixture_lambda'] == 0.:
        return 'lambda_zero'
    if all(not (c['n0'] and c['n1']) for c in ledger['cells'].values()):
        return 'all_single_class'
    return 'ordinary'


def loss_components(probabilities, targets, domains, *, arm, mixture_lambda=.25):
    """Return differentiable L/C/A and the detached six-cell coefficient ledger."""
    _validate_support(targets, domains, arm, mixture_lambda)
    if (probabilities.shape != targets.shape or probabilities.device != targets.device
            or probabilities.dtype != targets.dtype
            or probabilities.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Technical PAUSE: probabilities/targets must match in shape, device and floating dtype')
    if not bool((torch.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError('Technical PAUSE: invalid probabilities')
    clean = F.binary_cross_entropy(probabilities, targets, reduction='mean')
    ledger = _support_ledger(targets, domains, mixture_lambda)
    if _branch(ledger) != 'ordinary':
        return dict(loss=clean, clean_loss=clean, completed_loss=clean, ledger=ledger)
    entries = F.binary_cross_entropy(probabilities, targets, reduction='none')
    completed = clean.new_zeros(())
    for d in range(3):
        for t in range(2):
            cell = ledger['cells'][f'd{d}t{t}']
            if not cell['n']:
                continue
            means, variances, bmeans = {}, {}, {}
            for a in (0, 1):
                if not cell[f'n{a}']:
                    continue
                mask = (domains == d) & (targets[:, t] == a)
                scores = probabilities[mask, t]
                means[a] = scores.mean()
                variances[a] = (scores-means[a]).square().mean()
                bmeans[a] = entries[mask, t].mean()
                for key, value in ((f'mu{a}', means[a]), (f'var{a}', variances[a]),
                                   (f'bbar{a}', bmeans[a])):
                    cell[key] = float(value.detach())
            same = sum(cell[f'pi{a}']**2*bmeans[a] for a in bmeans)
            risk = same
            if len(means) == 2:
                gap = (1-means[1]+means[0]).square()
                variance = variances[1]+variances[0]
                qs = dict(Q_R=.5*(gap+variance),
                          Q_B=(1-means[1]).square()+means[0].square()+variance,
                          Q_M=.5*gap)
                risk = same+2*cell['pi1']*cell['pi0']*qs[_Q[arm]]
                cell.update({name: float(value.detach()) for name, value in qs.items()})
            cell.update(S=float(same.detach()), risk=float(risk.detach()))
            completed = completed+cell['weight']*risk
    loss = (1-mixture_lambda)*clean+mixture_lambda*completed
    return dict(loss=loss, clean_loss=clean, completed_loss=completed, ledger=ledger)


def _finite(tensors, label):
    if any(not bool(torch.isfinite(t).all()) for t in tensors if t is not None):
        raise FloatingPointError(f'Technical PAUSE: nonfinite {label}; discard unacknowledged transaction')


def _state_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _state_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _state_tensors(item)


def _diagnostics(named, before, optimizer):
    def reduce_group(items):
        active = [(name, p) for name, p in items if p.grad is not None]
        steps = [int(optimizer.state[p]['step'].item()) for _, p in active]
        def norm(values):
            squares = [v.detach().double().square().sum() for v in values]
            return math.sqrt(float(torch.stack(squares).sum())) if squares else 0.
        return dict(data_gradient_norm=norm([p.grad for _, p in active]),
                    parameter_update_norm=norm([p.detach()-before[name] for name, p in items]),
                    active_parameters=len(active), none_parameters=len(items)-len(active),
                    parameter_step_min=min(steps) if steps else None,
                    parameter_step_max=max(steps) if steps else None)
    result = reduce_group(named)
    result['parameter_groups'] = {}
    for prefix in sorted({name.split('.')[0] for name, _ in named}):
        items = [(name, p) for name, p in named if name.split('.')[0] == prefix]
        result['parameter_groups'][prefix] = dict(
            coordinates=sum(p.numel() for _, p in items),
            parameter_names=[name for name, _ in items], **reduce_group(items))
    return result


def train_step(model, optimizer, ema, x, y, config, context, row_ids=None, diagnose=False):
    """Perform one ordinary scalar backward, source Adam step and fixed EMA update."""
    synchronize(x.device)
    started = time.perf_counter()
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != len(y) or x.device != y.device:
        raise ValueError('Technical PAUSE: x must have source layout [F,B] colocated with y')
    if y.dtype != torch.float32:
        raise ValueError('Technical PAUSE: runtime targets must be FP32')
    arm, mixture_lambda = config['arm'], config['mixture_lambda']
    _validate_support(y, x[-1], arm, mixture_lambda)
    ledger = _support_ledger(y, x[-1], mixture_lambda)
    branch = _branch(ledger)
    ids = np.arange(len(y), dtype=np.int64) if row_ids is None else np.asarray(row_ids, dtype=np.int64)
    if ids.shape != (len(y),):
        raise ValueError('Technical PAUSE: row_ids must have shape [B]')
    named = list(model.named_parameters())
    before = {name: p.detach().clone() for name, p in named} if diagnose else None
    if branch != 'ordinary':
        value = source_clean_step(model, optimizer, ema, x, y, config)
        if not math.isfinite(value):
            raise FloatingPointError('Technical PAUSE: nonfinite source-clean loss')
        _finite([p.grad for _, p in named], 'source gradients')
        _finite(_state_tensors(model.state_dict()), 'source model')
        _finite(_state_tensors(optimizer.state_dict()), 'source Adam')
        _finite(_state_tensors(ema.state_dict()), 'source EMA')
        values = dict(loss=value, clean_loss=value, completed_loss=value)
    else:
        optimizer.zero_grad(set_to_none=True)
        probabilities = model(x)
        if probabilities.dtype != torch.float32:
            raise ValueError('Technical PAUSE: runtime probabilities must be FP32')
        components = loss_components(probabilities, y, x[-1], arm=arm, mixture_lambda=mixture_lambda)
        _finite([components[k] for k in ('loss', 'clean_loss', 'completed_loss')], 'loss')
        components['loss'].backward()
        _finite([p.grad for _, p in named], 'gradients')
        optimizer.step()
        update_ema(ema, model, config['rho'])
        ledger = components['ledger']
        values = {key: float(components[key].detach()) for key in ('loss', 'clean_loss', 'completed_loss')}
    diagnostics = _diagnostics(named, before, optimizer) if diagnose else None
    row_order_hash = state_digest(ids)
    synchronize(x.device)
    return dict(**values, branch=branch, ledger=ledger, row_order_hash=row_order_hash,
                optimizer_updates=1, ema_updates=1, ordinary_forwards=1, scalar_backwards=1,
                probe_forwards=0, view_forwards=0, algorithm_rng_draws=0,
                seconds=time.perf_counter()-started, diagnostics=diagnostics)
