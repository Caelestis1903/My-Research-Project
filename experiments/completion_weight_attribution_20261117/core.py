"""Cycle04 completion BCE and matched mass scaling, with source Adam/EMA."""
import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.auxiliary_objective_attribution_20261115.protocol import (
    cpu_copy, optimizer as source_optimizer, rng_state, state_digest)
from experiments.auxiliary_objective_attribution_20261115.stage2.core import synchronize
from experiments.conditional_information_momentum_20261102_v3.protocol import update_ema
from experiments.natural_quadratic_ranking_20261116.core import loss_components as nqr_components
from experiments.task_view_gradient_residual_momentum_20261116.core import source_clean_step

ARMS = ('COMPLETION-BCE-025', 'MASS-SCALE-025')
OBJECTIVES = ('C', 'S', 'W', 'N')
CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))
STATISTICS = ('mu0', 'mu1', 'var0', 'var1', 'bbar0', 'bbar1', 'H', 'Q_R')
_DATA = {'movielens': (20, 21927, 1e-4, .9828205985452511),
         'kuairand': (49, 40052, 1e-5, .995)}


def _support(targets, domains, arm, mixture_lambda):
    if arm not in ARMS or isinstance(mixture_lambda, bool) or mixture_lambda not in (0., .25):
        raise ValueError('Technical PAUSE: unregistered arm/lambda')
    if (targets.ndim != 2 or targets.shape[1] != 2 or len(targets) == 0
            or targets.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Technical PAUSE: targets require nonempty FP32/FP64 [B,2]')
    if domains.shape != (len(targets),) or domains.device != targets.device:
        raise ValueError('Technical PAUSE: domains require colocated [B]')
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError('Technical PAUSE: nonbinary labels')
    if not bool(((domains == 0) | (domains == 1) | (domains == 2)).all()):
        raise ValueError('Technical PAUSE: domains require integral 0/1/2')
    B = len(targets)
    cells = {}
    for d in range(3):
        mask = domains == d
        n = int(mask.sum())
        for t in range(2):
            n1 = int((targets[mask, t] == 1).sum())
            n0 = n-n1
            p0, p1 = (n0/n, n1/n) if n else (None, None)
            w = n/(2*B)
            opportunity = 2*p0*p1 if n else 0.
            cells[f'd{d}t{t}'] = dict(n=n, n0=n0, n1=n1, pi0=p0, pi1=p1,
                weight=w, opportunity=opportunity,
                removed_opposite_mass=w*mixture_lambda*opportunity,
                completion_bce_mass=w*(1-mixture_lambda*opportunity), opposite_mass=0.)
    r = sum(c['removed_opposite_mass'] for c in cells.values())
    m = 1-r
    branch = ('lambda_zero' if mixture_lambda == 0 else
              'all_single_class' if all(not(c['n0'] and c['n1']) for c in cells.values())
              else 'ordinary')
    for c in cells.values():
        c['scale_bce_mass'] = m*c['weight']
        c['bce_mass'] = c['completion_bce_mass'] if arm == ARMS[0] else c['scale_bce_mass']
        for a in (0, 1):
            completion = (1-mixture_lambda+mixture_lambda*c[f'pi{a}'])/(2*B) if c[f'n{a}'] else None
            scale = m/(2*B) if c[f'n{a}'] else None
            c[f'completion_coefficient{a}'] = completion
            c[f'scale_coefficient{a}'] = scale
            c[f'bce_coefficient{a}'] = completion if arm == ARMS[0] else scale
    completion_task = {str(t): sum(cells[f'd{d}t{t}']['completion_bce_mass'] for d in range(3))
                       for t in range(2)}
    return dict(schema_version=1, arm=arm, branch=branch, B=B, entries=2*B,
        mixture_lambda=mixture_lambda, bce_mass_scale=m, removed_opposite_mass=r,
        total_mass=m, bce_mass=m, opposite_mass=0.,
        task_mass=completion_task.copy() if arm == ARMS[0] else {'0': m/2, '1': m/2},
        completion_task_mass=completion_task, cells=cells)


def _statistics(ledger, probabilities, targets, domains, diagnose):
    entries = F.binary_cross_entropy(probabilities.detach(), targets, reduction='none') if diagnose else None
    for d in range(3):
        for t in range(2):
            c = ledger['cells'][f'd{d}t{t}']
            statuses = {}
            for key in STATISTICS:
                a = int(key[-1]) if key[-1] in '01' else None
                reason = ('empty_cell' if not c['n'] else
                          'absent_class' if a is not None and not c[f'n{a}'] else
                          'requires_both_classes' if key == 'Q_R' and not(c['n0'] and c['n1']) else
                          'observed' if diagnose else 'not_scheduled')
                c[key] = None
                statuses[key] = reason
            if diagnose and c['n']:
                means, variances, bmeans = {}, {}, {}
                for a in (0, 1):
                    if c[f'n{a}']:
                        mask = (domains == d) & (targets[:, t] == a)
                        scores = probabilities.detach()[mask, t]
                        means[a] = scores.mean()
                        variances[a] = (scores-means[a]).square().mean()
                        bmeans[a] = entries[mask, t].mean()
                        for key, value in ((f'mu{a}', means[a]), (f'var{a}', variances[a]),
                                           (f'bbar{a}', bmeans[a])):
                            c[key] = float(value)
                c['H'] = float(sum(c[f'pi{a}']**2*bmeans[a] for a in bmeans))
                if len(means) == 2:
                    c['Q_R'] = float(.5*((1-means[1]+means[0]).square()+variances[1]+variances[0]))
            c['statistics_status'] = statuses
    ledger.update(diagnose=bool(diagnose), statistics_status='observed' if diagnose else 'not_scheduled')


def loss_components(probabilities, targets, domains, *, arm, mixture_lambda=.25, diagnose=False):
    ledger = _support(targets, domains, arm, mixture_lambda)
    if (probabilities.shape != targets.shape or probabilities.device != targets.device
            or probabilities.dtype != targets.dtype):
        raise ValueError('Technical PAUSE: probabilities/targets shape/device/dtype mismatch')
    if not bool((torch.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError('Technical PAUSE: invalid probabilities')
    clean = F.binary_cross_entropy(probabilities, targets, reduction='mean')
    same = completion = scaled = clean
    if ledger['branch'] == 'ordinary':
        entries = F.binary_cross_entropy(probabilities, targets, reduction='none')
        same = clean.new_zeros(())
        for d in range(3):
            for t in range(2):
                c = ledger['cells'][f'd{d}t{t}']
                if c['n']:
                    h = sum(c[f'pi{a}']**2*entries[(domains == d) & (targets[:, t] == a), t].mean()
                            for a in (0, 1) if c[f'n{a}'])
                    same = same+c['weight']*h
        completion = (1-mixture_lambda)*clean+mixture_lambda*same
        scaled = ledger['bce_mass_scale']*clean
    nqr = nqr_components(probabilities, targets, domains, arm='NQR-025',
                         mixture_lambda=mixture_lambda)['loss'] if diagnose else None
    _statistics(ledger, probabilities, targets, domains, diagnose)
    return dict(loss=completion if arm == ARMS[0] else scaled, clean_loss=clean,
                same_class_loss=same, completion_loss=completion, scaled_loss=scaled,
                nqr_loss=nqr, branch=ledger['branch'], ledger=ledger)


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'Technical PAUSE: invalid {label}')


def _validate_call(x, y, config, context, row_ids, diagnose):
    if (x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != len(y)
            or x.device != y.device or y.dtype != torch.float32):
        raise ValueError('Technical PAUSE: runtime requires FP32 targets and source [F,B]')
    dataset = config.get('dataset')
    if dataset not in _DATA or context.get('dataset') != dataset:
        raise ValueError('Technical PAUSE: dataset mismatch')
    K, tail, wd, rho = _DATA[dataset]
    fixed = dict(cycle_id='cycle_04', objective='completion_weight_attribution_v1',
                 epochs=50, batch_size=40960, precision='float32', lr=.01,
                 betas=[.9, .999], eps=1e-8, weight_decay=wd, rho=rho,
                 steps_per_epoch=K, tail_rows=tail, loader_seed_offset=11000,
                 gate_training_detached=True, probe_gate_full_derivative=False, algorithm_rng_draws=0)
    if any(config.get(k) != v for k, v in fixed.items()) or any(
            any(part in k for part in ('schedule', 'phase_switch', 'off_updates')) for k in config):
        raise ValueError('Technical PAUSE: config differs from Cycle04')
    for key in ('data_hash', 'selection_hash'):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f'Technical PAUSE: missing {key}')
    _integer(context.get('seed'), 'seed', 2022)
    if context['seed'] not in range(2022, 2027):
        raise ValueError('Technical PAUSE: seed outside 2022..2026')
    purpose = context.get('purpose')
    if purpose == 'formal':
        if context.get('clock_kind') != 'formal' or config['mixture_lambda'] != .25:
            raise ValueError('Technical PAUSE: invalid formal purpose/lambda')
        for k in ('epoch', 'batch', 'global_step'):
            _integer(context.get(k), k, 1)
        e, b = context['epoch'], context['batch']
        if (e > 50 or b > K or context['global_step'] != (e-1)*K+b
                or len(y) != (tail if b == K else 40960)
                or bool(diagnose) != (b == 1 and e in (1, 10, 30, 50))):
            raise ValueError('Technical PAUSE: formal clock/cardinality/diagnostic mismatch')
    elif purpose in ('engineering_smoke', 'engineering_check'):
        if context.get('clock_kind') != 'engineering' or any(k in context for k in ('epoch', 'global_step')):
            raise ValueError('Technical PAUSE: invalid engineering clock')
        for k in ('engineering_call_ordinal', 'engineering_state_step', 'sample_epoch', 'sample_batch'):
            _integer(context.get(k), k, 1)
        if not isinstance(context.get('case_id'), str) or not isinstance(context.get('branch_role'), str):
            raise ValueError('Technical PAUSE: missing engineering identity')
        if context['sample_epoch'] != 1 or context['sample_batch'] not in (1, K):
            raise ValueError('Technical PAUSE: engineering sample clock mismatch')
        if purpose == 'engineering_smoke' and len(y) != (tail if context['sample_batch'] == K else 40960):
            raise ValueError('Technical PAUSE: engineering cardinality mismatch')
        if purpose == 'engineering_check' and context.get('fixture_rows') != len(y):
            raise ValueError('Technical PAUSE: CPU fixture cardinality must be explicit')
    else:
        raise ValueError('Technical PAUSE: unsupported purpose')
    if diagnose and (not isinstance(context.get('pre_loader_hash'), str) or not context['pre_loader_hash']):
        raise ValueError('Technical PAUSE: diagnosed call requires actual pre_loader_hash')
    if row_ids is None and purpose != 'engineering_check':
        raise ValueError('Technical PAUSE: real call requires source row IDs')
    ids = np.arange(len(y), dtype=np.int64) if row_ids is None else np.asarray(row_ids, dtype=np.int64)
    if ids.shape != (len(y),):
        raise ValueError('Technical PAUSE: row IDs require [B]')
    return _support(y, x[-1], config['arm'], config['mixture_lambda']), state_digest(ids)


def _tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _tensors(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from _tensors(v)


def _finite(values, label):
    if any(not bool(torch.isfinite(v).all()) for v in values if v is not None):
        raise FloatingPointError(f'Technical PAUSE: nonfinite {label}')


def parameter_activity(named):
    return {n: 'none' if p.grad is None else 'tensor_zero' if bool((p.grad == 0).all()) else 'nonzero'
            for n, p in named}


def adam_step_counters(named, optimizer):
    return {n: int(optimizer.state[p]['step'].item()) if 'step' in optimizer.state.get(p, {}) else None
            for n, p in named}


def _vector_metrics(left, right=None):
    """Full-coordinate norms: unused entries contribute zero, never a new .grad."""
    left_sq = sum(float(v.double().square().sum()) for v in left if v is not None)
    result = dict(norm=math.sqrt(left_sq))
    if right is not None:
        right_sq = sum(float(v.double().square().sum()) for v in right if v is not None)
        dot = sum(float(a.double().mul(b.double()).sum()) for a, b in zip(left, right)
                  if a is not None and b is not None)
        difference_sq = sum(float((a.double()-b.double()).square().sum()) if a is not None and b is not None
                            else float((a if a is not None else b).double().square().sum())
                            if a is not None or b is not None else 0. for a, b in zip(left, right))
        result.update(right_norm=math.sqrt(right_sq), difference_norm=math.sqrt(difference_sq),
                      cosine=dot/math.sqrt(left_sq*right_sq) if left_sq and right_sq else None,
                      null_reason=None if left_sq and right_sq else 'zero_norm')
    return result


def _step_summary(steps):
    present = [v for v in steps if v is not None]
    return dict(min=min(present) if present else None, max=max(present) if present else None,
                uninitialized=len(steps)-len(present))


def _group_metrics(named, gradients, updates, before_steps, after_steps, rho):
    groups = {'ALL': list(range(len(named)))}
    for prefix in sorted({n.split('.')[0] for n, _ in named}):
        groups[prefix] = [i for i, (n, _) in enumerate(named) if n.split('.')[0] == prefix]
    output = {}
    for group, indices in groups.items():
        sizes = [named[i][1].numel() for i in indices]
        objectives, pairs = {}, {}
        for objective in OBJECTIVES:
            grads = [gradients[objective][i] for i in indices]
            active = [v is not None for v in grads]
            zero = [v is not None and bool((v == 0).all()) for v in grads]
            objectives[objective] = dict(
                data_gradient_norm=_vector_metrics(grads)['norm'],
                parameter_update_norm=_vector_metrics([updates[objective][i] for i in indices])['norm'],
                active_parameters=sum(active), none_parameters=len(indices)-sum(active),
                tensor_zero_parameters=sum(zero), nonzero_parameters=sum(active)-sum(zero),
                active_coordinates=sum(s for s, a in zip(sizes, active) if a),
                none_coordinates=sum(s for s, a in zip(sizes, active) if not a),
                tensor_zero_coordinates=sum(s for s, z in zip(sizes, zero) if z),
                nonzero_coordinates=sum(s for s, a, z in zip(sizes, active, zero) if a and not z),
                pre_private_steps=_step_summary([before_steps[named[i][0]] for i in indices]),
                post_private_steps=_step_summary([after_steps[objective][named[i][0]] for i in indices]))
        for left, right in (('W', 'S'), ('N', 'W')):
            a, b = [gradients[left][i] for i in indices], [gradients[right][i] for i in indices]
            grad_metrics = _vector_metrics(a, b)
            update_metrics = _vector_metrics([updates[left][i] for i in indices], [updates[right][i] for i in indices])
            activity_diff = [(v is None) != (w is None) for v, w in zip(a, b)]
            pairs[f'{left}-{right}'] = dict(data_gradient=grad_metrics, parameter_update=update_metrics,
                activity_mask_difference_parameters=sum(activity_diff),
                activity_mask_difference_coordinates=sum(s for s, differs in zip(sizes, activity_diff) if differs),
                ema_difference_norm=(1-rho)*update_metrics['difference_norm'],
                ema_relationship='deltaEMA=(1-rho)*deltaTheta', ema_relationship_algebraic=True)
        output[group] = dict(coordinates=sum(sizes), parameter_names=[named[i][0] for i in indices],
                             objectives=objectives, pairs=pairs)
    return output


def train_step(model, optimizer, ema, x, y, config, context, row_ids=None, diagnose=False):
    ledger, row_hash = _validate_call(x, y, config, context, row_ids, diagnose)
    synchronize(x.device)
    started = time.perf_counter()
    named = list(model.named_parameters())
    before_steps = adam_step_counters(named, optimizer)
    host = 'W' if config['arm'] == ARMS[0] else 'S'
    components = None
    gradients, updates, post_steps, virtual_hashes = {}, {}, {}, {}
    diagnostic_seconds = 0.
    pre = None
    if diagnose:
        pre = dict(model_hash=state_digest(model.state_dict()), optimizer_hash=state_digest(optimizer.state_dict()),
                   rng_hash=state_digest(rng_state(x.device)), loader_hash=context['pre_loader_hash'])
        before_parameters = {n: p.detach().cpu().clone() for n, p in named}
        before_optimizer = cpu_copy(optimizer.state_dict())
        immutable_optimizer_hash = state_digest(before_optimizer)

    def observe(probabilities):
        nonlocal components, diagnostic_seconds
        tick = time.perf_counter()
        components = loss_components(probabilities, y, x[-1], arm=config['arm'],
                                     mixture_lambda=config['mixture_lambda'], diagnose=diagnose)
        _finite([components[k] for k in ('loss', 'clean_loss', 'same_class_loss', 'completion_loss', 'scaled_loss', 'nqr_loss')], 'loss')
        if diagnose:
            losses = dict(C=components['clean_loss'], S=components['scaled_loss'],
                          W=components['completion_loss'], N=components['nqr_loss'])
            for objective in OBJECTIVES:
                if objective != host:
                    gradients[objective] = tuple(None if g is None else g.detach().clone()
                        for g in torch.autograd.grad(losses[objective], tuple(p for _, p in named),
                                                     allow_unused=True, retain_graph=True))
            synchronize(x.device)
            diagnostic_seconds += time.perf_counter()-tick

    def virtual_steps():
        nonlocal diagnostic_seconds
        tick = time.perf_counter()
        gradients[host] = tuple(None if p.grad is None else p.grad.detach().clone() for _, p in named)
        for gs in gradients.values():
            _finite(gs, 'diagnostic gradient')
        m = components['ledger']['bce_mass_scale']
        for g, s in zip(gradients['C'], gradients['S']):
            if (g is None) != (s is None) or (g is not None and not torch.allclose(s, m*g, atol=2e-6, rtol=2e-5)):
                raise AssertionError('Technical PAUSE: g_S != m*g_C')
        observed_rng = state_digest(rng_state(x.device))
        live_optimizer_hash = state_digest(optimizer.state_dict())
        for objective in OBJECTIVES:
            clone = copy.deepcopy(model)
            clone_optimizer = source_optimizer(clone, config)
            clone_optimizer.load_state_dict(cpu_copy(before_optimizer))
            clone_named = list(clone.named_parameters())
            for (_, p), grad in zip(clone_named, gradients[objective]):
                p.grad = None if grad is None else grad.detach().clone()
            clone_optimizer.step()
            _finite(_tensors(clone.state_dict()), 'virtual model')
            _finite(_tensors(clone_optimizer.state_dict()), 'virtual Adam')
            virtual_hashes[objective] = dict(model_hash=state_digest(clone.state_dict()),
                                             optimizer_hash=state_digest(clone_optimizer.state_dict()))
            updates[objective] = tuple(p.detach().cpu()-before_parameters[n] for n, p in clone_named)
            post_steps[objective] = adam_step_counters(clone_named, clone_optimizer)
            del clone_named, clone_optimizer, clone
        if (state_digest(before_optimizer) != immutable_optimizer_hash
                or state_digest(optimizer.state_dict()) != live_optimizer_hash
                or state_digest(rng_state(x.device)) != observed_rng):
            raise AssertionError('Technical PAUSE: diagnostic clone mutated live/prestate/RNG')
        synchronize(x.device)
        diagnostic_seconds += time.perf_counter()-tick

    if ledger['branch'] != 'ordinary':
        if diagnose:
            handles = []
            def forward_observer(module, args, output):
                handles[0].remove()  # The observation closure must never enter a model clone.
                observe(output)
                return None
            def step_observer(opt, args, kwargs):
                virtual_steps()
                return None
            handles.append(model.register_forward_hook(forward_observer))
            handles.append(optimizer.register_step_pre_hook(step_observer))
            try:
                value = source_clean_step(model, optimizer, ema, x, y, config)
            finally:
                for handle in handles:
                    handle.remove()
        else:
            value = source_clean_step(model, optimizer, ema, x, y, config)
        if components is None:
            _statistics(ledger, None, y, x[-1], False)
            values = dict.fromkeys(('loss', 'clean_loss', 'same_class_loss', 'completion_loss', 'scaled_loss'), value)
            values['nqr_loss'] = None
    else:
        optimizer.zero_grad(set_to_none=True)
        observe(model(x))
        components['loss'].backward()
        _finite([p.grad for _, p in named], 'live gradients')
        if diagnose:
            virtual_steps()
        optimizer.step()
        update_ema(ema, model, config['rho'])
    if components is not None:
        ledger = components['ledger']
        values = {k: float(components[k].detach()) if components[k] is not None else None
                  for k in ('loss', 'clean_loss', 'same_class_loss', 'completion_loss', 'scaled_loss', 'nqr_loss')}
    if any(v is not None and not math.isfinite(v) for v in values.values()):
        raise FloatingPointError('Technical PAUSE: nonfinite scalar loss')
    for label, state in (('model', model.state_dict()), ('Adam', optimizer.state_dict()), ('EMA', ema.state_dict())):
        _finite(_tensors(state), label)
    _finite([p.grad for _, p in named], 'live gradient')
    activity = parameter_activity(named)
    steps = adam_step_counters(named, optimizer)
    for n, p in named:
        expected = before_steps[n] if p.grad is None else (before_steps[n] or 0)+1
        if steps[n] != expected:
            raise AssertionError(f'Technical PAUSE: private Adam step mismatch: {n}')
    diagnostics = None
    if diagnose:
        actual = dict(model_hash=state_digest(model.state_dict()), optimizer_hash=state_digest(optimizer.state_dict()))
        if actual != virtual_hashes[host]:
            raise AssertionError('Technical PAUSE: actual/virtual source Adam mismatch')
        actual_updates = tuple(p.detach().cpu()-before_parameters[n] for n, p in named)
        diff = _vector_metrics(actual_updates, updates[host])['difference_norm']
        if diff != 0.:
            raise AssertionError('Technical PAUSE: actual/virtual nonzero update difference')
        diagnostics = dict(schema_version=1, host_arm=config['arm'], counterfactual_at_host_state=True,
            row_order_hash=row_hash, pre_model_hash=pre['model_hash'], pre_optimizer_hash=pre['optimizer_hash'],
            pre_rng_hash=pre['rng_hash'], pre_loader_hash=pre['loader_hash'],
            objective_losses=dict(C=values['clean_loss'], S=values['scaled_loss'], W=values['completion_loss'], N=values['nqr_loss']),
            bce_mass_scale=ledger['bce_mass_scale'], removed_opposite_mass=ledger['removed_opposite_mass'],
            actual_objective=host, actual_post=actual, virtual_post=virtual_hashes,
            actual_virtual_exact=True, actual_virtual_update_difference_norm=diff,
            extra_autograd_sweeps=3, virtual_adam_steps=4,
            parameter_groups=_group_metrics(named, gradients, updates, before_steps, post_steps, config['rho']))
    synchronize(x.device)
    return dict(**values, branch=ledger['branch'], ledger=ledger, row_order_hash=row_hash,
        optimizer_updates=1, ema_updates=1, ordinary_forwards=1, scalar_backwards=1,
        probe_forwards=0, view_forwards=0, algorithm_rng_draws=0,
        extra_autograd_sweeps=3 if diagnose else 0, virtual_adam_steps=4 if diagnose else 0,
        seconds=time.perf_counter()-started, diagnostic_seconds=diagnostic_seconds,
        diagnostics=diagnostics, parameter_activity=activity, adam_step_counters=steps)
