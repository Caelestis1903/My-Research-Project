"""Cycle05 cell-mass BCE with frozen C/W/S reductions and source Adam/EMA."""
import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.auxiliary_objective_attribution_20261115.protocol import (
    cpu_copy, digest, load_protocol, optimizer as source_optimizer, rng_state, state_digest)
from experiments.auxiliary_objective_attribution_20261115.stage2.core import synchronize
from experiments.conditional_information_momentum_20261102_v3.protocol import update_ema
from experiments.completion_weight_attribution_20261117.core import (
    _support, _statistics, _tensors, _finite, _vector_metrics, _step_summary,
    parameter_activity, adam_step_counters, loss_components as frozen_components)
from experiments.task_view_gradient_residual_momentum_20261116.core import source_clean_step

ARMS = ('CELL-MASS-BCE-025',)
OBJECTIVES = ('C', 'S', 'U', 'W')
CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))
STATISTICS = ('mu0', 'mu1', 'var0', 'var1', 'bbar0', 'bbar1', 'H', 'Q_R')
_DATA = {'movielens': (20, 21927, 1e-4, .9828205985452511),
         'kuairand': (49, 40052, 1e-5, .995)}
_DATA_HASH = {d: digest(p['source']) for d, p in load_protocol()['datasets'].items()}
SELECTION_HASH = 'c67a9771fc5980f76f3fd4cfe0afd29a260ab987bbe816865452fbdc69590b89'
LOSS_FIELDS = ('loss', 'clean_loss', 'same_class_loss', 'completion_loss', 'scaled_loss', 'cell_mass_loss')


def _ledger(targets, domains, arm, mixture_lambda):
    if arm not in ARMS:
        raise ValueError('Technical PAUSE: unregistered U arm')
    return _augment_ledger(_support(targets, domains, 'COMPLETION-BCE-025', mixture_lambda))


def _augment_ledger(ledger):
    """Keep W/U cell mass as the same computed scalar, including its roundoff."""
    ledger['arm'] = ARMS[0]
    B, lam, m = ledger['B'], ledger['mixture_lambda'], ledger['bce_mass_scale']
    for c in ledger['cells'].values():
        c['w'] = c['weight']
        c['a'] = 1-lam*c['opportunity'] if c['n'] else None
        c['kappa'] = c['pi0']**2+c['pi1']**2 if c['n'] else None
        c['natural_bce_mass'] = c['weight']
        c['cell_mass_bce_mass'] = c['completion_bce_mass']
        c['bce_mass'] = c['cell_mass_bce_mass']
        c['coefficient_status'] = {}
        for a in (0, 1):
            present = bool(c[f'n{a}'])
            c[f'q{a}'] = 1-lam+lam*c[f'pi{a}'] if present else None
            c[f'natural_coefficient{a}'] = 1/(2*B) if present else None
            c[f'cell_mass_coefficient{a}'] = c['a']/(2*B) if present else None
            c[f'bce_coefficient{a}'] = c[f'cell_mass_coefficient{a}']
            for policy in ('natural', 'completion', 'cell_mass', 'scale'):
                c[f'{policy}_class_mass{a}'] = c[f'n{a}']*c[f'{policy}_coefficient{a}'] if present else 0.
            c['coefficient_status'][str(a)] = 'observed' if present else 'absent_class' if c['n'] else 'empty_cell'
        c['support_status'] = 'observed' if c['n'] else 'empty_cell'
    ledger['cell_mass_task_mass'] = ledger['completion_task_mass'].copy()
    ledger['scale_task_mass'] = {'0': m/2, '1': m/2}
    ledger['natural_task_mass'] = {'0': .5, '1': .5}
    ledger['task_mass'] = ledger['cell_mass_task_mass'].copy()
    ledger['shared_completion_cell_mass_exact'] = all(
        c['cell_mass_bce_mass'] == c['completion_bce_mass'] for c in ledger['cells'].values())
    return ledger


def loss_components(probabilities, targets, domains, *, arm=ARMS[0], mixture_lambda=.25, diagnose=False):
    if arm not in ARMS:
        raise ValueError('Technical PAUSE: unregistered U arm')
    frozen = frozen_components(probabilities, targets, domains, arm='COMPLETION-BCE-025',
                               mixture_lambda=mixture_lambda, diagnose=False)
    ledger = _augment_ledger(frozen['ledger'])
    _statistics(ledger, probabilities, targets, domains, diagnose)
    clean = frozen['clean_loss']
    cell_mass = clean
    if ledger['branch'] == 'ordinary':
        entries = F.binary_cross_entropy(probabilities, targets, reduction='none')
        cell_mass = entries.new_zeros(())
        for d in range(3):
            for t in range(2):
                c = ledger['cells'][f'd{d}t{t}']
                if c['n']:
                    cell_mass = cell_mass+(c['weight']*c['a'])*entries[domains == d, t].mean()
    mixed_delegate = ledger['branch'] == 'lambda_zero' and any(c['n0'] and c['n1'] for c in ledger['cells'].values())
    return dict(loss=cell_mass, clean_loss=clean,
        same_class_loss=None if mixed_delegate else frozen['same_class_loss'],
        same_class_loss_null_reason='not_computed_delegate' if mixed_delegate else None,
        completion_loss=frozen['completion_loss'], scaled_loss=frozen['scaled_loss'],
        cell_mass_loss=cell_mass, branch=ledger['branch'], ledger=ledger)


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
    fixed = dict(cycle_id='cycle_05', objective='cell_mass_bce_v1', arm=ARMS[0],
                 epochs=50, batch_size=40960, precision='float32', lr=.01,
                 betas=[.9, .999], eps=1e-8, weight_decay=wd, rho=rho,
                 steps_per_epoch=K, tail_rows=tail, loader_seed_offset=11000,
                 gate_training_detached=True, probe_gate_full_derivative=False, algorithm_rng_draws=0)
    if any(config.get(k) != v for k, v in fixed.items()) or any(
            any(part in k for part in ('schedule', 'switch', 'off_update')) for k in config):
        raise ValueError('Technical PAUSE: config differs from Cycle05')
    for key in ('data_hash', 'selection_hash'):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f'Technical PAUSE: missing {key}')
    if config['data_hash'] != _DATA_HASH[dataset] or config['selection_hash'] != SELECTION_HASH:
        raise ValueError('Technical PAUSE: data/selection identity mismatch')
    if config.get('mixture_lambda') not in (0., .25) or isinstance(config.get('mixture_lambda'), bool):
        raise ValueError('Technical PAUSE: unregistered lambda')
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
    if not isinstance(context.get('pre_loader_hash'), str) or not context['pre_loader_hash']:
        raise ValueError('Technical PAUSE: call requires actual pre_loader_hash')
    if row_ids is None and purpose != 'engineering_check':
        raise ValueError('Technical PAUSE: real call requires source row IDs')
    ids = np.arange(len(y), dtype=np.int64) if row_ids is None else np.asarray(row_ids, dtype=np.int64)
    if ids.shape != (len(y),):
        raise ValueError('Technical PAUSE: row IDs require [B]')
    if purpose == 'engineering_check' and x.device.type != 'cpu':
        raise ValueError('Technical PAUSE: engineering_check is CPU only')
    return _ledger(y, x[-1], config['arm'], config['mixture_lambda']), state_digest(ids)


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
                tensor_zero_coordinates=sum(int((g == 0).sum()) for g in grads if g is not None),
                nonzero_coordinates=sum(int((g != 0).sum()) for g in grads if g is not None),
                pre_private_steps=_step_summary([before_steps[named[i][0]] for i in indices]),
                post_private_steps=_step_summary([after_steps[objective][named[i][0]] for i in indices]))
        for left, right in (('U', 'W'), ('U', 'S'), ('S', 'C')):
            a, b = [gradients[left][i] for i in indices], [gradients[right][i] for i in indices]
            grad_metrics = _vector_metrics(a, b)
            update_metrics = _vector_metrics([updates[left][i] for i in indices], [updates[right][i] for i in indices])
            activity_diff = [(v is None) != (w is None) for v, w in zip(a, b)]
            pairs[f'{left}-{right}'] = dict(data_gradient=grad_metrics, parameter_update=update_metrics,
                activity_mask_difference_parameters=sum(activity_diff),
                activity_mask_difference_coordinates=sum(s for s, differs in zip(sizes, activity_diff) if differs),
                ema_difference_norm=(1-rho)*update_metrics['difference_norm'],
                ema_relationship='deltaEMA=(1-rho)*deltaTheta', ema_relationship_algebraic=True)
        output[group] = dict(coordinates=sum(sizes), objectives=objectives, pairs=pairs)
    return output


def train_step(model, optimizer, ema, x, y, config, context, row_ids=None, diagnose=False):
    ledger, row_hash = _validate_call(x, y, config, context, row_ids, diagnose)
    synchronize(x.device)
    started = time.perf_counter()
    named = list(model.named_parameters())
    before_steps = adam_step_counters(named, optimizer)
    host = 'U'
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
        immutable_parameters_hash = state_digest(before_parameters)

    def observe(probabilities):
        nonlocal components, diagnostic_seconds
        tick = time.perf_counter()
        components = loss_components(probabilities, y, x[-1], arm=config['arm'],
                                     mixture_lambda=config['mixture_lambda'], diagnose=diagnose)
        _finite([components[k] for k in LOSS_FIELDS], 'loss')
        if diagnose:
            losses = dict(C=components['clean_loss'], S=components['scaled_loss'],
                          U=components['cell_mass_loss'], W=components['completion_loss'])
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
        if (state_digest(before_parameters) != immutable_parameters_hash
                or state_digest(before_optimizer) != immutable_optimizer_hash
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
            values = dict.fromkeys(LOSS_FIELDS, value)
            if ledger['branch'] == 'lambda_zero' and any(c['n0'] and c['n1'] for c in ledger['cells'].values()):
                values['same_class_loss'] = None
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
                  for k in LOSS_FIELDS}
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
        point_id = digest(dict(context=context, row_order_hash=row_hash, pre=pre))
        diagnostics = dict(schema_version=1, point_id=point_id, context=copy.deepcopy(context),
            job_id=context.get('job_id', context.get('key')), physical_attempt_id=context.get('physical_attempt_id'),
            host_arm=config['arm'], counterfactual_at_host_state=True,
            row_order_hash=row_hash, pre_model_hash=pre['model_hash'], pre_optimizer_hash=pre['optimizer_hash'],
            pre_rng_hash=pre['rng_hash'], pre_loader_hash=pre['loader_hash'],
            objective_losses=dict(C=values['clean_loss'], S=values['scaled_loss'], U=values['cell_mass_loss'], W=values['completion_loss']),
            bce_mass_scale=ledger['bce_mass_scale'], removed_opposite_mass=ledger['removed_opposite_mass'],
            actual_objective=host, actual_post=actual, virtual_post=virtual_hashes,
            actual_virtual_exact=True, actual_virtual_update_difference_norm=diff,
            extra_autograd_sweeps=3, virtual_adam_steps=4, diagnostic_seconds=diagnostic_seconds,
            parameter_groups=_group_metrics(named, gradients, updates, before_steps, post_steps, config['rho']))
    synchronize(x.device)
    return dict(**values, same_class_loss_null_reason='not_computed_delegate' if values['same_class_loss'] is None else None,
        branch=ledger['branch'], ledger=ledger, row_order_hash=row_hash,
        optimizer_updates=1, ema_updates=1, ordinary_forwards=1, scalar_backwards=1,
        probe_forwards=0, view_forwards=0, algorithm_rng_draws=0,
        extra_autograd_sweeps=3 if diagnose else 0, virtual_adam_steps=4 if diagnose else 0,
        seconds=time.perf_counter()-started, diagnostic_seconds=diagnostic_seconds,
        diagnostics=diagnostics, parameter_activity=activity, adam_step_counters=steps)


def finite_checks():
    from .checks import check_core
    return check_core()
