"""Detached NDP-A input views; one ordinary coupled Adam and fixed EMA step."""
import hashlib
import math
import time

import torch
import torch.nn.functional as F

from experiments.auxiliary_objective_attribution_20261115.core import forward
from experiments.auxiliary_objective_attribution_20261115.stage2.core import (
    make_views, summary, synchronize,
)
from experiments.conditional_information_momentum_20261102_v3.protocol import update_ema

ARMS = ('AXIS-EVEN', 'AXIS-IID2', 'ISOTROPIC-EVEN')
RNG_KEY_VERSION = 'axis_even_v1'


def private_generator(device, context, task, stream):
    assert context['epoch'] >= 1 and context['batch'] >= 1
    key = '|'.join(str(v) for v in (
        RNG_KEY_VERSION, context['dataset'], context['seed'], context['epoch'],
        context['batch'], task, stream,
    ))
    seed = int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8], 'little')
    return torch.Generator(device=device).manual_seed(seed)


@torch.no_grad()
def make_replicas(deltas, arm, context):
    """Return task-major pairs and compact construction diagnostics."""
    assert arm in ARMS and len(deltas) == 2
    replicas = []; diagnostics = {}
    for task, source_delta in enumerate(deltas):
        delta = source_delta.detach()
        info = {}
        if arm == 'AXIS-EVEN':
            pair = (delta, -delta)
        elif arm == 'AXIS-IID2':
            signs = [torch.randint(
                0, 2, (len(delta), 1), dtype=torch.int64, device=delta.device,
                generator=private_generator(delta.device, context, task, stream),
            ).to(torch.float32).mul_(2).sub_(1) for stream in ('sign1', 'sign2')]
            pair = (signs[0]*delta, signs[1]*delta)
            if context.get('diagnose', False):
                info['sign_duplication_fraction'] = float((signs[0] == signs[1]).float().mean())
        else:
            q = torch.randn(delta.shape, dtype=torch.float32, device=delta.device,
                generator=private_generator(delta.device, context, task, 'isotropic'))
            qnorm = q.norm(dim=1, keepdim=True)
            # A zero Gaussian draw uses the registered coordinate fallback, without resampling.
            q[qnorm[:, 0] == 0, 0] = 1.
            u = delta.norm(dim=1, keepdim=True)*q/torch.where(qnorm > 0, qnorm, 1.)
            pair = (u, -u)
        if context.get('diagnose', False):
            radius = delta.norm(dim=1)
            actual = pair[0].norm(dim=1)
            error = (actual-radius).abs()/torch.where(radius > 0, radius, 1.)
            info.update(input_radius=summary(radius),
                input_norm_relative_error=float(error.max()))
            assert info['input_norm_relative_error'] <= 1e-6
            assert not torch.count_nonzero(pair[0][radius == 0])
        replicas.append(pair); diagnostics[f'task{task}'] = info
    return replicas, diagnostics


def losses(model, x, y, replicas, canonical=False):
    """Exact registered order and parenthesization; only each view's target contributes."""
    clean = F.binary_cross_entropy(model(x), y)
    if canonical:
        return clean, clean, clean, (clean, clean), None
    task_losses = []
    for task in range(2):
        pair = []
        for replica in range(2):
            probability = forward(model, x, replicas[task][replica])[0]
            pair.append(F.binary_cross_entropy(probability[:, task], y[:, task]))
        task_losses.append(pair)
    r1 = .5*(task_losses[0][0]+task_losses[1][0])
    r2 = .5*(task_losses[0][1]+task_losses[1][1])
    view = .5*(r1+r2)
    total = .75*clean+.25*view
    return total, clean, view, (r1, r2), task_losses


def support_ledger(x, y, canonical=False):
    """Conceptual natural mass, plus the canonical branch's actual coefficients."""
    batch = len(y); denominator = 2*batch
    assert batch > 0 and y.shape == (batch, 2)
    cells = {}
    for domain in range(3):
        for task in range(2):
            for label in (0, 1):
                count = int(((x[-1] == domain)&(y[:, task] == label)).sum())
                natural = count/denominator
                cells[f'd{domain}t{task}y{label}'] = dict(
                    count=count, clean_count=count, replica_counts=[count, count],
                    natural_weight_mass=natural, clean_weight_mass=.75*natural,
                    replica_weight_masses=[.125*natural, .125*natural],
                    total_weight_mass=natural,
                )
    assert sum(cell['count'] for cell in cells.values()) == denominator
    return dict(B=batch, entries=denominator, denominator=denominator,
        clean_entries=denominator, replica_entries=[denominator, denominator],
        clean_entry_coefficient=.75/denominator,
        replica_entry_coefficients=[.125/denominator, .125/denominator],
        total_entry_coefficient=1/denominator, clean_weight_mass=.75,
        replica_weight_masses=[.125, .125], total_weight_mass=1., task_weight_masses=[.5, .5],
        actual_clean_entry_coefficient=(1. if canonical else .75)/denominator,
        actual_replica_entry_coefficients=([0., 0.] if canonical else [.125/denominator]*2),
        zero_delta_canonical=canonical, cells=cells)


def norm(tensors):
    values = [v.detach().double().square().sum() for v in tensors if v is not None]
    return math.sqrt(float(torch.stack(values).sum())) if values else 0.


def train_step(model, optimizer, ema, x, y, config, context):
    """Return JSON-safe batch facts; context owns row hash and diagnostic scheduling."""
    assert config['arm'] in ARMS
    assert config['alpha'] == .03 and config['mixture_lambda'] == .25
    diagnose = context.get('diagnose', False)
    synchronize(x.device); started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    deltas, _, probe_info, probe_seconds, construction_seconds = make_views(model, x, y, .03)
    canonical = all(not torch.count_nonzero(delta) for delta in deltas)
    synchronize(x.device); construction_started = time.perf_counter()
    replicas, replica_info = (None, {}) if canonical else make_replicas(deltas, config['arm'], context)
    synchronize(x.device); construction_seconds += time.perf_counter()-construction_started
    training_started = time.perf_counter()
    total, clean, view, (r1, r2), task_losses = losses(model, x, y, replicas, canonical)
    assert torch.isfinite(total), 'Nonfinite training loss'
    ledger = support_ledger(x, y, canonical)
    diagnostics = dict(probe=probe_info, replicas=replica_info,
        rng_key_version=RNG_KEY_VERSION, view_rng_draws_skipped=canonical,
        rng_streams=([] if canonical or config['arm'] == 'AXIS-EVEN' else
            ['sign1', 'sign2'] if config['arm'] == 'AXIS-IID2' else ['isotropic']))
    if diagnose and config['arm'] != 'AXIS-IID2':
        diagnostics.update(E=float((view-clean).detach()), O=float((.5*(r1-r2)).detach()))
    total.backward()
    named = list(model.named_parameters())
    gradients = [p.grad for _, p in named if p.grad is not None]
    if gradients:
        assert torch.stack([torch.isfinite(g).all() for g in gradients]).all(), 'Nonfinite data gradient'
    if diagnose:
        before = [p.detach().clone() for _, p in named]
        diagnostics['gradient'] = dict(total_data_gradient_norm=norm(gradients),
            none_parameters=len(named)-len(gradients))
    optimizer.step()
    update_ema(ema, model, config['rho'])
    if diagnose:
        counters = [int(optimizer.state[p]['step'].item()) for _, p in named if p.grad is not None]
        diagnostics['gradient'].update(
            parameter_step_min=min(counters) if counters else None,
            parameter_step_max=max(counters) if counters else None,
            actual_update_l2_norm=norm([p.detach()-old for (_, p), old in zip(named, before)]))
    synchronize(x.device); training_seconds = time.perf_counter()-training_started
    return dict(loss=float(total.detach()), clean_loss=float(clean.detach()), view_loss=float(view.detach()),
        replica_losses=dict(R1=float(r1.detach()), R2=float(r2.detach())),
        task_view_losses={f'task{task}': ([None, None] if task_losses is None else
            [float(loss.detach()) for loss in task_losses[task]]) for task in range(2)},
        zero_delta_canonical=canonical, ledger=ledger, diagnostics=diagnostics,
        row_order_hash=context.get('row_order_hash'), optimizer_updates=1, ema_updates=1,
        probe_seconds=probe_seconds, construction_seconds=construction_seconds,
        training_seconds=training_seconds, seconds=time.perf_counter()-started)
