"""Four direct CPU fixture groups for the approved numerical contract."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from . import core
from experiments.auxiliary_objective_attribution_20261115.test_core import fixture, hooks
from experiments.auxiliary_objective_attribution_20261115.protocol import (
    optimizer, rng_state, state_digest,
)
from experiments.auxiliary_objective_attribution_20261115.stage2 import core as ndp
from experiments.task_view_gradient_residual_momentum_20261116.core import source_clean_step


def config(arm='AXIS-EVEN'):
    return dict(arm=arm, alpha=.03, mixture_lambda=.25, rho=.9828205985452511,
        lr=.01, weight_decay=1e-4)


def test_support_target_zero_symmetry_and_exact_iid_expectation():
    model, x, y, context = fixture(); context.update(batch=1, diagnose=True)
    before = state_digest(rng_state('cpu'))
    deltas, _, _, _, _ = core.make_views(model, x, y, .03)
    assert before == state_digest(rng_state('cpu')) and hooks(model) == 0
    assert all(p.grad is None for p in model.parameters())
    assert all(not delta.requires_grad for delta in deltas)
    deltas[0][0].zero_(); deltas[1].zero_()
    replicas, _ = core.make_replicas(deltas, 'AXIS-EVEN', context)
    seen = []; real_forward = core.forward
    def record_forward(model, x, delta):
        seen.append(delta)
        return real_forward(model, x, delta)
    with patch.object(core, 'forward', record_forward):
        total, clean, view, (r1, r2), task_losses = core.losses(model, x, y, replicas)
    assert len(seen) == 4
    assert all(actual is expected for actual, expected in zip(seen, [d for pair in replicas for d in pair]))
    outputs = [[real_forward(model, x, d)[0] for d in pair] for pair in replicas]
    explicit = [[F.binary_cross_entropy(outputs[t][j][:, t], y[:, t], reduction='sum')/len(y)
        for j in range(2)] for t in range(2)]
    expected_view = .5*(.5*(explicit[0][0]+explicit[1][0])+.5*(explicit[0][1]+explicit[1][1]))
    expected = .75*F.binary_cross_entropy(model(x), y)+.25*expected_view
    torch.testing.assert_close(total, expected)
    for t in range(2):
        for j in range(2):
            torch.testing.assert_close(task_losses[t][j], explicit[t][j])
    parameters = tuple(model.parameters())
    actual_grad = torch.autograd.grad(total, parameters, retain_graph=True, allow_unused=True)
    expected_grad = torch.autograd.grad(expected, parameters, allow_unused=True)
    for a, e in zip(actual_grad, expected_grad):
        assert (a is None) == (e is None)
        if a is not None:
            torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-6)
    assert core.norm(torch.autograd.grad(view, tuple(model.embedding.parameters()), retain_graph=True)) > 0
    differentiable = [[d.clone().requires_grad_() for d in pair] for pair in replicas]
    detached_total = core.losses(model, x, y, differentiable)[0]
    assert all(g is None for g in torch.autograd.grad(detached_total,
        [d for pair in differentiable for d in pair], allow_unused=True))
    reversed_total = core.losses(model, x, y, [pair[::-1] for pair in replicas])[0]
    assert torch.equal(total, reversed_total)
    # Enumerate all four independent sign pairs for one nonzero row-task, in FP64.
    plus = F.binary_cross_entropy(outputs[0][0][1, 0], y[1, 0]).double()
    minus = F.binary_cross_entropy(outputs[0][1][1, 0], y[1, 0]).double()
    combinations = torch.stack([.5*(a+b) for a in (plus, minus) for b in (plus, minus)])
    assert torch.equal(combinations.mean(), .5*(plus+minus))
    for batch in (1, 3, len(y)):
        for canonical in (False, True):
            ledger = core.support_ledger(x[:, :batch], y[:batch], canonical)
            assert ledger['denominator'] == ledger['entries'] == 2*batch
            assert ledger['replica_entries'] == [2*batch]*2
            assert ledger['clean_entry_coefficient'] == .75/(2*batch)
            assert ledger['replica_entry_coefficients'] == [.125/(2*batch)]*2
            assert ledger['actual_clean_entry_coefficient'] == (1. if canonical else .75)/(2*batch)
            assert ledger['actual_replica_entry_coefficients'] == ([0., 0.] if canonical else [.125/(2*batch)]*2)
            assert sum(c['count'] for c in ledger['cells'].values()) == 2*batch
            for key, cell in ledger['cells'].items():
                d, t, label = int(key[1]), int(key[3]), int(key[5])
                n = int(((x[-1, :batch] == d)&(y[:batch, t] == label)).sum())
                assert cell['count'] == cell['clean_count'] == n and cell['replica_counts'] == [n, n]
                assert cell['total_weight_mass'] == n/(2*batch)
                assert cell['clean_weight_mass'] == .75*n/(2*batch)
                assert cell['replica_weight_masses'] == [.125*n/(2*batch)]*2
            for task in range(2):
                assert abs(sum(c['total_weight_mass'] for k, c in ledger['cells'].items()
                    if k[3] == str(task))-.5) < 1e-12
    assert torch.equal(total, .75*clean+.25*(.5*(r1+r2)))
    assert hooks(model) == 0


def test_private_rng_exact_keys_norms_and_zero_gaussian_fallback():
    context = dict(dataset='movielens', seed=2022, epoch=1, batch=1, diagnose=True)
    delta = torch.arange(256*8, dtype=torch.float32).reshape(256, 8)/10000
    delta[0].zero_(); deltas = [delta, delta*.5]
    loader = torch.Generator().manual_seed(13022)
    loader_before = loader.get_state().clone(); global_before = state_digest(rng_state('cpu'))
    errors = []
    for arm in core.ARMS:
        replicas, diagnostic = core.make_replicas(deltas, arm, context)
        again, _ = core.make_replicas(deltas, arm, context)
        for task, d in enumerate(deltas):
            assert torch.equal(replicas[task][0], again[task][0])
            assert torch.equal(replicas[task][1], again[task][1])
            assert not torch.count_nonzero(replicas[task][0][0])
            errors.append(diagnostic[f'task{task}']['input_norm_relative_error'])
            if arm == 'AXIS-IID2':
                signs = []
                for j, stream in enumerate(('sign1', 'sign2')):
                    key = f'axis_even_v1|movielens|2022|1|1|{task}|{stream}'
                    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
                    g = torch.Generator(device='cpu').manual_seed(seed)
                    s = (2*torch.randint(0, 2, (len(d), 1), generator=g, dtype=torch.int64)-1).float()
                    signs.append(s)
                    assert torch.equal(replicas[task][j], s*d)
                assert 0 < diagnostic[f'task{task}']['sign_duplication_fraction'] < 1
                assert not torch.equal(signs[0], signs[1])
            elif arm == 'ISOTROPIC-EVEN':
                key = f'axis_even_v1|movielens|2022|1|1|{task}|isotropic'
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
                q = torch.randn(d.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
                expected = d.norm(dim=1, keepdim=True)*q/q.norm(dim=1, keepdim=True)
                assert torch.equal(replicas[task][0], expected)
            else:
                assert torch.equal(replicas[task][0], d)
            if arm != 'AXIS-IID2':
                assert torch.equal(replicas[task][0], -replicas[task][1])
        if arm != 'AXIS-EVEN':
            changed, _ = core.make_replicas(deltas, arm, dict(context, batch=2))
            assert not torch.equal(replicas[0][0], changed[0][0])
    with patch.object(core.torch, 'randn', side_effect=lambda shape, **kw: torch.zeros(shape)) as draw:
        fallback, _ = core.make_replicas(deltas, 'ISOTROPIC-EVEN', context)
        assert draw.call_count == 2
    for task, d in enumerate(deltas):
        assert torch.equal(fallback[task][0][:, 0], d.norm(dim=1))
        assert not torch.count_nonzero(fallback[task][0][:, 1:])
        assert not torch.count_nonzero(fallback[task][0][0])
    assert max(errors) <= 1e-6
    assert global_before == state_digest(rng_state('cpu'))
    assert torch.equal(loader_before, loader.get_state())
    print(json.dumps(dict(isotropic_input_norm_max_relative_error=max(errors))))


def test_source_adam_none_zero_counters_weight_decay_once_and_ema_once():
    model = torch.nn.Module()
    model.register_parameter('a', torch.nn.Parameter(torch.tensor([1., -2.])))
    model.register_parameter('b', torch.nn.Parameter(torch.tensor([.5, 3.])))
    model.register_buffer('buffer', torch.tensor([2.]))
    ema = copy.deepcopy(model); c = config(); c['weight_decay'] = .2
    opt = optimizer(model, c)
    assert not opt.defaults['foreach'] and not opt.defaults['fused'] and not opt.defaults['amsgrad']
    for step, active in enumerate(('a', 'b', 'a'), 1):
        old = {name: p.detach().clone() for name, p in model.named_parameters()}
        old_ema = copy.deepcopy(ema.state_dict())
        previous = {name: state_digest(opt.state[p]) for name, p in model.named_parameters() if p in opt.state}
        opt.zero_grad(set_to_none=True)
        (getattr(model, active).sum()*0).backward()
        assert getattr(model, active).grad is not None
        opt.step(); model.buffer.add_(1)
        core.update_ema(ema, model, c['rho'])
        for name, p in model.named_parameters():
            if name != active:
                assert p.grad is None and torch.equal(p, old[name])
                assert (name not in previous and p not in opt.state) or state_digest(opt.state[p]) == previous[name]
            elif step <= 2:
                state = opt.state[p]; h = .2*old[name]
                assert int(state['step']) == 1
                torch.testing.assert_close(state['exp_avg'], .1*h)
                torch.testing.assert_close(state['exp_avg_sq'], .001*h.square())
            expected_ema = old_ema[name].mul(c['rho']).add(p.detach(), alpha=1-c['rho'])
            assert torch.equal(ema.state_dict()[name], expected_ema)
        assert torch.equal(ema.buffer, model.buffer)
    assert int(opt.state[model.a]['step']) == 2 and int(opt.state[model.b]['step']) == 1


def assert_same_state(model, opt, ema, other, other_opt, other_ema):
    for actual, expected in ((model.state_dict(), other.state_dict()),
                             (opt.state_dict(), other_opt.state_dict()),
                             (ema.state_dict(), other_ema.state_dict())):
        assert state_digest(actual) == state_digest(expected)
    for a, b in zip(model.parameters(), other.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)


def test_canonical_source_identity_and_registered_steps():
    root = Path(__file__).resolve().parents[2]
    frozen = {
        'experiments/auxiliary_objective_attribution_20261115/core.py': '0999b534aba4f632f9fbd35a3313d5eeb4fd04405d323f881d2e39a7f4a2bfe3',
        'experiments/auxiliary_objective_attribution_20261115/stage2/core.py': 'fe662da1b8c00229a6a234bb7e00d659fab1d278470092531cdeefa6bb2f3d58',
        'experiments/task_view_gradient_residual_momentum_20261116/core.py': '5915d97983c79ccb33302146fde287419ef611caee5bda2f13898c973845c9d9',
    }
    for path, expected in frozen.items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest() == expected
    initial, x, y, context = fixture(); context.update(batch=1, diagnose=True, row_order_hash='cpu-fixture')
    for arm in core.ARMS:
        model = copy.deepcopy(initial); ema = copy.deepcopy(initial); c = config(arm)
        reference = copy.deepcopy(initial); reference_ema = copy.deepcopy(initial)
        opt = optimizer(model, c); ref_opt = optimizer(reference, c)
        real_views = core.make_views
        def zero_views(*args):
            return real_views(*args, force_zero=True)
        with patch.object(core, 'make_views', zero_views), patch.object(core, 'forward', side_effect=AssertionError('canonical view forward')):
            for batch in (len(y), 3):
                record = core.train_step(model, opt, ema, x[:, :batch], y[:batch], c, context)
                expected_loss = source_clean_step(reference, ref_opt, reference_ema, x[:, :batch], y[:batch], c)
                assert record['zero_delta_canonical'] and record['loss'] == expected_loss
                assert record['row_order_hash'] == 'cpu-fixture'
                assert record['optimizer_updates'] == record['ema_updates'] == 1
                assert_same_state(model, opt, ema, reference, ref_opt, reference_ema)
        # Exercise the actual new step against the explicit target-only objective and source Adam/EMA.
        model = copy.deepcopy(initial); ema = copy.deepcopy(initial); opt = optimizer(model, c)
        reference = copy.deepcopy(initial); reference_ema = copy.deepcopy(initial); ref_opt = optimizer(reference, c)
        global_before = state_digest(rng_state('cpu'))
        with patch.object(opt, 'step', wraps=opt.step) as step, patch.object(core, 'update_ema', wraps=core.update_ema) as update:
            record = core.train_step(model, opt, ema, x, y, c, context)
            assert step.call_count == update.call_count == 1
        assert global_before == state_digest(rng_state('cpu'))
        assert not record['zero_delta_canonical'] and hooks(model) == 0
        ref_opt.zero_grad(set_to_none=True)
        deltas, _, _, _, _ = ndp.make_views(reference, x, y, .03)
        replicas, _ = core.make_replicas(deltas, arm, context)
        clean = F.binary_cross_entropy(reference(x), y)
        a = [[F.binary_cross_entropy(core.forward(reference, x, replicas[t][j])[0][:, t], y[:, t])
            for j in range(2)] for t in range(2)]
        loss = .75*clean+.25*(.5*(.5*(a[0][0]+a[1][0])+.5*(a[0][1]+a[1][1])))
        loss.backward(); ref_opt.step(); core.update_ema(reference_ema, reference, c['rho'])
        assert record['loss'] == float(loss.detach())
        assert_same_state(model, opt, ema, reference, ref_opt, reference_ema)
        assert record['diagnostics']['gradient']['parameter_step_max'] == 1
        assert record['diagnostics']['gradient']['actual_update_l2_norm'] > 0
        assert core.norm([p.grad for p in model.embedding.parameters()]) > 0
        json.dumps(record, allow_nan=False)


if __name__ == '__main__':
    test_support_target_zero_symmetry_and_exact_iid_expectation()
    test_private_rng_exact_keys_norms_and_zero_gaussian_fallback()
    test_source_adam_none_zero_counters_weight_decay_once_and_ema_once()
    test_canonical_source_identity_and_registered_steps()
    print('PASS: four CPU fixture groups')
