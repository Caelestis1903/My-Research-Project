"""Finite registered CPU algebra, source-operation and diagnostic fixtures."""
import copy
import inspect
import json
import math
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from . import core
from experiments.auxiliary_objective_attribution_20261115 import core as auxiliary
from experiments.auxiliary_objective_attribution_20261115.stage2 import core as ndp
from experiments.auxiliary_objective_attribution_20261115.protocol import (
    make_model, optimizer, rng_state, state_digest, load_protocol,
)
from experiments.auxiliary_objective_attribution_20261115.test_core import fixture, hooks
from experiments.task_view_gradient_residual_momentum_20261116.core import source_clean_step
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.reference import MDMTRec as FixedReference


_RECORD_KEYS = {'loss', 'clean_loss', 'completed_loss', 'branch', 'ledger', 'row_order_hash',
                'optimizer_updates', 'ema_updates', 'ordinary_forwards', 'scalar_backwards',
                'probe_forwards', 'view_forwards', 'algorithm_rng_draws', 'seconds', 'diagnostics'}


def _config(arm, mixture_lambda=.25, dataset='movielens'):
    source = load_protocol()['datasets'][dataset]
    return dict(arm=arm, mixture_lambda=mixture_lambda, rho=source['rho'],
                lr=.01, weight_decay=source['weight_decay'], betas=[.9, .999], eps=1e-8)


def _close(actual, expected, dtype):
    torch.testing.assert_close(actual, expected,
                               atol=1e-10 if dtype == torch.float64 else 1e-6,
                               rtol=1e-10 if dtype == torch.float64 else 1e-5)


def _literal(probabilities, targets, domains, arm):
    """Enumerate uniform ordered pairs, including self-pairs, independently."""
    total = probabilities.new_zeros(())
    for d in range(3):
        rows = [i for i in range(len(targets)) if int(domains[i]) == d]
        if not rows:
            continue
        for t in range(2):
            pair_sum = probabilities.new_zeros(())
            for i in rows:
                for j in rows:
                    if targets[i, t] == targets[j, t]:
                        value = .5*(F.binary_cross_entropy(probabilities[i, t], targets[i, t])
                                    + F.binary_cross_entropy(probabilities[j, t], targets[j, t]))
                    else:
                        positive, negative = (i, j) if targets[i, t] == 1 else (j, i)
                        pp, pn = probabilities[positive, t], probabilities[negative, t]
                        value = ((1-pp).square()+pn.square() if arm == 'BRIER-OPPORTUNITY-025'
                                 else .5*(1-pp+pn).square())
                    pair_sum = pair_sum+value
            cell = pair_sum/(len(rows)**2)
            if arm == 'MEAN-GAP-025':
                classes = [probabilities[[i for i in rows if targets[i, t] == a], t] for a in (0, 1)]
                if all(len(scores) for scores in classes):
                    variance = sum(((scores-scores.mean()).square().sum()/len(scores)) for scores in classes)
                    cell = cell-(len(classes[0])*len(classes[1])/len(rows)**2)*variance
            total = total+len(rows)/(2*len(targets))*cell
    return total


def _mass(ledger):
    B, lam = ledger['B'], ledger['mixture_lambda']
    assert ledger['entries'] == 2*B and ledger['schema_version'] == 1
    assert list(ledger['cells']) == [f'd{d}t{t}' for d in range(3) for t in range(2)]
    assert ledger['total_mass'] == 1. and ledger['task_mass'] == {'0': .5, '1': .5}
    assert math.isclose(ledger['bce_mass']+ledger['opposite_mass'], 1., abs_tol=1e-15)
    for t in range(2):
        assert math.isclose(sum(c['weight'] for k, c in ledger['cells'].items() if k[3] == str(t)), .5, abs_tol=1e-15)
    for cell in ledger['cells'].values():
        assert cell['n0']+cell['n1'] == cell['n']
        assert cell['weight'] == cell['n']/(2*B)
        assert math.isclose(cell['bce_mass']+cell['opposite_mass'], cell['weight'], abs_tol=1e-15)
        for a in (0, 1):
            expected = (1-lam+lam*cell[f'n{a}']/cell['n'])/(2*B) if cell[f'n{a}'] else None
            assert cell[f'bce_coefficient{a}'] == expected
        coefficient_mass = sum(cell[f'n{a}']*(cell[f'bce_coefficient{a}'] or 0.) for a in (0, 1))
        assert math.isclose(coefficient_mass, cell['bce_mass'], abs_tol=1e-15)
        if not cell['n']:
            assert cell['pi0'] is cell['pi1'] is None
            assert all(cell[k] is None for k in core._STATISTICS)


def check_literal_values_gradients_and_contrasts():
    cases = [
        ([0]*4+[1]*4, [[0, 0], [1, 0], [0, 0], [1, 1], [0, 1], [0, 0], [0, 1], [1, 0]]),
        ([0, 0, 0, 1, 1, 2], [[0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1]]),
        ([0, 0, 1, 2], [[0, 1], [0, 1], [1, 0], [0, 0]]),
        ([2], [[1, 0]]),
    ]
    for dtype in (torch.float64, torch.float32):
        for domains, labels in cases:
            y = torch.tensor(labels, dtype=dtype)
            d = torch.tensor(domains)
            p = (.08+.84*torch.arange(1, y.numel()+1, dtype=dtype)/(y.numel()+1)).reshape_as(y).requires_grad_()
            outputs = {}
            for arm in core.ARMS:
                result = core.loss_components(p, y, d, arm=arm)
                literal = _literal(p, y, d, arm)
                clean = F.binary_cross_entropy(p, y)
                expected = .75*clean+.25*literal
                _close(result['completed_loss'], literal, dtype)
                _close(result['loss'], expected, dtype)
                for actual, reference in ((result['completed_loss'], literal), (result['loss'], expected)):
                    _close(torch.autograd.grad(actual, p, retain_graph=True)[0],
                           torch.autograd.grad(reference, p, retain_graph=True)[0], dtype)
                _mass(result['ledger'])
                json.dumps(result['ledger'], allow_nan=False)
                for key, cell in result['ledger']['cells'].items():
                    if not cell['n']:
                        continue
                    if not (cell['n0'] and cell['n1']):
                        assert all(cell[k] is None for k in ('Q_R', 'Q_B', 'Q_M'))
                    if core._branch(result['ledger']) == 'ordinary':
                        for a in (0, 1):
                            if cell[f'n{a}'] == 1:
                                assert cell[f'var{a}'] == 0.
                            elif cell[f'n{a}'] == 0:
                                assert all(cell[f'{k}{a}'] is None for k in ('mu', 'var', 'bbar'))
                outputs[arm] = result
            variance, brier_contrast = p.new_zeros(()), p.new_zeros(())
            for domain in range(3):
                for task in range(2):
                    a = [p[(d == domain) & (y[:, task] == label), task] for label in (0, 1)]
                    if not all(len(s) for s in a):
                        continue
                    v = sum((s-s.mean()).square().mean() for s in a)
                    factor = len(a[0])*len(a[1])/(len(y)*(len(a[0])+len(a[1])))
                    variance = variance+factor*.5*v
                    brier_contrast = brier_contrast+factor*.5*((1-a[1].mean()-a[0].mean()).square()+v)
            for left, right, expected in (('NQR-025', 'MEAN-GAP-025', variance),
                                           ('BRIER-OPPORTUNITY-025', 'NQR-025', brier_contrast)):
                difference = outputs[left]['completed_loss']-outputs[right]['completed_loss']
                _close(difference, expected, dtype)
                if expected.requires_grad:
                    _close(torch.autograd.grad(difference, p, retain_graph=True)[0],
                           torch.autograd.grad(expected, p, retain_graph=True)[0], dtype)
            pp, pn = p[0, 0], p[-1, 1]
            pair_difference = (1-pp).square()+pn.square()-.5*(1-pp+pn).square()
            identity = .5*(1-pp-pn).square()
            _close(pair_difference, identity, dtype)
            _close(torch.autograd.grad(pair_difference, p, retain_graph=True)[0],
                   torch.autograd.grad(identity, p)[0], dtype)
    return dict(status='PASS', dtypes=['float64', 'float32'], fixtures=len(cases), arms=3)


def check_half_probability_and_tail_denominators():
    for dtype in (torch.float64, torch.float32):
        for B in (1, 3, 7, 13, 21927, 40052):
            rows = torch.arange(B)
            domains = rows.remainder(3)
            y = torch.stack((rows.remainder(7) == 0, rows.remainder(5) < 2), dim=1).to(dtype)
            p = torch.full_like(y, .5, requires_grad=True)
            expected = (1-2*y)/B
            for arm in core.ARMS:
                result = core.loss_components(p, y, domains, arm=arm)
                _close(torch.autograd.grad(result['loss'], p)[0], expected, dtype)
                _mass(result['ledger'])
                if B > 3:
                    assert result['ledger']['opposite_mass'] > 0
            zero = core.loss_components(p, y, domains, arm=core.ARMS[0], mixture_lambda=0.)
            assert zero['loss'] is zero['clean_loss'] is zero['completed_loss']
            assert zero['ledger']['bce_mass'] == 1. and zero['ledger']['opposite_mass'] == 0.
    # Balanced classes have the registered .875/.125 split.
    result = core.loss_components(torch.full((2, 2), .5), torch.tensor([[0., 1.], [1., 0.]]),
                                  torch.tensor([0, 0]), arm=core.ARMS[0])
    assert result['ledger']['bce_mass'] == .875 and result['ledger']['opposite_mass'] == .125
    return dict(status='PASS', batch_sizes=[1, 3, 7, 13, 21927, 40052], arms=3)


def _same(model, opt, ema, reference, ref_opt, ref_ema):
    for a, b in ((model.state_dict(), reference.state_dict()),
                 (opt.state_dict(), ref_opt.state_dict()), (ema.state_dict(), ref_ema.state_dict())):
        assert state_digest(a) == state_digest(b)
    for a, b in zip(model.parameters(), reference.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)


def _record(record, B, diagnose):
    assert set(record) == _RECORD_KEYS
    assert record['ledger']['B'] == B
    assert record['optimizer_updates'] == record['ema_updates'] == record['ordinary_forwards'] == record['scalar_backwards'] == 1
    assert record['probe_forwards'] == record['view_forwards'] == record['algorithm_rng_draws'] == 0
    assert (record['diagnostics'] is not None) == diagnose
    assert record['seconds'] >= 0.
    _mass(record['ledger'])
    json.dumps(record, allow_nan=False)


def check_source_delegation_and_diagnostic_identity():
    initial, x, y, context = fixture()
    context['batch'] = 1
    loader = torch.Generator(device='cpu').manual_seed(13022)
    loader_before = loader.get_state().clone()
    rows = np.arange(100, 112, dtype=np.int32)[::-1]
    global_before = state_digest(rng_state('cpu'))
    cases = [(0., y, 'lambda_zero'), (.25, torch.zeros_like(y), 'all_single_class'), (.25, y, 'ordinary')]
    for arm in core.ARMS:
        for lam, labels, branch in cases:
            c = _config(arm, lam)
            model, reference = copy.deepcopy(initial), copy.deepcopy(initial)
            ema, ref_ema = copy.deepcopy(initial), copy.deepcopy(initial)
            opt, ref_opt = optimizer(model, c), optimizer(reference, c)
            for size in (len(y), 3):
                ids = rows[:size]
                with patch.object(core, 'source_clean_step', wraps=source_clean_step) as delegate:
                    record = core.train_step(model, opt, ema, x[:, :size], labels[:size], c, context,
                                             row_ids=ids, diagnose=True)
                assert delegate.call_count == (branch != 'ordinary')
                assert record['branch'] == branch
                assert record['row_order_hash'] == state_digest(np.asarray(ids, dtype=np.int64))
                _record(record, size, True)
                if branch != 'ordinary':
                    loss = source_clean_step(reference, ref_opt, ref_ema, x[:, :size], labels[:size], c)
                    assert record['loss'] == record['clean_loss'] == record['completed_loss'] == loss
                    for cell in record['ledger']['cells'].values():
                        assert all(cell[k] is None for k in core._STATISTICS)
                else:
                    # An independently enumerated scalar reaches the same real source model/Adam/EMA.
                    ref_opt.zero_grad(set_to_none=True)
                    p = reference(x[:, :size])
                    loss = .75*F.binary_cross_entropy(p, labels[:size])+.25*_literal(p, labels[:size], x[-1, :size], arm)
                    loss.backward()
                    for p_actual, p_reference in zip(model.parameters(), reference.parameters()):
                        assert (p_actual.grad is None) == (p_reference.grad is None)
                        if p_actual.grad is not None:
                            _close(p_actual.grad, p_reference.grad, torch.float32)
                    # Literal summation rounds differently; diagnostic identity below is bitwise.
                    ref_opt.step()
                    core.update_ema(ref_ema, reference, c['rho'])
                    for a, b in zip(model.parameters(), reference.parameters()):
                        _close(a, b, torch.float32)
                if branch != 'ordinary':
                    _same(model, opt, ema, reference, ref_opt, ref_ema)
            off_model, on_model = copy.deepcopy(initial), copy.deepcopy(initial)
            off_ema, on_ema = copy.deepcopy(initial), copy.deepcopy(initial)
            off_opt, on_opt = optimizer(off_model, c), optimizer(on_model, c)
            for size in (len(y), 3):
                off = core.train_step(off_model, off_opt, off_ema, x[:, :size], labels[:size], c, context)
                old = {name: p.detach().clone() for name, p in on_model.named_parameters()}
                on = core.train_step(on_model, on_opt, on_ema, x[:, :size], labels[:size], c, context, diagnose=True)
                _record(off, size, False)
                _record(on, size, True)
                _same(off_model, off_opt, off_ema, on_model, on_opt, on_ema)
                assert off['ledger'] == on['ledger']
                assert on['diagnostics']['parameter_step_max'] == (1 if size == len(y) else 2)
                groups = on['diagnostics']['parameter_groups']
                assert list(groups) == sorted(groups)
                for prefix, group in groups.items():
                    names = [name for name, _ in on_model.named_parameters() if name.split('.')[0] == prefix]
                    assert group['parameter_names'] == names
                    assert group['active_parameters']+group['none_parameters'] == len(names)
                    parameters = [(name, p) for name, p in on_model.named_parameters() if name in names]
                    expected_gradient = math.sqrt(sum(float(p.grad.double().square().sum())
                                                      for _, p in parameters if p.grad is not None))
                    expected_update = math.sqrt(sum(float((p.detach()-old[name]).double().square().sum())
                                                    for name, p in parameters))
                    assert math.isclose(group['data_gradient_norm'], expected_gradient, rel_tol=1e-12, abs_tol=1e-12)
                    assert math.isclose(group['parameter_update_norm'], expected_update, rel_tol=1e-12, abs_tol=1e-12)
                    assert group['coordinates'] == sum(p.numel() for _, p in parameters)
                for key in ('data_gradient_norm', 'parameter_update_norm'):
                    expected = math.sqrt(sum(g[key]**2 for g in groups.values()))
                    assert math.isclose(on['diagnostics'][key], expected, rel_tol=1e-12, abs_tol=1e-12)
    assert global_before == state_digest(rng_state('cpu'))
    assert torch.equal(loader_before, loader.get_state())
    assert hooks(initial) == 0
    return dict(status='PASS', arms=3, branches=3, steps_per_case=2,
                model_adam_ema_activity_rng_loader_identity='bitwise')


def check_source_graph_and_operation_counts():
    # The production factory receives synthetic features; no dataset is loaded.
    features = SimpleNamespace(columns=['feature', 'domain_indicator'], dense_columns=[],
                               vocab_sizes={'feature': 20, 'domain_indicator': 3})
    bundle = SimpleNamespace(features=features)
    initial = make_model(bundle, dict(perturbable_sparse_fields=['feature']))
    _, x, y, context = fixture()
    context['batch'] = 1
    assert 'self.gate[i](emb.detach())' in inspect.getsource(FixedReference.forward)
    for arm in core.ARMS:
        model, ema = copy.deepcopy(initial), copy.deepcopy(initial)
        c = _config(arm)
        opt = optimizer(model, c)
        before_rng = state_digest(rng_state('cpu'))
        with ExitStack() as stack:
            forward = stack.enter_context(patch.object(model, 'forward', wraps=model.forward))
            step = stack.enter_context(patch.object(opt, 'step', wraps=opt.step))
            update = stack.enter_context(patch.object(core, 'update_ema', wraps=core.update_ema))
            backward = stack.enter_context(patch.object(torch.autograd, 'backward', wraps=torch.autograd.backward))
            stack.enter_context(patch.object(torch.autograd, 'grad', side_effect=AssertionError('parameter probe')))
            stack.enter_context(patch.object(auxiliary, 'forward', side_effect=AssertionError('auxiliary/probe forward')))
            stack.enter_context(patch.object(ndp, 'make_views', side_effect=AssertionError('view construction')))
            gates = [stack.enter_context(patch.object(gate, 'forward', wraps=gate.forward)) for gate in model.gate]
            record = core.train_step(model, opt, ema, x, y, c, context, diagnose=True)
            assert forward.call_count == step.call_count == update.call_count == backward.call_count == 1
            assert forward.call_args.args == (x,) and not forward.call_args.kwargs
            assert all(gate.call_count == 1 and not gate.call_args.args[0].requires_grad for gate in gates)
        assert all(p.grad is not None for p in model.gate.parameters())
        assert any(p.grad is not None and bool(p.grad.any()) for p in model.embedding.parameters())
        assert before_rng == state_digest(rng_state('cpu')) and hooks(model) == 0
        _record(record, len(y), True)
    return dict(status='PASS', production_factory=True, gate_inputs_detached=True,
                gate_parameters_active=True, ordinary_forwards=1, scalar_backwards=1, probes=0)


class _ActivityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.score = torch.nn.Parameter(torch.tensor([.2, -.3]))
        self.zero = torch.nn.Parameter(torch.tensor([.7]))
        self.unused = torch.nn.Parameter(torch.tensor([.9]))
        self.register_buffer('buffer', torch.tensor([2.]))

    def forward(self, x):
        return (self.score+self.zero*0).sigmoid().expand(x.shape[1], 2)


def check_none_zero_activity_and_adam_ema():
    for arm in core.ARMS:
        for lam, single in ((0., False), (.25, True), (.25, False)):
            model = _ActivityModel()
            ema = copy.deepcopy(model)
            c = _config(arm, lam)
            opt = optimizer(model, c)
            x = torch.tensor([[1., 2.], [0., 0.]])
            y = torch.zeros((2, 2)) if single else torch.tensor([[0., 1.], [1., 0.]])
            old = copy.deepcopy(model.state_dict())
            record = core.train_step(model, opt, ema, x, y, c, dict(), diagnose=True)
            assert model.unused.grad is None and model.unused not in opt.state
            assert torch.equal(model.unused, old['unused'])
            assert model.zero.grad is not None and not bool(model.zero.grad.any())
            assert not torch.equal(model.zero, old['zero'])
            state = opt.state[model.zero]
            h = c['weight_decay']*old['zero']
            _close(state['exp_avg'], .1*h, torch.float32)
            _close(state['exp_avg_sq'], .001*h.square(), torch.float32)
            assert int(state['step']) == 1
            for name, p in model.named_parameters():
                expected = old[name].mul(c['rho']).add(p.detach(), alpha=1-c['rho'])
                assert torch.equal(ema.state_dict()[name], expected)
            assert torch.equal(ema.buffer, model.buffer)
            diag = record['diagnostics']
            assert diag['active_parameters'] == 2 and diag['none_parameters'] == 1
            assert diag['parameter_step_min'] == diag['parameter_step_max'] == 1
            unused = diag['parameter_groups']['unused']
            assert unused['parameter_step_min'] is unused['parameter_step_max'] is None
            assert unused['data_gradient_norm'] == unused['parameter_update_norm'] == 0.
            _record(record, 2, True)
    return dict(status='PASS', none_skips_decay_moments_counter=True, zero_active=True,
                source_adam_coupled_decay_once=True, ema_once_and_buffers_copied=True)


def run_core_checks():
    return {check.__name__: check() for check in (
        check_literal_values_gradients_and_contrasts,
        check_half_probability_and_tail_denominators,
        check_source_delegation_and_diagnostic_identity,
        check_source_graph_and_operation_counts,
        check_none_zero_activity_and_adam_ema,
    )}


if __name__ == '__main__':
    print(json.dumps(dict(status='PASS', core=run_core_checks()), indent=2, allow_nan=False))
