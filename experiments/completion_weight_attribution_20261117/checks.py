"""Finite CPU acceptance fixtures for the approved Cycle04 core."""
import argparse
import copy
import importlib
import io
import json
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from . import core
from experiments.auxiliary_objective_attribution_20261115.protocol import (
    cpu_copy, make_model, optimizer, restore_rng, rng_state, setup, state_digest)


def _equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return list(a) == list(b) and all(_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _exact(a, b):
    assert _equal(a, b)
    assert state_digest(a) == state_digest(b)


def _close(a, b, dtype=torch.float64):
    a, b = torch.as_tensor(a, dtype=dtype), torch.as_tensor(b, dtype=dtype)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    atol, rtol = (1e-12, 1e-10) if dtype == torch.float64 else (2e-6, 2e-5)
    assert bool((torch.abs(a-b) <= atol+rtol*torch.abs(b)).all()), (a, b)


def _config(arm, dataset='movielens', mixture_lambda=.25):
    K, tail, wd, rho = core._DATA[dataset]
    return dict(cycle_id='cycle_04', dataset=dataset, arm=arm, objective='completion_weight_attribution_v1',
        mixture_lambda=mixture_lambda, epochs=50, batch_size=40960, precision='float32', lr=.01,
        betas=[.9, .999], eps=1e-8, weight_decay=wd, rho=rho, steps_per_epoch=K, tail_rows=tail,
        loader_seed_offset=11000, gate_training_detached=True, probe_gate_full_derivative=False,
        algorithm_rng_draws=0, data_hash='finite_fixture', selection_hash='finite_fixture')


def _context(loader, rows, step=1):
    return dict(purpose='engineering_check', clock_kind='engineering', dataset='movielens', seed=2022,
        case_id='finite_core', engineering_call_ordinal=step, engineering_state_step=step,
        sample_epoch=1, sample_batch=1, branch_role='fixture', fixture_rows=rows,
        pre_loader_hash=state_digest(loader.get_state()))


def _batch(dtype=torch.float32):
    domains = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=dtype)
    y = torch.tensor([[0, 0], [0, 1], [0, 1], [1, 1], [1, 0], [1, 0], [1, 1]], dtype=dtype)
    x = torch.stack((torch.linspace(-.7, .9, len(y), dtype=dtype), domains))
    return x, y


def _loss_checks():
    fixtures = 0
    for dtype in (torch.float64, torch.float32):
        x, y = _batch(dtype)
        p = torch.tensor([[.17, .28], [.31, .47], [.56, .71], [.82, .86], [.63, .22], [.79, .34], [.73, .67]],
                         dtype=dtype, requires_grad=True)
        for arm in core.ARMS:
            result = core.loss_components(p, y, x[-1], arm=arm, diagnose=True)
            ledger = result['ledger']
            m, r = ledger['bce_mass_scale'], ledger['removed_opposite_mass']
            assert .875 <= m <= 1 and r > 0 and ledger['total_mass'] == m
            assert ledger['bce_mass'] == m and ledger['opposite_mass'] == 0
            entries = F.binary_cross_entropy(p, y, reduction='none')
            row = p.new_zeros(())
            rank = p.new_zeros(())
            for d in range(3):
                for t in range(2):
                    c = ledger['cells'][f'd{d}t{t}']
                    weighted_count = 0.
                    for a in (0, 1):
                        if c[f'n{a}']:
                            row = row+(entries[(x[-1] == d) & (y[:, t] == a), t]*(1-.25+.25*c[f'pi{a}'])).sum()/(2*len(y))
                            weighted_count += c[f'n{a}']*c[f'bce_coefficient{a}']
                        else:
                            assert c[f'bce_coefficient{a}'] is None and c[f'mu{a}'] is None
                    _close(weighted_count, c['bce_mass'])
                    if c['n0'] and c['n1']:
                        pos = p[(x[-1] == d) & (y[:, t] == 1), t]
                        neg = p[(x[-1] == d) & (y[:, t] == 0), t]
                        q = .5*(1-pos[:, None]+neg[None, :]).square().mean()
                        _close(c['Q_R'], q.detach(), dtype)
                        rank = rank+c['removed_opposite_mass']*q
                    else:
                        assert c['Q_R'] is None
            _close(result['completion_loss'], row, dtype)
            _close(result['scaled_loss'], m*result['clean_loss'], dtype)
            _close(result['nqr_loss'], result['completion_loss']+rank, dtype)
            _close(sum(c['bce_mass'] for c in ledger['cells'].values()), m)
            _close(sum(ledger['task_mass'].values()), m)
            _close(sum(ledger['completion_task_mass'].values()), m)
            assert ledger['completion_task_mass']['0'] != ledger['completion_task_mass']['1']
            for t in range(2):
                _close(sum(ledger['cells'][f'd{d}t{t}']['bce_mass'] for d in range(3)), ledger['task_mass'][str(t)])
            # Independent row/pair formulas check probability and parameter chain derivatives.
            for left, right in ((result['completion_loss'], row),
                                (result['scaled_loss'], m*result['clean_loss']),
                                (result['nqr_loss'], result['completion_loss']+rank)):
                a, = torch.autograd.grad(left, p, retain_graph=True)
                b, = torch.autograd.grad(right, p, retain_graph=True)
                _close(a, b, dtype)
            theta = torch.tensor([.4, -.2], dtype=dtype, requires_grad=True)
            probabilities = torch.sigmoid(x[0, :, None]*theta[None, :])
            chained = core.loss_components(probabilities, y, x[-1], arm=arm, diagnose=True)
            frozen = core.nqr_components(probabilities, y, x[-1], arm='NQR-025', mixture_lambda=.25)
            a, = torch.autograd.grad(chained['nqr_loss'], theta, retain_graph=True)
            b, = torch.autograd.grad(frozen['loss'], theta, retain_graph=True)
            _close(a, b, dtype)
            c_grad, = torch.autograd.grad(chained['clean_loss'], theta, retain_graph=True)
            s_grad, = torch.autograd.grad(chained['scaled_loss'], theta, retain_graph=True)
            _close(s_grad, m*c_grad, dtype)
            independent_w = probabilities.new_zeros(())
            independent_r = probabilities.new_zeros(())
            chained_entries = F.binary_cross_entropy(probabilities, y, reduction='none')
            for d in range(3):
                for t in range(2):
                    cell = chained['ledger']['cells'][f'd{d}t{t}']
                    for label in (0, 1):
                        if cell[f'n{label}']:
                            mask = (x[-1] == d) & (y[:, t] == label)
                            independent_w = independent_w+chained_entries[mask, t].sum()*cell[f'completion_coefficient{label}']
                    if cell['n0'] and cell['n1']:
                        pos = probabilities[(x[-1] == d) & (y[:, t] == 1), t]
                        neg = probabilities[(x[-1] == d) & (y[:, t] == 0), t]
                        independent_r = independent_r+cell['removed_opposite_mass']*.5*(1-pos[:, None]+neg[None, :]).square().mean()
            for left, right in ((chained['completion_loss'], independent_w),
                                (chained['nqr_loss'], independent_w+independent_r)):
                a, = torch.autograd.grad(left, theta, retain_graph=True)
                b, = torch.autograd.grad(right, theta, retain_graph=True)
                _close(a, b, dtype)
            c = ledger['cells']['d1t0']
            assert c['n0'] == 0 and c['pi0'] == 0 and c['statistics_status']['mu0'] == 'absent_class'
            assert c['completion_coefficient1'] == 1/(2*len(y))
            assert c['scale_coefficient1'] < c['completion_coefficient1']
            empty = ledger['cells']['d2t0']
            assert empty['pi0'] is None and empty['pi1'] is None and empty['opportunity'] == 0
            assert set(empty['statistics_status'].values()) == {'empty_cell'}
            off = core.loss_components(p, y, x[-1], arm=arm)
            assert off['nqr_loss'] is None
            assert all(off['ledger']['cells']['d0t0'][k] is None for k in core.STATISTICS)
            assert off['ledger']['cells']['d1t0']['statistics_status']['mu0'] == 'absent_class'
            assert off['ledger']['cells']['d1t0']['statistics_status']['Q_R'] == 'requires_both_classes'
            assert off['ledger']['cells']['d0t0']['statistics_status']['mu0'] == 'not_scheduled'
            for lam, labels, branch in ((0., y, 'lambda_zero'), (.25, torch.ones_like(y), 'all_single_class')):
                identity = core.loss_components(p, labels, x[-1], arm=arm, mixture_lambda=lam, diagnose=True)
                assert identity['branch'] == branch
                for key in ('loss', 'same_class_loss', 'completion_loss', 'scaled_loss', 'nqr_loss'):
                    _exact(identity[key], identity['clean_loss'])
                assert identity['ledger']['total_mass'] == 1
            fixtures += 1
    balanced = core.loss_components(torch.full((2, 2), .5), torch.tensor([[0., 0.], [1., 1.]]),
                                    torch.zeros(2), arm=core.ARMS[0])
    assert balanced['ledger']['bce_mass_scale'] == .875
    # Probability endpoints keep source BCE behavior.
    core.loss_components(torch.tensor([[0., 1.]]), torch.tensor([[0., 1.]]), torch.tensor([0.]), arm=core.ARMS[0])
    return dict(dtype_arm_fixtures=fixtures, probability_and_parameter_derivatives=True,
                explicit_pair_enumeration=True, empty_absent_single_lambda_zero=True)


class _FixtureModel(torch.nn.Module):
    constructions = 0

    def __init__(self):
        super().__init__()
        type(self).constructions += 1
        self.shared = torch.nn.Linear(1, 2)
        self.gate = torch.nn.Linear(1, 2)
        self.zero = torch.nn.Parameter(torch.tensor([.7]))
        self.unused = torch.nn.Parameter(torch.tensor([.8]))
        self.register_buffer('forwards', torch.zeros((), dtype=torch.int64))

    def forward(self, x):
        self.forwards.add_(1)
        feature = x[0, :, None]
        # This finite fixture exercises the ordinary detached gate and live RNG.
        jitter = torch.rand((), device=x.device)*.001
        return torch.sigmoid(self.shared(feature)+self.gate(feature.detach())+0*self.zero+jitter)


def _snapshot(model, opt, ema, loader):
    return cpu_copy(dict(model=model.state_dict(), optimizer=opt.state_dict(), ema=ema.state_dict(),
        gradients={n: p.grad for n, p in model.named_parameters()}, modes={n: m.training for n, m in model.named_modules()},
        ema_modes={n: m.training for n, m in ema.named_modules()}, rng=rng_state('cpu'), loader=loader.get_state()))


def _restore(model, opt, ema, loader, snapshot):
    model.load_state_dict(cpu_copy(snapshot['model']))
    opt.load_state_dict(cpu_copy(snapshot['optimizer']))
    ema.load_state_dict(cpu_copy(snapshot['ema']))
    for n, p in model.named_parameters():
        p.grad = None if snapshot['gradients'][n] is None else snapshot['gradients'][n].clone()
    for n, m in model.named_modules():
        m.training = snapshot['modes'][n]
    for n, m in ema.named_modules():
        m.training = snapshot['ema_modes'][n]
    loader.set_state(snapshot['loader'].clone())
    restore_rng(cpu_copy(snapshot['rng']), 'cpu')


def _isolated(a, b):
    if torch.is_tensor(a):
        assert a.data_ptr() != b.data_ptr() and a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    elif isinstance(a, dict):
        for k in a:
            _isolated(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        for x, y in zip(a, b):
            _isolated(x, y)


def _roundtrip(snapshot):
    buffer = io.BytesIO()
    torch.save(snapshot, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location='cpu', weights_only=False)
    _exact(snapshot, restored)
    _isolated(snapshot, restored)
    return restored


def _step_checks():
    comparisons = direct_source = diagnosed = live_calls = 0
    x, y = _batch()
    for arm in core.ARMS:
        for branch in ('ordinary', 'all_single_class', 'lambda_zero'):
            c = _config(arm, mixture_lambda=0. if branch == 'lambda_zero' else .25)
            labels = torch.ones_like(y) if branch == 'all_single_class' else y
            model = _FixtureModel()
            ema = copy.deepcopy(model)
            opt = optimizer(model, c)
            loader = torch.Generator().manual_seed(13022)
            for populated in (False, True):
                if populated:
                    # Initialize the later-unused parameter, then ensure None skips its populated state.
                    opt.zero_grad(set_to_none=True)
                    for p in model.parameters():
                        p.grad = torch.ones_like(p)*.01
                    opt.step()
                    core.update_ema(ema, model, c['rho'])
                before = _snapshot(model, opt, ema, loader)
                saved_hash = state_digest(before)
                off = core.train_step(model, opt, ema, x, labels, c, _context(loader, len(y)), diagnose=False)
                live_calls += 1
                off_state = _snapshot(model, opt, ema, loader)
                _restore(model, opt, ema, loader, _roundtrip(before))
                _isolated(before['optimizer'], opt.state_dict())
                constructions = _FixtureModel.constructions
                hooks = (len(model._forward_hooks), len(opt._optimizer_step_pre_hooks))
                on = core.train_step(model, opt, ema, x, labels, c, _context(loader, len(y)), diagnose=True)
                live_calls += 1
                diagnosed += 1
                assert _FixtureModel.constructions == constructions
                assert hooks == (len(model._forward_hooks), len(opt._optimizer_step_pre_hooks))
                on_state = _snapshot(model, opt, ema, loader)
                _exact(off_state, on_state)
                assert state_digest(before) == saved_hash
                assert on['branch'] == branch and off['branch'] == branch
                assert on['extra_autograd_sweeps'] == 3 and on['virtual_adam_steps'] == 4
                assert off['extra_autograd_sweeps'] == off['virtual_adam_steps'] == 0
                assert on['diagnostics']['actual_virtual_exact']
                assert on['diagnostics']['actual_virtual_update_difference_norm'] == 0
                assert on['parameter_activity']['zero'] == 'tensor_zero'
                assert on['parameter_activity']['unused'] == 'none'
                assert on['adam_step_counters']['unused'] == (before['optimizer']['state'][1]['step'].item() if populated else None)
                assert model.forwards.item() == before['model']['forwards'].item()+1
                _exact(ema.forwards, model.forwards)
                assert not torch.equal(before['model']['zero'], model.zero)
                _exact(before['model']['unused'], model.unused)
                if populated:
                    # Parameter IDs follow model registration: zero, unused, then shared/gate.
                    _exact(before['optimizer']['state'][1], opt.state_dict()['state'][1])
                assert on['diagnostics']['parameter_groups']['ALL']['objectives']['W']['tensor_zero_parameters'] >= 1
                json.dumps(on, allow_nan=False)
                comparisons += 1
                if branch != 'ordinary':
                    _restore(model, opt, ema, loader, before)
                    core.source_clean_step(model, opt, ema, x, labels, c)
                    direct_source += 1
                    _exact(on_state, _snapshot(model, opt, ema, loader))
                # Populated prefix restore and one-step suffix are exact, including next permutation.
                prefix = _snapshot(model, opt, ema, loader)
                prefix_hash = state_digest(prefix)
                core.train_step(model, opt, ema, x, labels, c, _context(loader, len(y), 2), diagnose=True)
                diagnosed += 1
                live_calls += 1
                expected = _snapshot(model, opt, ema, loader)
                next_order = torch.randperm(17, generator=loader)
                _restore(model, opt, ema, loader, _roundtrip(prefix))
                core.train_step(model, opt, ema, x, labels, c, _context(loader, len(y), 2), diagnose=False)
                live_calls += 1
                _exact(expected, _snapshot(model, opt, ema, loader))
                _exact(next_order, torch.randperm(17, generator=loader))
                assert state_digest(prefix) == prefix_hash
    return dict(neutrality_comparisons=comparisons, diagnosed_core_calls=diagnosed, core_calls=live_calls,
                direct_source_identity_calls=direct_source, extra_autograd_sweeps=3*diagnosed,
                virtual_source_adam_steps=4*diagnosed, bytesio_prefix_suffix=True,
                none_tensor_zero_lazy_populated=True, rng_loader_gradients_buffers_exact=True)


def _validation_checks():
    x, y = _batch()
    loader = torch.Generator().manual_seed(13022)
    c = _config(core.ARMS[0])
    context = _context(loader, len(y))
    model = _FixtureModel()
    ema = copy.deepcopy(model)
    opt = optimizer(model, c)
    before = _snapshot(model, opt, ema, loader)
    bad_cases = []
    for key, value in (('arm', 'NQR-025'), ('mixture_lambda', .5), ('lr', .02), ('gate_training_detached', False), ('schedule_epoch', 10)):
        candidate = dict(c, **{key: value})
        bad_cases.append(dict(x=x, y=y, config=candidate, context=context))
    for updates in ({'seed': True}, {'seed': 2021}, {'dataset': 'kuairand'}, {'fixture_rows': 2}, {'engineering_state_step': False}):
        bad_cases.append(dict(x=x, y=y, config=c, context=dict(context, **updates)))
    bad_y = y.clone(); bad_y[0, 0] = .5
    bad_x = x.clone(); bad_x[-1, 0] = 1.5
    for xx, yy in ((x, bad_y), (bad_x, y), (x[:, :0], y[:0]), (x, y.double()), (x[:, :-1], y)):
        bad_cases.append(dict(x=xx, y=yy, config=c, context=context))
    for case in bad_cases:
        try:
            core.train_step(model, opt, ema, **case)
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError('invalid call accepted')
        _exact(before, _snapshot(model, opt, ema, loader))
    for p in (torch.full_like(y, float('nan')), torch.full_like(y, -1.), y.double(), y[:, :1]):
        try:
            core.loss_components(p, y, x[-1], arm=core.ARMS[0])
        except ValueError:
            pass
        else:
            raise AssertionError('invalid probabilities accepted')
    cardinalities = 0
    for dataset, (K, tail, _, _) in core._DATA.items():
        config = _config(core.ARMS[0], dataset)
        for b, B, diagnostic in ((1, 40960, True), (K, tail, False)):
            yy = torch.zeros(B, 2)
            xx = torch.zeros(2, B)
            ctx = dict(purpose='formal', clock_kind='formal', dataset=dataset, seed=2022,
                       epoch=1, batch=b, global_step=b, pre_loader_hash=state_digest(loader.get_state()))
            ledger, _ = core._validate_call(xx, yy, config, ctx, np.arange(B, dtype=np.int64), diagnostic)
            assert ledger['B'] == B and ledger['entries'] == 2*B
            cardinalities += 1
    a = [torch.tensor([3.]), None]
    b = [None, torch.tensor([4.])]
    vectors = core._vector_metrics(a, b)
    assert vectors['norm'] == 3 and vectors['right_norm'] == 4 and vectors['difference_norm'] == 5 and vectors['cosine'] == 0
    zero = core._vector_metrics([torch.zeros(1), None], b)
    assert zero['cosine'] is None and zero['null_reason'] == 'zero_norm'
    # Independently expressed EMA differences use only the fixed algebra tolerance.
    initial = torch.tensor([.7, -.2])
    left, right = torch.tensor([.2, .3]), torch.tensor([.6, -.5])
    rho = c['rho']
    left_ema = initial.clone().mul_(rho).add_(left, alpha=1-rho)
    right_ema = initial.clone().mul_(rho).add_(right, alpha=1-rho)
    _close(left_ema-right_ema, (1-rho)*(left-right), torch.float32)
    return dict(rejected_pre_mutation_calls=len(bad_cases), rejected_probability_inputs=4,
                real_form_full_tail_cardinalities=cardinalities, full_coordinate_disjoint_zero_norm=True)


def _source_model_checks():
    # Tiny explicit feature schema; no dataset cache or Test object is opened.
    features = SimpleNamespace(columns=['user', 'domain_indicator'], dense_columns=[],
                               vocab_sizes={'user': 8, 'domain_indicator': 3})
    bundle = SimpleNamespace(features=features)
    x, y = _batch()
    x[0] = torch.arange(len(y))
    loader = torch.Generator().manual_seed(13022)
    comparisons = 0
    for arm in core.ARMS:
        config = _config(arm)
        model = make_model(bundle, {'perturbable_sparse_fields': ['user']})
        ema = copy.deepcopy(model)
        opt = optimizer(model, config)
        before = _snapshot(model, opt, ema, loader)
        core.train_step(model, opt, ema, x, y, config, _context(loader, len(y)), diagnose=True)
        expected = _snapshot(model, opt, ema, loader)
        _restore(model, opt, ema, loader, before)
        core.train_step(model, opt, ema, x, y, config, _context(loader, len(y)), diagnose=False)
        _exact(expected, _snapshot(model, opt, ema, loader))
        # Populated source identity with the same frozen ordinary M3oE.
        prefix = _snapshot(model, opt, ema, loader)
        zero = dict(config, mixture_lambda=0.)
        core.train_step(model, opt, ema, x, y, zero, _context(loader, len(y), 2), diagnose=True)
        expected = _snapshot(model, opt, ema, loader)
        _restore(model, opt, ema, loader, prefix)
        core.source_clean_step(model, opt, ema, x, y, zero)
        _exact(expected, _snapshot(model, opt, ema, loader))
        comparisons += 2
    return dict(actual_m3oe_comparisons=comparisons, core_calls=6, direct_source_calls=2,
                diagnosed_core_calls=4, extra_autograd_sweeps=12, virtual_source_adam_steps=16)


def check_core():
    original = cpu_copy(rng_state('cpu'))
    try:
        setup(2022, 'cpu')
        return dict(status='PASS', device='cpu', tolerance_fp64=dict(atol=1e-12, rtol=1e-10),
                    tolerance_fp32=dict(atol=2e-6, rtol=2e-5), exact_hash_tolerance=None,
                    losses=_loss_checks(), source_state=_step_checks(),
                    validation=_validation_checks(), source_model=_source_model_checks(),
                    tensor_storage='CPU RAM and BytesIO only', test_accessed=False)
    finally:
        restore_rng(original, 'cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('core', 'all'), default='core')
    args = parser.parse_args()
    result = {'core': check_core()}
    if args.scope == 'all':
        for name in ('protocol', 'runner', 'tables'):
            qualified = f'{__package__}.{name}'
            if importlib.util.find_spec(qualified) is None:
                result[name] = dict(status='INTEGRATION_PENDING', reason='runtime module not yet present')
            else:
                module = importlib.import_module(qualified)
                result[name] = module.finite_checks()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
