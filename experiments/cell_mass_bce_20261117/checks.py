"""Finite CPU fixtures for C05; no cache, owner IPC, GPU or Test access."""
import argparse
import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from . import core
from experiments.auxiliary_objective_attribution_20261115.protocol import (
    cpu_copy, make_model, optimizer, restore_rng, rng_state, setup, state_digest)
from experiments.completion_weight_attribution_20261117.checks import (
    _batch, _FixtureModel, _snapshot, _restore, _roundtrip, _exact, _close, _isolated)
from experiments.conditional_information_momentum_20261102_v3.protocol import metrics
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.protocol import selected_update


def _config(arm=core.ARMS[0], dataset='movielens', mixture_lambda=.25):
    K, tail, wd, rho = core._DATA[dataset]
    return dict(cycle_id='cycle_05', dataset=dataset, arm=arm, objective='cell_mass_bce_v1',
        mixture_lambda=mixture_lambda, epochs=50, batch_size=40960, precision='float32', lr=.01,
        betas=[.9, .999], eps=1e-8, weight_decay=wd, rho=rho, steps_per_epoch=K, tail_rows=tail,
        loader_seed_offset=11000, gate_training_detached=True, probe_gate_full_derivative=False,
        algorithm_rng_draws=0, data_hash=core._DATA_HASH[dataset], selection_hash=core.SELECTION_HASH)


def _context(loader, rows, step=1):
    return dict(purpose='engineering_check', clock_kind='engineering', dataset='movielens', seed=2022,
        case_id='finite_core', engineering_call_ordinal=step, engineering_state_step=step,
        sample_epoch=1, sample_batch=1, branch_role='fixture', fixture_rows=rows,
        pre_loader_hash=state_digest(loader.get_state()))


def _loss_checks():
    for dtype in (torch.float64, torch.float32):
        x, y = _batch(dtype)
        p = torch.tensor([[.17, .28], [.31, .47], [.56, .71], [.82, .86], [.63, .22], [.79, .34], [.73, .67]],
                         dtype=dtype, requires_grad=True)
        result = core.loss_components(p, y, x[-1], diagnose=True)
        ledger = result['ledger']
        frozen = core.frozen_components(p, y, x[-1], arm='COMPLETION-BCE-025', diagnose=False)
        for key in ('clean_loss', 'same_class_loss', 'completion_loss', 'scaled_loss'):
            _exact(result[key], frozen[key])
        m = ledger['bce_mass_scale']
        assert .875 <= m <= 1 and ledger['entries'] == 2*len(y)
        assert ledger['shared_completion_cell_mass_exact']
        entries = F.binary_cross_entropy(p, y, reduction='none')
        # Independently enumerate coefficients using explicit fixture cell counts.
        counts = (((3, 1), (1, 3)), ((0, 3), (2, 1)), ((0, 0), (0, 0)))
        hand = p.new_zeros(())
        covariance = p.new_zeros(())
        allocation = p.new_zeros(())
        for d in range(3):
            for t in range(2):
                c = ledger['cells'][f'd{d}t{t}']
                n0, n1 = counts[d][t]
                n = n0+n1
                assert (c['n0'], c['n1'], c['n']) == (n0, n1, n)
                assert c['completion_bce_mass'] == c['cell_mass_bce_mass'] == frozen['ledger']['cells'][f'd{d}t{t}']['completion_bce_mass']
                if not n:
                    assert c['a'] is c['kappa'] is c['pi0'] is c['pi1'] is None
                    assert set(c['statistics_status'].values()) == {'empty_cell'}
                    assert c['completion_bce_mass'] == c['cell_mass_bce_mass'] == 0
                    continue
                pi = (n0/n, n1/n)
                a = 1-.25*2*pi[0]*pi[1]
                kappa = pi[0]**2+pi[1]**2
                cell_mean = entries[x[-1] == d, t].mean()
                allocation = allocation+c['weight']*(a-m)*cell_mean
                for label in (0, 1):
                    mask = (x[-1] == d) & (y[:, t] == label)
                    if counts[d][t][label]:
                        hand = hand+entries[mask, t].sum()*a/(2*len(y))
                        covariance = covariance+.25*c['weight']*pi[label]*(pi[label]-kappa)*entries[mask, t].mean()
                        _close(c[f'cell_mass_class_mass{label}']-c[f'completion_class_mass{label}'],
                               counts[d][t][label]*(a-(.75+.25*pi[label]))/(2*len(y)))
                        _close(c[f'cell_mass_coefficient{label}'], a/(2*len(y)))
                        assert c[f'q{label}'] == .75+.25*pi[label]
                        if counts[d][t][label] == 1:
                            assert c[f'var{label}'] == 0
                    else:
                        assert c[f'cell_mass_coefficient{label}'] is c[f'mu{label}'] is None
                        assert c[f'cell_mass_class_mass{label}'] == c[f'completion_class_mass{label}'] == 0
                        assert c['statistics_status'][f'mu{label}'] == 'absent_class'
                _close(sum(c[f'cell_mass_class_mass{label}'] for label in (0, 1)), c['cell_mass_bce_mass'])
                _close(sum(c[f'completion_class_mass{label}'] for label in (0, 1)), c['completion_bce_mass'])
        for left, right in ((result['cell_mass_loss'], hand),
                            (result['completion_loss']-result['cell_mass_loss'], covariance),
                            (result['cell_mass_loss']-result['scaled_loss'], allocation)):
            _close(left, right, dtype)
            a, = torch.autograd.grad(left, p, retain_graph=True)
            b, = torch.autograd.grad(right, p, retain_graph=True)
            _close(a, b, dtype)
        _close(sum(c['cell_mass_bce_mass'] for c in ledger['cells'].values()), m)
        _close(sum(ledger['cell_mass_task_mass'].values()), m)
        assert ledger['task_mass'] == ledger['completion_task_mass'] == ledger['cell_mass_task_mass']
        assert ledger['task_mass']['0'] != ledger['task_mass']['1']
        off = core.loss_components(p, y, x[-1])['ledger']
        assert off['cells']['d0t0']['statistics_status']['mu0'] == 'not_scheduled'
        assert off['cells']['d1t0']['statistics_status']['mu0'] == 'absent_class'
        assert off['cells']['d1t0']['statistics_status']['Q_R'] == 'requires_both_classes'
        assert set(off['cells']['d2t0']['statistics_status'].values()) == {'empty_cell'}
        for lam, labels, branch in ((0., y, 'lambda_zero'), (.25, torch.ones_like(y), 'all_single_class')):
            delegated = core.loss_components(p, labels, x[-1], mixture_lambda=lam, diagnose=True)
            assert delegated['branch'] == branch
            for key in ('loss', 'completion_loss', 'scaled_loss', 'cell_mass_loss'):
                _exact(delegated[key], delegated['clean_loss'])
            if branch == 'lambda_zero':
                assert delegated['same_class_loss'] is None and delegated['same_class_loss_null_reason'] == 'not_computed_delegate'
            else:
                _exact(delegated['same_class_loss'], delegated['clean_loss'])
        balanced_y = torch.tensor([[0., 0.], [1., 1.]], dtype=dtype)
        balanced_p = torch.tensor([[.2, .4], [.7, .8]], dtype=dtype)
        balanced = core.loss_components(balanced_p, balanced_y, torch.zeros(2, dtype=dtype), diagnose=True)
        assert balanced['ledger']['total_mass'] == .875
        _close(balanced['completion_loss'], balanced['cell_mass_loss'], dtype)
        boundary_p = torch.tensor([[0., 1.], [1., 0.]], dtype=dtype, requires_grad=True)
        boundary_y = torch.tensor([[1., 1.], [0., 0.]], dtype=dtype)
        boundary = core.loss_components(boundary_p, boundary_y, torch.zeros(2, dtype=dtype))
        _exact(boundary['clean_loss'], F.binary_cross_entropy(boundary_p, boundary_y))
        boundary['loss'].backward()
        assert torch.isfinite(boundary_p.grad).all()
    return dict(dtype_fixtures=2, coefficient_covariance_allocation_gradients=True,
                frozen_C_H_W_S_exact=True, six_cell_W_U_mass_exact=True, natural_entries=14,
                balanced_single_empty_absent_singleton_boundary_lambda_zero=True)


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
                assert on['diagnostics']['parameter_groups']['ALL']['objectives']['U']['tensor_zero_parameters'] >= 1
                assert on['parameter_activity'] == off['parameter_activity']
                assert on['adam_step_counters'] == off['adam_step_counters']
                for field in core.LOSS_FIELDS:
                    assert on[field] == off[field]
                assert tuple(on['diagnostics']['objective_losses']) == core.OBJECTIVES
                assert tuple(on['diagnostics']['virtual_post']) == core.OBJECTIVES
                if branch == 'lambda_zero':
                    assert on['same_class_loss'] is None and off['same_class_loss'] is None
                if branch == 'ordinary':
                    _virtual_oracle(model, opt, ema, loader, before, x, labels, c, on)
                    _restore(model, opt, ema, loader, on_state)
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
                virtual_source_adam_steps=4*diagnosed, oracle_steps=8, oracle_forwards=8, bytesio_prefix_suffix=True,
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



def _virtual_oracle(model, opt, ema, loader, before, x, y, config, receipt):
    """Four independent ordinary source-Adam steps from the same fixture prestate."""
    fields = dict(C='clean_loss', S='scaled_loss', U='cell_mass_loss', W='completion_loss')
    for objective, field in fields.items():
        _restore(model, opt, ema, loader, before)
        opt.zero_grad(set_to_none=True)
        components = core.loss_components(model(x), y, x[-1], mixture_lambda=config['mixture_lambda'])
        components[field].backward()
        opt.step()
        expected = dict(model_hash=state_digest(model.state_dict()), optimizer_hash=state_digest(opt.state_dict()))
        assert expected == receipt['diagnostics']['virtual_post'][objective]


def _source_model_checks():
    features = SimpleNamespace(columns=['user', 'domain_indicator'], dense_columns=[],
                               vocab_sizes={'user': 8, 'domain_indicator': 3})
    model = make_model(SimpleNamespace(features=features), {'perturbable_sparse_fields': ['user']})
    ema = copy.deepcopy(model)
    config = _config()
    opt = optimizer(model, config)
    x, y = _batch()
    x[0] = torch.arange(len(y))
    loader = torch.Generator().manual_seed(13022)
    gate_inputs = []
    embedding_outputs = []
    handles = [gate.register_forward_pre_hook(lambda module, args: gate_inputs.append(
        (args[0].requires_grad, args[0].grad_fn))) for gate in model.gate]
    handles.append(model.embedding.register_forward_hook(
        lambda module, args, output: embedding_outputs.append(output.requires_grad)))
    try:
        p = model(x)
        assert p.shape == y.shape == (7, 2)
        F.binary_cross_entropy(p, y).backward()
        assert gate_inputs == [(False, None)]*6 and embedding_outputs == [True]
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.embedding.parameters())
        assert any(p.grad is not None for p in model.gate.parameters())
    finally:
        for handle in handles:
            handle.remove()
    before = _snapshot(model, opt, ema, loader)
    on = core.train_step(model, opt, ema, x, y, config, _context(loader, len(y)), diagnose=True)
    expected = _snapshot(model, opt, ema, loader)
    _restore(model, opt, ema, loader, before)
    off = core.train_step(model, opt, ema, x, y, config, _context(loader, len(y)), diagnose=False)
    _exact(expected, _snapshot(model, opt, ema, loader))
    assert on['parameter_activity'] == off['parameter_activity']
    groups = on['diagnostics']['parameter_groups']
    prefixes = sorted({n.split('.')[0] for n, _ in model.named_parameters()})
    assert list(groups) == ['ALL', *prefixes]
    assert {'embedding', 'gate', 'expert', 'domain_expert', 'task_expert', 'tower'} <= set(prefixes)
    assert groups['ALL']['coordinates'] == sum(p.numel() for p in model.parameters())
    for group in groups.values():
        assert 'parameter_names' not in group
        assert tuple(group['objectives']) == core.OBJECTIVES
        assert tuple(group['pairs']) == ('U-W', 'U-S', 'S-C')
    prefix = _snapshot(model, opt, ema, loader)
    zero = dict(config, mixture_lambda=0.)
    core.train_step(model, opt, ema, x, y, zero, _context(loader, len(y), 2), diagnose=True)
    expected = _snapshot(model, opt, ema, loader)
    _restore(model, opt, ema, loader, prefix)
    core.source_clean_step(model, opt, ema, x, y, zero)
    _exact(expected, _snapshot(model, opt, ema, loader))
    return dict(actual_m3oe_comparisons=2, core_calls=3, direct_source_calls=1,
                diagnosed_core_calls=2, extra_autograd_sweeps=6, virtual_source_adam_steps=8,
                separate_layout_backward_forwards=1, ordinary_gate_detach=True, prefixes=prefixes)


def _metrics_selection_checks():
    domains = np.repeat(np.arange(3), 4)
    y = np.tile(np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]), (3, 1))
    online = (.3+.4*y, .1+.8*y, .2+.6*y)
    fixed_ema = (.7-.4*y, .2+.6*y, .1+.8*y)
    best = dict(online=-1., fixed_ema=-1.)
    selected, selected_values, selected_hashes = {}, {}, {}
    rows = []
    for e, predictions in enumerate(zip(online, fixed_ema), 1):
        row = {}
        for stream, prediction in zip(best, predictions):
            value = metrics(y, prediction, domains)
            raw_bce = F.binary_cross_entropy(torch.tensor(prediction), torch.tensor(y)).item()
            _close(value['pooled_logloss'], raw_bce)
            _close(value['macro_logloss'], raw_bce)
            for d in range(3):
                for t in range(2):
                    _close(value[f'd{d}t{t}_logloss'], F.binary_cross_entropy(
                        torch.tensor(prediction[domains == d, t]), torch.tensor(y[domains == d, t])).item())
            if selected_update(selected, best, stream, e, value['macro_auc']):
                selected_values[stream] = value.copy()
                selected_hashes[stream] = state_digest(dict(stream=stream, epoch=e, state=torch.tensor([e])))
            row[stream] = value
        rows.append(row)
    assert selected == dict(online=1, fixed_ema=2)
    for stream, e in selected.items():
        assert selected_values[stream] == rows[e-1][stream]
        assert selected_hashes[stream] == state_digest(dict(stream=stream, epoch=e, state=torch.tensor([e])))
        assert selected_values[stream]['macro_logloss'] > min(r[stream]['macro_logloss'] for r in rows)
    undefined = metrics(np.zeros_like(y), np.full_like(y, .5), domains)
    assert undefined['macro_auc'] is None and all(undefined[f'd{d}t{t}_auc'] is None for d in range(3) for t in range(2))
    return dict(six_cell_source_metrics=True, independent_earliest_strict_selection=True,
                unscaled_loss_bound_to_auc_selected_state=True, undefined_auc_null=True,
                paired_families_and_table_counts='tables.finite_checks (runtime scope)')


def check_core():
    original = cpu_copy(rng_state('cpu'))
    try:
        setup(2022, 'cpu')
        return dict(status='PASS', device='cpu', tolerance_fp64=dict(atol=1e-12, rtol=1e-10),
                    tolerance_fp32=dict(atol=2e-6, rtol=2e-5), exact_hash_tolerance=None,
                    losses=_loss_checks(), source_state=_step_checks(),
                    validation=_validation_checks(), source_model=_source_model_checks(),
                    metrics_selection=_metrics_selection_checks(),
                    tensor_storage='CPU RAM and BytesIO only', test_accessed=False)
    finally:
        restore_rng(original, 'cpu')


def finite_checks():
    return check_core()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('check',))
    args = parser.parse_args()
    print(json.dumps({args.command: finite_checks()}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
