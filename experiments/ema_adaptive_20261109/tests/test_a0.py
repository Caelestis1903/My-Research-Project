import copy

import numpy as np
import pandas as pd
import pytest

from experiments.ema_adaptive_20261109 import analysis as a
from experiments.ema_adaptive_20261109.audit import Incomplete, SEEDS


def pulse_rows():
    rows = []
    for d in range(3):
        for t in range(2):
            for variant, auc, ll in [('F', .5, .8), ('fast', .6, .7), ('slow', .45, .9)]:
                rows.append(dict(dataset='kuairand', objective='gaussian_d0', seed=2022, epoch=5,
                    panel_id='P_A', domain=d, task=t, job_id='job', variant=variant,
                    group_id=np.nan if variant == 'F' else 'S', physical_readout_id=variant,
                    auc=auc, logloss=ll, state_hash=variant, actual_count=512, missing_reason='',
                    init_hash='i', order_hash='o', layout_hash='l', group_hash='g', identity_id='x'))
    return pd.DataFrame(rows)


def test_pairing_signs_alias_and_conflict():
    raw = pulse_rows()
    got = a.contrasts(pd.concat([raw, raw.iloc[:1]], ignore_index=True))
    assert len(got) == 6
    assert np.allclose(got.g_fast, .1) and np.allclose(got.g_slow, -.05)
    assert np.allclose(got.l_fast, .1) and np.allclose(got.l_slow, -.1)
    bad = raw.iloc[:1].copy(); bad.auc = .9
    with pytest.raises(Incomplete):
        a.contrasts(pd.concat([raw, bad], ignore_index=True))


def test_aggregate_before_action():
    raw = pd.DataFrame([dict(dataset='d', objective='o', seed=2022, epoch=5, domain=d, task=t,
        fast=([3., -1., -1.][d]), slow=0.) for d in range(3) for t in range(2)])
    got = a.scopes(raw, ['fast', 'slow'])
    assert got[got.scope == 'GLOBAL'].fast.iloc[0] == pytest.approx(1/3)
    assert got[got.scope == 'TASK1'].fast.iloc[0] == pytest.approx(1/3)
    assert np.argmax([0., 0., 0.]) == 0


def test_lag_does_not_bridge_missing_window_and_zero_stats():
    rows = []
    for epoch in (5, 15):
        for d in range(3):
            for t in range(2):
                rows.append(dict(dataset='d', objective='o', seed=2022, epoch=epoch,
                                 panel_id='P_A', domain=d, task=t, **{k: .1 for k in a.SIGNALS}))
    lag, stats = a.persistence(pd.DataFrame(rows))
    assert lag.empty
    assert np.isnan(a.correlation([1, 1], [2, 3], 'pearson'))
    s = a.sequence_stats([0, 1, -1], [1, 1, -1])
    assert s['nonzero_pairs'] == 2 and s['sign_agreement'] == 1
    assert s['zero_pair_fraction'] == pytest.approx(1/3)


def test_forward_fold_is_seed_and_time_disjoint():
    frame = pd.DataFrame([dict(dataset='kuairand', seed=seed, origin_epoch=e, target_epoch=e+5)
        for seed in SEEDS for e in (5, 10, 15, 25, 30, 35, 40, 45) for _ in range(12)])
    train, score = a.split(frame, 'kuairand', 2022)
    assert len(train) == 144 and len(score) == 60
    assert 2022 not in set(train.seed)
    assert train.target_epoch.max() < score.origin_epoch.min()


def test_fit_standardization_ignores_score_and_validation():
    train = pd.DataFrame([dict(objective='clean', domain=0, task=0, group_id='S', origin_epoch=e,
        current_fast=x, current_slow=-x, y_fastF=x/10, y_slowF=-x/10) for e, x in [(5, 1.), (10, 2.), (15, 3.)]])
    score = train.iloc[:1].copy()
    spec = {'feature_sets': {'C_persist': ['design', 'persist']}, 'feature_blocks': {'persist': ['current_fast', 'current_slow']}}
    first, transform = a.fit_predictions(train, score, spec, ['C_persist'])
    score['V_A_auc'] = 1000.; score['y_fastF'] = -1000.
    second, transform2 = a.fit_predictions(train, score, spec, ['C_persist'])
    assert np.array_equal(first['C_persist'], second['C_persist'])
    assert transform == transform2
    assert transform['C_persist']['continuous_mean'][1] == 2.


def gate_tables(dynamic=True, static=True):
    preds, stats = [], []
    for seed in SEEDS:
        for feature in ('P_A', 'P_B'):
            for panel in ('P_A', 'P_B', 'V_A', 'V_B'):
                for scope in ('GLOBAL', 'TASK0', 'TASK1'):
                    for model in ('C_design', 'C_simple', 'C_persist', 'C_carry'):
                        good = dynamic and model == 'C_persist'
                        preds.append(dict(dataset='kuairand', objective='gaussian_d0', seed=seed, feature_panel=feature,
                            score_panel=panel, scope=scope, model_id=model, MAE_D=.1 if good else .3,
                            chosen_minus_F=.1 if good else -.1, chosen_minus_uniform=.1 if good else -.1,
                            chosen_minus_C_design=.1 if good else -.1, chosen_minus_C_simple=.1 if good else -.1,
                            direction_correct=.9 if model == 'C_carry' else .5))
                    stats.append(dict(dataset='kuairand', objective='gaussian_d0', seed=seed, feature_panel=feature,
                        score_panel=panel, scope=scope, task_minus_F=.1 if static else -.1,
                        task_minus_uniform=.1 if static else -.1))
    return pd.DataFrame(preds), pd.DataFrame(stats)


def test_four_gates_and_seed_unit():
    p, s = gate_tables()
    assert a.gate(p, s)['state'] == 'PERSISTENCE_SUPPORTED'
    p, s = gate_tables(dynamic=False)
    assert a.gate(p, s)['state'] == 'STATIC_PREFERENCE_ONLY'
    p, s = gate_tables(dynamic=False, static=False)
    assert a.gate(p, s)['state'] == 'NO_ACTIONABLE_SIGNAL'
    assert a.gate(p, s, False)['state'] == 'SOURCE_INCOMPLETE'
    assert a.stable([-1., 1., 1., 1., 1.])['pass']
    result = a.seed_statistics([1, 2, 3, 4, 5])
    assert result['mean'] == 3 and result['sample_sd'] == pytest.approx(np.sqrt(2.5))
    assert result['ci95_high'] == pytest.approx(3 + 2.776445*np.sqrt(2.5/5))


def test_forecast_target_denominator_and_validation_cannot_select_actions():
    frame = a.contrasts(pulse_rows())
    frame['epoch'] = 10
    validation = frame.copy(); validation['panel_id'] = 'V_A'
    truth = pd.concat([frame, validation], ignore_index=True)
    oof = frame[['dataset', 'objective', 'seed', 'domain', 'task', 'group_id']].copy()
    oof['origin_epoch'], oof['target_epoch'] = 5, 10
    oof['feature_panel'], oof['model_id'] = 'P_A', 'C_design'
    oof['yhat_fast'], oof['yhat_slow'] = 0., 0.
    policy = pd.DataFrame([dict(dataset='kuairand', objective='gaussian_d0', seed=2022,
        feature_panel='P_A', scope='GLOBAL', static_action=0, fit_mean_fast=0., fit_mean_slow=0.)])
    _, first = a.evaluate_predictions(oof, truth, policy)
    assert first.direction_correct.eq(0).all()
    assert first.action.eq(0).all()
    truth.loc[truth.panel_id == 'V_A', ['g_fast', 'g_slow']] = [100., 200.]
    _, second = a.evaluate_predictions(oof, truth, policy)
    assert np.array_equal(first.action, second.action)


def test_static_task_reverse_is_diagnostic_after_global_transfer_passes():
    predictions, static = gate_tables(dynamic=False)
    reverse_tasks = (static.feature_panel == 'P_B') & static.scope.isin(['TASK0', 'TASK1'])
    static.loc[reverse_tasks, ['task_minus_F', 'task_minus_uniform']] = -.1
    result = a.gate(predictions, static)
    assert result['state'] == 'STATIC_PREFERENCE_ONLY'
    assert result['static'][0]['pass']
    assert all(scope['pass'] for scope in result['static'][1:])
    assert all(scope['reverse_role'] == 'diagnostic_only' for scope in result['static'][1:])
    assert any(not check['pass'] for check in result['static'][1]['checks'])
