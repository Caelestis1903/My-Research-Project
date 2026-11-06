"""Small checks for the statistical contracts that can change conclusions."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.ema_mechanism_20261106 import analysis as a

MANIFEST = Path(__file__).resolve().parents[3]/'research_plans/26-11-6/experiment_manifest.json'


def test_ridge_uses_sse_alpha_one_and_unpenalized_intercept():
    x = np.array([[-1.], [0.], [1.]])
    y = np.column_stack([2+3*x[:, 0], -4+6*x[:, 0]])
    prediction = a.ridge_predict(x, y, np.array([[0.], [2.]]))
    # X'X=2, hence coefficient 2/3 of the unconstrained slope.
    np.testing.assert_allclose(prediction, [[2., -4.], [6., 4.]], rtol=1e-14, atol=1e-14)


def test_action_keeps_F_and_fixed_tie_order():
    prediction = np.array([[-1., -2.], [0., 0.], [1., 1.], [0., 2.]])
    np.testing.assert_array_equal(a.action_indices(prediction), [0, 0, 1, 2])
    targets = np.array([[.1, .2], [.2, -.1], [-.3, -.2], [.1, .3]])
    result = a._row_outcomes(targets, prediction, np.zeros(4, dtype=int))
    np.testing.assert_allclose(result['chosen_minus_F'], [0, 0, -.3, .3])
    np.testing.assert_allclose(result['regret'], [.2, .2, .3, 0])


def test_source_logloss_endpoints_and_temperature_identity():
    p = np.array([0., 0., .25, .25, 1., 1.], dtype=np.float32)
    y = np.array([1, 0, 1, 0, 1, 0])
    expected = (200-np.log(.25)-np.log(.75))/6
    assert a.source_logloss(y, p) == pytest.approx(expected)
    unchanged = a.temperature_map(p, 1)
    assert unchanged.dtype == p.dtype
    np.testing.assert_array_equal(unchanged, p)
    transformed = a.temperature_map(p, .5)
    assert transformed[0] == 0 and transformed[-1] == 1
    assert np.all(np.diff(transformed) >= 0)
    assert transformed[2] == transformed[3]


def test_future_fold_seed_time_and_validation_row_isolation():
    spec = json.loads(MANIFEST.read_text())['predictive']
    rows = []
    for seed in a.SEEDS:
        for condition in a.CONDITIONS:
            for epoch in spec['train_origins']+spec['score_origins']:
                for panel in ('V_A', 'V_B'):
                    for opportunity in range(24):
                        rows.append(dict(seed=seed, objective=condition, origin_epoch=epoch,
                                         target_epoch=epoch+5, label_panel=panel))
    frame = pd.DataFrame(rows)
    train, score = a.predictive_split(frame, 2022, 'V_A', 'V_B', spec)
    assert len(train) == 576 and len(score) == 240
    assert 2022 not in set(train['seed']) and set(score['seed']) == {2022}
    assert train['target_epoch'].max() == 20 and score['origin_epoch'].min() == 25
    assert set(train['label_panel']) == {'V_A'} and set(score['label_panel']) == {'V_B'}
    with pytest.raises(ValueError, match='opposite'):
        a.predictive_split(frame, 2022, 'V_A', 'V_A', spec)


def test_continuous_scaling_fits_only_training_rows_and_keeps_onehot():
    spec = json.loads(MANIFEST.read_text())['predictive']
    train = pd.DataFrame([dict(objective='clean', domain=0, task=0, group_id='S', origin_epoch=e,
                               gap_rms=v, pulse_fastslow_rms=0., gap_pulse_cos=None, cos_missing_flag=1)
                          for e, v in [(5, 1.), (10, 3.)]])
    score = train.iloc[[0]].copy()
    score['gap_rms'] = 1000.
    x_train, x_score, receipt = a.feature_matrices(train, score, 'M0', spec)
    other = score.copy()
    other['gap_rms'] = -1000.
    _, _, other_receipt = a.feature_matrices(train, other, 'M0', spec)
    assert receipt == other_receipt
    np.testing.assert_array_equal(x_train[:, :48].sum(axis=1), [1, 1])
    assert np.max(np.abs(x_score[:, 48:])) > 100
    assert receipt['constant_flags'][-1]


def test_future_targets_come_from_new_pulse_and_24_support_rows():
    source = dict(schema_version=1, job_id='one', dataset='movielens', objective='clean', seed=2022,
                  epoch=5, global_step=100, commit_index=5, pulse_readouts=[])
    features = [dict(domain=d, task=t, group_id=g, panel_id='P_A', functional_support=True)
                for d in range(3) for t in range(2) for g in a.GROUPS if a.supported(d, t, g)]
    source['pulse_metrics'] = features
    future = dict(source, epoch=10, pulse_metrics=[], pulse_readouts=[])
    for panel in ('V_A', 'V_B'):
        for d in range(3):
            for t in range(2):
                future['pulse_readouts'].append(dict(panel_id=panel, domain=d, task=t, group_id=None,
                    variant='F', auc=.6, logloss=.5, state_hash='F10'))
                for group in a.GROUPS:
                    for variant, auc in [('fast', .61), ('slow', .59)]:
                        future['pulse_readouts'].append(dict(panel_id=panel, domain=d, task=t, group_id=group,
                            variant=variant, auc=auc, logloss=.49, state_hash=variant+'10'))
    rows = a.predictive_rows(None, [source, future])
    assert len(rows) == 48 and set(rows['target_epoch']) == {10}
    np.testing.assert_allclose(rows['y_fastF'], .01)
    np.testing.assert_allclose(rows['y_slowF'], -.01)
    assert set(rows['target_fast_state_hash']) == {'fast10'}


def test_missing_pair_padding_is_never_indexed():
    pairs = np.full((6, 4, 2), -1, dtype=np.int32)
    pairs[0, 0] = [0, 1]
    identity = dict(pairs=pairs, pair_counts=np.array([1, 0, 0, 0, 0, 0]))
    np.testing.assert_array_equal(a._panel_pairs(identity, 'V_mech', 0), [[0, 1]])
    assert a._pair_scores(np.array([.8, .1]), a._panel_pairs(identity, 'V_mech', 1)).size == 0


def test_seed_summary_averages_correlated_rows_before_seed_statistics():
    rows = pd.DataFrame([dict(dataset='x', seed=s, value=float(s-2022))
                         for s in a.SEEDS for _ in range(30)])
    result = a.summarize_seeds(rows, ['dataset']).iloc[0]
    assert result['n_seeds'] == 5
    assert result['mean'] == 2
    assert result['sample_sd'] == pytest.approx(np.std(range(5), ddof=1))


def test_ragged_validation_masks_metrics_pairs_flips_and_calibration(tmp_path, monkeypatch):
    indices = np.full((3, 6), -1, dtype=np.int64)
    y = np.full((3, 6, 2), -1, dtype=np.float32)
    side = np.full((3, 6), -1, dtype=np.int8)
    p = np.zeros((3, 6, 2), dtype=np.float32)
    for d, count in enumerate((4, 6, 4)):
        indices[d, :count] = np.arange(count)
        y[d, :count, 0] = np.arange(count) % 2
        y[d, :count, 1] = 1-y[d, :count, 0]
        side[d, :count] = np.repeat([0, 1], count//2)
        p[d, :count] = .2+.6*y[d, :count]
    pairs = np.tile(np.array([[1, 0], [3, 2]], dtype=np.int32), (6, 1, 1))
    # Task1 has the opposite labels.
    pairs[1::2] = pairs[1::2, :, ::-1]
    identity = dict(indices=indices, y=y, validation_row_side=side, pairs=pairs,
                    pair_counts=np.full(6, 2))
    for row_side in (None, 0, 1):
        result = a._panel_metric_matrix(identity, p, row_side)
        for c in range(6):
            d, t = c//2, c % 2
            valid = indices[d] >= 0
            if row_side is not None:
                valid &= side[d] == row_side
            assert result[c, 0] == a.cell_auc(y[d, valid, t], p[d, valid, t])
            assert result[c, 1] == a.source_logloss(y[d, valid, t], p[d, valid, t])
    np.testing.assert_array_equal(a._panel_pairs(identity, 'V_mech', 0), pairs[0])
    invalid = dict(identity, pairs=pairs.copy())
    invalid['pairs'][0, 0, 0] = 4
    with pytest.raises(ValueError, match='invalid container'):
        a._panel_pairs(invalid, 'V_mech', 0)

    rank_y = np.tile([0, 1, 0, 1], (6, 1)).astype(np.float32)
    train = dict(rank_y=rank_y, rank_pairs=np.tile([[1, 0], [3, 2]], (6, 1, 1)))
    monkeypatch.setattr(a, 'panel_identity', lambda out, ds, seed, panel: identity if panel == 'V_mech' else train)
    monkeypatch.setattr(a, '_prediction', lambda out, record, panel, rid: {'p': p} if panel == 'V_mech' else {'rank_p': .2+.6*rank_y})
    study = SimpleNamespace(manifest={'datasets': {'movielens': {'aliases': {'F': 'H2'}}}})
    record = dict(job_id='masked', dataset='movielens', objective='clean', seed=2022, epoch=20,
        validation={'H2': {}}, pulse_readouts=[{}],
        predictions=[dict(panel_id=panel, readout_id=rid, physical_readout_id=rid)
                     for panel in ('P_A', 'P_B', 'V_mech') for rid in ('H2', 'fast_S')])
    flipped = a.rank_flips(study, tmp_path, [record])
    v = flipped[flipped['panel_id'] == 'V_mech']
    np.testing.assert_allclose(v['exact_panel_auc'], 1.)
    np.testing.assert_allclose(v['exact_auc_delta'], 0.)

    original_ll = a.source_logloss
    observed_labels = []
    def checked_ll(labels, probability):
        assert np.all(np.isin(labels, [0, 1]))
        observed_labels.append(len(labels))
        return original_ll(labels, probability)
    monkeypatch.setattr(a, 'source_logloss', checked_ll)
    def temperature_fit(fn, **kwargs):
        fn(1.)
        return SimpleNamespace(success=True, x=1., nit=1)
    monkeypatch.setattr(a, 'minimize_scalar', temperature_fit)
    records = [dict(dataset='movielens', objective='clean', seed=s, epoch=e,
                    model_hash={'online': 'online', 'H2': 'F'}) for s in a.SEEDS for e in (20, 40)]
    result = a._calibration_controls(study, tmp_path, records,
        {'triggered': True, 'strata': [{'dataset': 'movielens', 'objective': 'clean'}]})
    assert len(result) == 240 and set(observed_labels) == {2, 3}
