"""Audit producer batch composition, importance moments and short-batch clocks."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import v3_readout_20261016 as r


def moments(counts, weights):
    n = counts.sum(axis=1)
    first, second = counts @ weights, counts @ np.square(weights)
    return np.column_stack((first ** 2 / second,
        second / n - np.square(first / n),
        np.max(np.where(counts > 0, weights[None, :], -np.inf), axis=1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    rows, sources, comparisons, frames = [], [], [], {}
    for text in spec['runs']:
        directory = Path(text)
        producer = json.loads((directory / 'receipt.json').read_text())
        assert producer['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
        config = json.loads((directory / 'config.json').read_text())
        for name in ('config.json', 'trajectory.csv', 'batch_trace.csv'):
            assert r.base.sha(directory / name) == producer['artifact_hashes'][name]
        frame = pd.read_csv(directory / 'batch_trace.csv')
        trajectory = pd.read_csv(directory / 'trajectory.csv').set_index('epoch')
        epochs = list(range(1, producer['epochs'] + 1))
        assert trajectory.index.tolist() == epochs
        assert sorted(frame.epoch.unique()) == epochs
        assert not frame.duplicated(['epoch', 'batch']).any()
        counts = frame[['d0', 'd1', 'd2']].to_numpy(dtype=np.int64)
        assert (counts >= 0).all() and np.array_equal(counts.sum(1), frame.n)
        weights = np.asarray(config['p']) / np.asarray(config['q'])
        calculated = moments(counts, weights)
        reported = frame[['ess', 'weight_variance', 'max_weight']].to_numpy()
        errors = np.max(abs(calculated - reported), axis=0)
        relative = np.max(abs(calculated - reported) / np.maximum(abs(reported), 1), axis=0)
        assert np.isfinite(calculated).all() and np.max(relative) < 1e-10
        for epoch, part in frame.groupby('epoch', sort=True):
            assert part.batch.tolist() == list(range(len(part)))
            assert int(part.n.sum()) == sum(config['data']['train_domain_counts'].values())
            exposure = part[['d0', 'd1', 'd2']].sum().to_numpy(dtype=np.int64)
            assert np.array_equal(exposure, trajectory.loc[epoch, ['exposure_d0', 'exposure_d1', 'exposure_d2']].to_numpy(dtype=np.int64))
            short = part[part.n < config['batch_size']]
            assert len(short) <= 1 and (part.n <= config['batch_size']).all()
            tail = None if short.empty else int(short.iloc[0].batch)
            if config['order'] == 'NATURAL' and tail is not None:
                assert tail == len(part) - 1
            rows.append(dict(run=str(directory), seed=config['seed'], optimizer=config['optimizer'],
                arm=config['arm'], replacement=config['replacement'], order=config['order'],
                epoch=int(epoch), actual_final_budget=producer['epochs'], batches=len(part),
                sample_draws=int(part.n.sum()), d0_draws=int(exposure[0]),
                d1_draws=int(exposure[1]), d2_draws=int(exposure[2]),
                each_task_draws_implied_by_fully_observed_two_task_training=int(part.n.sum()),
                mean_ess_fraction=float((part.ess / part.n).mean()),
                min_ess_fraction=float((part.ess / part.n).min()),
                max_ess_fraction=float((part.ess / part.n).max()),
                mean_weight_variance=float(part.weight_variance.mean()),
                max_importance_weight=float(part.max_weight.max()),
                short_batch_size=None if short.empty else int(short.iloc[0].n),
                short_batch_actual_position=tail,
                short_batch_moved_from_last=tail is not None and tail != len(part) - 1))
        frames[str(directory)] = frame
        sources.append(dict(run=str(directory), producer_receipt_sha=r.base.sha(directory / 'receipt.json'),
            actual_epochs=producer['epochs'], batches_checked=len(frame),
            config_sha=r.base.sha(directory / 'config.json'),
            trace_sha=r.base.sha(directory / 'batch_trace.csv'),
            trajectory_sha=r.base.sha(directory / 'trajectory.csv'),
            max_absolute_errors=errors.tolist(), max_relative_errors=relative.tolist()))
    integer_fields = ['n', 'd0', 'd1', 'd2']
    for item in spec['comparisons']:
        left, right = frames[item['baseline']], frames[item['variant']]
        assert left.epoch.tolist() == right.epoch.tolist()
        kind = item['intervention']
        assert kind in ('PERM', 'ORDER')
        for epoch, a in left.groupby('epoch'):
            b = right[right.epoch == epoch]
            if kind == 'PERM':
                assert np.array_equal(a[integer_fields], b[integer_fields])
            else:
                assert sorted(map(tuple, a[integer_fields].to_numpy())) == sorted(map(tuple, b[integer_fields].to_numpy()))
            for metric in ('ess', 'weight_variance', 'max_weight'):
                av, bv = a[metric].to_numpy(), b[metric].to_numpy()
                if kind == 'ORDER':
                    av, bv = np.sort(av), np.sort(bv)
                assert np.allclose(av, bv, rtol=1e-10, atol=1e-10)
        comparisons.append(dict(**item, epochs_checked=int(left.epoch.max()),
            batches_checked=len(left), identity='PASS', scope='Actual domain counts and weight moments; losses and parameter paths not equated.'))
    args.out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(args.out / 'epoch_weight_and_tail.csv', index=False)
    r.base.dump(args.out / 'comparison_identities.json', comparisons)
    r.base.dump(args.out / 'receipt.json', dict(status='PASS', source_sha=r.base.sha(__file__),
        manifest_sha=r.base.sha(args.manifest), sources=sources, runs=len(sources),
        actual_epochs_audited=len(rows), comparisons=len(comparisons),
        numerical_tolerance_relative=1e-10, independent_review=False,
        actual_trace_epoch_and_batch_indices=True, no_test_performance_access=True,
        task_count_scope='Both tasks observed for each training row in unchanged two-task BCE implementation; implied count, not a new missingness measurement.',
        scope='Finite producer trace/weight/tail identities; ESS is not gradient covariance or downstream performance causality. Full raw batch NPZ identity checks remain in sampler receipts.'))


if __name__ == '__main__':
    main()
