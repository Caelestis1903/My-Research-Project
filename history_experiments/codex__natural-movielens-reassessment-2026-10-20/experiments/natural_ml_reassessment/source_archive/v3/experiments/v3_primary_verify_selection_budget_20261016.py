"""Primary verification of actual sparse frozen selection-budget readouts."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import v3_readout_20261016 as r


def main():
    name = sys.argv[1]
    out = r.OUT / 'analysis' / name
    receipt_path = out / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'COMPLETE'
    assert receipt['source_sha'] == r.base.sha(Path(__file__).with_name(
        'v3_selection_budget_confirmation_20261016.py'))
    assert receipt['original_readout_sha'] == r.base.sha(r.__file__)
    assert receipt['analysis_rules_sha'] == r.base.sha(r.OUT / 'analysis_rules_frozen.json')
    assert receipt['design_sha'] == r.base.sha(r.OUT / 'selection_budget_sparse_confirmation_design.json')
    for file, digest in receipt['hashes'].items():
        assert r.base.sha(out / file) == digest
    run = Path(receipt['source_run'])
    producer = json.loads((run / 'receipt.json').read_text())
    assert r.base.sha(run / 'receipt.json') == receipt['producer_receipt_sha']
    count = producer['epochs']
    assert count == receipt['actual_epochs']
    config = json.loads((run / 'config.json').read_text())
    assert config['seed'] == receipt['seed'] and config['arm'] == receipt['arm']
    for file in ('config.json', 'trajectory.csv', 'metrics_selection.csv'):
        assert r.base.sha(run / file) == producer['artifact_hashes'][file]
    split_file = r.OUT / 'splits' / f'{config["dataset"]}_seed{config["seed"]}.npz'
    assert r.base.sha(split_file) == config['split_sha256']
    selection = np.load(split_file)['selection']
    bundle = r.audit.bundle(config['dataset'], config['seed'])
    cells, points = [], []
    for epoch in range(1, count + 1):
        file = f'validation_epoch{epoch:03d}.npz'
        assert r.base.sha(run / file) == producer['artifact_hashes'][file] == receipt['all_epoch_prediction_hashes'][file]
        actual = r.train.metrics(bundle, np.load(run / file)['predictions'], 'val', selection)
        cells.extend(dict(x, epoch=epoch) for x in actual)
        points.append([np.mean([x[m] for x in actual]) for m in ('auc', 'logloss')])
    point = np.asarray(points)
    raw = pd.read_csv(run / 'metrics_selection.csv')
    merged = pd.DataFrame(cells).merge(raw, on=['epoch', 'domain', 'task'], suffixes=('_new', '_old'), validate='one_to_one')
    cell_error = max(float(abs(merged[m + '_new'] - merged[m + '_old']).max()) for m in ('auc', 'logloss'))
    assert len(merged) == 6 * count and cell_error < 1e-12
    reported_points = pd.read_csv(out / 'selection_point_all_epochs.csv')
    assert reported_points.epoch.tolist() == list(range(1, count + 1))
    point_error = float(abs(reported_points[['auc', 'logloss']].to_numpy() - point).max())
    assert point_error < 1e-12
    trajectory = pd.read_csv(run / 'trajectory.csv')
    objective = trajectory.train_empirical_objective.to_numpy()
    expected = []
    for start in range(1, count - 2):
        peak = int(np.argmax(point[:start, 0]))
        window = np.arange(start, start + 3)
        if (all(point[window, 0] < point[peak, 0]) and
                all(point[window, 1] > point[peak, 1]) and objective[start + 2] < objective[peak]):
            expected.append((peak, window))
    needed = sorted({int(i) + 1 for peak, window in expected for i in [peak, *window]})
    assert needed == receipt['needed_epochs'] and len(expected) == receipt['eligible_windows']
    checks = json.loads((out / 'candidate_windows.json').read_text())
    assert len(checks) == len(expected)
    all_upper, first, raw_rows = [], None, []
    if needed:
        cache = np.load(out / 'selection_bootstrap_needed_epochs.npz')
        assert cache['epochs'].tolist() == needed
        bootstrap = cache['selection']
        assert bootstrap.shape == (1000, len(needed), 2) and np.isfinite(bootstrap).all()
        lookup = {epoch: i for i, epoch in enumerate(needed)}
        for (peak, window), check in zip(expected, checks):
            upper = np.quantile(bootstrap[:, [lookup[int(i) + 1] for i in window], 0]
                - bootstrap[:, lookup[peak + 1], None, 0], .975, axis=0)
            confirmed = bool(all(upper < 0))
            assert check['start_epoch'] == int(window[0]) + 1 and check['peak_epoch'] == peak + 1
            assert check['window_epochs'] == (window + 1).tolist()
            assert np.array_equal(upper, check['upper975']) and confirmed == check['confirmed']
            all_upper.append(upper.tolist())
            if confirmed and first is None:
                first = int(window[0]) + 1
        for b in range(1000):
            for i, epoch in enumerate(needed):
                raw_rows.append(dict(bootstrap=b, epoch=epoch, auc=bootstrap[b, i, 0], logloss=bootstrap[b, i, 1]))
    confirmed = first is not None
    assert receipt['confirmed_overfitting'] == confirmed
    assert receipt['confirmed_overfitting_onset'] == first
    prior, last = float(objective[-10:-5].mean()), float(objective[-5:].mean())
    gain = (prior - last) / max(abs(prior), 1e-12)
    raw_path = r.OUT / 'archive/formal' / (name + '_selection_bootstrap_needed_epochs.csv')
    assert not raw_path.exists()
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    result = dict(status='PASS', time=time.time(), independent_review=False,
        source_sha=r.base.sha(__file__), receipt_sha=r.base.sha(receipt_path),
        producer_receipt_sha=receipt['producer_receipt_sha'], actual_epochs=count,
        arm=config['arm'], all_prediction_files_rehashed=count,
        point_recomputed_from_actual_predictions=True, point_rows=count, cell_rows=6 * count,
        max_cell_error=cell_error, max_point_error=point_error,
        eligible_windows=[int(w[0]) + 1 for _, w in expected],
        needed_epochs=needed, all1000_bootstrap_upper975=all_upper,
        confirmed_overfitting=confirmed, confirmed_overfitting_onset=first,
        gain=gain, plateau=gain <= .005, extension_required=gain > .005 and not confirmed,
        actual_full_cache_oracle_sha=r.base.sha(r.OUT / 'selection_budget_sparse_actual_full_cache_oracle.json'),
        raw_rows=len(raw_rows), raw_archive=str(raw_path), raw_archive_sha=r.base.sha(raw_path),
        scope='Budget-only confirmation; full final paired and all mandatory prefix readouts remain required.', test_access=False)
    certificate = r.OUT / (name + '_primary_verification.json')
    assert not certificate.exists()
    r.base.dump(certificate, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
