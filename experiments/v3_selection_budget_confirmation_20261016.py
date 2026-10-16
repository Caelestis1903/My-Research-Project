"""Exact frozen selection-overfit confirmation, bootstrap only needed epochs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import v3_readout_20261016 as r


def eligible_windows(point, objective):
    windows = []
    for start in range(1, len(point) - 2):
        peak = int(np.argmax(point[:start, 0]))
        window = np.arange(start, start + 3)
        if (np.all(point[window, 0] < point[peak, 0]) and
                np.all(point[window, 1] > point[peak, 1]) and
                objective[start + 2] < objective[peak]):
            windows.append(dict(start=start, peak=peak, window=window.tolist()))
    return windows


def confirm(windows, bootstrap, epoch_indices):
    lookup = {int(epoch): i for i, epoch in enumerate(epoch_indices)}
    records = []
    for item in windows:
        peak = lookup[item['peak']]
        window = [lookup[x] for x in item['window']]
        upper = np.quantile(bootstrap[:, window, 0] - bootstrap[:, peak, None, 0], .975, axis=0)
        records.append(dict(start_epoch=item['start'] + 1, peak_epoch=item['peak'] + 1,
            window_epochs=[x + 1 for x in item['window']], upper975=upper.tolist(),
            confirmed=bool(np.all(upper < 0))))
    first = next((x for x in records if x['confirmed']), None)
    return dict(confirmed_overfitting=first is not None,
        confirmed_overfitting_onset=None if first is None else first['start_epoch'],
        confirmation_peak_epoch=None if first is None else first['peak_epoch']), records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert r.base.sha(r.__file__) == '655e284c7c32052c2aea951ccf6005befc2dc8561c302736e775a9007c0dfe7c'
    freeze = r.OUT / 'analysis_rules_frozen.json'
    assert json.loads(freeze.read_text())['status'] == 'FROZEN'
    producer_receipt_sha = r.base.sha(args.run / 'receipt.json')
    started = time.time()
    producer = json.loads((args.run / 'receipt.json').read_text())
    config = json.loads((args.run / 'config.json').read_text())
    assert producer['status'] == 'COMPLETE_TRAINING_TEST_DEFERRED'
    for name in ('config.json', 'trajectory.csv', 'metrics_selection.csv'):
        assert r.base.sha(args.run / name) == producer['artifact_hashes'][name]
    split_path = r.OUT / 'splits' / f'{config["dataset"]}_seed{config["seed"]}.npz'
    assert r.base.sha(split_path) == config['split_sha256']
    bundle = r.audit.bundle(config['dataset'], config['seed'])
    split = np.load(split_path)['selection']
    domains = bundle.val_x[:, -1].astype(int)
    training = pd.read_csv(args.run / 'trajectory.csv')
    epochs = list(range(1, producer['epochs'] + 1))
    assert training.epoch.tolist() == epochs and len(epochs) >= 50
    point, cells, prediction_hashes = [], [], {}
    for epoch in epochs:
        path = args.run / f'validation_epoch{epoch:03d}.npz'
        digest = r.base.sha(path)
        assert digest == producer['artifact_hashes'][path.name]
        prediction_hashes[path.name] = digest
        prediction = np.load(path)['predictions']
        metric = r.train.metrics(bundle, prediction, 'val', split)
        point.append([np.mean([x[name] for x in metric]) for name in ('auc', 'logloss')])
        cells.extend(dict(x, epoch=epoch) for x in metric)
    point = np.asarray(point)
    raw_cells = pd.read_csv(args.run / 'metrics_selection.csv')
    merged = pd.DataFrame(cells).merge(raw_cells, on=['epoch', 'domain', 'task'],
                                      suffixes=('_recomputed', '_producer'), validate='one_to_one')
    assert len(merged) == len(epochs) * 6
    point_error = max(float(abs(merged[m + '_recomputed'] - merged[m + '_producer']).max())
                      for m in ('auc', 'logloss'))
    assert point_error < 1e-12
    objective = training.train_empirical_objective.to_numpy()
    windows = eligible_windows(point, objective)
    needed = sorted({i for w in windows for i in [w['peak'], *w['window']]})
    args.out.mkdir(parents=True, exist_ok=False)
    if needed:
        predictions = np.asarray([[np.load(args.run / f'validation_epoch{i+1:03d}.npz')['predictions'] for i in needed]])
        bootstrap = r.bootstrap_curves(bundle.val_y, domains, predictions, split,
            1000, config['seed'] + 20261016, args.out / 'progress.json')[:, 0]
        result, checks = confirm(windows, bootstrap, needed)
        np.savez_compressed(args.out / 'selection_bootstrap_needed_epochs.npz',
                            selection=bootstrap, epochs=np.asarray(needed) + 1)
    else:
        result, checks = confirm([], np.empty((1000, 0, 2)), [])
    pd.DataFrame(dict(epoch=epochs, auc=point[:, 0], logloss=point[:, 1])).to_csv(
        args.out / 'selection_point_all_epochs.csv', index=False)
    pd.DataFrame(cells).to_csv(args.out / 'selection_point_cells.csv', index=False)
    r.base.dump(args.out / 'candidate_windows.json', checks)
    assert r.base.sha(args.run / 'receipt.json') == producer_receipt_sha
    r.base.dump(args.out / 'receipt.json', dict(status='COMPLETE',
        mode='EXACT_SELECTION_ONLY_BUDGET_CONFIRMATION', **result,
        source_sha=r.base.sha(__file__), original_readout_sha=r.base.sha(r.__file__),
        analysis_rules_sha=r.base.sha(freeze), source_run=str(args.run),
        producer_receipt_sha=producer_receipt_sha,
        design_sha=r.base.sha(r.OUT / 'selection_budget_sparse_confirmation_design.json'),
        elapsed_seconds=time.time() - started,
        actual_epochs=len(epochs), seed=config['seed'], arm=config['arm'],
        all_epoch_prediction_hashes=prediction_hashes,
        point_recomputed_from_ALL_actual_epochs=True, max_producer_cell_error=point_error,
        eligible_windows=len(windows), needed_epochs=[i + 1 for i in needed],
        bootstrap_count=1000 if needed else 0,
        omitted_epoch_bootstraps_not_fabricated=True,
        first_auc_degradation_NOT_computed=True, diagnostic_access=False,
        test_performance_read=False, independent_review=False,
        hashes={p.name: r.base.sha(p) for p in args.out.iterdir() if p.is_file()},
        scope='Only exact frozen selection budget confirmation; final full paired1000/200 readouts remain required.'))


if __name__ == '__main__':
    main()
