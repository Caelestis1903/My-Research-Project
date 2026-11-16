"""Raw endpoints, complete cell metrics and registered paired-seed arithmetic only."""
import gzip
import json
from pathlib import Path

import numpy as np
from .protocol import (ARMS, DATASETS, SEEDS, BUNDLE, cycle_path, results, epoch_files,
    validate_attempt, assert_old_unchanged, sha_file, write_json)
from experiments.auxiliary_objective_attribution_20261115.stage2.tables import write_csv, paired


def paired_stats(values):
    # Reuse the historical helper and apply the registered full-precision t multiplier.
    out = paired(values); values = np.asarray(values, dtype=float)
    radius = 2.7764451051977987*out['sample_sd']/np.sqrt(5)
    out.update(ci95_low=out['mean']-radius, ci95_high=out['mean']+radius,
        negative_seeds=int((values < 0).sum()), zero_seeds=int((values == 0).sum()), n=5,
        statistical_unit='seed', interval='descriptive_95_percent')
    return out


def export(attempt):
    attempt = cycle_path(attempt); manifest = validate_attempt(attempt, require_smoke=True); records = results(attempt)
    expected = {(d, a, s) for d in DATASETS for a in ('CLEAN', 'NDP-A')+ARMS for s in SEEDS}
    by_key = {(r['job']['dataset'], r['job']['config']['arm'], r['job']['seed']): r for r in records}
    assert len(records) == len(by_key) == 50 and set(by_key) == expected
    endpoints = []; curves = []; cells = []; costs = []; attempts = []; weights = []; diagnostics = []; order_hashes = {}
    for result in records:
        j = result['job']; arm = j['config']['arm']; reused = arm in ('CLEAN', 'NDP-A')
        base = dict(dataset=j['dataset'], arm=arm, seed=j['seed'], key=j['key'])
        assert result['epochs'] == 50 and result['test_accessed'] is False
        steps = 1000 if j['dataset'] == 'movielens' else 2450
        assert result['global_step'] == result['counts']['optimizer_updates'] == result['counts']['ema_updates'] == steps
        epochs = epoch_files(attempt, result); assert set(epochs) == set(range(1, 51))
        for epoch, (_, data) in sorted(epochs.items()):
            order_key = (j['dataset'], j['seed'], epoch)
            if order_key in order_hashes:
                assert order_hashes[order_key] == data['epoch_order_hash'], 'Cross-arm loader identity failed'
            order_hashes[order_key] = data['epoch_order_hash']
            for kind, value in data['validation'].items():
                curves.append(dict(**base, kind=kind, epoch=epoch, **value))
                for domain in range(3):
                    for task in range(2):
                        prefix = f'd{domain}t{task}_'
                        fields = {k[len(prefix):]: v for k, v in value.items() if k.startswith(prefix)}
                        fields.setdefault('negative', fields['count']-fields['positive'])
                        cells.append(dict(**base, kind=kind, epoch=epoch, domain=domain, task=task, **fields))
            for cell, value in data.get('objective_ledger', {}).get('cells', {}).items():
                weights.append(dict(**base, epoch=epoch, cell_label=cell,
                    batch_count=data['objective_ledger']['batches'], **value))
            if not reused:
                path = attempt/data['support_log']; assert sha_file(path) == data['support_log_sha256']
                if epoch in (1, 10, 25, 40, 50):
                    with gzip.open(path, 'rt') as stream:
                        first = json.loads(next(stream))
                    diagnostics.append(dict(**base, epoch=epoch, batch=1, row_order_hash=first['row_order_hash'],
                        loss=first['loss'], clean_loss=first['clean_loss'], view_loss=first['view_loss'],
                        replica_losses=json.dumps(first['replica_losses'], sort_keys=True),
                        diagnostics=json.dumps(first['diagnostics'], sort_keys=True)))
        for kind, value in result['selected_validation'].items():
            selected = max(sorted(epochs), key=lambda e: epochs[e][1]['validation'][kind]['macro_auc'])
            assert selected == result['selected_epochs'][kind]
            data = epochs[selected][1]; assert data['validation'][kind] == value
            assert data['model_hash' if kind == 'online' else 'ema_hash'] == result['selected_hashes'][kind]
            endpoints.append(dict(**base, split='validation', endpoint='selected', kind=kind, epoch=selected,
                model_hash=result['selected_hashes'][kind], **value))
        for epoch in (10, 30, 50):
            for kind, value in epochs[epoch][1]['validation'].items():
                endpoints.append(dict(**base, split='validation', endpoint=f'E{epoch}', kind=kind, epoch=epoch, **value))
        physical = None; missing = []
        if not reused:
            physical = 0.
            for directory in sorted((attempt/'runs'/j['key']).glob('attempt*')):
                path = directory/'worker_cost.json'
                if path.exists():
                    value = json.loads(path.read_text()); physical += value['seconds']
                    attempts.append(dict(**base, attempt=value['attempt'], status=value['status'], seconds=value['seconds']))
                else:
                    missing.append(directory.name)
        costs.append(dict(**base, reused=reused, epochs=50, steps=steps, worker_seconds=result['worker_seconds'],
            measured_attempt_worker_seconds=physical, missing_terminal_costs=json.dumps(missing),
            peak_allocated_bytes=result['peak_allocated_bytes'], peak_reserved_bytes=result['peak_reserved_bytes'],
            peak_host_rss_bytes=result.get('peak_host_rss_bytes'), data_load_seconds=result.get('data_load_seconds')))
    groups = {}
    for row in endpoints:
        groups.setdefault(tuple(row[k] for k in ('dataset', 'arm', 'endpoint', 'kind')), {})[row['seed']] = row
    metrics = sorted({k for row in endpoints for k in row if k.endswith(('_auc', '_logloss'))})
    summaries = []; seed_deltas = []; differences = []
    for key, values in sorted(groups.items()):
        assert set(values) == set(SEEDS); base = dict(zip(('dataset', 'arm', 'endpoint', 'kind'), key))
        for metric in metrics:
            numbers = [values[s][metric] for s in SEEDS]
            if None not in numbers:
                summaries.append(dict(**base, metric=metric, mean=float(np.mean(numbers)),
                    sample_sd=float(np.std(numbers, ddof=1)), n=5))
    for dataset in DATASETS:
        for endpoint in ('selected', 'E10', 'E30', 'E50'):
            for control in ('CLEAN', 'NDP-A', 'ISOTROPIC-EVEN', 'AXIS-IID2'):
                for kind in ('online', 'fixed_ema'):
                    a = groups[dataset, 'AXIS-EVEN', endpoint, kind]; b = groups[dataset, control, endpoint, kind]
                    base = dict(dataset=dataset, arm='AXIS-EVEN', control=control, endpoint=endpoint,
                        kind=kind, reference_kind=kind)
                    for metric in metrics:
                        if any(a[s][metric] is None or b[s][metric] is None for s in SEEDS):
                            continue
                        deltas = [a[s][metric]-b[s][metric] for s in SEEDS]
                        differences.append(dict(**base, metric=metric, **paired_stats(deltas)))
                        seed_deltas.extend(dict(**base, metric=metric, seed=s, candidate=a[s][metric],
                            reference=b[s][metric], delta=delta) for s, delta in zip(SEEDS, deltas))
    directory = attempt/'tables'
    for name, rows in (('ENDPOINTS', endpoints), ('VALIDATION_CURVES', curves), ('CELL_METRICS', cells),
        ('COSTS', costs), ('PHYSICAL_ATTEMPTS', attempts), ('WEIGHTS', weights), ('SUMMARY', summaries), ('PAIRED_SEEDS', seed_deltas),
        ('PAIRED_SUMMARY', differences), ('GRADIENT_DIAGNOSTICS', diagnostics)):
        write_csv(directory/f'{name}.csv', rows)
    report = dict(status='COMPLETE', trajectories=50, new_trajectories=30, reused_trajectories=20,
        new_optimizer_updates=sum(r['global_step'] for r in records if r['job']['config']['arm'] in ARMS),
        measured_attempt_worker_seconds=sum(r['seconds'] for r in attempts),
        endpoints=len(endpoints), directory=str(directory), source_hash=manifest['source_hash'],
        design_bundle_sha256=BUNDLE, old_inventories=assert_old_unchanged(attempt), test_accessed=False)
    assert report['new_optimizer_updates'] == 51750
    write_json(attempt/'TABLES_RECEIPT.json', report)
    complete_path = attempt/'COMPLETE_RUNNER_REPORT.json'
    complete = json.loads(complete_path.read_text())
    complete.update(tables_receipt=report, physical_attempt_costs=attempts,
                    physical_cost_missing=[dict(key=r['key'], attempts=json.loads(r['missing_terminal_costs']))
                                           for r in costs if json.loads(r['missing_terminal_costs'])])
    write_json(complete_path, complete)
    return report
