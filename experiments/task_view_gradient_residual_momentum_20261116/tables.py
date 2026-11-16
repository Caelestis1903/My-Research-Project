"""Raw validation exports and preregistered five-seed paired arithmetic."""
import gzip
import json
from pathlib import Path
import numpy as np
from .protocol import SEEDS, DATASETS, ARMS, results, epoch_files
from experiments.auxiliary_objective_attribution_20261115.stage2.tables import write_csv, paired


def export(attempt):
    attempt = Path(attempt); records = results(attempt)
    assert len(records) == 70 and all(r['epochs'] == 50 and not r['test_accessed'] for r in records)
    by_key = {(r['job']['dataset'], r['job']['config']['arm'], r['job']['seed']): r for r in records}
    assert len(by_key) == 70
    assert set(by_key) == {(d, a, s) for d in DATASETS for a in ('CLEAN', 'NDP-A')+ARMS for s in SEEDS}
    endpoints = []; curves = []; costs = []; weights = []; diagnostics = []; order_hashes = {}
    for result in records:
        j = result['job']; arm = j['config']['arm']; reused = arm in ('CLEAN', 'NDP-A')
        base = dict(dataset=j['dataset'], arm=arm, seed=j['seed'], key=j['key'])
        epochs = epoch_files(attempt, result); assert set(epochs) == set(range(1, 51))
        expected = 1000 if j['dataset'] == 'movielens' else 2450
        assert result['global_step'] == result['counts']['optimizer_updates'] == result['counts']['ema_updates'] == expected
        for epoch, (_, data) in sorted(epochs.items()):
            key = (j['dataset'], j['seed'], epoch)
            if key in order_hashes:
                assert order_hashes[key] == data['epoch_order_hash'], 'Natural loader order mismatch'
            order_hashes[key] = data['epoch_order_hash']
            for kind, values in data['validation'].items():
                curves.append(dict(**base, kind=kind, epoch=epoch, **values))
            for cell, value in data.get('objective_ledger', {}).get('cells', {}).items():
                fields = value if isinstance(value, dict) else dict(clean_count=value, view_count=value)
                weights.append(dict(**base, epoch=epoch, cell_label=cell,
                    batch_count=data['objective_ledger']['batches'], **fields))
            if not reused:
                path = attempt/data['support_log']
                if epoch in (1, 10, 25, 40, 50):
                    with gzip.open(path, 'rt') as stream:
                        first = json.loads(next(stream))
                    diagnostic = first.pop('gradient_diagnostics', {})
                    diagnostics.append(dict(**base, epoch=epoch,
                        **{k: v for k, v in first.items() if k != 'epoch' and (isinstance(v, (int, float, bool)) or v is None)},
                        **{k: v for k, v in diagnostic.items() if k != 'parameter_groups'},
                        parameter_groups=json.dumps(diagnostic.get('parameter_groups', {}), sort_keys=True)))
        for kind, values in result['selected_validation'].items():
            selected_epoch = max(sorted(epochs), key=lambda e: epochs[e][1]['validation'][kind]['macro_auc'])
            assert selected_epoch == result['selected_epochs'][kind]
            data = epochs[selected_epoch][1]
            assert data['validation'][kind] == values
            assert data['model_hash' if kind == 'online' else 'ema_hash'] == result['selected_hashes'][kind]
            endpoints.append(dict(**base, split='validation', endpoint='selected', kind=kind,
                epoch=selected_epoch, model_hash=result['selected_hashes'][kind], **values))
        for epoch in (10, 30, 50):
            for kind, values in epochs[epoch][1]['validation'].items():
                endpoints.append(dict(**base, split='validation', endpoint=f'E{epoch}', kind=kind, epoch=epoch, **values))
        costs.append(dict(**base, epochs=50, steps=result['global_step'], worker_seconds=result['worker_seconds'],
            peak_allocated_bytes=result['peak_allocated_bytes'], peak_reserved_bytes=result['peak_reserved_bytes'],
            data_load_seconds=result.get('data_load_seconds'), reused=reused))
    groups = {}
    for row in endpoints:
        groups.setdefault(tuple(row[k] for k in ('dataset', 'arm', 'endpoint', 'kind')), {})[row['seed']] = row
    metrics = sorted({k for row in endpoints for k in row if k.endswith(('_auc', '_logloss'))})
    summaries = []; seed_differences = []; differences = []
    for key, values in sorted(groups.items()):
        assert set(values) == set(SEEDS)
        base = dict(zip(('dataset', 'arm', 'endpoint', 'kind'), key))
        for metric in metrics:
            v = [values[s][metric] for s in SEEDS]
            if None not in v:
                summaries.append(dict(**base, metric=metric, mean=float(np.mean(v)), sample_sd=float(np.std(v, ddof=1)), n=5))
    contrasts = [(a, c) for a in ARMS for c in ('CLEAN', 'NDP-A')]
    contrasts += [('TV-GRM-010', 'TV-GRM-SHUFFLE-010'), ('TV-GRM-010', 'TV-GRM-INSTANT-010')]
    for dataset in DATASETS:
        for endpoint in ('selected', 'E10', 'E30', 'E50'):
            for arm, control in contrasts:
                kinds = [('online', 'online'), ('fixed_ema', 'fixed_ema')]
                if control == 'CLEAN':
                    kinds.append(('fixed_ema', 'online'))
                for kind, reference_kind in kinds:
                    a = groups[dataset, arm, endpoint, kind]; b = groups[dataset, control, endpoint, reference_kind]
                    base = dict(dataset=dataset, arm=arm, control=control, endpoint=endpoint,
                                kind=kind, reference_kind=reference_kind)
                    for metric in metrics:
                        if any(a[s][metric] is None or b[s][metric] is None for s in SEEDS):
                            continue
                        deltas = [a[s][metric]-b[s][metric] for s in SEEDS]
                        differences.append(dict(**base, metric=metric, **paired(deltas)))
                        seed_differences.extend(dict(**base, metric=metric, seed=s, delta=v)
                                                for s, v in zip(SEEDS, deltas))
    ranking = []
    for arm in ARMS[:3]:
        means = {d: float(np.mean([groups[d, arm, 'selected', 'fixed_ema'][s]['macro_auc']-
            groups[d, 'CLEAN', 'selected', 'fixed_ema'][s]['macro_auc'] for s in SEEDS])) for d in DATASETS}
        ranking.append(dict(arm=arm, gamma=int(arm[-3:])/100, **means,
                            retention_score=float(np.mean(list(means.values())))))
    ranking.sort(key=lambda row: (-row['retention_score'], row['gamma']))
    directory = attempt/'tables'
    for name, rows in (('ENDPOINTS', endpoints), ('VALIDATION_CURVES', curves), ('COSTS', costs), ('WEIGHTS', weights),
        ('SUMMARY', summaries), ('PAIRED_SEEDS', seed_differences), ('PAIRED_SUMMARY', differences),
        ('GRADIENT_DIAGNOSTICS', diagnostics), ('CANDIDATE_RANKING', ranking)):
        write_csv(directory/f'{name}.csv', rows)
    return dict(status='COMPLETE', trajectories=70, new_trajectories=50, endpoints=len(endpoints),
                directory=str(directory), test_accessed=False)
