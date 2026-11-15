"""Scalar exports only; scientific interpretation belongs to the orchestrator."""
import csv
import json
from pathlib import Path
import numpy as np
from .selection import paired
from .protocol import SEEDS


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def export(attempt, records):
    attempt = Path(attempt); rows = []; costs = []
    for result in records:
        job = result['job']; c = job['config']
        base = dict(dataset=job['dataset'], arm=c['arm'], alpha=c['alpha'], seed=job['seed'], key=job['key'])
        for kind, values in result['selected_validation'].items():
            rows.append(dict(**base, split='validation', endpoint='selected', kind=kind,
                epoch=result['selected_epochs'][kind], model_hash=result['selected_hashes'][kind], **values))
        for epoch, estimators in result['common_validation'].items():
            for kind, values in estimators.items():
                rows.append(dict(**base, split='validation', endpoint='E'+epoch, kind=kind, epoch=int(epoch), **values))
        costs.append(dict(**base, epochs=result['epochs'], steps=result['global_step'],
            worker_seconds=result['worker_seconds'], peak_allocated_bytes=result['peak_allocated_bytes'],
            peak_reserved_bytes=result['peak_reserved_bytes']))
    for path in sorted((attempt/'test_endpoints').glob('*.json')):
        result = json.loads(path.read_text()); job = result['job']; c = job['config']
        rows.append(dict(dataset=job['dataset'], arm=c['arm'], alpha=c['alpha'], seed=job['seed'], key=job['key'],
            split='test', endpoint='selected', kind=result['kind'], epoch=result['epoch'], model_hash=result['model_hash'],
            **result['metrics']))
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ('dataset','arm','alpha','split','endpoint','kind'))
        groups.setdefault(key, {})[row['seed']] = row
    summaries = []; differences = []; seed_differences = []
    metric_names = sorted({k for row in rows for k in row if k.endswith(('_auc','_logloss'))})
    for key, seeds in sorted(groups.items()):
        if set(seeds) != set(SEEDS):
            continue
        base = dict(zip(('dataset','arm','alpha','split','endpoint','kind'),key))
        for metric in metric_names:
            values = [seeds[s].get(metric) for s in SEEDS]
            if any(v is None for v in values):
                continue
            summaries.append(dict(**base, metric=metric, n=5, mean=float(np.mean(values)), sample_sd=float(np.std(values,ddof=1))))
        dataset, arm, alpha, split, endpoint, kind = key
        if endpoint != 'selected':
            continue
        for control in ('CLEAN','A','B','C','RANDOM','GAUSSIAN'):
            reference_kinds = [kind] if control != arm else []
            if kind == 'fixed_ema' and control in ('CLEAN', arm):
                reference_kinds.append('online')
            for reference_kind in reference_kinds:
                reference_key = (dataset,control,.03 if control == 'CLEAN' else alpha,split,endpoint,reference_kind)
                reference = groups.get(reference_key)
                if reference is None or set(reference) != set(SEEDS):
                    continue
                comparison = dict(control=control, reference_kind=reference_kind)
                for metric in metric_names:
                    if any(seeds[s].get(metric) is None or reference[s].get(metric) is None for s in SEEDS):
                        continue
                    stat = paired([seeds[s][metric]-reference[s][metric] for s in SEEDS])
                    differences.append(dict(**base, **comparison, metric=metric, mean=stat['mean'],
                        sample_sd=stat['sample_sd'], ci95_low=stat['ci95'][0],ci95_high=stat['ci95'][1],
                        positive_seeds=stat['positive_seeds']))
                    for seed, value in zip(SEEDS, stat['values']):
                        seed_differences.append(dict(**base, **comparison, metric=metric, seed=seed, delta=value))
    directory = attempt/'tables'
    for name, values in (('ENDPOINTS',rows),('COSTS',costs),('SUMMARY',summaries),
                         ('PAIRED_SUMMARY',differences),('PAIRED_SEEDS',seed_differences)):
        write_csv(directory/f'{name}.csv', values)
    return dict(status='COMPLETE', endpoints=len(rows), completed_trajectories=len(records), directory=str(directory))
