"""Registered Cycle02 validation analysis; run only after all 30 trajectories finish."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ATTEMPT = ROOT / 'artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02/attempt1'
TABLES = ATTEMPT / 'tables'
ARMS = ('NQR-025', 'BRIER-OPPORTUNITY-025', 'MEAN-GAP-025', 'CLEAN')
DATASETS = ('movielens', 'kuairand')
SEEDS = tuple(range(2022, 2027))
KINDS = ('online', 'fixed_ema')
ENDPOINTS = ('selected', 'E10', 'E30', 'E50')
CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))
TRAIN_COUNTS = {'movielens': [155949, 318599, 325619], 'kuairand': [172782, 1743898, 89452]}


def read_csv(name):
    with (TABLES / name).open(newline='') as f:
        return list(csv.DictReader(f))


def dump_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    assert len(values) == 5 and all(math.isfinite(x) for x in values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = 2.7764451051977987 * sd / math.sqrt(5)
    return dict(mean=mean, sd=sd, ci95_low=mean-half, ci95_high=mean+half,
                positive=sum(x > 0 for x in values), negative=sum(x < 0 for x in values),
                zero=sum(x == 0 for x in values), n=5)


def main():
    receipt = json.loads((TABLES / 'EXPORT_RECEIPT.json').read_text())
    assert receipt['new_trajectories'] == 30 and receipt['reused_clean'] == 10
    assert receipt['useful_batches'] == 51750 and receipt['test_accessed'] is False
    endpoints = read_csv('ENDPOINTS.csv')
    assert len(endpoints) == 320
    index = {}
    for original in endpoints:
        row = dict(original)
        for name in list(row):
            if name.endswith(('_auc', '_logloss')):
                row[name] = float(row[name])
        row['epoch'] = int(row['epoch'])
        row['seed'] = int(row['seed'])
        key = row['dataset'], row['arm'], row['seed'], row['endpoint'], row['estimator']
        assert key not in index
        index[key] = row
    means = []
    for dataset in DATASETS:
        for arm in ARMS:
            for kind in KINDS:
                for endpoint in ENDPOINTS:
                    rows = [index[dataset, arm, seed, endpoint, kind] for seed in SEEDS]
                    metrics = [k for k in rows[0] if k.endswith(('_auc', '_logloss'))] + ['epoch']
                    for metric in metrics:
                        means.append(dict(dataset=dataset, arm=arm, estimator=kind, endpoint=endpoint,
                                          metric=metric, **describe([r[metric] for r in rows])))
    selected_pairs = [
        (ARMS[0], 'fixed_ema', 'CLEAN', 'fixed_ema'),
        (ARMS[0], 'fixed_ema', 'CLEAN', 'online'),
        (ARMS[0], 'online', 'CLEAN', 'online'),
        (ARMS[0], 'fixed_ema', ARMS[1], 'fixed_ema'),
        (ARMS[0], 'fixed_ema', ARMS[2], 'fixed_ema'),
        (ARMS[1], 'fixed_ema', 'CLEAN', 'fixed_ema'),
        (ARMS[2], 'fixed_ema', 'CLEAN', 'fixed_ema'),
    ]
    common_pairs = [(a, k, 'CLEAN', k) for a in ARMS[:3] for k in KINDS]
    common_pairs += [(ARMS[0], k, a, k) for a in ARMS[1:3] for k in KINDS]
    pairs, summaries = [], []
    for dataset in DATASETS:
        for endpoint in ENDPOINTS:
            for left, lk, right, rk in selected_pairs if endpoint == 'selected' else common_pairs:
                comparison = f'{left}/{lk} minus {right}/{rk}'
                paired = [(index[dataset, left, seed, endpoint, lk], index[dataset, right, seed, endpoint, rk]) for seed in SEEDS]
                metrics = [k for k in paired[0][0] if k.endswith(('_auc', '_logloss'))] + ['epoch']
                for metric in metrics:
                    values = []
                    for (l, r), seed in zip(paired, SEEDS):
                        delta = l[metric] - r[metric]
                        values.append(delta)
                        pairs.append(dict(dataset=dataset, comparison=comparison, endpoint=endpoint, metric=metric,
                                          seed=seed, left=l[metric], right=r[metric], delta=delta,
                                          left_epoch=l['epoch'], right_epoch=r['epoch'],
                                          left_state_hash=l['state_hash'], right_state_hash=r['state_hash']))
                    summaries.append(dict(dataset=dataset, comparison=comparison, endpoint=endpoint,
                                          metric=metric, **describe(values)))
    lookup = {(r['dataset'], r['comparison'], r['endpoint'], r['metric']): r for r in summaries}
    primary_comparison = f'{ARMS[0]}/fixed_ema minus CLEAN/fixed_ema'
    primary = {}
    for dataset in DATASETS:
        auc = lookup[dataset, primary_comparison, 'selected', 'macro_auc']
        ll = lookup[dataset, primary_comparison, 'selected', 'macro_logloss']
        cells = {c: lookup[dataset, primary_comparison, 'selected', c+'_auc'] for c in CELLS}
        min_cell = min(r['mean'] for r in cells.values())
        primary[dataset] = dict(auc=auc, logloss=ll, cell_auc=cells, min_cell_mean_delta=min_cell,
                                macro_target_met=auc['mean'] >= .003,
                                cell_protection_met=min_cell >= -.001,
                                auc_target_and_cell_protection_met=auc['mean'] >= .003 and min_cell >= -.001,
                                comprehensive_loss_decrease=ll['mean'] < 0)
    positive_datasets = sum(primary[d]['auc']['mean'] > 0 for d in DATASETS)
    if positive_datasets == 0:
        outcome = 'NO_CONTINUATION_SIGNAL'
    elif positive_datasets == 1:
        outcome = 'MIXED_DATASET_SIGNAL'
    elif all(primary[d]['auc_target_and_cell_protection_met'] for d in DATASETS):
        outcome = 'VALIDATION_SCREEN_AUC_TARGET_AND_CELL_PROTECTION_MET_BOTH_DATASETS'
    else:
        outcome = 'POSITIVE_VALIDATION_SIGNAL_BELOW_TARGET'
    diagnostics = read_csv('DIAGNOSTICS.csv')
    diagnostic_means = []
    groups = {}
    for row in diagnostics:
        if row['parameter_group'] != 'ALL':
            continue
        key = row['dataset'], row['arm'], int(row['epoch'])
        groups.setdefault(key, []).append(row)
    for (dataset, arm, epoch), rows in sorted(groups.items()):
        assert sorted(int(r['seed']) for r in rows) == list(SEEDS)
        for metric in ('data_gradient_norm', 'parameter_update_norm'):
            diagnostic_means.append(dict(dataset=dataset, arm=arm, epoch=epoch, metric=metric,
                                         **describe([float(r[metric]) for r in rows])))
    stat_groups = {}
    prevalence = {}
    for row in read_csv('STATISTICS.csv'):
        epoch = int(row['epoch'])
        if row['statistic'] == 'Q_R' and epoch == 1:
            key = row['dataset'], row['cell']
            counts = tuple(int(row[k]) for k in ('n', 'n0', 'n1'))
            if key in prevalence:
                assert prevalence[key] == counts
            prevalence[key] = counts
        if epoch not in (1, 10, 25, 40, 50):
            continue
        count = float(row['observed_count'])
        value = float(row['count_weighted_sum']) / count if count else None
        stat_groups.setdefault((row['dataset'], row['arm'], epoch, row['cell'], row['statistic']), []).append(value)
    statistic_means = []
    for (dataset, arm, epoch, cell, metric), values in sorted(stat_groups.items()):
        assert len(values) == 5
        valid = [v for v in values if v is not None]
        statistic_means.append(dict(dataset=dataset, arm=arm, epoch=epoch, cell=cell, statistic=metric,
                                    valid_seeds=len(valid), mean=statistics.mean(valid) if valid else None,
                                    sd=statistics.stdev(valid) if len(valid)>1 else None,
                                    aggregation='equal_seed_mean_of_class_or_cell_count_weighted_batch_statistics'))
    costs = read_csv('COSTS.csv')
    physical = read_csv('PHYSICAL_ATTEMPTS.csv')
    cost_means = []
    for dataset in DATASETS:
        for arm in ARMS:
            rows = [r for r in costs if r['dataset'] == dataset and r['arm'] == arm]
            assert len(rows) == 5
            for metric in ('physical_worker_seconds', 'training_seconds', 'evaluation_seconds', 'diagnostic_seconds',
                           'serialization_seconds', 'transfer_seconds', 'peak_full_process_bytes'):
                values = [float(r[metric]) for r in rows if r.get(metric) not in ('', None)]
                cost_means.append(dict(dataset=dataset, arm=arm, metric=metric, measured_seeds=len(values),
                                       mean=statistics.mean(values) if values else None,
                                       sd=statistics.stdev(values) if len(values)>1 else None))
    dump_csv('ENDPOINT_MEANS.csv', means)
    dump_csv('ALL_PAIRED_SEEDS.csv', pairs)
    dump_csv('ALL_PAIRED_SUMMARY.csv', summaries)
    dump_csv('DIAGNOSTIC_MEANS.csv', diagnostic_means)
    dump_csv('STATISTIC_MEANS.csv', statistic_means)
    dump_csv('COST_MEANS.csv', cost_means)
    appendix = ['# Cycle02 numerical appendix', '',
                'Natural validation populations; seeds 2022–2026. Each estimator selects its earliest strict maximum independently. Selected LogLoss uses that same state.', '']
    for dataset in DATASETS:
        for endpoint in ENDPOINTS:
            for kind in KINDS:
                appendix += [f'## {dataset} / {endpoint} / {kind}', '',
                             '| Arm | Seed | Epoch | Macro AUC | Macro LogLoss | State SHA256 |',
                             '|---|---:|---:|---:|---:|---|']
                for arm in ARMS:
                    for seed in SEEDS:
                        r = index[dataset, arm, seed, endpoint, kind]
                        appendix.append(f'| {arm} | {seed} | {r["epoch"]} | {r["macro_auc"]:.12f} | {r["macro_logloss"]:.12f} | {r["state_hash"]} |')
                appendix.append('')
        for kind in KINDS:
            for arm in ARMS:
                appendix += [f'## {dataset} / selected / {kind} / {arm} — cells', '',
                             '| Seed | ' + ' | '.join(f'{c} {m}' for c in CELLS for m in ('AUC', 'LL')) + ' |',
                             '|---:|' + '---:|'*12]
                for seed in SEEDS:
                    r = index[dataset, arm, seed, 'selected', kind]
                    appendix.append('| '+str(seed)+' | '+' | '.join(f'{r[c+"_"+m]:.12f}' for c in CELLS for m in ('auc','logloss'))+' |')
                appendix.append('')
    (OUT/'NUMERICAL_APPENDIX.md').write_text('\n'.join(appendix)+'\n')
    facts = dict(candidate='NQR-025', decision=outcome, primary=primary, new_trajectories=30,
                 reused_clean=10, new_epochs=1500, new_useful_updates=51750,
                 train_domain_counts=TRAIN_COUNTS,
                 train_cell_counts={f'{d}/{c}': dict(n=n, n0=n0, n1=n1, prevalence=n1/n) for (d,c),(n,n0,n1) in prevalence.items()},
                 physical_attempts=len(physical), physical_status_counts={s:sum(r['status']==s for r in physical) for s in set(r['status'] for r in physical)},
                 total_measured_physical_worker_seconds=sum(float(r['seconds']) for r in physical),
                 test_accessed=False, primary_estimator='independently_selected_fixed_ema',
                 source_hash=receipt['source_hash'], design_bundle_sha256=receipt['design_bundle_sha256'],
                 export_receipt_sha256=sha(TABLES/'EXPORT_RECEIPT.json'),
                 inputs={name:sha(TABLES/name) for name in ('ENDPOINTS.csv','DIAGNOSTICS.csv','STATISTICS.csv','COSTS.csv','PHYSICAL_ATTEMPTS.csv')})
    (OUT/'DECISION.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(decision=outcome, primary=primary, paired_seed_rows=len(pairs),
                          paired_summary_rows=len(summaries), test_accessed=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
