"""Parent scientific arithmetic for the registered Cycle03 comparisons.

Run only after all twenty new trajectories and their numeric export finish.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ATTEMPT = ROOT / 'artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/attempt1'
TABLES = ATTEMPT / 'tables'
ARMS = ('NQR-EARLY10', 'NQR-LATE10', 'NQR-025', 'CLEAN')
DATASETS = ('movielens', 'kuairand')
SEEDS = tuple(range(2022, 2027))
KINDS = ('online', 'fixed_ema')
ENDPOINTS = ('selected', 'E10', 'E30', 'E40', 'E50')
CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))
CONTRASTS = ((ARMS[0], 'CLEAN'), (ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]),
             (ARMS[1], 'CLEAN'), (ARMS[1], ARMS[2]))
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


def describe(values, metric):
    assert len(values) == 5 and all(math.isfinite(x) for x in values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = 2.7764451051977987 * sd / math.sqrt(5)
    positive, negative = sum(x > 0 for x in values), sum(x < 0 for x in values)
    return dict(mean=mean, sd=sd, ci95_low=mean-half, ci95_high=mean+half,
                positive=positive, negative=negative, zero=sum(x == 0 for x in values), n=5,
                favorable=negative if metric.endswith('_logloss') else positive if metric != 'epoch' else None,
                unfavorable=positive if metric.endswith('_logloss') else negative if metric != 'epoch' else None,
                favorable_direction='negative' if metric.endswith('_logloss') else 'positive' if metric != 'epoch' else None)


def main():
    receipt = json.loads((TABLES / 'EXPORT_RECEIPT.json').read_text())
    assert receipt['new_trajectories'] == 20 and receipt['reused_clean'] == receipt['reused_constant_nqr'] == 10
    assert receipt['useful_batches'] == 34500 and receipt['test_accessed'] is False
    endpoints = read_csv('ENDPOINTS.csv')
    assert len(endpoints) == 400
    index = {}
    for original in endpoints:
        row = dict(original)
        for name in list(row):
            if name.endswith(('_auc', '_logloss')):
                row[name] = float(row[name])
        row['epoch'], row['seed'] = int(row['epoch']), int(row['seed'])
        key = row['dataset'], row['arm'], row['seed'], row['endpoint'], row['estimator']
        assert key not in index
        index[key] = row
    metrics = [k for k in endpoints[0] if k.endswith(('_auc', '_logloss'))] + ['epoch']
    means, pairs, summaries = [], [], []
    for dataset in DATASETS:
        for endpoint in ENDPOINTS:
            for kind in KINDS:
                for arm in ARMS:
                    rows = [index[dataset, arm, seed, endpoint, kind] for seed in SEEDS]
                    for metric in metrics:
                        means.append(dict(dataset=dataset, arm=arm, estimator=kind, endpoint=endpoint,
                                          metric=metric, **describe([r[metric] for r in rows], metric)))
                contrasts = [(a, kind, b, kind) for a, b in CONTRASTS]
                if endpoint == 'selected':
                    contrasts += [(ARMS[0], 'fixed_ema', 'CLEAN', 'online'),
                                  (ARMS[0], 'fixed_ema', ARMS[0], 'online'),
                                  ('CLEAN', 'fixed_ema', 'CLEAN', 'online')]
                    if kind == 'online':
                        contrasts = contrasts[:len(CONTRASTS)]
                for left, lk, right, rk in contrasts:
                    comparison = f'{left}/{lk} minus {right}/{rk}'
                    for metric in metrics:
                        values = []
                        for seed in SEEDS:
                            l, r = index[dataset, left, seed, endpoint, lk], index[dataset, right, seed, endpoint, rk]
                            delta = l[metric] - r[metric]
                            values.append(delta)
                            pairs.append(dict(dataset=dataset, comparison=comparison, endpoint=endpoint, metric=metric,
                                              seed=seed, left=l[metric], right=r[metric], delta=delta,
                                              favorable=(delta < 0 if metric.endswith('_logloss') else delta > 0) if metric != 'epoch' else None,
                                              left_epoch=l['epoch'], right_epoch=r['epoch'],
                                              left_state_hash=l['state_hash'], right_state_hash=r['state_hash']))
                        summaries.append(dict(dataset=dataset, comparison=comparison, endpoint=endpoint,
                                              metric=metric, **describe(values, metric)))
    lookup = {(r['dataset'], r['comparison'], r['endpoint'], r['metric']): r for r in summaries}
    comparison = f'{ARMS[0]}/fixed_ema minus CLEAN/fixed_ema'
    primary = {}
    for dataset in DATASETS:
        auc, ll = (lookup[dataset, comparison, 'selected', m] for m in ('macro_auc', 'macro_logloss'))
        cells = {c: lookup[dataset, comparison, 'selected', c+'_auc'] for c in CELLS}
        minimum = min(r['mean'] for r in cells.values())
        primary[dataset] = dict(auc=auc, logloss=ll, cell_auc=cells, min_cell_mean_delta=minimum,
            macro_target_met=auc['mean'] >= .003, cell_protection_met=minimum >= -.001,
            auc_target_and_cell_protection_met=auc['mean'] >= .003 and minimum >= -.001,
            comprehensive_loss_decrease=ll['mean'] < 0)
    positives = sum(primary[d]['auc']['mean'] > 0 for d in DATASETS)
    outcome = ('NO_CONTINUATION_SIGNAL' if positives == 0 else 'MIXED_DATASET_SIGNAL' if positives == 1 else
               'VALIDATION_SCREEN_AUC_TARGET_AND_CELL_PROTECTION_MET_BOTH_DATASETS'
               if all(primary[d]['auc_target_and_cell_protection_met'] for d in DATASETS) else
               'POSITIVE_VALIDATION_SIGNAL_BELOW_TARGET')
    timing = {d: {k: {m: lookup[d, f'{ARMS[0]}/{k} minus {ARMS[1]}/{k}', 'E50', m]
                      for m in ('macro_auc', 'macro_logloss')} for k in KINDS} for d in DATASETS}
    structure = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for kind in KINDS:
                late, clean = (index[dataset, a, seed, 'selected', kind] for a in (ARMS[1], 'CLEAN'))
                bound = clean['epoch'] <= 40
                if bound:
                    assert late['macro_auc'] >= clean['macro_auc']
                e, n = (index[dataset, a, seed, 'E10', kind] for a in (ARMS[0], ARMS[2]))
                assert e['state_hash'] == n['state_hash']
                assert all(e[m] == n[m] for m in metrics)
                structure.append(dict(dataset=dataset, seed=seed, estimator=kind,
                    clean_selected_epoch=clean['epoch'], late_selected_epoch=late['epoch'],
                    late_selected_pre41=late['epoch'] <= 40, clean_optimum_in_exact_late_prefix=bound,
                    late_minus_clean_auc=late['macro_auc']-clean['macro_auc'], early_E10_is_reused_identity=True))
    diagnostics, groups = [], {}
    for row in read_csv('DIAGNOSTICS.csv'):
        if row['parameter_group'] == 'ALL' and row['available'] == 'True':
            groups.setdefault((row['dataset'], row['arm'], int(row['epoch'])), []).append(row)
    for (dataset, arm, epoch), rows in sorted(groups.items()):
        assert sorted(int(r['seed']) for r in rows) == list(SEEDS)
        for metric in ('data_gradient_norm', 'parameter_update_norm'):
            diagnostics.append(dict(dataset=dataset, arm=arm, epoch=epoch, metric=metric,
                                    **describe([float(r[metric]) for r in rows], metric)))
    prevalence, stat_groups = {}, {}
    for row in read_csv('STATISTICS.csv'):
        epoch = int(row['epoch'])
        if row['statistic'] == 'Q_R' and epoch == 1:
            key, counts = (row['dataset'], row['cell']), tuple(int(row[k]) for k in ('n','n0','n1'))
            assert key not in prevalence or prevalence[key] == counts
            prevalence[key] = counts
        if epoch not in (1,10,11,30,40,41,50):
            continue
        count = float(row['observed_count'])
        stat_groups.setdefault((row['dataset'], row['arm'], epoch, row['cell'], row['statistic']), []).append(
            float(row['count_weighted_sum']) / count if count else None)
    statistics_rows = []
    for (dataset, arm, epoch, cell, metric), values in sorted(stat_groups.items()):
        assert len(values) == 5
        valid = [v for v in values if v is not None]
        statistics_rows.append(dict(dataset=dataset, arm=arm, epoch=epoch, cell=cell, statistic=metric,
            valid_seeds=len(valid), mean=statistics.mean(valid) if valid else None,
            sd=statistics.stdev(valid) if len(valid)>1 else None,
            aggregation='equal_seed_mean_of_class_or_cell_count_weighted_batch_statistics'))
    costs, physical = read_csv('COSTS.csv'), read_csv('PHYSICAL_ATTEMPTS.csv')
    cost_means = []
    for dataset in DATASETS:
        for arm in ARMS:
            rows = [r for r in costs if r['dataset'] == dataset and r['arm'] == arm]
            assert len(rows) == 5
            for metric in ('physical_worker_seconds','training_seconds','evaluation_seconds','diagnostic_seconds',
                           'serialization_seconds','transfer_seconds','peak_full_process_bytes'):
                values = [float(r[metric]) for r in rows if r.get(metric) not in ('',None)]
                cost_means.append(dict(dataset=dataset, arm=arm, metric=metric, measured_seeds=len(values),
                    mean=statistics.mean(values) if values else None, sd=statistics.stdev(values) if len(values)>1 else None))
    for name, rows in (('ENDPOINT_MEANS.csv',means),('ALL_PAIRED_SEEDS.csv',pairs),('ALL_PAIRED_SUMMARY.csv',summaries),
                       ('DIAGNOSTIC_MEANS.csv',diagnostics),('STATISTIC_MEANS.csv',statistics_rows),
                       ('COST_MEANS.csv',cost_means),('SELECTION_STRUCTURE.csv',structure)):
        dump_csv(name,rows)
    appendix = ['# Cycle03 numerical appendix','','Seeds 2022–2026; natural validation; earliest strict Macro AUC maximum per stream. Selected LL uses that same state.','']
    for dataset in DATASETS:
        for endpoint in ENDPOINTS:
            for kind in KINDS:
                appendix += [f'## {dataset} / {endpoint} / {kind}','','| Arm | Seed | Epoch | Macro AUC | Macro LL | State SHA256 |','|---|---:|---:|---:|---:|---|']
                for arm in ARMS:
                    for seed in SEEDS:
                        r = index[dataset,arm,seed,endpoint,kind]
                        appendix.append(f'| {arm} | {seed} | {r["epoch"]} | {r["macro_auc"]:.12f} | {r["macro_logloss"]:.12f} | {r["state_hash"]} |')
                appendix.append('')
        for kind in KINDS:
            for arm in ARMS:
                appendix += [f'## {dataset} / selected / {kind} / {arm} — cells','','| Seed | '+' | '.join(f'{c} {m}' for c in CELLS for m in ('AUC','LL'))+' |','|---:|'+'---:|'*12]
                for seed in SEEDS:
                    r = index[dataset,arm,seed,'selected',kind]
                    appendix.append('| '+str(seed)+' | '+' | '.join(f'{r[c+"_"+m]:.12f}' for c in CELLS for m in ('auc','logloss'))+' |')
                appendix.append('')
    (OUT/'NUMERICAL_APPENDIX.md').write_text('\n'.join(appendix)+'\n')
    facts = dict(candidate=ARMS[0],decision=outcome,primary=primary,E50_early_minus_late=timing,
        new_trajectories=20,reused_clean=10,reused_constant_nqr=10,new_epochs=1000,new_useful_updates=34500,
        train_domain_counts=TRAIN_COUNTS,
        train_cell_counts={f'{d}/{c}':dict(n=n,n0=n0,n1=n1,prevalence=n1/n) for (d,c),(n,n0,n1) in prevalence.items()},
        physical_attempts=len(physical),physical_status_counts={s:sum(r['status']==s for r in physical) for s in set(r['status'] for r in physical)},
        total_measured_physical_worker_seconds=sum(float(r['seconds']) for r in physical),test_accessed=False,
        primary_estimator='independently_selected_fixed_ema',source_hash=receipt['source_hash'],
        design_bundle_sha256=receipt['design_bundle_sha256'],export_receipt_sha256=sha(TABLES/'EXPORT_RECEIPT.json'),
        inputs={name:sha(TABLES/name) for name in ('ENDPOINTS.csv','DIAGNOSTICS.csv','STATISTICS.csv','COSTS.csv','PHYSICAL_ATTEMPTS.csv')})
    (OUT/'DECISION.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(decision=outcome,primary=primary,E50_early_minus_late=timing,
        paired_seed_rows=len(pairs),paired_summary_rows=len(summaries),test_accessed=False),ensure_ascii=False))


if __name__ == '__main__':
    main()
