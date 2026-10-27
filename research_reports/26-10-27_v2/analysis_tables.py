"""Recompute final seed-level tables from the completed frozen experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT.parents[1] / 'artifacts/mdmt_relation_basis_v2/revision1'
SEEDS = [2022, 2023, 2024, 2025]
METHODS = ['B_REFERENCE', 'DIRECT', 'READ', 'TARGET_ONLY', 'LOWRANK_RELATION', 'FULL_RELATION']


def read(path):
    return json.loads(Path(path).read_text())


def stats(values):
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if not len(x):
        return dict(n=0, mean=None, sample_sd=None, ci95_low=None, ci95_high=None)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else None
    half = float(t.ppf(.975, len(x)-1)*sd/np.sqrt(len(x))) if sd is not None else None
    return dict(n=len(x), mean=mean, sample_sd=sd,
                ci95_low=mean-half if half is not None else None,
                ci95_high=mean+half if half is not None else None)


def main():
    result = read(ROOT/'RESULTS_SUMMARY.json')
    rows = result['seed_rows']
    assert result['completed_adaptation_runs'] == 48 and len(rows) == 288
    assert sum(r['role'] == 'E' for r in rows) == 96
    assert len(list((ARTIFACT/'workers').glob('*/seed*/COMPLETE.json'))) == 8
    metric_names = ['macro_auc', 'macro_logloss', 'natural_logloss',
                    'pooled_auc_task0', 'pooled_auc_task1']
    metric_names += [f'd{d}_t{q}_{m}' for q in range(2) for d in range(3) for m in ['auc', 'logloss']]
    index = {(r['dataset'], r['method'], r['role'], r['state'], r['seed']): r for r in rows}
    assert len(index) == 288
    summary, paired, stages, tasks = [], [], [], []
    for dataset in ['movielens', 'kuairand']:
        for role in ['V', 'Q', 'E']:
            for state in ['selected', 'final']:
                for method in METHODS:
                    group = [index[dataset, method, role, state, s] for s in SEEDS]
                    for metric in metric_names:
                        summary.append(dict(dataset=dataset, method=method, role=role, state=state,
                                            metric=metric, **stats([r[metric] for r in group])))
                    for task in range(2):
                        values = {key: [] for key in ['task_macro_auc', 'task_macro_logloss', 'task_natural_logloss', 'pooled_auc']}
                        for row in group:
                            auc = [row[f'd{d}_t{task}_auc'] for d in range(3)]
                            values['task_macro_auc'].append(float(np.mean([v for v in auc if v is not None])))
                            ll = [row[f'd{d}_t{task}_logloss'] for d in range(3)]
                            ns = [row[f'd{d}_t{task}_n'] for d in range(3)]
                            values['task_macro_logloss'].append(float(np.mean(ll)))
                            values['task_natural_logloss'].append(float(np.average(ll, weights=ns)))
                            values['pooled_auc'].append(row[f'pooled_auc_task{task}'])
                        for metric, vals in values.items():
                            tasks.append(dict(dataset=dataset, method=method, role=role, state=state,
                                              task=task, metric=metric, **stats(vals)))
                for control in [m for m in METHODS if m != 'LOWRANK_RELATION']:
                    for metric in metric_names:
                        vals = []
                        for seed in SEEDS:
                            a = index[dataset, 'LOWRANK_RELATION', role, state, seed][metric]
                            b = index[dataset, control, role, state, seed][metric]
                            vals.append(None if a is None or b is None else a-b)
                        valid = [v for v in vals if v is not None]
                        lower = 'logloss' in metric
                        paired.append(dict(dataset=dataset, method='LOWRANK_RELATION', control=control,
                                           role=role, state=state, metric=metric, **stats(vals),
                                           improved_seed_count=sum(v < 0 if lower else v > 0 for v in valid),
                                           **{f'seed{s}': v for s, v in zip(SEEDS, vals)}))
            for method in METHODS:
                for metric in metric_names:
                    vals = []
                    for seed in SEEDS:
                        a = index[dataset, method, role, 'final', seed][metric]
                        b = index[dataset, method, role, 'selected', seed][metric]
                        vals.append(None if a is None or b is None else a-b)
                    stages.append(dict(dataset=dataset, method=method, role=role, metric=metric,
                                       comparison='final_minus_selected', **stats(vals),
                                       **{f'seed{s}': v for s, v in zip(SEEDS, vals)}))
    for name, records in [('METRIC_SUMMARY_ALL.csv', summary), ('PAIRED_ALL_CONTROLS.csv', paired),
                           ('TRAINING_STAGE_COMPARISON.csv', stages), ('PER_TASK_SUMMARY.csv', tasks)]:
        pd.DataFrame(records).to_csv(ROOT/name, index=False)

    diagnostics = []
    for dataset in ['movielens', 'kuairand']:
        for method in ['READ', 'LOWRANK_RELATION', 'FULL_RELATION']:
            for seed in SEEDS:
                z = read(ARTIFACT/'runs'/dataset/method/f'seed{seed}'/'diagnostics.json')
                record = dict(dataset=dataset, method=method, seed=seed,
                              formation_evidence=z['formation_evidence'], flags=';'.join(z['flags']),
                              mean_route_entropy=float(np.mean(z['entropy'])),
                              mean_NULL_mass=float(np.asarray(z['route_mean'])[:, :, -1].mean()),
                              max_source_LL_excess=max(v['delta_natural_logloss'] for v in z['source_retention']),
                              max_source_output_deviation_rms=max(v['output_deviation_rms'] for v in z['source_retention']))
                for intervention in ['all_root_minus_full', 'constant_minus_full']:
                    for metric in ['macro_logloss', 'macro_auc', 'natural_logloss']:
                        record[intervention+'_'+metric] = z[intervention][metric]
                for source, distance in enumerate(z['basis_output_rms_distance']):
                    record[f'source{source}_basis_output_rms_distance'] = distance
                diagnostics.append(record)
    pd.DataFrame(diagnostics).to_csv(ROOT/'FUNCTION_DIAGNOSTICS.csv', index=False)

    profile = []
    audit = read(ROOT/'AUDIT_DATA.json')
    for dataset, description in audit['datasets'].items():
        for role, details in description['roles'].items():
            for cell in details['cells']:
                profile.append(dict(dataset=dataset, role=role, **cell,
                                    positive_rate=cell['positive']/cell['n']))
    pd.DataFrame(profile).to_csv(ROOT/'DATA_ROLE_PROFILE.csv', index=False)
    print(json.dumps(dict(seed_rows=len(rows), E_rows=96, metric_summary=len(summary),
                          paired_contrasts=len(paired), task_summary=len(tasks), diagnostics=len(diagnostics))))


if __name__ == '__main__':
    main()
