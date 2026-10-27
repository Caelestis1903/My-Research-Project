"""Seed-level summaries of actual exported measurements; no training or E selection."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import t


def interval(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0, mean=None, sd=None, ci_low=None, ci_high=None)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else None
    half = float(t.ppf(.975, len(a)-1)*sd/np.sqrt(len(a))) if sd is not None else None
    return dict(n=len(a), mean=mean, sd=sd,
                ci_low=mean-half if half is not None else None,
                ci_high=mean+half if half is not None else None)


def collect(root):
    seeds, cells = [], []
    paths = [(p, False) for p in (root/'runs').glob('*/*/seed*/*_metrics.json')]
    paths += [(p, True) for p in (root/'workers').glob('*/seed*/*_NATIVE_RAW_metrics.json')]
    for p, native in paths:
        m = json.loads(p.read_text())
        if native:
            dataset, seed_text = p.parts[-3:-1]
            method = 'NATIVE_RAW'
        else:
            dataset, method, seed_text = p.parts[-4:-1]
        role, state = p.stem.removesuffix('_metrics').split('_', 1)
        seed = int(seed_text[4:])
        base = dict(dataset=dataset, method=method, seed=seed, role=role, state=state,
                    phase='development' if seed in (2022, 2023) else 'locked')
        for key in ('macro_auc', 'macro_logloss', 'natural_logloss'):
            seeds.append(dict(**base, metric=key, value=m[key]))
        for task, auc in enumerate(m['pooled_auc_by_task']):
            seeds.append(dict(**base, metric=f'pooled_auc_task{task}', value=auc))
        for task in (0, 1):
            for key in ('auc', 'logloss'):
                values = [c[key] for c in m['cells'] if c['task'] == task and c[key] is not None]
                seeds.append(dict(**base, metric=f'task{task}_macro_{key}',
                                  value=float(np.mean(values)) if values else None))
        cells.extend(dict(**base, **c) for c in m['cells'])
    return pd.DataFrame(seeds), pd.DataFrame(cells)


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=True)
    seeds, cells = collect(root)
    if seeds.empty:
        return {'status': 'NO_COMPLETE_MEASUREMENTS'}
    keys = ['dataset', 'phase', 'role', 'state', 'method', 'metric']
    assert not seeds.duplicated(keys+['seed']).any(), 'Duplicate seed measurements'
    seeds.to_csv(output/'SEED_METRICS_LONG.csv', index=False)
    cells.to_csv(output/'CELL_METRICS.csv', index=False)
    summaries = [dict(zip(keys, k), **interval(g.value)) for k, g in seeds.groupby(keys)]
    pd.DataFrame(summaries).to_csv(output/'METRIC_SUMMARY.csv', index=False)
    selection = {}
    for dataset in ('movielens', 'kuairand'):
        q = seeds[(seeds.dataset == dataset)&(seeds.phase == 'development')&
                  (seeds.role == 'Q')&(seeds.state == 'selected')&
                  (seeds.metric == 'macro_logloss')]
        controls = q[q.method.isin(['TARGET_MATCH', 'FIXED_READ', 'STATIC_LOCAL'])]
        if len(controls) != 6:
            continue
        selection[dataset] = controls.groupby('method').value.mean().sort_values().index[0]
    paired, paired_cells = [], []
    for row in seeds.to_dict('records'):
        for reference in dict.fromkeys(['NATIVE_RAW', 'TARGET_MATCH', selection.get(row['dataset'])]):
            if reference is None or reference == row['method']:
                continue
            match = seeds[(seeds.dataset == row['dataset'])&(seeds.seed == row['seed'])&
                          (seeds.role == row['role'])&(seeds.metric == row['metric'])&
                          (seeds.method == reference)&
                          (seeds.state == ('NATIVE_RAW' if reference == 'NATIVE_RAW' else row['state']))]
            if len(match) == 1 and row['value'] is not None and pd.notna(row['value']):
                paired.append(dict(**row, reference=reference, baseline=float(match.iloc[0].value),
                                   delta=row['value']-float(match.iloc[0].value)))
    p = pd.DataFrame(paired)
    p.to_csv(output/'PAIRED_SEED_DELTAS.csv', index=False)
    if not p.empty:
        pk = keys+['reference']
        ps = [dict(zip(pk, k), **interval(g.delta),
                   improved=int((g.delta > 0).sum()) if 'auc' in k[5] else int((g.delta < 0).sum()))
              for k, g in p.groupby(pk)]
        pd.DataFrame(ps).to_csv(output/'PAIRED_SUMMARY.csv', index=False)
    for row in cells.to_dict('records'):
        for reference in dict.fromkeys(['TARGET_MATCH', selection.get(row['dataset'])]):
            if reference is None or reference == row['method']:
                continue
            match = cells[(cells.dataset == row['dataset'])&(cells.seed == row['seed'])&
                          (cells.role == row['role'])&(cells.state == row['state'])&
                          (cells.method == reference)&(cells.cell == row['cell'])]
            if len(match) == 1:
                for metric in ('auc', 'logloss'):
                    if pd.notna(row[metric]) and pd.notna(match.iloc[0][metric]):
                        paired_cells.append(dict(**row, reference=reference, metric=metric,
                            delta=row[metric]-float(match.iloc[0][metric])))
    pc = pd.DataFrame(paired_cells)
    pc.to_csv(output/'PAIRED_CELL_DELTAS.csv', index=False)
    if not pc.empty:
        ck = ['dataset', 'phase', 'role', 'state', 'method', 'reference', 'domain', 'task', 'metric']
        cs = [dict(zip(ck, k), **interval(g.delta)) for k, g in pc.groupby(ck)]
        pd.DataFrame(cs).to_csv(output/'PAIRED_CELL_SUMMARY.csv', index=False)
    component_rows = []
    for candidate, reference in [('CLUSTER_ONLY','STATIC_LOCAL'), ('RELATION_ONLY','FIXED_READ'),
                                  ('CLUSTER_RELATION','CLUSTER_ONLY'), ('CLUSTER_RELATION','RELATION_ONLY')]:
        joined = seeds[seeds.method == candidate].merge(seeds[seeds.method == reference],
            on=['dataset','seed','phase','role','state','metric'], suffixes=('_a','_c'), validate='one_to_one')
        for r in joined.to_dict('records'):
            component_rows.append({k:r[k] for k in ['dataset','seed','phase','role','state','metric']} |
                dict(candidate=candidate, reference=reference, delta=r['value_a']-r['value_c']))
    component = pd.DataFrame(component_rows)
    component.to_csv(output/'MATCHED_COMPONENT_DELTAS.csv', index=False)
    if not component.empty:
        ck = ['dataset','phase','role','state','metric','candidate','reference']
        summaries = [dict(zip(ck, k), **interval(g.delta)) for k, g in component.groupby(ck)]
        pd.DataFrame(summaries).to_csv(output/'MATCHED_COMPONENT_SUMMARY.csv', index=False)
    return dict(status='MEASURED_SUMMARIES', Cstar_by_development_Q=selection,
                seed_metric_records=len(seeds), cell_records=len(cells))


if __name__ == '__main__':
    check = interval([1., 3.])
    assert check['n'] == 2 and check['mean'] == 2. and np.isclose(check['sd'], np.sqrt(2.))
    assert interval([1.])['ci_low'] is None
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.output), ensure_ascii=False))
