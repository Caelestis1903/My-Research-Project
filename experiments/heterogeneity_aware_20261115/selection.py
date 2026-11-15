"""Predeclared validation-only arithmetic; seeds are independent units."""
import numpy as np
from .protocol import SEEDS, DATASETS, ARMS


def grouped(records):
    groups = {}
    for record in records:
        c = record['job']['config']
        key = (record['job']['dataset'], c['arm'], c['alpha'])
        groups.setdefault(key, {})[record['job']['seed']] = record
    return groups


def scores(groups, dataset, arm, alpha=.03, kind='fixed_ema'):
    group = groups[dataset, arm, alpha]
    assert set(group) == set(SEEDS), 'Five complete seeds required'
    assert all(r['epochs'] == 50 for r in group.values()), 'Full 50epochs required'
    return np.asarray([group[s]['selected_validation'][kind]['macro_auc'] for s in SEEDS])


def paired(values):
    values = np.asarray(values, dtype=float)
    assert len(values) == 5 and np.isfinite(values).all()
    mean = float(values.mean()); sd = float(values.std(ddof=1)); radius = 2.776445*sd/np.sqrt(5)
    return dict(seeds=list(SEEDS), values=values.tolist(), mean=mean, sample_sd=sd,
                ci95=[mean-radius, mean+radius], positive_seeds=int((values > 0).sum()))


def pilot_decision(records):
    groups = grouped(records); evidence = {}; eligible = []
    for dataset in DATASETS:
        for arm in ARMS:
            scores(groups, dataset, arm)
    for arm in ('A', 'B', 'C'):
        comparisons = {}; positive_datasets = 0; margins = []
        for dataset in DATASETS:
            candidate = scores(groups, dataset, arm)
            differences = {control:paired(candidate-scores(groups, dataset, control))
                           for control in ('CLEAN', 'RANDOM', 'GAUSSIAN')}
            continuation = all(value['mean'] > 0 for value in differences.values())
            positive_datasets += int(continuation)
            margins.append(min(value['mean'] for value in differences.values()))
            comparisons[dataset] = dict(continuation=continuation, comparisons=differences)
        evidence[arm] = comparisons
        if positive_datasets:
            eligible.append((arm, positive_datasets, float(np.mean(margins))))
    preference = {'C':0, 'B':1, 'A':2}
    eligible.sort(key=lambda item:(-item[1], -item[2], preference[item[0]]))
    routes = [item[0] for item in eligible[:2]]
    return dict(status='CONTINUE' if routes else 'NO_CONTINUATION_SIGNAL', routes=routes,
                evidence=evidence, test_accessed=False,
                later_stages='PENDING' if routes else 'NOT_TRIGGERED')


def hpo_selection(records, routes):
    groups = grouped(records); selected = {}; evidence = {}
    for arm in routes:
        candidates = []
        for alpha in (.01,.03,.05):
            values = [float(scores(groups, d, arm, alpha).mean()) for d in DATASETS]
            candidates.append((alpha, float(np.mean(values))))
        alpha, _ = max(candidates, key=lambda item:(item[1], -item[0]))
        comparisons = {d:{c:paired(scores(groups,d,arm,alpha)-scores(groups,d,c))
                          for c in ('CLEAN','RANDOM','GAUSSIAN')} for d in DATASETS}
        eligible = any(all(record['mean'] > 0 for record in cells.values()) for cells in comparisons.values())
        evidence[arm] = dict(alpha_scores=candidates, alpha=alpha, comparisons=comparisons, eligible=eligible)
        if eligible:
            selected[arm] = alpha
    configurations = set()
    for arm, alpha in selected.items():
        controls = {'A':('CLEAN','A','B','C','RANDOM','GAUSSIAN'),
                    'B':('CLEAN','B','C','RANDOM','GAUSSIAN'),
                    'C':('CLEAN','C','RANDOM','GAUSSIAN')}[arm]
        for dataset in DATASETS:
            for control in controls:
                configurations.add((dataset, control, .03 if control == 'CLEAN' else alpha))
    return dict(status='FREEZE_CANDIDATES' if selected else 'NO_CONTINUATION_SIGNAL',
                selected=selected, evidence=evidence, configurations=[list(v) for v in sorted(configurations)],
                seed_pool_reused=True, test_accessed=False)
