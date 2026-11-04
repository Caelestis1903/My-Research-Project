"""Recompute stage, task, cell and seed contrasts from archived endpoints."""
import csv
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(2022, 2027))
ARMS = ('G-D0', 'G-D02', 'G-D2', 'FIXED_EMA', 'M3OE', 'HYPERGATE', 'AUTOIFS')
CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))
METRICS = ('macro_auc', 'macro_logloss', 'sample_weighted_cell_auc',
           'sample_weighted_cell_logloss', 'pooled_task_mean_auc', 'pooled_logloss',
           'task0_macro_auc', 'task1_macro_auc', 'task0_macro_logloss', 'task1_macro_logloss',
           'worst_cell_auc') + tuple(c + '_' + m for c in CELLS for m in ('auc', 'logloss'))


def stats(values):
    assert len(values) == 5 and all(math.isfinite(v) for v in values)
    mean, sd = statistics.fmean(values), statistics.stdev(values)
    half = 2.7764451051977987 * sd / math.sqrt(5)
    return dict(values=values, mean=mean, sd=sd, ci95=[mean-half, mean+half],
                positive=sum(v > 0 for v in values), negative=sum(v < 0 for v in values))


def main():
    rows = list(csv.DictReader((HERE / 'ENDPOINTS.csv').open()))
    lookup = {}
    for row in rows:
        estimator = 'online' if row['arm'] in ('M3OE', 'HYPERGATE', 'AUTOIFS') else 'fixed_ema'
        if row['arm'] in ARMS and row['estimator'] == estimator and row['split'] in ('test', 'validation'):
            key = (row['arm'], row['checkpoint'], row['split'], int(row['seed']))
            assert key not in lookup, key
            lookup[key] = row
    assert len(lookup) == len(ARMS) * 3 * 2 * 5
    result = dict(absolute={}, paired={})
    stages = []
    for checkpoint in ('selected', 'E10', 'E50'):
        result['absolute'][checkpoint], result['paired'][checkpoint] = {}, {}
        for split in ('test', 'validation'):
            absolute = {}
            for arm in ARMS:
                values = [lookup[arm, checkpoint, split, seed] for seed in SEEDS]
                for row in values:
                    assert abs(float(row['macro_auc']) - statistics.fmean(float(row[c+'_auc']) for c in CELLS)) < 1e-12
                    assert abs(float(row['macro_logloss']) - statistics.fmean(float(row[c+'_logloss']) for c in CELLS)) < 1e-12
                absolute[arm] = dict(epochs=[int(row['epoch']) for row in values],
                    metrics={m: stats([float(row[m]) for row in values]) for m in METRICS})
            result['absolute'][checkpoint][split] = absolute
            paired = {}
            for arm in ('G-D0', 'G-D02', 'G-D2'):
                for reference in ARMS:
                    if reference == arm:
                        continue
                    contrast = {}
                    for metric in METRICS:
                        a, b = absolute[arm]['metrics'][metric]['values'], absolute[reference]['metrics'][metric]['values']
                        contrast[metric] = stats([x-y for x, y in zip(a, b)])
                    paired[arm + '_vs_' + reference] = contrast
                    stages.append(dict(checkpoint=checkpoint, split=split, arm=arm, reference=reference,
                        **{m: contrast[m]['mean'] for m in METRICS}))
            result['paired'][checkpoint][split] = paired
    with (HERE / 'STAGE_EFFECTS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stages[0]))
        writer.writeheader()
        writer.writerows(stages)
    (HERE / 'ANALYSIS_VALUES.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({arm: {m: result['absolute']['selected']['test'][arm]['metrics'][m]
                           for m in ('macro_auc', 'macro_logloss')} for arm in ARMS}, indent=2))


if __name__ == '__main__':
    main()
