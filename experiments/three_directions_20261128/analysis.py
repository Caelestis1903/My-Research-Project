"""Summarize a completed A1 or B1 Validation stage with paired seed statistics."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--stage', choices=('A1', 'B1'), required=True)
    args = parser.parse_args()
    stage = args.stage
    p = args.artifacts.resolve() / stage


    def read(name):
        return json.loads((p / (name + '.json')).read_text())

    def write_csv(name, rows):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (p / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                              for key, value in row.items()} for row in rows)

    status = read('STAGE_STATUS')
    assert status['status'] == 'COMPLETE' and status['test_accesses'] == 0
    seeds = range(2022, 2027)
    datasets = ('movielens', 'kuairand')
    readouts = ('online', 'fixed_ema')
    contrasts = ({'P0-B0': {'P0': 1, 'B0': -1}, 'GP0-G0': {'GP0': 1, 'G0': -1},
                  'GP0-P0': {'GP0': 1, 'P0': -1}, 'G0-B0': {'G0': 1, 'B0': -1},
                  'GP0-B0': {'GP0': 1, 'B0': -1},
                  'interaction': {'GP0': 1, 'G0': -1, 'P0': -1, 'B0': 1}}
                 if stage == 'A1' else
                 {'B1-B0': {'B1': 1, 'B0': -1}, 'B2-B1': {'B2': 1, 'B1': -1},
                  'B2-B0': {'B2': 1, 'B0': -1}})
    thresholds = {'P0-B0': .002, 'GP0-G0': .001, 'B2-B1': .001}
    endpoints, costs = read('ENDPOINTS'), read('COST')
    assert all(row['split'] == 'validation' for row in endpoints)
    selected = read('SELECTED')
    assert len(selected) == status['completed'] * len(readouts)
    lookup = {(r['dataset'], r['readout'], r['checkpoint'], r['arm'], r['seed']): r for r in endpoints}
    assert len(lookup) == len(endpoints)
    metrics = tuple(key for key in endpoints[0] if key.endswith(('_auc', '_logloss')))
    six = [f'd{d}t{t}_auc' for d in range(3) for t in range(2)]
    worker = defaultdict(float)
    observer = defaultdict(float)
    for cost in costs:
        worker[cost['job']] += cost['worker_seconds']
        observer[cost['job']] += cost['observer_seconds']

    summary, deltas, decisions = [], [], []
    for dataset in datasets:
        for readout in readouts:
            for checkpoint in sorted({r['checkpoint'] for r in endpoints}, key=lambda k: (k != 'selected', k)):
                for contrast, weights in contrasts.items():
                    ranges = {seed:{arm:lookup[dataset, readout, checkpoint, arm, seed] for arm in weights} for seed in seeds}
                    avg = {}
                    base = dict(stage=stage, dataset=dataset, readout=readout, checkpoint=checkpoint, contrast=contrast)
                    for seed, arms in ranges.items():
                        assert len({row['data_hash'] for row in arms.values()}) == 1
                        values = {metric:sum(scale * arms[arm][metric] for arm, scale in weights.items()) for metric in metrics}
                        deltas.append(dict(base, seed=seed, **{f'{a}_epoch':row['epoch'] for a,row in arms.items()}, **values))
                    for metric in metrics:
                        diffs = [sum(scale * arms[arm][metric] for arm, scale in weights.items()) for arms in ranges.values()]
                        estimates = (mean(diffs), stdev(diffs))
                        width = 2.7764451051977987 * estimates[1] / len(diffs)**.5
                        avg[metric] = estimates[0]
                        summary.append(dict(base, metric=metric, n=len(diffs), paired_mean=estimates[0], paired_sd=estimates[1],
                                            ci95_low=estimates[0]-width, ci95_high=estimates[0]+width, positive=sum(v>0 for v in diffs)))
                    if checkpoint != 'selected':
                        continue
                    low_cell = min(six, key=lambda cell:avg[cell])
                    macro = next(row for row in reversed(summary) if row['metric']=='macro_auc')
                    threshold = thresholds.get(contrast)
                    gate = None if threshold is None else avg['macro_auc'] >= threshold and avg[low_cell] >= -.001
                    times = {arm:mean(worker[arms[arm]['job_key']] for arms in ranges.values()) for arm in weights}
                    stage_observer = {arm:mean(observer[arms[arm]['job_key']] for arms in ranges.values()) for arm in weights}
                    row = dict(base, mean_delta_auc=avg['macro_auc'], mean_delta_ll=avg['macro_logloss'],
                               mean_traffic_auc=avg['sample_weighted_cell_auc'], mean_pooled_auc=avg['pooled_task_mean_auc'],
                               ci95_low=macro['ci95_low'], ci95_high=macro['ci95_high'], positive=macro['positive'],
                               min_cell=low_cell, min_cell_delta=avg[low_cell], six_cell_protection=avg[low_cell]>=-.001,
                               practical_threshold=threshold, validation_gate_pass=gate, mean_arm_worker_seconds=times,
                               mean_arm_observer_seconds=stage_observer, cell_mean_delta_auc={cell:avg[cell] for cell in six})
                    decisions.append(row)
                    print(dataset, readout, contrast, f"AUC {row['mean_delta_auc']:+.6f}",
                          f"CI [{row['ci95_low']:+.6f},{row['ci95_high']:+.6f}]", f"{row['positive']}/5",
                          f"LL {row['mean_delta_ll']:+.6f}", low_cell, f"{row['min_cell_delta']:+.6f}", f"gate={gate}")

    for fn, rows in [('endpoints.csv', endpoints), ('per_seed_selected.csv', selected), ('six_cell_selected.csv', read('SIX_CELL')),
                     ('costs.csv', costs), ('paired_summary.csv', summary), ('paired_seed.csv', deltas)]:
        write_csv(fn, rows)
    decision_file = p / 'decisions.json'
    record = json.loads(decision_file.read_text()) if decision_file.exists() else {}
    record.update(stage=stage, split='validation', paired_ci='two_sided_student_t_df4',
                  validated_comparisons=decisions, test_accesses=0)
    decision_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


if __name__ == "__main__":
    main()
