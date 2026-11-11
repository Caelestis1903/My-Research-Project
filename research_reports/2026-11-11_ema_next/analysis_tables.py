"""Recompute report statistics from released scalar evidence; no model evaluation."""
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEDS = set(range(2022, 2027))


def read(name):
    with gzip.open(ROOT / name, 'rt') as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with gzip.open(ROOT / name, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def stats(values):
    assert len(values) == 5
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = 2.7764451052 * sd / math.sqrt(5)
    return dict(values=values, n=5, mean=mean, sd=sd,
                ci95=[mean - half, mean + half], positive_seeds=sum(x > 0 for x in values))


def main():
    endpoints = read('ENDPOINTS.csv.gz')
    groups = defaultdict(dict)
    for row in endpoints:
        groups[(row['dataset'], row['condition'], row['candidate'], row['split'])][int(row['seed'])] = row
    metrics = [k for k in endpoints[0] if k.endswith(('_auc', '_logloss'))]
    summary = []
    for (ds, condition, candidate, split), data in sorted(groups.items()):
        assert set(data) == SEEDS
        for metric in metrics:
            summary.append(dict(dataset=ds, condition=condition, candidate=candidate,
                                split=split, metric=metric,
                                **stats([float(data[s][metric]) for s in sorted(SEEDS)])))
    write('ENDPOINT_SUMMARY.csv.gz', summary)

    baseline = defaultdict(dict)
    for row in read('PUBLISHED_BASELINES.csv.gz'):
        baseline[(row['dataset'], row['method'])][int(row['seed'])] = row
    paired = read('PAIRED.csv.gz')
    keys = {(r['dataset'], r['condition'], r['candidate'], r['control'], r['metric']) for r in paired}
    comparisons = {(r['dataset'], r['condition'], r['candidate'], r['control']) for r in paired}
    for ds, condition, candidate, split in groups:
        control = candidate.replace('_all_', '_S_')
        if split == 'test' and candidate.startswith('B_H') and control != candidate:
            assert (ds, condition, control, split) in groups
            comparisons.add((ds, condition, candidate, control))
    for ds, condition, candidate, control in sorted(comparisons):
        data = groups[(ds, condition, candidate, 'test')]
        ref = baseline[(ds, control.split(':', 1)[1])] if control.startswith('PUBLISHED:') else groups[(ds, condition, control, 'test')]
        assert set(data) == set(ref) == SEEDS
        for metric in metrics:
            key = (ds, condition, candidate, control, metric)
            if key in keys:
                continue
            cv = [float(data[s][metric]) for s in sorted(SEEDS)]
            bv = [float(ref[s][metric]) for s in sorted(SEEDS)]
            paired.append(dict(dataset=ds, condition=condition, candidate=candidate,
                               control=control, metric=metric, **stats([a-b for a, b in zip(cv, bv)]),
                               candidate_values=cv, control_values=bv,
                               comparison_class='report_derived_same_frozen_endpoints'))
    write('PAIRED.csv.gz', paired)

    stages = defaultdict(dict)
    with gzip.open(ROOT / 'evidence/EPOCH_RECORDS.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row['family'] != 'PARENT' or row['epoch'] not in (20, 35, 50):
                continue
            ds = row['dataset']
            winner = 'B_H40_all_logit' if ds == 'movielens' else 'B_H277_all_logit'
            for candidate, control in [(winner, 'F'), ('B_SHARED', 'F'), (winner, winner.replace('_all_', '_S_'))]:
                if candidate not in row['readouts'] or control not in row['readouts']:
                    continue
                a = row['readouts'][candidate]['metrics']
                b = row['readouts'][control]['metrics']
                for metric in metrics:
                    key = (ds, row['objective'], row['epoch'], candidate, control, metric)
                    stages[key][row['seed']] = a[metric] - b[metric]
    stage_rows = []
    for (ds, condition, epoch, candidate, control, metric), data in sorted(stages.items()):
        assert set(data) == SEEDS
        stage_rows.append(dict(dataset=ds, condition=condition, epoch=epoch,
                               candidate=candidate, control=control, metric=metric,
                               **stats([data[s] for s in sorted(SEEDS)])))
    write('STAGE_COMPARISONS.csv.gz', stage_rows)

    lookup = {(r['dataset'], r['condition'], r['candidate'], r['control'], r['metric']): r for r in paired}
    for formal in json.loads((ROOT / 'FORMAL_DECISIONS.json').read_text()):
        key = (formal['dataset'], formal['condition'], formal['candidate'], 'PUBLISHED:' + formal['control'])
        auc = float(lookup[key + ('macro_auc',)]['mean'])
        ll = float(lookup[key + ('macro_logloss',)]['mean'])
        guard = all(float(lookup[key + (f'd{d}t{t}_auc',)]['mean']) >= -.001
                    for d in range(3) for t in range(2))
        assert abs(auc - formal['delta_auc']) < 1e-12 and abs(ll - formal['delta_logloss']) < 1e-12
        assert formal['auc_pass'] == (auc >= .003 and guard)
    print(dict(endpoint_summary_rows=len(summary), paired_rows=len(paired), stage_rows=len(stage_rows), formal_checks=12))


if __name__ == '__main__':
    main()
