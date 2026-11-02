"""Aggregate recorded endpoints using seeds as the replicate unit."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def collect(attempt):
    rows = []
    for path in sorted((attempt / 'runs').glob('*/result.json')):
        r = json.loads(path.read_text()); j = r['job']; c = j['config']
        for checkpoint in ('selected', 'E10', 'E50'):
            for kind, metrics in r[checkpoint + '_validation'].items():
                rows.append(dict(dataset=j['dataset'], arm=c['arm'], lambda_pair=c['lambda_pair'],
                    seed=j['seed'], key=j['key'], role='validation', checkpoint=checkpoint,
                    estimator=kind, epoch=r['selected_epochs'][kind] if checkpoint == 'selected' else int(checkpoint[1:]),
                    state_hash=r[checkpoint + '_hashes'][kind], **metrics))
    for path in sorted((attempt / 'test_endpoints').glob('*.json')):
        r = json.loads(path.read_text()); j = r['job']; c = j['config']
        rows.append(dict(dataset=j['dataset'], arm=c['arm'], lambda_pair=c['lambda_pair'],
            seed=j['seed'], key=j['key'], role='test', checkpoint=r['checkpoint'],
            estimator=r['model_kind'], epoch=r['epoch'], state_hash=r['model_hash'], **r['metrics']))
    return pd.DataFrame(rows)


def tables(attempt, output):
    output.mkdir(parents=True, exist_ok=True)
    frame = collect(attempt)
    if frame.empty:
        raise ValueError('No completed trajectories')
    identity = ['dataset', 'arm', 'lambda_pair', 'seed', 'role', 'checkpoint', 'estimator']
    assert not frame.duplicated(identity).any(), 'Duplicate scientific endpoint'
    frame.to_csv(output / 'endpoints.csv', index=False)
    metrics = [c for c in frame if c.endswith(('_auc', '_logloss'))]
    group = ['dataset', 'arm', 'lambda_pair', 'role', 'checkpoint', 'estimator']
    long = frame.melt(id_vars=identity, value_vars=metrics, var_name='metric', value_name='value')
    summary = long.groupby(group + ['metric']).value.agg(['count', 'mean', 'std']).reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    paired = []
    for keys, data in frame.groupby(['dataset', 'role', 'checkpoint', 'estimator']):
        variants = list(data.groupby(['arm', 'lambda_pair']))
        for (arm, lam), a in variants:
            for (control, control_lam), b in variants:
                if (arm, lam) == (control, control_lam):
                    continue
                matched = a.merge(b, on='seed', suffixes=('_a', '_b'), validate='one_to_one')
                for _, row in matched.iterrows():
                    for metric in metrics:
                        paired.append(dict(zip(['dataset', 'role', 'checkpoint', 'estimator'], keys),
                            arm=arm, lambda_pair=lam, control=control, control_lambda=control_lam,
                            seed=int(row.seed), metric=metric, difference=row[metric+'_a']-row[metric+'_b']))
    pairs = pd.DataFrame(paired)
    pairs.to_csv(output / 'paired_seed.csv', index=False)
    grouping = ['dataset', 'role', 'checkpoint', 'estimator', 'arm', 'lambda_pair', 'control', 'control_lambda', 'metric']
    stats = pairs.groupby(grouping).difference.agg(n='count', mean='mean', sd='std', positive=lambda x: (x > 0).sum(), negative=lambda x: (x < 0).sum()).reset_index()
    half = t.ppf(.975, stats.n-1) * stats.sd / np.sqrt(stats.n)
    stats['ci95_low'] = stats['mean']-half; stats['ci95_high'] = stats['mean']+half
    stats.to_csv(output / 'paired_summary.csv', index=False)
    return frame, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frame, stats = tables(args.attempt, args.output)
    print(json.dumps(dict(endpoints=len(frame), paired_comparisons=len(stats))))
