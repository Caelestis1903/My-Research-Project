"""Build report tables from frozen scalar evaluation records."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.ema_view_design_20261102_v2.gates import paired

REFERENCES = {
    'A1': ['A0'], 'J0': ['A0', 'A1'], 'J1': ['A0', 'A1', 'J0'],
    'F0': ['A0', 'A1'], 'F1': ['A0', 'A1', 'F0'],
    'RG': ['A0', 'A1'], 'AG': ['A0', 'A1'], 'TG': ['A0', 'A1', 'RG', 'AG'],
}


def build(attempt, out):
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for role in ('validation', 'test'):
        frame = pd.read_csv(attempt / f'{role}_metrics.csv')
        assert len(frame) == 144 and frame.key.nunique() == 36
        assert set(frame.seed) == {2022, 2024}
        frame.to_csv(out / f'{role}_metrics.csv', index=False)
        frames.append(frame.assign(role=role))
    data = pd.concat(frames, ignore_index=True)
    metrics = [c for c in data if c.endswith(('_auc', '_logloss'))]
    dimensions = ['role', 'dataset', 'split_id', 'arm', 'kappa', 'model_kind', 'checkpoint']
    summary = data.groupby(dimensions)[metrics].agg(['mean', 'std'])
    summary.columns = [f'{a}_{b}' for a, b in summary.columns]
    summary = summary.reset_index()
    summary['n'] = 2
    summary.to_csv(out / 'endpoint_summary.csv', index=False)
    effects = pd.read_csv(attempt / 'paired_effects.csv')
    effects = effects[(effects.phase == 'development') & effects.apply(
        lambda row: row.reference in REFERENCES.get(row.arm, []), axis=1)]
    effects.to_csv(out / 'paired_effects.csv.gz', index=False, compression='gzip')
    primary = effects[(effects.model_kind == 'fixed_ema') & (effects.checkpoint == 'selected')
                      & effects.metric.isin(['macro_auc', 'macro_logloss'])]
    primary.to_csv(out / 'primary_effects.csv', index=False)
    cells = []
    ema_effects = []
    minimum_gains = []
    for (role, dataset, checkpoint), group in data.groupby(['role', 'dataset', 'checkpoint']):
        ema = group[group.model_kind == 'fixed_ema']
        baseline = ema[ema.arm == 'A0'].set_index('seed')
        for arm, records in ema.groupby('arm'):
            for d in range(3):
                for t in range(2):
                    auc = records.set_index('seed')[f'd{d}t{t}_auc']
                    ll = records.set_index('seed')[f'd{d}t{t}_logloss']
                    delta = auc - baseline[f'd{d}t{t}_auc']
                    cells.append(dict(role=role, dataset=dataset, checkpoint=checkpoint,
                        arm=arm, cell=f'd{d}t{t}', auc=auc.mean(), logloss=ll.mean(),
                        delta_auc=delta.mean(), positive_seeds=int((delta > 0).sum()),
                        delta_logloss=(ll-baseline[f'd{d}t{t}_logloss']).mean(),
                        count=int(records[f'd{d}t{t}_count'].iloc[0]),
                        positive=int(records[f'd{d}t{t}_positive'].iloc[0])))
        for arm, records in group.groupby('arm'):
            online = records[records.model_kind == 'online'].set_index('seed').to_dict('index')
            average = records[records.model_kind == 'fixed_ema'].set_index('seed').to_dict('index')
            for metric in ('macro_auc', 'macro_logloss'):
                e = paired(average, online, metric)
                ema_effects.append(dict(role=role, dataset=dataset, checkpoint=checkpoint,
                    arm=arm, metric=metric, **{k:json.dumps(v) if isinstance(v, list) else v for k,v in e.items()}))
        for kind, records in group.groupby('model_kind'):
            reference = records[records.arm == 'A0'].set_index('seed')
            for _, record in records.iterrows():
                base = reference.loc[record.seed]
                gains = {f'd{d}t{t}': record[f'd{d}t{t}_auc']-base[f'd{d}t{t}_auc']
                         for d in range(3) for t in range(2)}
                worst = min(gains.values())
                minimum_gains.append(dict(role=role, dataset=dataset, checkpoint=checkpoint,
                    model_kind=kind, arm=record.arm, seed=record.seed,
                    minimum_paired_cell_gain=worst,
                    attaining_cells=','.join(k for k,v in gains.items() if v == worst),
                    difference_of_worst_auc=record.worst_cell_auc-base.worst_cell_auc))
    pd.DataFrame(cells).to_csv(out / 'cell_effects.csv', index=False)
    pd.DataFrame(ema_effects).to_csv(out / 'ema_effects.csv', index=False)
    pd.DataFrame(minimum_gains).to_csv(out / 'minimum_cell_gains.csv', index=False)
    epochs = pd.read_csv(attempt / 'epoch_metrics.csv')
    assert len(epochs) == 36 * 50 * 2
    epochs.to_csv(out / 'epoch_metrics.csv.gz', index=False, compression='gzip')
    stages = epochs[epochs.arm.eq('A0') & epochs.epoch.isin([1, 5, 10, 25, 50])]
    stages.to_csv(out / 'baseline_stages.csv', index=False)
    for name in ('run_ledger.csv', 'trial_ledger.csv', 'physical_cost_ledger.csv', 'failure_ledger.csv'):
        shutil.copyfile(attempt / name, out / name)
    selected = summary[summary.role.eq('test') & summary.model_kind.eq('fixed_ema')
                       & summary.checkpoint.eq('selected')]
    print(selected[['dataset', 'arm', 'macro_auc_mean', 'macro_auc_std',
                    'macro_logloss_mean', 'macro_logloss_std']].to_string(index=False))
    print('test_primary_against_A0')
    print(primary[primary.role.eq('test') & primary.reference.eq('A0')][
        ['dataset', 'arm', 'metric', 'values', 'mean', 'sample_sd', 'ci95', 'positive']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'tables')
    args = parser.parse_args()
    build(args.attempt, args.out)
