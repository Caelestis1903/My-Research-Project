"""Seed uncertainty for all cells/marginals at globally frozen checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import v3_readout_20261016 as readout

base = readout.base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--aggregate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--joint-window', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    receipts = {group: [json.loads((Path(d) / 'receipt.json').read_text()) for d in directories]
                for group, directories in manifest['groups'].items()}
    assert all(r['status'] == 'COMPLETE' for rs in receipts.values() for r in rs)
    windows = {group: min(r['common_pre_degradation_epochs'] for r in rs)
               for group, rs in receipts.items()}
    if args.joint_window:
        windows = {group: min(windows.values()) for group in windows}
    trajectory = pd.read_csv(args.aggregate / 'domain_task_trajectory.csv')
    cells = pd.read_csv(args.aggregate / 'cell_trajectory.csv')
    keys = ['group', 'seed', 'split', 'arm', 'epoch']
    cells['cell'] = 'd' + cells.domain.astype(str) + 't' + cells.task.astype(str)
    wide = cells.pivot(index=keys, columns='cell', values=['auc', 'logloss'])
    wide.columns = [cell + '_' + metric for metric, cell in wide.columns]
    trajectory = trajectory.merge(wide.reset_index(), on=keys, validate='one_to_one')
    variables = [c for c in trajectory if c.endswith('_auc') or c.endswith('_logloss')]
    rows = []
    for group, rs in receipts.items():
        seeds = sorted(r['seed'] for r in rs)
        assert len(seeds) == len(set(seeds))
        assert sorted(trajectory[trajectory.group == group].seed.unique()) == seeds
        minimum_budget = min(int(trajectory[(trajectory.group == group) &
                              (trajectory.seed == seed)].epoch.max()) for seed in seeds)
        for receipt in rs:
            seed = receipt['seed']
            for split in ('selection', 'diagnostic'):
                frame = trajectory[(trajectory.group == group) &
                                   (trajectory.seed == seed) & (trajectory.split == split)]
                arms = {arm: frame[frame.arm == arm].set_index('epoch').sort_index()
                        for arm in ('SN', 'SB')}
                assert list(arms['SN'].index) == list(arms['SB'].index)
                assert list(arms['SN'].index) == list(range(1, len(arms['SN']) + 1))
                candidates = []
                for epoch in (5, 10, 15, 20, 30, 40, 50, 100):
                    if epoch <= minimum_budget:
                        candidates.append((f'common_epoch{epoch}',
                            {arm: arms[arm].loc[epoch, variables] for arm in arms},
                            {'SN': epoch, 'SB': epoch}))
                for label, horizon in (('AULC_1_10', 10), ('AULC_1_20', 20),
                                       ('common_pre_AULC', windows[group])):
                    candidates.append((label,
                        {arm: arms[arm].loc[1:horizon, variables].mean() for arm in arms},
                        {'SN': horizon, 'SB': horizon}))
                for rule, epochs in receipt['selected_epochs'].items():
                    candidates.append((f'selected_{rule}',
                        {arm: arms[arm].loc[epochs[arm], variables] for arm in arms}, epochs))
                epochs = receipt['selected_epochs']['macro_auc']
                for choosing_arm in ('SN', 'SB'):
                    epoch = epochs[choosing_arm]
                    candidates.append((f'common_{choosing_arm}_selected_macro_auc',
                        {arm: arms[arm].loc[epoch, variables] for arm in arms},
                        {'SN': epoch, 'SB': epoch}))
                for label, values, epochs in candidates:
                    for variable in variables:
                        sn, sb = float(values['SN'][variable]), float(values['SB'][variable])
                        rows.append(dict(group=group, seed=seed, split=split, readout=label,
                            variable=variable, SN=sn, SB=sb, difference=sb-sn,
                            SN_epoch_or_horizon=epochs['SN'], SB_epoch_or_horizon=epochs['SB']))
    result = pd.DataFrame(rows)
    result.to_csv(args.out / 'seed_readouts.csv', index=False)
    summaries = []
    for key, frame in result.groupby(['group', 'split', 'readout', 'variable']):
        assert not frame.seed.duplicated().any(), key
        group = key[0]
        assert len(frame) == len(receipts[group]), key
        summaries.append(dict(zip(('group', 'split', 'readout', 'variable'), key)) |
            readout.paired_summary(frame.sort_values('seed').difference) |
            dict(SN_mean=float(frame.SN.mean()), SB_mean=float(frame.SB.mean())))
    base.dump(args.out / 'summary.json', summaries)
    base.dump(args.out / 'receipt.json', dict(status='COMPLETE', source_sha=base.sha(__file__),
        manifest_sha=base.sha(args.manifest), windows=windows, joint_window=args.joint_window,
        aggregate_sources={name: base.sha(args.aggregate / name) for name in
            ('receipt.json', 'domain_task_trajectory.csv', 'cell_trajectory.csv')},
        scope='Globally frozen checkpoints applied to every cell. Seed CIs do not include '
              'outer LR-selection uncertainty; adaptive final budgets retain explicit epochs. '
              'No per-cell checkpoint selection, test prediction or training change.'))


if __name__ == '__main__':
    main()
