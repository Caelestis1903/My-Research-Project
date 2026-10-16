"""Descriptive initial-budget plots; individual seeds retained, no inference gates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import v3_readout_20261016 as r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    curves, sources, early = {}, [], []
    for group, directory in manifest['groups'].items():
        directory = Path(directory)
        receipt = json.loads((directory / 'receipt.json').read_text())
        assert receipt['status'] == 'COMPLETE'
        csv = directory / 'domain_task_trajectory.csv'
        frame = pd.read_csv(csv)
        frame = frame[(frame.split == 'diagnostic') & (frame.epoch <= 50)]
        assert sorted(frame.seed.unique().tolist()) == [2022, 2023, 2024, 2025]
        for metric in ('auc', 'logloss'):
            pivot = frame.pivot(index=['seed', 'epoch'], columns='arm',
                                values=f'macro_{metric}')
            gap = (pivot.SB - pivot.SN).unstack('epoch').sort_index()
            assert gap.shape == (4, 50) and not gap.isna().any().any()
            curves[group, metric] = gap
        for variable in ('d0_auc', 'd1_auc', 'd2_auc', 't0_auc', 't1_auc'):
            pivot = frame[frame.epoch <= 22].pivot(index=['seed', 'epoch'],
                        columns='arm', values=variable)
            values = (pivot.SB - pivot.SN).groupby('seed').mean()
            for seed, value in values.items():
                early.append(dict(group=group, seed=int(seed), variable=variable,
                                  difference=float(value), horizon=22))
        sources.append(dict(group=group, directory=str(directory),
                            receipt_sha=r.base.sha(directory / 'receipt.json'),
                            trajectory_sha=r.base.sha(csv)))
    args.out.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, ((group, _), color) in zip(axes.flat,
            zip(manifest['groups'].items(), plt.get_cmap('tab10').colors)):
        values = curves[group, 'auc']
        for seed, row in values.iterrows():
            axis.plot(values.columns, row, color=color, alpha=.28, linewidth=.8)
        axis.plot(values.columns, values.mean(), color=color, linewidth=2,
                  label='4-seed mean; thin lines: individual seeds')
        axis.axhline(0, color='black', linewidth=.6)
        axis.axvline(22, color='gray', linestyle=':', linewidth=.8)
        axis.set(title=group, ylabel='Diagnostic macro AUC: SB - SN', xlabel='Epoch')
        axis.grid(alpha=.15)
        axis.legend(fontsize=7)
    fig.suptitle('Initial 50 epochs; selected endpoints and adequate-budget results separate')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(args.out / f'initial50_auc_trajectories.{suffix}', dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4))
    for group in manifest['groups']:
        values = curves[group, 'logloss']
        line, = axis.plot(values.columns, values.mean(), label=group)
        axis.fill_between(values.columns, values.min(), values.max(),
                          alpha=.10, color=line.get_color())
    axis.axhline(0, color='black', linewidth=.6)
    axis.set(xlabel='Epoch', ylabel='Diagnostic macro logloss: SB - SN',
             title='Mean and observed seed range (range is not a confidence interval)')
    axis.grid(alpha=.15)
    axis.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(args.out / f'initial50_logloss_trajectories.{suffix}', dpi=180)
    plt.close(fig)
    pd.DataFrame(early).to_csv(args.out / 'joint22_domain_task_seed_gaps.csv', index=False)
    r.base.dump(args.out / 'receipt.json', dict(status='COMPLETE',
        source_sha=r.base.sha(__file__), manifest_sha=r.base.sha(args.manifest),
        sources=sources, independent_review=False,
        scope='Descriptive initial50 diagnostic plots, thin individual seed trajectories; '
              'logloss band is observed seed range, not CI. No selected-epoch/budget '
              'equivalence claim, no condition/extension decisions or test access.'))


if __name__ == '__main__':
    main()
