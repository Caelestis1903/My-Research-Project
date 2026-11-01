"""Plot completed A0/A1 validation trajectories and paired seed effects."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', required=True, type=Path)
args = parser.parse_args()
report = Path(__file__).resolve().parent
figures = report / 'figures'
figures.mkdir(exist_ok=True)
tables = []
patterns = ['runs/*/*/A0/seed*/attempt1/primary/validation_epoch_metrics.csv',
            'runs/development/*/A1/seed*/attempt1/primary/validation_epoch_metrics.csv',
            'runs/development/*/A1/seed*/attempt1/replay/validation_epoch_metrics.csv']
for pattern in patterns:
    for path in args.run_dir.glob(pattern):
        parts = path.relative_to(args.run_dir / 'runs').parts
        seed = int(parts[3].removeprefix('seed'))
        if seed not in (2022, 2024):
            continue
        table = pd.read_csv(path)
        tables.append(table.assign(dataset=parts[1], arm=parts[2], seed=seed))
curves = pd.concat(tables, ignore_index=True)
assert not curves.duplicated(['dataset', 'arm', 'seed', 'epoch', 'model_kind']).any()
assert curves.groupby(['dataset', 'arm', 'epoch', 'model_kind']).seed.nunique().eq(2).all()
styles = {'online': ('#777777', '-', 'Online'), 'fixed_ema': ('#0072B2', '-', 'Fixed EMA'),
          'delayed5': ('#D55E00', '--', 'Delayed EMA'), 'swa20': ('#009E73', '-', 'SWA E31+'),
          'adaptive_ema': ('#D55E00', '--', 'Adaptive EMA'), 'replay_rho': ('#009E73', ':', 'Rho replay')}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
for row, dataset in enumerate(('movielens', 'kuairand')):
    for column, arm in enumerate(('A0', 'A1')):
        ax = axes[row, column]
        part = curves[(curves.dataset == dataset) & (curves.arm == arm)]
        for kind, group in part.groupby('model_kind', sort=False):
            values = group.groupby('epoch').macro_auc.agg(['mean', 'std'])
            color, line, label = styles[kind]
            ax.plot(values.index, values['mean'], color=color, linestyle=line, label=label, linewidth=1.5)
            ax.fill_between(values.index, values['mean'] - values['std'], values['mean'] + values['std'],
                            color=color, alpha=.09)
        ax.axvline(10, color='#999999', linewidth=.8, linestyle=':')
        ax.set(title=f'{dataset.title()} / {arm}', xlabel='Epoch', ylabel='Validation macro AUC', xlim=(1, 50))
        ax.grid(alpha=.15)
        ax.legend(fontsize=8, loc='lower right')
fig.suptitle('Two development seeds: mean and sample SD; SWA begins at E31')
for suffix in ('png', 'pdf'):
    fig.savefig(figures / f'time_validation_curves.{suffix}', dpi=180)
plt.close(fig)

a0 = pd.read_csv(report / 'a_group_validation_paired.csv')
a1 = pd.read_csv(report / 'a1_c2_validation_endpoints.csv')
a0 = a0[(a0.checkpoint == 'selected') & (a0.metric == 'macro_auc')]
effects = []
for record in a0.to_dict('records'):
    effects.append({k: record[k] for k in ('dataset', 'seed', 'effect', 'delta')})
selected = a1[a1.checkpoint == 'selected'].set_index(['dataset', 'seed', 'model_kind'])
for dataset in ('movielens', 'kuairand'):
    for seed in (2022, 2024):
        for effect, right in [('adaptive-fixed', 'fixed_ema'), ('adaptive-replay', 'replay_rho')]:
            effects.append(dict(dataset=dataset, seed=seed, effect=effect,
                                delta=selected.loc[(dataset, seed, 'adaptive_ema'), 'macro_auc'] -
                                selected.loc[(dataset, seed, right), 'macro_auc']))
effects = pd.DataFrame(effects)
order = ['fixed-online', 'delayed-fixed', 'swa-fixed', 'adaptive-fixed', 'adaptive-replay']
labels = ['A0: fixed - online', 'A0: delayed - fixed', 'A0: SWA - fixed',
          'A1: adaptive - fixed', 'A1: adaptive - replay']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
for ax, dataset in zip(axes, ('movielens', 'kuairand')):
    part = effects[effects.dataset == dataset]
    for seed, color, offset in [(2022, '#0072B2', -.08), (2024, '#D55E00', .08)]:
        values = part[part.seed == seed].set_index('effect').loc[order, 'delta']
        ax.scatter(values, np.arange(len(order)) + offset, color=color, s=36, label=f'Seed {seed}', zorder=3)
    means = part.groupby('effect').delta.mean().loc[order]
    ax.scatter(means, np.arange(len(order)), marker='D', color='black', s=22, label='Mean', zorder=4)
    ax.axvline(0, color='#777777', linewidth=.8)
    ax.set(yticks=np.arange(len(order)), yticklabels=labels, xlabel='Paired selected validation AUC change',
           title=dataset.title())
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.15)
    ax.legend(fontsize=8)
fig.suptitle('Seed-level effects; paired t95% intervals are in the accompanying tables')
for suffix in ('png', 'pdf'):
    fig.savefig(figures / f'time_selected_validation_effects.{suffix}', dpi=180)
plt.close(fig)
print('Wrote two standalone figures in PNG and PDF.')

stability_path = report / 'b1_probability_stability.csv'
if stability_path.exists():
    stability = pd.read_csv(stability_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for ax, dataset in zip(axes, ('movielens', 'kuairand')):
        for seed, color in ((2022, '#0072B2'), (2024, '#D55E00')):
            group = stability[(stability.dataset == dataset) & (stability.seed == seed)].sort_values('epoch')
            assert len(group) == 50
            ax.plot(group.epoch, group.adjacent_epoch_correlation, color=color,
                    label=f'{seed}: adjacent epoch')
            ax.plot(group.epoch, group.epoch1_correlation, color=color, linestyle='--',
                    label=f'{seed}: E1 anchor')
        ax.set(title=dataset.title(), xlabel='Epoch', ylabel='Row-aligned probability correlation',
               xlim=(1, 50), ylim=(-.05, 1.05))
        ax.grid(alpha=.15)
        ax.legend(fontsize=8, ncols=2)
    fig.suptitle('B1: probability ranking continuity on the same target rows')
    for suffix in ('png', 'pdf'):
        fig.savefig(figures / f'b1_probability_stability.{suffix}', dpi=180)
    plt.close(fig)
    print('Wrote B1 probability stability in PNG and PDF.')
