"""Publication figures from the report's numerical tables."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
ARMS = ['A1', 'J0', 'J1', 'F0', 'F1', 'RG', 'AG', 'TG']
NAMES = {'movielens': 'MovieLens-1M', 'kuairand': 'KuaiRand'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=200, bbox_inches='tight')
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


effects = pd.read_csv(HERE / 'tables/primary_effects.csv')
effects = effects[(effects.role == 'test') & (effects.reference == 'A0') & (effects.metric == 'macro_auc')]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
for ax, dataset in zip(axes, NAMES):
    group = effects[effects.dataset == dataset].set_index('arm')
    for y, arm in enumerate(ARMS):
        row = group.loc[arm]
        values = json.loads(row['values'])
        assert len(values) == 2
        ax.scatter(values[0], y, marker='o', color='#2878a8', s=28, label='2022' if y == 0 else None)
        ax.scatter(values[1], y, marker='^', color='#df8f44', s=30, label='2024' if y == 0 else None)
        ax.scatter(row['mean'], y, marker='|', color='black', s=120, label='Mean' if y == 0 else None)
    ax.axvline(0, color='black', lw=.8)
    ax.axvline(.001, color='#42926b', ls='--', lw=.8, label='Practical gate')
    ax.set(yticks=range(len(ARMS)), yticklabels=ARMS, title=NAMES[dataset] + ' (development n=2)',
           xlabel='Selected test macro AUC difference vs A0 + EMA')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
axes[0].legend(fontsize=8)
save(fig, 'selected_seed_effects')

cells = pd.read_csv(HERE / 'tables/cell_effects.csv')
cells = cells[(cells.role == 'test') & (cells.checkpoint == 'selected')]
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
for ax, dataset in zip(axes, NAMES):
    matrix = cells[cells.dataset == dataset].pivot(index='arm', columns='cell', values='delta_auc').loc[ARMS]
    limit = max(float(np.abs(matrix.to_numpy()).max()), .001)
    im = ax.imshow(matrix, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    for i in range(len(matrix)):
        for j in range(6):
            value = matrix.iloc[i, j]
            ax.text(j, i, f'{value:+.4f}' if limit < .01 else f'{value:+.3f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(value) > .55 * limit else 'black')
    ax.set(xticks=range(6), xticklabels=matrix.columns, yticks=range(len(ARMS)), yticklabels=ARMS,
           title=NAMES[dataset] + ': mean test cell AUC difference')
    fig.colorbar(im, ax=ax, shrink=.75, label='Difference vs A0 + EMA (panel-specific scale)')
save(fig, 'cell_effects')

epochs = pd.read_csv(HERE / 'tables/epoch_metrics.csv.gz')
epochs = epochs[epochs.arm == 'A0']
fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True)
for col, dataset in enumerate(NAMES):
    for row, metric in enumerate(('macro_auc', 'macro_logloss')):
        ax = axes[row, col]
        for kind, color, label in [('online', '#2878a8', 'Online'), ('fixed_ema', '#df8f44', 'Fixed EMA')]:
            group = epochs[(epochs.dataset == dataset) & (epochs.model_kind == kind)]
            values = group.groupby('epoch')[metric].agg(['mean', 'min', 'max'])
            ax.plot(values.index, values['mean'], color=color, label=label)
            ax.fill_between(values.index, values['min'], values['max'], color=color, alpha=.14)
        ax.set(xlabel='Epoch', ylabel='Validation ' + metric.replace('_', ' '))
        ax.grid(alpha=.2)
        if row == 0:
            ax.set_title(NAMES[dataset] + ': A0, mean and seed range (n=2)')
        ax.legend(fontsize=8)
save(fig, 'baseline_training_stages')
print('Wrote three PNG/PDF figure pairs')
