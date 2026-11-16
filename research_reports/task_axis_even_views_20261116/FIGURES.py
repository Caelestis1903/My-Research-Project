"""Static research figures from the completed validation exports."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TABLES = HERE.parents[1]/'artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1/tables'
p = pd.read_csv(HERE/'PRIMARY_PAIRED.csv')
d = pd.read_csv(TABLES/'PAIRED_SEEDS.csv')
d = d[d.endpoint.eq('selected') & d.kind.eq('fixed_ema') & d.metric.eq('macro_auc')]
controls = ['CLEAN', 'NDP-A', 'ISOTROPIC-EVEN', 'AXIS-IID2']
fig, axes = plt.subplots(2, 4, figsize=(12, 5), layout='constrained')
for row, dataset in enumerate(['movielens', 'kuairand']):
    for col, control in enumerate(controls):
        ax = axes[row, col]
        q = p[p.dataset.eq(dataset) & p.control.eq(control) & p.metric.eq('macro_auc')].iloc[0]
        values = d[d.dataset.eq(dataset) & d.control.eq(control)].sort_values('seed').delta.to_numpy()*1000
        ax.scatter(values, np.linspace(.12, .40, 5), color='#777777', s=22)
        ax.errorbar(q['mean']*1000, 0, xerr=np.array([
            [(q['mean']-q.ci95_low)*1000], [(q.ci95_high-q['mean'])*1000]]),
            fmt='D', color='#225b88', capsize=4)
        ax.axvline(0, color='#b04b3b', linewidth=.9)
        ax.set(yticks=[], ylim=(-.15, .55), xlabel='AUC delta x 1000', title=f'{dataset}: vs {control}')
        ax.grid(axis='x', alpha=.18)
fig.suptitle('Selected EMA, AXIS-EVEN minus control: five seed dots and descriptive 95% CI')
fig.savefig(HERE/'PAIRED_AUC.png', dpi=180)
plt.close(fig)

e = pd.read_csv(TABLES/'ENDPOINTS.csv')
e = e[e.endpoint.eq('selected') & e.kind.eq('fixed_ema')]
cells = [f'd{domain}t{task}' for domain in range(3) for task in range(2)]
arms = ['AXIS-EVEN', 'AXIS-IID2', 'ISOTROPIC-EVEN']
matrices = []
for dataset in ['movielens', 'kuairand']:
    rows = e[e.dataset.eq(dataset)].groupby('arm')[[f'{c}_auc' for c in cells]].mean()
    matrices.append(np.array([rows.loc[a]-rows.loc['CLEAN'] for a in arms])*1000)
limit = max(float(np.abs(m).max()) for m in matrices)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), layout='constrained')
for ax, dataset, values in zip(axes, ['movielens', 'kuairand'], matrices):
    im = ax.imshow(values, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    ax.set(xticks=range(6), xticklabels=cells, yticks=range(3), yticklabels=arms, title=dataset)
    for i in range(3):
        for j in range(6):
            ax.text(j, i, f'{values[i,j]:+.2f}', ha='center', va='center',
                    color='white' if abs(values[i,j]) > limit*.55 else 'black', fontsize=9)
fig.colorbar(im, ax=axes, label='Mean cell AUC delta vs CLEAN x 1000')
fig.suptitle('Five-seed selected EMA cell means')
fig.savefig(HERE/'CELL_AUC_DELTAS.png', dpi=180)
plt.close(fig)
