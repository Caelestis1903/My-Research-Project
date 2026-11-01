"""Figures from frozen, completed numerical tables; development seeds stay separate."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
TABLE = OUT / 'tables'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 180, 'pdf.fonttype': 42})


def save(fig, name):
    fig.savefig(FIG / (name + '.png'), bbox_inches='tight')
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    paired = pd.read_csv(TABLE / 'paired_effects.csv')
    effects = [f'{a}-A1:fixed_ema' for a in ('B1', 'B2', 'B3')] + [
        f'{a}-{r}:fixed_ema' for a, r in (('B1', 'R1'), ('B2', 'R2'), ('B3', 'R3'))]
    selected = paired[(paired.checkpoint == 'selected') & (paired.subset == 'development') &
                      (paired.metric == 'macro_auc')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for i, dataset in enumerate(('movielens', 'kuairand')):
        for j, split in enumerate(('validation', 'test')):
            ax = axes[i, j]
            part = selected[(selected.dataset == dataset) & (selected.split == split)]
            for y, effect in enumerate(effects):
                group = part[part.effect == effect].sort_values('seed')
                if group.empty:
                    continue
                ax.plot(group.delta, [y] * len(group), 'o', color='#7089ac', alpha=.8, ms=5)
                ax.plot(group.delta.mean(), y, 'D', color='#142c4d', ms=6)
            ax.axvline(0, color='#888888', lw=.8)
            ax.axvline(.001, color='#cf8a30', lw=.8, ls=':')
            ax.set_yticks(range(len(effects)), [e.replace(':fixed_ema', '') for e in effects])
            ax.set_title(f'{dataset.title()} | {split} | development n=2')
            ax.grid(axis='x', alpha=.2)
            ax.invert_yaxis()
            ax.set_xlabel('Paired selected macro AUC difference')
    fig.suptitle('Fixed EMA: seed differences (circles) and mean (diamond)\nMissing controls were not dispatched; test does not change promotion', y=1.02)
    fig.tight_layout()
    save(fig, 'final_development_effects')

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    cells = [f'd{d}t{t}_auc' for d in range(3) for t in range(2)]
    heat = paired[(paired.checkpoint == 'selected') & (paired.subset == 'development')]
    arrays = []
    for dataset in ('movielens', 'kuairand'):
        for split in ('validation', 'test'):
            part = heat[(heat.dataset == dataset) & (heat.split == split)]
            arrays.append(np.array([[part[(part.effect == effect) & (part.metric == cell)].delta.mean()
                                     for cell in cells] for effect in effects]))
    bound = max(np.nanmax(np.abs(a)) for a in arrays)
    for ax, arr, (dataset, split) in zip(axes.flat, arrays,
            [(d, s) for d in ('movielens', 'kuairand') for s in ('validation', 'test')]):
        im = ax.imshow(arr, cmap='RdBu', vmin=-bound, vmax=bound, aspect='auto')
        ax.set_xticks(range(6), [c.replace('_auc', '') for c in cells])
        ax.set_yticks(range(len(effects)), [e.replace(':fixed_ema', '') for e in effects])
        ax.set_title(f'{dataset.title()} | {split} | development n=2')
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                if np.isfinite(arr[y, x]):
                    ax.text(x, y, f'{arr[y, x]:+.3f}', ha='center', va='center', fontsize=8,
                            color='white' if abs(arr[y, x]) > bound * .6 else 'black')
                else:
                    ax.text(x, y, 'N/A', ha='center', va='center', fontsize=8, color='#666666')
    fig.suptitle('Selected fixed EMA: six-cell paired AUC differences\nMeans of two seed differences; colors do not encode uncertainty', y=1.02)
    fig.tight_layout(rect=(0, 0, .92, 1))
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.025, pad=.04, label='Mean paired AUC difference')
    save(fig, 'final_domain_task_effects')

    confirmed = paired[(paired.checkpoint == 'selected') & (paired.subset == 'confirmation') &
                       (paired.metric == 'macro_auc') & paired.effect.isin(effects)]
    if not confirmed.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for y, ((dataset, split, effect), group) in enumerate(confirmed.groupby(['dataset', 'split', 'effect'])):
            ax.plot(group.delta, [y] * len(group), 'o', color='#7089ac')
            ax.plot(group.delta.mean(), y, 'D', color='#142c4d')
            ax.text(0, y, '', ha='left')
        labels = [f'{d} {s} {e} (n={len(g)})' for (d, s, e), g in confirmed.groupby(['dataset', 'split', 'effect'])]
        ax.set_yticks(range(len(labels)), labels)
        ax.axvline(0, color='#888888', lw=.8)
        ax.set_xlabel('Paired selected macro AUC difference')
        ax.set_title('Confirmation seeds only (2023, 2025, 2026)')
        fig.tight_layout()
        save(fig, 'final_confirmation_effects')


if __name__ == '__main__':
    main()
