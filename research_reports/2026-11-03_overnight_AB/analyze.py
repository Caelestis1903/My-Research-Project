"""Render the completed five-seed study's selected comparisons; no training."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (2022, 2023, 2024, 2025, 2026)
DATASETS = ('movielens', 'kuairand')
CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))
CP = 'C_MIX_P01_SERIAL_V1'
REFS = (('A0', 'online', 'M3 online'), ('A0', 'fixed_ema', 'A0 EMA'),
        ('C_MIX', 'fixed_ema', 'C'), ('P01', 'fixed_ema', 'P'),
        ('HYPERGATE', 'online', 'HG'), ('AUTOIFS', 'online', 'AI'),
        ('GAUSSIAN_A1', 'fixed_ema', 'Gaussian'))


def summary(values):
    values = np.asarray(values, dtype=float)
    assert values.shape == (5,) and np.isfinite(values).all()
    mean, sd = math.fsum(values)/5, float(values.std(ddof=1))
    radius = 2.7764451051977987 * sd / np.sqrt(5)
    return dict(values=values.tolist(), mean=mean, sd=sd,
                ci95=[mean-radius, mean+radius], positive=int((values > 0).sum()),
                negative=int((values < 0).sum()))


def main():
    with (HERE/'ENDPOINTS.csv').open() as f:
        all_rows = [r for r in csv.DictReader(f) if r['split'] == 'test']
    rows = [r for r in all_rows if r['checkpoint'] == 'selected']
    index = {}
    for row in rows:
        key = (row['dataset'], row['arm'], int(row['seed']), row['estimator'])
        assert key not in index, f'Duplicate selected endpoint: {key}'
        index[key] = row
    def vector(dataset, arm, estimator, metric):
        return np.array([float(index[dataset, arm, seed, estimator][metric]) for seed in SEEDS])
    # The complete protocol is required before any result figure is written.
    for dataset in DATASETS:
        for arm, estimator in [(CP, 'fixed_ema'), *[(a, e) for a, e, _ in REFS]]:
            for seed in SEEDS:
                assert (dataset, arm, seed, estimator) in index, f'Incomplete: {dataset}/{arm}/{seed}'
    selected, comparisons, interaction = {}, {}, {}
    for dataset in DATASETS:
        selected[dataset], comparisons[dataset] = {}, {}
        for arm, estimator in [(CP, 'fixed_ema'), (CP, 'online'), *[(a, e) for a, e, _ in REFS]]:
            selected[dataset][f'{arm}/{estimator}'] = dict(
                auc=summary(vector(dataset, arm, estimator, 'macro_auc')),
                logloss=summary(vector(dataset, arm, estimator, 'macro_logloss')),
                epochs=[int(index[dataset, arm, seed, estimator]['epoch']) for seed in SEEDS])
        for arm, estimator, label in REFS:
            metrics = ('macro_auc', 'macro_logloss', 'sample_weighted_cell_auc',
                       'sample_weighted_cell_logloss', 'pooled_task_mean_auc', 'pooled_logloss',
                       *[f'{c}_{m}' for c in CELLS for m in ('auc', 'logloss')],
                       *[f'task{t}_macro_{m}' for t in range(2) for m in ('auc', 'logloss')])
            delta = {m: summary(vector(dataset, CP, 'fixed_ema', m)-vector(dataset, arm, estimator, m))
                     for m in metrics}
            delta['macro_auc_pass'] = delta['macro_auc']['mean'] >= .003
            delta['cell_protection_pass'] = all(delta[f'{c}_auc']['mean'] >= -.001 for c in CELLS)
            delta['logloss_improved'] = delta['macro_logloss']['mean'] < 0
            comparisons[dataset][label] = delta
        interaction[dataset] = {m: summary(vector(dataset, CP, 'fixed_ema', m)
            - vector(dataset, 'C_MIX', 'fixed_ema', m) - vector(dataset, 'P01', 'fixed_ema', m)
            + vector(dataset, 'A0', 'fixed_ema', m)) for m in ('macro_auc', 'macro_logloss')}

    full_index = {(r['dataset'], r['arm'], int(r['seed']), r['estimator'], r['checkpoint']): r
                  for r in all_rows}
    assert len(full_index) == len(all_rows)
    stage_rows = []
    for dataset in DATASETS:
        for checkpoint in ('selected', 'E10', 'E50'):
            for arm, estimator, label in REFS:
                for seed in SEEDS:
                    candidate = full_index[dataset, CP, seed, 'fixed_ema', checkpoint]
                    reference = full_index[dataset, arm, seed, estimator, checkpoint]
                    row = dict(dataset=dataset, checkpoint=checkpoint, seed=seed,
                               reference=label, candidate_epoch=candidate['epoch'],
                               reference_epoch=reference['epoch'], candidate_hash=candidate['model_hash'],
                               reference_hash=reference['model_hash'])
                    for metric in ('macro_auc', 'macro_logloss', 'sample_weighted_cell_auc',
                                   'sample_weighted_cell_logloss', 'pooled_task_mean_auc', 'pooled_logloss',
                                   *[f'{c}_{m}' for c in CELLS for m in ('auc', 'logloss')]):
                        row[f'delta_{metric}'] = float(candidate[metric])-float(reference[metric])
                    stage_rows.append(row)
    with (HERE/'STAGE_EFFECTS.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)

    figures = HERE/'figures'
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), constrained_layout=True)
    colors = plt.cm.tab10(np.arange(5))
    labels = [r[2] for r in REFS]
    for di, dataset in enumerate(DATASETS):
        for mi, metric in enumerate(('macro_auc', 'macro_logloss')):
            ax = axes[di, mi]
            for j, label in enumerate(labels):
                s = comparisons[dataset][label][metric]
                vals = np.array(s['values'])*1000
                ax.scatter(j+np.linspace(-.15, .15, 5), vals, c=colors, s=23,
                           alpha=.85, zorder=3)
                ax.errorbar(j, s['mean']*1000,
                    yerr=np.array([[s['mean']-s['ci95'][0]], [s['ci95'][1]-s['mean']]])*1000,
                    fmt='D', color='black', markersize=4, capsize=4, zorder=4)
            ax.axhline(0, color='#555555', linewidth=.8)
            if mi == 0:
                ax.axhline(3, color='#238443', linestyle='--', linewidth=1)
            ax.set_xticks(range(len(labels)), labels)
            ax.set_title(f'{dataset}: CP fixed EMA minus reference')
            ax.set_ylabel(('Delta AUC' if mi == 0 else 'Delta LogLoss')+' (x 1e-3)')
            ax.grid(axis='y', alpha=.2)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=colors[i], label=str(s))
               for i, s in enumerate(SEEDS)]
    fig.legend(handles=handles, loc='outside lower center', ncol=5, frameon=False)
    fig.savefig(figures/'paired_selected.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    for di, dataset in enumerate(DATASETS):
        for mi, metric in enumerate(('auc', 'logloss')):
            ax = axes[di, mi]
            values = np.array([[comparisons[dataset][label][f'{c}_{metric}']['mean']*1000
                                for c in CELLS] for label in labels])
            limit = max(float(abs(values).max()), .1)
            im = ax.imshow(values, cmap='RdYlGn' if mi == 0 else 'RdYlGn_r',
                           vmin=-limit, vmax=limit, aspect='auto')
            for i in range(len(labels)):
                for j in range(6):
                    ax.text(j, i, f'{values[i,j]:+.2f}', ha='center', va='center', fontsize=8)
                    if mi == 0 and values[i, j] < -1:
                        ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, linewidth=2, color='black'))
            ax.set_xticks(range(6), CELLS)
            ax.set_yticks(range(len(labels)), labels)
            ax.set_title(f'{dataset}: delta {metric} (x 1e-3)')
            fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
    fig.savefig(figures/'six_cell_deltas.png', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(selected=selected, comparisons=comparisons, interaction=interaction),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
