"""Reproduce the report's descriptive tables and figures from frozen scalar records."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ATTEMPT = ROOT / 'artifacts/mdmt_history_information_20261103/attempt1'
ARMS = ['A0f', 'A1', 'M', 'T0', 'TM', 'H_REAL', 'H_MIX', 'C_MIX', 'SCALE']
ep = pd.read_csv(HERE / 'endpoints.csv')
cells = pd.read_csv(HERE / 'six_cell.csv')
epochs = pd.read_csv(HERE / 'validation_epochs.csv')
assert len(ep) == 252 and len(cells) == 1512 and ep.key.nunique() == 21
assert not ep.duplicated(['key', 'split', 'model_kind', 'checkpoint']).any()
keys = ['dataset', 'seed', 'split', 'model_kind', 'checkpoint']
metrics = [c for c in ep if c.endswith(('_auc', '_logloss'))]
base = ep[ep.arm.eq('A0f')][keys + metrics]
delta = ep.merge(base, on=keys, suffixes=('', '_A0f'), validate='many_to_one')
for m in metrics:
    delta[m + '_delta'] = delta[m] - delta[m + '_A0f']
delta.to_csv(HERE / 'comparisons_to_A0.csv', index=False)

group = ['key', 'dataset', 'seed', 'arm', 'split', 'model_kind', 'checkpoint']
axis_rows = []
for axis in ('domain', 'task'):
    x = cells.groupby(group + [axis], as_index=False)[['auc', 'logloss']].mean()
    x = x.rename(columns={axis: 'index'})
    x['aggregation_axis'] = axis
    axis_rows.append(x)
pd.concat(axis_rows, ignore_index=True).to_csv(HERE / 'domain_task_summary.csv', index=False)

diag_rows = []
for result_file in sorted((ATTEMPT / 'runs').glob('*/result.json')):
    result = json.loads(result_file.read_text())
    if result.get('provenance'):
        continue
    j = result['job']
    samples = []
    for file in sorted(result_file.parent.glob('attempt*/epoch_*.json')):
        e = json.loads(file.read_text())
        samples.extend(e['diagnostics'])
    row = dict(dataset=j['dataset'], arm=j['config']['arm'], seed=j['seed'], sampled_steps=len(samples))
    for field in ('constraints', 'projection_distance', 'history_norm_ratio', 'parameter_distance',
                  'delta_over_gradient', 'transport_clip_factor', 'transport_moment_ratio'):
        values = [s[field] for s in samples if field in s]
        if values:
            row[field + '_mean'] = float(np.mean(values))
            if field in ('constraints', 'projection_distance'):
                row[field + '_positive_fraction'] = float(np.mean(np.array(values) > 1e-12))
            if field == 'transport_clip_factor':
                row['transport_clipped_fraction'] = float(np.mean(np.array(values) < 1))
    if j['config']['arm'] in ('M', 'TM', 'H_MIX', 'C_MIX'):
        for task in ('0', '1'):
            for level in ('1', '2', '3'):
                values = [s['task_support_rates'][task][level] for s in samples if 'task_support_rates' in s]
                if values:
                    row[f'task{task}_support{level}_mean'] = float(np.mean(values))
    diag_rows.append(row)
pd.DataFrame(diag_rows).to_csv(HERE / 'sampled_diagnostics.csv', index=False)

training = epochs[epochs.model_kind.eq('online') & epochs.epoch.isin([1, 10, 50])]
training[['key', 'dataset', 'seed', 'arm', 'epoch', 'training_loss', 'training_real_loss']].to_csv(HERE / 'training_loss_endpoints.csv', index=False)

figdir = HERE / 'figures'
figdir.mkdir(exist_ok=True)
selected = delta[delta.model_kind.eq('fixed_ema') & delta.checkpoint.eq('selected') & delta.seed.eq(2022)]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
for ax, dataset in zip(axes, ('movielens', 'kuairand')):
    x = selected[selected.dataset.eq(dataset)]
    arms = [a for a in ARMS[1:] if a in set(x.arm)]
    for i, split in enumerate(('validation', 'test')):
        y = x[x.split.eq(split)].set_index('arm').loc[arms, 'macro_auc_delta']
        ax.bar(np.arange(len(arms)) + (i - .5) * .36, y, width=.36, label=split)
    ax.axhline(0, color='black', linewidth=.7)
    ax.set_xticks(np.arange(len(arms)), arms, rotation=40, ha='right')
    ax.set_title(dataset + ' | seed 2022 | fixed EMA selected')
    ax.set_ylabel('Macro AUC difference vs A0f')
    ax.legend()
fig.savefig(figdir / 'selected_auc.png', dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
cell_metrics = [f'd{d}t{t}_auc_delta' for d in range(3) for t in range(2)]
test = selected[selected.split.eq('test')]
fig.suptitle('Fixed EMA | validation-selected checkpoints')
scale = max(abs(test[cell_metrics].to_numpy()).max(), 1e-6)
for ax, dataset in zip(axes, ('movielens', 'kuairand')):
    x = test[test.dataset.eq(dataset)].set_index('arm')
    arms = [a for a in ARMS[1:] if a in x.index]
    z = x.loc[arms, cell_metrics].to_numpy()
    im = ax.imshow(z, cmap='RdBu', vmin=-scale, vmax=scale, aspect='auto')
    ax.set_xticks(range(6), [m[:4] for m in cell_metrics])
    ax.set_yticks(range(len(arms)), arms)
    ax.set_title(dataset + ' | test AUC difference | seed 2022')
    for i in range(len(arms)):
        for k in range(6):
            ax.text(k, i, f'{z[i,k]:+.3f}', ha='center', va='center', fontsize=8)
fig.colorbar(im, ax=axes, label='AUC difference vs A0f')
fig.savefig(figdir / 'six_cell_auc.png', dpi=170)
plt.close(fig)

def markdown(frame):
    def value(v):
        return f'{v:.6f}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] + ['| ' + ' | '.join(value(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])

parts = ['# 数值附录', '', '所有数值来自冻结端点。新候选仅seed 2022；seed 2024的A0f/A1为历史引用。AUC越高越好，LogLoss越低越好。单seed差没有样本SD或t区间；完整统计见paired_seed.csv和paired_summary.csv。', '', 'online/EMA × selected/E10/E50 × validation/test的全部252行见endpoints.csv；全部1512个域任务格见six_cell.csv；每epoch指标见validation_epochs.csv。', '']
for dataset in ('movielens', 'kuairand'):
    for split in ('validation', 'test'):
        x = delta[delta.dataset.eq(dataset) & delta.split.eq(split) & delta.model_kind.eq('fixed_ema') & delta.checkpoint.eq('selected')].sort_values(['seed', 'arm'])
        cols = ['seed', 'arm', 'epoch', 'macro_auc', 'macro_auc_delta', 'macro_logloss', 'macro_logloss_delta', 'sample_weighted_cell_auc', 'pooled_task_mean_auc']
        parts += [f'## {dataset} / {split} / fixed EMA selected', '', markdown(x[cols]), '']
        for metric in ('auc', 'logloss'):
            cols = ['seed', 'arm'] + [f'd{d}t{t}_{metric}' for d in range(3) for t in range(2)]
            parts += [f'### 六格 {metric}', '', markdown(x[cols]), '']
(HERE / 'NUMERICAL_APPENDIX.md').write_text('\n'.join(parts) + '\n')
print(json.dumps(dict(endpoints=len(ep), cells=len(cells), trajectories=ep.key.nunique(), new_trajectories=len(diag_rows), figures=2)))
