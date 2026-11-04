"""Render seed-level numerical appendices and figures from completed scalar exports."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
METHODS = ['C_MIX', 'H_MIX', 'P01', 'M', 'Gaussian']
DATASETS = ['movielens', 'kuairand']
CELLS = [f'd{d}t{t}' for d in range(3) for t in range(2)]


def table(frame):
    columns = list(frame.columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---'] * len(columns)) + '|']
    lines.extend('| ' + ' | '.join(str(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def main():
    completion = json.loads((HERE / 'COMPLETION.json').read_text())
    assert completion['status'] == 'COMPLETE' and completion['new_runs'] == 30
    endpoints = pd.read_csv(HERE / 'endpoints.csv')
    paired = pd.read_csv(HERE / 'paired_seed.csv')
    assert len(endpoints) == 660
    selected = endpoints.query("split == 'test' and checkpoint == 'selected'").copy()
    primary = paired.query("split == 'test' and checkpoint == 'selected' and model_kind == 'fixed_ema'").copy()
    assert primary.groupby(['dataset', 'arm', 'comparison']).seed.nunique().eq(5).all()
    summary = paired.query("split == 'test'").groupby(
        ['dataset', 'arm', 'model_kind', 'checkpoint', 'comparison'])[
        [c for c in paired.columns if c.startswith('delta_')]].mean().reset_index()
    summary.to_csv(HERE / 'stage_effects.csv', index=False)
    decisions = json.loads((HERE / 'DECISIONS.json').read_text())
    lookup = {(r['dataset'], r['arm'], r['comparison']): r for r in decisions}
    outcomes = []
    for dataset in DATASETS:
        for arm in METHODS:
            total, component = [lookup[(dataset, arm, comparison)] for comparison in ['total', 'component']]
            outcomes.append(dict(dataset=dataset, arm=arm,
                total_auc=total['mean_delta_auc'], component_auc=component['mean_delta_auc'],
                total_logloss=total['mean_delta_logloss'], component_logloss=component['mean_delta_logloss'],
                auc_status=total['auc_status'],
                violated_cells=json.dumps(total['violated_cells'], ensure_ascii=False),
                total_ll_decreased=total['logloss_decreased'],
                component_ll_decreased=component['logloss_decreased']))
    outcomes = pd.DataFrame(outcomes)
    outcomes.to_csv(HERE / 'selected_outcomes.csv', index=False)

    appendix = ['# 五 seed 数值附录', '',
        '主端点为各估计器独立 validation 选点后的 test；AUC/LogLoss 同列，seed 为重复单位。', '',
        '全部 online/EMA、selected/E10/E50 与 validation/test 原值见 [endpoints.csv](endpoints.csv)；'
        '逐 seed 差和样本 SD、配对 t95% 区间见 [paired_seed.csv](paired_seed.csv)、[paired_summary.csv](paired_summary.csv)。', '']
    ledger = ['### 11/03 五 seed 补全：固定 M3oE 内部候选与控制', '',
        '协议：原行序自然80/10/10、train-only预处理、fixed1 M3oE、batch40960、50epoch；'
        'Adam lr=.01、coupled wd ML1e−4/KR1e−5；EMA ρ ML=.9828205985452511/KR=.995。'
        'selected按validation六格macro AUC最早严格最大值。每行n=5，种子2022–2026。'
        '本组为内部候选评价；与10/29发表模型的不同人口、选点结果分组保留。', '',
        '[报告](research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md) · '
        '[全部逐seed端点](research_reports/2026-11-03_mdmt_five_seed_completion/endpoints.csv) · '
        '[选点和来源](research_reports/2026-11-03_mdmt_five_seed_completion/RUN_MANIFEST.json)', '']
    metric_columns = [f'{cell}_{metric}' for cell in CELLS for metric in ['auc', 'logloss']]
    metric_columns += ['macro_auc', 'macro_logloss', 'sample_weighted_cell_auc',
                       'sample_weighted_cell_logloss', 'pooled_task_mean_auc', 'pooled_logloss']
    means = endpoints.query("split == 'test'").groupby(
        ['dataset', 'arm', 'model_kind', 'checkpoint'])[metric_columns].mean().reset_index()
    means.to_csv(HERE / 'absolute_test_means.csv', index=False)
    for dataset in DATASETS:
        appendix += [f'## {dataset}：selected逐seed', '']
        rows = []
        for _, row in selected[selected.dataset.eq(dataset)].sort_values(['arm', 'model_kind', 'seed']).iterrows():
            item = dict(method=row.arm, estimator=row.model_kind, seed=row.seed, epoch=row.epoch)
            item.update({cell: f'{row[cell+"_auc"]:.9f} / {row[cell+"_logloss"]:.9f}' for cell in CELLS})
            item['Macro AUC / LL'] = f'{row.macro_auc:.9f} / {row.macro_logloss:.9f}'
            rows.append(item)
        appendix += [table(pd.DataFrame(rows)), '']
        for checkpoint in ['selected', 'E10', 'E50']:
            ledger += [f'**{dataset} / {checkpoint}**', '']
            rows = []
            subset = means[means.dataset.eq(dataset) & means.checkpoint.eq(checkpoint)]
            for _, row in subset.sort_values(['arm', 'model_kind']).iterrows():
                item = {'配置': f'{row.arm} · {row.model_kind}', 'n': 5}
                item.update({cell: f'{row[cell+"_auc"]:.6f} / {row[cell+"_logloss"]:.6f}' for cell in CELLS})
                for label, auc, ll in [('Macro AUC / LL', 'macro_auc', 'macro_logloss'),
                    ('Traffic AUC / LL', 'sample_weighted_cell_auc', 'sample_weighted_cell_logloss'),
                    ('Pooled AUC / LL', 'pooled_task_mean_auc', 'pooled_logloss')]:
                    item[label] = f'{row[auc]:.6f} / {row[ll]:.6f}'
                rows.append(item)
            ledger += [table(pd.DataFrame(rows)), '']
    (HERE / 'NUMERICAL_APPENDIX.md').write_text('\n'.join(appendix))
    (HERE / 'LEDGER_SECTION.md').write_text('\n'.join(ledger))

    figure_dir = HERE / 'figures'
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row_id, dataset in enumerate(DATASETS):
        for column, metric in enumerate(['macro_auc', 'macro_logloss']):
            ax = axes[row_id, column]
            for offset, comparison, color in [(-.16, 'total', '#245a81'), (.16, 'component', '#ba5f25')]:
                subset = primary[primary.dataset.eq(dataset) & primary.comparison.eq(comparison)]
                for index, arm in enumerate(METHODS):
                    values = subset[subset.arm.eq(arm)].sort_values('seed')['delta_'+metric].to_numpy()
                    assert len(values) == 5
                    y = index + offset
                    ax.scatter(values, y + np.linspace(-.035, .035, 5), color=color, s=18, alpha=.6)
                    mean, interval = values.mean(), 2.776445105 * values.std(ddof=1) / np.sqrt(5)
                    ax.errorbar(mean, y, xerr=interval, fmt='D', color=color, markersize=5,
                                capsize=3, label=comparison if index == 0 else None)
            ax.axvline(0, color='gray', linewidth=.8)
            if metric == 'macro_auc':
                ax.axvline(.003, color='#39744a', linestyle='--', linewidth=1)
            ax.set_yticks(range(len(METHODS)), METHODS)
            ax.invert_yaxis()
            ax.set_title(f'{dataset}: delta {metric}')
            ax.set_xlabel('Delta (seed dots; mean with paired t95%)')
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.grid(axis='x', alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.04), ncol=2, frameon=False)
    fig.savefig(figure_dir / 'paired_auc_logloss.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    matrices = {}
    for dataset in DATASETS:
        for comparison in ['total', 'component']:
            subset = primary[primary.dataset.eq(dataset) & primary.comparison.eq(comparison)]
            matrices[(dataset, comparison)] = subset.groupby('arm')[[f'delta_{c}_auc' for c in CELLS]].mean().reindex(METHODS).to_numpy()
    limit = max(np.abs(values).max() for values in matrices.values())
    for i, dataset in enumerate(DATASETS):
        for j, comparison in enumerate(['total', 'component']):
            ax, values = axes[i, j], matrices[(dataset, comparison)]
            plotted = ax.imshow(values, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
            ax.set_xticks(range(6), CELLS)
            ax.set_yticks(range(5), METHODS)
            ax.set_title(f'{dataset}: {comparison} mean delta AUC')
            for y in range(5):
                for x in range(6):
                    ax.text(x, y, f'{values[y,x]:+.4f}', ha='center', va='center', fontsize=8,
                            color='white' if abs(values[y,x]) > limit*.6 else 'black')
                    if values[y,x] < -.001:
                        ax.add_patch(plt.Rectangle((x-.5, y-.5), 1, 1, fill=False, edgecolor='#111111', linewidth=2))
    fig.colorbar(plotted, ax=axes, label='Mean delta AUC; outlined cells below -0.001', shrink=.8)
    fig.savefig(figure_dir / 'cell_delta_auc.png', dpi=180)
    plt.close(fig)
    print(outcomes.to_string(index=False))


if __name__ == '__main__':
    main()
