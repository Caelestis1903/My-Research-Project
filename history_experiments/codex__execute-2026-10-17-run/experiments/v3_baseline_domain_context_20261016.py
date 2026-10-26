"""Describe baseline domain context from verified fixed-stage diagnostic cells."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research_plans/2026-10-16_v3/execution_20261016'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    aggregate = OUT / 'analysis/V3_selection_baselines_final_aggregate'
    receipt = json.loads((aggregate / 'receipt.json').read_text())
    assert receipt['status'] == 'COMPLETE'
    config_pins = {}
    for item in receipt['sources']:
        parent = Path(item['pair_directory']) / 'receipt.json'
        assert sha(parent) == item['receipt_sha']
        pair = json.loads(parent.read_text())
        for run, digest in zip(pair['source_runs'], pair['source_run_receipt_sha']):
            directory = Path(run)
            assert sha(directory / 'receipt.json') == digest
            producer = json.loads((directory / 'receipt.json').read_text())
            config = directory / 'config.json'
            assert sha(config) == producer['artifact_hashes']['config.json']
            config_pins[run] = sha(config)
    frame = pd.read_csv(aggregate / 'cell_trajectory.csv')
    frame = frame[(frame.split == 'diagnostic') & frame.epoch.isin([10, 50])].copy()
    assert len(frame) == 7 * 4 * 2 * 2 * 3 * 2
    assert sorted(frame.seed.unique()) == [2022, 2023, 2024, 2025]
    rows = []
    configs = {run: json.loads((Path(run) / 'config.json').read_text()) for run in frame.run.unique()}
    for row in frame.to_dict('records'):
        assert row['run'] in config_pins
        config = configs[row['run']]
        domain, task = int(row['domain']), int(row['task'])
        data = config['data']
        train = data['cell_stats']['train'][f'd{domain}t{task}']
        n = int(data['train_domain_counts'][str(domain)])
        assert train['count'] == n
        rows.append(row | dict(dataset=config['dataset'], train_n=n,
            train_n_pos=train['positive_count'], train_positive_rate=train['positive_rate'],
            physical_domain=data['physical_domain_identity'][domain],
            size_role=data['role_by_domain'][str(domain)], task_readout=str(task)))
    cells = pd.DataFrame(rows)
    macro = cells.groupby(['group', 'seed', 'epoch', 'domain', 'arm'], as_index=False).agg(
        auc=('auc', 'mean'), logloss=('logloss', 'mean'), train_n=('train_n', 'first'),
        dataset=('dataset', 'first'), physical_domain=('physical_domain', 'first'),
        size_role=('size_role', 'first'))
    macro['task_readout'] = 'domain_macro_equal_two_tasks'
    combined = pd.concat([cells, macro], ignore_index=True)
    summary = []
    for key, part in combined.groupby(['group', 'epoch', 'domain', 'task_readout']):
        sn = part[part.arm == 'SN'].sort_values('seed')
        sb = part[part.arm == 'SB'].sort_values('seed')
        assert sn.seed.tolist() == sb.seed.tolist() == [2022, 2023, 2024, 2025]
        assert part.train_n.nunique() == 1
        result = dict(zip(['group', 'epoch', 'domain', 'task_readout'], key))
        result |= dict(dataset=sn.dataset.iloc[0], train_n=int(sn.train_n.iloc[0]),
                       physical_domain=sn.physical_domain.iloc[0], size_role=sn.size_role.iloc[0])
        for metric in ['auc', 'logloss']:
            for arm, values in [('SN', sn[metric].to_numpy()), ('SB', sb[metric].to_numpy()),
                                ('SB_minus_SN', sb[metric].to_numpy() - sn[metric].to_numpy())]:
                result.update({f'{arm}_{metric}_mean': float(values.mean()),
                               f'{arm}_{metric}_min': float(values.min()),
                               f'{arm}_{metric}_max': float(values.max()),
                               f'{arm}_{metric}_seed_values': json.dumps(values.tolist())})
        result['auc_gap_negative_seeds'] = int((sb.auc.to_numpy() < sn.auc.to_numpy()).sum())
        summary.append(result)
    table = pd.DataFrame(summary)
    assert len(table) == 7 * 2 * 3 * 3
    target = OUT / 'analysis/baseline_domain_context'
    target.mkdir(exist_ok=False)
    cells.to_csv(target / 'all7_fixed10_50_diagnostic_cells_with_train_context.csv', index=False)
    table.to_csv(target / 'all7_domain_and_task_descriptive_summary.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    groups = [('fixedSGD_01', 'KuaiRand: SGD 0.1'), ('calSGD_03', 'KuaiRand: SGD 0.3'),
              ('calAdam_0003', 'KuaiRand: Adam 0.003'), ('MovieLensSGD_03', 'MovieLens: SGD 0.3')]
    domain_table = table[table.task_readout == 'domain_macro_equal_two_tasks']
    for ax, (group, title) in zip(axes.flat, groups):
        selected = domain_table[domain_table.group == group]
        for epoch, color, offset in [(10, '#8c8c8c', -.008), (50, '#1764a2', .008)]:
            view = selected[selected.epoch == epoch].sort_values('domain')
            assert len(view) == 3
            y = view.SN_auc_mean.to_numpy()
            ax.errorbar(view.train_n, y,
                yerr=[y - view.SN_auc_min.to_numpy(), view.SN_auc_max.to_numpy() - y],
                fmt='o', color=color, markersize=5, capsize=3, label=f'Epoch {epoch}')
            for _, row in view.iterrows():
                ax.annotate('d' + str(int(row.domain)), (row.train_n, row.SN_auc_mean),
                            xytext=(4, 7 if epoch == 50 else -11), textcoords='offset points',
                            fontsize=8, color=color)
        ax.set_xscale('log')
        ax.set_title(title)
        ax.set_xlabel('Training rows per domain (log scale)')
        ax.set_ylabel('SN diagnostic domain macro AUC')
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('Domain size and baseline performance: model and stage matter', fontsize=12)
    fig.text(.5, .015, 'M3OE; domain AUC = equal mean of two task AUCs. Bars show observed four-seed ranges, not confidence intervals.\n'
             'N is not randomized. Cross-environment equal epochs do not match updates or examples. Panel y-axis scales differ.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .96))
    figures = []
    for extension in ['png', 'pdf']:
        path = OUT / 'figures' / ('baseline_domain_size_and_performance.' + extension)
        fig.savefig(path, dpi=180)
        figures.append(path)
    plt.close(fig)
    files = list(target.glob('*.csv')) + figures
    result = dict(status='COMPLETE', independent_review=False, source_sha=sha(Path(__file__)),
                  scope_sha=sha(OUT / 'baseline_domain_context_scope.json'),
                  aggregate_receipt_sha=sha(aggregate / 'receipt.json'),
                  aggregate_cells_sha=sha(aggregate / 'cell_trajectory.csv'), config_pins=config_pins,
                  cell_rows=len(cells), summary_rows=len(table),
                  outputs={str(path.relative_to(ROOT)): sha(path) for path in files},
                  inference='Descriptive only; no new bootstrap, p-value, regression, causal identification or gate.',
                  test_access=False)
    (target / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='COMPLETE', cell_rows=len(cells), summary_rows=len(table))))


if __name__ == '__main__':
    main()
