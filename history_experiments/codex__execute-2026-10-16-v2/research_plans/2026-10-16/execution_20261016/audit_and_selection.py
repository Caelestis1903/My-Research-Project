"""Read-only historical artifact audit and Stage 3 descriptive calculations."""
from pathlib import Path
import csv
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT / 'research_plans/2026-10-15/execution_20261015'
B1 = OLD / 'B1_reconstruction_20261016'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def category(path):
    n = path.name.lower()
    if 'config' in n and n.endswith('.json'): return 'config'
    if 'parameter_manifest' in n: return 'parameter_manifest'
    if 'manifest' in n and n.endswith('.json'): return 'manifest'
    if 'selection' in n or 'selected_epoch' in n: return 'selection'
    if 'metric' in n and n.endswith('.csv'): return 'metrics'
    if n.endswith('.npz') and ('predict' in n or n.startswith(('test_epoch', 'validation_epoch'))): return 'prediction'
    if n.endswith(('.pth', '.pt')):
        return 'optimizer_state' if 'optimizer' in n or 'resume' in n or 'moment' in n else 'checkpoint'
    return None


def audit():
    rows, roots = [], {}
    patterns = {'S4': 'stage_S4*', 'S5': 'stage_S5*', 'S6': 'stage_S6*',
                'S8': 'stage_S8*'}
    for stage, pattern in patterns.items():
        roots[stage] = sorted(p for p in (ROOT / 'experiments').glob(pattern) if p.is_dir())
    roots['AO04'] = [ROOT / 'research_R2/formal_runs_20261012']
    roots['B1_RECONSTRUCTION'] = [B1 / 'runs']
    for stage, paths in roots.items():
        for directory in paths:
            for p in sorted(directory.rglob('*')):
                if not p.is_file(): continue
                kind = category(p)
                if kind:
                    rows.append({'stage': stage, 'path': str(p.relative_to(ROOT)),
                                 'category': kind, 'size': p.stat().st_size})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'artifact_inventory.csv', index=False)
    counts = frame.groupby(['stage', 'category']).size().unstack(fill_value=0)
    counts.to_csv(OUT / 'artifact_category_counts.csv')
    checks = []
    def check(p, expected, receipt):
        actual = sha(p) if p.is_file() else None
        checks.append({'path': str(p.relative_to(ROOT)), 'expected': expected, 'actual': actual,
                       'match': actual == expected, 'receipt': str(receipt.relative_to(ROOT))})
    receipt = B1 / 'analysis/analysis_receipt.json'
    for name, expected in json.loads(receipt.read_text())['output_sha256'].items():
        check(receipt.parent / name, expected, receipt)
    receipt = OLD / 'C2_numeric_oracle_20261016/run_receipt.json'
    for name, expected in json.loads(receipt.read_text())['artifact_hashes'].items():
        check(receipt.parent / name, expected, receipt)
    receipt = OLD / 'output_checksums.json'
    old_checks = json.loads(receipt.read_text())
    # The old checksum index maps paths relative to its execution directory.
    for name, expected in old_checks.items():
        if isinstance(expected, str) and len(expected) == 64:
            check(OLD / name, expected, receipt)
    pd.DataFrame(checks).to_csv(OUT / 'reused_artifact_hash_checks.csv', index=False)
    missing = json.loads((OLD / 'missing_artifacts.json').read_text())
    remaining = []
    for record in missing:
        name = record.get('expected_path')
        if name:
            remaining.append({**record, 'current_exists': (ROOT / name).is_file(),
                              'current_audit_scope': 'explicit historical path; alternatives indexed separately'})
    dump('historical_missing_recheck.json', remaining)
    dump('audit_receipt.json', {'status': 'ACCESSIBLE_EVIDENCE_AUDITED_HISTORICAL_IDENTITY_INCOMPLETE',
         'time_unix': time.time(), 'script_sha256': sha(__file__), 'artifact_rows': len(rows),
         'receipt_hash_checks': len(checks), 'hash_mismatches': sum(not c['match'] for c in checks),
         'missing_explicit_paths': sum(not c['current_exists'] for c in remaining),
         'search_scope': [str(p.relative_to(ROOT)) for dirs in roots.values() for p in dirs],
         'large_state_hashes_recomputed': False, 'training_runs': 0,
         'independent_review': False})
    assert all(c['match'] for c in checks), 'Reused evidence changed; inspect before analysis'


def selection():
    cell = pd.read_csv(B1 / 'analysis/cell_metrics.csv')
    assert len(cell) == 4800
    assert not cell.duplicated(['seed', 'arm', 'split', 'epoch', 'domain', 'task']).any()
    aggregate = pd.read_csv(B1 / 'analysis/aggregate_metrics.csv')
    timing = pd.read_csv(B1 / 'analysis/selected_timing_decomposition.csv')
    for p, t in [('policy_at_F', 'timing_on_DH'), ('policy_at_DH', 'timing_on_F')]:
        assert np.max(np.abs(timing.delta - timing[p] - timing[t])) < 1e-12
    ranks, trajectories, endpoint = [], [], []
    fig, ax = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for j, seed in enumerate([2022, 2023, 2024, 2025]):
        for arm, style in [('F', '-'), ('DH', '--')]:
            own = aggregate[(aggregate.seed == seed) & (aggregate.arm == arm) & (aggregate.metric == 'auc')]
            for quantity in ['macro', 'target', 'non_target', 'd0t0', 'd0t1', 'd1t0', 'd1t1', 'd2t0', 'd2t1']:
                values = own[own.quantity == quantity].pivot(index='epoch', columns='split', values='value').sort_index()
                if len(values) != 50: raise ValueError(quantity)
                rho = float(spearmanr(values.validation, values.test).statistic)
                tau = float(kendalltau(values.validation, values.test).statistic)
                ranks.append(dict(seed=seed, arm=arm, quantity=quantity, n_epochs=50,
                                  val_test_epoch_spearman=rho, val_test_epoch_kendall=tau))
                if quantity in ['macro', 'target', 'non_target']:
                    for split in ['validation', 'test']:
                        color = dict(macro='black', target='tab:blue', non_target='tab:orange')[quantity]
                        ax[int(split == 'test'), j].plot(values.index, values[split], style, color=color,
                                                  alpha=.8, label=f'{arm}/{quantity}')
            epochs = json.loads((B1 / f'run_metadata/seed{seed}/{arm}/selection_frozen.json').read_text())['selection']
            for k, split in enumerate(['validation', 'test']):
                ax[k, j].axvline(epochs['R0'], ls=style, color='tab:green', alpha=.5)
                ax[k, j].set_title(f'{seed} {split}'); ax[k, j].set_xlabel('epoch (0 based)')
            for epoch in range(50):
                c = cell[(cell.seed == seed) & (cell.arm == arm) & (cell.epoch == epoch)]
                v = c[c.split == 'validation'].sort_values(['domain','task'])
                t = c[c.split == 'test'].sort_values(['domain','task'])
                trajectories.append(dict(seed=seed, arm=arm, epoch=epoch,
                    cell_rank_spearman=float(spearmanr(v.auc, t.auc).statistic)))
            for rule, epoch in epochs.items():
                a = own[(own.epoch == epoch) & (own.quantity == 'macro')].set_index('split').value
                endpoint.append(dict(seed=seed, arm=arm, rule=rule, selected_epoch=epoch,
                    val_macro_auc=a['validation'], test_macro_auc=a['test']))
    ax[0, 0].legend(fontsize=6, ncol=2)
    fig.savefig(OUT / 'selection_trajectories.png', dpi=150)
    fig.savefig(OUT / 'selection_trajectories.pdf'); plt.close(fig)
    pd.DataFrame(ranks).to_csv(OUT / 'val_test_epoch_rank_agreement.csv', index=False)
    pd.DataFrame(trajectories).to_csv(OUT / 'val_test_cell_rank_agreement.csv', index=False)
    pd.DataFrame(endpoint).to_csv(OUT / 'selected_endpoints.csv', index=False)
    r0 = timing[(timing.rule == 'R0') & (timing.metric == 'auc')]
    r0.to_csv(OUT / 'r0_timing_decomposition.csv', index=False)
    pd.DataFrame(ranks).pivot_table(index=['seed', 'arm'], columns='quantity',
                                   values='val_test_epoch_spearman').to_csv(OUT / 'epoch_rank_matrix.csv')
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    vals = pd.DataFrame(ranks)
    for n, q in enumerate(['macro','target','non_target']):
        v = vals[vals.quantity == q]
        ax.scatter(np.arange(len(v)) + n*.15, v.val_test_epoch_spearman, label=q)
    ax.axhline(0, color='gray', lw=.6); ax.set_ylim(-1, 1)
    ax.set_xticks(np.arange(8)+.15, [f'{s}/{a}' for s in [2022,2023,2024,2025] for a in ['F','DH']], rotation=40)
    ax.set_ylabel('Spearman across 50 epochs'); ax.legend()
    fig.savefig(OUT / 'rank_agreement.png', dpi=150); plt.close(fig)
    dump('selection_receipt.json', {'status': 'COMPLETE_RECONSTRUCTION_FORENSIC_ONLY',
        'training_runs': 0, 'cell_rows': len(cell), 'rank_pairs': len(ranks),
        'source_cell_sha256': sha(B1 / 'analysis/cell_metrics.csv'),
        'source_aggregate_sha256': sha(B1 / 'analysis/aggregate_metrics.csv'),
        'timing_identities_max_residual': float(max(np.max(np.abs(timing.delta-timing.policy_at_F-timing.timing_on_DH)),
           np.max(np.abs(timing.delta-timing.policy_at_DH-timing.timing_on_F)))),
        'test_role': 'retrospective_forensic_no_selection', 'independent_review': False})


if __name__ == '__main__':
    audit()
    selection()
    print('AUDIT_AND_SELECTION_DONE', flush=True)
