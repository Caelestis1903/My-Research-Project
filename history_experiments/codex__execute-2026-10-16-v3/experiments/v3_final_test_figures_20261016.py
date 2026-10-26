"""Plot all frozen baseline test summaries after primary numerical verification."""
from pathlib import Path
import json
import hashlib
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research_plans/2026-10-16_v3/execution_20261016'

def load(path):
    return json.loads(path.read_text())

def main():
    receipt = OUT / 'analysis/test_all_cohorts_primary_verification/receipt.json'
    assert load(receipt)['status'] == 'PASS'
    groups = list(load(OUT/'test_baseline7_summary_manifest.json')['groups'])
    data = load(OUT/'analysis/test_baseline7_summary/test_seed_summary.json')
    cells = load(OUT/'analysis/test_baseline7_summary/test_cell_seed_summary.json')
    figure_dir = OUT/'figures'
    figure_dir.mkdir(exist_ok=True)
    files = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, metric, label in zip(axes, ('macro_auc', 'macro_logloss'), ('Macro AUC: positive favors SB', 'Macro LogLoss: negative favors SB')):
        rows = [next(x for x in data if (x['group'], x['readout'], x['metric']) == (g, 'selected_macro_auc', metric)) for g in groups]
        assert all(x['n'] == 4 for x in rows)
        values = np.array([x['mean'] for x in rows])
        errors = np.array([[x['mean']-x['ci_low'] for x in rows], [x['ci_high']-x['mean'] for x in rows]])
        ax.errorbar(values, np.arange(len(groups)), xerr=errors, fmt='o', capsize=4, color='#2364aa')
        ax.axvline(0, color='gray', linewidth=1)
        ax.set_xlabel('SB - SN; ' + label)
        ax.grid(axis='x', alpha=.25)
        ax.set_yticks(range(len(groups)), groups)
    axes[0].invert_yaxis()
    fig.suptitle('Frozen test at validation-selected AUC epochs | all 4 training seeds\nMean and 95% seed bootstrap CI; exact two-sided sign test minimum p=0.125')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        path = figure_dir/('final_test_selected_macro_gaps.'+suffix)
        assert not path.exists()
        fig.savefig(path, dpi=170, bbox_inches='tight')
        files.append(path)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    records = []
    for ax, metric in zip(axes, ('auc', 'logloss')):
        rows = [[next(x for x in cells if (x['group'], x['readout'], x['domain'], x['task'], x['metric']) == (g, 'selected_macro_auc', d, t, metric)) for d in range(3) for t in range(2)] for g in groups]
        matrix = np.array([[x['mean'] for x in line] for line in rows])
        extent = max(float(abs(matrix).max()), 1e-12)
        im = ax.imshow(matrix, cmap='RdBu', vmin=-extent, vmax=extent)
        for i, line in enumerate(rows):
            for j, x in enumerate(line):
                assert x['n'] == 4
                ax.text(j, i, f"{x['mean']:+.4f}\n{x['positive']}+/{x['negative']}-", ha='center', va='center', fontsize=8, color='white' if abs(matrix[i,j])>extent*.58 else 'black')
                records.append(dict(group=groups[i], domain=j//2, task=j%2, metric=metric, **{k:x[k] for k in ('n','mean','ci_low','ci_high','positive','negative')}))
        ax.set_xticks(range(6), [f'd{d}t{t}' for d in range(3) for t in range(2)])
        ax.set_yticks(range(len(groups)), groups)
        ax.set_title(f'{metric}: SB - SN mean; seed sign counts')
        fig.colorbar(im, ax=ax, shrink=.7)
    fig.suptitle('All six physical domain-task cells | frozen test at validation-selected AUC epochs\nCells are not independent training-seed replicates; AUC and LogLoss favor opposite directions')
    fig.tight_layout()
    for suffix in ('png','pdf'):
        path = figure_dir/('final_test_all_six_cells.'+suffix)
        assert not path.exists()
        fig.savefig(path, dpi=170, bbox_inches='tight')
        files.append(path)
    plt.close(fig)
    path=OUT/'archive/formal/final_test_selected_all_six_cell_plot_values.csv'
    assert not path.exists()
    pd.DataFrame(records).to_csv(path,index=False)
    files.append(path)
    result=dict(status='PASS',time=time.time(),independent_review=False,source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),primary_verification_sha=hashlib.sha256(receipt.read_bytes()).hexdigest(),all_groups=groups,selection_rule='FrozenvalidationmacroAUC; no test choices',cell_rows=len(records),hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    path=OUT/'final_test_figure_receipt.json'
    assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n')

if __name__ == '__main__':
    main()
