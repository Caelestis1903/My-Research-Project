"""Render already verified V11 E10 summaries without new statistical estimation."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'research_plans/2026-10-16_v3/execution_20261016'
    geo = out / 'analysis/V11_shadow_geometry_all4/all4_geometry.csv'
    stats = out / 'analysis/V11_shadow_statistics/macro_and_train_contrast_summary.csv'
    g, s = pd.read_csv(geo), pd.read_csv(stats)
    conditions = ['swap_m', 'swap_v', 'full_shadow', 'beta1_zero_v_only', 'beta2_zero', 'm_only']
    labels = ['Swap m', 'Swap v', 'Full shadow', 'Beta1 = 0', 'Beta2 = 0', 'm only']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
    rows = []
    for arm, offset, color, marker in [('SN', -.12, '#1764a2', 'o'), ('SB', .12, '#ce5b1b', 's')]:
        for index, condition in enumerate(conditions):
            a = g[(g.epoch == 10) & (g.current_arm == arm) & (g.condition == condition)
                  & (g.norm_control == 'raw') & (g.block == 'ALL')].sort_values('seed')
            b = s[(s.epoch == 10) & (s.current_arm == arm) & (s.condition == condition)
                  & (s.norm_control == 'match_native_global_norm') & (s.split == 'diagnostic')
                  & (s.variable == 'macro_auc') & (s.contrast == 'condition_minus_native')]
            assert a.seed.tolist() == [2022, 2023, 2024, 2025] and len(b) == 1
            b = b.iloc[0]
            assert b.n == 4
            y = index + offset
            for ax, variable in zip(axes[:2], ['delta_norm_relative_native', 'cosine_to_native']):
                vals = a[variable].to_numpy()
                ax.scatter(vals, np.full(4, y), s=15, alpha=.45, color=color, marker=marker)
                ax.scatter(vals.mean(), y, s=43, color=color, marker=marker,
                           label=arm if index == 0 else None)
            axes[2].errorbar(b['mean'], y, xerr=[[b['mean'] - b.ci_low], [b.ci_high - b['mean']]],
                             color=color, marker=marker, capsize=2, markersize=5)
            rows.append(dict(condition=condition, arm=arm,
                             raw_norm_ratio_mean=a.delta_norm_relative_native.mean(),
                             raw_cosine_mean=a.cosine_to_native.mean(),
                             normmatched_auc_delta=b['mean'], ci_low=b.ci_low, ci_high=b.ci_high))
    axes[0].set_xscale('log')
    axes[0].axvline(1, color='.65', linewidth=.8)
    axes[1].axvline(1, color='.65', linewidth=.8)
    axes[2].axvline(0, color='.65', linewidth=.8)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc='upper left')
    for ax, title, xlabel in zip(axes,
         ['Raw update magnitude', 'Raw update direction', 'AUC after global norm matching'],
         ['Norm / native norm (log scale)', 'Cosine to same-q native update', 'Diagnostic macro AUC delta vs native']):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis='x', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('V11 local state interventions at epoch 10: four training seeds', fontsize=12)
    fig.text(.5, .025, 'Geometry: dots = seeds; solid markers = means. AUC bars: existing seed-bootstrap 95% intervals.\n'
             'Native-SN anchors, same current q and batch. Global norm matching does not fix block norms or establish long-run mediation.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .1, 1, .94))
    folder = out / 'figures'
    artifacts = []
    for suffix in ['png', 'pdf']:
        path = folder / ('V11_shadow_epoch10_geometry_and_auc.' + suffix)
        fig.savefig(path, dpi=180)
        artifacts.append(path)
    plt.close(fig)
    csv = folder / 'V11_shadow_epoch10_geometry_and_auc.csv'
    pd.DataFrame(rows).to_csv(csv, index=False)
    artifacts.append(csv)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = dict(status='COMPLETE', scope='Visualization only of already primary-verified statistics; no new hypothesis test or experiment.',
                   independent_review=False, source_sha=sha(Path(__file__)),
                   inputs={str(path.relative_to(root)): sha(path) for path in [geo, stats]},
                   artifacts={str(path.relative_to(root)): sha(path) for path in artifacts},
                   rows=len(rows), test_access=False)
    (folder / 'V11_shadow_epoch10_geometry_and_auc_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
