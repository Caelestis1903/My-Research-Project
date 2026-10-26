"""Standalone figures from validated N2 summaries; no model/test access."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import n17_common as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.summary / 'receipt.json').read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['anchors'] == 16
    r = pd.read_csv(args.summary / 'raw_moments.csv')
    m = pd.read_csv(args.summary / 'mean_mapping_contrasts.csv')
    r = r[r.block == 'ALL'].set_index(['anchor_arm', 'epoch', 'seed', 'arm', 'variant'])
    m = m[m.block == 'ALL'].set_index(['anchor_arm', 'epoch', 'seed', 'variant'])
    anchors = [(a, e, s) for a in ('SN', 'SB') for e in (10, 40) for s in range(2032, 2036)]
    xs = np.arange(16); colors = {'IID': '#2369a5', 'quota': '#d88723'}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    values = {}
    for v, offset in [('IID', -.09), ('quota', .09)]:
        y = [r.loc[(*k, 'SB', v), 'empirical_covariance_trace'] / r.loc[(*k, 'SN', v), 'empirical_covariance_trace'] for k in anchors]
        values[v] = y; ax.scatter(xs + offset, y, s=27, color=colors[v], label=v)
    for i in xs: ax.plot([i - .09, i + .09], [values['IID'][i], values['quota'][i]], color='.65', lw=.7, zorder=0)
    ax.axhline(1, color='.5', lw=.8); ax.set(title='A. Total gradient covariance trace ratio', ylabel='SB / SN'); ax.legend(frameon=False)
    ax = axes[0, 1]
    for arm, offset, color in [('SN', -.09, '#2369a5'), ('SB', .09, '#bd4a48')]:
        y = [100 * r.loc[(*k, arm, 'IID'), 'predicted_between_fraction'] for k in anchors]
        ax.scatter(xs + offset, y, s=27, color=color, label=arm)
    ax.set(yscale='log', title='B. Domain-count share of predicted IID trace', ylabel='Percent (log scale)'); ax.legend(frameon=False)
    ax = axes[1, 0]
    for arm, color, side in [('SN', '#2369a5', -1), ('SB', '#bd4a48', 1)]:
        for v, marker, shift in [('IID', 'o', .03), ('quota', 'x', .15)]:
            y = [r.loc[(*k, arm, v), 'empirical_over_predicted_trace'] for k in anchors]
            ax.scatter(xs + side * shift, y, marker=marker, color=color, s=24, label=arm + '-' + v)
    ax.axhline(1, color='.5', lw=.8); ax.set(title='C. Empirical / decomposed total trace', ylabel='Ratio; decomposition estimates use same bank'); ax.legend(frameon=False, ncol=2, fontsize=8)
    ax = axes[1, 1]
    for v, offset in [('IID', -.09), ('quota', .09)]:
        y = [m.loc[(*k, v), 'mean_update_difference_norm'] / m.loc[(*k, v), 'mean_update_difference_mc_rms_upper_bound'] for k in anchors]
        ax.scatter(xs + offset, y, s=27, color=colors[v], label=v)
    ax.axhline(1, color='.5', lw=.8, ls='--', label='MC RMS scale (not CI)')
    ax.set(title='D. Fixed-state Adam q mean-update contrast', ylabel='Contrast norm / sum of marginal MC RMS', ylim=(0, 1.1)); ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.set_xlim(-.6, 15.6); ax.set_xticks([1.5, 5.5, 9.5, 13.5], ['SN E10', 'SN E40', 'SB E10', 'SB E40'])
        ax.set_xlabel('Anchor path and epoch; four points = seeds 2032–2035')
        for x in (3.5, 7.5, 11.5): ax.axvline(x, color='.9', lw=.7, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    figure.suptitle('N2: 4 training seeds, 2 paths, 2 stages; B=40,960, K=512 per arm', fontsize=13)
    args.out.mkdir(parents=True, exist_ok=False)
    for ext in ('png', 'pdf'): figure.savefig(args.out / ('fixed_point_overview.' + ext), dpi=180)
    plt.close(figure)
    c.dump(args.out / 'receipt.json', dict(status='COMPLETE', source_sha=c.sha(__file__),
        summary_receipt_sha=c.sha(args.summary / 'receipt.json'),
        input_sha={name: c.sha(args.summary / name) for name in ('raw_moments.csv', 'mean_mapping_contrasts.csv')},
        scope='points reuse four training seeds across path/stage; no extra independent replicates; MC RMS is not a confidence bound; same-bank decomposition', test_read=False))


if __name__ == '__main__':
    main()
