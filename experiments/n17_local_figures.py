"""N3 descriptive figures from completed primary readouts only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import n17_common as c


def load(directory):
    receipt = json.loads((directory / 'receipt.json').read_text())
    assert receipt['status'] == 'COMPLETE' and receipt['training_seed_unit']
    seeds = pd.read_csv(directory / 'seed_estimands.csv')
    summary = pd.read_csv(directory / 'conditional_summary.csv')
    return seeds, summary


def effect(axis, seeds, summary, x, epoch, anchor, block, case, estimand, color, offset, label=None):
    filters = dict(epoch=epoch, anchor_arm=anchor, block=block, case=case, estimand=estimand, split='diagnostic', metric='auc')
    def subset(frame):
        mask = np.ones(len(frame), dtype=bool)
        for key, value in filters.items():
            mask &= frame[key].to_numpy() == value
        return frame[mask]
    points = subset(seeds).sort_values('seed')
    group = subset(summary)
    if len(group) == 0:
        return False
    assert len(group) == 1 and len(points) == 4 and points.seed.nunique() == 4
    row = group.iloc[0]
    assert row['n'] == 4
    jitter = np.linspace(-.045, .045, 4)
    axis.scatter(x + offset + jitter, points['mean'], color=color, alpha=.55, s=14, zorder=2)
    axis.errorbar(x + offset, row['mean'], yerr=[[row['mean'] - row.seed_t95_low], [row.seed_t95_high - row['mean']]],
                  fmt='D', color=color, markersize=4, capsize=3, lw=1, label=label, zorder=3)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--confirmation', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    frozen = json.loads(a.candidate.read_text())
    assert frozen['status'] == 'FROZEN'
    candidate = frozen['candidate']
    dev = load(a.development)
    panel_count = 2 if candidate is not None and a.confirmation else 1
    figure, axes = plt.subplots(panel_count, 1, figsize=(10, 4 * panel_count), squeeze=False, constrained_layout=True)
    axis = axes[0, 0]
    effects = [('C_full', 'vs_SB', 'full', '#2369a5'), ('C_norm', 'vs_SB', 'norm', '#d88723'),
               ('C_direction', 'vs_SB', 'direction', '#3c8d68'), ('C_factorial', 'norm_direction_interaction', 'interaction', '#a25b9b')]
    groups = [('SN', 10), ('SN', 40), ('SB', 10), ('SB', 40)]
    for i, (case, estimand, label, color) in enumerate(effects):
        for x, (anchor, epoch) in enumerate(groups):
            effect(axis, *dev, x, epoch, anchor, 'C', case, estimand, color, (i - 1.5) * .17, label if x == 0 else None)
    axis.set_xticks(range(4), [f'{anchor} E{epoch}' for anchor, epoch in groups])
    axis.set_title('A. Whole C: development, four training seeds per path and stage')
    axis.legend(frameon=False, ncol=4, fontsize=9)
    sources = [a.development]
    if panel_count == 2:
        confirm = load(a.confirmation)
        gate = json.loads((a.confirmation / 'confirmation_gate.json').read_text())
        assert gate['candidate_sha'] == c.sha(a.candidate)
        sources.append(a.confirmation)
        axis = axes[1, 0]
        block, epoch = candidate['block'], candidate['epoch']
        for i, (suffix, estimand, label, color) in enumerate([('_full', 'vs_SB', 'full minus SB', '#2369a5'), ('_factorial', 'direction_residual', 'full minus norm', '#bd4a48')]):
            for x, (data, anchor) in enumerate([(dev, 'SN'), (dev, 'SB'), (confirm, 'SN'), (confirm, 'SB')]):
                effect(axis, *data, x, epoch, anchor, block, block + suffix, estimand, color, (i - .5) * .22, label if x == 0 else None)
        axis.set_xticks(range(4), ['Dev SN path', 'Dev SB path', 'Confirm SN path', 'Confirm SB path'])
        axis.set_title(f'B. Frozen candidate {block}, E{epoch}; development and independent confirmation')
        axis.legend(frameon=False, ncol=2, fontsize=9)
    for axis in axes.flat:
        axis.axhline(0, color='.4', lw=.8)
        for boundary in [-.0001, .0001]:
            axis.axhline(boundary, color='.7', ls='--', lw=.6)
        axis.set_ylabel('Diagnostic macro AUC difference')
        axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle('Points: training-seed means; diamonds/bars: mean and pointwise seed t95 CI\nMC draws are conditional repetitions, not additional training seeds', fontsize=11)
    a.out.mkdir(parents=True, exist_ok=False)
    for extension in ['png', 'pdf']:
        figure.savefig(a.out / ('local_counterfactuals.' + extension), dpi=180)
    plt.close(figure)
    c.dump(a.out / 'receipt.json', dict(status='COMPLETE', source_sha=c.sha(__file__), candidate_sha=c.sha(a.candidate),
        inputs={str(directory): {name: c.sha(directory / name) for name in ['receipt.json', 'seed_estimands.csv', 'conditional_summary.csv']} for directory in sources},
        test_read=False, scope='descriptive pointwise four-training-seed t intervals; no reselection; dashed lines indicate the preregistered local practical magnitude, not a significance test'))


if __name__ == '__main__':
    main()
