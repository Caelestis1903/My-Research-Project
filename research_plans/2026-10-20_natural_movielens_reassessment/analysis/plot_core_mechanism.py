"""Scientific figure of the completed core fixed-state MC trace comparisons."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'core_mechanism'


def plot():
    assert json.loads((OUT / 'analysis_status.json').read_text())['complete']
    f = pd.read_csv(OUT / 'covariance.csv')
    assert len(f) == 160
    stages = ['initial_state.pt', 'epoch09.pt', 'epoch39.pt', 'selected_state.pt', 'epoch49.pt']
    panels = [
        ('movielens_natural', 'full50', 'Natural full support / 1,000 updates'),
        ('movielens_artificial_d2_fixed2022', 'full50', 'Artificial fixed support / 250 updates'),
        ('movielens_natural_matched_n187500', 'full50', 'Natural matched N / 250 updates'),
        ('movielens_natural', 'matched250', 'Natural full support / 250 updates'),
    ]
    # Population names are manifest identities. Do not silently omit a cohort.
    assert set(zip(f.population, f.budget)) == {(a, b) for a, b, _ in panels}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (population, budget, title) in zip(axes.flat, panels):
        g = f[(f.population == population) & (f.budget == budget)]
        for arm, offset, color in [('SN', -.12, '#1766a0'), ('SB', .12, '#c66410')]:
            for i, checkpoint in enumerate(stages):
                z = g[(g.arm == arm) & (g.checkpoint == checkpoint)].sort_values('seed')
                assert len(z) == 4
                x = i + offset + np.linspace(-.035, .035, 4)
                ax.scatter(x, z.SB_over_SN_trace, color=color, alpha=.7, s=27,
                    label=f'{arm} training anchor' if i == 0 else None)
                ax.plot([i + offset - .06, i + offset + .06], [z.SB_over_SN_trace.mean()] * 2,
                    color=color, lw=2)
        ax.axhline(1, color='#333333', lw=1, ls='--')
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(5), ['INIT', 'E10', 'E40', 'SELECTED', 'E50'])
        ax.set_ylabel('SB / SN gradient trace')
        ax.grid(axis='y', alpha=.2)
        if g.SB_over_SN_trace.max() / g.SB_over_SN_trace.min() > 5:
            ax.set_yscale('log')
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('Core fixed-state gradient covariance: population and checkpoint', fontsize=14)
    fig.text(.5, .015, 'Points: four optimizer seeds; bars: mean. MC512 per estimator, B40,960. '
        'No MC confidence interval.\nCheckpoint categories are not a chronological curve; SELECTED may precede E10/E40.',
        ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    fig.savefig(OUT / 'gradient_trace_ratios.png', dpi=180)
    fig.savefig(OUT / 'gradient_trace_ratios.pdf')
    plt.close(fig)


if __name__ == '__main__':
    plot()
