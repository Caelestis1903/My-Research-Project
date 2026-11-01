"""Export seed-level selected-test effects from the frozen numerical tables."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    out = Path(__file__).resolve().parent / 'figures'
    out.mkdir(exist_ok=True)
    pairs = pd.read_csv(run / 'paired_effects.csv')
    summary = pd.read_csv(run / 'effect_summary.csv')
    pairs = pairs[(pairs.split == 'test') & (pairs.checkpoint == 'selected')]
    summary = summary[(summary.split == 'test') & (summary.checkpoint == 'selected')]
    effects = [('EMA_A0', 'Adam EMA - online'),
               ('augmentation', 'Aug EMA - Adam EMA'),
               ('AIG_independent', 'AIG EMA - Aug EMA')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), constrained_layout=True)
    for row, dataset in enumerate(['movielens', 'kuairand']):
        for col, metric in enumerate(['macro_auc', 'macro_logloss']):
            ax = axes[row, col]
            available = [(effect, label) for effect, label in effects
                         if effect in set(summary[summary.dataset == dataset].effect)]
            for y, (effect, _) in enumerate(available):
                sel = (pairs.dataset == dataset) & (pairs.metric == metric) & (pairs.effect == effect)
                values = pairs[sel].sort_values('seed').delta.to_numpy()
                assert len(values) == 5
                stat = summary[(summary.dataset == dataset) & (summary.metric == metric)
                               & (summary.effect == effect)].iloc[0]
                ax.scatter(values, y + np.linspace(-.12, .12, 5), color='#377ca8', s=22, zorder=3)
                ax.errorbar(stat['mean'], y, xerr=[[stat['mean'] - stat.ci95_low],
                    [stat.ci95_high - stat['mean']]], fmt='D', color='#b63b32', capsize=4, ms=5)
            ax.axvline(0, color='#888888', lw=1)
            ax.set_yticks(range(len(available)), [label for _, label in available])
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=.2)
            ax.set_title(('MovieLens-1M' if row == 0 else 'KuaiRand-Pure') +
                         (' | AUC: higher is better' if col == 0 else ' | LogLoss: lower is better'))
            ax.set_xlabel('Paired difference; blue: seeds, red: mean and 95% t interval')
    fig.savefig(out / 'selected_seed_effects.png', dpi=180)
    fig.savefig(out / 'selected_seed_effects.pdf')
    plt.close(fig)

    gates = pd.read_csv(run / 'gate_epoch_summary.csv')
    gates = gates[gates.stage == 'S4']
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for row, dataset in enumerate(['movielens', 'kuairand']):
        for col, group in enumerate(['E', 'P']):
            ax = axes[row, col]
            for trial, color in [('M1', '#c26d29'), ('M2', '#377ca8')]:
                values = gates[(gates.dataset == dataset) & (gates.group == group)
                               & (gates.trial == trial)]
                assert set(values.seed) == {2022, 2024} and len(values) == 100
                curve = values.groupby('epoch').gate.agg(['mean', 'std'])
                ax.plot(curve.index, curve['mean'], label=trial, color=color)
                ax.fill_between(curve.index, curve['mean'] - curve['std'],
                    curve['mean'] + curve['std'], color=color, alpha=.15)
            ax.axhline(.5, color='#888888', ls=':', lw=1)
            ax.set_ylim(0, 1)
            ax.set_title(dataset + ' | group ' + group)
            ax.set_xlabel('Epoch; two development seeds, mean +/- sample SD')
            ax.set_ylabel('Gate = lambda / lambda_max')
            ax.legend(frameon=False)
    fig.savefig(out / 'innovation_gate_curves.png', dpi=180)
    fig.savefig(out / 'innovation_gate_curves.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
