"""Render the completed experiment's existing numerical tables."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', default=str(Path(__file__).parent / 'figures'))
    args = parser.parse_args()
    root, out = Path(args.run_dir), Path(args.output)
    completed = json.loads((root / 'completion.json').read_text())
    assert completed['status'] == 'COMPLETE'
    assert completed['phase1_runs'] == 40
    out.mkdir(parents=True, exist_ok=True)
    effects = pd.read_csv(root / 'effect_summary.csv')
    pairs = pd.read_csv(root / 'paired_effects.csv')
    epochs = pd.read_csv(root / 'epoch_metrics.csv')
    diagnostics = pd.read_csv(root / 'optimizer_state_diagnostics.csv')
    phase1 = epochs[epochs.arm.isin(['A0', 'A1', 'A2', 'A3'])]
    assert len(phase1) == 2000
    assert phase1.groupby(['dataset', 'arm', 'seed']).epoch.count().eq(50).all()
    datasets = ['movielens', 'kuairand']
    colors = dict(zip(['A0', 'A1', 'A2', 'A3'], ['#526579', '#d68a23', '#4f9c8f', '#8c59a0']))
    contrasts = ['P0_A0_shadow-online', 'A1-A0_online', 'A2-A1_online', 'A3-A1_online', 'A3-A2_online']
    labels = ['A0 shadow - online', 'A1 - A0', 'A2 - A1 (equivalent)', 'A3 - A1', 'A3 - A2']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for row, dataset in enumerate(datasets):
        for col, split in enumerate(['validation', 'test']):
            ax = axes[row, col]
            for y, effect in enumerate(contrasts):
                query = (effects.dataset.eq(dataset) & effects.split.eq(split) &
                         effects.checkpoint.eq('selected') & effects.metric.eq('macro_auc') & effects.effect.eq(effect))
                r = effects.loc[query].iloc[0]
                raw = pairs[(pairs.dataset == dataset) & (pairs.split == split) &
                            (pairs.checkpoint == 'selected') & (pairs.metric == 'macro_auc') & (pairs.effect == effect)].sort_values('seed')
                assert len(raw) == 5
                ax.plot([r.ci_low * 1000, r.ci_high * 1000], [y, y], color='#526579', lw=2)
                ax.scatter(raw.value * 1000, y + np.linspace(-.14, .14, 5), s=17, color='#90a8be', alpha=.8)
                ax.scatter(r['mean'] * 1000, y, color='#b83947', s=40, marker='D', zorder=3)
            ax.axvline(0, color='#777777', lw=.8)
            ax.set_yticks(range(len(labels)), labels if col == 0 else [])
            ax.invert_yaxis()
            ax.set_title(f'{dataset} / {split}')
            ax.grid(axis='x', alpha=.2)
            ax.set_xlabel('Selected macro AUC difference (x 0.001); paired 95% t interval')
    fig.tight_layout()
    fig.savefig(out / 'selected_effects.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for row, dataset in enumerate(datasets):
        for col, kind in enumerate(['online', 'shadow']):
            ax = axes[row, col]
            for arm, color in colors.items():
                selected = phase1[(phase1.dataset == dataset) & (phase1.arm == arm)]
                curve = selected.groupby('epoch')[f'val_{kind}_macro_auc'].agg(['mean', 'std'])
                ax.plot(curve.index, curve['mean'], label=arm, color=color)
                ax.fill_between(curve.index, curve['mean'] - curve['std'], curve['mean'] + curve['std'], color=color, alpha=.1)
            ax.set_title(f'{dataset} / {kind} validation')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Six-cell macro AUC')
            ax.grid(alpha=.2); ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'validation_curves.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for row, dataset in enumerate(datasets):
        for col, metric in enumerate(['injection_ratio', 'q_delta_cosine']):
            ax = axes[row, col]
            for arm in ['A2', 'A3']:
                table = diagnostics[(diagnostics.dataset == dataset) & (diagnostics.arm == arm)]
                curve = table.groupby('epoch')[metric].agg(['mean', 'std'])
                ax.plot(curve.index, curve['mean'], label=arm, color=colors[arm])
                ax.fill_between(curve.index, curve['mean'] - curve['std'], curve['mean'] + curve['std'], color=colors[arm], alpha=.15)
            ax.set_title(f'{dataset} / {metric}')
            ax.set_xlabel('Epoch; last actual batch only')
            ax.grid(alpha=.2); ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'momentum_diagnostics.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
