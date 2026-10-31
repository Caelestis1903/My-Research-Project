"""Rebuild report tables and figures from the completed GA31-v2 numeric records."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(root, out):
    out.mkdir(parents=True, exist_ok=True)
    completion = json.loads((root / 'completion.json').read_text())
    assert completion['status'] == 'COMPLETE' and completion['completed'] == 40
    assert completion['test_evaluated'] and completion['epochs_each'] == 50
    frozen = json.loads((root / 'selection_frozen.json').read_text())
    test = pd.read_csv(root / 'test_metrics.csv')
    validation = pd.read_csv(root / 'validation_metrics.csv')
    effects = pd.read_csv(root / 'effect_summary.csv')
    pairs = pd.read_csv(root / 'paired_effects.csv')
    assert len(test) == len(validation) == 120
    assert len(test[['dataset', 'arm', 'seed', 'checkpoint']].drop_duplicates()) == 120
    test.to_csv(out / 'TEST_METRICS.csv', index=False)
    validation.to_csv(out / 'VALIDATION_METRICS.csv', index=False)
    effects.to_csv(out / 'EFFECT_SUMMARY.csv', index=False)
    pairs.to_csv(out / 'PAIRED_EFFECTS.csv', index=False)
    pd.read_csv(root / 'arm_summary.csv').to_csv(out / 'ARM_SUMMARY.csv', index=False)

    profiles = []
    for dataset in ('movielens', 'kuairand'):
        manifest = json.loads((root / 'data' / dataset / 'data_manifest.json').read_text())
        for split, cells in manifest['cell_stats'].items():
            for cell, record in cells.items():
                profiles.append(dict(dataset=dataset, split=split, cell=cell, **record))
    pd.DataFrame(profiles).to_csv(out / 'DOMAIN_PROFILE.csv', index=False)

    runs, costs, probe_seeds, epochs = [], [], [], []
    for item in frozen['selected']:
        directory = Path(item['output_dir'])
        if not directory.is_absolute() and not directory.exists():
            directory = root / 'runs' / item['dataset'] / item['arm'] / f"seed{item['seed']}" / directory.name
        runtime = json.loads((directory / 'runtime_stats.json').read_text())
        config = json.loads((directory / 'resolved_config.json').read_text())
        key = {k: item[k] for k in ('dataset', 'arm', 'seed')}
        selected = test[(test.dataset == key['dataset']) & (test.arm == key['arm']) &
                        (test.seed == key['seed']) & (test.checkpoint == 'selected')]
        assert len(selected) == 1 and runtime['epochs'] == 50
        runs.append(dict(**key, selected_epoch=item['selected_epoch'], selected_step=item['global_step'],
                         validation_auc=item['selected_validation']['macro_auc'],
                         validation_logloss=item['selected_validation']['macro_logloss'],
                         training_device=config['device'], **selected.iloc[0].drop(labels=list(key)).to_dict()))
        costs.append(dict(**key, device=config['device'], training_seconds=runtime['training_seconds'],
                          probe_seconds=runtime['probe']['seconds'], probe_batches=runtime['probe']['batches'],
                          probe_backwards=runtime['probe']['backwards'], probe_forwards=runtime['probe']['forwards'],
                          training_backwards=runtime['training_backwards'],
                          training_forwards=runtime['training_forwards_including_validation'],
                          original_exposures=runtime['original_exposures'], extra_views=runtime['extra_views'],
                          steps=runtime['global_step'], peak_memory_bytes=runtime['peak_memory_bytes'],
                          snapshot_tensor_bytes=runtime['snapshot_tensor_bytes']))
        assert runtime['probe']['source_immutable'] and runtime['probe']['rng_restored']
        curve = pd.read_csv(directory / 'epoch_metrics.csv')
        assert curve.epoch.tolist() == list(range(1, 51))
        epochs.append(curve.assign(**key))
        probe = pd.read_csv(directory / 'gradient_probe.csv')
        assert probe.source_immutable.all()
        probe['mix_extra_validation_delta'] = probe.mix_validation_delta - probe.real_validation_delta
        probe['aug_extra_validation_delta'] = probe.extended_aug_validation_delta - probe.real_validation_delta
        for anchor, epoch in [('E1', 1), ('E10', 10), ('selected', item['selected_epoch'])]:
            portion = probe[probe.epoch == epoch]
            numeric = [c for c in portion.select_dtypes(include='number').columns
                       if c not in ('seed', 'epoch', 'global_step', 'batch', 'noise_seed')]
            for block, group in portion.groupby('block'):
                probe_seeds.append(dict(**key, anchor=anchor, epoch=epoch, block=block,
                                        batches=len(group), **group[numeric].mean().to_dict()))

    run_frame = pd.DataFrame(runs).sort_values(['dataset', 'seed', 'arm'])
    run_frame.to_csv(out / 'SELECTED_RUNS.csv', index=False)
    cost_frame = pd.DataFrame(costs).sort_values(['dataset', 'seed', 'arm'])
    cost_frame.to_csv(out / 'COSTS.csv', index=False)
    probe_frame = pd.DataFrame(probe_seeds)
    probe_frame.to_csv(out / 'PROBE_SEED.csv', index=False)
    numeric = [c for c in probe_frame.select_dtypes(include='number').columns
               if c not in ('seed', 'epoch', 'batches')]
    summary = probe_frame.groupby(['dataset', 'arm', 'anchor', 'block'])[numeric].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(out / 'PROBE_SUMMARY.csv')
    curve_frame = pd.concat(epochs, ignore_index=True)
    curve_frame.to_csv(out / 'EPOCH_METRICS.csv', index=False)
    assert len(curve_frame) == 2000 and cost_frame.steps.sum() == 69000
    assert cost_frame.original_exposures.sum() == 2806299000
    assert cost_frame.extra_views.sum() == 122700500
    launch = pd.read_csv(root / 'launch_ledger.csv')
    facts = dict(completion=completion, working_scale=frozen['working_scale'],
                 original_exposures=int(cost_frame.original_exposures.sum()),
                 extra_views=int(cost_frame.extra_views.sum()), adam_steps=int(cost_frame.steps.sum()),
                 training_worker_seconds=float(cost_frame.training_seconds.sum()),
                 probe_worker_seconds=float(cost_frame.probe_seconds.sum()),
                 probe_batches=int(cost_frame.probe_batches.sum()),
                 probe_backwards=int(cost_frame.probe_backwards.sum()),
                 probe_forwards=int(cost_frame.probe_forwards.sum()),
                 counterfactual_adam_steps=int(3 * cost_frame.probe_batches.sum()),
                 first_launch_unix=float(launch.time.min()),
                 final_numeric_write_unix=(root / 'completion.json').stat().st_mtime,
                 max_process_peak_bytes=int(cost_frame.peak_memory_bytes.max()),
                 snapshots_bytes=int(cost_frame.snapshot_tensor_bytes.sum()),
                 selected_epoch_ranges=run_frame.groupby(['dataset', 'arm']).selected_epoch.agg(['min', 'max']).reset_index().to_dict('records'))
    (out / 'ANALYSIS_FACTS.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2) + '\n')

    colors = dict(P0='#315c9a', P1='#159489', P2='#e78b2c', P3='#b94b66')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for row, dataset in enumerate(('movielens', 'kuairand')):
        for col, metric in enumerate(('macro_auc', 'macro_logloss')):
            ax = axes[row, col]
            for arm, color in colors.items():
                frame = curve_frame[(curve_frame.dataset == dataset) & (curve_frame.arm == arm)]
                table = frame.pivot(index='epoch', columns='seed', values='val_' + metric)
                assert table.shape == (50, 5)
                mean, sd = table.mean(axis=1), table.std(axis=1, ddof=1)
                ax.plot(table.index, mean, label=arm, color=color)
                ax.fill_between(table.index, mean - sd, mean + sd, color=color, alpha=.13)
            ax.set(title=f'{dataset}: validation {metric}', xlabel='Epoch')
            ax.grid(alpha=.2)
            if row == 0 and col == 0: ax.legend(ncol=4, fontsize=9)
    fig.savefig(out / 'VALIDATION_CURVES.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    names = ['A1', 'A0.1', 'S0', 'S1', 'I']
    for row, dataset in enumerate(('movielens', 'kuairand')):
        for col, metric in enumerate(('macro_auc', 'macro_logloss')):
            ax = axes[row, col]
            data = effects[(effects.subset == 'five_seed') & (effects.split == 'test') &
                           (effects.checkpoint == 'selected') & (effects.dataset == dataset) &
                           (effects.metric == metric)].set_index('effect').loc[names]
            for index, name in enumerate(names):
                values = pairs[(pairs.subset == 'five_seed') & (pairs.split == 'test') &
                               (pairs.checkpoint == 'selected') & (pairs.dataset == dataset) &
                               (pairs.metric == metric) & (pairs.effect == name)].sort_values('seed').value
                ax.scatter(values, index + np.linspace(-.11, .11, 5), s=15, color='#9ca6b8')
            ax.errorbar(data['mean'], range(5),
                        xerr=np.stack([data['mean'] - data.ci_low, data.ci_high - data['mean']]),
                        fmt='o', capsize=3, color='#315c9a')
            ax.axvline(0, color='black', linewidth=.8)
            ax.set(yticks=range(5), yticklabels=names, title=f'{dataset}: selected test {metric}',
                   xlabel='Paired difference; 95% t interval, n=5')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=.2)
    fig.savefig(out / 'FACTORIAL_EFFECTS.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    main(args.run_dir, args.output)
