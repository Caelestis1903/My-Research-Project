"""Rebuild GA31-R3 report tables and figures from numeric records."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def summarize_seeds(frame, keys, excluded=()):
    columns = [c for c in frame.select_dtypes(include='number').columns
               if c not in set(keys) | set(excluded) | {'seed'}]
    table = frame.groupby(keys)[columns].agg(['mean', 'std', 'count'])
    table.columns = ['_'.join(c) for c in table.columns]
    return table.reset_index()


def collect(root, out):
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((root / 'selection_frozen.json').read_text())
    baseline = json.loads((root / 'baseline_frozen.json').read_text())
    candidate = json.loads((root / 'candidate_frozen.json').read_text())
    completion = json.loads((root / 'completion.json').read_text())
    assert completion['status'] == 'COMPLETE'
    selections = frozen['selected']
    count = len(selections)
    assert count in (35, 40, 45)
    test = pd.read_csv(root / 'test_metrics.csv')
    validation = pd.read_csv(root / 'validation_metrics.csv')
    assert len(test) == len(validation) == 3 * count
    assert len(test[['dataset', 'arm', 'seed', 'checkpoint']].drop_duplicates()) == 3 * count
    test.to_csv(out / 'TEST_METRICS.csv', index=False)
    validation.to_csv(out / 'VALIDATION_METRICS.csv', index=False)
    for source, target in [('paired_effects.csv', 'PAIRED_EFFECTS.csv'),
                           ('effect_summary.csv', 'EFFECT_SUMMARY.csv'),
                           ('arm_summary.csv', 'ARM_SUMMARY.csv')]:
        pd.read_csv(root / source).to_csv(out / target, index=False)

    curves, costs, selected, temporal, increments, counterfactuals, scales, view_probes = [], [], [], [], [], [], [], []
    for item in selections:
        key = {k: item[k] for k in ('dataset', 'arm', 'seed')}
        directory = Path(item['output_dir'])
        if not directory.exists():
            directory = root / 'runs' / key['dataset'] / key['arm'] / f"seed{key['seed']}" / directory.name
        runtime = json.loads((directory / 'runtime_stats.json').read_text())
        assert runtime['epochs'] == 50
        cost = {k: v for k, v in runtime.items() if isinstance(v, (int, float, str, bool))}
        config = json.loads((directory / 'resolved_config.json').read_text())
        cost['device'] = config['device']
        cost['loss_scale'] = config['config']['loss_scale']
        cost['weight_decay'] = config['config']['weight_decay']
        cost['eps'] = config['config']['eps']
        cost['num_views'] = config['config']['num_views']
        cost['online_probe_seconds'] = runtime['probe']['seconds']
        cost['online_probe_gradients'] = runtime['probe']['gradients']
        costs.append(dict(cost, **key))
        curve = pd.read_csv(directory / 'epoch_metrics.csv')
        assert curve.epoch.tolist() == list(range(1, 51))
        curves.append(curve.assign(**key))
        row = test[(test.dataset == key['dataset']) & (test.arm == key['arm']) &
                   (test.seed == key['seed']) & (test.checkpoint == 'selected')].iloc[0].to_dict()
        selected.append(dict(row, selected_epoch=item['selected_epoch'],
                             selected_validation_auc=item['selected_validation']['macro_auc'],
                             selected_validation_logloss=item['selected_validation']['macro_logloss']))
        for filename, destination in [('temporal_probe.csv', temporal), ('optimizer_scale.csv', scales)]:
            if (directory / filename).exists():
                frame = pd.read_csv(directory / filename).assign(**key)
                if 'source_immutable' in frame:
                    assert frame.source_immutable.all()
                if filename in ('increment_probe.csv', 'counterfactual_update.csv', 'view_probe.csv'):
                    frame['is_E1'] = frame.epoch == 1
                    frame['is_E10'] = frame.epoch == 10
                    frame['is_selected'] = frame.epoch == item['selected_epoch']
                destination.append(frame)

    by_key = {(s['dataset'], s['arm'], s['seed']): s for s in selections}
    for directory in sorted((root / 'probes').glob('*/*/seed*/attempt*')):
        dataset, arm, seed = directory.parts[-4:-1]
        key = dict(dataset=dataset, arm=arm, seed=int(seed[4:]))
        item = by_key[dataset, arm, key['seed']]
        for filename, destination in [('increment_probe.csv', increments),
                                       ('counterfactual_update.csv', counterfactuals),
                                       ('view_probe.csv', view_probes)]:
            frame = pd.read_csv(directory / filename).assign(**key)
            assert frame.source_immutable.all()
            frame['is_E1'] = frame.epoch == 1
            frame['is_E10'] = frame.epoch == 10
            frame['is_selected'] = frame.epoch == item['selected_epoch']
            destination.append(frame)

    curve = pd.concat(curves, ignore_index=True)
    assert len(curve) == count * 50
    curve.to_csv(out / 'EPOCH_METRICS.csv', index=False)
    cost_frame = pd.DataFrame(costs)
    cost_frame.to_csv(out / 'COSTS.csv', index=False)
    assert int(cost_frame.global_step.sum()) == completion['original_optimizer_steps']
    pd.DataFrame(selected).sort_values(['dataset', 'arm', 'seed']).to_csv(out / 'SELECTED_RUNS.csv', index=False)
    if scales:
        pd.concat(scales, ignore_index=True).to_csv(out / 'OPTIMIZER_SCALE_OBSERVATIONS.csv', index=False)

    time_rows = pd.concat(temporal, ignore_index=True)
    assert len(time_rows) == 120 and time_rows.extra_gradients.sum() == 360
    time_rows['working_baseline'] = time_rows.apply(lambda r: r.arm == baseline['baselines'][r.dataset], axis=1)
    time_rows.to_csv(out / 'TEMPORAL_OBSERVATIONS.csv', index=False)
    time_numeric = ['delta1_norm', 'delta2_norm', 'mean_delta_norm', 'view_variance', 'S', 'C_previous', 'signal_normalized_C']
    time_seed = time_rows.groupby(['dataset', 'arm', 'seed', 'epoch'])[time_numeric].mean().reset_index()
    time_seed.to_csv(out / 'TEMPORAL_SEED.csv', index=False)
    summarize_seeds(time_seed, ['dataset', 'arm', 'epoch']).to_csv(out / 'TEMPORAL_SUMMARY.csv', index=False)

    increment = pd.concat(increments, ignore_index=True)
    assert len(increment) <= 60
    increment.to_csv(out / 'INCREMENT_OBSERVATIONS.csv', index=False)
    seed_records = []
    numeric = [c for c in increment.select_dtypes(include='number').columns
               if c not in ('seed', 'epoch', 'global_step', 'batch')]
    for anchor in ('E1', 'E10', 'selected'):
        part = increment[increment['is_' + anchor]]
        table = part.groupby(['dataset', 'arm', 'seed'])[numeric].mean().reset_index()
        table['anchor'] = anchor
        seed_records.append(table)
    increment_seed = pd.concat(seed_records, ignore_index=True)
    increment_seed.to_csv(out / 'INCREMENT_SEED.csv', index=False)
    summarize_seeds(increment_seed, ['dataset', 'arm', 'anchor']).to_csv(out / 'INCREMENT_SUMMARY.csv', index=False)

    cf = pd.concat(counterfactuals, ignore_index=True)
    identity = ['dataset', 'arm', 'seed', 'epoch', 'batch', 'panel']
    utility = [c for c in cf.columns if c.startswith('delta_')]
    real = cf[cf.branch == 'real'].set_index(identity)[utility]
    assert real.index.is_unique
    for column in utility:
        reference = cf.set_index(identity).index.map(real[column])
        cf['extra_' + column] = cf[column].to_numpy() - np.asarray(reference)
    cf.to_csv(out / 'COUNTERFACTUAL_OBSERVATIONS.csv', index=False)
    cf_numeric = utility + ['extra_' + c for c in utility] + [
        'update_norm', 'update_difference_norm', 'update_cosine_to_real', 'm_change_norm', 'v_change_norm']
    cf_records = []
    for anchor in ('E1', 'E10', 'selected'):
        part = cf[cf['is_' + anchor]]
        table = part.groupby(['dataset', 'arm', 'seed', 'branch', 'panel'])[cf_numeric].mean().reset_index()
        table['anchor'] = anchor
        cf_records.append(table)
    cf_seed = pd.concat(cf_records, ignore_index=True)
    cf_seed.to_csv(out / 'COUNTERFACTUAL_SEED.csv', index=False)
    summarize_seeds(cf_seed, ['dataset', 'arm', 'anchor', 'branch', 'panel']).to_csv(out / 'COUNTERFACTUAL_SUMMARY.csv', index=False)
    if view_probes:
        views = pd.concat(view_probes, ignore_index=True)
        views.to_csv(out / 'VIEW_OBSERVATIONS.csv', index=False)

    profiles = []
    for dataset in ('movielens', 'kuairand'):
        manifest = json.loads((root / 'data' / dataset / 'data_manifest.json').read_text())
        for split, cells in manifest['cell_stats'].items():
            for cell, record in cells.items():
                profiles.append(dict(dataset=dataset, split=split, cell=cell, **record))
    pd.DataFrame(profiles).to_csv(out / 'DOMAIN_PROFILE.csv', index=False)
    probe_ledger = json.loads((root / 'probe_ledger.json').read_text())
    assert probe_ledger['offline_gradients'] == len(increment) * 14
    assert probe_ledger['offline_counterfactual_steps'] == len(cf) // 2
    facts = dict(completion=completion, baselines=baseline, candidate=candidate,
                 formal_runs=count, epochs=len(curve), temporal_observations=len(time_rows),
                 temporal_gradients=int(time_rows.extra_gradients.sum()), offline_batches=len(increment),
                 offline_gradients=probe_ledger['offline_gradients'], counterfactual_steps=probe_ledger['offline_counterfactual_steps'],
                 original_exposures=int(cost_frame.original_exposures.sum()), extra_views=int(cost_frame.extra_views.sum()),
                 training_forwards=int(cost_frame.training_forwards_including_validation.sum()),
                 training_backwards=int(cost_frame.training_backwards.sum()),
                 worker_seconds_including_online=float(cost_frame.training_seconds.sum()),
                 online_probe_worker_seconds=float(cost_frame.online_probe_seconds.sum()),
                 offline_probe_worker_seconds=probe_ledger['offline_worker_seconds'],
                 max_peak_memory_bytes=int(cost_frame.peak_memory_bytes.max()),
                 snapshot_tensor_bytes=int(cost_frame.snapshot_tensor_bytes.sum()))
    (out / 'ANALYSIS_FACTS.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2) + '\n')
    figures(curve, increment_seed, pd.read_csv(root / 'effect_summary.csv'),
            pd.read_csv(root / 'paired_effects.csv'), out)


def figures(curve, increment, effects, pairs, out):
    palette = {'B-ML': '#315c9a', 'L2-ML': '#7958a8', 'B-KR': '#315c9a',
               'M1': '#e78b2c', 'G-IID2': '#159489', 'G-ANTI2': '#b94b66'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for row, dataset in enumerate(('movielens', 'kuairand')):
        for col, metric in enumerate(('macro_auc', 'macro_logloss')):
            ax = axes[row, col]
            for arm, frame in curve[curve.dataset == dataset].groupby('arm', sort=False):
                table = frame.pivot(index='epoch', columns='seed', values='val_' + metric)
                assert table.shape == (50, 5)
                mean, sd = table.mean(axis=1), table.std(axis=1, ddof=1)
                ax.plot(table.index, mean, label=arm, color=palette[arm])
                ax.fill_between(table.index, mean - sd, mean + sd, color=palette[arm], alpha=.12)
            ax.set(title=f'{dataset}: validation {metric}', xlabel='Epoch')
            ax.grid(alpha=.2)
            if col == 0:
                ax.legend(fontsize=8, ncol=3)
    fig.savefig(out / 'VALIDATION_CURVES.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    order = ['M1-baseline', 'G-IID2-baseline', 'G-ANTI2-baseline', 'G-ANTI2-G-IID2', 'L2-ML-B-ML']
    for row, dataset in enumerate(('movielens', 'kuairand')):
        for col, metric in enumerate(('macro_auc', 'macro_logloss')):
            ax = axes[row, col]
            part = effects[(effects.dataset == dataset) & (effects.split == 'test') &
                           (effects.checkpoint == 'selected') & (effects.metric == metric)].set_index('effect')
            names = [name for name in order if name in part.index]
            part = part.loc[names]
            for index, name in enumerate(names):
                values = pairs[(pairs.dataset == dataset) & (pairs.split == 'test') &
                               (pairs.checkpoint == 'selected') & (pairs.metric == metric) &
                               (pairs.effect == name)].sort_values('seed').value.to_numpy()
                assert len(values) == 5
                ax.scatter(values, index + np.linspace(-.1, .1, 5), s=15, color='#a1aab8')
            ax.errorbar(part['mean'], range(len(names)),
                        xerr=np.stack([part['mean'] - part.ci_low, part.ci_high - part['mean']]),
                        fmt='o', capsize=3, color='#315c9a')
            ax.axvline(0, color='black', linewidth=.8)
            ax.set(yticks=range(len(names)), yticklabels=names, title=f'{dataset}: selected test {metric}',
                   xlabel='Paired difference; 95% t interval, n=5')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=.2)
            ax.xaxis.set_major_locator(MaxNLocator(5))
    fig.savefig(out / 'SELECTED_EFFECTS.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, dataset in zip(axes, ('movielens', 'kuairand')):
        part = increment[(increment.dataset == dataset) & (increment.anchor == 'selected')]
        for index, (_, r) in enumerate(part.sort_values('seed').iterrows()):
            ax.scatter(r.gaussian_variance, r.gaussian_signal, color='#159489', label='Gaussian' if index == 0 else None)
            ax.scatter(r.mask_variance, r.mask_signal, color='#e78b2c', label='FieldMask-1' if index == 0 else None, marker='s')
        ax.axhline(0, color='black', linewidth=.7)
        ax.set_xscale('log')
        ax.set_yscale('symlog', linthresh=1e-10)
        ax.set(xlabel='Conditional view variance (seed average)', ylabel='Signed signal estimate', title=dataset)
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(out / 'INCREMENT_SIGNAL_VARIANCE.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.run_dir, args.output)
