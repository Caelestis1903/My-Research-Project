"""Parent-owned analysis of the completed, registered cycle01 matrix."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_reports.task_view_gradient_residual_momentum_20261116.ANALYSIS import markdown


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ATTEMPT = ROOT / 'artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1'
TABLES = ATTEMPT / 'tables'
SEEDS = {2022, 2023, 2024, 2025, 2026}
DATASETS = ('movielens', 'kuairand')
ARMS = ('CLEAN', 'NDP-A', 'AXIS-EVEN', 'AXIS-IID2', 'ISOTROPIC-EVEN')
CELLS = [f'd{d}t{t}' for d in range(3) for t in range(2)]
PAIR_KEYS = ['dataset', 'arm', 'control', 'endpoint', 'kind', 'reference_kind', 'metric']


def main():
    progress = json.loads((ATTEMPT / 'progress.json').read_text())
    receipt = json.loads((ATTEMPT / 'TABLES_RECEIPT.json').read_text())
    manifest = json.loads((ATTEMPT / 'RESOURCE_1_SUPERSESSION.json').read_text())['run_manifest']
    assert progress['status'] == 'COMPLETE' and progress['complete'] == 30
    assert receipt['status'] == 'COMPLETE' and receipt['new_optimizer_updates'] == 51750
    assert receipt['source_hash'] == manifest['source_hash'] and not receipt['test_accessed']
    endpoints = pd.read_csv(TABLES / 'ENDPOINTS.csv')
    paired = pd.read_csv(TABLES / 'PAIRED_SUMMARY.csv')
    deltas = pd.read_csv(TABLES / 'PAIRED_SEEDS.csv')
    costs = pd.read_csv(TABLES / 'COSTS.csv')
    diagnostics = pd.read_csv(TABLES / 'GRADIENT_DIAGNOSTICS.csv')
    selected = endpoints[endpoints.endpoint.eq('selected')].copy()
    expected = {(d, a, s, k) for d in DATASETS for a in ARMS
                for s in SEEDS for k in ('online', 'fixed_ema')}
    assert len(endpoints) == 400 and len(selected) == len(expected) == 100
    assert set(zip(selected.dataset, selected.arm, selected.seed, selected.kind)) == expected
    assert not endpoints.duplicated(['dataset', 'arm', 'seed', 'endpoint', 'kind']).any()
    assert endpoints.split.eq('validation').all()
    for metric in ('auc', 'logloss'):
        assert np.allclose(endpoints[[f'{c}_{metric}' for c in CELLS]].mean(axis=1),
                           endpoints[f'macro_{metric}'], rtol=0, atol=1e-12)
    assert not deltas.duplicated(PAIR_KEYS + ['seed']).any()
    assert np.allclose(deltas.candidate - deltas.reference, deltas.delta, rtol=0, atol=1e-12)
    groups = deltas.groupby(PAIR_KEYS).delta
    stats = groups.agg(['mean', 'std', 'size'])
    reference = paired.set_index(PAIR_KEYS)
    stats = stats.reindex(reference.index)
    assert len(stats) == len(reference) and stats['size'].eq(5).all()
    radius = 2.7764451051977987 * stats['std'] / np.sqrt(5)
    for calculated, recorded in ((stats['mean'], reference['mean']),
                                 (stats['std'], reference.sample_sd),
                                 (stats['mean'] - radius, reference.ci95_low),
                                 (stats['mean'] + radius, reference.ci95_high)):
        assert np.allclose(calculated, recorded, rtol=0, atol=1e-12)
    for sign, predicate in (('positive_seeds', lambda x: x > 0),
                             ('negative_seeds', lambda x: x < 0),
                             ('zero_seeds', lambda x: x == 0)):
        assert groups.apply(lambda x: int(predicate(x).sum())).reindex(reference.index).eq(reference[sign]).all()

    metrics = [c for c in endpoints if c.endswith(('_auc', '_logloss'))]
    group_keys = ['dataset', 'arm', 'endpoint', 'kind']
    means = endpoints.groupby(group_keys, sort=False)[metrics + ['epoch']].mean().reset_index()
    means.rename(columns={'epoch': 'selected_or_fixed_epoch_mean'}, inplace=True)
    means['n'] = 5
    means.to_csv(HERE / 'STAGE_CELL_MEANS.csv', index=False)
    selected_means = means[means.endpoint.eq('selected')].copy()
    selected_means.to_csv(HERE / 'SELECTED_CELL_MEANS.csv', index=False)
    primary = paired[paired.endpoint.eq('selected') & paired.kind.eq('fixed_ema')
                     & paired.metric.isin(['macro_auc', 'macro_logloss'])].copy()
    assert len(primary) == 16
    primary.to_csv(HERE / 'PRIMARY_PAIRED.csv', index=False)

    populations = {d: json.loads((ATTEMPT / f'{d}_data_manifest.json').read_text())['train_domain_counts']
                   for d in DATASETS}
    mean_lookup = means.set_index(group_keys)
    domain_task = paired[paired.metric.str.match(
        r'(d[0-2]t[01]|domain[0-2]_macro|task[01]_macro)_(auc|logloss)$')].copy()
    counts = []
    baseline_means = []
    for row in domain_task.itertuples():
        domain = next((d for d in range(3) if row.metric.startswith((f'd{d}t', f'domain{d}_'))), None)
        counts.append(None if domain is None else populations[row.dataset][domain])
        baseline_means.append(mean_lookup.loc[
            (row.dataset, row.control, row.endpoint, row.kind), row.metric])
    domain_task['train_domain_rows'] = counts
    domain_task['control_metric_mean'] = baseline_means
    domain_task.to_csv(HERE / 'DOMAIN_TASK_PAIRED.csv', index=False)

    flat = pd.json_normalize(diagnostics.diagnostics.map(json.loads))
    diagnostic_columns = [c for c in (
        'E', 'O', 'gradient.total_data_gradient_norm', 'gradient.actual_update_l2_norm',
        'probe.task0.delta_norm.mean', 'probe.task1.delta_norm.mean',
        'replicas.task0.input_norm_relative_error', 'replicas.task1.input_norm_relative_error')
        if c in flat]
    diagnostic_values = pd.concat(
        [diagnostics[['dataset', 'arm', 'seed', 'epoch']].reset_index(drop=True), flat[diagnostic_columns]], axis=1)
    diagnostic_means = diagnostic_values.groupby(['dataset', 'arm', 'epoch'])[diagnostic_columns].mean()
    diagnostic_means['n_seeds'] = diagnostic_values.groupby(['dataset', 'arm', 'epoch']).seed.nunique()
    assert diagnostic_means.n_seeds.eq(5).all()
    diagnostic_means.reset_index().to_csv(HERE / 'DIAGNOSTIC_MEANS.csv', index=False)

    new_costs = costs[~costs.arm.isin(['CLEAN', 'NDP-A'])]
    assert len(new_costs) == 30 and int(new_costs.steps.sum()) == 51750
    assert int(new_costs.epochs.sum()) == 1500
    cost_means = costs.groupby(['dataset', 'arm'])[
        ['worker_seconds', 'steps', 'peak_reserved_bytes', 'peak_allocated_bytes']].mean()
    cost_means['seconds_per_update'] = cost_means.worker_seconds / cost_means.steps
    cost_means.reset_index().to_csv(HERE / 'COST_MEANS.csv', index=False)
    scores = primary[primary.control.eq('CLEAN') & primary.metric.eq('macro_auc')].set_index('dataset')
    positive = {d: bool(scores.loc[d, 'mean'] > 0) for d in DATASETS}
    outcome = ('PROVISIONAL_CROSS_DATASET_SIGNAL' if all(positive.values()) else
               'MIXED_SIGNAL' if any(positive.values()) else 'NO_CONTINUATION_SIGNAL')
    control_signals = []
    for dataset in DATASETS:
        clean = selected[selected.dataset.eq(dataset) & selected.arm.eq('CLEAN')
                         & selected.kind.eq('fixed_ema')].set_index('seed')
        for arm in ('AXIS-IID2', 'ISOTROPIC-EVEN'):
            rows = selected[selected.dataset.eq(dataset) & selected.arm.eq(arm)
                            & selected.kind.eq('fixed_ema')].set_index('seed')
            x = rows.macro_auc-clean.macro_auc
            radius = 2.7764451051977987*x.std(ddof=1)/np.sqrt(5)
            control_signals.append(dict(dataset=dataset, arm=arm, mean=float(x.mean()),
                ci95_low=float(x.mean()-radius), ci95_high=float(x.mean()+radius),
                positive_seeds=int((x > 0).sum()), negative_seeds=int((x < 0).sum()), n=5))
    attempts = pd.read_csv(TABLES/'PHYSICAL_ATTEMPTS.csv')
    launches = [json.loads(s) for s in (ATTEMPT/'launch_ledger.jsonl').read_text().splitlines()]
    exits = [json.loads(s) for s in (ATTEMPT/'worker_exit_ledger.jsonl').read_text().splitlines()]
    assert len(launches) == len(exits) == len(attempts)
    decision = dict(candidate='AXIS-EVEN', candidate_decision=outcome,
        all_primary_dataset_nonpositive_gate_triggered=not any(positive.values()),
        primary_delta_macro_auc={d: float(scores.loc[d, 'mean']) for d in DATASETS},
        primary_paired=primary.to_dict(orient='records'), train_domain_counts=populations,
        observed_unregistered_controls=control_signals,
        training_source_hash=manifest['source_hash'], design_bundle_sha256=manifest['design_bundle_sha256'],
        execution_policy_history_sha256=hashlib.sha256(
            (ATTEMPT/'CURRENT_UTILIZATION_EXECUTION_POLICY.jsonl').read_bytes()).hexdigest(),
        champion=None, test_accessed=False, test_status='SEALED', normal_termination='USER_STOPPED',
        cost=dict(new_trajectories=30, reused_trajectories=20, epochs=1500,
                  optimizer_updates=51750, ema_updates=51750,
                  worker_seconds=float(new_costs.worker_seconds.sum()),
                  measured_attempt_worker_seconds=float(new_costs.measured_attempt_worker_seconds.sum()),
                  physical_attempt_status_counts={str(k): int(v) for k, v in attempts.status.value_counts().items()},
                  physical_worker_span_seconds=max(s['observed_at'] for s in exits)-min(s['time'] for s in launches)),
        source_tables={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(TABLES.glob('*.csv'))},
        next_action='Parent scientific interpretation, context publication, then fresh native Astra in this conversation.')
    (HERE / 'DECISION.json').write_text(json.dumps(decision, indent=2, allow_nan=False) + '\n')

    columns = ['arm', 'n', 'selected_or_fixed_epoch_mean'] + [f'{c}_auc' for c in CELLS] + [
        'macro_auc', 'macro_logloss', 'sample_weighted_cell_auc', 'pooled_task_mean_auc', 'worst_cell_auc']
    sections = ['# Task-axis even views numerical appendix',
                'Validation; seeds2022–2026, n=5. AUC deltas above zero and LogLoss deltas below zero '
                'favor AXIS-EVEN. Paired intervals are descriptive; seed is the statistical unit.']
    ledger = ['## 2026-11-16 Task-axis even views：完整五seed验证记录',
              'M3oE fixed1，自然人口，50epochs/batch40960/FP32；30条新增、20条CLEAN/NDP-A历史对照。'
              'Online/EMA独立选点；注册候选仅AXIS-EVEN。完整数字见research_reports/task_axis_even_views_20261116。']
    for dataset in DATASETS:
        for kind in ('online', 'fixed_ema'):
            rows = selected_means[selected_means.dataset.eq(dataset) & selected_means.kind.eq(kind)]
            sections += [f'## {dataset} / {kind} / selected means', markdown(rows[columns])]
            per_seed = selected[selected.dataset.eq(dataset) & selected.kind.eq(kind)]
            sections += [f'## {dataset} / {kind} / selected per seed',
                         markdown(per_seed[['arm', 'seed', 'epoch', 'macro_auc', 'macro_logloss', 'model_hash']])]
            compact = rows[['arm', 'n']].copy()
            for cell in CELLS:
                compact[cell + ' AUC / LL'] = rows.apply(
                    lambda r: f"{r[cell + '_auc']:.9f} / {r[cell + '_logloss']:.9f}", axis=1)
            compact['Macro AUC / LL'] = rows.apply(
                lambda r: f'{r.macro_auc:.9f} / {r.macro_logloss:.9f}', axis=1)
            ledger += [f'### {dataset} / {kind}', markdown(compact)]
    show = ['dataset', 'control', 'metric', 'mean', 'sample_sd', 'ci95_low', 'ci95_high',
            'positive_seeds', 'negative_seeds', 'zero_seeds']
    sections += ['## Selected fixed-EMA paired contrasts', markdown(primary[show]),
                 '## Common-epoch comparisons', markdown(paired[
                     paired.kind.eq('fixed_ema') & paired.metric.isin(['macro_auc', 'macro_logloss'])
                     & ~paired.endpoint.eq('selected')][['endpoint'] + show]),
                 '## Worker costs', markdown(cost_means.reset_index())]
    (HERE / 'NUMERICAL_APPENDIX.md').write_text('\n\n'.join(sections) + '\n')
    (HERE / 'LEDGER_ENTRY.md').write_text('\n\n'.join(ledger) + '\n')
    print(json.dumps({k: decision[k] for k in ('candidate_decision', 'primary_delta_macro_auc', 'cost')}, indent=2))


if __name__ == '__main__':
    main()
