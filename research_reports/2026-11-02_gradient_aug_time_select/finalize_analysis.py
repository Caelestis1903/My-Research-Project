"""Post-run numerical presentation only; never changes selection or launches training.

Run from repository root after completion.json and state_release.json exist.
Raw counters stay unchanged. The presentation explicitly marks the known mask-only
energy measurement scope, reviewed in DIAGNOSTIC_ACCOUNTING_REVIEW.md.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.gradient_aug_time_select_20261102.analyze import summarize

REPORT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    RUN = parser.parse_args().run_dir
    completion = json.loads((RUN / 'completion.json').read_text())
    release = json.loads((RUN / 'state_release.json').read_text())
    assert completion['test_evaluated'] and release['released']
    tables = REPORT / 'tables'
    decisions = REPORT / 'decisions'
    tables.mkdir(exist_ok=True)
    decisions.mkdir(exist_ok=True)
    for name in ('validation_metrics.csv', 'test_metrics.csv', 'selected_metrics.csv',
                 'paired_effects.csv', 'effect_summary.csv', 'arm_summary.csv',
                 'trial_ledger.csv', 'costs.csv', 'test_costs.csv',
                 'failure_ledger.csv', 'launch_ledger.csv', 'validation_epoch_metrics.csv'):
        shutil.copyfile(RUN / name, tables / name)
    for path in RUN.glob('*.json'):
        if any(word in path.stem for word in ('decision', 'frozen', 'completion', 'release',
                                             'historical', 'analysis_receipt', 'throughput')):
            shutil.copyfile(path, decisions / path.name)

    # Arm summaries use seeds, never rows/epochs/cells, as repetition units.
    summaries, population, worst = [], [], []
    extra_paired, extra_summary = [], []
    for split in ('validation', 'test'):
        frame = pd.read_csv(RUN / f'{split}_metrics.csv')
        assert not frame.duplicated(['dataset', 'arm', 'seed', 'model_kind', 'checkpoint']).any()
        metrics = [c for c in frame if c.endswith(('_auc', '_logloss'))]
        cells = [f'd{d}t{t}' for d in range(3) for t in range(2)]
        for (dataset, endpoint), part in frame.groupby(['dataset', 'checkpoint']):
            groups = {(a, k): g.set_index('seed') for (a, k), g in part.groupby(['arm', 'model_kind'])}
            contrasts = [('A0:delayed-fixed', [('A0', 'delayed5', 1), ('A0', 'fixed_ema', -1)]),
                         ('A0:swa-fixed', [('A0', 'swa20', 1), ('A0', 'fixed_ema', -1)])]
            for arm in ('B1', 'B2', 'B3'):
                contrasts.append((arm + ':EMA_interaction_vs_A1', [(arm, 'fixed_ema', 1),
                    (arm, 'online', -1), ('A1', 'fixed_ema', -1), ('A1', 'online', 1)]))
            for effect, terms in contrasts:
                if any((a, k) not in groups for a, k, sign in terms):
                    continue
                common = set.intersection(*(set(groups[a, k].index) for a, k, sign in terms))
                for subset, seeds in (('all', sorted(common)),
                                      ('development', sorted(common & {2022, 2024})),
                                      ('confirmation', sorted(common & {2023, 2025, 2026}))):
                    if len(seeds) < 2:
                        continue
                    for metric in metrics:
                        diffs = []
                        for seed in seeds:
                            delta = sum(sign * float(groups[a, k].loc[seed, metric]) for a, k, sign in terms)
                            diffs.append(delta)
                            extra_paired.append(dict(split=split, dataset=dataset, checkpoint=endpoint,
                                subset=subset, effect=effect, metric=metric, seed=seed, delta=delta))
                        extra_summary.append(dict(split=split, dataset=dataset, checkpoint=endpoint,
                            subset=subset, effect=effect, metric=metric, **summarize(diffs)))
        for (dataset, arm, kind, endpoint), group in frame.groupby(['dataset', 'arm', 'model_kind', 'checkpoint']):
            for subset, seeds in (('all', (2022, 2023, 2024, 2025, 2026)),
                                  ('development', (2022, 2024)),
                                  ('confirmation', (2023, 2025, 2026))):
                part = group[group.seed.isin(seeds)]
                if len(part) < 2:
                    continue
                for metric in metrics:
                    summaries.append(dict(split=split, dataset=dataset, arm=arm, model_kind=kind,
                        checkpoint=endpoint, subset=subset, seeds=','.join(map(str, sorted(part.seed))),
                        metric=metric, **summarize(part[metric])))
            for _, row in group.iterrows():
                values = {cell: float(row[cell + '_auc']) for cell in cells}
                minimum = min(values.values())
                worst.append(dict(split=split, dataset=dataset, arm=arm, seed=int(row.seed),
                    model_kind=kind, checkpoint=endpoint, epoch=int(row.epoch),
                    worst_cells='|'.join(c for c, v in values.items() if v == minimum), worst_auc=minimum))
        base = frame[(frame.arm == 'A0') & (frame.model_kind == 'fixed_ema') &
                     (frame.checkpoint == 'selected')]
        for dataset, group in base.groupby('dataset'):
            for cell in cells:
                counts = group[cell + '_count'].unique()
                positive = group[cell + '_positive'].unique()
                assert len(counts) == len(positive) == 1
                population.append(dict(split=split, dataset=dataset, cell=cell,
                    count=int(counts[0]), positive=int(positive[0]),
                    positive_rate=float(positive[0] / counts[0]),
                    baseline_auc_mean=float(group[cell + '_auc'].mean()),
                    baseline_logloss_mean=float(group[cell + '_logloss'].mean()),
                    baseline_seeds=','.join(map(str, sorted(group.seed)))))
    pd.DataFrame(summaries).to_csv(tables / 'arm_summary_by_subset.csv', index=False)
    pd.DataFrame(population).to_csv(tables / 'population_and_baseline_difficulty.csv', index=False)
    pd.DataFrame(worst).to_csv(tables / 'worst_cell_identity.csv', index=False)
    pd.DataFrame(extra_paired).to_csv(tables / 'supplemental_paired_effects.csv', index=False)
    pd.DataFrame(extra_summary).to_csv(tables / 'supplemental_effect_summary.csv', index=False)

    raw = pd.read_csv(RUN / 'costs.csv')
    present = raw.copy()
    present['raw_mask_fields'] = raw['mask_fields']
    present['raw_perturbation_energy'] = raw['perturbation_energy']
    scope = []
    replay_rows, validation_rows, pass_rows, mask_rows, matching = [], [], [], [], []
    for index, row in present.iterrows():
        root = Path(row.output_dir)
        combined = json.loads((root / 'runtime_stats.json').read_text())
        masked = row.trial.startswith(('B1', 'R1'))
        if masked:
            scope.append('measured_mask_channel_primary_plus_replay')
        elif row.trial == 'A0':
            scope.append('structural_zero_no_augmentation')
            present.at[index, 'mask_fields'] = np.nan
        else:
            scope.append('unmeasured_unmasked_augmentation')
            present.at[index, 'mask_fields'] = np.nan
            present.at[index, 'perturbation_energy'] = np.nan
        replay_rows.append(dict(dataset=row.dataset, trial=row.trial, seed=int(row.seed),
            training_passes=int(combined['training_passes']),
            replay_required=row.trial != 'A0',
            replay_verified=combined.get('replay_verified', False),
            final_optimizer_hash=combined['final_optimizer_hash']))
        for kind, costs in combined['validation_costs'].items():
            validation_rows.append(dict(dataset=row.dataset, trial=row.trial, seed=int(row.seed),
                model_kind=kind, **costs))
        for pass_name in ('primary', 'replay'):
            directory = root if root.name == pass_name else root / pass_name
            if not directory.exists():
                continue
            stats = json.loads((directory / 'runtime_stats.json').read_text())
            pass_rows.append(dict(dataset=row.dataset, trial=row.trial, seed=int(row.seed),
                execution_attempt=RUN.name, scope='successful_current_job',
                pass_name=pass_name, **{k: stats[k] for k in ('training_seconds', 'original_exposures',
                'extra_views', 'training_forwards', 'gradient_calls', 'optimizer_steps',
                'shadow_updates', 'validation_forwards', 'validation_wall_seconds',
                'training_gpu_event_ms', 'peak_memory_bytes', 'peak_reserved_bytes',
                'lambda_steps', 'lambda_sample_weight')}))
        if masked:
            mask = pd.read_csv(root / 'primary' / 'mask_stats.csv')
            counts = np.stack(mask['counts'].map(json.loads))
            energy = np.stack(mask['energies'].map(json.loads))
            n = mask['target_rows'].sum()
            fields = counts.shape[1]
            mask_rows.append(dict(dataset=row.dataset, trial=row.trial, seed=int(row.seed),
                pass_name='primary', steps=len(mask), target_row_exposures=int(n), fields=fields,
                selected_fields=int(counts.sum()), selected_rate=float(counts.sum() / (n * fields)),
                perturbation_energy=float(energy.sum()),
                actual_perturbation_rms=float(np.sqrt(energy.sum() / (n * fields * 16))),
                energy_scope='pre_lambda_embedding_perturbation'))
            if row.trial.startswith('R1'):
                source = Path(str(root).replace('/R1/', '/B1/')) / 'primary' / 'mask_stats.csv'
                ref = pd.read_csv(source)
                assert list(mask.global_step) == list(ref.global_step)
                rc = np.stack(ref['counts'].map(json.loads))
                re = np.stack(ref['energies'].map(json.loads))
                matching.append(dict(dataset=row.dataset, seed=int(row.seed),
                    count_exact=bool(np.array_equal(counts, rc)),
                    maximum_absolute_energy_error=float(np.abs(energy - re).max()),
                    maximum_relative_energy_error=float((np.abs(energy - re) / np.maximum(re, 1e-30)).max()),
                    source_total_energy=float(re.sum()), control_total_energy=float(energy.sum())))
    present['energy_measurement_scope'] = scope
    present.to_csv(tables / 'costs_presentation.csv', index=False, na_rep='N/A')
    # The runner's R3 ledger enumerates raw sources once, including failed retries
    # and partial passes. Do not add inherited cumulative receipts a second time.
    physical = pd.read_csv(RUN / 'physical_cost_ledger.csv')
    shutil.copyfile(RUN / 'physical_cost_ledger.csv', tables / 'physical_cost_ledger.csv')
    for name in ('physical_cost_sources.json', 'execution_cost_continuity.json'):
        shutil.copyfile(RUN / name, decisions / name)
    additive = ['training_passes', 'training_seconds', 'original_exposures', 'extra_views',
                'view_draws', 'training_forwards', 'gradient_calls', 'optimizer_steps',
                'shadow_updates', 'validation_forwards', 'validation_wall_seconds',
                'validation_gpu_event_ms', 'training_gpu_event_ms', 'selection_checks']
    totals = {k: float(physical[k].sum()) for k in additive}
    totals.update(scientific_unique_runs=len(raw),
        physical_ledger_records=len(physical),
        source_attempts=sorted(physical.source_attempt.unique()),
        complete_passes=int(physical.complete_passes.sum()),
        partial_passes=int(physical.partial_passes.sum()),
        physical_totals_scope='known totals or lower bounds; trace counters count committed records',
        unavailable_partial_components=sorted(set().union(*physical.unavailable_fields.map(json.loads))),
        lower_bound_fields=sorted(set().union(*physical.lower_bound_fields.map(json.loads))),
        current_successful_worker_peak_allocated_bytes=int(raw.peak_memory_bytes.max()),
        current_successful_worker_peak_reserved_bytes=int(raw.peak_reserved_bytes.max()),
        wall_seconds_from_original_task_start=completion['wall_seconds'],
        test_costs=pd.read_csv(RUN / 'test_costs.csv').drop(columns=['dataset', 'trial', 'seed']).sum().to_dict(),
        scope='all failed and successful execution sources; scientific sample size unchanged',
        stage0_excluded=True)
    (tables / 'all_attempt_cost_totals.json').write_text(json.dumps(totals, indent=2) + '\n')
    pd.DataFrame(pass_rows).to_csv(tables / 'physical_pass_costs.csv', index=False)
    pd.DataFrame(validation_rows).to_csv(tables / 'validation_costs_by_kind.csv', index=False)
    pd.DataFrame(replay_rows).to_csv(tables / 'replay_identity.csv', index=False)
    pd.DataFrame(mask_rows).to_csv(tables / 'mask_dose_primary.csv', index=False)
    pd.DataFrame(matching).to_csv(tables / 'r1_count_energy_matching.csv', index=False)
    (REPORT / 'FINAL_ANALYSIS_RECEIPT.json').write_text(json.dumps(dict(
        completion_status=completion['status'], state_released=release['released'],
        unique_runs=len(raw), test_rows=len(pd.read_csv(RUN / 'test_metrics.csv')),
        scientific_unit='seed', energy_scope_corrected_in_presentation_only=True,
        source_unchanged=True), indent=2) + '\n')


if __name__ == '__main__':
    main()
