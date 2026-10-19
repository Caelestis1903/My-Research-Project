"""Descriptive evidence tables from complete collected scopes, without new inference."""
import argparse
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import t
from cm19_collect import FIELDS
from cm19_train import OUT, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['development', 'test'], required=True)
    args = parser.parse_args()
    root = OUT / (args.split + '_readout')
    receipt = json.loads((root / 'collection_receipt.json').read_text())
    assert receipt['independent_runs'] == 856 and receipt['stage'] == 'all'
    cells = pd.read_csv(root / 'seed_cell_metrics.csv')
    macro = pd.read_csv(root / 'seed_macro_metrics.csv')
    paths = pd.read_csv(root / 'path_diagnostics.csv.gz')
    target = root / 'evidence'
    target.mkdir(exist_ok=True)
    keys = FIELDS + ['label', 'probe', 'intervention']
    paths.groupby(keys, dropna=False).agg(
        mean_delta_p=('delta_p', 'mean'), max_delta_p=('max_delta', 'max'),
        cells_exact_zero=('delta_p', lambda x: int((x == 0).sum())),
        cell_rows=('delta_p', 'size'), n_seeds=('seed', 'nunique'),
        mean_delta_auc=('delta_auc', 'mean'),
        mean_delta_logloss=('delta_logloss', 'mean'),
    ).reset_index().to_csv(target / 'probe_condition_summary.csv', index=False)
    paths.groupby(keys + ['seed'], dropna=False).agg(
        mean_delta_p=('delta_p', 'mean'), max_delta_p=('max_delta', 'max'),
        cells_exact_zero=('delta_p', lambda x: int((x == 0).sum())),
        cell_rows=('delta_p', 'size'),
    ).reset_index().to_csv(target / 'probe_seed_summary.csv', index=False)
    # Keep single-task model comparisons on the same task and domain as fixed.
    natural = cells[cells.intervention == 'none']
    native = natural[natural.stage == 'G0']
    fixed = native[native.model == 'fixed']
    matched = native[native.model != 'fixed'].merge(
        fixed, on=['dataset', 'optimizer', 'seed', 'checkpoint', 'domain', 'task'],
        suffixes=('_model', '_fixed'), validate='many_to_one')
    matched['delta_auc'] = matched.auc_model - matched.auc_fixed
    matched['delta_logloss'] = matched.logloss_model - matched.logloss_fixed
    kk = ['dataset', 'model_model', 'optimizer', 'checkpoint', 'domain', 'task']
    matched[kk + ['seed', 'delta_auc', 'delta_logloss']].rename(
        columns={'model_model': 'model'}).to_csv(target / 'task_matched_fixed_seed_cells.csv', index=False)
    summarize(matched, kk, ['delta_auc', 'delta_logloss']).rename(
        columns={'model_model': 'model'}).to_csv(target / 'task_matched_fixed_cell_summary.csv', index=False)
    # Selection and final are separate estimands, not independent replications.
    stage = macro[macro.checkpoint == 'final'].merge(
        macro[macro.checkpoint == 'selected'], on=FIELDS + ['seed', 'run_id'],
        suffixes=('_final', '_selected'), validate='one_to_one')
    stage['delta_auc'] = stage.macro_auc_final - stage.macro_auc_selected
    stage['delta_logloss'] = stage.macro_logloss_final - stage.macro_logloss_selected
    stage[FIELDS + ['seed', 'delta_auc', 'delta_logloss']].to_csv(target / 'stage_seed_deltas.csv', index=False)
    summarize(stage, FIELDS, ['delta_auc', 'delta_logloss']).to_csv(target / 'stage_summary.csv', index=False)
    profiles = pd.read_csv(OUT / 'offline_readout/model_cost_profiles.csv')
    costs = pd.read_csv(root / 'run_costs.csv').merge(
        profiles[['dataset', 'model', 'variant', 'rows',
                  'forward_dense_flops_per_example', 'forward_backward_dense_flops_per_example',
                  'inference_seconds_median', 'inference_seconds_p95']],
        on=['dataset', 'model', 'variant'], validate='many_to_one')
    assert len(costs) == 856 and not costs.forward_backward_dense_flops_per_example.isna().any()
    auxiliary = []
    for row in costs.itertuples():
        extra = 0
        if row.model == 'original':
            rr = OUT / 'runs' / row.run_id
            config = json.loads((rr / 'config.json').read_text())
            result = json.loads((rr / 'receipt.json').read_text())
            n, batch = config['train_rows'], config['batch_size']
            cycles, remainder = divmod(result['arch_steps'], math.ceil(n / batch))
            extra = cycles * n + remainder * batch
        auxiliary.append(extra)
    costs['auxiliary_training_rows'] = auxiliary
    costs['estimated_dense_primary_backward_flops'] = costs.examples * costs.forward_backward_dense_flops_per_example
    costs['estimated_dense_auxiliary_backward_flops'] = costs.auxiliary_training_rows * costs.forward_backward_dense_flops_per_example
    costs['estimate_scope'] = 'Small-batch supported dense forward/backward FLOPs scaled by exact row exposures; excludes optimizer, embedding, normalization, evaluation and diagnostics. CPU latency shared-host, not isolated/full-batch.'
    costs.to_csv(target / 'costs_with_profile_estimates.csv', index=False)
    # These are full-test fixed-state interventions, not retraining contrasts.
    if args.split == 'test':
        neutral = cells[cells.intervention != 'none'].merge(
            natural[['run_id', 'checkpoint', 'domain', 'task', 'auc', 'logloss']],
            on=['run_id', 'checkpoint', 'domain', 'task'], suffixes=('', '_natural'),
            validate='many_to_one')
        neutral['delta_auc'] = neutral.auc - neutral.auc_natural
        neutral['delta_logloss'] = neutral.logloss - neutral.logloss_natural
        kk = FIELDS + ['checkpoint', 'intervention']
        neutral[kk + ['seed', 'domain', 'task', 'n', 'delta_auc', 'delta_logloss']].to_csv(
            target / 'neutral_seed_cell_deltas.csv', index=False)
        summarize(neutral, kk + ['domain', 'task'], ['delta_auc', 'delta_logloss']).to_csv(
            target / 'neutral_cell_summary.csv', index=False)
        seed = neutral.groupby(kk + ['seed'], dropna=False)[['delta_auc', 'delta_logloss']].mean().reset_index()
        seed.to_csv(target / 'neutral_macro_seed_deltas.csv', index=False)
        summarize(seed, kk, ['delta_auc', 'delta_logloss']).to_csv(target / 'neutral_macro_summary.csv', index=False)
    dump(target / 'receipt.json', dict(
        source_split=args.split, independent_runs=856, complete=True,
        unit='Four optimization seeds on one fixed population; no multiplicity correction.',
        probe_scope='Within-seed domain/task cells are descriptive, not independent seed replications.',
        signs='All deltas candidate-minus-reference; neutral intervention deltas neutral-minus-natural.',
    ))
    print(f'Wrote complete {args.split} descriptive evidence tables.')


def summarize(frame, keys, metrics):
    rows = []
    for values, block in frame.groupby(keys, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        assert len(block) == 4 and block.seed.nunique() == 4, (keys, values, len(block))
        row = dict(zip(keys, values))
        for metric in metrics:
            delta = block[metric]
            mean = delta.mean()
            sd = delta.std(ddof=1)
            half = t.ppf(.975, 3) * sd / np.sqrt(4)
            row.update({metric + '_mean': mean, metric + '_sd': sd,
                        metric + '_ci95_low': mean - half, metric + '_ci95_high': mean + half,
                        metric + '_positive_seeds': int((delta > 0).sum()),
                        metric + '_negative_seeds': int((delta < 0).sum()),
                        metric + '_zero_seeds': int((delta == 0).sum())})
        row['n_seeds'] = 4
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    main()
