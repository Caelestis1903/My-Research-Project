"""Primary descriptive summaries of the closed 160-anchor core diagnostics."""
from pathlib import Path
import gzip
import hashlib
import json

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / 'core_mechanism'
GROUP = ['population', 'budget', 'arm', 'checkpoint']
SEEDS = {2022, 2023, 2024, 2025}


def summarize(frame, groups, metrics):
    rows = []
    for key, g in frame.groupby(groups, dropna=False):
        assert set(g.seed) == SEEDS and len(g) == 4
        for metric in metrics:
            values = g[metric].to_numpy(float)
            rows.append(dict(zip(groups, key), metric=metric, n_seeds=4,
                defined=int(np.isfinite(values).sum()), mean=float(np.mean(values)),
                min=float(np.min(values)), max=float(np.max(values)),
                positive=int((values > 0).sum()), negative=int((values < 0).sum()),
                above_one=int((values > 1).sum()), below_one=int((values < 1).sum())))
    return pd.DataFrame(rows)


def analyze():
    status = json.loads((OUT / 'analysis_status.json').read_text())
    assert status['complete'] and status['completed'] == 160 and not status['missing']
    sources = pd.read_csv(OUT / 'source_index.csv')
    covariance = pd.read_csv(OUT / 'covariance.csv')
    blocks = pd.read_csv(OUT / 'block_covariance.csv')
    finite = pd.read_csv(OUT / 'finite_pool_identity.csv')
    grid = pd.read_csv(OUT / 'same_baseline_adam_grid.csv')
    risk = pd.read_csv(OUT / 'functional_vs_training_derivative.csv')
    summarize(covariance, GROUP, ['SB_over_SN_trace', 'mean_difference_over_RMS_sum']).to_csv(
        OUT / 'four_seed_covariance_summary.csv', index=False)
    totals = covariance[['run_id', 'checkpoint', 'SN_trace', 'SB_trace']].rename(
        columns={'SN_trace': 'SN_all_trace', 'SB_trace': 'SB_all_trace'})
    blocks = blocks.merge(totals, on=['run_id', 'checkpoint'], validate='many_to_one')
    blocks['SN_trace_share'] = blocks.SN_trace / blocks.SN_all_trace
    blocks['SB_trace_share'] = blocks.SB_trace / blocks.SB_all_trace
    blocks.to_csv(OUT / 'partition_trace_shares.csv', index=False)
    summarize(blocks, GROUP + ['partition'], ['SB_over_SN_trace', 'SN_trace_share',
        'SB_trace_share']).to_csv(OUT / 'four_seed_partition_summary.csv', index=False)

    raw_norms = []
    snapshots = []
    for row in sources.itertuples():
        path = Path(row.path)
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row.sha256
        x = json.loads(raw)
        raw_norms.append(dict(run_id=row.run_id, checkpoint=row.checkpoint,
            finite_oracle_gradient_norm=x['finite_stratified_pool_oracle']['objective_mean_gradient_norm']))
        snapshots.append(dict(path=str(path), sha256=row.sha256, bytes=len(raw), diagnostic=x))
    finite = finite.merge(pd.DataFrame(raw_norms), on=['run_id', 'checkpoint'], validate='many_to_one')
    finite['identity_error_over_oracle_norm'] = finite.expected_gradient_error_norm / finite.finite_oracle_gradient_norm
    finite['MC_mean_error_over_RMS_SE'] = finite.MC_error_vs_exact_mean_norm / finite.MC_mean_RMS_SE
    finite.to_csv(OUT / 'finite_pool_relative_errors.csv', index=False)
    summarize(finite, GROUP + ['estimator'], ['identity_error_over_oracle_norm', 'MC_over_exact',
        'MC_mean_error_over_RMS_SE']).to_csv(OUT / 'four_seed_finite_pool_summary.csv', index=False)

    # A switched synthetic moment is compared at the same theta/native baseline.
    # This reports formula-space norm changes, not a trained model or risk change.
    edges = []
    for (run_id, checkpoint), g in grid.groupby(['run_id', 'checkpoint']):
        indexed = g.set_index(['m_source', 'v_source', 'current_source'])
        base = g.iloc[0][['population', 'budget', 'arm', 'seed']].to_dict()
        for factor, level in [('m_source', 0), ('v_source', 1), ('current_source', 2)]:
            for key in indexed.index:
                if key[level] != 'full':
                    continue
                target_key = list(key)
                target_key[level] = 'target'
                full = indexed.loc[key, 'applied_shared_delta_norm']
                target = indexed.loc[tuple(target_key), 'applied_shared_delta_norm']
                other = [f'{name}={key[i]}' for i, name in enumerate(
                    ['m_source', 'v_source', 'current_source']) if i != level]
                edges.append(dict(base, run_id=run_id, checkpoint=checkpoint,
                    switched_factor=factor, held_sources=','.join(other),
                    full_norm=full, target_norm=target, target_minus_full_norm=target-full,
                    target_over_full_norm=target/full if full else np.nan))
    edges = pd.DataFrame(edges)
    edges.to_csv(OUT / 'same_baseline_grid_switches.csv', index=False)
    summarize(edges, GROUP + ['switched_factor', 'held_sources'], ['target_minus_full_norm',
        'target_over_full_norm']).to_csv(OUT / 'four_seed_grid_switch_summary.csv', index=False)
    risk['difference_over_training_norm'] = risk.gradient_difference_norm / risk.training_derivative_norm
    risk['partition_difference_share'] = risk.block_gradient_difference_squared_norm / risk.gradient_difference_norm.pow(2)
    risk.to_csv(OUT / 'functional_derivative_relative_differences.csv', index=False)
    summarize(risk, GROUP + ['partition'], ['difference_over_training_norm',
        'partition_difference_share']).to_csv(OUT / 'four_seed_functional_derivative_summary.csv', index=False)

    archive = OUT / 'raw_diagnostic_snapshots.jsonl.gz'
    payload = ''.join(json.dumps(x, sort_keys=True, ensure_ascii=False) + '\n' for x in snapshots).encode()
    archive.write_bytes(gzip.compress(payload, mtime=0))
    (OUT / 'summary_receipt.json').write_text(json.dumps(dict(complete=True, anchors=160,
        optimizer_seeds=4, raw_diagnostic_archive=str(archive),
        raw_diagnostic_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        raw_diagnostic_count=len(snapshots),
        uncertainty='Four-seed descriptive ranges; no trace MCSE, population CI, or multiplicity correction inferred.',
        grid_scope='Synthetic two-event C-only formula norm, without theta installation or risk evaluation.'), indent=2) + '\n')
    print('Core summaries complete: 160 anchors; four optimizer seeds per group.')


if __name__ == '__main__':
    analyze()
