import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARTIFACT = ROOT / 'artifacts/26-11-16_augmentation_gradient_momentum'
TABLES = ARTIFACT / 'attempt1/tables'
SEEDS = [2022, 2023, 2024, 2025, 2026]
ARMS = ['CLEAN', 'NDP-A', 'TV-GRM-005', 'TV-GRM-010', 'TV-GRM-020',
        'TV-GRM-SHUFFLE-010', 'TV-GRM-INSTANT-010']
DATASETS = ['movielens', 'kuairand']
CELLS = [f'd{domain}t{task}' for domain in range(3) for task in range(2)]


def markdown(frame, digits=9):
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    for values in frame.itertuples(index=False, name=None):
        cells = [f'{value:.{digits}f}' if isinstance(value, (float, np.floating)) else str(value)
                 for value in values]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    endpoints = pd.read_csv(TABLES / 'ENDPOINTS.csv')
    summaries = pd.read_csv(TABLES / 'SUMMARY.csv')
    paired = pd.read_csv(TABLES / 'PAIRED_SUMMARY.csv')
    differences = pd.read_csv(TABLES / 'PAIRED_SEEDS.csv')
    diagnostics = pd.read_csv(TABLES / 'GRADIENT_DIAGNOSTICS.csv')
    costs = pd.read_csv(TABLES / 'COSTS.csv')
    ranking = pd.read_csv(TABLES / 'CANDIDATE_RANKING.csv')
    selected = endpoints[endpoints.endpoint.eq('selected')].copy()
    assert len(selected) == 140 and set(selected.seed) == set(SEEDS)
    assert selected.groupby(['dataset', 'arm', 'kind']).size().eq(5).all()
    assert selected.split.eq('validation').all()
    for metric in ['auc', 'logloss']:
        assert np.allclose(endpoints[[f'{cell}_{metric}' for cell in CELLS]].mean(axis=1),
                           endpoints[f'macro_{metric}'], rtol=0, atol=1e-12)
    keys = ['dataset', 'arm', 'control', 'endpoint', 'kind', 'reference_kind', 'metric']
    assert not differences.duplicated(keys + ['seed']).any()
    assert set(differences.seed) == set(SEEDS)
    grouped = differences.groupby(keys).delta.agg(['mean', 'size'])
    reference = paired.set_index(keys)
    assert len(grouped) == len(reference) and grouped['size'].eq(5).all()
    assert np.allclose(grouped.reindex(reference.index)['mean'], reference['mean'], rtol=0, atol=1e-12)
    primary = paired[paired.endpoint.eq('selected') & paired.kind.eq('fixed_ema')
                     & paired.reference_kind.eq('fixed_ema')
                     & paired.metric.isin(['macro_auc', 'macro_logloss'])].copy()
    ledger = []
    for (dataset, kind, arm), rows in selected.groupby(['dataset', 'kind', 'arm'], sort=False):
        entry = dict(dataset=dataset, kind=kind, arm=arm, n=5)
        metrics = [name for name in rows.columns if name.endswith(('_auc', '_logloss'))]
        for metric in metrics:
            entry[metric] = float(rows[metric].mean())
        entry['selected_epoch_mean'] = float(rows.epoch.mean())
        ledger.append(entry)
    ledger = pd.DataFrame(ledger)
    ledger.to_csv(HERE / 'SELECTED_CELL_MEANS.csv', index=False)
    primary.to_csv(HERE / 'PRIMARY_PAIRED.csv', index=False)
    ledger_lines = ['## 2026-11-16 TV-GRM：完整五seed验证记录',
                    'M3oE fixed1，自然人口，50epochs/batch40960/FP32，seeds2022–2026。'
                    '50条新增+20条历史CLEAN/NDP-A复用；所有行是validation、独立online/EMA选点。'
                    '本组为内部方法/匹配控制，不作跨协议论文baseline排名；Test未开启。'
                    '[完整分析](research_reports/task_view_gradient_residual_momentum_20261116/REPORT.md)，'
                    '[逐seed/六格/总体原值](artifacts/26-11-16_augmentation_gradient_momentum/attempt1/tables/ENDPOINTS.csv)，'
                    '[五seed均值](research_reports/task_view_gradient_residual_momentum_20261116/SELECTED_CELL_MEANS.csv)。']
    for dataset in DATASETS:
        for kind in ['online', 'fixed_ema']:
            numeric_rows = []
            for row in ledger[ledger.dataset.eq(dataset) & ledger.kind.eq(kind)].to_dict(orient='records'):
                numeric = {'arm': row['arm'], 'n': 5}
                numeric.update({cell: f"{row[f'{cell}_auc']:.9f} / {row[f'{cell}_logloss']:.9f}" for cell in CELLS})
                numeric['Macro AUC / LL'] = f"{row['macro_auc']:.9f} / {row['macro_logloss']:.9f}"
                numeric['Traffic AUC'] = row['sample_weighted_cell_auc']
                numeric['Pooled AUC'] = row['pooled_task_mean_auc']
                numeric['Worst-cell AUC'] = row['worst_cell_auc']
                numeric_rows.append(numeric)
            ledger_lines += [f'### {dataset} / {kind}', markdown(pd.DataFrame(numeric_rows))]
    (HERE / 'LEDGER_ENTRY.md').write_text('\n\n'.join(ledger_lines) + '\n')
    sections = ['# TV-GRM numerical appendix',
                'All scores are validation scores. Seeds are the statistical units; n=5. '
                'Online and fixed EMA select their own earliest strict Macro AUC maximum. '
                'All per-seed cells, common epochs and state hashes are in ENDPOINTS.csv.']
    for dataset in DATASETS:
        for kind in ['online', 'fixed_ema']:
            rows = selected[selected.dataset.eq(dataset) & selected.kind.eq(kind)].copy()
            clean = rows[rows.arm.eq('CLEAN')].set_index('seed')
            rows['delta_macro_auc'] = rows.apply(lambda row: row.macro_auc - clean.loc[row.seed, 'macro_auc'], axis=1)
            rows['delta_macro_logloss'] = rows.apply(lambda row: row.macro_logloss - clean.loc[row.seed, 'macro_logloss'], axis=1)
            sections += [f'## {dataset} / {kind} / selected per seed',
                         markdown(rows[['arm', 'seed', 'epoch', 'macro_auc', 'macro_logloss',
                                        'delta_macro_auc', 'delta_macro_logloss', 'model_hash']]),
                         f'## {dataset} / {kind} / six-cell means',
                         markdown(ledger[ledger.dataset.eq(dataset) & ledger.kind.eq(kind)].drop(columns=['dataset', 'kind']))]
    sections += ['## Selected five-seed paired estimates', markdown(primary),
                 '## Common-epoch Macro AUC and LogLoss', markdown(summaries[
                     summaries.metric.isin(['macro_auc', 'macro_logloss']) & ~summaries.endpoint.eq('selected')]),
                 '## Retention ranking', markdown(ranking)]
    (HERE / 'NUMERICAL_APPENDIX.md').write_text('\n\n'.join(sections) + '\n')
    chosen = ranking.iloc[0]
    all_nonpositive = bool((ranking[DATASETS] <= 0).all().all())
    eligible = bool(any(float(chosen[dataset]) > 0 for dataset in DATASETS))
    new_costs = costs[~costs.arm.isin(['CLEAN', 'NDP-A'])]
    assert len(new_costs) == 50 and int(new_costs.steps.sum()) == 86250
    diagnostic_summary = []
    for (dataset, arm), rows in diagnostics.groupby(['dataset', 'arm']):
        diagnostic_summary.append(dict(dataset=dataset, arm=arm, diagnostic_points=len(rows),
            cap_fraction=float(rows.cap_active.mean()), mean_cap_scale=float(rows.cap_scale.mean()),
            mean_clean_residual_cosine=float(rows.clean_residual_cosine.mean()),
            mean_q_residual_cosine=(float(rows.q_residual_cosine.mean())
                                    if rows.q_residual_cosine.notna().any() else None),
            max_shuffle_relative_error=(float(rows.shuffle_relative_error.max())
                                        if rows.shuffle_relative_error.notna().any() else None)))
    manifest_path = ARTIFACT / 'attempt1/run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    decision = dict(candidate='TV-GRM', candidate_decision='NO_CONTINUATION_SIGNAL',
        scientific_summary='No cross-dataset increment over CLEAN fixed EMA; weak positive MovieLens gamma=.20 signal.',
        all_gamma_all_dataset_nonpositive_gate_triggered=all_nonpositive,
        retention_rank_winner=str(chosen.arm), retention_score=float(chosen.retention_score),
        retention_winner_has_positive_dataset_mean=eligible, champion=(str(chosen.arm) if eligible else None),
        retain_all_states_until_review=True, primary_paired=primary.to_dict(orient='records'),
        ranking=ranking.to_dict(orient='records'), diagnostics=diagnostic_summary,
        cost=dict(new_trajectories=50, reused_trajectories=20, new_epochs=2500,
                  parameter_updates=86250, ema_updates=86250,
                  worker_seconds=float(new_costs.worker_seconds.sum()),
                  peak_allocated_bytes=int(new_costs.peak_allocated_bytes.max()),
                  peak_reserved_bytes=int(new_costs.peak_reserved_bytes.max())),
        training_source_hash=manifest['source_hash'], design_sha256=manifest['design_sha256'],
        source_tables={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in TABLES.glob('*.csv')},
        test_accessed=False, test_status='NOT_TRIGGERED', hpo_status='NOT_TRIGGERED',
        execution_mode='CURRENT_CONVERSATION_ONLY', normal_termination='USER_STOPPED',
        next_action='Independent evidence review and main-context publication, then fresh native Astra in this conversation.',
        evidence_strength=dict(cross_dataset_increment='MEDIUM', movielens_gamma020='WEAK',
                               residual_direction='WEAK', residual_history='WEAK'))
    assert not all_nonpositive and not eligible
    (HERE / 'DECISION.json').write_text(json.dumps(decision, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: decision[key] for key in ['candidate_decision',
        'all_gamma_all_dataset_nonpositive_gate_triggered', 'retention_rank_winner',
        'retention_score', 'champion', 'cost', 'diagnostics']}, indent=2))


if __name__ == '__main__':
    main()
