"""Primary four-seed summaries after all frozen tests for one transfer study close."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from analyze_core import interval

HERE = Path(__file__).resolve().parent
RECIPE = ['study','population','scope','dataset','optimizer','method','suite','variant',
    'decay_mode','schedule','fraction','backbone','noise_interaction','stage_group','budget','lr']
SEEDS = {2022, 2023, 2024, 2025}


def analyze(study):
    index = pd.read_csv(HERE/'transfer/run_config_index.csv').fillna('UNSPECIFIED')
    index = index[index.study == study].set_index('run_id')
    new = index[index.scope != 'historical']
    expected = {'N19': 316, 'CM19': 508}[study]
    assert len(new) == expected, 'all frozen study scopes must finish test collection first'
    macro = pd.read_csv(HERE/'transfer/run_task_macro_metrics.csv')
    macro = macro[(macro.study == study) & (macro.intervention == 'none')].fillna('UNSPECIFIED')
    assert set(new.index) <= set(macro.run_id)
    pairs = pd.read_csv(HERE/'transfer/paired_seed_deltas.csv')
    pairs = pairs[(pairs.task == -1) & pairs.candidate_run_id.isin(index.index)]
    definitions = json.loads((HERE/'transfer/comparison_definitions.json').read_text())
    metadata = []
    for record in definitions:
        ca = record['candidate_run_ids']
        if not set(ca) <= set(index.index):
            continue
        ref = record['reference_run_ids']
        a = index.loc[ca]
        b = index.loc[ref]
        assert set(a.seed) == set(b.seed) == SEEDS and len(a) == len(b) == 4
        row = dict(comparison=record['comparison'], meaning=record['meaning'])
        for side, frame in [('candidate', a), ('reference', b)]:
            for column in RECIPE:
                assert frame[column].nunique() == 1, (record['comparison'], column)
                row[side+'_'+column] = frame[column].iloc[0]
        metadata.append(row)
    metadata = pd.DataFrame(metadata)
    out = HERE/(study.lower()+'_phase')
    out.mkdir(exist_ok=True)
    metadata.to_csv(out/'comparison_metadata.csv', index=False)
    summaries = pd.read_csv(HERE/'transfer/paired_summary.csv')
    summaries = summaries[summaries.comparison.isin(metadata.comparison)].merge(
        metadata, on=['comparison','meaning'], validate='many_to_one')
    summaries.to_csv(out/'all_four_seed_comparison_summaries.csv', index=False)
    absolute = []
    for key, g in macro.groupby(RECIPE+['checkpoint','task'], dropna=False):
        assert set(g.seed) == SEEDS and len(g) == 4
        for metric in ['macro_auc','macro_logloss','support_p_cell_auc','support_p_cell_logloss',
                'exposure_p_cell_auc','exposure_p_cell_logloss','eval_p_cell_auc','eval_p_cell_logloss']:
            absolute.append(dict(zip(RECIPE+['checkpoint','task'], key), metric=metric,
                **interval(g[metric])))
    pd.DataFrame(absolute).to_csv(out/'four_seed_absolute_readouts.csv', index=False)
    domain_task = []
    for axis in ['domain','task']:
        raw = pd.read_csv(HERE/f'transfer_cells/paired_{axis}_deltas.csv')
        raw = raw[raw.comparison.isin(metadata.comparison)]
        for key, g in raw.groupby(['comparison','checkpoint',axis]):
            assert set(g.seed) == SEEDS and len(g) == 4
            for metric in ['auc','logloss']:
                domain_task.append(dict(zip(['comparison','checkpoint',axis],key), axis=axis,
                    metric=metric, reference_auc=g.baseline_auc.mean(),
                    reference_logloss=g.baseline_logloss.mean(), **interval(g[metric])))
    pd.DataFrame(domain_task).to_csv(out/'four_seed_domain_task_summary.csv', index=False)
    roles = pd.read_csv(HERE/'transfer_cells/actual_size_roles_and_tradeoffs.csv')
    roles = roles[roles.comparison.isin(metadata.comparison)]
    role_columns = ['reference_small_gain_large_decline','reference_both_smaller_gain_largest_decline',
        'candidate_small_gain_large_decline','candidate_both_smaller_gain_largest_decline']
    trades = []
    for key, g in roles.groupby(['comparison','checkpoint']):
        assert set(g.seed) == SEEDS and len(g) == 4
        trades.append(dict(comparison=key[0], checkpoint=key[1],
            **{column:int(g[column].sum()) for column in role_columns},
            changed_size_order_seeds=int((~g.same_size_order).sum())))
    pd.DataFrame(trades).to_csv(out/'four_seed_actual_size_tradeoffs.csv', index=False)

    # Retain the original support/exposure estimands for both sides; they can
    # change across populations. eval-p shares the actual common test weights.
    keyed = macro.set_index(['run_id','checkpoint','task'])
    weighted = []
    for record in pairs.itertuples():
        ca = record.candidate_run_id
        ref = record.reference_run_id
        tasks = json.loads(index.loc[ca, 'native_tasks'])
        ref_tasks = json.loads(index.loc[ref, 'native_tasks'])
        reference_task = tasks[0] if len(tasks) == 1 and len(ref_tasks) == 2 else -1
        a = keyed.loc[(ca, record.checkpoint, -1)]
        b = keyed.loc[(ref, record.checkpoint, reference_task)]
        for weighting in ['support_p_cell','exposure_p_cell','eval_p_cell']:
            metric = weighting+'_logloss'
            weighted.append(dict(comparison=record.comparison,checkpoint=record.checkpoint,seed=record.seed,
                weighting=weighting,delta=float(a[metric]-b[metric]),
                interpretation='each-condition support/exposure weights; eval-p uses common matched-task test weights'))
    weighted = pd.DataFrame(weighted)
    weighted.to_csv(out/'paired_seed_weighted_logloss.csv', index=False)
    rows = []
    for key, g in weighted.groupby(['comparison','checkpoint','weighting']):
        assert set(g.seed) == SEEDS and len(g) == 4
        rows.append(dict(zip(['comparison','checkpoint','weighting'],key), **interval(g.delta)))
    pd.DataFrame(rows).to_csv(out/'four_seed_weighted_logloss_summary.csv', index=False)
    collection = json.loads((HERE/'transfer/collection_status.json').read_text())
    (out/'analysis_status.json').write_text(json.dumps(dict(study=study, complete_study=True,
        new_tested_configurations=len(new), historical_configurations=len(index)-len(new),
        complete_comparisons=len(metadata),
        all_studies_complete=collection['new_run_count'] == 824 and not collection['missing'],
        scope='descriptive optimizer-seed t95; repeated test population, no familywise or population confirmation',
        weight_scope='support/exposure can differ between conditions; prescribed exposure weights are not exact realized truncated domain counts'),indent=2)+'\n')
    print(study, 'complete phase tables:', len(new), 'new configurations;', len(metadata), 'comparisons')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', choices=['N19','CM19'], required=True)
    analyze(parser.parse_args().study)
