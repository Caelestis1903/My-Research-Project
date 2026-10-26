"""Compact view of completed four-seed tables for scientific interpretation."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import numpy as np
from analyze_core_four_seed import REPORT, stats, read


def table(name):
    with (REPORT/'tables'/name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    if read(REPORT/'CORE_FOUR_SEED_ANALYSIS_RECEIPT.json')['status']!='PASS':
        raise PermissionError('Complete frozen four-seed analysis required')
    roles={'Q_explore','validation','test_exposed','Q_legacy_confirm_exposed'}
    methods={'NATIVE','LOCAL','H_DIRECT','H_DIRECT_EFFORT','PILOT_PROB','FULL_SOURCE_ALL',
             'RCF','POOLED_TARGET','LOCAL_ACTIVE','K1_WIDE','SOURCE_INIT_RANDOM',
             'PILOT_LOGIT','GROUP_CAL','FULL_SOURCE','FULL_TARGET','RANDOM_FEATURE','ETA_DIRECT_MATCHED'}
    summary=table('CORE_four_seed_metric_summary.csv')
    macro=[r for r in summary if r['role'] in roles and r['training_stage']=='selected'
           and r['level']=='macro' and r['method'] in methods and r['metric'] in ('auc','logloss','natural_logloss')]
    contrasts=table('CORE_four_seed_paired_summary.csv')
    wanted={('RCF',ref) for ref in ('H_DIRECT','H_DIRECT_EFFORT','FULL_SOURCE_ALL','POOLED_TARGET',
                                   'LOCAL_ACTIVE','NATIVE','PILOT_PROB','K1_WIDE','SOURCE_INIT_RANDOM')}
    wanted|={('PILOT_PROB','NATIVE'),('PILOT_PROB','H_DIRECT'),('LOCAL','NATIVE')}
    wanted|={('PILOT_LOGIT','PILOT_PROB'),('FULL_SOURCE','GROUP_CAL'),('FULL_SOURCE','FULL_TARGET'),
              ('FULL_SOURCE','RANDOM_FEATURE'),('FULL_SOURCE_ALL','H_DIRECT'),
              ('FULL_SOURCE_ALL','H_DIRECT_EFFORT'),('ETA_DIRECT_MATCHED','H_DIRECT')}
    pairs=[r for r in contrasts if r['role'] in roles and r['training_stage']=='selected'
           and (r['candidate'],r['reference']) in wanted and r['metric'] in ('auc','logloss','natural_logloss')]
    worst=defaultdict(list)
    for row in table('CORE_four_seed_worst_cells.csv'):
        if row['role'] not in roles or row['training_stage']!='selected' or row['method'] not in methods:continue
        worst[row['dataset'],row['method'],row['role']].append(row)
    worst_summary=[]
    for key,entries in worst.items():
        if {int(r['seed']) for r in entries}!={5201,5202,5203,5204}:raise RuntimeError('Four worst-cell seeds required')
        for metric in ('worst_cell_auc','worst_cell_logloss'):
            worst_summary.append(dict(zip(('dataset','method','role'),key),metric=metric,**stats([r[metric] for r in entries])))
    retention=defaultdict(list)
    for row in table('CORE_four_seed_source_retention.csv'):
        retention[row['dataset'],row['method'],row['role']].append(row)
    retention_summary=[]
    for key,entries in retention.items():
        if len({r['seed'] for r in entries})!=4:raise RuntimeError('Four retention seeds required')
        retention_summary.append(dict(zip(('dataset','method','role'),key),n_training_seeds=4,
            function_instances=len(entries),source_budget_exceeded=sum(r['source_budget_exceeded']=='True' for r in entries),
            foreign_R_exceeded=sum(r['rms_budget_exceeded']=='True' for r in entries),
            max_S=max(float(r['S']) for r in entries),max_R=max(float(r['R']) for r in entries),
            mean_K=float(np.mean([float(r['K']) for r in entries])),max_K=max(float(r['K']) for r in entries)))
    result=dict(status='PASS',minimum_training_seeds=4,macro=macro,contrasts=pairs,
        worst_cells=worst_summary,source_retention=retention_summary,
        native_training_stage=table('CORE_four_seed_native_epoch24_minus_selected.csv'),
        bootstrap_invalid_total=sum(int(r['invalid_replicates']) for r in table('CORE_four_seed_within_seed_bootstrap.csv')),
        note='Descriptive view only; test/legacy cannot change frozen selection or resource gates. Function counts are not training replicates.')
    (REPORT/'FINAL_FOUR_SEED_FINDINGS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',macro_rows=len(macro),paired_rows=len(pairs),worst_cells=len(worst_summary))))


if __name__=='__main__':main()
