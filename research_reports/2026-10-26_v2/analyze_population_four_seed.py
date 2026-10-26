"""Keep natural-population, weighted-cell and pooled estimands separate."""
from collections import defaultdict
import csv
import json
import numpy as np
from analyze_core_four_seed import REPORT, read, write, stats


def main():
    if read(REPORT/'CORE_FOUR_SEED_ANALYSIS_RECEIPT.json')['status']!='PASS':raise PermissionError('Complete four-seed core required')
    with (REPORT/'tables/CORE_four_seed_per_seed_metrics.csv').open(newline='') as stream:raw=list(csv.DictReader(stream))
    grouped=defaultdict(list)
    for row in raw:
        if row['training_stage']=='selected' and row['role'] in ('Q_explore','validation','test_exposed','Q_legacy_confirm_exposed'):
            grouped[row['dataset'],row['method'],row['role'],int(row['seed'])].append(row)
    values=[];by_condition=defaultdict(dict)
    for (dataset,method,role,seed),rows in grouped.items():
        macro=next(r for r in rows if r['level']=='macro');cells=[r for r in rows if r['level']=='cell'];tasks=[r for r in rows if r['level']=='task']
        if len(cells)!=6 or len(tasks)!=2:raise RuntimeError('Six cells and two tasks required')
        out=dict(dataset=dataset,method=method,role=role,seed=seed,natural_logloss=float(macro['natural_logloss']),
                 sample_weighted_cell_auc=sum(float(r['auc'])*int(r['n']) for r in cells)/sum(int(r['n']) for r in cells),
                 pooled_task_mean_auc=float(np.mean([float(r['pooled_auc']) for r in tasks])))
        values.append(out);by_condition[dataset,method,role][seed]=out
    pairs=[]
    for (dataset,candidate,role),entries in by_condition.items():
        if set(entries)!={5201,5202,5203,5204}:raise RuntimeError('Four independent population seeds required')
        for reference in ('NATIVE','H_DIRECT','PILOT_PROB','POOLED_TARGET','LOCAL_ACTIVE'):
            other=by_condition.get((dataset,reference,role))
            if not other or candidate==reference:continue
            for metric in ('natural_logloss','sample_weighted_cell_auc','pooled_task_mean_auc'):
                pairs.append(dict(dataset=dataset,candidate=candidate,reference=reference,role=role,metric=metric,
                                  **stats([entries[s][metric]-other[s][metric] for s in sorted(entries)])))
    write('CORE_population_four_seed_per_seed.csv',values);write('CORE_population_four_seed_paired_summary.csv',pairs)
    (REPORT/'POPULATION_FOUR_SEED_ANALYSIS_RECEIPT.json').write_text(json.dumps(dict(status='PASS',minimum_training_seeds=4,
         primary_metric_unchanged='six-cell MacroLL',estimands=['natural row-mean LL','sample-weighted cell AUC','mean of the two task-pooled AUCs'],
         warning='Weighted cell AUC and pooled AUC are different; population metrics do not retroactively change the primary gate'),indent=2)+'\n')
    for r in pairs:
        if r['role']=='test_exposed' and (r['candidate'],r['reference']) in [('RCF','NATIVE'),('RCF','H_DIRECT'),('PILOT_PROB','H_DIRECT')]:print(json.dumps(r))


if __name__=='__main__':main()
