"""S1 development analysis after all13 conditions have four trained seeds."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

REPORT=Path(__file__).resolve().parent
REPO=REPORT.parents[1]
ACTIVE=REPO/'artifacts/mdmt_rcf_20261026/v2_four_seed'
SEEDS=[5201,5202,5203,5204]
METHODS=['NATIVE','LOCAL','B_REFIT','H_DIRECT','ETA_DIRECT_MATCHED','H_DIRECT_EFFORT','GROUP_CAL','PILOT_PROB','PILOT_LOGIT','FULL_SOURCE','FULL_SOURCE_ALL','FULL_TARGET','RANDOM_FEATURE']


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(name,rows):
    rows=list(rows);path=REPORT/'tables'/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(rows)


def stat(values):
    a=np.asarray(values,float)
    if len(a)!=4 or not np.isfinite(a).all():raise RuntimeError('Four finite training-seed measurements required')
    m=float(a.mean());sd=float(a.std(ddof=1));d=3.182446305284263*sd/2
    return dict(n=4,mean=m,sample_sd=sd,t95_low=m-d,t95_high=m+d)


def main():
    all_data={};receipt=[];per_seed=[];cells=[];domain_task=[];groups=[];summaries=[];pairs=[];pair_values=[]
    for dataset in ['movielens','kuairand']:
        for seed in SEEDS:
            path=ACTIVE/'summaries'/f'{dataset}_{seed}.json';summary=read(path)
            for method in METHODS:
                completed=ACTIVE/'runs'/dataset/f'seed{seed}'/method/'TRAINING_COMPLETE.json'
                c=read(completed)
                if c['status']!='COMPLETE' or c['statistical_seed']!=seed:raise RuntimeError('Missing actual training evidence')
                receipt.append(dict(dataset=dataset,seed=seed,method=method,sha256=sha(completed)))
                m=summary['metrics'][method];all_data[dataset,seed,method]=m
                per_seed.append(dict(dataset=dataset,seed=seed,method=method,role='Q_explore',macro_logloss=m['macro_logloss'],macro_auc=m['macro_auc'],natural_logloss=m['natural_logloss'],pooled_auc_task0=m['pooled_auc'][0],pooled_auc_task1=m['pooled_auc'][1],worst_cell_auc=m['worst_cell_auc']))
                cells.extend(dict(dataset=dataset,seed=seed,method=method,role='Q_explore',**x) for x in m['cells'])
                for level in ['domain','task']:
                    domain_task.extend(dict(dataset=dataset,seed=seed,method=method,role='Q_explore',level=level,**x) for x in m[level])
                groups.extend(dict(dataset=dataset,seed=seed,method=method,**x) for x in summary['group_metrics'].get(method,[]))
    for dataset in ['movielens','kuairand']:
        for method in METHODS:
            for metric in ['macro_logloss','macro_auc','natural_logloss','worst_cell_auc']:
                summaries.append(dict(dataset=dataset,method=method,metric=metric,**stat([all_data[dataset,s,method][metric] for s in SEEDS])))
            for reference in ['NATIVE','LOCAL','H_DIRECT','H_DIRECT_EFFORT','PILOT_PROB','GROUP_CAL','FULL_SOURCE_ALL','FULL_TARGET','RANDOM_FEATURE']:
                if method==reference:continue
                for metric in ['macro_logloss','macro_auc']:
                    diff=[all_data[dataset,s,method][metric]-all_data[dataset,s,reference][metric] for s in SEEDS]
                    common=dict(dataset=dataset,candidate=method,reference=reference,metric=metric)
                    pairs.append(dict(common,**stat(diff),negative_seeds=sum(v<0 for v in diff),positive_seeds=sum(v>0 for v in diff)))
                    pair_values.extend(dict(common,seed=s,delta=v) for s,v in zip(SEEDS,diff))
    write('S1_Q_four_seed_per_seed.csv',per_seed)
    write('S1_Q_four_seed_six_cells.csv',cells)
    write('S1_Q_four_seed_domain_task.csv',domain_task)
    write('S1_Q_four_seed_groups.csv',groups)
    write('S1_Q_four_seed_summary.csv',summaries)
    write('S1_Q_four_seed_paired_summary.csv',pairs)
    write('S1_Q_four_seed_paired_differences.csv',pair_values)
    (REPORT/'S1_FOUR_SEED_ANALYSIS_RECEIPT.json').write_text(json.dumps(dict(status='PASS',datasets=['movielens','kuairand'],training_seeds=SEEDS,conditions=26,completed_condition_seed_receipts=104,role='Q_explore',exposed_test_not_used=True,cohort='development',actual_training_receipt_hashes=receipt),indent=2)+'\n')
    compact=[]
    for dataset in ['movielens','kuairand']:
        for method in METHODS:
            compact.append(dict(dataset=dataset,method=method,mean_ll=np.mean([all_data[dataset,s,method]['macro_logloss'] for s in SEEDS]),mean_auc=np.mean([all_data[dataset,s,method]['macro_auc'] for s in SEEDS])))
    print(json.dumps(compact,indent=2))


if __name__=='__main__':main()
