"""Post-freeze analysis of saved information only; no model states required."""
from pathlib import Path
import argparse,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from analysis_pipeline import ROOT,METHODS,analyze_files,load_prediction,table,write
from experiments.mdmt_dfd_20261026.metrics import evaluate_probabilities,paired_summary
from experiments.mdmt_dfd_20261026.data import RoleBundle

def path_for(rec,role):
    return ROOT/'final_evaluation'/rec['dataset']/f"seed{rec['seed']}"/rec['stage']/rec['arm']/rec['variant']/f'{role}.npz'

def run(role):
    f=json.loads((ROOT/'FROZEN_EVALUATION_SPEC.json').read_text())
    records=f['predictors'];output=ROOT/'analysis'/'final'
    for state in ['selected','read_pass10_or_fixed']:
        files={}
        for seed in [5101,5102]:
            files[seed]={}
            for method in METHODS:
                variant='read_pass10' if state!='selected' and method in ['H_DIRECT','ETA_DIRECT','UNIFORM','GEOM','SHUFFLE','LOCAL_CAP','DFD'] else 'selected'
                matches=[r for r in records if r['dataset']=='movielens' and r['seed']==seed and r['stage']=='p2' and r['arm']==method and r['variant']==variant]
                if len(matches)!=1:raise RuntimeError('Frozen predictor variant mismatch')
                files[seed][method]=path_for(matches[0],role)
        analyze_files('movielens',role,state,files,output/'movielens'/role/state,True)
    for dataset in ['movielens','kuairand']:
        rows=[];cells=[];all_metrics={}
        data=RoleBundle(dataset)
        for seed in [5101,5102]:
            all_metrics[seed]={}
            for rec in records:
                if rec['dataset']!=dataset or rec['seed']!=seed or rec['stage']!='prefix':continue
                a=load_prediction(path_for(rec,role));m=evaluate_probabilities(a['y'],a['p'],a['domain'])
                name=rec['arm']+'_'+rec['variant'];all_metrics[seed][name]=m
                support=data.supports(a['row_id'])
                rows.append(dict(dataset=dataset,seed=seed,method=name,role=role,macro_LL=m['macro_logloss'],macro_AUC=m['macro_auc'],natural_LL=m['natural_logloss'],pooled_AUC0=m['pooled_auc'][0],pooled_AUC1=m['pooled_auc'][1],known_user_fraction=None if support is None else float(support[:,0].mean()),known_item_fraction=None if support is None else float(support[:,1].mean()),predictor_hash=rec['predictor_hash']))
                for c in m['cells']:cells.append(dict(dataset=dataset,seed=seed,method=name,role=role,**c))
        target=output/dataset/role/'prefix';target.mkdir(parents=True,exist_ok=True)
        table(target/'macro.csv',rows);table(target/'six_cells.csv',cells)
        comparisons={}
        for name in ['NATIVE_epoch24','LOCAL_pass12']:
            comparisons[name]={field:paired_summary([all_metrics[s][name][field]-all_metrics[s]['NATIVE_selected'][field] for s in [5101,5102]]) for field in ['macro_logloss','macro_auc']}
        write(target/'paired_seed_statistics.json',dict(direction='variant minus NATIVE_selected',comparisons=comparisons))
    print(role,'all final analysis complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',required=True,choices=['validation','Q_explore','test_exposed','Q_legacy_confirm_exposed']);run(p.parse_args().role)
