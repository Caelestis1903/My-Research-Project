"""Describe frozen label support and existing baseline results; no fitting or new evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
ART=Path('/home_new/xzf_natural_movielens_reassessment/data')
OUT=ROOT/'label_context'
OUT.mkdir(exist_ok=True)
rows=[];sources=[]
for pop in ('movielens_natural','movielens_artificial_d2_fixed2022','movielens_natural_matched_n187500'):
    path=ART/(pop+'.json');data=json.loads(path.read_text())['data']
    sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for split,cells in data['cell_stats'].items():
        for cell,stats in cells.items():
            domain=int(cell[1]);task=int(cell[3]);train=data['cell_stats']['train'][cell]['positive_rate']
            rate=stats['positive_rate']
            constant_ll=-rate*np.log(train)-(1-rate)*np.log1p(-train)
            rows.append(dict(population=pop,split=split,domain=domain,task=task,
                n=stats['count'],positive_count=stats['positive_count'],positive_rate=rate,
                entropy_bits=stats['entropy'],train_constant_logloss=constant_ll,
                positive_rate_minus_train=rate-train))
pd.DataFrame(rows).to_csv(OUT/'frozen_label_support.csv',index=False)
path=ROOT/'core/AC-2D/seed_cell_metrics.csv'
sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
f=pd.read_csv(path)
f=f[(f.arm=='SN')&f.checkpoint.isin(['selected','final'])]
g=f.groupby(['population','budget','checkpoint','domain','task']).agg(
    n_seeds=('seed','nunique'),auc_mean=('auc','mean'),auc_sd=('auc','std'),
    logloss_mean=('logloss','mean'),logloss_sd=('logloss','std'),
    constant_logloss=('train_constant_test_logloss','first'),
    logloss_skill=('logloss_skill_vs_train_constant','mean')).reset_index()
assert g.n_seeds.eq(4).all()
g.to_csv(OUT/'stage2d_SN_baseline_cells.csv',index=False)
(OUT/'source_files.json').write_text(json.dumps(sources,indent=2)+'\n')
print('label cells',len(rows),'existing full-seed baseline cells',len(g))
print(g[(g.population=='movielens_natural')&(g.budget=='full50')].round(6).to_string(index=False))
