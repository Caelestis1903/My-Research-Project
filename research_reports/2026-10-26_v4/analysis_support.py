"""Fit-identity support strata, solely from frozen prediction files."""
from pathlib import Path
import csv,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from analysis_pipeline import ROOT,METHODS,load_prediction,table
from experiments.mdmt_dfd_20261026.data import RoleBundle
from experiments.mdmt_dfd_20261026.metrics import auc,losses

data=RoleBundle('movielens');rows=[];cells=[]
for seed in [5101,5102]:
    for role in ['validation','Q_explore','test_exposed','Q_legacy_confirm_exposed']:
        for method in METHODS:
            path=ROOT/'final_evaluation/movielens'/f'seed{seed}'/'p2'/method/'selected'/f'{role}.npz'
            a=load_prediction(path);support=data.supports(a['row_id']);ll=losses(a['y'],a['p'])
            groups={'known_user':support[:,0],'unknown_user':~support[:,0]}
            groups.update({f'user{int(u)}_item{int(i)}':(support[:,0]==u)&(support[:,1]==i) for u in [False,True] for i in [False,True]})
            for name,mask in groups.items():
                values=[]
                for q in range(2):
                    for d in range(3):
                        selected=mask&(a['domain']==d);n=int(selected.sum());y=a['y'][selected,q]
                        v=dict(dataset='movielens',seed=seed,role=role,method=method,group=name,domain=d,task=q,n=n,positive=int(y.sum()),negative=int(n-y.sum()),AUC=auc(y,a['p'][selected,q]),LL=float(ll[selected,q].mean()) if n else None)
                        cells.append(v);values.append(v)
                rows.append(dict(dataset='movielens',seed=seed,role=role,method=method,group=name,n=int(mask.sum()),fraction=float(mask.mean()),macro_LL=float(np.mean([v['LL'] for v in values])) if all(v['LL'] is not None for v in values) else None,macro_AUC=float(np.mean([v['AUC'] for v in values])) if all(v['AUC'] is not None for v in values) else None,natural_LL=float(ll[mask].mean()) if mask.any() else None))
out=ROOT/'analysis'/'support';out.mkdir(parents=True,exist_ok=True)
table(out/'macro.csv',rows);table(out/'six_cells.csv',cells)
(out/'INDEX.json').write_text(json.dumps(dict(definition='Known raw user/item if present anywhere in natural fit; no evaluation labels used for group construction',rows=len(rows),cells=len(cells),single_class_empty_cell='NA, no silent dropped-cell macro',kuairand='Raw user/item IDs unavailable, no invented strata',no_model_inference=True),indent=2)+'\n')
print('Support analysis complete',len(rows),len(cells),flush=True)
