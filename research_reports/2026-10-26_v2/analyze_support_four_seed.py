"""Prespecified support/OOV strata from frozen predictions and natural IDs.

Descriptive diagnostics only. No stratum is used to choose a model or policy.
"""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from analyze_core_four_seed import read, sha, write, stats, ACTIVE, REPORT, require_complete_export_inventory

PREPARED=Path('/home/xzf/data/mdmt_next_20261024_strict_v1')
METHODS=('NATIVE','LOCAL','H_DIRECT','H_DIRECT_EFFORT','PILOT_PROB',
         'FULL_SOURCE_ALL','RCF','POOLED_TARGET','LOCAL_ACTIVE')
ROLES=('Q_explore','test_exposed','Q_legacy_confirm_exposed')


def main():
    inventory=read(ACTIVE/'FOUR_SEED_ACCEPTANCE.json')
    if inventory['status']!='PASS' or not inventory['analysis_authorized']:
        raise PermissionError('Four independent training seeds required')
    index=read(ACTIVE/'ARTIFACT_INDEX.json')
    require_complete_export_inventory(inventory,index)
    natural={};sources={};counts=[];rows=[];missing=[]
    for dataset in ('movielens','kuairand'):
        prep=PREPARED/dataset;manifest=read(prep/'DATA_MANIFEST.json')
        x=np.load(prep/'x.npy',mmap_mode='r');fit=np.load(prep/'rows_fit.npy')
        schema=manifest['features'];columns=schema['columns']
        categorical=[columns.index(c) for c in schema['sparse_columns'] if c!='domain_indicator']
        oov=np.any(x[:,categorical]==0,axis=1)
        natural[dataset]=(x,fit,oov,manifest)
        if dataset=='movielens':
            raw=pd.read_csv(manifest['raw_path'],usecols=['user_id','movie_id'])
            user=raw.user_id.to_numpy();item=raw.movie_id.to_numpy()
            if len(user)!=len(x):raise RuntimeError('Raw/prepared row alignment changed')
            for source in range(3):
                sf=fit[np.asarray(x[fit,-1],int)==source]
                sources[source]=(np.isin(user,np.unique(user[sf])),np.isin(item,np.unique(item[sf])))
        else:
            missing.append(dict(dataset=dataset,diagnostic='original user/item source-domain support',
                                status='MISSING',reason='Processed raw CSV has no trustworthy original user/item IDs; feature OOV is available and is not a substitute'))
    for prefix in index['prefixes']:
        dataset=prefix['dataset'];seed=int(prefix['seed']);root=Path(prefix['root'])
        if read(root/'EXPORT_ACCEPTANCE.json')['status']!='PASS':raise PermissionError('Raw export acceptance required')
        x,fit,oov,manifest=natural[dataset]
        for role in ROLES:
            with np.load(root/'prefix'/f'{role}.npz',allow_pickle=False) as data:
                row_id=data['row_id'];domain=data['domain'];y=data['y'];group=data['group']
                global_support=data['support'] if 'support' in data.files else None
            masks=[('feature_oov',-1,'ANY_OOV',oov[row_id]),
                   ('feature_oov',-1,'NO_OOV',~oov[row_id])]
            if global_support is not None:
                for j,name in enumerate(('user','item')):
                    masks.extend([('global_fit_'+name,-1,'SEEN',global_support[:,j]),
                                  ('global_fit_'+name,-1,'UNSEEN',~global_support[:,j])])
                for source,(us,it) in sources.items():
                    masks.extend([('source_fit_user_item',source,'BOTH_SEEN',us[row_id]&it[row_id]),
                                  ('source_fit_user_item',source,'EITHER_UNSEEN',~(us[row_id]&it[row_id]))])
            for kind,source,stratum,mask in masks:
                for target in range(3):
                    if source==target:continue
                    cell=mask&(domain==target)
                    counts.append(dict(dataset=dataset,seed=seed,role=role,kind=kind,source=source,
                                       stratum=stratum,target=target,n=int(cell.sum()),
                                       target_role_n=int((domain==target).sum())))
            for method in METHODS:
                with np.load(root/method/f'{role}.npz',allow_pickle=False) as data:
                    if not np.array_equal(data['row_id'],row_id):raise RuntimeError('Unpaired prediction rows')
                    p=data['p'].astype(float)
                for kind,source,stratum,mask in masks:
                    for target in range(3):
                        if source==target:continue
                        for task in range(2):
                            selected=mask&(domain==target);yy=y[selected,task];pp=np.clip(p[selected,task],1e-7,1-1e-7)
                            n=len(yy);pos=int(yy.sum())
                            rows.append(dict(dataset=dataset,seed=seed,method=method,role=role,kind=kind,
                                source=source,stratum=stratum,target=target,task=task,n=n,positive=pos,negative=n-pos,
                                auc=float(roc_auc_score(yy,pp)) if 0<pos<n else None,
                                logloss=float(np.mean(-yy*np.log(pp)-(1-yy)*np.log1p(-pp))) if n else None))
    keys=('dataset','method','role','kind','source','stratum','target','task')
    keyed=defaultdict(dict)
    for row in rows:keyed[tuple(row[k] for k in keys)][row['seed']]=row
    summaries=[];paired=[]
    for key,entries in keyed.items():
        if set(entries)!={5201,5202,5203,5204}:raise RuntimeError('Missing independent support seed')
        common=dict(zip(keys,key))
        for metric in ('auc','logloss'):
            if all(row[metric] is not None for row in entries.values()):
                summaries.append(dict(common,metric=metric,**stats([entries[s][metric] for s in sorted(entries)])))
            for reference in ('NATIVE','H_DIRECT','PILOT_PROB','POOLED_TARGET','LOCAL_ACTIVE'):
                other=keyed.get((key[0],reference)+key[2:])
                if reference==key[1] or not other:continue
                if all(entries[s][metric] is not None and other[s][metric] is not None for s in entries):
                    paired.append(dict(common,reference=reference,metric=metric,
                        **stats([entries[s][metric]-other[s][metric] for s in sorted(entries)])))
    write('SUPPORT_four_seed_counts.csv',counts)
    write('SUPPORT_four_seed_per_seed.csv',rows)
    write('SUPPORT_four_seed_metric_summary.csv',summaries)
    write('SUPPORT_four_seed_paired_summary.csv',paired)
    user_cells=defaultdict(list)
    for row in rows:
        if row['kind']=='global_fit_user':
            user_cells[row['dataset'],row['method'],row['role'],row['stratum'],row['seed']].append(row)
    user_pairs=[]
    for role in ROLES:
        for stratum in ('SEEN','UNSEEN'):
            for candidate,reference in (('RCF','H_DIRECT'),('PILOT_PROB','NATIVE'),
                                         ('RCF','POOLED_TARGET'),('RCF','LOCAL_ACTIVE')):
                for metric in ('auc','logloss'):
                    values=[]
                    for seed in (5201,5202,5203,5204):
                        c=user_cells['movielens',candidate,role,stratum,seed]
                        other=user_cells['movielens',reference,role,stratum,seed]
                        if len(c)!=6 or len(other)!=6 or any(r[metric] is None for r in c+other):break
                        values.append(float(np.mean([r[metric] for r in c])-np.mean([r[metric] for r in other])))
                    if len(values)==4:
                        user_pairs.append(dict(dataset='movielens',role=role,stratum=stratum,candidate=candidate,
                                               reference=reference,metric=metric,**stats(values)))
    write('SUPPORT_ML_user_macro_contrasts.csv',user_pairs)
    receipt=dict(status='PASS',minimum_training_seeds=4,methods=list(METHODS),roles=list(ROLES),
        stratum_selection='fit-only original-ID membership and frozen categorical OOV index0; no outcome-driven threshold',
        source_support='ML source-domain fit user+item overlap; source-indexed full-predictor diagnostics do not identify one source causal contribution',
        missing=missing,metrics_rows=len(rows),frozen_manifest_sha256=sha(ACTIVE/'FROZEN_MANIFEST.json'))
    (REPORT/'SUPPORT_FOUR_SEED_ANALYSIS_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',metric_rows=len(rows),missing=missing)))


if __name__=='__main__':main()
