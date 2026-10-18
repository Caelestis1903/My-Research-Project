"""Read completed runs; development decisions stay separate from frozen test.

No training, checkpoint selection, retuning or per-seed arm exclusion occurs here.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from n19_data import OUT,dump,population
from n19_train import metrics

COMPARISONS={
 'stability': [('task_cf','fixed'),('task_dro','fixed'),('task_cf','task_mask'),('task_dro','task_mask'),('task_cf','vrex'),('task_dro','vrex'),('global_cf','fixed')],
 'noise': [('axis_channel','ind_channel'),('joint_axis','nested_ind'),('joint_axis','nested_axis'),('nested_axis','nested_ind'),('joint_axis','axis_channel'),('axis_channel','naive'),('joint_axis','naive'),('axis_channel','audit_fixed'),('joint_axis','audit_fixed'),('audit_fixed','naive'),('audit_only','naive'),('clean_oracle','naive')],
 'sampling': [('mask_copula','mask_independent'),('mask_copula','mask_antithetic'),('mask_copula','mask_positive'),('mask_copula_cv','mask_independent_cv'),('mask_copula_cv','mask_copula'),('mask_independent_cv','mask_independent'),('mask_copula','fixed'),('mask_copula_cv','fixed')],
 'natural': [('original_sample','fixed'),('original','fixed'),('mmoe','fixed'),('ple','fixed'),('rank_uniform','fixed'),('rank_protected','rank_uniform'),('rank_protected','fixed'),('cell_codebook','domain_codebook'),('cell_codebook','rank_uniform'),('domain_codebook','rank_uniform'),('interaction','fixed'),('interaction','domain_mix'),('interaction','full_mix'),('reliability_interaction','interaction'),('regret_router','full_mix'),('regret_router','utility_router'),('coalition_router','regret_router'),('coalition_router','utility_router')]
}
for candidate in ['rank_uniform','rank_protected','domain_codebook','cell_codebook','domain_mix','axis_mix','full_mix','interaction','reliability_interaction','utility_router','regret_router','coalition_router','entropy_router']:
    for control in ['original_sample','mmoe','ple']:
        COMPARISONS['natural'].append((candidate,control))
COMPARISONS['natural'].extend([('regret_router','fixed'),('coalition_router','fixed')])
COMPARISONS['natural'].extend([('regret_router','entropy_router'),('coalition_router','entropy_router'),('entropy_router','full_mix'),('entropy_router','fixed'),('entropy_router','utility_router')])
COMPARISONS['natural'].extend([('interaction','axis_mix'),('reliability_interaction','axis_mix'),('axis_mix','domain_mix'),('full_mix','axis_mix')])
COMPARISONS['natural'].extend([('cell_codebook','fixed'),('domain_codebook','fixed'),
 ('reliability_interaction','fixed'),('reliability_interaction','domain_mix'),('reliability_interaction','full_mix'),
 ('coalition_router','full_mix'),('axis_mix','fixed'),('domain_mix','fixed'),('full_mix','fixed'),
 ('utility_router','fixed'),('utility_router','full_mix')])
COMPARISONS['stability'].extend([('task_cf','original_sample'),('task_dro','original_sample')])
COMPARISONS['noise'].extend([('axis_channel','original_sample'),('joint_axis','original_sample')])
COMPARISONS['noise'].extend([('profile_axis','axis_channel'),('profile_axis','audit_fixed'),('profile_nested','nested_axis'),('profile_joint','joint_axis'),('profile_joint','profile_nested'),('profile_joint','audit_joint'),('profile_joint','nested_ind'),('profile_joint','audit_fixed'),('profile_joint','naive')])
COMPARISONS['noise'].extend([('nested_axis','axis_channel'),('nested_ind','ind_channel'),('profile_nested','profile_axis')])
COMPARISONS['sampling'].extend([('mask_residual_cv','mask_copula_cv'),('mask_residual_cv','mask_independent_cv'),('mask_residual_cv','mask_antithetic_cv'),('mask_antithetic_cv','mask_independent_cv'),('mask_copula_cv','mask_antithetic_cv'),('mask_residual_cv','fixed')])

def interval(values):
    x=np.asarray(values,float);n=len(x);mean=float(x.mean())
    half=float(student_t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n)) if n>1 else None
    return dict(n_seeds=n,mean=mean,ci95_low=mean-half if half is not None else None,
        ci95_high=mean+half if half is not None else None,positive_seeds=int((x>0).sum()),negative_seeds=int((x<0).sum()))

def collect(phase):
    cells=[];resources=[];cache={}
    for path in sorted((OUT/'runs').glob('*')):
        if not (path/'receipt.json').exists():continue
        config=json.loads((path/'config.json').read_text());receipt=json.loads((path/'receipt.json').read_text())
        if config['engineering'] or receipt['status']!='COMPLETE_TRAINING_TEST_DEFERRED':continue
        suite=config['suite'];dataset=config['dataset'];key=(dataset,suite,config.get('noise_interaction',0.))
        if key not in cache:cache[key]=population(*key)
        data=cache[key];base=dict(run_id=config['run_id'],suite=suite,dataset=dataset,method=config['method'],seed=config['seed'],tag=config['run_id'].split('__seed')[1].split('__',1)[1] if '__' in config['run_id'].split('__seed')[1] else '')
        observed_per_visit=receipt['labels_queried']/(config['epochs']*config['train_rows']) if suite=='sampling' else (2. if suite=='natural' and config['method']=='fixed' else None)
        resources.append(dict(**base,**receipt,epochs_budget=config['epochs'],model_parameters=config['parameters'],p6_label_observations_per_row_visit=observed_per_visit))
        for stage in ['selected','final']:
            file=path/(f'validation_{stage}.npz' if phase=='development' else f'test_{stage}.npz')
            if not file.exists():continue
            z=np.load(file);ix=data.development if phase=='development' else np.arange(len(z['labels']))
            for record in metrics(z['predictions'][ix],z['labels'][ix],z['domains'][ix]):
                record.update(base,stage=stage,phase=phase,train_n=int((data.train_x[data.main,-1]==record['domain']).sum()))
                prevalence=record['positives']/record['n'];h=-prevalence*np.log(prevalence)-(1-prevalence)*np.log1p(-prevalence)
                record.update(prevalence=prevalence,empirical_label_entropy=float(h),logloss_skill_vs_empirical_constant=float(1-record['logloss']/h))
                cells.append(record)
    frame=pd.DataFrame(cells);resource=pd.DataFrame(resources)
    if frame.empty:return frame,pd.DataFrame(),resource
    aggregates=[]
    columns=['run_id','suite','dataset','method','seed','tag','stage','phase']
    for key,group in frame.groupby(columns,dropna=False):
        rec=dict(zip(columns,key));weights=group.train_n.to_numpy(float);weights/=weights.sum()
        pop=group.n.to_numpy(float);pop/=pop.sum()
        rec.update(macro_auc=group.auc.mean(),macro_logloss=group.logloss.mean(),
            train_weighted_auc=float(group.auc@weights),train_weighted_logloss=float(group.logloss@weights),
            evaluation_weighted_auc=float(group.auc@pop),evaluation_weighted_logloss=float(group.logloss@pop),
            worst_cell_auc=float(group.auc.min()),worst_cell_logloss=float(group.logloss.max()))
        for task in range(2):
            g=group[group.task==task];rec[f'task{task}_auc']=g.auc.mean();rec[f'task{task}_logloss']=g.logloss.mean()
        aggregates.append(rec)
    aggregate=pd.DataFrame(aggregates)
    # Full-label natural reference shares the sampling data, main split and seeds.
    reference=aggregate[(aggregate.suite=='natural')&(aggregate.dataset=='movielens')&(aggregate.method=='fixed')].copy()
    if len(reference):reference['suite']='sampling';aggregate=pd.concat([aggregate,reference],ignore_index=True)
    cellref=frame[(frame.suite=='natural')&(frame.dataset=='movielens')&(frame.method=='fixed')].copy()
    if len(cellref):cellref['suite']='sampling';frame=pd.concat([frame,cellref],ignore_index=True)
    return frame,aggregate,resource

def summarize(phase):
    target=OUT/('development_readout' if phase=='development' else 'test_readout');target.mkdir(exist_ok=True)
    cells,aggregate,resources=collect(phase)
    if cells.empty:dump(target/'status.json',dict(status='NO_COMPLETED_RUNS',phase=phase));return
    cells.to_csv(target/'cells.csv',index=False);aggregate.to_csv(target/'runs.csv',index=False);resources.to_csv(target/'resources.csv',index=False)
    means=[]
    for key,g in aggregate.groupby(['suite','dataset','method','tag','stage']):
        row=dict(zip(['suite','dataset','method','tag','stage'],key));row['n_seeds']=g.seed.nunique()
        for metric in ['macro_auc','macro_logloss','train_weighted_auc','train_weighted_logloss','evaluation_weighted_auc','evaluation_weighted_logloss','worst_cell_auc','worst_cell_logloss','task0_auc','task1_auc','task0_logloss','task1_logloss']:
            stats=interval(g[metric]);row[metric]=stats['mean'];row[metric+'_sd']=float(g[metric].std(ddof=1)) if len(g)>1 else None
        means.append(row)
    pd.DataFrame(means).to_csv(target/'method_means.csv',index=False)
    comparisons=[];cellpairs=[]
    for (suite,dataset,stage,tag),g in aggregate.groupby(['suite','dataset','stage','tag']):
        for method,control in COMPARISONS[suite]:
            a=g[g.method==method].set_index('seed');b=g[g.method==control].set_index('seed');seeds=a.index.intersection(b.index)
            if not len(seeds):continue
            base=dict(suite=suite,dataset=dataset,stage=stage,tag=tag,method=method,control=control,seeds=','.join(map(str,sorted(seeds))))
            for metric in ['macro_auc','macro_logloss','train_weighted_auc','train_weighted_logloss','evaluation_weighted_auc','evaluation_weighted_logloss','worst_cell_auc','worst_cell_logloss','task0_auc','task1_auc','task0_logloss','task1_logloss']:
                comparisons.append(dict(**base,metric=metric,**interval(a.loc[seeds,metric].values-b.loc[seeds,metric].values)))
            cg=cells[(cells.suite==suite)&(cells.dataset==dataset)&(cells.stage==stage)&(cells.tag==tag)]
            for (domain,task),c in cg.groupby(['domain','task']):
                ca=c[c.method==method].set_index('seed');cb=c[c.method==control].set_index('seed');ss=ca.index.intersection(cb.index)
                if not len(ss):continue
                for metric in ['auc','logloss']:
                    cellpairs.append(dict(**base,domain=domain,task=task,metric=metric,train_n=int(ca.train_n.iloc[0]),baseline=float(cb.loc[ss,metric].mean()),**interval(ca.loc[ss,metric].values-cb.loc[ss,metric].values)))
    pd.DataFrame(comparisons).to_csv(target/'paired_comparisons.csv',index=False)
    pd.DataFrame(cellpairs).to_csv(target/'paired_cells.csv',index=False)
    dump(target/'status.json',dict(status='PARTIAL' if phase=='development' else 'FROZEN_READOUT',phase=phase,
        runs=int(resources.shape[0]),warning='Paired t intervals are descriptive with four seeds; cells/tasks are correlated. Historical test reuse remains exploratory.',test_used_for_method_changes=False))
    selected=pd.DataFrame(means);selected=selected[(selected.stage=='selected')]
    print(selected[['suite','dataset','method','tag','n_seeds','macro_auc','macro_logloss','train_weighted_auc']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','test'],default='development');a=p.parse_args();summarize(a.phase)
