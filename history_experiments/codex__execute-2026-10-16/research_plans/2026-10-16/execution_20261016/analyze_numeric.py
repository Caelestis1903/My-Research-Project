"""Numerical aggregation only; scientific interpretation is in mechanism_analysis.md."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
OLD=OUT.parents[2]/'research_plans/2026-10-15/execution_20261015/C2_numeric_oracle_20261016'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,v):(OUT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
receipts=sorted((OUT/'numeric_oracle').rglob('receipt.json'))
assert len(receipts)==8
frames={k:[] for k in ['estimator_statistics','state_trace','state_contrasts','coordinate_contrasts']}
checks=[]; covariance_checks=[]
for p in receipts:
    receipt=json.loads(p.read_text());assert receipt['status']=='PASS_NUMERIC_ONLY'
    for n,h in receipt['artifact_hashes'].items():
        actual=sha(p.parent/n);checks.append({'path':str((p.parent/n).relative_to(OUT)),
             'expected':h,'actual':actual,'match':actual==h});assert actual==h
    for k in frames:frames[k].append(pd.read_csv(p.parent/f'{k}.csv'))
    oldpath=OLD/receipt['dataset']/f"seed{receipt['seed']}"/'projected_covariance.npz'
    with np.load(oldpath,allow_pickle=False) as old,np.load(p.parent/'coordinate_covariance.npz',allow_pickle=False) as new:
        for a in ['Equal','Sample']:
            for q in ['Natural','Balanced']:
                expected=old[f'{a}_{q}_B16_exact']; actual=new[f'{a}_{q}_B16']
                error=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-15))
                covariance_checks.append(dict(dataset=receipt['dataset'],seed=receipt['seed'],objective=a,
                    sampler=q,relative_covariance_residual=error));assert error<1e-10
for k in frames:
    frames[k]=pd.concat(frames[k],ignore_index=True)
    frames[k].to_csv(OUT/f'{k}.csv',index=False)
pd.DataFrame(checks).to_csv(OUT/'numeric_hash_checks.csv',index=False)
pd.DataFrame(covariance_checks).to_csv(OUT/'old_new_covariance_checks.csv',index=False)
e=frames['estimator_statistics'];s=frames['state_contrasts']
ratios=[]
for (ds,seed,a,b),v in e.groupby(['dataset','seed','objective','batch_size']):
    n=v[v.sampler=='Natural'].iloc[0]; q=v[v.sampler=='Balanced'].iloc[0]
    ratios.append(dict(dataset=ds,seed=seed,objective=a,batch_size=b,
        projected_covariance_trace_ratio=q.projected_cov_trace/n.projected_cov_trace,
        natural_ess_fraction=n.mean_ess_fraction,balanced_ess_fraction=q.mean_ess_fraction))
ratios=pd.DataFrame(ratios);ratios.to_csv(OUT/'estimator_paired_ratios.csv',index=False)
summary=s.groupby(['dataset','objective','batch_size','optimizer','variable','partition']).agg(
    seeds=('seed','count'),mean_difference_norm=('mean_difference_norm','mean'),
    min_difference_norm=('mean_difference_norm','min'),max_difference_norm=('mean_difference_norm','max'),
    mean_mc_se_norm=('mc_se_vector_norm','mean'),minimum_max_coordinate_z=('max_coordinate_z','min'),
    maximum_max_coordinate_z=('max_coordinate_z','max'),
    mean_pathwise_rms=('pathwise_rms_difference','mean'))
summary.to_csv(OUT/'numeric_seed_summary.csv')
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,dataset in zip(axes,['kuairand','movielens']):
    for marker,opt in [('o','Adam'),('s','SGD')]:
        for color,obj in [('tab:blue','Equal'),('tab:orange','Sample')]:
            v=s[(s.dataset==dataset)&(s.optimizer==opt)&(s.objective==obj)&
                (s.variable=='cumulative')&(s.partition=='C')]
            ax.scatter(v.batch_size,v.mean_difference_norm,label=f'{opt}/{obj}',c=color,marker=marker,alpha=.7)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_title(dataset)
    ax.set_xlabel('IID batch size');ax.set_ylabel('Norm of mean cumulative delta difference (C projection)')
    ax.legend(fontsize=7)
fig.savefig(OUT/'state_divergence.png',dpi=160);fig.savefig(OUT/'state_divergence.pdf')
plt.close(fig)
# SGD has no persistent m/v: the SGD mhat/vhat columns are instantaneous g/g² only.
sgd=s[(s.optimizer=='SGD')&(s.variable=='cumulative')]
adam=s[(s.optimizer=='Adam')&(s.variable=='cumulative')]
dump('numeric_analysis_receipt.json',{'status':'PASS_NUMERICAL_AGGREGATION_ONLY','cases':len(receipts),
    'training_runs':0,'artifact_hash_checks':len(checks),'all_hashes_match':True,
    'old_new_covariance_comparisons':len(covariance_checks),
    'max_covariance_relative_residual':float(pd.DataFrame(covariance_checks).relative_covariance_residual.max()),
    'max_exact_gradient_relative_residual':float(e.exact_gradient_relative_residual.max()),
    'max_gradient_mc_z':float(e.mc_gradient_max_z.max()),
    'max_sgd_cumulative_contrast_coordinate_z':float(sgd.max_coordinate_z.max()),
    'min_adam_cumulative_contrast_max_coordinate_z':float(adam.max_coordinate_z.min()),
    'elapsed_producer_seconds_sum':sum(json.loads(p.read_text())['elapsed_seconds'] for p in receipts),
    'independent_review':False,'scientific_interpretation':'MAIN_AGENT_REQUIRED',
    'sgd_state_columns_note':'m/v zero; mhat/vhat denote instantaneous gradient/square, not persistent state',
    'scope':'initial theta, 48-row finite pool, selected coordinates, fixed-theta replay; not training',
    'source_sha256':sha(__file__)})
print('NUMERIC_AGGREGATION_8_CASES_COMPLETE')
