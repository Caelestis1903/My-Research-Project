"""Exploratory domain/task associations, explicitly not causal regressions."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from v2_20261016 import OUT,dump,sha


def main():
    a=pd.read_csv(OUT/'analysis'/'aggregate_metrics.csv');effects=[]
    for (dataset,opt,variant,seed),z in a.query("split=='validation' and checkpoint=='selected'").groupby(['dataset','optimizer','variant','seed']):
        n=z[z.arm=='SN'];b=z[z.arm=='SB']
        if n.empty or b.empty:continue
        n=n.iloc[0];b=b.iloc[0];nr=Path(n.run);br=Path(b.run)
        config=json.loads((nr/'config.json').read_text());data=config['data'];profiles=data['cell_stats']['train']
        en=pd.read_csv(nr/'exposure.csv');eb=pd.read_csv(br/'exposure.csv')
        norms=OUT/'probes'/dataset/opt/f'seed{seed}'/'task_gradient_norms.csv'
        covariances=norms.with_name('task_covariance.csv')
        grad=pd.read_csv(norms) if norms.exists() else None;cov=pd.read_csv(covariances) if covariances.exists() else None
        for d in range(3):
            exp_n=float(en[f'exposure_d{d}'].sum());exp_b=float(eb[f'exposure_d{d}'].sum())
            for t in range(2):
                profile=profiles[f'd{d}t{t}'];baseline=float(n[f'auc_d{d}t{t}']);auc_delta=float(b[f'auc_d{d}t{t}']-baseline)
                row=dict(dataset=dataset,optimizer=opt,variant=variant,seed=seed,domain=d,task=t,
                    train_count=profile['count'],positive_rate=profile['positive_rate'],label_entropy=profile['entropy'],
                    baseline_auc=baseline,baseline_logloss=float(n[f'logloss_d{d}t{t}']),delta_auc=auc_delta,
                    delta_logloss=float(b[f'logloss_d{d}t{t}']-n[f'logloss_d{d}t{t}']),
                    exposure_n=exp_n,exposure_b=exp_b,exposure_ratio=exp_b/exp_n,
                    unique_n=int(en.iloc[-1][f'unique_cumulative_d{d}']),unique_b=int(eb.iloc[-1][f'unique_cumulative_d{d}']),
                    repeat_p95_n=float(en.iloc[-1][f'repeat_p95_d{d}']),repeat_p95_b=float(eb.iloc[-1][f'repeat_p95_d{d}']))
                if grad is not None:
                    zz=grad.query("snapshot=='selected' and domain==@d and task==@t")
                    row['baseline_gradient_norm']=float(zz.groupby('microbatch').gradient_norm.apply(lambda v:math.sqrt(np.sum(v*v))).mean())
                    row['projected_gradient_covariance_trace']=float(cov.query("snapshot=='selected' and domain==@d and task==@t").projected_within_domain_microbatch_covariance_trace.sum())
                effects.append(row)
    cells=pd.DataFrame(effects);cells.to_csv(OUT/'analysis'/'domain_task_profiles.csv',index=False)
    fits=[];correlations=[]
    for (dataset,opt,variant),z in cells.groupby(['dataset','optimizer','variant']):
        if 'projected_gradient_covariance_trace' not in z or z.projected_gradient_covariance_trace.isna().any():continue
        x=np.column_stack([np.log(z.train_count),1-z.baseline_auc,np.log(z.exposure_ratio),z.projected_gradient_covariance_trace])
        sd=x.std(0);standard=(x-x.mean(0))/np.where(sd>0,sd,1);design=np.column_stack([np.ones(len(x)),standard])
        beta,residual,rank,singular=np.linalg.lstsq(design,z.delta_auc,rcond=None)
        condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None
        for j,name in enumerate(['intercept','log_size','baseline_1_minus_auc','log_exposure_ratio','projected_covariance_trace']):
            fits.append(dict(dataset=dataset,optimizer=opt,variant=variant,term=name,coefficient=float(beta[j]),
                design_rank=int(rank),columns=5,n=len(z),physical_domain_clusters=3,condition_number=condition,
                status='DESCRIPTIVE_ONLY' if rank==5 else 'RANK_DEFICIENT_UNIDENTIFIED',
                inference='no causal interpretation; only3domains; seed/task repetitions not independent domains'))
        for name in ['train_count','baseline_auc','baseline_logloss','exposure_ratio','projected_gradient_covariance_trace']:
            corr=float(z[name].corr(z.delta_auc));correlations.append(dict(dataset=dataset,optimizer=opt,variant=variant,feature=name,correlation=corr))
    pd.DataFrame(fits).to_csv(OUT/'analysis'/'exploratory_association_regression.csv',index=False)
    pd.DataFrame(correlations).to_csv(OUT/'analysis'/'exploratory_associations.csv',index=False)
    trade=[]
    for keys,z in cells.groupby(['dataset','optimizer','variant','seed']):
        bydom=z.groupby('domain').delta_auc.mean();counts=z.groupby('domain').train_count.first();large=int(counts.idxmax())
        small=[d for d in counts.index if d!=large]
        trade.append(dict(zip(['dataset','optimizer','variant','seed'],keys),large_domain=large,
            large_effect=float(bydom[large]),**{f'd{d}_effect':float(bydom[d]) for d in range(3)},
            both_small_improve_large_declines=bool(all(bydom[d]>0 for d in small) and bydom[large]<0),
            any_small_improves_large_declines=bool(any(bydom[d]>0 for d in small) and bydom[large]<0)))
    pd.DataFrame(trade).to_csv(OUT/'analysis'/'small_large_tradeoff.csv',index=False)
    dump(OUT/'analysis'/'domain_receipt.json',dict(status='COMPLETE',profile_rows=len(cells),source_sha256=sha(__file__),
        size_not_difficulty=True,regression_exploratory_only=True,physical_clusters_per_environment=3))
    print('Domain/task exploratory profiles',len(cells))

if __name__=='__main__':main()
