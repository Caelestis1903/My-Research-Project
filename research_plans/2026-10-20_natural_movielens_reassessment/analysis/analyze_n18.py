"""Primary reanalysis of original and natural N18 data-only evidence."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'n18';OUT.mkdir(exist_ok=True)
ROOTS={'artificial_historical':Path('/home/xzf/Code/phase0-3-20261018-git/research_plans/2026-10-18/execution/theory_data'),
       'natural':Path('/home_new/xzf_natural_movielens_reassessment/N18/theory_data')}
rows=[];sources=[];bounds=[];audits=[];cells=[];times=[]
for population,root in ROOTS.items():
    for name in ('P2_mask_bounds.csv','P2_equal_budget_audit.csv','data_cells.csv','P1_ML_rating_time_environments.csv','P1_ML_time_definition.json'):
        p=root/name;sources.append(dict(population=population,name=name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    f=pd.read_csv(root/'P2_mask_bounds.csv');f['population']=population
    assert len(f)==24 and set(f.seed)=={2022,2023,2024,2025}
    f['naive_width']=f.naive_upper-f.naive_lower;f['logic_width']=f.logic_upper-f.logic_lower
    f['relative_width_reduction']=1-f.logic_width/f.naive_width
    f['naive_risk_sign_identified']=(f.risk_difference_lower>0)|(f.risk_difference_upper<0)
    f['logic_risk_sign_identified']=(f.logic_risk_difference_lower>0)|(f.logic_risk_difference_upper<0)
    f['oracle_contained']=(f.oracle_positive_rate>=f.logic_lower)&(f.oracle_positive_rate<=f.logic_upper)
    f['same_information_lp_absolute_error']=abs(f.logic_upper-f.ordinary_same_information_lp_upper)
    assert f.oracle_contained.all() and f.same_information_lp_absolute_error.max()<1e-12
    bounds.append(f)
    a=pd.read_csv(root/'P2_equal_budget_audit.csv');a['population']=population
    a['random_error']=a.random_estimate-a.oracle;a['stratified_error']=a.logic_stratified_estimate-a.oracle
    a['stratified_se_smaller']=a.logic_stratified_se<a.random_se;audits.append(a)
    cc=pd.read_csv(root/'data_cells.csv');cc=cc[cc.dataset=='movielens'].copy();cc['population']=population;cells.append(cc)
    tt=pd.read_csv(root/'P1_ML_rating_time_environments.csv');tt['population']=population;times.append(tt)
    for mech,g in f.groupby('mechanism'):
        aa=a[a.mechanism==mech]
        rows.append(dict(population=population,mechanism=mech,n_records=len(g),n_mask_seeds=g.seed.nunique(),
            oracle_positive_rate=g.oracle_positive_rate.mean(),observed_positive_rate=g.observed_positive_rate.mean(),
            naive_width=g.naive_width.mean(),logic_width=g.logic_width.mean(),
            mean_relative_width_reduction=g.relative_width_reduction.mean(),
            naive_sign_identified=int(g.naive_risk_sign_identified.sum()),logic_sign_identified=int(g.logic_risk_sign_identified.sum()),
            lp_max_error=g.same_information_lp_absolute_error.max(),
            random_mean_se=aa.random_se.mean(),stratified_mean_se=aa.logic_stratified_se.mean(),
            stratified_se_smaller=int(aa.stratified_se_smaller.sum()),
            random_rmse=np.sqrt(np.mean(aa.random_error**2)),stratified_rmse=np.sqrt(np.mean(aa.stratified_error**2)),
            actual_random_labels=int(aa.label_budget.sum()),actual_stratified_labels=int(aa.actual_stratified_labels.sum())))
pd.DataFrame(sources).to_csv(OUT/'source_index.csv',index=False)
summary=pd.DataFrame(rows);summary.to_csv(OUT/'population_mechanism_summary.csv',index=False)
bounds=pd.concat(bounds);audits=pd.concat(audits)
bounds.to_csv(OUT/'all_bounds.csv',index=False);audits.to_csv(OUT/'all_audits.csv',index=False)
pd.concat(cells).to_csv(OUT/'data_cells.csv',index=False);pd.concat(times).to_csv(OUT/'time_environments.csv',index=False)
bounds.groupby(['population','mechanism','domain'])[['naive_width','logic_width','relative_width_reduction','oracle_positive_rate','observed_positive_rate']].agg(['mean','std','min','max']).to_csv(OUT/'domain_mask_summary.csv')
audits.groupby(['population','mechanism','domain'])[['random_se','logic_stratified_se','random_error','stratified_error']].agg(['mean','std','min','max']).to_csv(OUT/'domain_audit_summary.csv')
fig,axs=plt.subplots(1,2,figsize=(9,3.5))
for i,mech in enumerate(('MAR','MNAR')):
    g=summary[summary.mechanism==mech].set_index('population').loc[list(ROOTS)]
    x=np.arange(2)
    axs[i].bar(x-.17,g.naive_width,.34,label='Observation only')
    axs[i].bar(x+.17,g.logic_width,.34,label='Nested labels / same-info LP')
    axs[i].set_xticks(x,['Artificial historical','Natural']);axs[i].set_title(mech)
    axs[i].set_ylabel('Mean prevalence interval width');axs[i].set_ylim(0,.8)
axs[1].legend(fontsize=8);fig.tight_layout()
fig.savefig(OUT/'n18_identification_widths.png',dpi=180);fig.savefig(OUT/'n18_identification_widths.pdf');plt.close(fig)
(OUT/'analysis_receipt.json').write_text(json.dumps(dict(status='COMPLETE',training_runs=0,
    aggregation='unweighted mean over 3 physical domains x 4 mask/audit repetitions; fixed population per condition',
    new_result_type='finite-population controlled missingness, not neural performance',test_metrics_read=False),indent=2)+'\n')
print(summary.to_string(index=False))
