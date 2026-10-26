"""Forensic analysis of saved CSVs and verified paired prediction arrays."""
from pathlib import Path
import pandas as pd,numpy as np,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
rows=[]
for ds in ['kuairand','movielens']:
 p=ROOT/f'experiments/data/stage_S6_R2_raw_results/{ds}'
 for split,part in [('validation','val'),('test','test')]:
  df=pd.read_csv(p/f's6r2_partA_{part}_metrics_{ds}.csv')
  for rule,g in df.groupby('rule'):
   a=g[g.arm=='DH'].set_index('seed');b=g[g.arm=='F'].set_index('seed')
   for seed in sorted(set(a.index)&set(b.index)):
    if split=='test' and (not a.loc[seed].test_available or not b.loc[seed].test_available):continue
    for metric in ['M','T','N','D0','D1','D2','WD','WC','d0t0','d0t1','d1t0','d1t1','d2t0','d2t1']:
     av=float(a.loc[seed,metric]);bv=float(b.loc[seed,metric])
     if not np.isfinite(av+bv):continue
     rows.append(dict(dataset=ds,split=split,rule=rule,seed=int(seed),metric=metric,delta=av-bv,DH=av,F=bv,DH_epoch=int(a.loc[seed,'selected_epoch']),F_epoch=int(b.loc[seed,'selected_epoch']),grade='L2',comparison='exact selected CSV' if rule=='R0' else 'alternative selected CSV; verify test source, not grid equivalence'))
d=pd.DataFrame(rows);d.to_csv(OUT/'s6r2_paired_recomputed.csv',index=False)
g=d.groupby(['dataset','split','rule','metric']).delta
summary=g.agg(['count','mean','std','min','max']).reset_index();summary['positive_seeds']=g.apply(lambda a:int((a>0).sum())).to_numpy();summary.to_csv(OUT/'s6r2_summary.csv',index=False)
# Recompute historical S5 effects rather than trust aggregate table.
pdrows=pd.read_csv(ROOT/'experiments/data/stage_S5_outputs/stage_S5_paired_results.csv')
print('S5 paired schema',pdrows.columns.tolist(),flush=True)
for col in ['delta_auc','delta_logloss']:
 if col in pdrows:
  keys=[c for c in ['contrast','checkpoint','domain','task'] if c in pdrows]
  pdrows.groupby(keys)[col].agg(['count','mean','std','min','max']).reset_index().to_csv(OUT/f's5_recomputed_{col}.csv',index=False)
# Logged sensitivity factors: valid descriptors, not frozen-model Jacobians.
r=[]
for p in (ROOT/'research_R2/formal_runs_20261012').rglob('fusion_log.csv'):
 df=pd.read_csv(p)
 if 'rho_dt_0' not in df:continue
 for t in range(2):
  for dom in range(3):
   c=t*3+dom;rho=pd.to_numeric(df[f'rho_dt_{c}'],errors='coerce')
   coeff=df.sigmoid_aD*(1-df.sigmoid_bD)*rho*(1-rho)
   r.append(dict(source=str(p.relative_to(ROOT)),cell=c,domain=dom,task=t,epochs=df.epoch.nunique(),rho_min=rho.min(),rho_max=rho.max(),rho_final=rho.iloc[-1],saturated_epochs=int(((rho<.01)|(rho>.99)).sum()),derivative_scalar_min=coeff.min(),derivative_scalar_max=coeff.max(),derivative_scalar_final=coeff.iloc[-1],gate_entropy_final=float(df[f'gate_entropy_{c}'].iloc[-1]),status='LOG_ONLY_NO_EXPERT_DIFFERENCE_OR_LOGIT_JACOBIAN'))
pd.DataFrame(r).to_csv(OUT/'ao04_rho_log_diagnostics.csv',index=False)
print(summary[(summary.rule=='R0')&(summary.metric.isin(['M','T','N']))].to_string(index=False),flush=True)
