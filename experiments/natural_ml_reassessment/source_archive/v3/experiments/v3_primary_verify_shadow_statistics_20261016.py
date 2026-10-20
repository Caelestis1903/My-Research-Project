from pathlib import Path
import sys,json,hashlib,time,itertools
import numpy as np,pandas as pd
R=Path('/home/xzf/Code/phase0-3-20261016-v3-git');O=R/'research_plans/2026-10-16_v3/execution_20261016';D=O/'analysis/V11_shadow_statistics';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((D/'receipt.json').read_text());assert r['status']=='COMPLETE' and r['source_sha']==sha(R/'experiments/v3_shadow_statistics_20261016.py') and not r['test_access']
for n,h in r['hashes'].items():assert sha(D/n)==h
A=pd.read_csv(D/'absolute_seed_readouts.csv');V=pd.read_csv(D/'seed_contrasts.csv');S=json.loads((D/'contrast_summary.json').read_text());key=['seed','epoch','split','variable','condition','norm_control','current_arm'];lookup=A.set_index(key).value.to_dict();maxabs=0.
# Independent absolute reconstruction from actual parent cell tables.
for src in r['sources']:
 P=Path(src['parent'])
 for n,h in src['hashes'].items():assert sha(P/n)==h
 pr=json.loads((P/'receipt.json').read_text());p=np.asarray(json.loads((Path(pr['source_runs'][0])/'config.json').read_text())['p']);cells=pd.read_csv(P/'cells.csv');summary=pd.read_csv(P/'summary.csv');md=['seed','epoch','condition','norm_control','current_arm']
 for kk,g in cells.groupby(md+['split']):
  seed,epoch,condition,norm,arm,split=kk
  for metric in ('auc','logloss'):
   values={'macro_'+metric:g[metric].mean(),'sample_weighted_cell_'+metric:np.average(g[metric],weights=g.n),'train_objective_weighted_cell_'+metric:np.average(g[metric],weights=p[g.domain.to_numpy(int)])}
   values.update({f'd{d}_{metric}':sub[metric].mean() for d,sub in g.groupby('domain')});values.update({f't{t}_{metric}':sub[metric].mean() for t,sub in g.groupby('task')});values.update({f'd{x.domain}t{x.task}_{metric}':getattr(x,metric) for x in g.itertuples()})
   for var,val in values.items():maxabs=max(maxabs,abs(lookup[seed,epoch,split,var,condition,norm,arm]-val))
 for x in summary.itertuples():maxabs=max(maxabs,abs(lookup[x.seed,x.epoch,'train','train_empirical_objective',x.condition,x.norm_control,x.current_arm]-x.train_empirical_objective))
assert maxabs<1e-12
err=0.
for x in V.itertuples():
 def get(c,n,a):return lookup[x.seed,x.epoch,x.split,x.variable,c,n,a]
 c,n,a=x.condition,x.norm_control,x.current_arm
 if x.contrast=='condition_minus_BASE':value=get(c,n,a)-get('BASE','BASE','BASE')
 elif x.contrast=='condition_minus_native':value=get(c,n,a)-get('native',n,a)
 elif x.contrast=='normmatched_minus_raw':value=get(c,n,a)-get(c,'raw',a)
 else:
  gap=get(c,n,'SB')-get(c,n,'SN')
  if x.contrast=='current_q_gap':value=gap
  elif x.contrast=='history_by_current_q_interaction':value=gap-get('native',n,'SB')+get('native',n,'SN')
  elif x.contrast=='normmatched_minus_raw_q_gap':value=gap-get(c,'raw','SB')+get(c,'raw','SN')
  else:raise AssertionError(x.contrast)
 err=max(err,abs(value-x.difference))
assert err<1e-12
K=['epoch','split','variable','condition','norm_control','current_arm','contrast'];groups={k:g.sort_values('seed').difference.to_numpy() for k,g in V.groupby(K)};matrix=np.stack([groups[tuple(s[k] for k in K)] for s in S]);assert matrix.shape==(len(S),4)
# One frozen set of bootstrap counts; independent vectorized checks across ALL summaries.
draw=np.random.default_rng(20261016).integers(0,4,size=(20000,4));weights=np.stack([(draw==i).sum(axis=1)/4 for i in range(4)],axis=1);sign=np.array(list(itertools.product([-1,1],repeat=4)));cierr=0.;meanerr=0.
for start in range(0,len(S),32):
 M=matrix[start:start+32];boot=weights@M.T;CI=np.quantile(boot,[.025,.975],axis=0);perm=sign@M.T/4
 for i,s in enumerate(S[start:start+32]):
  values=M[i];assert s['n']==4 and s['all_seeds_present'] and s['observed_seeds']==[2022,2023,2024,2025];meanerr=max(meanerr,abs(s['mean']-values.mean()));cierr=max(cierr,abs(s['ci_low']-CI[0,i]),abs(s['ci_high']-CI[1,i]));assert s['positive']==int((values>0).sum()) and s['negative']==int((values<0).sum());assert abs(s['paired_sign_randomization_p']-np.mean(abs(perm[:,i])>=abs(values.mean())-1e-15))<1e-12
assert cierr<1e-12 and meanerr<1e-12
out=dict(status='PASS_ALL_ABSOLUTE_CONTRAST_AND_SEED_STATISTICS',time=time.time(),independent_review=False,receipt_sha=sha(D/'receipt.json'),absolute_rows=len(A),contrasts=len(V),summary_groups=len(S),max_absolute_reconstruction_error=maxabs,max_contrast_error=err,max_mean_error=meanerr,max_bootstrap_CI_error=cierr,all_summary_sign_randomization_exact=True,test_access=False,scope='Actual four training seeds only, stage/variable separated. No localone-step to long-run necessity extrapolation; direction supplement pending.')
(O/'V11_shadow_statistics_actual_primary_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
