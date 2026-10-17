from pathlib import Path
import json,hashlib,time,itertools
import numpy as np,pandas as pd
R=Path('/home/xzf/Code/phase0-3-20261016-v3-git');O=R/'research_plans/2026-10-16_v3/execution_20261016';D=O/'analysis/V7_local_all4_statistics';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
r=load(D/'receipt.json');assert r['status']=='COMPLETE' and r['source_sha']=='2cab2e40e03ea0c705fa9bf95332ffef36b6720cc4acad0e4e7652a156daa95f'
for n,h in r['hashes'].items():assert sha(D/n)==h
A=pd.read_csv(D/'absolute_seed_readouts.csv');V=pd.read_csv(D/'seed_contrasts.csv');S=load(D/'contrast_summary.json');lookup=A.set_index(['seed','epoch','split','variable','condition']).value.to_dict();ae=0.
for src in r['sources']:
 P=Path(src['directory'])
 for n,h in src['hashes'].items():assert sha(P/n)==h
 pr=load(P/'receipt.json');p=np.asarray(pr['config']['p']);cells=pd.read_csv(P/'cells.csv');summ=pd.read_csv(P/'summary.csv')
 for (condition,split),g in cells.groupby(['condition','split']):
  for metric in ('auc','logloss'):
   vals={'macro_'+metric:g[metric].mean(),'sample_weighted_cell_'+metric:np.average(g[metric],weights=g.n),'train_objective_weighted_cell_'+metric:np.average(g[metric],weights=p[g.domain.to_numpy(int)])};vals.update({f'd{d}_{metric}':sub[metric].mean() for d,sub in g.groupby('domain')});vals.update({f't{t}_{metric}':sub[metric].mean() for t,sub in g.groupby('task')});vals.update({f'd{x.domain}t{x.task}_{metric}':getattr(x,metric) for x in g.itertuples()})
   for var,val in vals.items():ae=max(ae,abs(lookup[src['seed'],src['epoch'],split,var,condition]-val))
 for x in summ.itertuples():ae=max(ae,abs(lookup[src['seed'],src['epoch'],'train','train_empirical_objective',x.condition]-x.train_empirical_objective))
ce=0.
for x in V.itertuples():
 get=lambda c:lookup[x.seed,x.epoch,x.split,x.variable,c]
 n,b,c=get('SN'),get('SB'),get(x.condition)
 if x.contrast=='condition_minus_BASE':value=c-get('BASE')
 elif x.contrast=='SB_minus_SN':value=b-n
 elif x.contrast=='hybrid_minus_SN':value=c-n
 elif x.contrast=='hybrid_minus_SB':value=c-b
 elif x.contrast=='attenuation':value=1-abs(c-n)/abs(b-n) if abs(b-n)>=1e-8 else np.nan
 else:raise AssertionError(x.contrast)
 if np.isnan(value):assert np.isnan(x.difference)
 else:ce=max(ce,abs(value-x.difference))
assert ae<1e-12 and ce<1e-9
K=['epoch','split','variable','condition','contrast'];groups={k:g.sort_values('seed').difference.to_numpy() for k,g in V.groupby(K)};se=0.
for n in range(1,5):
 subset=[s for s in S if s['n_observed']==n];rng=np.random.default_rng(20261016);draw=rng.integers(0,n,(20000,n));W=np.stack([(draw==i).sum(1)/n for i in range(n)],1);signs=np.array(list(itertools.product([-1,1],repeat=n)))
 for start in range(0,len(subset),32):
  part=subset[start:start+32];M=np.stack([groups[tuple(s[k] for k in K)][np.isfinite(groups[tuple(s[k] for k in K)])] for s in part]);ci=np.quantile(W@M.T,[.025,.975],axis=0);perm=signs@M.T/n
  for i,s in enumerate(part):
   assert s['n']==n;se=max(se,abs(s['mean']-M[i].mean()),abs(s['ci_low']-ci[0,i]),abs(s['ci_high']-ci[1,i]));assert s['positive']==int((M[i]>0).sum()) and s['negative']==int((M[i]<0).sum());assert s['paired_sign_randomization_p']==float(np.mean(abs(perm[:,i])>=abs(M[i].mean())-1e-15))
assert se<1e-9
proof=dict(status='PASS_ALL_ACTUAL_POINTS_CONTRASTS_AND_SEED_STATISTICS',time=time.time(),independent_review=False,receipt_sha=sha(D/'receipt.json'),absolute_rows=len(A),contrast_rows=len(V),summary_groups=len(S),max_absolute_error=ae,max_contrast_error=ce,max_statistics_error=se,test_access=False)
f=O/'V7_local_all4_statistics_primary_verification.json';assert not f.exists();f.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
