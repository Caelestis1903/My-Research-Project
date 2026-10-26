import pathlib,json,hashlib,time,itertools
import numpy as np,pandas as pd
r=pathlib.Path.cwd();o=r/'research_plans/2026-10-16_v3/execution_20261016';a=o/'analysis/V5_Adam_final_matched_aggregate';m=o/'analysis/V5_Adam_final_matched_joint_marginals';c=o/'analysis/V5_Adam_final_matched_contrasts'
def sh(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ld(p):return json.loads(pathlib.Path(p).read_text())
mx=0.;checks=0
def close(x,y):
 global mx,checks
 x=np.asarray(x,float);y=np.asarray(y,float);e=float(np.max(np.abs(x-y)));assert e<2e-12,e;mx=max(mx,e);checks+=x.size
manifest=ld(o/'V5_Adam_final_matched_aggregate_manifest.json');cells=pd.read_csv(a/'cell_trajectory.csv');traj=pd.read_csv(a/'domain_task_trajectory.csv');rows=pd.read_csv(m/'seed_readouts.csv');contr=pd.read_csv(c/'seed_contrasts.csv');budget={2022:50,2023:100,2024:100,2025:50}
for group,ds in manifest['groups'].items():
 for directory in ds:
  p=pathlib.Path(directory);rc=ld(p/'receipt.json');seed=rc['seed'];T=budget[seed]
  for name,digest in rc['hashes'].items():assert sh(p/name)==digest,(p,name)
  for source,digest in zip(rc['source_runs'],rc['source_run_receipt_sha']):
   rr=pathlib.Path(source);assert sh(rr/'receipt.json')==digest;assert ld(rr/'receipt.json')['epochs']==T
  orig=pd.read_csv(p/'cell_trajectory.csv');actual=cells[(cells.group==group)&(cells.seed==seed)].drop(columns='group').reset_index(drop=True);pd.testing.assert_frame_equal(orig.reset_index(drop=True),actual,check_exact=False,atol=1e-14,rtol=0)
  for (split,arm,epoch),part in orig.groupby(['split','arm','epoch']):
   t=traj[(traj.group==group)&(traj.seed==seed)&(traj.split==split)&(traj.arm==arm)&(traj.epoch==epoch)].iloc[0]
   for metric in ['auc','logloss']:
    close(t['macro_'+metric],part[metric].mean())
    for d,sub in part.groupby('domain'):close(t[f'd{d}_{metric}'],sub[metric].mean())
    for task,sub in part.groupby('task'):close(t[f't{task}_{metric}'],sub[metric].mean())
  point=np.load(p/'bootstrap_curves.npz')['point']
  for idx,x in rows[(rows.group==group)&(rows.seed==seed)].iterrows():
   var=x.variable;metric=var.rsplit('_',1)[1]
   def vals(arm):
    k=x.SN_epoch_or_horizon if arm=='SN' else x.SB_epoch_or_horizon
    part=orig[(orig.split==x.split)&(orig.arm==arm)]
    part=part[part.epoch<=k] if ('AULC' in x.readout) else part[part.epoch==k]
    if var.startswith('d') and var[1].isdigit() and var[2]=='t':part=part[(part.domain==int(var[1]))&(part.task==int(var[3]))]
    elif var.startswith('d') and var[1].isdigit():part=part[part.domain==int(var[1])]
    elif var.startswith('t') and var[1].isdigit():part=part[part.task==int(var[1])]
    elif var.startswith('sample_weighted_cell_'):
     return float(part.groupby('epoch').apply(lambda z:np.average(z[metric],weights=z.n),include_groups=False).mean())
    elif var.startswith('train_objective_weighted_cell_'):
     weights=np.array(ld(pathlib.Path(rc['source_runs'][0])/'config.json')['p'])
     return float(part.groupby('epoch').apply(lambda z:np.average(z[metric],weights=weights[z.domain.to_numpy(int)]),include_groups=False).mean())
    return float(part[metric].mean())
   sn,sb=vals('SN'),vals('SB');close([x.SN,x.SB,x.difference],[sn,sb,sb-sn])
keys=['seed','split','readout','variable']
b=rows[rows.group=='Adam_WR'].set_index(keys)
for group in ['Adam_WR','Adam_PERM']:
 q=rows[rows.group==group].set_index(keys).sort_index();z=contr[contr.group==group].set_index(keys).sort_index();ref=b.loc[q.index]
 close(z.interaction,q.difference-ref.difference);close(z.SN_condition_minus_baseline,q.SN-ref.SN);close(z.SB_condition_minus_baseline,q.SB-ref.SB)
for directory,frame,col in [(m,rows,'difference'),(c,contr,'interaction')]:
 for s in ld(directory/'summary.json'):
  v=frame[(frame.group==s['group'])&(frame.split==s['split'])&(frame.readout==s['readout'])&(frame.variable==s['variable'])].sort_values('seed')[col].to_numpy();assert len(v)==4;close(s['seed_values'],v);close(s['mean'],v.mean());close(s['sd'],v.std(ddof=1));assert s['positive']==int((v>0).sum()) and s['negative']==int((v<0).sum())
  rng=np.random.default_rng(20261016);boot=np.mean(v[rng.integers(0,4,size=(20000,4))],axis=1);close([s['ci_low'],s['ci_high']],np.quantile(boot,[.025,.975]));perm=np.array([np.mean(v*np.array(sign)) for sign in itertools.product([-1,1],repeat=4)]);close(s['paired_sign_randomization_p'],np.mean(abs(perm)>=abs(v.mean())-1e-15))
result=dict(status='PASS',time=time.time(),independent_review=False,actual_budgets=budget,point_and_statistics_values_checked=checks,max_error=mx,full_pair_output_hashes_and_current_producer_receipts_verified=True,contrast_receipt_sha=sh(c/'receipt.json'),test_access=False,scope='All source cell trajectories, domain/task/macro marginals, every fixed/selected seed readout and all paired contrasts and bootstrap CI/sign summaries independently reproduced. fixed_budget matched within each seed; selection chosen epochs generally unmatched.')
(o/'V5_Adam_final_matched_primary_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
