import sys,pathlib,json,hashlib,time
sys.path.insert(0,'experiments')
import numpy as np,pandas as pd
r=pathlib.Path.cwd();p=r/'research_plans/2026-10-16_v3/execution_20261016';sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest();load=lambda q:json.loads(q.read_text());write=lambda q,x:q.write_text(json.dumps(x,indent=2)+'\n')
sampler=p/'analysis/V5_seed2025_final350_sampler_identity';weight=p/'analysis/V5_seed2025_final350_weight_tail';qstatus=load(p/'V5_seed2025_final350_sampler_and_weight_CPU_manifest.status.json');assert qstatus['status']=='COMPLETE' and all(x['status']=='COMPLETE' and x['exit_code']==0 for x in qstatus['tasks'].values())
sr=load(sampler/'receipt.json');wr=load(weight/'receipt.json');review=load(p/'V5_seed2025_final350_sampler_and_weight_activation_review.json');assert sr['status']==wr['status']=='PASS'
assert sr['source_sha']==review['sources']['v3_sampler_readout_20261016.py']==sha(r/'experiments/v3_sampler_readout_20261016.py')
assert wr['source_sha']==review['sources']['v3_final_batch_weight_tail_audit_20261016.py']==sha(r/'experiments/v3_final_batch_weight_tail_audit_20261016.py')
inputs=load(p/'V5_seed2025_candidate_final350_weight_tail_inputs.json');assert sr['comparisons']==inputs['comparisons'];producers={};traces={};trajectories={};critical=0;endpoints=set()
for text in inputs['runs']:
 d=pathlib.Path(text);pr=load(d/'receipt.json');cfg=load(d/'config.json');assert pr['epochs']==350 and pr['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
 for name in ['config.json','trajectory.csv','batch_trace.csv','optimizer_observations.csv','exposure_final.npz']:
  assert sha(d/name)==pr['artifact_hashes'][name];critical+=1
 producers[text]=(pr,cfg);traces[text]=pd.read_csv(d/'batch_trace.csv');trajectories[text]=pd.read_csv(d/'trajectory.csv').set_index('epoch')
identity=pd.read_csv(sampler/'intervention_identity.csv');assert len(identity)==1400 and set(identity.identity)=={'PASS'} and len(sr['comparisons'])==4
for comp in inputs['comparisons']:
 part=identity[(identity.baseline==comp['baseline'])&(identity.variant==comp['variant'])&(identity.intervention==comp['intervention'])].sort_values('epoch');assert part.epoch.tolist()==list(range(1,351))
 for row in part.itertuples():
  name=f'batches_epoch{row.epoch:03d}.npz'
  for text,hashval in [(row.baseline,row.baseline_batch_sha),(row.variant,row.variant_batch_sha)]:
   assert producers[text][0]['artifact_hashes'][name]==hashval
   if row.epoch in [1,350]:
    file=pathlib.Path(text)/name
    if str(file) not in endpoints:assert sha(file)==hashval;endpoints.add(str(file))
 assert part.order_changed.all() if comp['intervention']=='ORDER' else not part.order_changed.any()
coverage=pd.read_csv(sampler/'coverage_repetition.csv');assert len(coverage)==6300
for text,(pr,cfg) in producers.items():
 part=coverage[coverage.run==text];assert len(part)==1050
 for row in part.itertuples():
  t=trajectories[text].loc[row.epoch];d=int(row.domain);N=int(cfg['data']['train_domain_counts'][str(d)]);E=int(t[f'exposure_d{d}']);U=int(t[f'unique_epoch_d{d}'])
  assert row.pool_support==N and row.draws==E and row.unique==U and row.repeated_draws==E-U
  assert row.min_repeated_draws_at_fixed_quota==max(0,E-N)
  assert abs(row.coverage_epoch-U/N)<1e-12 and abs(row.coverage_cumulative-t[f'coverage_d{d}'])<1e-12
  hist={int(k):int(v) for k,v in (x.split(':') for x in row.repeat_histogram.split(';'))};assert sum(hist.values())==N and sum(k*v for k,v in hist.items())==E and sum(v for k,v in hist.items() if k>0)==U
 first=pd.read_csv(sampler/'first_seen_summary.csv');sub=first[first.run==text];assert len(sub)==3
 assert sub.pool_support.sum()==sum(cfg['data']['train_domain_counts'].values()) and (sub.seen_rows+sub.never_seen==sub.pool_support).all()
 for row in sub.itertuples():
  t=trajectories[text];assert row.total_draws==t[f'exposure_d{row.domain}'].sum()
  assert row.seen_rows==t.iloc[-1][f'unique_cumulative_d{row.domain}']
mov=pd.read_csv(sampler/'moving_theta_optimizer_observations.csv');assert len(mov)==50400
for text in producers:
 original=pd.read_csv(pathlib.Path(text)/'optimizer_observations.csv');actual=mov[mov.run==text][original.columns].reset_index(drop=True);pd.testing.assert_frame_equal(original,actual,check_exact=False,atol=1e-13,rtol=1e-13)
write(p/'V5_seed2025_final350_sampler_actual_primary_verification.json',dict(status='PASS_ACTUAL_FINAL350_SAMPLER_IDENTITY',time=time.time(),independent_review=False,receipt_sha=sha(sampler/'receipt.json'),actual_comparisons=4,actual_arm_comparison_epochs=1400,current_sources=6,critical_physical_hashes=critical,all_recorded_batch_hashes_match_current_producers=True,physical_endpoint_batch_files_rehashed=len(endpoints),coverage_rows=6300,moving_theta_rows=50400,hashes={x.name:sha(x) for x in sampler.iterdir() if x.is_file()},scope='Frozenproducer physicallyverified ALLNPZ androw/domain/orderidentity. Primary independentlybinds ALLrecorded hashes tocurrentproducers, rehashesphysicalfirst/lastbatchfiles andallcriticalfiles, reconstructs ALLcoverage/repeat hist/counts andrawtelemetry copies. No zero-repeat/unique-repetition-necessity claim.',test_access=False));print('SAMPLERrootPASS',flush=True)
# Independent actual trace formulas and short-tail position reconstruction.
epochs=pd.read_csv(weight/'epoch_weight_and_tail.csv');assert len(epochs)==2100 and wr['actual_epochs_audited']==2100 and wr['runs']==6;maximum=0.;summaries=[]
for text,(pr,cfg) in producers.items():
 df=traces[text];assert len(df)==350*49 and not df.duplicated(['epoch','batch']).any();n=df.n.to_numpy();counts=df[['d0','d1','d2']].to_numpy();w=np.asarray(cfg['p'])/np.asarray(cfg['q']);assert np.array_equal(counts.sum(1),n)
 first=(counts*w).sum(1);second=(counts*w*w).sum(1);ess=first*first/second;mu=first/n;variance=(counts*(w[None,:]-mu[:,None])**2).sum(1)/n;mx=np.max(np.where(counts>0,w[None,:],-np.inf),axis=1)
 for computed,col in [(ess,'ess'),(variance,'weight_variance'),(mx,'max_weight')]:
  error=float(np.max(abs(computed-df[col].to_numpy())/np.maximum(abs(df[col].to_numpy()),1)));assert error<1e-10;maximum=max(maximum,error)
 moved=0
 for e,part in df.groupby('epoch',sort=True):
  assert part.batch.tolist()==list(range(49));assert int(part.n.sum())==sum(cfg['data']['train_domain_counts'].values());short=part[part.n<cfg['batch_size']];assert len(short)==1;position=int(short.iloc[0].batch);tail=int(short.iloc[0].n);assert tail==40052
  if cfg['order']=='ORIGINAL':assert position==48
  moved+=position!=48
  x=epochs[(epochs.run==text)&(epochs.epoch==e)];assert len(x)==1;x=x.iloc[0]
  assert x.short_batch_actual_position==position and x.short_batch_size==tail and bool(x.short_batch_moved_from_last)==(position!=48)
  assert abs(x.mean_ess_fraction-(part.ess/part.n).mean())<1e-12 and abs(x.mean_weight_variance-part.weight_variance.mean())<1e-12
 summaries.append(dict(run=pathlib.Path(text).name,seed=cfg['seed'],optimizer=cfg['optimizer'],arm=cfg['arm'],replacement=cfg['replacement'],order=cfg['order'],epochs=350,ess_fraction_mean=float((df.ess/df.n).mean()),weight_variance_mean=float(df.weight_variance.mean()),max_weight=float(df.max_weight.max()),tail_moved_epochs=moved))
for c in inputs['comparisons']:
 a,b=traces[c['baseline']],traces[c['variant']]
 for e in range(1,351):
  left,right=a[a.epoch==e],b[b.epoch==e];fields=['n','d0','d1','d2'];x,y=left[fields].to_numpy(),right[fields].to_numpy()
  if c['intervention']=='PERM':assert np.array_equal(x,y)
  else:assert sorted(map(tuple,x))==sorted(map(tuple,y))
old=p/'analysis/V5_completed34_actual_final_batch_weight_tail';oldproof=load(p/'V5_completed34_batch_weight_tail_primary_verification.json');assert sha(old/'epoch_weight_and_tail.csv')==oldproof['epoch_csv_sha'];assert sha(old/'primary_per_run_summary.csv')==oldproof['primary_per_run_summary_sha']
all_epochs=pd.concat([pd.read_csv(old/'epoch_weight_and_tail.csv'),epochs],ignore_index=True);all_summary=pd.concat([pd.read_csv(old/'primary_per_run_summary.csv'),pd.DataFrame(summaries)],ignore_index=True);assert len(all_epochs)==4200 and all_epochs.run.nunique()==40 and not all_epochs.duplicated(['run','epoch']).any();assert len(all_summary)==40 and not all_summary.run.duplicated().any();assert int(all_summary.epochs.sum())==4200
order=all_epochs[all_epochs.order=='PERMUTED'];assert len(order)==1000;out=p/'analysis/V5_all40_final_batch_weight_tail_primary';assert not out.exists();out.mkdir();all_epochs.to_csv(out/'epoch_weight_and_tail.csv',index=False);all_summary.to_csv(out/'primary_per_run_summary.csv',index=False)
result=dict(status='PASS_ALL40_ACTUAL_FINALS',time=time.time(),independent_review=False,new6_receipt_sha=sha(weight/'receipt.json'),old34_primary_sha=sha(p/'V5_completed34_batch_weight_tail_primary_verification.json'),all_actual_runs=40,all_actual_arm_epochs=4200,all_actual_batches=int(all_epochs.batches.sum()),actual_ORDER_arm_epochs=len(order),actual_ORDER_tail_moved_epochs=int(order.short_batch_moved_from_last.sum()),new6_summary=summaries,max_new_formula_relative_error=maximum,all_ORIGINAL_short_tails_last_primary_asserted=True,scope='OriginalNATURALguardinactive forORIGINALconfigs; primaryactualpositions independentlyasserted. Actualtrace/weights/counts identities, no ESS/covariance/performance necessity inference. SourceallrawNPZ identities fromsamplerreceipt/rootpins.',output_hashes={x.name:sha(x) for x in out.iterdir() if x.is_file()},test_access=False)
assert result['all_actual_batches']==205800;write(out/'receipt.json',result);write(p/'V5_all40_final_batch_weight_tail_primary_verification.json',result);print(json.dumps({k:result[k] for k in ['status','all_actual_runs','all_actual_arm_epochs','all_actual_batches','actual_ORDER_arm_epochs','actual_ORDER_tail_moved_epochs','max_new_formula_relative_error']}),flush=True)
