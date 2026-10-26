from pathlib import Path
import json,hashlib,time,csv,sys,subprocess
import pandas as pd
R=Path('/home/xzf/Code/phase0-3-20261016-v3-git');O=R/'research_plans/2026-10-16_v3/execution_20261016';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
def write(p,x):
 assert not p.exists();p.write_text(json.dumps(x,indent=2)+'\n')
s=load(O/'V8_final350_strong_GPU_cell_manifest.status.json');assert s['status']=='COMPLETE' and all(t['status']=='COMPLETE' and t['exit_code']==0 for t in s['tasks'].values())
mf=O/'V8_final_domain_task_context_manifest.json';spec=load(mf);records=[];groups={};support_ref=None
for item in spec['records']:
 if item['seed']!=2025:continue
 pair=load(Path(item['pair'])/'receipt.json');groups[item['group']]=[]
 for arm,source,digest,cell_dir in zip(('SN','SB'),pair['source_runs'],pair['source_run_receipt_sha'],item['cell_dirs']):
  run=Path(source);out=Path(cell_dir);pr=load(run/'receipt.json');c=load(run/'config.json');rc=load(out/'receipt.json');assert sha(run/'receipt.json')==digest and pr['epochs']==350;assert rc['status']=='PASS' and rc['training_receipt_sha']==digest and rc['config_sha']==sha(run/'config.json') and rc['source_sha']==sha(R/'experiments/v3_cell_fullbudget_20261016.py') and rc['actual_budget']==350
  epochs=sorted(set([10,50,350,pr['selected_epoch']]));assert rc['epochs']==epochs
  sup=pd.read_csv(out/'cell_support.csv');perf=pd.read_csv(out/'cell_train_performance.csv');traj=pd.read_csv(run/'trajectory.csv').set_index('epoch');assert len(sup)==24 and len(perf)==6*len(epochs)
  reduced=sup[['split','domain','task','n','positive']].sort_values(['split','domain','task']).reset_index(drop=True)
  if support_ref is None:support_ref=reduced
  else:pd.testing.assert_frame_equal(support_ref,reduced)
  errors=[]
  for e in epochs:
   part=perf[perf.epoch==e];assert len(part)==6;er=abs(part.empirical_objective_contribution.sum()-traj.loc[e,'train_empirical_objective']);assert er<1e-6+1e-5*abs(traj.loc[e,'train_empirical_objective']);errors.append(float(er))
  record=dict(run=str(run),cell_dir=str(out),arm=arm,group=item['group'],cell_receipt_sha=sha(out/'receipt.json'),support_sha=sha(out/'cell_support.csv'),performance_sha=sha(out/'cell_train_performance.csv'),epochs=epochs,actual_budget=350,objective_sum_max_error=max(errors))
  records.append(record);groups[item['group']].append(dict(run=run,config=c,producer=pr,performance=perf))
strong=[x for x in records if x['group']=='strong'];write(O/'V8_final350_strong_cell_pair_primary_verification.json',dict(status='PASS_CELL_PAIR_OUTPUTS_COUPLING_STILL_PENDING',time=time.time(),independent_review=False,records=strong,test_access=False))
checks=[]
for ai,arm in enumerate(('SN','SB')):
 peers=[groups[g][ai] for g in ('clean','mild','strong')];base=peers[0]
 for peer in peers[1:]:
  for key in ['seed','arm','optimizer','lr','batch_size','replacement','order','duplication','p','q','importance_weight','data','init_sha256','split_sha256','source_hashes','weight_decay']:
   assert peer['config'][key]==base['config'][key],key
  keys=['epoch','domain','task','actual_exposure','positive_exposure','unique_exposed'];a=base['performance'];b=peer['performance'];a=a[a.epoch.isin([10,50,350])][keys].sort_values(keys[:3]).reset_index(drop=True);b=b[b.epoch.isin([10,50,350])][keys].sort_values(keys[:3]).reset_index(drop=True);pd.testing.assert_frame_equal(a,b)
 for epoch in range(1,351):
  hashes=[]
  for peer in peers:
   name=f'batches_epoch{epoch:03d}.npz';h=sha(peer['run']/name);assert h==peer['producer']['artifact_hashes'][name];hashes.append(h)
  assert len(set(hashes))==1
  checks.append(dict(arm=arm,epoch=epoch,three_corruption_batch_files_sha=hashes[0],physically_rehashed_files=3))
 if arm=='SN':print('SN all1050physicalbatchhashes PASS',flush=True)
f=O/'archive/formal/V8_seed2025_final350_batch_identity.csv';assert not f.exists();pd.DataFrame(checks).to_csv(f,index=False)
proof=O/'V8_seed2025_final350_cells_and_coupling_primary_verification.json';write(proof,dict(status='PASS_ALL6_ACTUAL350_CELLS_AND_ALL2100_BATCH_HASHES',time=time.time(),independent_review=False,records=records,all6_support_and_labels_identical=True,common_epoch_exposure_positive_exposure_unique_counts_identical_across_corruption=True,physical_batch_hash_files_verified=2100,batch_identity_csv=str(f.relative_to(O)),batch_identity_csv_sha=sha(f),scope='All350 sourcebatches physically rehashed; sameq row/order/quota and support/labels fixed across severity. Feature law/J_c changes; no causal-size or scalar-difficulty isolation.',test_access=False))
pins=[]
for item in spec['records']:
 for d in item['cell_dirs']:
  path=Path(d);rc=load(path/'receipt.json');assert rc['status']=='PASS';pins.append(dict(cell_dir=d,cell_receipt_sha=sha(path/'receipt.json'),support_sha=sha(path/'cell_support.csv'),performance_sha=sha(path/'cell_train_performance.csv')))
assert len(pins)==24
activation=O/'V8_final_domain_task_context_activation_review.json';write(activation,dict(status='PASS_ACTUAL_ALL24_CELLS_AND_COUPLING',time=time.time(),manifest_sha=sha(mf),cell_records=pins,independent_review=False,coupling_350_sha=sha(proof),coupling_200_sha=sha(O/'V8_seed2023_final200_cells_and_coupling_primary_verification.json'),test_access=False))
cmd=[sys.executable,str(R/'experiments/v3_domain_task_context_20261016.py'),'--manifest',str(mf),'--activation',str(activation),'--out',str(O/'analysis/V8_final_domain_task_context')];t=time.time();proc=subprocess.Popen(cmd,cwd=R);rc=proc.wait();write(O/'V8_final_domain_task_context_actual_launch.json',dict(pid=proc.pid,command=cmd,started=t,finished=time.time(),exit_code=rc,source_sha=sha(R/'experiments/v3_domain_task_context_20261016.py')));assert rc==0
print('V8coupling2100physicalhashes andcontext actual PASS',flush=True)
