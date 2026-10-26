import sys,json,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd
R=Path('/home/xzf/Code/phase0-3-20261016-v3-git');O=R/'research_plans/2026-10-16_v3/execution_20261016'
sys.path.insert(0,str(R/'experiments'));import v3_readout_20261016 as r
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
results=[]
for dirname in sys.argv[1:]:
 D=O/'analysis'/dirname;receipt=json.loads((D/'receipt.json').read_text());assert receipt['status']=='COMPLETE'
 assert receipt['source_sha'] in ['3e2024481c2a5f86a24c34b6a503c370a096f3bc7ec0b1c39694684998fa2900','655e284c7c32052c2aea951ccf6005befc2dc8561c302736e775a9007c0dfe7c','0dc81c105f6ed5627c4749501d01838ca558e4534fafdda2ccbca8f01c48c087']
 assert receipt['freeze_sha']==sha(O/'analysis_rules_frozen.json')
 for name,h in receipt['hashes'].items():assert sha(D/name)==h,(D,name)
 if receipt['source_sha']=='0dc81c105f6ed5627c4749501d01838ca558e4534fafdda2ccbca8f01c48c087':
  assert sha(R/'experiments/v3_project_pair_prefix_20261016.py')==receipt['source_sha']
  prov=json.loads((D/'prefix_provenance.json').read_text());parent=Path(prov['parent']);pr=json.loads((parent/'receipt.json').read_text());T=prov['epochs']
  assert prov['status']=='EXACT_PREFIX_REUSE' and prov['test_access'] is False and prov['new_resampling'] is False and prov['independent_replicates_added'] is False
  assert sha(parent/'receipt.json')==prov['parent_receipt_sha'] and sha(parent/'bootstrap_curves.npz')==prov['parent_cache_sha']==pr['hashes']['bootstrap_curves.npz']
  with np.load(parent/'bootstrap_curves.npz') as zz,np.load(D/'bootstrap_curves.npz') as xx:
   for key in ['selection','diagnostic','point']:assert np.array_equal(xx[key],zz[key][:,:,:T]),key
   assert np.array_equal(xx['epochs'],zz['epochs'][:T])
  for i,link in enumerate(prov['sources']):
   child,ancestor=Path(link['prefix']),Path(link['parent']);cp=json.loads((child/'receipt.json').read_text());ap=json.loads((ancestor/'receipt.json').read_text())
   assert str(child)==receipt['source_runs'][i] and str(ancestor)==pr['source_runs'][i]
   assert sha(child/'receipt.json')==link['prefix_receipt_sha']==receipt['source_run_receipt_sha'][i]
   assert sha(ancestor/'receipt.json')==link['parent_receipt_sha']==pr['source_run_receipt_sha'][i]
   assert link['identical_prediction_and_model_files']==2*T and cp['epochs']==T<=ap['epochs']
   assert sha(child/'config.json')==sha(ancestor/'config.json')
   for epoch in range(1,T+1):
    for name in [f'validation_epoch{epoch:03d}.npz',f'epoch{epoch:03d}_model.pt']:
     assert cp['artifact_hashes'][name]==ap['artifact_hashes'][name]
     if epoch in [1,T]:assert sha(child/name)==sha(ancestor/name)==cp['artifact_hashes'][name]
   ct=pd.read_csv(child/'trajectory.csv');at=pd.read_csv(ancestor/'trajectory.csv');pd.testing.assert_frame_equal(ct,at.iloc[:T].reset_index(drop=True))
 z=np.load(D/'bootstrap_curves.npz');p=z['point'];epochs=z['epochs'];assert list(epochs)==list(range(1,len(epochs)+1));assert z['selection'].shape[0]==1000 and z['diagnostic'].shape[0]==200
 cells=pd.read_csv(D/'cell_trajectory.csv');err=0.;budget=[]
 for i,(source,h) in enumerate(zip(receipt['source_runs'],receipt['source_run_receipt_sha'])):
  d=Path(source);assert sha(d/'receipt.json')==h;pr=json.loads((d/'receipt.json').read_text());assert pr['epochs']==len(epochs)
  for name in ('trajectory.csv','config.json'):assert sha(d/name)==pr['artifact_hashes'][name]
  t=pd.read_csv(d/'trajectory.csv');assert t.epoch.tolist()==epochs.tolist()
  a=t.train_empirical_objective.to_numpy();prior=float(a[-10:-5].mean());recent=float(a[-5:].mean());gain=(prior-recent)/max(abs(prior),1e-12)
  on=r.onset(p[0,i],z['selection'][:,i],a);assert on==receipt['selection_onsets'][i]
  on2=r.onset(p[1,i],z['diagnostic'][:,i],a);assert on2==receipt['onsets'][i]
  b=receipt['optimization_budget'][i];assert abs(b['relative_improvement']-gain)<1e-12 and b['extension_required']==(gain>.005 and not on['confirmed_overfitting']);budget.append(b)
  for j,split in enumerate(('selection','diagnostic')):
   pp=cells[(cells.arm==('SN','SB')[i])&(cells.split==split)]
   assert len(pp)==len(epochs)*6
   v=pp.groupby('epoch')[['auc','logloss']].mean().to_numpy();err=max(err,float(abs(v-p[j,i]).max()))
 assert err<1e-12
 indices=r.selection_indices(p[0]);assert receipt['selected_epochs']=={rule:{arm:int(epochs[ix]) for arm,ix in zip(('SN','SB'),idx)} for rule,idx in indices.items()}
 sb=pd.read_csv(D/'selection_bootstrap.csv');assert len(sb)==4000
 for rule,idx in r.selection_indices(z['selection']).items():
  rr=sb[sb.rule==rule].sort_values('replicate');assert np.array_equal(rr.SN_epoch,epochs[idx[:,0]]) and np.array_equal(rr.SB_epoch,epochs[idx[:,1]])
 result=dict(pair=dirname,status='PASS',receipt_sha=sha(D/'receipt.json'),actual_epochs=len(epochs),max_point_error=err,selection_bootstrap_choices_checked=4000,budgets=budget)
 results.append(result)
out=O/(sys.argv[1]+'_primary_verification.json');out.write_text(json.dumps(dict(status='PASS',time=time.time(),independent_review=False,results=results,test_access=False,scope='Immutable paired and exact prefix validation: all cache arrays bitexact parent prefix, all producer model/prediction digest entries match, physical first/last files and all configs/trajectories/current receipts checked; all sixcell points,4000reselection choices,onsets,budget independently reproduced. No new independent draws or stopping assertion.'),indent=2)+'\n');print(json.dumps(results))
