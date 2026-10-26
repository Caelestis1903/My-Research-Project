from pathlib import Path
import json,sys,hashlib,subprocess,time
import numpy as np,pandas as pd
R=Path('/home/xzf/Code/phase0-3-20261016-v3-git');O=R/'research_plans/2026-10-16_v3/execution_20261016';sys.path.insert(0,str(R/'experiments'));import v3_audit_20261016 as audit
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
freeze=load(O/'test_readout_frozen.json');assert freeze['status']=='FROZEN';rec=load(O/'test_prefix_oracle_item.json');x=rec['item'];parent=Path(x['parent']);target=Path(x['target']);native=load(parent/'test_readout_receipt.json');assert native['status']=='COMPLETE' and native['test_global_freeze_sha']==sha(O/'test_readout_frozen.json');assert len(native['epochs'])==100
q=load(O/'test_native_oracle_manifest.status.json');assert q['status']=='COMPLETE' and all(t['status']=='COMPLETE' and t['exit_code']==0 for t in q['tasks'].values())
if not (target/'test_readout_receipt.json').exists():
 cmd=[sys.executable,str(R/'experiments/v3_project_test_prefix_20261016.py'),'--manifest',str(O/'test_prefix_reuse_manifest.json'),'--index',str(rec['index'])];start=time.time();proc=subprocess.Popen(cmd,cwd=R);rc=proc.wait();launch=dict(pid=proc.pid,command=cmd,started=start,finished=time.time(),exit_code=rc,source_sha=sha(R/'experiments/v3_project_test_prefix_20261016.py'));assert not (O/'test_prefix_oracle_actual_projection_launch.json').exists();(O/'test_prefix_oracle_actual_projection_launch.json').write_text(json.dumps(launch,indent=2)+'\n');assert rc==0
 
else:
 assert load(O/'test_prefix_oracle_actual_projection_launch.json')['exit_code']==0
 assert load(target/'test_readout_receipt.json')['test_global_freeze_sha']==sha(O/'test_readout_frozen.json')
c=load(target/'config.json');pc=load(parent/'config.json');assert c==pc and sha(target/'config.json')==x['config_sha'];b=audit.bundle(c['dataset'],c['seed']);assert b.manifest_fields()==c['data'];assert c['corruption']==0
tr=load(target/'receipt.json');pr=load(parent/'receipt.json');table=pd.read_csv(target/'metrics_test.csv').set_index(['epoch','domain','task']);pt=pd.read_csv(parent/'metrics_test.csv');pd.testing.assert_frame_equal(table.reset_index().sort_index(axis=1),pt[pt.epoch<=50].reset_index(drop=True).sort_index(axis=1),check_exact=False,atol=1e-14,rtol=0)
rows=[];error=0.
for e in range(1,51):
 mn=f'epoch{e:03d}_model.pt';assert sha(target/mn)==sha(parent/mn)==tr['artifact_hashes'][mn]==pr['artifact_hashes'][mn]
 predname=f'test_epoch{e:03d}.npz';predsha=sha(parent/predname);assert predsha==native['hashes'][predname];pred=np.load(parent/predname)['predictions'];assert pred.shape==b.test_y.shape
 for d in range(3):
  mask=b.test_x[:,-1]==d
  for t in range(2):
   y=b.test_y[mask,t].astype(float);score=pred[mask,t].astype(float);_,idx=np.unique(score,return_inverse=True);pos=np.bincount(idx,weights=y);neg=np.bincount(idx,weights=1-y);auc=float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/(y.sum()*(len(y)-y.sum())));p=np.clip(score,1e-12,1-1e-12);ll=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)));a=table.loc[(e,d,t)];err=max(abs(auc-a.auc),abs(ll-a.logloss));error=max(error,err);assert int(a.n)==len(y) and int(a.n_pos)==int(y.sum());assert err<2e-12;rows.append(dict(epoch=e,domain=d,task=t,n=len(y),n_pos=int(y.sum()),auc=auc,logloss=ll,max_error=err,physical_model_sha=sha(target/mn),physical_test_prediction_sha=predsha))
f=O/'archive/formal/test_first_AdaTask23_SN100_to50_oracle.csv';assert not f.exists();pd.DataFrame(rows).to_csv(f,index=False)
proof=dict(status='PASS_ACTUAL_ALL50x6_CELLS_AND_MODEL_PREDICTION_HASHES',time=time.time(),independent_review=False,global_freeze_sha=sha(O/'test_readout_frozen.json'),parent=str(parent),target=str(target),epochs=50,cells=300,max_error=error,oracle_csv_sha=sha(f),current_parent_producer_sha=sha(parent/'receipt.json'),current_target_producer_sha=sha(target/'receipt.json'),target_test_receipt_sha=sha(target/'test_readout_receipt.json'),actual_bundle_metadata_equal_config=True,independent_formula='AUC tiedscore bin cumulative negative count; LL independent float64 clip1e-12; actualtargettest_y labels. No repeated model inference.',newly_evaluated_models=0,bulk_prefix_projection_authorized=True)
f=O/'test_first_prefix_actual_primary_oracle.json';assert not f.exists();f.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
