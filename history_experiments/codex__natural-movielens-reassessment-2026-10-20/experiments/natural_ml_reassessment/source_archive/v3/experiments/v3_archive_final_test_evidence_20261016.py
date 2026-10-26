from pathlib import Path
import json,hashlib,time,sys,subprocess,shutil
import pandas as pd
R=Path('/home/xzf/Code/phase0-3-20261016-v3-git');O=R/'research_plans/2026-10-16_v3/execution_20261016';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
assert load(O/'analysis/test_all_cohorts_primary_verification/receipt.json')['status']=='PASS'
inputs=load(O/'test_source_inputs.json')['records'];assert len(inputs)==156
mf=O/'final_all156_test_archive_reference_manifest.json';assert not mf.exists();mf.write_text(json.dumps(dict(non_executable_archive_reference_manifest=True,scope='Inputreferences forfrozenarchiver only; commandfields areparser-compatiblepathreferences, NOTclaims ofexecutedtraining/inference argv. Actualnativeargv/PIDs/exitcodes inGPUstatuses/queuelaunches.',tasks=[dict(name=Path(x['run']).name,command=['archive-reference-only','--run-dir',x['run']]) for x in inputs]),indent=2)+'\n')
cmd=[sys.executable,'experiments/v3_archive_20261016.py','--manifest',str(mf),'--out',str(O/'archive/formal')];t=time.time();proc=subprocess.Popen(cmd,cwd=R,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);output,_=proc.communicate();assert proc.returncode==0,output;result=json.loads(output);assert result['runs']==156
launch=O/'final_all156_test_archive_actual_launch.json';assert not launch.exists();launch.write_text(json.dumps(dict(pid=proc.pid,command=cmd,started=t,finished=time.time(),exit_code=proc.returncode,source_sha=sha(R/'experiments/v3_archive_20261016.py'),output=result),indent=2)+'\n')
index=load(Path(result['index']));copied=0
for entry in index['runs']:
 for x in entry['files']:
  assert sha(Path(entry['archive'])/x['name'])==x['sha256']
  assert sha(Path(entry['local_artifacts'])/x['name'])==x['sha256'];copied+=1
records=[];reuse=0;raw=[]
for x in inputs:
 d=Path(x['run']);T=x['epochs'];assert sha(d/'receipt.json')==x['producer_receipt_sha'];out=O/'archive/formal'/d.name/f'budget{T:03d}';rc=load(d/'test_readout_receipt.json');row=dict(run=str(d),physical_run=str(d.resolve()),epochs=T,producer_receipt_sha=sha(d/'receipt.json'),config_sha=sha(d/'config.json'),test_receipt_sha=sha(d/'test_readout_receipt.json'),metrics_sha=sha(d/'metrics_test.csv'),archive_dir=str(out.relative_to(O)))
 if (d/'test_prediction_reuse_provenance.json').exists():
  src=d/'test_prediction_reuse_provenance.json';dest=out/src.name
  if dest.exists():assert sha(dest)==sha(src)
  else:shutil.copyfile(src,dest)
  assert sha(dest)==rc['hashes'][src.name];assert not list(d.glob('test_epoch*.npz'));row['reuse_provenance_sha']=sha(src);reuse+=1
 else:
  assert rc['source_sha']=='655e284c7c32052c2aea951ccf6005befc2dc8561c302736e775a9007c0dfe7c';assert len(rc['hashes'])==T
  raw.append(dict(run=str(d),physical_run=str(d.resolve()),epochs=T,training_receipt_sha=x['producer_receipt_sha'],config_sha=x['config_sha'],test_receipt_sha=row['test_receipt_sha'],model_hashes=x['model_hashes'],test_prediction_hashes=rc['hashes']))
 records.append(row)
assert reuse==36 and len(raw)==120 and sum(x['epochs'] for x in raw)==13500
f=O/'final_test_local_raw_hash_index.json';assert not f.exists();f.write_text(json.dumps(dict(status='LOCAL_RAW_RETAINED_NO_BINARY_UPLOAD',time=time.time(),test_global_freeze_sha=sha(O/'test_readout_frozen.json'),native_sources=raw,all156_source_archives=records,raw_prediction_files=13500,exact_prefix_prediction_reuses=36,scope='Models/testNPZ retained onlocal artifact disk; perfilehashes/config/data/producer pins permit tracing. 36derivedsources referenceactualnativefiles, neverextraindependentmodels/seeds. No datasets/weights/cache/logs uploaded.'),indent=2)+'\n')
c=load(O/'orchestrator_checkpoint.json');states=[]
for name in c['active_gpu_status_files']:
 s=load(O/name);assert all(t['status'] in ['COMPLETE','FAILED'] for t in s['tasks'].values()),name;states.append(dict(status_file=name,actual_state=s))
for name in ['test_native_oracle_manifest.status.json','test_native_bulk_manifest.status.json','test_remaining_prefix_projection_CPU_manifest.status.json','test_all14_summary_serialization_retry_CPU_manifest.status.json']:
 s=load(O/name);assert s['status']=='COMPLETE' and all(x['status']=='COMPLETE' and x['exit_code']==0 for x in s['tasks'].values())
f2=O/'final_actual_GPU_queue_state_archive.json';assert not f2.exists();f2.write_text(json.dumps(dict(status='ALL_REGISTERED_GPU_QUEUES_TERMINAL',time=time.time(),states=states,historical_failed_attempts_not_deleted=True,required_frozen_test_all_actual_exit0=True, historical_failed_summary_attempt=load(O/'test_all14_summary_CPU_manifest.status.json'), serialization_repair=load(O/'test_summary_serialization_repair_primary_self_review.json')),indent=2)+'\n')
proof=dict(status='PASS_ALL156_TEST_ARCHIVES_AND_RAW_INDEX',time=time.time(),independent_review=False,archive_index=result['index'],archive_index_sha=sha(Path(result['index'])),critical_copied_files_physically_verified=copied,reuse_provenance_files=36,native_sources=120,native_test_prediction_files=13500,required_source_paths=156,required_model_path_records=sum(x['epochs'] for x in inputs),raw_test_cell_metric_rows=sum(x['epochs']*6 for x in inputs),raw_index_sha=sha(f),all_GPU_state_archive_sha=sha(f2),scope='Archivecompacttruthfulmetadata/metrics/receipts+36provenance; weights/predNPZ/datasets staylocalbySHA. Allnativepredictionhashes havealready beenphysicallychecked infrozenv2; thisarchive doesnotfalselyclaimanotherfullNPZrehash.')
p=O/'final_test_archive_primary_verification.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
