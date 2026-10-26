from pathlib import Path
import json,hashlib,time,sys
import pandas as pd,numpy as np
R=Path('/home/xzf/Code/phase0-3-20261016-v3-git');O=R/'research_plans/2026-10-16_v3/execution_20261016';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text());stage=sys.argv[1];assert stage in ['prefix','summaries'];freeze=load(O/'test_readout_frozen.json');assert freeze['status']=='FROZEN';rec=load(O/'test_remaining_CPU_pipeline_prepared_review.json')
assert sha(O/'test_readout_frozen.json')==rec['frozen_global_sha']
if stage=='prefix':
 for label in ['oracle','bulk']:
  q=load(O/f'test_native_{label}_manifest.status.json');assert q['status']=='COMPLETE' and all(t['status']=='COMPLETE' and t['exit_code']==0 for t in q['tasks'].values())
 inputs=load(O/'test_native_parent_inputs.json')['records'];stem='test_remaining_prefix_projection';expected=120
else:
 q=load(O/'test_remaining_prefix_projection_CPU_manifest.status.json');assert q['status']=='COMPLETE' and all(t['status']=='COMPLETE' and t['exit_code']==0 for t in q['tasks'].values());assert load(O/'test_first_prefix_actual_primary_oracle.json')['bulk_prefix_projection_authorized']
 inputs=load(O/'test_source_inputs.json')['records'];stem='test_all14_summary';expected=156
records=[];ends=0
for x in inputs:
 d=Path(x['run']);assert sha(d/'receipt.json')==x['producer_receipt_sha'] and sha(d/'config.json')==x['config_sha'];p=load(d/'receipt.json');q=load(d/'test_readout_receipt.json');T=p['epochs'];assert T==x['epochs'];assert q['status']=='COMPLETE' and q['test_global_freeze_sha']==sha(O/'test_readout_frozen.json') and q['epochs']==list(range(1,T+1))
 a=pd.read_csv(d/'metrics_test.csv');assert len(a)==T*6 and not a.duplicated(['epoch','domain','task']).any() and np.isfinite(a[['auc','logloss']]).all().all();assert set(a.epoch)==set(range(1,T+1))
 if q['source_sha']==freeze['test_readout_source_sha']:
  assert len(q['hashes'])==T
  if stage=='prefix':
   for e in sorted(set([1,T])):
    f=f'test_epoch{e:03d}.npz';assert sha(d/f)==q['hashes'][f];ends+=1
 else:
  assert stage=='summaries' and q['source_sha']==freeze['test_prefix_reuse_source_sha']
  for f,h in q['hashes'].items():assert sha(d/f)==h
 records.append(dict(run=str(d),actual_epochs=T,producer_receipt_sha=sha(d/'receipt.json'),test_receipt_sha=sha(d/'test_readout_receipt.json'),test_metrics_sha=sha(d/'metrics_test.csv'),test_metric_rows=len(a)))
assert len(records)==expected
mf=O/(stem+'_CPU_manifest.json');assert sha(mf)==rec['manifests'][mf.name];review=O/(stem+'_activation_review.json');assert not review.exists();review.write_text(json.dumps(dict(status='PASS_ACTIVATE_FROZEN_TEST_DERIVED_OUTPUTS',time=time.time(),independent_review=False,manifest_sha=sha(mf),test_global_freeze_sha=sha(O/'test_readout_frozen.json'),sources=rec['sources'],actual_input_records=records,actual_physical_endpoint_test_prediction_hashes=ends,first_oracle_sha=sha(O/'test_first_prefix_actual_primary_oracle.json'),resource='max2IOprefixworkers' if stage=='prefix' else 'max8CPUsummaryworkers afterallGPUtestfinished; noCPUbootstrapGPUworkers.',scope='Inputcritical/endpointprecheckonly; allnativepredictionhashes+fullnumericreconstruction muststillpassfrozenv2 aftersummaries. No newtest-basedcondition.'),indent=2)+'\n');print('ACTIVATED',stem,len(records),'actualinputs',sum(x['test_metric_rows'] for x in records),'rawrows')
