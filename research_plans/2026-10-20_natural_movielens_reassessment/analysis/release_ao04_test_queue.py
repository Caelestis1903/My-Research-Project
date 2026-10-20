"""Publish the designed AO04 single-command readout after primary scope freeze."""
from pathlib import Path
import hashlib,json
ART=Path('/home_new/xzf_natural_movielens_reassessment')
ANA=Path(__file__).resolve().parent
PYTHON='/home/xzf/miniconda3/envs/phase0-3-env/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def release():
 root=ART/'AO04';freeze=root/'TEST_FREEZE.json';f=json.loads(freeze.read_text())
 assert f['global_test_unlock'] is True
 closure=ANA/'ao04_freeze/closure.json';c=json.loads(closure.read_text())
 assert c['training_complete'] and not c['test_read'] and c['development_disposition']=='CONFIGURATIONS_AND_SELECTION_FROZEN'
 assert sha(closure)==f['closure_sha256']
 configs=json.loads((root/'manifest.json').read_text())['configurations'];assert len(configs)==12
 assert set(f['runs'])=={c['id'] for c in configs}
 inputs=[root/'manifest.json',freeze,closure];artifacts=[]
 for cfg in configs:
  run=Path(cfg['run_dir']);entry=f['runs'][cfg['id']]
  assert sha(run/'config.json')==entry['config_sha256'] and sha(run/'selection.json')==entry['selection_sha256']
  receipt=json.loads((run/'local_fusion_diagnostics/receipt.json').read_text())
  assert receipt['status']=='COMPLETE_RAW_DIAGNOSTIC' and receipt['training_updates']==0 and not receipt['test_read']
  inputs.extend([run/'receipt.json',run/'selection.json',run/'local_fusion_diagnostics/receipt.json'])
  artifacts.extend([run/'metrics_test.csv',run/'predictions_test.npz'])
 artifacts.extend([root/'_final_test_evaluated'/name for name in ['metrics_test.csv','predictions_test.npz','test_evaluation_manifest.json']])
 digest=sha(freeze);rid='AO04_full_freeze_'+digest[:12]
 task=dict(run_id='AO04_readout_all12_'+digest[:12],study='AO04',scope='movielens_natural',config_path=str(root/'manifest.json'),run_dir=str(root),
  command=[PYTHON,'-m','experiments.natural_ml_extensions.ao04','test','--device','cuda:0'],receipt=str(root/'test_receipt.json'),
  receipt_checks=dict(status='COMPLETE_FROZEN_TEST',optimization_steps=0,freeze_sha256=digest),required_files=[str(x) for x in inputs],
  artifacts=[str(x) for x in artifacts],profile_group='TEST_AO04')
 payload=dict(schema='natural_movielens_test_release_v1',family='AO04',release_id=rid,status='RELEASED',test_authorized=True,
  parent_freeze_path=str(freeze),parent_freeze_sha256=digest,tasks=[task])
 content=(json.dumps(payload,indent=2)+'\n').encode();path=ART/'runner/released_test_queues/AO04'/(rid+'.json');path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert path.read_bytes()==content
 else:
  temporary=path.with_suffix('.json.tmp');temporary.write_bytes(content);temporary.replace(path)
 archive=ANA/'ao04_freeze'/path.name
 if archive.exists():assert archive.read_bytes()==content
 else:archive.write_bytes(content)
 info=dict(release_path=str(path),sha256=sha(path),formal_configurations=12,readout_tasks=1,parent_freeze_sha256=digest,zero_update_evaluation_only=True)
 (ANA/'ao04_freeze/release_receipt.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':release()
