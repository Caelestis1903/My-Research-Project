"""Publish one primary-frozen transfer scope; no training or metric interpretation."""
from pathlib import Path
import argparse,hashlib,json
ART=Path('/home_new/xzf_natural_movielens_reassessment')
ANA=Path(__file__).resolve().parent
PYTHON='/home/xzf/miniconda3/envs/phase0-3-env/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def release(study,scope):
 root=ART/study/scope;freeze=root/'TEST_FREEZE.json';f=json.loads(freeze.read_text())
 folder=ANA/'transfer_freeze'/study/scope;closure=folder/'closure.json';c=json.loads(closure.read_text())
 assert c['training_complete'] and not c['test_read'] and c['development_disposition']=='CONFIGURATIONS_AND_SELECTION_FROZEN'
 assert c['all_required_diagnostics_complete'] and sha(closure)==f['closure_sha256']
 configs=[r for r in json.loads((ART/'transfer_manifest.json').read_text())['runs'] if r['study']==study and r['scope']==scope]
 assert configs and set(c['run_ids'])=={r['run_id'] for r in configs}
 assert set(f['runs'])=={r['historical_run_id'] for r in configs}
 digest=sha(freeze);tasks=[]
 for cfg in configs:
  run=Path(cfg['run_dir']);entry=f['runs'][cfg['historical_run_id']];native=json.loads((run/'config.json').read_text());states=Path(native['state_directory'])
  assert sha(run/'config.json')==entry['config_sha256'] and sha(run/'selection_frozen.json')==entry['selection_sha256']
  assert entry['global_run_id']==cfg['run_id']
  artifacts=[run/'test_receipt.json']
  if study=='N19':artifacts.extend([run/name for name in ['metrics_test.json','diagnostics_test.json','test_selected.npz','test_final.npz']])
  else:artifacts.extend([run/'metrics_test.csv',run/'inference_selected.json',run/'inference_final.json',states/'test_selected.npz',states/'test_final.npz'])
  assert all(path.is_relative_to(root) for path in artifacts)
  # Scope is the common output envelope for sibling runs/ and states/; only
  # this task's explicit outputs may be archived during a retry.
  tasks.append(dict(run_id='transfer_readout_'+cfg['run_id'],study=study,scope=scope,config_path=cfg['config_path'],run_dir=str(root),
   command=[PYTHON,'-m','experiments.natural_ml_transfer.evaluate','test','--config',cfg['config_path'],'--device','cuda:0'],
   receipt=str(run/'reassessment_test_receipt.json'),receipt_checks=dict(status='COMPLETE_FROZEN_TEST',optimization_steps=0,freeze_sha256=digest),
   required_files=[cfg['config_path'],str(freeze),str(closure),str(run/'selection_frozen.json'),str(states/'selected.pt'),str(states/'last.pt')],
   artifacts=[str(x) for x in artifacts],profile_group='TEST_'+study))
 rid=study+'_'+scope+'_freeze_'+digest[:12]
 payload=dict(schema='natural_movielens_test_release_v1',family='transfer',release_id=rid,study=study,scope=scope,
  status='RELEASED',test_authorized=True,parent_freeze_path=str(freeze),parent_freeze_sha256=digest,tasks=tasks)
 content=(json.dumps(payload,indent=2)+'\n').encode();path=ART/'runner/released_test_queues/transfer'/(rid+'.json');path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert path.read_bytes()==content
 else:
  tmp=path.with_suffix('.json.tmp');tmp.write_bytes(content);tmp.replace(path)
 archive=folder/path.name
 if archive.exists():assert archive.read_bytes()==content
 else:archive.write_bytes(content)
 info=dict(release_path=str(path),sha256=sha(path),formal_configurations=len(tasks),parent_freeze_sha256=digest,zero_update_evaluation_only=True)
 (folder/'release_receipt.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--study',choices=['N19','CM19'],required=True);p.add_argument('--scope',required=True);a=p.parse_args();release(a.study,a.scope)
