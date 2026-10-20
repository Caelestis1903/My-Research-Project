"""Primary pre-test scope inventory: preserve every frozen run and diagnostic."""
from pathlib import Path
import argparse,hashlib,json
import pandas as pd
ART=Path('/home_new/xzf_natural_movielens_reassessment')
ANA=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(study,scope):
 manifest=ART/'transfer_manifest.json';configs=json.loads(manifest.read_text())['runs'];byconfig={c['config_path']:c for c in configs}
 configs=[c for c in configs if c['study']==study and c['scope']==scope];assert configs
 folder=ANA/'transfer_freeze'/study/scope;folder.mkdir(parents=True,exist_ok=True)
 rows=[]
 for c in configs:
  run=Path(c['run_dir']);rp=run/'receipt.json';r=json.loads(rp.read_text());ep=run/'reassessment_receipt.json';e=json.loads(ep.read_text())
  assert r['status']==e['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
  assert e['global_run_id']==r['global_run_id']==c['run_id'] and not e['test_evaluated']
  assert e['receipt_sha256']==sha(rp) and e['checkpoint_index_sha256']==sha(run/'checkpoint_index.json')
  assert r['actual_main_examples']>0 and sum(r['actual_main_domain_exposures'])==r['actual_main_examples']
  rows.append(dict(run_id=c['run_id'],historical_run_id=c['historical_run_id'],run_dir=str(run),config_path=c['config_path'],
   seed=c['seed'],population=c['population'],budget=c['budget'],epochs=r['epochs'],steps=r['steps'],actual_main_examples=r['actual_main_examples'],
   domain_exposures=json.dumps(r['actual_main_domain_exposures']),independent_training_support=r['independent_training_support'],
   receipt_path=str(rp),receipt_sha256=sha(rp),reassessment_receipt_sha256=sha(ep),selection_sha256=sha(run/'selection_frozen.json')))
 dm=ART/'transfer_diagnostic_manifest.json';tasks=[t for t in json.loads(dm.read_text())['tasks'] if t['study']==study and t['scope']==scope];assert tasks
 diagnostics=[]
 for task in tasks:
  c=byconfig[task['config_path']];p=ART/study/scope/'diagnostic_receipts'/(c['historical_run_id']+'__'+task['kind']+'.json');r=json.loads(p.read_text())
  assert r['status']=='COMPLETE_RAW_DIAGNOSTIC' and r['training_updates']==0 and not r['test_read'] and r['global_run_id']==c['run_id']
  diagnostics.append(dict(diagnostic_id=task['id'],kind=task['kind'],path=str(p),sha256=sha(p)))
 inventory=folder/'scope_training_inventory.csv';pd.DataFrame(rows).to_csv(inventory,index=False)
 index=folder/'diagnostic_receipt_index.csv';pd.DataFrame(diagnostics).to_csv(index,index=False)
 closure=dict(scope=scope,study=study,training_complete=True,all_required_diagnostics_complete=True,
  development_disposition='CONFIGURATIONS_AND_SELECTION_FROZEN',test_read=False,
  run_ids=[c['run_id'] for c in configs],training_configurations=len(configs),required_diagnostics=len(tasks),global_study_complete=False,
  configuration_policy='all pre-registered configurations and earliest validation selection retained; no result-driven deletion, retuning or expansion',
  proofs=[dict(path=str(p.resolve()),sha256=sha(p)) for p in [manifest,dm,inventory,index]])
 path=folder/'closure.json';payload=(json.dumps(closure,indent=2)+'\n').encode()
 if path.exists():assert path.read_bytes()==payload,'scope closure already frozen; do not overwrite'
 else:path.write_bytes(payload)
 print(json.dumps(dict(closure=str(path.resolve()),sha256=sha(path),configurations=len(configs),diagnostics=len(tasks)),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--study',choices=['N19','CM19'],required=True);p.add_argument('--scope',required=True);a=p.parse_args();prepare(a.study,a.scope)
