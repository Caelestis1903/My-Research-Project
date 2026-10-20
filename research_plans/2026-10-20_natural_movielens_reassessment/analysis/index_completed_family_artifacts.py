"""Hash completed AO04 or transfer scopes, preserving all server artifact paths."""
from pathlib import Path
import argparse,hashlib,json,gzip
ART=Path('/home_new/xzf_natural_movielens_reassessment')
ANA=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main(study,scope):
 if study=='AO04':
  assert scope is None
  root=ART/'AO04';configs=json.loads((root/'manifest.json').read_text())['configurations']
  r=json.loads((root/'test_receipt.json').read_text());assert r['status']=='COMPLETE_FROZEN_TEST' and r['optimization_steps']==0 and r['freeze_sha256']==sha(root/'TEST_FREEZE.json')
 else:
  assert scope
  root=ART/study/scope;configs=[r for r in json.loads((ART/'transfer_manifest.json').read_text())['runs'] if r['study']==study and r['scope']==scope];assert configs
  digest=sha(root/'TEST_FREEZE.json')
  for cfg in configs:
   r=json.loads((Path(cfg['run_dir'])/'reassessment_test_receipt.json').read_text());assert r['status']=='COMPLETE_FROZEN_TEST' and r['optimization_steps']==0 and r['freeze_sha256']==digest
 out=ANA/'family_artifacts'/study/(scope or 'all');out.mkdir(parents=True,exist_ok=True);rows=[];small=[]
 for p in sorted(root.rglob('*')):
  if not p.is_file():continue
  digest=sha(p);size=p.stat().st_size
  rows.append(dict(study=study,scope=scope or 'all',path=str(p),bytes=size,sha256=digest))
  if p.suffix=='.json' and size<=100000:
   small.append(dict(path=str(p),sha256=digest,content=json.loads(p.read_text())))
 for name,values in [('artifact_index',rows),('small_json_snapshots',small)]:
  raw=('\n'.join(json.dumps(x,sort_keys=True) for x in values)+'\n').encode();(out/(name+'.jsonl.gz')).write_bytes(gzip.compress(raw,mtime=0))
 status=dict(complete=True,study=study,scope=scope,training_configurations=len(configs),actual_file_hashes=len(rows),total_bytes=sum(r['bytes'] for r in rows),small_json_snapshots=len(small),server_root=str(root),scope_note='successful frozen scope; failed attempts remain separately in ledger/runner archives')
 (out/'index_status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--study',choices=['AO04','N19','CM19'],required=True);p.add_argument('--scope');a=p.parse_args();main(a.study,a.scope)
