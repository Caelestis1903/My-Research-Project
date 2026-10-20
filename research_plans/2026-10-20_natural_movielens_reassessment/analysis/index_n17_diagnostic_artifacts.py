"""Bind completed N17 diagnostic artifacts to server paths and actual SHA bytes."""
from pathlib import Path
import hashlib,json,gzip
ART=Path('/home_new/xzf_natural_movielens_reassessment/N17')
OUT=Path(__file__).resolve().parent/'n17_diagnostic_artifacts'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 base=json.loads((ART/'diagnostic_manifest.json').read_text())['configurations'];extra=json.loads((ART/'extra_precision_manifest.json').read_text())['configurations']
 configs=base+extra;assert len(base)==48 and len(extra)==14
 OUT.mkdir(exist_ok=True);rows=[];small=[]
 for cfg in configs:
  folder=Path(cfg['out']);r=json.loads((folder/'extension_receipt.json').read_text());assert r['status']=='COMPLETE' and r['test_read'] is False
  for p in sorted(folder.rglob('*')):
   if not p.is_file():continue
   rows.append(dict(diagnostic_id=cfg['id'],kind=cfg['kind'],path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
   if p.suffix=='.json':small.append(dict(diagnostic_id=cfg['id'],path=str(p),sha256=rows[-1]['sha256'],content=json.loads(p.read_text())))
 for name,records in [('artifact_index',rows),('small_json_snapshots',small)]:
  payload=('\n'.join(json.dumps(x,sort_keys=True) for x in records)+'\n').encode();p=OUT/(name+'.jsonl.gz');p.write_bytes(gzip.compress(payload,mtime=0))
 (OUT/'index_status.json').write_text(json.dumps(dict(complete=True,base_diagnostics=48,extra_precision_chunks=14,actual_file_hashes=len(rows),total_bytes=sum(r['bytes'] for r in rows),small_json_snapshots=len(small),test_read=False,checkpoint_bindings='actual checkpoint SHA checked by mechanism analysis/source_index.csv'),indent=2)+'\n')
 print('indexed',len(rows),'actual diagnostic artifacts')
if __name__=='__main__':main()
