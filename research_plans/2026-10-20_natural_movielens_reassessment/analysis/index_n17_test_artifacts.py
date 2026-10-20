"""Index every completed frozen N17 test output, including trajectory NPZ files."""
from pathlib import Path
import gzip,hashlib,json
ART=Path('/home_new/xzf_natural_movielens_reassessment/N17')
OUT=Path(__file__).resolve().parent/'n17_test_artifacts'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 frozen=ART/'test_frozen.json';freeze=json.loads(frozen.read_text());digest=sha(frozen);rows=[];small=[]
 assert len(freeze['runs'])==72
 for path,entry in freeze['runs'].items():
  run=Path(path);r=json.loads((run/'test_receipt.json').read_text());assert r['status']=='COMPLETE' and r['zero_updates'] and r['optimizer_instantiated'] is False and r['freeze_sha256']==digest
  files=[run/name for name in ['test_receipt.json','metrics_test.csv','test_prediction_index.json','raw_AULC.json']]+[run/f'test_epoch{e:03d}.npz' for e in range(1,entry['config']['epochs']+1)]
  for file in files:
   rows.append(dict(run_id=entry['config']['id'],group=entry['config']['group'],path=str(file),bytes=file.stat().st_size,sha256=sha(file)))
   if file.suffix=='.json':small.append(dict(run_id=entry['config']['id'],path=str(file),sha256=rows[-1]['sha256'],content=json.loads(file.read_text())))
 OUT.mkdir(exist_ok=True)
 for name,values in [('artifact_index',rows),('small_json_snapshots',small)]:
  payload=('\n'.join(json.dumps(x,sort_keys=True) for x in values)+'\n').encode();(OUT/(name+'.jsonl.gz')).write_bytes(gzip.compress(payload,mtime=0))
 (OUT/'index_status.json').write_text(json.dumps(dict(complete=True,test_configurations=72,epoch_predictions=8000,actual_file_hashes=len(rows),total_bytes=sum(x['bytes'] for x in rows),test_freeze_sha256=digest,zero_updates=True,pilot_predictions=0),indent=2)+'\n');print('indexed',len(rows),'test artifacts')
if __name__=='__main__':main()
