"""Index complete N17 training models; preserve pilot and shared-prefix provenance."""
from pathlib import Path
import csv,gzip,hashlib,io,json
ART=Path('/home_new/xzf_natural_movielens_reassessment/N17')
OUT=Path(__file__).resolve().parent/'artifact_index';OUT.mkdir(exist_ok=True)
def sha(raw):return hashlib.sha256(raw).hexdigest()
configs={}
for name in ['base_manifest.json','post_pilot_manifest.json','H200_manifest.json']:
    for c in json.loads((ART/name).read_text())['configurations']:
        if c.get('run_dir'):configs[c['id']]=c
assert len(configs)==88
rows=[];records=[];bindings=[]
for c in configs.values():
    run=Path(c['run_dir']);receipt=json.loads((run/'receipt.json').read_text())
    assert receipt['status']=='COMPLETE_TRAINING_TEST_DEFERRED' and receipt['epochs']==c['epochs']
    raw=(run/'all_model_index.json').read_bytes();digest=sha(raw);models=json.loads(raw)
    assert digest==receipt['all_model_index_sha256'] and len(models)==c['epochs']
    for name in ['config.json','receipt.json','all_model_index.json','selection_frozen.json']:
        source=run/name;payload=source.read_bytes()
        records.append(dict(run_id=c['id'],group=c['group'],pilot=c['group']=='pilot',path=str(source),sha256=sha(payload),record=json.loads(payload)))
    bindings.append(dict(run_id=c['id'],group=c['group'],pilot=c['group']=='pilot',seed=c['seed'],arm=c['arm'],lr=c['lr'],population=c['population'],run_dir=str(run),
        epochs=receipt['epochs'],main_steps=receipt['steps'],examples=receipt['examples'],selected_epoch=receipt['selected_epoch'],
        receipt_sha256=sha((run/'receipt.json').read_bytes()),all_model_index_sha256=digest,resume_from=c.get('resume_from',''),test_read=False))
    for model in models:
        path=Path(model['path'])
        rows.append(dict(run_id=c['id'],group=c['group'],pilot=c['group']=='pilot',epoch=model['epoch'],
            shared_H100_prefix=c['group']=='H200' and model['epoch']<=100,path=str(path),producer_recorded_sha256=model['sha256'],
            exists_at_index=path.exists(),bytes_at_index=path.stat().st_size if path.exists() else None,
            hash_source=str(run/'all_model_index.json'),hash_source_sha256=digest,heavy_tensor_rehashed=False))
def csv_gz(name,values):
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    (OUT/name).write_bytes(gzip.compress(s.getvalue().encode(),mtime=0))
csv_gz('n17_training_model_index.csv.gz',rows)
csv_gz('n17_training_run_bindings.csv.gz',bindings)
(OUT/'n17_training_small_records.jsonl.gz').write_bytes(gzip.compress((''.join(json.dumps(x,sort_keys=True)+'\n' for x in records)).encode(),mtime=0))
status=dict(training_configs=88,pilot_configs=16,formal_H100=64,H200_continuations=8,model_references=len(rows),
    shared_prefix_references=sum(x['shared_H100_prefix'] for x in rows),unique_model_paths=len(set(x['path'] for x in rows)),
    missing_model_paths=sum(not x['exists_at_index'] for x in rows),heavy_models_rehashed=False,
    hash_provenance='producer all_model_index bound to completed training receipt; test freeze will independently hash model bytes',
    test_read=False,scope='training model captures and small metadata only; not yet test or diagnostic index')
(OUT/'n17_training_index_status.json').write_text(json.dumps(status,indent=2)+'\n');print(status)
