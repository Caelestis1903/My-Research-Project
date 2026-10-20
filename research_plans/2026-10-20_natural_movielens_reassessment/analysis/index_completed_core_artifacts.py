"""Export producer-recorded hashes and absolute paths without rehashing heavy tensors."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

ART=Path('/home_new/xzf_natural_movielens_reassessment')
OUT=Path(__file__).resolve().parent/'artifact_index'
OUT.mkdir(exist_ok=True)
manifest=json.loads((ART/'core_manifest.json').read_text())
runs=[(c['run_id'],Path(c['run_dir'])) for c in manifest['runs']]
v2=sorted((ART/'V2').glob('*/*/seed*/run_manifest.json'))
runs += [('D-V2__'+'__'.join(p.parent.relative_to(ART/'V2').parts),p.parent) for p in v2]
assert len(runs)==136
rows=[];sources=[];snapshots=[]
for run_id,run in runs:
    path=run/'run_manifest.json';raw=path.read_bytes();data=json.loads(raw)
    assert data['status']=='COMPLETE'
    source_sha=hashlib.sha256(raw).hexdigest()
    sources.append(dict(run_id=run_id,path=str(path),sha256=source_sha,producer_status=data['status']))
    snapshots.append(dict(run_id=run_id,source_path=str(path),source_sha256=source_sha,manifest=data))
    for name,digest in sorted(data['artifact_hashes'].items()):
        artifact=run/name
        rows.append(dict(run_id=run_id,path=str(artifact),producer_recorded_sha256=digest,
            exists_at_index=artifact.exists(),bytes_at_index=artifact.stat().st_size if artifact.exists() else None,
            hash_source_manifest=str(path),hash_source_manifest_sha256=source_sha,
            hash_scope='producer-recorded artifact bytes at completion; no heavy tensor rehash in this exporter'))
stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for name,payload in [('completed_core_artifacts.csv.gz',stream.getvalue().encode()),
    ('completed_core_manifests.jsonl.gz',(''.join(json.dumps(r,sort_keys=True)+'\n' for r in snapshots)).encode())]:
    (OUT/name).write_bytes(gzip.compress(payload,mtime=0))
(OUT/'completed_core_manifest_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
status=dict(complete_runs=len(runs),core=120,v2=16,artifacts=len(rows),
    missing_current_paths=sum(not r['exists_at_index'] for r in rows),
    heavy_tensors_rehashed=False,hash_sources_snapshotted=True,
    remaining_families_not_yet_indexed=['N17','AO04','N19','CM19','core offline mechanisms','V9 full trajectory readout'])
(OUT/'index_status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2))

# Independent, zero-update trajectory readouts are indexed separately from training.
readout=ART/'V9_trajectory_readout'
receipts=sorted(readout.glob('*/receipt.json'))
if len(receipts)==8:
    entries=[];records=[]
    for path in receipts:
        raw=path.read_bytes();j=json.loads(raw);assert j['status']=='COMPLETE' and j['optimizer_updates']==0
        assert j['epochs_evaluated']==50
        digest=hashlib.sha256(raw).hexdigest()
        records.append(dict(path=str(path),sha256=digest,receipt=j))
        for name,value in sorted(j['artifacts'].items()):
            artifact=path.parent/name
            entries.append(dict(run_id=j['run_id'],path=str(artifact),producer_recorded_sha256=value,
                exists_at_index=artifact.exists(),bytes_at_index=artifact.stat().st_size if artifact.exists() else None,
                source_receipt=str(path),source_receipt_sha256=digest,optimizer_updates=0))
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
    (OUT/'v9_full_readout_artifacts.csv.gz').write_bytes(gzip.compress(s.getvalue().encode(),mtime=0))
    (OUT/'v9_full_readout_receipts.json').write_text(json.dumps(records,indent=2)+'\n')
    status['v9_zero_update_readout_tasks']=8;status['v9_readout_artifacts']=len(entries)
    status['v9_missing_current_paths']=sum(not r['exists_at_index'] for r in entries)
    status['remaining_families_not_yet_indexed'].remove('V9 full trajectory readout')
    (OUT/'index_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print('V9 zero-update readout tasks',len(receipts),'artifacts',len(entries))
