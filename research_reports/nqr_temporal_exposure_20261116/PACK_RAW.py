"""Archive completed numeric records and explicit reused controls, never tensors."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
N=ROOT/'artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/attempt1'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):
            h.update(data)
    return h.hexdigest()


def main():
    tick=time.perf_counter()
    read=lambda p:json.loads(p.read_text())
    acceptance=read(N/'PARENT_STORAGE_ACCEPTANCE.json')
    assert acceptance['status']=='ACCEPTED'
    receipt=read(N/'tables/EXPORT_RECEIPT.json')
    assert receipt['new_trajectories']==20 and receipt['useful_batches']==34500 and receipt['test_accessed'] is False
    archive=N/'RAW_RUNS.tar.gz'
    assert not archive.exists(), 'Reuse completed archive; do not silently rebuild publication bytes'
    controls=[]; external=[]
    for entry in read(N/'reused_reference_index.json')['entries']:
        prefix=f'reused_controls/{entry["key"]}'
        files=[dict(path=entry['result_path'],sha256=entry['result_sha256'],archive_path=prefix+'/result.json')]
        files += [dict(path=r['path'],sha256=r['sha256'],archive_path=f'{prefix}/epoch_{i:02d}.json')
                  for i,r in enumerate(entry['epochs'],1)]
        assert len(files)==51
        controls.append(dict(key=entry['key'],dataset=entry['dataset'],arm=entry['arm'],seed=entry['seed'],files=files))
        external.extend((Path(r['path']),r['archive_path'],r['sha256'],'reused_numeric_control') for r in files)
    assert len(controls)==20
    mapping=dict(type='20 exact reused numeric CLEAN/constant-NQR trajectories; no live old IPC',controls=controls)
    (N/'CONTROL_ARCHIVE_MAP.json').write_text(json.dumps(mapping,indent=2)+'\n')
    with (N/'tables/DIAGNOSTICS.csv').open(newline='') as f:
        historical=sorted({r['source_path'] for r in csv.DictReader(f) if r['available']=='True' and r['reused']=='True'})
    for i,path in enumerate(historical):
        p=Path(path)
        assert p.is_file() and not p.is_relative_to(N)
        external.append((p,f'reused_diagnostic_batches/{i:03d}_{p.name}',sha(p),'reused_numeric_diagnostic'))
    members=[]; sources=[]
    for p in sorted(N.rglob('*')):
        if not p.is_file() or p.name=='guardian.json' or p.name.startswith('RAW_RUNS'):
            continue
        sources.append((p,str(p.relative_to(N)),None,'new_cycle_numeric'))
    sources.extend(external)
    assert len({arc for _,arc,_,_ in sources})==len(sources)
    for p,arc,expected,kind in sources:
        assert not p.is_symlink() and p.suffix.lower() not in ('.pt','.pth','.ckpt','.pkl','.safetensors','.npy','.npz')
        assert 'guardian.json' not in arc
        actual=sha(p)
        assert expected is None or actual==expected
        members.append(dict(path=str(p),archive_path=arc,sha256=actual,bytes=p.stat().st_size,kind=kind))
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        for p,arc,_,_ in sources:
            tar.add(p,arcname=arc,recursive=False)
    parts=[]; combined=hashlib.sha256()
    with archive.open('rb') as f:
        for i,data in enumerate(iter(lambda:f.read(8*1024*1024),b'')):
            part=archive.with_name(archive.name+f'.part-{i:02d}')
            assert not part.exists()
            part.write_bytes(data);combined.update(data)
            parts.append(dict(path=part.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    assert sum(p['bytes'] for p in parts)==archive.stat().st_size
    archive_sha=combined.hexdigest()
    assert archive_sha==sha(archive)
    with tarfile.open(archive,'r:gz') as tar:
        headers=tar.getmembers()
        assert [h.name for h in headers]==[r['archive_path'] for r in members]
        assert [h.size for h in headers]==[r['bytes'] for r in members]
    manifest=dict(archive=archive.name,sha256=archive_sha,bytes=archive.stat().st_size,member_count=len(members),
        uncompressed_bytes=sum(r['bytes'] for r in members),parts=parts,members=members,
        source_hash=receipt['source_hash'],design_bundle_sha256=receipt['design_bundle_sha256'],test_accessed=False,
        tensor_files=False,credential_files_excluded=['guardian.json'],seconds=time.perf_counter()-tick)
    (N/'RAW_RUNS_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('members','parts')}|dict(parts=len(parts),reused_controls=len(controls),historical_diagnostic_logs=len(historical))))


if __name__=='__main__':
    main()
