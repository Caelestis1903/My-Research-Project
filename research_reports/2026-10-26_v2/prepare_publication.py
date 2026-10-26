"""Prepare bounded Git delivery; large CSV tables are lossless gzip data.

No network or branch update is performed here. Raw CSV/large NPZ remain on the
server. Publication manifests are metadata, never model/checkpoint archives.
"""
from pathlib import Path
import gzip
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parent
CODE=ROOT/'experiments/mdmt_rcf_20261026'
FINAL_REPORTS=['FINAL_REPORT.md','IMPLEMENTATION_REVIEW.md','DATA_AND_IDENTITY.md',
               'SOURCE_INCREMENT_REPORT.md','FUNCTION_FORMATION_REPORT.md',
               'SCALE_REPORT.md','COST_REPORT.md','OPEN_QUESTIONS.md']


def file_identity(path):
    raw=path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)),bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        git_blob_sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())


def main():
    for name in FINAL_REPORTS+['ARTIFACT_INDEX.json','EXPORT_ACCEPTANCE.json','CLEANUP_RECEIPT.json']:
        if not (REPORT/name).is_file():raise FileNotFoundError(name)
    for name in ['EXPORT_ACCEPTANCE.json','CLEANUP_RECEIPT.json']:
        if json.loads((REPORT/name).read_text())['status'] not in ('PASS','COMPLETE'):
            raise PermissionError('Actual final acceptance required')
    compressed=[];omitted=set()
    for path in sorted((REPORT/'tables').glob('*.csv')):
        if path.stat().st_size<=256*1024:continue
        raw=path.read_bytes();target=path.with_suffix('.csv.gz')
        target.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
        if gzip.decompress(target.read_bytes())!=raw:raise RuntimeError('Lossless CSV verification failed')
        omitted.add(path)
        compressed.append(dict(raw_csv=file_identity(path),published_gzip=file_identity(target)))
        rel=str(path.relative_to(REPORT));new=str(target.relative_to(REPORT))
        for name in FINAL_REPORTS:
            report=REPORT/name;text=report.read_text()
            text=re.sub(re.escape(rel)+r'(?![.\w])',lambda _:new,text)
            report.write_text(text)
    code=[file_identity(p) for p in sorted(CODE.rglob('*'))
          if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    # Campaign source_identity hashes repository-relative keys from paths.code.
    source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(CODE.rglob('*.py'))
                  if '__pycache__' not in str(p)}
    for name in ('SOURCE_MANIFEST.json','SCALE_SOURCE_MANIFEST.json'):
        p=CODE/'vendor_snapshot'/name
        source_files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    source_hash=hashlib.sha256(json.dumps(source_files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    state=json.loads((REPORT/'ORCHESTRATOR_STATE.json').read_text())
    if source_hash!=state['observer_source_hash']:raise RuntimeError('Published source differs from final reviewed observer identity')
    code_manifest=dict(status='PREPARED_FINAL_SOURCE_FOR_PUBLICATION',
        current_observer_source_hash=source_hash,
        source_version_note='Per-run loaded training hashes and FROZEN remain separate from post-training delivery observer source',
        files=code)
    (REPORT/'PUBLICATION_CODE_MANIFEST.json').write_text(json.dumps(code_manifest,indent=2)+'\n')
    files=[]
    for root in (CODE,REPORT):
        for p in sorted(root.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or '.matplotlib' in p.parts:continue
            if p in omitted or p.name.startswith('.publication_') or p.name=='PUBLICATION_FILE_MANIFEST.json':continue
            if p.suffix=='.pyc':continue
            if p.suffix not in {'.py','.md','.json','.yaml','.yml','.csv','.gz','.png','.pdf'}:
                raise RuntimeError('Unexpected publication file type '+str(p))
            files.append(file_identity(p))
    manifest=dict(status='PREPARED_LOCAL_ONLY',remote_branch='research-plan/2026-10-26-v2-mechanisms',
        files=files,compressed_csv=compressed,
        self_excluded_from_its_own_hash=True,
        gzip_note='Raw CSV remains on server. Published .csv.gz is verified lossless (gzip mtime0); decompress before csv-module scripts. No large NPZ, mmap, model or checkpoint is published.')
    dest=REPORT/'PUBLICATION_FILE_MANIFEST.json';dest.write_text(json.dumps(manifest,indent=2)+'\n')
    plan=files+[file_identity(dest)]
    (REPORT/'.publication_upload_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(status='PREPARED_LOCAL_ONLY',files=len(plan),bytes=sum(r['bytes'] for r in plan),
                          compressed_tables=len(compressed),largest_file_bytes=max(r['bytes'] for r in plan))))


if __name__=='__main__':main()
