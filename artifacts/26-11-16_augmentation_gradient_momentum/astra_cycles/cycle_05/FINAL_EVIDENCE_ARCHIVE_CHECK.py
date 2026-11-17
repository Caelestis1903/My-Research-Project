"""Independent C05 final archive readback; public numeric evidence only."""
from pathlib import Path, PurePosixPath
import collections
import hashlib
import json
import tarfile

W=Path('/home/xzf/Code/phase0-3_tvg_rm_20261116')
A=W/'artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05'
N=A/'attempt1'
R=Path('/tmp/phase0-3-c05-01a1e9a9/attempt1')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def load(path):return json.loads(path.read_text())

class PartStream:
    def __init__(self, paths):self.paths=iter(paths);self.current=None
    def read(self,n):
        assert n>=0
        blocks=[]
        while n:
            if self.current is None:
                try:self.current=next(self.paths).open('rb')
                except StopIteration:break
            b=self.current.read(n)
            if not b:self.current.close();self.current=None;continue
            blocks.append(b);n-=len(b)
        return b''.join(blocks)
    def close(self):
        if self.current is not None:self.current.close()


def main():
    m=load(N/'RAW_RUNS_MANIFEST.json');index=load(N/'RAW_RUNS_PART_INDEX.json')
    assert m['source_hash']=='f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d'
    assert index['parts']==m['parts'] and index['archive_sha256']==m['sha256'] and index['archive_bytes']==m['bytes']
    assert m['tensor_files']==0 and m['test_accessed'] is index['test_accessed'] is False
    combined=hashlib.sha256();total=0;paths=[]
    for i,p in enumerate(m['parts']):
        path=N/p['path'];assert path.parent==N and path.name.startswith('RAW_RUNS.tar.gz.part-')
        assert sha(path)==p['sha256'] and path.stat().st_size==p['bytes']
        assert 0<p['bytes']<=8388608 and (i==len(m['parts'])-1 or p['bytes']==8388608)
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1048576),b''):combined.update(b);total+=len(b)
        paths.append(path)
    assert total==m['bytes'] and combined.hexdigest()==m['sha256']
    expected=m['members'];assert len(expected)==m['member_count'] and len({x['member'] for x in expected})==len(expected)
    required=[R/'PARTIAL_TELEMETRY_BYTE_PROVENANCE.json',R/'REVIEW_SCHEMA_TEXT_PROVENANCE.json',R/'retired_prelaunch_01/SOURCE55_TEXT_PROVENANCE.json']
    required.extend(Path(x['complete_prefix_path']) for x in load(R/'PARTIAL_TELEMETRY_BYTE_PROVENANCE.json')['files'])
    members_by_source={x['source_path']:x for x in expected}
    for path in required:assert str(path) in members_by_source and members_by_source[str(path)]['sha256']==sha(path)
    source=load(R/'source_receipt.json')
    for x in source['files']:
        match=[r for r in expected if r['category']=='source' and r['member']=='attempt1/source_snapshot/'+x['path']]
        assert len(match)==1 and match[0]['sha256']==x['sha256']
    categories=collections.Counter(x['category'] for x in expected)
    assert categories['source']==55 and categories['control']==1530 and categories['design']==7
    assert sum(x['bytes'] for x in expected)==m['uncompressed_bytes']
    stream=PartStream(paths);count=0
    try:
        with tarfile.open(fileobj=stream,mode='r|gz') as archive:
            for header in archive:
                row=expected[count];p=PurePosixPath(header.name)
                assert not p.is_absolute() and '..' not in p.parts and header.isfile()
                assert header.name==row['member'] and header.size==row['bytes']
                h=hashlib.sha256()
                with archive.extractfile(header) as f:
                    for b in iter(lambda:f.read(1048576),b''):h.update(b)
                assert h.hexdigest()==row['sha256'],header.name
                count+=1
    finally:stream.close()
    assert count==m['member_count']
    result=dict(status='PASS',reviewer_id='01a1eab1-ed17-7f51-92bc-4dc70500384c',
        archive_sha256=m['sha256'],archive_bytes=m['bytes'],parts=len(paths),members_readback=count,
        all_part_hashes_and_combined_archive_hash_verified=True,all_member_types_lengths_and_hashes_verified=True,
        categories=dict(categories),manifest_sha256=sha(N/'RAW_RUNS_MANIFEST.json'),
        index_sha256=sha(N/'RAW_RUNS_PART_INDEX.json'),
        incomplete_compressed_bytes_recoverable_from_exact_hex_wrapper=True,
        incomplete_prefixes_and_schema_wrapper_and_retired55_in_archive=True,
        source_hash=m['source_hash'],design_bundle_sha256=m['design_bundle_sha256'],test_accessed=False)
    (A/'FINAL_EVIDENCE_ARCHIVE_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
