"""Prepare an isolated Git index for the reviewed Cycle04 numeric publication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
A=ROOT/'artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04'
N=A/'attempt2'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def git(*args,input=None,env=None):
    return subprocess.check_output(['git',*args],cwd=ROOT,input=input,env=env).decode().strip()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--base-tree',required=True)
    args=parser.parse_args()
    read=lambda p:json.loads(p.read_text())
    review_path=A/'FINAL_EVIDENCE_REVIEW_R1.json'
    review=read(review_path)
    assert review['verdict'] in ('PASS','PASS WITH FIXES') and not review.get('blocking_issues',[])
    assert read(N/'PARENT_STORAGE_ACCEPTANCE.json')['status']=='ACCEPTED'
    assert (OUT/'REPORT.md').is_file() and (OUT/'DATA_AVAILABILITY.md').is_file()
    assert git('rev-parse','HEAD')==args.expected_head
    assert git('diff','--cached','--name-only')==''
    original=Path(git('rev-parse','--git-path','index')).resolve()
    prepared=Path('/tmp/phase0-3-cycle04-publication.index')
    backup=Path('/tmp/phase0-3-cycle04-original.index')
    shutil.copyfile(original,backup);shutil.copyfile(original,prepared)
    env=dict(os.environ,GIT_INDEX_FILE=str(prepared),GIT_NO_LAZY_FETCH='1')
    paths={ROOT/name for name in ('AGENTS.md','buchong.md','EXP_AI_CONTEXT.md','MDMT_ALGORITHM_METRICS_LEDGER.md')}
    context=ROOT/'research_context/experiments'
    paths.update(context/name for name in ('Experimental_Global_Summary.md','Experimental_Evidence_Ledger.md',
        'Experimental_Experiment_Index.md','Experimental_Open_Questions.md','Experimental_Stable_Findings.md'))
    plan=ROOT/'research_plans/26-11-16_augmentation_gradient_momentum'
    paths.add(plan/'00_EXECUTION_RUNBOOK.md')
    for directory in (OUT,A,plan/'astra_cycles/cycle_04',ROOT/'experiments/completion_weight_attribution_20261117'):
        paths.update(p for p in directory.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json','.csv','.png'))
    paths.add(A.parent/'cycle_03/PUBLICATION_RECEIPT.json')
    paths.update(p for p in N.iterdir() if p.is_file() and p.suffix in ('.json','.md')
        and p.name not in ('guardian.json','STORAGE_OWNER.json'))
    archived_tables=[]
    for p in (N/'tables').iterdir():
        if p.is_file() and p.stat().st_size<=8*1024*1024:paths.add(p)
        elif p.is_file():archived_tables.append(p.name)
    archive=read(N/'RAW_RUNS_MANIFEST.json')
    for part in archive['parts']:
        path=N/part['path'];assert path.stat().st_size==part['bytes'] and sha(path)==part['sha256']
        paths.add(path)
    paths.discard(A/'PUBLICATION_RECEIPT.json')
    rows=[]
    for path in sorted(paths):
        assert path.is_file() and not path.is_symlink() and path.is_relative_to(ROOT)
        assert path.name not in ('guardian.json','STORAGE_OWNER.json')
        assert path.suffix.lower() not in ('.pt','.pth','.ckpt','.pkl','.safetensors','.npy','.npz')
        data=path.read_bytes();assert len(data)<100*1024*1024
        expected=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        actual=git('hash-object','-w','--stdin',input=data);assert actual==expected
        rows.append(dict(path=str(path.relative_to(ROOT)),absolute_path=str(path),bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),git_sha1=actual))
    entries=''.join(f"100644 {r['git_sha1']}\t{r['path']}\n" for r in rows)
    git('update-index','--add','--index-info',input=entries.encode(),env=env)
    tree=git('write-tree','--missing-ok',env=env)
    assert sha(original)==sha(backup) and git('rev-parse','HEAD')==args.expected_head
    manifest=dict(phase='FINAL_EVIDENCE_PASS_PREPARED_INDEX',repository='Caelestis1903/My-Research-Project',
        branch='main',expected_head=args.expected_head,base_tree_sha=args.base_tree,expected_tree=tree,
        files=rows,prepared_index=str(prepared),original_index=str(original),original_index_backup=str(backup),
        original_index_sha256=sha(original),excluded_private=['guardian.json','STORAGE_OWNER.json'],
        archived_only_tables=sorted(archived_tables),final_evidence_review=str(review_path.relative_to(ROOT)),
        final_evidence_review_sha256=sha(review_path))
    destination=Path('/tmp/phase0-3-cycle04-publication-manifest.json')
    destination.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(manifest=str(destination),files=len(rows),bytes=sum(r['bytes'] for r in rows),expected_tree=tree)))


if __name__=='__main__':main()
