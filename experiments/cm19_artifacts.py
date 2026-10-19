"""Checksum completed research states and predictions without reading metrics."""
import argparse,hashlib,json,time
from pathlib import Path
import pandas as pd
from cm19_train import OUT,dump


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    manifest=json.loads((OUT/'run_manifest.json').read_text());runs=manifest['runs']
    complete=[c for c in runs if (OUT/'runs'/c['run_id']/'receipt.json').exists()]
    tests=sum((OUT/'runs'/c['run_id']/'test_receipt.json').exists() for c in runs)
    if not args.allow_partial:assert len(complete)==len(runs) and tests==len(runs),'Final index requires every training and frozen test receipt.'
    target=OUT/'ARTIFACT_INDEX.csv';old={}
    if target.exists():old={r['path']:r for r in pd.read_csv(target,dtype=str).to_dict('records')}
    frozen=json.loads((OUT/'TEST_FREEZE.json').read_text()) if (OUT/'TEST_FREEZE.json').exists() else None
    rows=[];reused=0;hashed=0
    for i,c in enumerate(complete):
        out=OUT/'runs'/c['run_id'];cfg=json.loads((out/'config.json').read_text());test_ready=(out/'test_receipt.json').exists()
        for file in sorted(Path(cfg['state_directory']).iterdir()):
            if file.suffix not in ['.pt','.npz'] or file.name.startswith('test_') and not test_ready:continue
            stat=file.stat();prev=old.get(str(file));same=prev and prev['bytes']==str(stat.st_size) and prev['mtime_ns']==str(stat.st_mtime_ns)
            if same:digest=prev['sha256'];reused+=1
            else:
                h=hashlib.sha256()
                with file.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
                digest=h.hexdigest();hashed+=1
            if frozen and file.name in ['selected.pt','last.pt']:
                label='selected' if file.name=='selected.pt' else 'final'
                assert digest==frozen['runs'][c['run_id']]['checkpoint_sha256'][label],file
            rows.append(dict(run_id=c['run_id'],stage=c['stage'],dataset=c['dataset'],model=c['model'],variant=c['variant'],seed=c['seed'],file=file.name,kind='checkpoint' if file.suffix=='.pt' else 'frozen_test_predictions' if file.name.startswith('test_') else 'diagnostic_probe',path=str(file),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=digest))
        if (i+1)%25==0:print(json.dumps(dict(indexed_completed_runs=i+1,scope_runs=len(complete))),flush=True)
    temp=target.with_suffix('.tmp');pd.DataFrame(rows).to_csv(temp,index=False);temp.replace(target)
    dump(OUT/'artifact_index_receipt.json',dict(updated_unix=time.time(),planned_runs=len(runs),training_receipts=len(complete),test_receipts=tests,files=len(rows),hashed_this_pass=hashed,reused_unchanged_files=reused,bytes=sum(r['bytes'] for r in rows),complete=not args.allow_partial and len(complete)==len(runs) and tests==len(runs),states_and_predictions_stored_locally=True,large_artifacts_committed=False,cache_rule='Only completed-run files; test predictions require test receipt; reuse SHA256 only with unchanged absolute path, byte length and nanosecond mtime.'))
    print(json.dumps(dict(indexed_files=len(rows),completed_run_scope=len(complete),final_index=not args.allow_partial)),flush=True)


if __name__=='__main__':main()
