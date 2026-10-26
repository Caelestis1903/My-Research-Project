"""Snapshot compact completed-run evidence; never copies datasets or weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--name-prefix',default='');a=p.parse_args();manifest=json.loads(a.manifest.read_text())
 state_path=a.manifest.with_suffix('.status.json');states=json.loads(state_path.read_text())['tasks'] if state_path.exists() else {};entries=[]
 for task in manifest['tasks']:
  if not task['name'].startswith(a.name_prefix):continue
  command=task['command']
  if '--run-dir' not in command:continue
  directory=Path(command[command.index('--run-dir')+1]);receipt_path=directory/'receipt.json'
  if not receipt_path.exists():continue
  receipt=json.loads(receipt_path.read_text())
  if receipt.get('status')!='COMPLETE_TRAINING_TEST_DEFERRED':continue
  target=a.out/directory.name/f'budget{receipt["epochs"]:03d}';target.mkdir(parents=True,exist_ok=True);copied=[]
  for name in ('config.json','receipt.json','trajectory.csv','metrics_selection.csv','selection_frozen.json','batch_trace.csv','optimizer_observations.csv','metrics_test.csv','test_readout_receipt.json'):
   source=directory/name
   if not source.exists():continue
   if name in ('metrics_test.csv','test_readout_receipt.json') and not (directory/'test_readout_receipt.json').exists():continue
   if name in receipt['artifact_hashes']:assert sha(source)==receipt['artifact_hashes'][name],f'Producer hash mismatch {source}'
   dest=target/name
   if dest.exists():assert sha(dest)==sha(source),f'Archive overwrite refused {dest}'
   else:shutil.copyfile(source,dest)
   copied.append(dict(name=name,sha256=sha(dest),bytes=dest.stat().st_size))
  launch=states.get(task['name'],{});entry=dict(run=directory.name,budget=receipt['epochs'],local_artifacts=str(directory.resolve()),archive=str(target),files=copied,actual_launch=launch,manifest_sha=sha(a.manifest));entries.append(entry)
 index=a.out/(a.manifest.stem+'_snapshot_index.json');index.parent.mkdir(parents=True,exist_ok=True)
 # Index snapshots are append-only by their own content hash, supporting later
 # addition of deferred test files without changing any existing archived raw file.
 content=json.dumps(dict(status='ARCHIVED_COMPLETED_ONLY',manifest=str(a.manifest),runs=entries),indent=2)
 digest=hashlib.sha256(content.encode()).hexdigest();version=index.with_name(index.stem+'_'+digest[:12]+'.json')
 if version.exists():assert version.read_text()==content
 else:version.write_text(content)
 print(json.dumps(dict(index=str(version),runs=len(entries),files=sum(len(e['files']) for e in entries))))
if __name__=='__main__':main()
