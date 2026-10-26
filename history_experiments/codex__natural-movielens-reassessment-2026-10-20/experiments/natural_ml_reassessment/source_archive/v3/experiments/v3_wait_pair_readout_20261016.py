"""Mechanical dependency waiter for an already authorized paired readout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

def main():
 p=argparse.ArgumentParser();p.add_argument('--natural',type=Path,required=True);p.add_argument('--balanced',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=50);a=p.parse_args()
 while True:
  ready=[]
  for directory in (a.natural,a.balanced):
   path=directory/'receipt.json'
   receipt=json.loads(path.read_text()) if path.exists() else {}
   ready.append(receipt.get('status')=='COMPLETE_TRAINING_TEST_DEFERRED' and receipt.get('epochs')==a.epochs)
   if (directory/'failure.json').exists():raise RuntimeError(f'Training failed: {directory}')
  if all(ready):break
  time.sleep(20)
 command=[sys.executable,'experiments/v3_readout_20261016.py','--mode','pair','--natural',str(a.natural),'--balanced',str(a.balanced),'--out',str(a.out),'--bootstraps','1000']
 print(json.dumps(dict(status='DEPENDENCIES_READY',command=command,time=time.time())),flush=True)
 return subprocess.call(command)
if __name__=='__main__':sys.exit(main())
