"""Two-device execution queue. Collects raw exit codes and completion receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from .bootstrap import ROOT, PYTHON, dump
from .manifest import configurations

def execute(cfg,physical_device,gate):
    receipt=Path(cfg['run_dir'])/'run_manifest.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('status')=='COMPLETE':
        return dict(run_id=cfg['run_id'],status='ALREADY_COMPLETE',exit_code=0)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(physical_device))
    log=ROOT/'logs'/(cfg['run_id']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    command=cfg['command']+['--review-report',str(gate)]
    with log.open('a') as handle:
        process=subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT,check=False)
    record=dict(run_id=cfg['run_id'],exit_code=process.returncode,device=str(physical_device),stdout=str(log),
                status='COMPLETE' if process.returncode==0 else 'FAILED')
    dump(ROOT/'queue_receipts'/(cfg['run_id']+'.json'),record)
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',action='append',required=True)
    p.add_argument('--population',action='append');p.add_argument('--budget',choices=['full50','matched250'])
    p.add_argument('--devices',nargs='+',default=['0','1']);p.add_argument('--review-report',default=str(ROOT/'review_gate.json'))
    a=p.parse_args();runs=[c for c in configurations() if c['stage'] in a.stage
        and (not a.population or c['population'] in a.population) and (not a.budget or c['budget']==a.budget)]
    # Deterministic assignment; each worker processes its own ordered stream.
    def worker(index):
        records=[]
        for cfg in runs[index::len(a.devices)]:
            record=execute(cfg,a.devices[index],a.review_report);records.append(record)
            print(json.dumps(record),flush=True)
            if record['exit_code']!=0:break
        return records
    with ThreadPoolExecutor(max_workers=len(a.devices)) as pool:
        records=list(pool.map(worker,range(len(a.devices))))
    if any(r['exit_code'] for group in records for r in group):raise SystemExit(1)

if __name__=='__main__':main()
