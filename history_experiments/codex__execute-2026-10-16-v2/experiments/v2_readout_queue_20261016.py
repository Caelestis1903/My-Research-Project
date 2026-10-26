"""Mechanical two-process CPU readout queue; no scientific branch decisions."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time
from v2_20261016 import OUT,ROOT,sha,dump
PYTHON='/home/xzf/miniconda3/envs/phase0-3-env/bin/python'


def main():
    lock=(OUT/'readout_queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    review=json.loads((OUT/'analysis_review_verdict.json').read_text());assert review['verdict']=='PASS_PRIMARY_SELF_REVIEW'
    for name,h in review['source_hashes'].items():assert sha(ROOT/'experiments'/name)==h,(name,'source changed after review')
    tasks=[]
    for seed in (2022,2023,2024,2025):
        for arm in ('SN','SB','EN','EB'):
            run=OUT/'runs'/'kuairand'/'adam'/f'seed{seed}'/arm/'attempt1'
            identity=f'kuairand_adam_iid_{arm}_{seed}'
            tasks.append(dict(id='selection_'+identity,script='v2_selection_20261016.py',run=run,
                receipt=OUT/'analysis'/'selection_bootstrap'/(identity+'_receipt.json')))
        for optimizer in ('adam','sgd'):
            run=OUT/'runs'/'kuairand'/optimizer/f'seed{seed}'/'SN'/'attempt1'
            receipt=OUT/'probes'/'kuairand'/optimizer/f'seed{seed}'/'receipt.json'
            tasks.append(dict(id=f'probe_{optimizer}_{seed}',script='v2_probe_20261016.py',run=run,receipt=receipt))
            if optimizer=='adam':
                tasks.append(dict(id=f'weight_{seed}',script='v2_weight_audit_20261016.py',run=run,
                    dependency=receipt,receipt=OUT/'analysis'/'weight_equivalence'/f'seed{seed}'/'receipt.json'))
    for task in tasks:
        task['status']='COMPLETE' if task['receipt'].exists() else 'PENDING'
    children={};(OUT/'logs').mkdir(exist_ok=True)
    while True:
        for pid,(proc,task,log) in list(children.items()):
            code=proc.poll()
            if code is None:continue
            log.close();task.update(status='COMPLETE' if code==0 and task['receipt'].exists() else 'FAILED',exit_code=code,end=time.time());del children[pid]
        for task in tasks:
            if len(children)>=2:break
            if task['status']!='PENDING':continue
            if not (task['run']/'run_manifest.json').exists():continue
            if task.get('dependency') and not task['dependency'].exists():continue
            logpath=OUT/'logs'/(task['id']+'.log');log=logpath.open('a')
            cmd=[PYTHON,'-B','-u',str(ROOT/'experiments'/task['script']),'--run',str(task['run'])]
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            task.update(status='RUNNING',pid=proc.pid,command=cmd,stdout=str(logpath),start=time.time());children[proc.pid]=(proc,task,log)
        done=all(t['status'] in ('COMPLETE','FAILED') for t in tasks)
        serial=[{k:str(v) if isinstance(v,Path) else v for k,v in t.items()} for t in tasks]
        dump(OUT/'readout_queue_status.json',dict(status='FINISHED_PENDING_PRIMARY_ANALYSIS' if done else 'RUNNING',pid=os.getpid(),updated=time.time(),tasks=serial))
        if done:return
        time.sleep(10)

if __name__=='__main__':main()
