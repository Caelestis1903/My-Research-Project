"""Mechanical persistent GPU queue; records raw run status, never research conclusions."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'research_plans/2026-10-16/stage2d_execution_20261016'
WORKER=ROOT/'experiments/stage2d_20261016.py'
PYTHON='/home/xzf/miniconda3/envs/phase0-3-env/bin/python'
PUBLICATION=ROOT.parent/'phase0-3-20261016-git'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(path,v):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(path)
def gpu_snapshot():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    return [dict(zip(['index','total','used'],map(int,r.split(',')))) for r in s.strip().splitlines()]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    engineering=json.loads((OUT/'engineering_verification.json').read_text())
    profile=json.loads((OUT/'cuda_profile.json').read_text())
    assert engineering['status']=='PASS_ENGINEERING_ONLY' and profile['status']=='PASS'
    assert engineering['worker_sha256']==profile['worker_sha256']==sha(WORKER)
    review=json.loads((OUT/'review_verdict.json').read_text())
    assert review['verdict']=='PRIMARY_REVIEW_ACCEPTED_FOR_STAGE2D' and review['worker_sha256']==sha(WORKER)
    commit=subprocess.check_output(['git','-C',str(PUBLICATION),'rev-parse','HEAD'],text=True).strip()
    for p in [WORKER,Path(__file__).resolve()]:
        content=subprocess.check_output(['git','-C',str(PUBLICATION),'show',commit+':'+str(p.relative_to(ROOT))])
        assert hashlib.sha256(content).hexdigest()==sha(p),'executed source not committed'
    assert shutil.disk_usage(OUT).free>20*1024**3
    reservation=float(profile['reservation_mib'])
    tasks=[]
    for seed in [2022,2023,2024,2025]:
        for arm in ['SN','SB','EN','EB']:
            directory=OUT/'runs'/f'seed{seed}'/arm/'attempt1'
            task=dict(seed=seed,arm=arm,attempt=1,run_dir=str(directory),status='PENDING',code_commit=commit)
            if directory.exists():
                manifest=directory/'run_manifest.json'
                if manifest.exists() and json.loads(manifest.read_text()).get('status')=='COMPLETE':task['status']='COMPLETE'
                else:raise RuntimeError('Existing incomplete run needs explicit recovery: '+str(directory))
            tasks.append(task)
    children={};(OUT/'logs').mkdir(exist_ok=True)
    def event(value):
        with (OUT/'queue_events.jsonl').open('a') as f:f.write(json.dumps(dict(time_unix=time.time(),**value))+'\n')
    event(dict(event='QUEUE_START',pid=os.getpid(),code_commit=commit,worker_sha256=sha(WORKER)))
    while True:
        for pid,(process,task,log) in list(children.items()):
            code=process.poll()
            if code is not None:
                log.close();manifest=Path(task['run_dir'])/'run_manifest.json'
                valid=code==0 and manifest.exists() and json.loads(manifest.read_text()).get('status')=='COMPLETE'
                task.update(status='COMPLETE' if valid else 'FAILED',exit_code=code,end_unix=time.time())
                event(dict(event='EXIT',seed=task['seed'],arm=task['arm'],pid=pid,exit_code=code,status=task['status']))
                del children[pid]
        snapshots=gpu_snapshot()
        for gpu in sorted(snapshots,key=lambda g:g['used']/g['total']):
            active=[t for _,t,_ in children.values() if t['gpu']==gpu['index']]
            if len(active)>=5:continue
            if gpu['used']>.9*gpu['total']:continue
            pending=next((t for t in tasks if t['status']=='PENDING'),None)
            if pending is None:continue
            # Before any first-epoch producer receipt, allow full initialization headroom.
            initializing=sum(reservation for t in active if not (Path(t['run_dir'])/'progress.json').exists())
            if gpu['used']+initializing+reservation>.9*gpu['total']:continue
            logpath=OUT/'logs'/f"seed{pending['seed']}_{pending['arm']}_attempt1.log"
            log=logpath.open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index']),
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONFAULTHANDLER='1')
            cmd=[PYTHON,'-u',str(WORKER),'--mode','train','--arm',pending['arm'],'--seed',str(pending['seed']),
                 '--run-dir',pending['run_dir'],'--code-commit',commit]
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            pending.update(status='RUNNING',pid=process.pid,gpu=gpu['index'],command=cmd,
                stdout=str(logpath),start_unix=time.time(),reservation_mib=reservation)
            children[process.pid]=(process,pending,log)
            event(dict(event='START',pid=process.pid,seed=pending['seed'],arm=pending['arm'],gpu=gpu))
        done=all(t['status'] in ['COMPLETE','FAILED'] for t in tasks)
        status='FINISHED_PENDING_PRIMARY_ANALYSIS' if done else ('RUNNING' if children else 'WAITING_GPU_HEADROOM')
        dump(OUT/'queue_status.json',dict(study_id='STAGE2D_KR_IID_20261016',status=status,pid=os.getpid(),
            updated_unix=time.time(),tasks=tasks,gpus=snapshots,max_processes_per_gpu=5,memory_fraction_limit=.9,
            reservation_mib=reservation,worker_sha256=sha(WORKER),code_commit=commit))
        if done:return
        time.sleep(10)


if __name__=='__main__':main()
