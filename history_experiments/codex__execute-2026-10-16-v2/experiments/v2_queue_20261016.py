"""Persistent mechanical queue; consumes primary-authored task manifest."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'research_plans/2026-10-16_v2/execution_20261016'
WORKER=ROOT/'experiments/v2_20261016.py'
PYTHON='/home/xzf/miniconda3/envs/phase0-3-env/bin/python'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)
def gpus():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    return [dict(zip(('index','total','used'),map(int,row.split(',')))) for row in out.strip().splitlines()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);args=p.parse_args()
    manifest=args.manifest.resolve();tasks=json.loads(manifest.read_text())['tasks'];wave=manifest.stem
    lock=(OUT/(wave+'.lock')).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    profile=json.loads((OUT/'cuda_profile.json').read_text());review=json.loads((OUT/'review_verdict.json').read_text())
    assert profile['status']=='PASS' and review['verdict']=='PASS_PRIMARY_SELF_REVIEW'
    assert profile['worker_sha256']==review['worker_sha256']==sha(WORKER)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for source in (WORKER,Path(__file__).resolve()):
        assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+str(source.relative_to(ROOT))],cwd=ROOT)).hexdigest()==sha(source)
    assert shutil.disk_usage(OUT).free>20*1024**3
    reservation=profile['reservation_mib'];children={};(OUT/'logs').mkdir(exist_ok=True)
    for task in tasks:
        directory=Path(task['run_dir']);receipt=directory/'run_manifest.json'
        if directory.exists():
            assert receipt.exists() and json.loads(receipt.read_text())['status']=='COMPLETE',str(directory)
            task['status']='COMPLETE'
        else:task['status']='PENDING'
    def event(x):
        with (OUT/(wave+'_events.jsonl')).open('a') as f:f.write(json.dumps(dict(time=time.time(),**x))+'\n')
    event(dict(event='QUEUE_START',pid=os.getpid(),commit=commit))
    while True:
        # Primary may add a reviewed, prerequisite-satisfied next phase. Existing
        # commands are immutable; a single scheduler keeps GPU admission atomic.
        known={t['id']:t for t in tasks}
        for entry in json.loads(manifest.read_text())['tasks']:
            if entry['id'] in known:
                assert entry['args']==known[entry['id']]['args'] and entry['run_dir']==known[entry['id']]['run_dir']
            else:
                assert not Path(entry['run_dir']).exists()
                entry['status']='PENDING';tasks.append(entry)
        for pid,(process,task,log) in list(children.items()):
            code=process.poll()
            if code is None:continue
            log.close();receipt=Path(task['run_dir'])/'run_manifest.json'
            good=code==0 and receipt.exists() and json.loads(receipt.read_text())['status']=='COMPLETE'
            task.update(status='COMPLETE' if good else 'FAILED',exit_code=code,end=time.time())
            event(dict(event='EXIT',id=task['id'],pid=pid,exit_code=code,status=task['status']));del children[pid]
        snapshots=gpus()
        for gpu in sorted(snapshots,key=lambda x:x['used']/x['total']):
            active=[t for _,t,_ in children.values() if t['gpu']==gpu['index']]
            # Include every other v2 queue wave in the per-device five-worker cap.
            other=[]
            for statusfile in OUT.glob('*_status.json'):
                if statusfile.name==wave+'_status.json':continue
                try:
                    status=json.loads(statusfile.read_text())
                    for task in status.get('tasks',[]):
                        if task.get('status')=='RUNNING' and task.get('gpu')==gpu['index']:
                            os.kill(task['pid'],0);other.append(task)
                except (OSError,ValueError,KeyError):pass
            if len(active)+len(other)>=5:continue
            pending=next((t for t in tasks if t['status']=='PENDING'),None)
            if pending is None:continue
            task_reservation=float(pending.get('reservation_mib',reservation))
            initializing=sum(float(t.get('reservation_mib',reservation)) for t in active+other if not (Path(t['run_dir'])/'progress.json').exists())
            if gpu['used']+initializing+task_reservation>.9*gpu['total']:continue
            logpath=OUT/'logs'/(pending['id']+'.log');log=logpath.open('a')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index']),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2',PYTHONFAULTHANDLER='1',PYTHONDONTWRITEBYTECODE='1')
            program=ROOT/pending.get('worker',str(WORKER.relative_to(ROOT)))
            task_commit=pending.get('code_commit',commit)
            if program!=WORKER:
                task_review=json.loads((ROOT/pending['review_receipt']).read_text())
                assert task_review['verdict']=='PASS_PRIMARY_SELF_REVIEW' and task_review['worker_sha256']==sha(program)
                assert hashlib.sha256(subprocess.check_output(['git','show',task_commit+':'+str(program.relative_to(ROOT))],cwd=ROOT)).hexdigest()==sha(program)
            cmd=[PYTHON,'-B','-u',str(program),'--run-dir',pending['run_dir'],'--code-commit',task_commit]+pending['args']
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            pending.update(status='RUNNING',gpu=gpu['index'],pid=proc.pid,command=cmd,stdout=str(logpath),start=time.time(),reservation_mib=task_reservation)
            children[proc.pid]=(proc,pending,log);event(dict(event='START',id=pending['id'],pid=proc.pid,gpu=gpu))
        done=all(t['status'] in ('COMPLETE','FAILED') for t in tasks)
        dump(OUT/(wave+'_status.json'),dict(status='FINISHED_PENDING_PRIMARY_ANALYSIS' if done else 'RUNNING',
            pid=os.getpid(),updated=time.time(),tasks=tasks,gpus=snapshots,reservation_mib=reservation,commit=commit))
        if done:return
        time.sleep(10)

if __name__=='__main__':main()
