"""Persistent mechanical queue. Records process results, makes no scientific decisions."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    lock=a.manifest.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=json.loads(a.manifest.read_text());tasks=m['tasks'];status=a.manifest.with_suffix('.status.json');done={};active={}
    if status.exists():
        old=json.loads(status.read_text())
        for name,v in old['tasks'].items():
            if v['status'] in ('COMPLETE','FAILED'):done[name]=v
            elif v['status']=='RUNNING':assert not Path(f'/proc/{v["pid"]}').exists(),'live child; do not duplicate'
    while True:
        for name,(proc,gpu,log,cmd,started) in list(active.items()):
            code=proc.poll()
            if code is not None:
                log.close();done[name]=dict(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,gpu=gpu,command=cmd,pid=proc.pid,started=started,finished=time.time());del active[name]
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()
        devices={int((v:=row.split(','))[0]):dict(uuid=v[1].strip(),total=float(v[2]),used=float(v[3])) for row in rows}
        procrows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
        external={gpu:set() for gpu in devices}
        for row in procrows:
            if ',' not in row:continue
            uuid,pid=row.split(',')
            for gpu,d in devices.items():
                if uuid.strip()==d['uuid']:external[gpu].add(int(pid))
        reserved={gpu:0 for gpu in devices}
        for task in tasks:
            name=task['name']
            if name in done or name in active:continue
            candidates=[]
            for gpu,d in devices.items():
                ours={v[0].pid for v in active.values() if v[1]==gpu};count=len(external[gpu]|ours)
                starting=sum(m['reservation_mib'] for v in active.values() if v[1]==gpu and v[0].pid not in external[gpu])
                need=d['used']+starting+reserved[gpu]+m['reservation_mib']
                if count<5 and need<.9*d['total']:candidates.append((count,need,gpu))
            if not candidates:break
            _,_,gpu=min(candidates);cmd=[x.replace('{device}',f'cuda:{gpu}') for x in task['command']]
            log=Path(task['log']);log.parent.mkdir(parents=True,exist_ok=True);handle=log.open('ab')
            proc=subprocess.Popen(cmd,cwd=m['cwd'],stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
            active[name]=(proc,gpu,handle,cmd,time.time())
            # active imports are already accounted for in starting on the next admission.
        current={**done,**{name:dict(status='RUNNING',pid=v[0].pid,gpu=v[1],command=v[3],started=v[4]) for name,v in active.items()}}
        for task in tasks:current.setdefault(task['name'],dict(status='PENDING'))
        write(status,dict(status='RUNNING' if active else ('WAITING_CAPACITY' if len(done)<len(tasks) else 'FINISHED'),queue_pid=os.getpid(),time=time.time(),tasks=current))
        if len(done)==len(tasks):break
        time.sleep(10)

if __name__=='__main__':main()
