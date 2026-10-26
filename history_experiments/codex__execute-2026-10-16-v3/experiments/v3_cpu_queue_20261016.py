"""Durable mechanical CPU command queue, no scientific interpretation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);args=p.parse_args()
    lock=args.manifest.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads(args.manifest.read_text());pending=list(manifest['tasks']);active={};done={}
    status=args.manifest.with_suffix('.status.json');assert not status.exists(),'Existing queue state requires primary review'
    while pending or active:
        for name,(process,log,task) in list(active.items()):
            code=process.poll()
            if code is not None:
                log.close();done[name]=dict(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,pid=process.pid,command=task['command']);del active[name]
        while pending and len(active)<manifest.get('max_cpu_processes',2):
            task=pending.pop(0);logpath=Path(task['log']);logpath.parent.mkdir(parents=True,exist_ok=True);log=logpath.open('ab')
            process=subprocess.Popen(task['command'],cwd=manifest['cwd'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
            active[task['name']]=(process,log,task)
        states={**done,**{name:dict(status='RUNNING',pid=value[0].pid,command=value[2]['command']) for name,value in active.items()}}
        for task in pending:states[task['name']]=dict(status='PENDING')
        write(status,dict(status='RUNNING' if active else 'COMPLETE',queue_pid=os.getpid(),tasks=states,time=time.time()))
        if active:time.sleep(10)

if __name__=='__main__':main()
