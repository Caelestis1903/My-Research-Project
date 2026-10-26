"""GPU readout queue with measured host-memory reservations; mechanical only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--preceding-status',type=Path,action='append',default=[]);a=p.parse_args()
    lock=a.manifest.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads(a.manifest.read_text());tasks=manifest['tasks'];status=a.manifest.with_suffix('.status.json')
    completed={};active={}
    if status.exists():
        prior=json.loads(status.read_text())
        for name,state in prior['tasks'].items():
            if state['status'] in ('COMPLETE','FAILED'):completed[name]=state
        assert not any(x['status']=='RUNNING' and Path(f'/proc/{x["pid"]}').exists() for x in prior['tasks'].values()),'Live prior child: do not duplicate'
    while True:
        for name,(proc,gpu,log,command,started) in list(active.items()):
            code=proc.poll()
            if code is not None:
                log.close();completed[name]=dict(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,gpu=gpu,command=command,pid=proc.pid,started=started,finished=time.time());del active[name]
        pending=[t for t in tasks if t['name'] not in completed and t['name'] not in active]
        mem=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
        devices={int(row.split(',')[0]):[float(x) for x in row.split(',')[1:]] for row in mem.strip().splitlines()}
        external=[];prior_pending=False
        for preceding in a.preceding_status:
            prior=json.loads(preceding.read_text())
            prior_pending=prior_pending or any(x['status']=='PENDING' for x in prior['tasks'].values())
            external.extend(x for x in prior['tasks'].values() if x['status']=='RUNNING' and Path(f'/proc/{x["pid"]}').exists())
        external=list({x['pid']:x for x in external}.values())
        host_available=float(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))/1024
        for task in pending:
            host_reservation=task.get('host_reservation_mib',4096)
            if host_available<host_reservation+4096:break
            if prior_pending:break
            choices=[]
            for gpu,(total,used) in devices.items():
                count=sum(x[1]==gpu for x in active.values())+sum(x['gpu']==gpu for x in external)
                # Include reservation for children still importing/initializing, admission per device.
                reserved=count*manifest.get('reservation_mib',1800)
                estimate=max(used,reserved)+task.get('reservation_mib',manifest.get('reservation_mib',1800))
                if count<5 and estimate<total*.9:choices.append((count,estimate,gpu))
            if not choices:break
            _,_,gpu=min(choices);command=[s.replace('{device}',f'cuda:{gpu}') for s in task['command']]
            logpath=Path(task['log']);logpath.parent.mkdir(parents=True,exist_ok=True);log=logpath.open('ab')
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=manifest['cwd'],start_new_session=True,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
            active[task['name']]=(proc,gpu,log,command,time.time());devices[gpu][1]+=task.get('reservation_mib',manifest.get('reservation_mib',1800));host_available-=host_reservation
        current={**completed,**{name:dict(status='RUNNING',gpu=value[1],pid=value[0].pid,command=value[3],started=value[4]) for name,value in active.items()}}
        for task in tasks:
            if task['name'] not in current:current[task['name']]=dict(status='PENDING')
        write(status,dict(status='RUNNING' if active else ('WAITING_PREDECESSOR_OR_CAPACITY' if pending else 'COMPLETE'),queue_pid=os.getpid(),time=time.time(),tasks=current))
        if len(completed)==len(tasks):break
        time.sleep(10)
    return 1 if any(x['status']=='FAILED' for x in completed.values()) else 0

if __name__=='__main__':sys.exit(main())
