"""Mechanical GPU queue. No method selection or scientific interpretation."""
import argparse,json,os,subprocess,time
from pathlib import Path
from n19_data import ROOT,OUT,dump

def gpu_memory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    return [(int(a),float(b),float(c)) for a,b,c in [row.split(',') for row in raw.splitlines()]]

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=OUT/'queue_manifest.json');a=p.parse_args()
    running={};done={};status_path=a.manifest.with_suffix('.status.json');events=a.manifest.with_suffix('.events.jsonl')
    if status_path.exists():done=json.loads(status_path.read_text()).get('completed',{})
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    while True:
        manifest=json.loads(a.manifest.read_text())
        for name,(proc,handle,device) in list(running.items()):
            code=proc.poll()
            if code is not None:
                handle.close();done[name]=dict(exit_code=code,device=device,finished_time=time.time());del running[name]
                with events.open('a') as f:f.write(json.dumps(dict(event='FINISH',name=name,exit_code=code,device=device,time=time.time()))+'\n')
        pending=[t for t in manifest['tasks'] if t['name'] not in done and t['name'] not in running]
        for device,total,used in gpu_memory():
            slots=sum(v[2]==device for v in running.values())
            while pending and slots<5 and used+manifest.get('reservation_mib',1800)<.9*total:
                task=pending.pop(0);log=Path(task['log']);log.parent.mkdir(parents=True,exist_ok=True);handle=log.open('w')
                cmd=[s.replace('{device}',f'cuda:{device}') for s in task['command']]
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=env)
                running[task['name']]=(proc,handle,device);slots+=1;used+=manifest.get('reservation_mib',1800)
                with events.open('a') as f:f.write(json.dumps(dict(event='START',name=task['name'],pid=proc.pid,device=device,time=time.time(),command=cmd))+'\n')
        state=dict(status='RUNNING',sealed=manifest.get('sealed',False),completed=done,
                   running={name:dict(pid=v[0].pid,device=v[2]) for name,v in running.items()},
                   total=len(manifest['tasks']),updated=time.time())
        if manifest.get('sealed',False) and not running and not pending:
            state['status']='COMPLETE' if all(v['exit_code']==0 for v in done.values()) else 'COMPLETE_WITH_FAILURES'
            dump(status_path,state);print(state['status'],len(done),flush=True);return
        dump(status_path,state);time.sleep(5)

if __name__=='__main__':main()
