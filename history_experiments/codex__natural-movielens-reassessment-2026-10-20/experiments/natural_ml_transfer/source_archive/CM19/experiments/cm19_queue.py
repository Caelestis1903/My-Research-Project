"""Mechanical GPU queue. No algorithm choices, result interpretation or retries."""
import argparse,json,os,subprocess,time
from pathlib import Path
from cm19_train import ROOT,OUT,ART,PYTHON,dump

class AdoptedProcess:
    """Observe an existing training PID without restarting or changing it."""
    def __init__(self,pid,run_id,out):self.pid=pid;self.run_id=run_id;self.out=out
    def alive(self):
        try:
            stat=Path(f'/proc/{self.pid}/stat').read_text().rsplit(')',1)[1].split()[0]
            cmd=Path(f'/proc/{self.pid}/cmdline').read_bytes()
        except FileNotFoundError:return False
        if stat=='Z' or not cmd:return False
        if b'cm19_train.py' not in cmd or (self.run_id+'.json').encode() not in cmd:raise RuntimeError(f'PID identity changed: {self.pid}')
        return True
    def poll(self):
        if self.alive():return None
        return 0 if (self.out/'receipt.json').exists() else 1

def memory_key(c):return (c['model'],c['variant'],c['dataset'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(OUT/'run_manifest.json'));p.add_argument('--stages',default='G0,G0M,G0MC,G1,G2,G3,G4,G4LONG');p.add_argument('--smoke',action='store_true');p.add_argument('--limit',type=int,default=5);p.add_argument('--gpus',default='0,1');p.add_argument('--adopt-active');p.add_argument('--check-adoption',action='store_true');a=p.parse_args()
    manifest=json.loads(Path(a.manifest).read_text());stages=a.stages.split(',');runs=[r for r in manifest['runs'] if r['stage'] in stages]
    if a.smoke:
        # Full physical batch and epoch validation for representative classes; engineering only.
        runs=[next(r for r in runs if r['dataset']=='kuairand' and r['model']==model and r['seed']==2022 and r['optimizer']=='adam' and r['variant']=='native') for model in ['fixed','star0','pepnet','hamur0','mmoe','ple']]
    label='engineering' if a.smoke else 'formal';logroot=OUT/'logs'/label;logroot.mkdir(parents=True,exist_ok=True)
    tasks=[];active=[];history=[];gpus=[int(s) for s in a.gpus.split(',')]
    adoption={j['run_id']:j for j in json.loads(Path(a.adopt_active).read_text())['running']} if a.adopt_active else {}
    for r in runs:
        out=OUT/('engineering' if a.smoke else 'runs')/r['run_id']
        if r['run_id'] in adoption:
            old=adoption[r['run_id']];proc=AdoptedProcess(old['pid'],r['run_id'],out)
            if proc.alive():
                if a.check_adoption:log=None
                else:log=(logroot/(r['run_id']+'.log')).open('a')
                active.append(dict(config=r,process=proc,log=log,gpu=old['gpu'],pid=old['pid'],started=old['started'],status='RUNNING',adopted=True));continue
        if (out/'receipt.json').exists():history.append(dict(run_id=r['run_id'],status='ALREADY_COMPLETE'));continue
        if out.exists():raise RuntimeError(f'Unfinished directory requires primary review: {out}')
        tasks.append(r)
    if a.check_adoption:
        print(json.dumps(dict(adopted=[dict(run_id=j['config']['run_id'],pid=j['pid'],gpu=j['gpu']) for j in active],already_complete=len(history),pending=len(tasks),new_processes_started=0)));return
    reservations={}
    for folder in ['engineering','runs']:
        for path in (OUT/folder).glob('*/receipt.json'):
            receipt=json.loads(path.read_text());cfg=json.loads((path.parent/'config.json').read_text());key=memory_key(cfg);reservations[key]=max(reservations.get(key,0),receipt['peak_reserved_mib']+384)
    while tasks or active:
        for job in active[:]:
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();job['finished']=time.time();job['exit_code']=rc
            if job.get('adopted'):job['exit_code_inferred_from_receipt']=True
            out=OUT/('engineering' if a.smoke else 'runs')/job['config']['run_id'];receipt=out/'receipt.json'
            job['status']='COMPLETE' if rc==0 and receipt.exists() else 'FAILED'
            if receipt.exists():
                key=memory_key(job['config']);reservations[key]=max(reservations.get(key,0),json.loads(receipt.read_text())['peak_reserved_mib']+384)
            row={k:v for k,v in job.items() if k not in ['process','log','config']};row['run_id']=job['config']['run_id'];history.append(row);active.remove(job)
            with (OUT/f'queue_{label}.events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
        mem={int(v[0]):(float(v[1]),float(v[2])) for v in [line.split(',') for line in smi.strip().splitlines()]}
        smi_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
        process_mem={int(v[0]):float(v[1]) for v in [line.split(',') for line in smi_processes.strip().splitlines() if line.strip()]}
        def allocated(j):return (ART/('engineering' if a.smoke else 'runs')/j['config']['run_id']/'step000001.pt').exists()
        for job in active:
            if allocated(job) and job['pid'] in process_mem:
                key=memory_key(job['config']);reservations[key]=max(reservations.get(key,0),process_mem[job['pid']]+384)
        for gpu in gpus:
            while tasks and sum(j['gpu']==gpu for j in active)<(1 if a.smoke else min(5,a.limit)):
                used,total=mem[gpu]
                # Config is emitted before the full batch allocates activations.
                # The first completed update/checkpoint observes that allocation.
                if any(j['gpu']==gpu and not allocated(j) for j in active):break
                if used>=.9*total:break
                chosen=None
                for r in tasks:
                    need=reservations.get(memory_key(r),6000)
                    if used+need<=.9*total:chosen=r;break
                if chosen is None:break
                tasks.remove(chosen);cmd=[PYTHON,'-B',str(ROOT/'experiments/cm19_train.py'),'--config',str(OUT/'configs'/(chosen['run_id']+'.json')),'--device',f'cuda:{gpu}']
                if a.smoke:cmd.append('--smoke')
                log=(logroot/(chosen['run_id']+'.log')).open('w');env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
                process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
                active.append(dict(config=chosen,process=process,log=log,gpu=gpu,pid=process.pid,started=time.time(),status='RUNNING'))
                print(json.dumps(dict(event='START',run_id=chosen['run_id'],pid=process.pid,gpu=gpu)),flush=True);break
        dump(OUT/f'queue_{label}.status.json',dict(updated=time.time(),pending=len(tasks),running=[dict(run_id=j['config']['run_id'],gpu=j['gpu'],pid=j['pid'],started=j['started']) for j in active],completed=sum(r['status'] in ['COMPLETE','ALREADY_COMPLETE'] for r in history),failed=[r for r in history if r['status']=='FAILED'],total=len(runs)))
        if tasks or active:time.sleep(3)
    if any(r['status']=='FAILED' for r in history):raise SystemExit(1)
if __name__=='__main__':main()
