"""Mechanical GPU queue. No algorithm choices, result interpretation or retries."""
import argparse,json,os,subprocess,time
from pathlib import Path
from cm19_train import ROOT,OUT,PYTHON,dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(OUT/'run_manifest.json'));p.add_argument('--stages',default='G0,G0M,G0MC,G1,G2,G3,G4,G4LONG');p.add_argument('--smoke',action='store_true');p.add_argument('--limit',type=int,default=5);p.add_argument('--gpus',default='0,1');a=p.parse_args()
    manifest=json.loads(Path(a.manifest).read_text());stages=a.stages.split(',');runs=[r for r in manifest['runs'] if r['stage'] in stages]
    if a.smoke:
        # Full physical batch and epoch validation for representative classes; engineering only.
        runs=[next(r for r in runs if r['dataset']=='kuairand' and r['model']==model and r['seed']==2022 and r['optimizer']=='adam' and r['variant']=='native') for model in ['fixed','star0','pepnet','hamur0','mmoe','ple']]
    label='engineering' if a.smoke else 'formal';logroot=OUT/'logs'/label;logroot.mkdir(parents=True,exist_ok=True)
    tasks=[];active=[];history=[];gpus=[int(s) for s in a.gpus.split(',')]
    for r in runs:
        out=OUT/('engineering' if a.smoke else 'runs')/r['run_id']
        if (out/'receipt.json').exists():history.append(dict(run_id=r['run_id'],status='ALREADY_COMPLETE'));continue
        if out.exists():raise RuntimeError(f'Unfinished directory requires primary review: {out}')
        tasks.append(r)
    reservations={}
    for path in (OUT/'engineering').glob('*/receipt.json'):
        receipt=json.loads(path.read_text());cfg=json.loads((path.parent/'config.json').read_text());reservations[cfg['model']]=receipt['peak_reserved_mib']+384
    while tasks or active:
        for job in active[:]:
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();job['finished']=time.time();job['exit_code']=rc
            out=OUT/('engineering' if a.smoke else 'runs')/job['config']['run_id'];receipt=out/'receipt.json'
            job['status']='COMPLETE' if rc==0 and receipt.exists() else 'FAILED'
            if receipt.exists():reservations[job['config']['model']]=json.loads(receipt.read_text())['peak_reserved_mib']+384
            row={k:v for k,v in job.items() if k not in ['process','log','config']};row['run_id']=job['config']['run_id'];history.append(row);active.remove(job)
            with (OUT/f'queue_{label}.events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
        mem={int(v[0]):(float(v[1]),float(v[2])) for v in [line.split(',') for line in smi.strip().splitlines()]}
        for gpu in gpus:
            while tasks and sum(j['gpu']==gpu for j in active)<(1 if a.smoke else min(5,a.limit)):
                used,total=mem[gpu]
                # Allow startup processes to allocate before admitting another; memory gate is per GPU.
                if any(j['gpu']==gpu and not (OUT/('engineering' if a.smoke else 'runs')/j['config']['run_id']/'config.json').exists() for j in active):break
                if used>=.9*total:break
                chosen=None
                for r in tasks:
                    need=reservations.get(r['model'],2200 if not r['model'].startswith('hamur') else 5000)
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
