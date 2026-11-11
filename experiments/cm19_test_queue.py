"""Mechanical queue for frozen post-training test readout only."""
import argparse,json,os,subprocess,time
from cm19_train import ROOT,OUT,PYTHON,dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1');p.add_argument('--limit',type=int,default=5);a=p.parse_args()
    freeze=json.loads((OUT/'TEST_FREEZE.json').read_text());tasks=[r for r in freeze['runs'] if not (OUT/'runs'/r/'test_receipt.json').exists()];active=[];done=len(freeze['runs'])-len(tasks);failed=[]
    logroot=OUT/'logs/test';logroot.mkdir(parents=True,exist_ok=True)
    while tasks or active:
        for j in active[:]:
            rc=j['proc'].poll()
            if rc is None:continue
            j['log'].close();active.remove(j)
            if rc==0 and (OUT/'runs'/j['run_id']/'test_receipt.json').exists():done+=1
            else:failed.append(dict(run_id=j['run_id'],exit_code=rc))
        smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
        mem={int(v[0]):(float(v[1]),float(v[2])) for v in [s.split(',') for s in smi.strip().splitlines()]}
        for gpu in [int(g) for g in a.gpus.split(',')]:
            if not tasks:break
            used,total=mem[gpu]
            if sum(j['gpu']==gpu for j in active)>=min(5,a.limit) or used+3000>.9*total:continue
            run=tasks.pop(0);log=(logroot/(run+'.log')).open('w');env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
            proc=subprocess.Popen([PYTHON,'-B',str(ROOT/'experiments/cm19_evaluate.py'),'--run-id',run,'--device',f'cuda:{gpu}'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
            active.append(dict(run_id=run,gpu=gpu,proc=proc,log=log,pid=proc.pid))
        dump(OUT/'test_queue.status.json',dict(updated=time.time(),completed=done,failed=failed,pending=len(tasks),running=[{k:v for k,v in j.items() if k not in ['proc','log']} for j in active],total=len(freeze['runs'])))
        if tasks or active:time.sleep(3)
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
