"""Mechanical CPU diagnostics alongside GPU training, no metric-based decisions."""
import json,os,subprocess,time
from cm19_train import ROOT,OUT,PYTHON,dump

def main():
    manifest=json.loads((OUT/'run_manifest.json').read_text());jobs=[];seen=set()
    for c in manifest['runs']:
        if c['stage']=='G0' and c['model'] in ['fixed','star0','star1']:
            jobs.append(dict(run_id=c['run_id'],kind='local_mechanism',script='cm19_local_mechanism.py',receipt='local_mechanism_receipt.json'))
        key=(c['model'],c['variant'],c['dataset'])
        if c['seed']==2022 and c['optimizer']=='adam' and key not in seen:
            seen.add(key);jobs.append(dict(run_id=c['run_id'],kind='cost_profile',script='cm19_costs.py',receipt='cost_profile.json'))
    jobs=[j for j in jobs if not (OUT/'runs'/j['run_id']/j['receipt']).exists()];total=len(jobs);active=[];done=0;failed=[];logs=OUT/'logs/offline';logs.mkdir(parents=True,exist_ok=True)
    while jobs or active:
        for j in active[:]:
            rc=j['proc'].poll()
            if rc is None:continue
            j['log'].close();active.remove(j)
            if rc==0 and (OUT/'runs'/j['run_id']/j['receipt']).exists():done+=1
            else:failed.append(dict(run_id=j['run_id'],kind=j['kind'],exit_code=rc))
        for j in jobs[:]:
            if len(active)>=2:break
            if not (OUT/'runs'/j['run_id']/'receipt.json').exists():continue
            jobs.remove(j);log=(logs/(j['run_id']+'__'+j['kind']+'.log')).open('w');env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
            proc=subprocess.Popen([PYTHON,'-B',str(ROOT/'experiments'/j['script']),'--run-id',j['run_id'],'--device','cpu'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
            active.append({**j,'proc':proc,'log':log,'pid':proc.pid})
        dump(OUT/'offline_queue.status.json',dict(updated=time.time(),total=total,completed=done,failed=failed,pending=len(jobs),running=[{k:v for k,v in j.items() if k not in ['proc','log']} for j in active]))
        if jobs or active:time.sleep(5)
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
