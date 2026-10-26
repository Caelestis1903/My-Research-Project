from pathlib import Path
import os,subprocess,json,time,hashlib
OUT=Path(__file__).resolve().parent
PYTHON='/home/xzf/miniconda3/envs/phase0-3-env/bin/python'
worker=OUT/'state_oracle.py'
source=hashlib.sha256(worker.read_bytes()).hexdigest()
records=[]
for dataset in ['kuairand','movielens']:
 for seed in [2022,2023,2024,2025]:
  case=OUT/'numeric_oracle'/dataset/f'seed{seed}'
  if case.exists():
   records.append(dict(dataset=dataset,seed=seed,status='EXISTS_INSPECT_BEFORE_RESUME'))
   continue
  log=OUT/f'oracle_{dataset}_{seed}.log'
  cmd=[PYTHON,str(worker),'--dataset',dataset,'--seed',str(seed)]
  rec=dict(dataset=dataset,seed=seed,command=cmd,source_sha256=source,start_unix=time.time(),status='RUNNING')
  records.append(rec)
  (OUT/'numeric_queue_status.json').write_text(json.dumps(records,indent=2))
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
  with log.open('w') as stream:
   p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env)
   rec['pid']=p.pid
   (OUT/'numeric_queue_status.json').write_text(json.dumps(records,indent=2))
   rec['exit_code']=p.wait()
  rec['end_unix']=time.time();rec['status']='EXIT0_PENDING_ANALYSIS' if rec['exit_code']==0 else 'FAILED'
  (OUT/'numeric_queue_status.json').write_text(json.dumps(records,indent=2))
  assert hashlib.sha256(worker.read_bytes()).hexdigest()==source
