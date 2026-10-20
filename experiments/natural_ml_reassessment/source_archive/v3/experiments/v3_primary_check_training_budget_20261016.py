import json,hashlib,time,sys
from pathlib import Path
import numpy as np,pandas as pd
R=Path('/home/xzf/Code/phase0-3-20261016-v3-git');O=R/'research_plans/2026-10-16_v3/execution_20261016';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mp=O/sys.argv[1];m=json.loads(mp.read_text());q=json.loads(mp.with_suffix('.status.json').read_text());records=[]
for task in m['tasks']:
 c=task['command'];D=Path(c[c.index('--run-dir')+1]);T=int(c[c.index('--epochs')+1]);Q=q['tasks'][task['name']];assert Q['status']=='COMPLETE' and Q['exit_code']==0
 r=json.loads((D/'receipt.json').read_text());assert r['epochs']==T and r['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
 for n in ('config.json','trajectory.csv','metrics_selection.csv',f'epoch{T:03d}.pt',f'validation_epoch{T:03d}.npz',f'batches_epoch{T:03d}.npz'):assert sha(D/n)==r['artifact_hashes'][n],n
 t=pd.read_csv(D/'trajectory.csv');assert t.epoch.tolist()==list(range(1,T+1));a=t.train_empirical_objective.to_numpy();gain=(a[-10:-5].mean()-a[-5:].mean())/max(abs(a[-10:-5].mean()),1e-12)
 metric=pd.read_csv(D/'metrics_selection.csv').groupby('epoch')[['auc','logloss']].mean().to_numpy();assert metric.shape==(T,2)
 eligible=[]
 if gain>.005:
  for start in range(1,T-2):
   peak=int(np.argmax(metric[:start,0]));w=np.arange(start,start+3)
   if np.all(metric[w,0]<metric[peak,0]) and np.all(metric[w,1]>metric[peak,1]) and a[start+2]<a[peak]:eligible.append(start+1)
 records.append(dict(run=str(D),epochs=T,receipt_sha=sha(D/'receipt.json'),critical_sha_checks=6,prior5_objective=float(a[-10:-5].mean()),last5_objective=float(a[-5:].mean()),gain=float(gain),plateau=bool(gain<=.005),eligible_selection_confirmation_windows=eligible if gain>.005 else None,actual_queue_record=Q))
result=dict(status='PASS_ACTUAL_TRAINING_INPUTS',time=time.time(),independent_review=False,records=records,forced_extension=any(x['gain']>.005 and x['eligible_selection_confirmation_windows']==[] for x in records),all_plateau=all(x['plateau'] for x in records),needs_actual_bootstrap_for_nonplateau=any(x['gain']>.005 and x['eligible_selection_confirmation_windows'] for x in records),test_access=False)
(O/sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}));print(json.dumps([{k:x[k] for k in ('run','gain','plateau','eligible_selection_confirmation_windows')} for x in records]))
