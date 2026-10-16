"""Materialize approved raw run matrices from explicitly frozen LR choices."""
import argparse
import json
from pathlib import Path
import sys
import v3_audit_20261016 as audit
OUT=audit.OUT;ROOT=audit.ROOT

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['V4','V5','V8','V9','V10'],required=True);p.add_argument('--calibration',type=Path,required=True);a=p.parse_args()
 c=json.loads(a.calibration.read_text());assert c['status']=='FROZEN';lrs=c['selected_lr'];tasks=[];pairs=[];reuse=[]
 def add(group,opt,seed,arm,extra=None,dataset='kuairand',baseline=False):
  extra=extra or [];name=f'{group}__{dataset}__{opt}__{arm}__seed{seed}';directory=OUT/'runs'/name
  if baseline:
   fixed={'sgd':.1,'adam':.01}
   if lrs[opt]==fixed[opt]:directory=OUT/'runs'/f'V2__{dataset}__{opt}__{arm}__seed{seed}';reuse.append(str(directory));return directory
  command=[sys.executable,'experiments/v3_train_20261016.py','--dataset',dataset,'--seed',str(seed),'--optimizer',opt,'--lr',str(lrs['adam' if opt=='adatask' else opt]),'--arm',arm,'--epochs','50','--device','{device}','--run-dir',str(directory),*extra]
  tasks.append(dict(name=name,command=command,log=str(directory/'stdout.log'),reservation_mib=3000 if '--batch-size' in extra else 1900));return directory
 def paired(group,opt,extra=None,dataset='kuairand',baseline=False):
  for seed in audit.SEEDS:
   runs=[add(group,opt,seed,arm,extra,dataset,baseline) for arm in ('SN','SB')]
   pairs.append(dict(group=group,optimizer=opt,dataset=dataset,seed=seed,natural=str(runs[0]),balanced=str(runs[1])))
 if a.phase=='V4':
  for opt in ('adam','sgd'):paired('V4final',opt,baseline=True)
  paired('V4batch2B','sgd',['--batch-size','81920'])
 elif a.phase=='V5':
  for opt in ('adam','sgd'):paired('V5PERM',opt,['--replacement','PERM'])
  paired('V5ORDER','sgd',['--order','PERMUTED'])
 elif a.phase=='V8':
  for prob in (.1,.3):paired('V8corruption'+str(prob),'sgd',['--corruption',str(prob)])
 elif a.phase=='V9':paired('V9','sgd',dataset='movielens')
 elif a.phase=='V10':paired('V10','adatask')
 path=OUT/(a.phase+'_training_manifest.json');assert not path.exists()
 audit.base.dump(path,dict(cwd=str(ROOT),reservation_mib=1900,phase=a.phase,tasks=tasks,calibration_receipt=str(a.calibration),calibration_sha=audit.base.sha(a.calibration),reuse=reuse))
 audit.base.dump(OUT/(a.phase+'_pairs.json'),dict(pairs=pairs))
if __name__=='__main__':main()
