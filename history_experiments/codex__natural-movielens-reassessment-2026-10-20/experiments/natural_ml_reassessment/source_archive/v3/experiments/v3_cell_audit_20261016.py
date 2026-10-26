"""Raw domain-task size, positive-rate, loss contribution and exposure audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import v3_audit_20261016 as audit
import v3_train_20261016 as train
base=audit.base;OUT=audit.OUT

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 c=json.loads((a.run/'config.json').read_text());receipt=json.loads((a.run/'receipt.json').read_text());b=audit.bundle(c['dataset'],c['seed']);small=int(np.argmin(c['p']));train.corruption(b,c['corruption'],small)
 frozen=np.load(OUT/'splits'/f'{c["dataset"]}_seed{c["seed"]}.npz');model=base.fixed_model(b,c['seed']).to(a.device)
 epochs=sorted(set([10,50,receipt['selected_epoch']]));exposure=np.zeros(len(b.train_x),dtype=np.int32);statistics=[];losses=[]
 # All row counts are exact replay counts, not the target quota approximation.
 for epoch in range(1,max(epochs)+1):
  data=np.load(a.run/f'batches_epoch{epoch:03d}.npz');np.add.at(exposure,data['rows'],1)
  if epoch not in epochs:continue
  state=torch.load(a.run/f'epoch{epoch:03d}_model.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);pred=train.predict(model,b,'train',a.device)
  cells=train.metrics(b,pred,'train');total=0.
  for row in cells:
   d,t=row['domain'],row['task'];pool=np.flatnonzero(b.train_x[:,-1]==d);yy=b.train_y[pool,t];pp=np.clip(pred[pool,t].astype(float),1e-12,1-1e-12)
   bce=torch.nn.functional.binary_cross_entropy(torch.from_numpy(pred[pool,t]),torch.from_numpy(yy),reduction='none').double().numpy();contribution=float(bce.sum()/(2*len(b.train_x)));total+=contribution
   row.update(epoch=epoch,seed=c['seed'],arm=c['arm'],positive_rate=float(yy.mean()),empirical_objective_contribution=contribution,actual_exposure=int(exposure[pool].sum()),positive_exposure=int(exposure[pool]@yy),unique_exposed=int(np.count_nonzero(exposure[pool])),run=str(a.run));losses.append(row)
  expected=float(pd.read_csv(a.run/'trajectory.csv').set_index('epoch').loc[epoch,'train_empirical_objective'])
  assert abs(total-expected)<=1e-6+1e-5*abs(expected),(total,expected)
 for split,subset in [('train',None),('val_all',None),('selection',frozen['selection']),('diagnostic',frozen['diagnostic'])]:
  xx=b.train_x if split=='train' else b.val_x;yy=b.train_y if split=='train' else b.val_y
  indices=np.arange(len(xx)) if subset is None else subset
  for d in range(3):
   chosen=indices[xx[indices,-1]==d]
   for t in range(2):statistics.append(dict(split=split,domain=d,task=t,n=len(chosen),positive=int(yy[chosen,t].sum()),positive_rate=float(yy[chosen,t].mean()),seed=c['seed'],dataset=c['dataset'],run=str(a.run)))
 pd.DataFrame(statistics).to_csv(a.out/'cell_support.csv',index=False);pd.DataFrame(losses).to_csv(a.out/'cell_train_performance.csv',index=False)
 base.dump(a.out/'receipt.json',dict(status='PASS',source_sha=base.sha(__file__),run=str(a.run),config_sha=base.sha(a.run/'config.json'),epochs=epochs,test_performance_read=False,objective_decomposition_verified=True))
if __name__=='__main__':main()
