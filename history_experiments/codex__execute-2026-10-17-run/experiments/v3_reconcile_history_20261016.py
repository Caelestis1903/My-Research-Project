"""Existing-artifact numerical reconciliation of v2/v3 fixed-SGD readouts."""
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
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 identities=[];scores=[];cells=[];snapshots=[];hashes=[]
 for seed in audit.SEEDS:
  b=audit.bundle('kuairand',seed);split=np.load(OUT/'splits'/f'kuairand_seed{seed}.npz')
  for arm in ('SN','SB'):
   old=audit.OLD/f'runs/kuairand/sgd/seed{seed}/{arm}/attempt1';new=OUT/'runs'/f'V2__kuairand__sgd__{arm}__seed{seed}'
   c0=json.loads((old/'config.json').read_text());c1=json.loads((new/'config.json').read_text());s0=json.loads((old/'selection_frozen.json').read_text());s1=json.loads((new/'selection_frozen.json').read_text())
   assert c0['init_sha']==c1['init_sha256'] and c0['data']['split_row_fingerprints']==c1['data']['split_row_fingerprints']
   x=pd.read_csv(old/'batch_trace.csv');y=pd.read_csv(new/'batch_trace.csv');assert len(x)==len(y)==2450
   for key in ('epoch','batch','n','d0','d1','d2','row_ids_sha256'):assert np.array_equal(x[key],y[key]),key
   source_changes={key:[value,c1['source_hashes'].get(key)] for key,value in c0['source_hashes'].items() if c1['source_hashes'].get(key)!=value}
   identities.append(dict(seed=seed,arm=arm,batches_identical=2450,initial_hash_identical=True,data_split_fingerprints_identical=True,max_online_loss_difference=float(abs(x.weighted_loss-y.weighted_loss).max()),source_hash_changes=json.dumps(source_changes)))
   for epoch in (10,50):
    paths=[old/f'epoch{epoch:02d}.pt',new/f'epoch{epoch:03d}_model.pt'];states=[torch.load(path,map_location='cpu',weights_only=False)['model'] for path in paths]
    assert states[0].keys()==states[1].keys();maximum=0.;sq=0.;reference=0.
    for key,value in states[0].items():
     assert value.shape==states[1][key].shape
     diff=value.double()-states[1][key].double();maximum=max(maximum,float(diff.abs().max()));sq+=float(diff.square().sum());reference+=float(value.double().square().sum())
    snapshots.append(dict(seed=seed,arm=arm,epoch=epoch,max_abs_difference=maximum,l2_difference=sq**.5,relative_l2_difference=(sq/max(reference,1e-300))**.5))
   stored=np.load(old/'validation_selected.npz');assert np.array_equal(stored['row_ids'],b.val_row_ids) and np.array_equal(stored['labels'],b.val_y) and np.array_equal(stored['domains'],b.val_x[:,-1].astype(int))
   old_epoch=s0['selected_epoch'];new_epoch=s1['selected_epoch'];old_prediction=stored['predictions'];new_same=np.load(new/f'validation_epoch{old_epoch:03d}.npz')['predictions'];new_selected=np.load(new/f'validation_epoch{new_epoch:03d}.npz')['predictions']
   row=dict(seed=seed,arm=arm,old_selected_epoch=old_epoch,new_selected_epoch=new_epoch,max_prediction_difference_same_epoch=float(abs(old_prediction-new_same).max()),rms_prediction_difference_same_epoch=float(np.sqrt(np.mean(np.square(old_prediction.astype(float)-new_same)))))
   for role,prediction in (('old_full_selected',old_prediction),('new_at_old_full_selected_epoch',new_same),('new_half_selected',new_selected)):
    for name,subset in (('full_validation',None),('selection_half',split['selection'])):
     results=train.metrics(b,prediction,'val',subset)
     for metric in ('auc','logloss'):row[role+'__'+name+'__'+metric]=float(np.mean([cell[metric] for cell in results]))
     for cell in results:cell.update(seed=seed,arm=arm,role=role,evaluation_population=name);cells.append(cell)
   assert abs(row['old_full_selected__full_validation__auc']-s0['validation_macro_auc'])<1e-12
   scores.append(row)
   for path in (old/'config.json',old/'selection_frozen.json',old/'validation_selected.npz',new/'config.json',new/'selection_frozen.json'):
    hashes.append(dict(path=str(path),sha256=base.sha(path)))
 pd.DataFrame(identities).to_csv(a.out/'training_identity.csv',index=False);pd.DataFrame(snapshots).to_csv(a.out/'parameter_snapshots.csv',index=False);frame=pd.DataFrame(scores);frame.to_csv(a.out/'reconciled_scores.csv',index=False);pd.DataFrame(cells).to_csv(a.out/'reconciled_cells.csv',index=False)
 paired=[]
 for key in (c for c in frame.columns if '__' in c):
  table=frame.pivot(index='seed',columns='arm',values=key)
  for seed,row in table.iterrows():paired.append(dict(seed=seed,readout=key,SN=row['SN'],SB=row['SB'],difference=row['SB']-row['SN']))
 pd.DataFrame(paired).to_csv(a.out/'paired_reconciliation.csv',index=False);base.dump(a.out/'source_hashes.json',hashes)
 base.dump(a.out/'receipt.json',dict(status='COMPLETE_NUMERICAL_AUDIT',source_sha=base.sha(__file__),runs=8,snapshot_comparisons=16,identical_training_batches=8*2450,no_training_or_test_executed=True,source_hash_changes_retained_for_primary_review=True))
if __name__=='__main__':main()
