"""Paired optimizer-configuration interactions at frozen readouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import v3_readout_20261016 as r
base=r.base

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 manifest=json.loads(a.manifest.read_text());runs={};minimum=10**9
 for opt in ('adam','sgd'):
  runs[opt]={}
  for directory in map(Path,manifest[opt]):
   receipt=json.loads((directory/'receipt.json').read_text());assert receipt['status']=='COMPLETE' and receipt['optimizer']==opt
   runs[opt][receipt['seed']]=(receipt,np.load(directory/'bootstrap_curves.npz'),pd.read_csv(directory/'paired_readouts.csv'))
   minimum=min(minimum,receipt['common_pre_degradation_epochs'])
 seeds=sorted(runs['adam']);assert seeds==sorted(runs['sgd']);rows=[]
 for seed in seeds:
  for split_index,split in enumerate(('selection','diagnostic')):
   for horizon,label in ((minimum,'PRIMARY_common_pre_AULC'),(10,'AULC_1_10'),(20,'AULC_1_20')):
    for j,metric in enumerate(('auc','logloss')):
     delta={opt:float((runs[opt][seed][1]['point'][split_index,1,:horizon,j]-runs[opt][seed][1]['point'][split_index,0,:horizon,j]).mean()) for opt in runs}
     rows.append(dict(seed=seed,split=split,readout=label,horizon=horizon,metric=metric,adam_delta=delta['adam'],sgd_delta=delta['sgd'],interaction=delta['adam']-delta['sgd']))
  adam=runs['adam'][seed][2];sgd=runs['sgd'][seed][2];merged=adam.merge(sgd,on=['split','readout','metric'],suffixes=('_adam','_sgd'),validate='one_to_one')
  for row in merged.to_dict('records'):rows.append(dict(seed=seed,split=row['split'],readout=row['readout'],horizon=None,metric=row['metric'],adam_delta=row['difference_adam'],sgd_delta=row['difference_sgd'],interaction=row['difference_adam']-row['difference_sgd']))
 frame=pd.DataFrame(rows);summary=[]
 # AULC_1_10/20 appears both directly and in paired_readouts, so scope duplicate
 # representation by retaining the direct numerical point-array version.
 frame=frame.drop_duplicates(['seed','split','readout','metric'],keep='first')
 frame.to_csv(a.out/'seed_interactions.csv',index=False)
 for key,subset in frame.groupby(['split','readout','metric']):
  row=dict(zip(('split','readout','metric'),key));row.update(r.paired_summary(subset.sort_values('seed').interaction));summary.append(row)
 primary=next(row for row in summary if row['split']=='diagnostic' and row['readout']=='PRIMARY_common_pre_AULC' and row['metric']=='auc')
 required_same_sign=(3*len(seeds)+3)//4
 trigger=max(primary['positive'],primary['negative'])>=required_same_sign and (primary['ci_low']>0 or primary['ci_high']<0) and abs(primary['mean'])>=.003
 base.dump(a.out/'summary.json',summary);base.dump(a.out/'gate.json',dict(status='TRIGGERED' if trigger else 'NOT_TRIGGERED',primary_interaction=primary,common_window=minimum,
  criteria='>=75% seeds same sign AND paired seed95%CI excludes0 AND absmean>=.003',seeds=seeds,required_same_sign=required_same_sign,scope='optimizer configurations, LR-specific; state causality not established',
  source_sha=base.sha(__file__),manifest_sha=base.sha(a.manifest)))
if __name__=='__main__':main()
