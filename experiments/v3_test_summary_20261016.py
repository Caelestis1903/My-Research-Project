"""Frozen-test numerical summary using already frozen selection decisions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import v3_readout_20261016 as r
base=r.base;OUT=r.OUT

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 freeze=json.loads((OUT/'test_readout_frozen.json').read_text());assert freeze['status']=='FROZEN' and freeze['all_condition_definitions_frozen']
 assert freeze['analysis_rules_sha']==base.sha(OUT/'analysis_rules_frozen.json');a.out.mkdir(parents=True,exist_ok=False);results=[];cell_results=[];selection_uncertainty=[]
 manifest=json.loads(a.manifest.read_text())
 assert base.sha(a.manifest) in freeze['test_summary_manifest_hashes'], 'Test summary manifest must be frozen before test'
 for group,pairs in manifest['groups'].items():
  observed_pre=min(json.loads((Path(directory)/'receipt.json').read_text())['common_pre_degradation_epochs'] for directory in pairs)
  common_pre=int(manifest.get('common_pre_windows',{}).get(group,observed_pre))
  assert 1<=common_pre<=observed_pre
  for directory in map(Path,pairs):
   receipt=json.loads((directory/'receipt.json').read_text());seed=receipt['seed'];budget=len(np.load(directory/'bootstrap_curves.npz')['epochs']);curves=[];frames=[]
   for source in map(Path,receipt['source_runs']):
    completion=json.loads((source/'test_readout_receipt.json').read_text());assert completion['status']=='COMPLETE' and completion['test_global_freeze_sha']==base.sha(OUT/'test_readout_frozen.json')
    c=json.loads((source/'config.json').read_text());p_train=np.array(c['p']);cells=pd.read_csv(source/'metrics_test.csv');cells=cells[cells.epoch<=budget];assert len(cells)==budget*6;frames.append(cells)
    records=[]
    for epoch,part in cells.groupby('epoch'):
     item=dict(epoch=epoch)
     for metric in ('auc','logloss'):
      item['macro_'+metric]=part[metric].mean();item['sample_weighted_cell_'+metric]=np.average(part[metric],weights=part.n)
      item['train_objective_weighted_cell_'+metric]=np.average(part[metric],weights=p_train[part.domain.to_numpy(dtype=int)])
      for d,subset in part.groupby('domain'):item[f'd{d}_{metric}']=subset[metric].mean()
      for t,subset in part.groupby('task'):item[f't{t}_{metric}']=subset[metric].mean()
     records.append(item)
    curves.append(pd.DataFrame(records).set_index('epoch'))
   readouts={f'common_epoch{e}':(e,e) for e in (5,10,15,20,30,40,50,100) if e<=budget}
   readouts['final_epoch']=(budget,budget)
   for rule,chosen in receipt['selected_epochs'].items():
    readouts['selected_'+rule]=(chosen['SN'],chosen['SB']);readouts['common_SN_selected_'+rule]=(chosen['SN'],chosen['SN']);readouts['common_SB_selected_'+rule]=(chosen['SB'],chosen['SB'])
   for label,(ne,be) in readouts.items():
    for metric in curves[0].columns:
     sn,sb=curves[0].loc[ne,metric],curves[1].loc[be,metric]
     results.append(dict(group=group,seed=seed,readout=label,metric=metric,SN_epoch=ne,SB_epoch=be,analysis_budget=budget,SN=sn,SB=sb,difference=sb-sn))
    left=frames[0][frames[0].epoch==ne];right=frames[1][frames[1].epoch==be];joined=left.merge(right,on=['domain','task'],suffixes=('_SN','_SB'),validate='one_to_one')
    for row in joined.to_dict('records'):
     for metric in ('auc','logloss'):cell_results.append(dict(group=group,seed=seed,readout=label,domain=row['domain'],task=row['task'],metric=metric,SN_epoch=ne,SB_epoch=be,SN=row[metric+'_SN'],SB=row[metric+'_SB'],difference=row[metric+'_SB']-row[metric+'_SN']))
   intervals={f'AULC_1_{end}':end for end in (10,20,50) if end<=budget}
   intervals[f'common_pre_AULC_1_{common_pre}']=common_pre
   for label,end in intervals.items():
    assert end<=budget
    for metric in curves[0].columns:
     sn,sb=(curve.loc[1:end,metric].mean() for curve in curves)
     results.append(dict(group=group,seed=seed,readout=label,metric=metric,SN_epoch=None,SB_epoch=None,analysis_budget=budget,SN=sn,SB=sb,difference=sb-sn))
    averaged=[frame[frame.epoch<=end].groupby(['domain','task'])[['auc','logloss']].mean() for frame in frames]
    for domain,task in averaged[0].index:
     for metric in ('auc','logloss'):
      sn,sb=(frame.loc[(domain,task),metric] for frame in averaged)
      cell_results.append(dict(group=group,seed=seed,readout=label,domain=domain,task=task,metric=metric,SN_epoch=None,SB_epoch=None,SN=sn,SB=sb,difference=sb-sn))
   bs=pd.read_csv(directory/'selection_bootstrap.csv')
   for rule,part in bs.groupby('rule'):
    effect=curves[1].loc[part.SB_epoch.to_numpy(),'macro_auc'].to_numpy()-curves[0].loc[part.SN_epoch.to_numpy(),'macro_auc'].to_numpy()
    selection_uncertainty.append(dict(group=group,seed=seed,rule=rule,test_auc_difference_at_reselected_epochs_ci_low=np.quantile(effect,.025),ci_high=np.quantile(effect,.975),scope='selection row uncertainty conditional on fitted test trajectories; not test row or training seed uncertainty'))
 frame=pd.DataFrame(results);frame.to_csv(a.out/'seed_test_readouts.csv',index=False);cells=pd.DataFrame(cell_results);cells.to_csv(a.out/'seed_cell_test_readouts.csv',index=False);summaries=[];cell_summary=[]
 for source,keys,destination in ((frame,['group','readout','metric'],summaries),(cells,['group','readout','domain','task','metric'],cell_summary)):
  for key,part in source.groupby(keys):
   assert not part.seed.duplicated().any();item=dict(zip(keys,key));item.update(r.paired_summary(part.sort_values('seed').difference));item.update(SN_mean=part.SN.mean(),SB_mean=part.SB.mean());destination.append(item)
 base.dump(a.out/'test_seed_summary.json',summaries);base.dump(a.out/'test_cell_seed_summary.json',cell_summary);pd.DataFrame(selection_uncertainty).to_csv(a.out/'test_selection_uncertainty.csv',index=False)
 base.dump(a.out/'receipt.json',dict(status='COMPLETE',source_sha=base.sha(__file__),test_global_freeze_sha=base.sha(OUT/'test_readout_frozen.json'),manifest_sha=base.sha(a.manifest),test_used_for_choices=False))
if __name__=='__main__':main()
