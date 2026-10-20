"""Descriptive stage/capacity/cost tables. No candidate selection or training."""
import json
import numpy as np
import pandas as pd
from n19_data import OUT,dump

def main():
    target=OUT/'dynamics';target.mkdir(exist_ok=True);cellrows=[];trajrows=[];ranks=[];reliability=[]
    for path in sorted((OUT/'runs').glob('*')):
        if not (path/'receipt.json').exists():continue
        c=json.loads((path/'config.json').read_text())
        if c['engineering']:continue
        tag=c['run_id'].split('__seed')[1].split('__',1)[1] if '__' in c['run_id'].split('__seed')[1] else ''
        base=dict(run_id=c['run_id'],suite=c['suite'],dataset=c['dataset'],method=c['method'],seed=c['seed'],tag=tag)
        cells=pd.read_csv(path/'metrics_validation.csv');cells=cells[cells.split=='development'].copy()
        for key,value in base.items():cells[key]=value
        cellrows.append(cells)
        traj=pd.read_csv(path/'trajectory.csv')
        for key,value in base.items():traj[key]=value
        trajrows.append(traj)
        if (path/'rank_allocation.json').exists():
            r=json.loads((path/'rank_allocation.json').read_text())
            for domain,count in enumerate(r['active_ranks']):ranks.append(dict(**base,domain=domain,active_rank=count,spectral_error=r['spectral_reconstruction_relative_errors'][domain]))
        for file in path.glob('reliability_epoch*.json'):
            r=json.loads(file.read_text());confidence=np.array(r['confidence']);evidence=np.array(r['evidence']);slopes=np.array(r['slopes'])
            reliability.append(dict(**base,epoch=r['epoch'],mean_confidence=float(confidence.mean()),min_confidence=float(confidence.min()),fraction_agree=float((evidence==1).mean()),median_abs_slope=float(np.median(np.abs(slopes))),extra_seconds=r['extra_seconds']))
    if not cellrows:return
    cells=pd.concat(cellrows,ignore_index=True);traj=pd.concat(trajrows,ignore_index=True)
    cells.to_csv(target/'cell_trajectories.csv',index=False);traj.to_csv(target/'training_trajectories.csv',index=False)
    cells.groupby(['suite','dataset','method','tag','epoch','domain','task'])[['auc','logloss']].agg(['mean','std','count']).to_csv(target/'mean_cell_trajectories.csv')
    traj.groupby(['suite','dataset','method','tag','epoch'])[['selection_auc','development_auc','objective','epoch_seconds','parameters']].agg(['mean','std','count']).to_csv(target/'mean_training_trajectories.csv')
    stages=[]
    for key,g in cells.groupby(['run_id','suite','dataset','method','seed','domain','task']):
        base=dict(zip(['run_id','suite','dataset','method','seed','domain','task'],key));g=g.set_index('epoch')
        for a,b,label in [(1,10,'early'),(9,10,'pruning_transition'),(10,20,'post_transition'),(20,int(g.index.max()),'late')]:
            if a not in g.index or b not in g.index or a==b:continue
            stages.append(dict(**base,stage=label,from_epoch=a,to_epoch=b,auc_change=float(g.loc[b,'auc']-g.loc[a,'auc']),logloss_change=float(g.loc[b,'logloss']-g.loc[a,'logloss'])))
    pd.DataFrame(stages).to_csv(target/'stage_changes.csv',index=False)
    if ranks:pd.DataFrame(ranks).to_csv(target/'rank_allocations.csv',index=False)
    if reliability:pd.DataFrame(reliability).to_csv(target/'reliability_updates.csv',index=False)
    dump(target/'scope.json',dict(completed_formal_runs=traj.run_id.nunique(),metrics='development only',
        warning='Stage changes are descriptive; epoch9-to10 is not a randomized causal pruning effect. Compare the paired rank arms and fixed baseline.'))
    print('Dynamics:',traj.run_id.nunique(),'completed full-training runs')

if __name__=='__main__':main()
