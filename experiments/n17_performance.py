"""Primary-orchestrator endpoint tables and training-seed uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import n17_common as c
from n17_n1_control import paired

def curve(run,split='test'):
    conf=json.loads((run/'config.json').read_text());source=pd.read_csv(run/('metrics_test.csv' if split=='test' else 'metrics_selection.csv'));records=[];pdom=np.asarray(conf['p'])
    for epoch,part in source.groupby('epoch'):
        row=dict(epoch=int(epoch))
        for metric in ('auc','logloss'):
            row['macro_'+metric]=float(part[metric].mean());row['train_p_cell_'+metric]=float(np.average(part[metric],weights=pdom[part.domain.to_numpy(dtype=int)]));row['evaluation_p_cell_'+metric]=float(np.average(part[metric],weights=part.n))
            for item in part.itertuples():row[f'd{item.domain}t{item.task}_{metric}']=float(getattr(item,metric))
            for d,sub in part.groupby('domain'):row[f'd{d}_{metric}']=float(sub[metric].mean())
            for task,sub in part.groupby('task'):row[f't{task}_{metric}']=float(sub[metric].mean())
        records.append(row)
    frame=pd.DataFrame(records).set_index('epoch');training=pd.read_csv(run/'trajectory.csv').set_index('epoch')
    frame['train_J']=training.train_empirical_objective;frame['steps']=training.steps;frame['examples']=training.examples;frame['wall_seconds']=training.elapsed_seconds
    return frame,source

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--comparisons',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    comparisons=json.loads(a.comparisons.read_text());a.out.mkdir(parents=True,exist_ok=False);raw=[];cells=[]
    for comp in comparisons['comparisons']:
        runs=[Path(comp[arm]) for arm in ('SN','SB')];budget=int(comp.get('budget',100));curves=[];frames=[];chosen=[]
        for run in runs:
            receipt=json.loads((run/'test_receipt.json').read_text());assert receipt['status']=='COMPLETE'
            x,frame=curve(run);assert all(epoch in x.index for epoch in range(1,budget+1));curves.append(x);frames.append(frame)
            selection=pd.read_csv(run/'trajectory.csv').set_index('epoch');chosen.append(int(selection.loc[1:budget].selection_macro_auc.idxmax()))
        readouts={f'E{epoch}':(epoch,epoch) for epoch in (10,20,50,100,200) if epoch<=budget};readouts['selected']=(chosen[0],chosen[1])
        for readout,epochs in readouts.items():
            for metric in curves[0].columns:
                sn,sb=[float(x.loc[epoch,metric]) for x,epoch in zip(curves,epochs)];raw.append(dict(group=comp['group'],seed=comp['seed'],readout=readout,metric=metric,SN=sn,SB=sb,difference=sb-sn,SN_epoch=epochs[0],SB_epoch=epochs[1],budget=budget,estimand=comp['estimand']))
        for end in (20,100):
            if end>budget:continue
            for metric in curves[0].columns:
                if metric in ('steps','examples','wall_seconds'):continue
                sn,sb=[float(x.loc[1:end,metric].mean()) for x in curves];raw.append(dict(group=comp['group'],seed=comp['seed'],readout=f'AULC1_{end}',metric=metric,SN=sn,SB=sb,difference=sb-sn,SN_epoch=None,SB_epoch=None,budget=budget,estimand=comp['estimand']))
    frame=pd.DataFrame(raw);frame.to_csv(a.out/'paired_seed_endpoints.csv',index=False);summaries=[]
    for (group,readout,metric,estimand),part in frame.groupby(['group','readout','metric','estimand']):
        assert not part.seed.duplicated().any();part=part.sort_values('seed');summaries.append(dict(group=group,readout=readout,metric=metric,estimand=estimand,**paired(part.difference),SN_mean=float(part.SN.mean()),SB_mean=float(part.SB.mean())))
    c.dump(a.out/'seed_summary.json',summaries)
    primary=[x for x in summaries if x['group'] in ('N1_common_adam','N1_common_sgd') and x['readout']=='selected' and x['metric']=='macro_auc'];primary.sort(key=lambda x:x['exact_sign_randomization_p']);assert len(primary)==2
    running=0.
    for index,row in enumerate(primary):running=max(running,(2-index)*row['exact_sign_randomization_p']);row['holm_adjusted_p']=min(1.,running)
    c.dump(a.out/'primary_common_eta_Holm.json',primary);c.dump(a.out/'receipt.json',dict(status='COMPLETE',source_sha=c.sha(__file__),comparison_sha=c.sha(a.comparisons),scope='training-seed uncertainty; historical test reused under frozen protocol; weighted cell AUC is not pooled AUC'))

if __name__=='__main__':main()
