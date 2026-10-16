"""Aggregate paired numerical readouts across training seeds, retaining cells."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import v3_readout_20261016 as readout
base=readout.base;OUT=readout.OUT

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);manifest=json.loads(a.manifest.read_text());records=[];cells=[];bootstrap=[];stages=[];identities=[];train_weights={}
    for group,pairs in manifest['groups'].items():
        receipts=[json.loads((Path(directory)/'receipt.json').read_text()) for directory in pairs]
        common=min(r['common_pre_degradation_epochs'] for r in receipts)
        for directory,receipt in zip(map(Path,pairs),receipts):
            assert receipt['status']=='COMPLETE';seed=receipt['seed']
            config=json.loads((Path(receipt['source_runs'][0])/'config.json').read_text());train_weights[(group,seed)]=np.array(config['p'])
            frame=pd.read_csv(directory/'paired_readouts.csv');frame['group']=group;frame['seed']=seed;records.extend(frame.to_dict('records'))
            array=np.load(directory/'bootstrap_curves.npz');point=array['point']
            for j,split in enumerate(('selection','diagnostic')):
                for k,metric in enumerate(('auc','logloss')):
                    values=point[j,:,:common,k].mean(1)
                    records.append(dict(group=group,seed=seed,split=split,readout=f'four_seed_common_pre_AULC_1_{common}',metric=metric,SN=values[0],SB=values[1],difference=values[1]-values[0]))
            for arm,onset,budget in zip(('SN','SB'),receipt['onsets'],receipt['optimization_budget']):
                stages.append(dict(group=group,seed=seed,**onset,**budget))
            frame=pd.read_csv(directory/'cell_trajectory.csv');frame['group']=group;cells.extend(frame.to_dict('records'))
            frame=pd.read_csv(directory/'selection_bootstrap.csv')
            for rule,rows in frame.groupby('rule'):
                values=rows['selection_auc_difference'].to_numpy();diag=rows['diagnostic_auc_at_reselected_epochs'].to_numpy()
                bootstrap.append(dict(group=group,seed=seed,rule=rule,replicates=len(rows),selection_ci_low=np.quantile(values,.025),selection_ci_high=np.quantile(values,.975),
                    diagnostic_at_reselection_ci_low=np.quantile(diag,.025),diagnostic_at_reselection_ci_high=np.quantile(diag,.975),
                    SN_epoch_histogram=json.dumps(rows.SN_epoch.value_counts().sort_index().to_dict()),SB_epoch_histogram=json.dumps(rows.SB_epoch.value_counts().sort_index().to_dict())))
            identities.append(dict(group=group,seed=seed,pair_directory=str(directory),receipt_sha=base.sha(directory/'receipt.json')))
    allrows=pd.DataFrame(records);allrows.to_csv(a.out/'seed_readouts.csv',index=False);summary=[]
    for key,rows in allrows.groupby(['group','split','readout','metric']):
        assert not rows.seed.duplicated().any(),key
        row=dict(zip(('group','split','readout','metric'),key));row.update(readout.paired_summary(rows.sort_values('seed').difference));row.update(SN_mean=rows.SN.mean(),SB_mean=rows.SB.mean());summary.append(row)
    base.dump(a.out/'seed_summary.json',summary)
    pd.DataFrame([{**r,'seed_values':json.dumps(r['seed_values'])} for r in summary]).to_csv(a.out/'seed_summary.csv',index=False)
    pd.DataFrame(bootstrap).to_csv(a.out/'selection_uncertainty.csv',index=False);pd.DataFrame(stages).to_csv(a.out/'training_stages.csv',index=False)
    frame=pd.DataFrame(cells);frame.to_csv(a.out/'cell_trajectory.csv',index=False)
    # Within-cell AUC average weighted by split row counts, distinct from pooled
    # cross-domain AUC, so heterogeneous score calibration is not silently mixed.
    marginal=[]
    for key,rows in frame.groupby(['group','seed','split','arm','epoch']):
        item=dict(zip(('group','seed','split','arm','epoch'),key))
        for metric in ('auc','logloss'):
            item['macro_'+metric]=rows[metric].mean();item['sample_weighted_cell_'+metric]=np.average(rows[metric],weights=rows.n)
            item['train_objective_weighted_cell_'+metric]=np.average(rows[metric],weights=train_weights[(key[0],key[1])][rows.domain.to_numpy(dtype=int)])
            for domain,part in rows.groupby('domain'):item[f'd{domain}_{metric}']=part[metric].mean()
            for task,part in rows.groupby('task'):item[f't{task}_{metric}']=part[metric].mean()
        marginal.append(item)
    pd.DataFrame(marginal).to_csv(a.out/'domain_task_trajectory.csv',index=False)
    # Scientific figures show individual seeds as well as the seed mean.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    marginal=pd.DataFrame(marginal)
    for group,rows in marginal[marginal.split=='diagnostic'].groupby('group'):
        fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
        for ax,metric in zip(axes,('macro_auc','macro_logloss')):
            for arm,color in (('SN','#2469ad'),('SB','#c45132')):
                subset=rows[rows.arm==arm]
                for _,seedrows in subset.groupby('seed'):ax.plot(seedrows.epoch,seedrows[metric],color=color,alpha=.22,lw=.8)
                avg=subset.groupby('epoch')[metric].mean();ax.plot(avg.index,avg.values,color=color,label=arm,lw=2)
            ax.set(xlabel='Completed epoch',ylabel=metric.replace('_',' '));ax.legend(frameon=False)
        fig.suptitle(group+' | diagnostic | thin: individual seeds')
        safe=''.join(c if c.isalnum() or c in '-_' else '_' for c in group)
        fig.savefig(a.out/f'{safe}_trajectory.png',dpi=180);fig.savefig(a.out/f'{safe}_trajectory.pdf');plt.close(fig)
    base.dump(a.out/'receipt.json',dict(status='COMPLETE',source_sha=base.sha(__file__),sources=identities,training_seed_uncertainty_separate_from_row_bootstrap=True))
if __name__=='__main__':main()
