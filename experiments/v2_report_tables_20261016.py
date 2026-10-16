"""Final descriptive tables; no scientific branch decisions or training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_20261016 import OUT,sha,dump
from v2_analyze_20261016 import summarize


def main():
    target=OUT/'analysis'
    for name in ('wave1_status.json','readout_queue_status.json'):
        state=json.loads((OUT/name).read_text())
        assert all(t['status']=='COMPLETE' for t in state['tasks']),name+' incomplete'
    assert json.loads((target/'receipt.json').read_text())['phase']=='FINAL'
    a=pd.read_csv(target/'aggregate_metrics.csv')
    selected=a.query("checkpoint=='selected'")
    rows=[]
    for keys,z in selected.groupby(['dataset','optimizer','variant','split','arm']):
        assert sorted(z.seed)==[2022,2023,2024,2025]
        for metric in ('auc_macro','logloss_macro','auc_sample_weighted','logloss_sample_weighted'):
            values=z.sort_values('seed')[metric].to_numpy()
            rows.append(dict(zip(['dataset','optimizer','variant','split','arm'],keys),metric=metric,
                mean=float(values.mean()),sd=float(values.std(ddof=1)),n=4,
                **{f'seed{s}':float(v) for s,v in zip((2022,2023,2024,2025),values)}))
    pd.DataFrame(rows).to_csv(target/'selected_absolute_performance.csv',index=False)
    epochs=selected.query("split=='validation'")[['dataset','optimizer','variant','seed','arm','selected_epoch']]
    epochs.to_csv(target/'selected_epochs.csv',index=False)
    d=pd.read_csv(target/'paired_differences.csv');interactions=[]
    # Differences of paired sampling effects describe modification, not mediation.
    for name,left,right,match in [
        ('cyclic_minus_iid',d.query("variant=='cyclic'"),d.query("variant=='iid'"),['dataset','optimizer']),
        ('adatask_minus_adam',d.query("optimizer=='adatask'"),d.query("optimizer=='adam'"),['dataset','variant']),
        ('sgd_minus_adam',d.query("optimizer=='sgd'"),d.query("optimizer=='adam'"),['dataset','variant']),
        ('movielens_minus_kuairand',d.query("dataset=='movielens'"),d.query("dataset=='kuairand'"),['optimizer','variant'])]:
        join=match+['split','checkpoint','contrast','epoch','metric','seed']
        merged=left.merge(right,on=join,suffixes=('_left','_right'),validate='one_to_one')
        for keys,z in merged.groupby(join[:-1]):
            if sorted(z.seed)!=[2022,2023,2024,2025]:continue
            interactions.append(dict(comparison=name,**dict(zip(join[:-1],keys)),
                **summarize((z.sort_values('seed').delta_left-z.sort_values('seed').delta_right).to_numpy()),
                inference='paired descriptive interaction; no equivalence or causal mediation claim'))
    pd.DataFrame(interactions).to_csv(target/'sampling_effect_interactions.csv',index=False)
    objectives=[]
    z=a.query("dataset=='kuairand' and optimizer=='adam' and variant=='iid'")
    for equal,sample in [('EN','SN'),('EB','SB')]:
        x=z[z.arm==equal];y=z[z.arm==sample]
        for split in ('validation','test'):
            for key in ('selected','epoch01','epoch05','epoch10','epoch20','epoch30','epoch40','epoch50'):
                xx=x[(x.split==split)&(x.checkpoint==key)];yy=y[(y.split==split)&(y.checkpoint==key)]
                if key.startswith('epoch') and split=='validation':
                    epoch=int(key[5:]);xx=x[(x.split==split)&(x.checkpoint=='trajectory')&(x.epoch==epoch)]
                    yy=y[(y.split==split)&(y.checkpoint=='trajectory')&(y.epoch==epoch)]
                p=xx.merge(yy,on='seed',suffixes=('_e','_s'),validate='one_to_one')
                if sorted(p.seed)!=[2022,2023,2024,2025]:continue
                for metric in ('auc_macro','logloss_macro'):
                    objectives.append(dict(contrast=equal+'-'+sample,split=split,checkpoint=key,metric=metric,
                        **summarize((p.sort_values('seed')[metric+'_e']-p.sort_values('seed')[metric+'_s']).to_numpy()),
                        scope='objective-changing comparison, not objective-equivalent sampling'))
    pd.DataFrame(objectives).to_csv(target/'objective_weighting_comparisons.csv',index=False)
    boot=[pd.read_csv(p) for p in sorted((target/'selection_bootstrap').glob('*.csv'))]
    probabilities=pd.concat(boot,ignore_index=True);probabilities.to_csv(target/'selection_grid_probabilities.csv',index=False)
    bootrows=[]
    for keys,z in probabilities.groupby(['dataset','optimizer','variant','arm','seed']):
        assert abs(z.probability.sum()-1)<1e-12
        positive=z[z.probability>0];modal=z.sort_values(['probability','epoch'],ascending=[False,True]).iloc[0]
        bootrows.append(dict(zip(['dataset','optimizer','variant','arm','seed'],keys),
            modal_epoch=int(modal.epoch),modal_probability=float(modal.probability),
            entropy=float(-(positive.probability*np.log(positive.probability)).sum()),
            scope='200 stratified validation-row draws conditional on seven saved trained checkpoints'))
    pd.DataFrame(bootrows).to_csv(target/'selection_grid_uncertainty.csv',index=False)
    budgets=[];profiles=[];seen=set()
    for path in sorted((OUT/'runs').glob('*/*/seed*/*/attempt*/run_manifest.json')):
        c=json.loads(path.read_text())['config'];trace=pd.read_csv(path.parent/'batch_trace.csv')
        budgets.append(dict(dataset=c['dataset'],optimizer=c['optimizer'],variant=c['sampler'],seed=c['seed'],arm=c['arm'],
            epochs=c['epochs'],updates=len(trace),row_draws=int(trace.n.sum()),
            train_rows=sum(c['data']['train_domain_counts'].values()),run=str(path.parent)))
        identity=(c['dataset'],c['seed'])
        if identity not in seen:
            seen.add(identity)
            for split,cells in c['data']['cell_stats'].items():
                denominator=sum(v['count'] for k,v in cells.items() if k.endswith('t0'))
                for cell,values in cells.items():
                    domain,task=cell[1:].split('t')
                    profiles.append(dict(dataset=c['dataset'],seed=c['seed'],split=split,
                        domain=int(domain),task=int(task),domain_share=values['count']/denominator,**values))
    pd.DataFrame(budgets).to_csv(target/'completed_training_budgets.csv',index=False)
    pd.DataFrame(profiles).to_csv(target/'split_cell_profiles.csv',index=False)
    for dataset in a.dataset.unique():
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        trajectories=a[(a.dataset==dataset)&(a.split=='validation')&(a.checkpoint=='trajectory')]
        for (opt,variant,arm),z in trajectories.groupby(['optimizer','variant','arm']):
            yy=z.groupby('epoch').mean(numeric_only=True)
            label=f'{opt}/{variant}/{arm}'
            for axis,metric in zip(axes,('auc_macro','logloss_macro')):axis.plot(yy.index,yy[metric],label=label)
        for axis,metric in zip(axes,('Macro AUC','Macro LogLoss')):
            axis.set(xlabel='Completed epoch',ylabel=metric);axis.legend(fontsize=6,ncol=2)
        fig.tight_layout();fig.savefig(target/f'{dataset}_all_trajectories.png',dpi=180)
        fig.savefig(target/f'{dataset}_all_trajectories.pdf');plt.close(fig)
    dump(target/'report_tables_receipt.json',dict(status='COMPLETE',source_sha256=sha(__file__),
        formal_runs=len(budgets),selection_bootstrap_runs=len(boot),causal_claims_generated=False))
    print('Final descriptive report tables complete',len(budgets),len(interactions))


if __name__=='__main__':main()
