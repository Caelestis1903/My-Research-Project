"""Deterministic seed-level tables and preregistered screening rules."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .protocol import protocol,write_json

EFFECTS={'A1':{'P2':1,'P0':-1},'A0.1':{'P3':1,'P1':-1},
         'S0':{'P1':1,'P0':-1},'S1':{'P3':1,'P2':-1},
         'I':{'P3':1,'P1':-1,'P2':-1,'P0':1}}


def summarize(values):
    values=np.asarray(values,dtype=float);n=len(values);mean=float(values.mean());sd=float(values.std(ddof=1))
    margin={4:3.182446305284263,5:2.7764451051977987}[n]*sd/np.sqrt(n)
    return dict(n=n,mean=mean,std=sd,ci_low=mean-margin,ci_high=mean+margin,
                positive=int((values>0).sum()),negative=int((values<0).sum()),
                practically_small=int((np.abs(values)<.001).sum()))


def build_tables(run_dir,p):
    root=Path(run_dir);summaries=[];pairs=[];arm_summaries=[]
    for split in ('validation','test'):
        frame=pd.read_csv(root/f'{split}_metrics.csv')
        assert len(frame)==120
        metrics=[c for c in frame if c.endswith(('_auc','_logloss'))]
        for subset,seeds in [('five_seed',list(range(2022,2027))),('four_seed_sensitivity',list(range(2022,2026)))]:
            selected=frame[frame.seed.isin(seeds)]
            for (dataset,checkpoint),group in selected.groupby(['dataset','checkpoint']):
                for metric in metrics:
                    table=group.pivot(index='seed',columns='arm',values=metric).loc[seeds]
                    for arm in table:
                        arm_summaries.append(dict(split=split,subset=subset,dataset=dataset,checkpoint=checkpoint,
                                                   metric=metric,arm=arm,**summarize(table[arm])))
                    for effect,weights in EFFECTS.items():
                        values=sum(table[arm]*weight for arm,weight in weights.items())
                        for seed,value in values.items():pairs.append(dict(split=split,subset=subset,dataset=dataset,
                            checkpoint=checkpoint,metric=metric,effect=effect,seed=seed,value=float(value)))
                        summaries.append(dict(split=split,subset=subset,dataset=dataset,checkpoint=checkpoint,
                                               metric=metric,effect=effect,**summarize(values)))
    pd.DataFrame(pairs).to_csv(root/'paired_effects.csv',index=False)
    pd.DataFrame(summaries).to_csv(root/'effect_summary.csv',index=False)
    pd.DataFrame(arm_summaries).to_csv(root/'arm_summary.csv',index=False)
    summary=pd.DataFrame(summaries);frozen=json.loads((root/'selection_frozen.json').read_text());decisions={}
    for subset in ('five_seed','four_seed_sensitivity'):
        n=5 if subset=='five_seed' else 4;direction=4 if n==5 else 3
        decisions[subset]={}
        for dataset in p['datasets']:
            def row(split,effect,metric):
                rows=summary[(summary.subset==subset)&(summary.dataset==dataset)&(summary.split==split)&
                             (summary.checkpoint=='selected')&(summary.effect==effect)&(summary.metric==metric)]
                assert len(rows)==1
                return rows.iloc[0].to_dict()
            if subset=='five_seed':scale=frozen['working_scale'][dataset]
            else:
                scale=frozen['four_seed_sensitivity_working_scale'][dataset]
            main='A1' if scale==1 else 'A0.1';other='A0.1' if scale==1 else 'A1'
            def pass_gain(split,effect):
                value=row(split,effect,'macro_auc');return value['mean']>=.001 and value['positive']>=direction
            largest=int(np.argmax(p['datasets'][dataset]['train_domain_counts']))
            guard_metrics=[f'd{largest}t0_auc',f'd{largest}t1_auc','pooled_task_mean_auc','worst_cell_auc']
            guards=[]
            for metric in guard_metrics:
                value=row('test',main,metric)
                if value['mean']<=-.002 and value['negative']>=direction:guards.append(value)
            loss=row('test',main,'macro_logloss')
            if loss['mean']>=.002 and loss['positive']>=direction:guards.append(loss)
            small=all(abs(row('test',effect,'macro_auc')['mean'])<.0005 and
                      row('test',effect,'macro_auc')['practically_small']>=direction for effect in ('A1','A0.1'))
            validation=pass_gain('validation',main);test=pass_gain('test',main)
            status='SUPPORTED' if validation and test else 'SCALE_DEPENDENT_CANDIDATE' if pass_gain('test',other) else 'SMALL_EFFECT' if small else 'NO_PROMOTION'
            if guards:status='TRADEOFF'
            decisions[subset][dataset]=dict(working_scale=scale,effect=main,validation_gate=validation,test_gate=test,
                other_scale_validation_gate=pass_gain('validation',other),other_scale_test_gate=pass_gain('test',other),
                small_effect=small,guards=guards,status=status)
    decisions['stop_after_round']=True
    write_json(root/'DECISIONS.json',decisions)
    return decisions


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--protocol')
    args=parser.parse_args();build_tables(args.run_dir,protocol(args.protocol) if args.protocol else protocol())
