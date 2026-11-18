"""Deterministic seed-level comparisons; no model state input."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .protocol import protocol,write_json,guards


def summarize(values):
    x=np.asarray(values,dtype=float);n=len(x);mean=float(x.mean());sd=float(x.std(ddof=1))
    assert n==5
    half=2.7764451051977987*sd/np.sqrt(n)
    return dict(n=n,mean=mean,std=sd,ci_low=mean-half,ci_high=mean+half,
                positive=int((x>0).sum()),negative=int((x<0).sum()))


def build_tables(run_dir,p):
    root=Path(run_dir);baseline=json.loads((root/'baseline_frozen.json').read_text())['baselines']
    candidates=json.loads((root/'candidate_frozen.json').read_text())
    pairs=[];summaries=[];arms=[]
    for split in ('validation','test'):
        frame=pd.read_csv(root/f'{split}_metrics.csv')
        assert len(frame)==len(frame[['dataset','arm','seed','checkpoint']].drop_duplicates())
        metrics=[k for k in frame if k.endswith(('_auc','_logloss'))]
        for dataset in p['datasets']:
            contrasts=[('M1-baseline','M1',baseline[dataset]),('G-IID2-baseline','G-IID2',baseline[dataset])]
            if dataset=='movielens':contrasts.append(('L2-ML-B-ML','L2-ML','B-ML'))
            if candidates['gates'][dataset]['triggered']:
                contrasts.extend([('G-ANTI2-baseline','G-ANTI2',baseline[dataset]),('G-ANTI2-G-IID2','G-ANTI2','G-IID2')])
            for checkpoint in ('selected','E10','E50'):
                table=frame[(frame.dataset==dataset)&(frame.checkpoint==checkpoint)].set_index(['arm','seed'])
                for metric in metrics:
                    for arm in table.index.get_level_values('arm').unique():
                        values=table.loc[arm].sort_index()[metric].values
                        arms.append(dict(split=split,dataset=dataset,checkpoint=checkpoint,arm=arm,metric=metric,**summarize(values)))
                    for effect,left,right in contrasts:
                        values=[]
                        for seed in p['seeds']:
                            a=float(table.loc[(left,seed),metric]);b=float(table.loc[(right,seed),metric]);delta=a-b;values.append(delta)
                            pairs.append(dict(split=split,dataset=dataset,checkpoint=checkpoint,effect=effect,arm=left,reference_arm=right,
                                              metric=metric,seed=seed,arm_value=a,reference_value=b,value=delta))
                        summaries.append(dict(split=split,dataset=dataset,checkpoint=checkpoint,effect=effect,metric=metric,**summarize(values)))
    paired=pd.DataFrame(pairs);summary=pd.DataFrame(summaries)
    paired.to_csv(root/'paired_effects.csv',index=False);summary.to_csv(root/'effect_summary.csv',index=False)
    pd.DataFrame(arms).to_csv(root/'arm_summary.csv',index=False)
    decisions={}
    for dataset in p['datasets']:
        decisions[dataset]={}
        for arm in ('M1','G-IID2','G-ANTI2'):
            if arm=='G-ANTI2' and not candidates['gates'][dataset]['triggered']:
                decisions[dataset][arm]=dict(status='NOT_TRIGGERED',gate=candidates['gates'][dataset]);continue
            effect=arm+'-baseline';records={}
            for split in ('validation','test'):
                rows=paired[(paired.dataset==dataset)&(paired.checkpoint=='selected')&(paired.effect==effect)&(paired.split==split)]
                deltas={k:g.sort_values('seed').value.values for k,g in rows.groupby('metric')}
                auc=deltas['macro_auc'];passed=float(auc.mean())>=.001 and int((auc>0).sum())>=4
                records[split]=dict(gain_gate=passed,mean_delta_auc=float(auc.mean()),positive_seeds=int((auc>0).sum()),guards=guards(deltas,p,dataset))
            iid_gate=None
            if arm=='G-ANTI2':
                row=summary[(summary.dataset==dataset)&(summary.checkpoint=='selected')&(summary.effect=='G-ANTI2-G-IID2')&
                            (summary.split=='test')&(summary.metric=='macro_auc')].iloc[0]
                iid_gate=bool(row['mean']>=.001 and row.positive>=4)
            status='TRADEOFF' if records['test']['guards'] else 'GO' if records['test']['gain_gate'] and (iid_gate is not False) else 'NO_PROMOTION'
            decisions[dataset][arm]=dict(status=status,validation=records['validation'],test=records['test'],same_K_iid_gate=iid_gate)
    write_json(root/'DECISIONS.json',dict(study_id=p['study_id'],baselines=baseline,methods=decisions,stop_after_round=True))
    for filename in ('temporal_probe.csv','optimizer_scale.csv'):
        files=list((root/'runs').glob('*/*/seed*/attempt*/'+filename))
        if files:
            chunks=[]
            for path in files:
                table=pd.read_csv(path);table['dataset']=path.parts[-5];table['arm']=path.parts[-4];table['seed']=int(path.parts[-3][4:]);chunks.append(table)
            combined=pd.concat(chunks,ignore_index=True);combined.to_csv(root/filename,index=False)
            if filename=='temporal_probe.csv':
                numeric=[c for c in combined.select_dtypes(include='number') if c not in ('seed','epoch','batch','global_step')]
                grouped=combined.groupby(['dataset','arm','seed','epoch'])[numeric].mean().reset_index()
                grouped['working_baseline']=[arm==baseline[d] for d,arm in zip(grouped.dataset,grouped.arm)]
                grouped.to_csv(root/'temporal_summary.csv',index=False)
    for filename in ('increment_probe.csv','view_probe.csv','counterfactual_update.csv'):
        files=list((root/'probes').glob('*/*/seed*/attempt*/'+filename))
        if files:pd.concat([pd.read_csv(f) for f in files],ignore_index=True).to_csv(root/filename,index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();build_tables(args.run_dir,protocol())
