"""Seed-level arithmetic and preregistered gates; no scientific interpretation."""
import numpy as np
import pandas as pd
from .protocol import write_json


def summarize(values):
    x=np.asarray(values,dtype=float);n=len(x)
    sd=float(x.std(ddof=1)) if n>1 else 0.
    critical={2:12.706204736432095,3:4.302652729696142,5:2.7764451051977987}.get(n)
    half=None if critical is None else critical*sd/np.sqrt(n)
    mean=float(x.mean())
    return dict(n=n,mean=mean,sample_sd=sd,ci95_low=None if half is None else mean-half,
        ci95_high=None if half is None else mean+half,positive=int((x>0).sum()),negative=int((x<0).sum()))


def compare(candidate,reference,dataset):
    assert set(candidate)==set(reference) and len(candidate)>=2
    keys=[k for k in next(iter(candidate.values())) if k.endswith(('_auc','_logloss'))]
    deltas={k:[float(candidate[s][k])-float(reference[s][k]) for s in sorted(candidate)] for k in keys}
    summary={k:summarize(v) for k,v in deltas.items()};auc=summary['macro_auc'];ll=summary['macro_logloss']
    largest=2 if dataset=='movielens' else 1
    guards=[k for k in ['worst_cell_auc',f'd{largest}t0_auc',f'd{largest}t1_auc'] if summary[k]['mean']<-.002]
    if ll['mean']>=.002:guards.append('macro_logloss')
    positive=auc['positive']==len(candidate)
    passed=auc['mean']>=.001 and positive and not guards
    exploratory=auc['mean']>=.0003 and positive and not guards
    matched=auc['mean']>=.0005 and positive and not guards
    tradeoff=(ll['negative']==len(candidate) or summary['worst_cell_auc']['positive']==len(candidate))
    return dict(passed=passed,exploratory=exploratory,matched_exceeded=matched,guards=guards,
        tradeoff=tradeoff,control_needed=(passed or exploratory or tradeoff),deltas=deltas,summary=summary)


def values(results,dataset,trial,kind='fixed_ema',seeds=None):
    return {r['seed']:r['selected_validation'][kind] for r in results if r['dataset']==dataset and
        r['trial']==trial and (seeds is None or r['seed'] in seeds)}


def build_tables(out):
    paired=[];summaries=[];arms=[]
    for split in ('validation','test'):
        frame=pd.read_csv(out/f'{split}_metrics.csv')
        metrics=[k for k in frame if k.endswith(('_auc','_logloss'))]
        for (dataset,checkpoint),table in frame.groupby(['dataset','checkpoint']):
            groups={(a,k):g.set_index('seed') for (a,k),g in table.groupby(['arm','model_kind'])}
            contrasts=[]
            for arm,kind in groups:
                if kind!='online' and (arm,'online') in groups:contrasts.append((f'{arm}:{kind}-online',(arm,kind),(arm,'online')))
                if arm!='A0' and ('A0',kind) in groups:contrasts.append((f'{arm}-A0:{kind}',(arm,kind),('A0',kind)))
                if arm not in ('A0','A1') and ('A1',kind) in groups:contrasts.append((f'{arm}-A1:{kind}',(arm,kind),('A1',kind)))
            for b,r in [('B1','R1'),('B2','R2'),('B3','R3')]:
                for kind in ('online','fixed_ema','adaptive_ema','replay_rho'):
                    if (b,kind) in groups and (r,kind) in groups:contrasts.append((f'{b}-{r}:{kind}',(b,kind),(r,kind)))
            if ('A1','adaptive_ema') in groups and ('A1','replay_rho') in groups:
                contrasts.append(('C2:adaptive-replay',('A1','adaptive_ema'),('A1','replay_rho')))
            if ('A1','adaptive_ema') in groups and ('A1','fixed_ema') in groups:
                contrasts.append(('C2:adaptive-fixed',('A1','adaptive_ema'),('A1','fixed_ema')))
            for (arm,kind),g in groups.items():
                for metric in metrics:arms.append(dict(split=split,dataset=dataset,checkpoint=checkpoint,arm=arm,
                    model_kind=kind,metric=metric,**summarize(g[metric])))
            for effect,left,right in contrasts:
                common=sorted(set(groups[left].index)&set(groups[right].index))
                for subset,seeds in [('all',common),('development',[s for s in common if s in (2022,2024)]),
                    ('confirmation',[s for s in common if s in (2023,2025,2026)])]:
                    if len(seeds)<2:continue
                    for metric in metrics:
                        diffs=[]
                        for seed in seeds:
                            a=float(groups[left].loc[seed,metric]);b=float(groups[right].loc[seed,metric]);diffs.append(a-b)
                            paired.append(dict(split=split,dataset=dataset,checkpoint=checkpoint,effect=effect,subset=subset,
                                metric=metric,seed=seed,candidate=a,reference=b,delta=a-b))
                        summaries.append(dict(split=split,dataset=dataset,checkpoint=checkpoint,effect=effect,subset=subset,
                            metric=metric,**summarize(diffs)))
    pd.DataFrame(paired).to_csv(out/'paired_effects.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'effect_summary.csv',index=False)
    pd.DataFrame(arms).to_csv(out/'arm_summary.csv',index=False)
    write_json(out/'analysis_receipt.json',dict(status='COMPLETE',paired_rows=len(paired),summary_rows=len(summaries),seed_is_unit=True))
