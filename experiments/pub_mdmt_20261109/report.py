"""Seed-unit numerical summaries; scientific interpretation belongs to the orchestrator."""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
from .common import read_json,write_json,SEEDS,DATASETS


def report(out):
    out=Path(out); target=out/'tables'; target.mkdir(parents=True,exist_ok=True)
    results=[read_json(p) for phase in ('formal','combination') for p in sorted((out/phase).glob('*/RESULT.json'))]
    if not (out/'EXPERIMENT_COMPLETE.json').exists(): raise ValueError('Complete the fixed experiment matrix before reporting')
    records=[dict(**{k:v for k,v in e.items() if k!='metrics'},**e['metrics']) for r in results for e in r['endpoints']]
    df=pd.DataFrame(records); df.to_csv(target/'ENDPOINTS.csv',index=False,float_format='%.17g')
    cols=[c for c in df if c.endswith(('_auc','_logloss')) or c=='pooled_logloss']
    long=df.melt(id_vars=['dataset','kind','split','anchor','seed','epoch','state_hash'],value_vars=cols,var_name='metric')
    summary=long.groupby(['dataset','kind','split','anchor','metric']).value.agg(['count','mean','std']).reset_index(); summary.to_csv(target/'SUMMARY.csv',index=False,float_format='%.17g')
    pairs=[]
    for arm in ('C0','C1','C2','G0','G1','G2'):
        pairs.extend([(arm,'A'),(arm,'F')])
    pairs.extend([('C0','G0'),('C1','G1'),('C2','G2'),('C1','C0'),('C2','C1')])
    for kind in set(df.kind):
        if kind.startswith('E') and kind.endswith('F'): pairs.extend([(kind,'A'),(kind,'F'),(kind,'C'+kind[1])])
    raw=[]; paired=[]
    for ds,split,anchor in itertools.product(DATASETS,('validation','test'),('selected','E10','E50')):
        section=df[(df.dataset==ds)&(df.split==split)&(df.anchor==anchor)]
        for method,control in pairs:
            a=section[section.kind==method].set_index('seed'); b=section[section.kind==control].set_index('seed')
            if len(a)==0 or len(b)==0: continue
            if set(a.index)!=set(SEEDS) or set(b.index)!=set(SEEDS): raise ValueError('Incomplete paired seeds')
            delta=a.loc[list(SEEDS),cols]-b.loc[list(SEEDS),cols]
            for seed,row in delta.iterrows(): raw.append(dict(dataset=ds,split=split,anchor=anchor,method=method,control=control,seed=seed,**row.to_dict()))
            for metric in cols:
                values=delta[metric].to_numpy(); mean=float(np.mean(values)); sd=float(np.std(values,ddof=1)); half=2.776445105*sd/np.sqrt(5)
                paired.append(dict(dataset=ds,split=split,anchor=anchor,method=method,control=control,metric=metric,n=5,mean=mean,std=sd,ci_low=mean-half,ci_high=mean+half,positive=int((values>0).sum()),negative=int((values<0).sum())))
    pd.DataFrame(raw).to_csv(target/'SEED_PAIRS.csv',index=False,float_format='%.17g'); pd.DataFrame(paired).to_csv(target/'PAIRED.csv',index=False,float_format='%.17g')
    costs=[dict(dataset=r['job']['dataset'],arm=r['job']['arm'],seed=r['job']['seed'],**r['cost']) for r in results]; pd.DataFrame(costs).to_csv(target/'COSTS.csv',index=False,float_format='%.17g')
    decisions=[]
    for ds in DATASETS:
        for method,control in pairs:
            rows=[r for r in raw if r['dataset']==ds and r['split']=='test' and r['anchor']=='selected' and r['method']==method and r['control']==control]
            if not rows: continue
            auc=float(np.mean([r['macro_auc'] for r in rows])); cells={f'd{d}t{t}':float(np.mean([r[f'd{d}t{t}_auc'] for r in rows])) for d in range(3) for t in range(2)}; ll=float(np.mean([r['macro_logloss'] for r in rows]))
            decisions.append(dict(dataset=ds,method=method,control=control,n=5,macro_auc_gain=auc,cell_auc_gains=cells,auc_qualified=auc>=.003,cell_protection=all(v>=-.001 for v in cells.values()),macro_ll_difference=ll,ll_improved=ll<0))
    write_json(target/'DECISIONS.json',decisions)
    stale=[]
    for result in results:
        if not result['job']['arm'].startswith('C'): continue
        directory=out/'formal'/f"{result['job']['dataset']}_{result['job']['arm']}_seed{result['job']['seed']}"
        trace=[json.loads(line) for line in (directory/'ALPHA.jsonl').read_text().splitlines()]
        for row in frequency100_staleness(trace): stale.append(dict(dataset=result['job']['dataset'],arm=result['job']['arm'],seed=result['job']['seed'],**row))
    pd.DataFrame(stale).to_csv(target/'FREQUENCY100_DIAGNOSTIC.csv',index=False,float_format='%.17g')
    display=['dataset','kind','seed','epoch']+[f'd{d}t{t}_{m}' for d in range(3) for t in range(2) for m in ('auc','logloss')]+['macro_auc','macro_logloss','sample_weighted_cell_auc','sample_weighted_cell_logloss','pooled_task_mean_auc','pooled_logloss']
    # Markdown uses pandas formatting without requiring an optional tabulate dependency.
    table=df[(df.split=='test')&(df.anchor=='selected')][display]
    lines=['# PUB-MDMT 数值附表','','完整Validation/Test selected/E10/E50、逐seed配对差、SD与95%配对t区间见本目录CSV。','','| '+' | '.join(display)+' |','| '+' | '.join(['---']*len(display))+' |']
    for row in table.itertuples(index=False,name=None): lines.append('| '+' | '.join(f'{v:.9f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    (target/'NUMERICAL_TABLES.md').write_text('\n'.join(lines)+'\n')
    return dict(status='COMPLETE',physical_trajectories=len(results),endpoint_rows=len(df),paired_summaries=len(paired),tables=str(target))


def frequency100_staleness(trace):
    """Offline coefficient holding on the observed trajectory; no performance prediction."""
    anchors={}; values={}
    for row in trace:
        for group,source in row['groups'].items():
            if source['alpha'] is None: continue
            alpha=np.asarray(source['alpha']); old=anchors.get(group)
            refresh=(row['step']-1)%100==0 or old is None or old[0]!=source['members']
            if refresh: anchors[group]=(source['members'],alpha.copy(),row['step'])
            held=anchors[group]; score=float(np.linalg.norm(alpha-held[1])/max(np.linalg.norm(alpha),1e-30))
            values.setdefault(group,[]).append((score,row['step']-held[2],refresh))
    return [dict(group=g,observed_coefficient_steps=len(v),mean_relative_alpha_distance=float(np.mean([x[0] for x in v])),max_relative_alpha_distance=float(max(x[0] for x in v)),mean_age_steps=float(np.mean([x[1] for x in v])),frequency100_refreshes=sum(x[2] for x in v),nominal_refresh_fraction=sum(x[2] for x in v)/len(v)) for g,v in values.items()]
