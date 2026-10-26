"""Keep original S5/S6 CSV evidence separate from new natural performance."""
from pathlib import Path
import hashlib
import re
import numpy as np
import pandas as pd
from analyze_core import interval

HERE=Path(__file__).resolve().parent
OUT=HERE/'controlled_history';OUT.mkdir(exist_ok=True)
OLD=Path('/home/xzf/Code/phase0-3/experiments/data')
frames=[];sources=[]
for stage in ('AC-S5','AC-S6'):
    for dataset,suffix in [('kuairand',''),('movielens','ML')]:
        for arm in ('F','T','HT','DH'):
            for seed in (2022,2023,2024,2025):
                path=OLD/f'stage_{stage[3:]}{suffix}_raw_results'/arm/f'seed{seed}/joint/metrics_test.csv'
                f=pd.read_csv(path)
                assert not f.duplicated(['checkpoint','domain','task']).any()
                assert f.groupby('checkpoint').size().eq(6).all()
                f['checkpoint']=f.checkpoint.map(lambda s:'common'+str(int(s.rsplit('_',1)[-1])).zfill(2) if s.startswith('common_epoch_') else s)
                f['stage']=stage;f['population']=dataset+'_natural_historical' if dataset=='kuairand' else 'movielens_artificial_historical'
                f['arm']=arm;f['seed']=seed;f['budget']='historical_full50';f['completed_epoch']=f.stage_epoch+1
                f['run_id']='historical__'+stage+'__'+dataset+'__'+arm+'__'+str(seed)
                f['main_steps']=np.nan;f['arch_steps']=np.nan
                f['n']=f.test_sample_count;f['n_pos']=f.test_positive_count;f['train_p_weight']=f.domain_share/2
                f['evidence_grade']='L2 saved CSV; old state/optimizer lineage not reconstructed'
                frames.append(f)
                sources.append(dict(stage=stage,population=f.population.iloc[0],arm=arm,seed=seed,path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),grade='historical raw CSV, not old predictions reevaluated'))
    new=HERE/'core'/stage/'seed_cell_metrics.csv'
    if new.exists():
        f=pd.read_csv(new);f['evidence_grade']='new frozen-prediction metric reconstruction';frames.append(f)
cell=pd.concat(frames,ignore_index=True);cell.to_csv(OUT/'all_cell_metrics.csv',index=False)
pd.DataFrame(sources).to_csv(OUT/'historical_source_index.csv',index=False)
macro=[]
keys=['stage','population','budget','arm','seed','checkpoint']
for key,g in cell.groupby(keys):
    assert len(g)==6
    target=1 if key[1]=='kuairand_natural_historical' else 2
    for task in (-1,0,1):
        gg=g if task==-1 else g[g.task==task]
        macro.append(dict(zip(keys,key),task=task,completed_epoch=int(gg.completed_epoch.iloc[0]),
            macro_auc=gg.auc.mean(),macro_logloss=gg.logloss.mean(),
            target_auc=gg[gg.domain==target].auc.mean(),target_logloss=gg[gg.domain==target].logloss.mean(),
            non_target_auc=gg[gg.domain!=target].auc.mean(),non_target_logloss=gg[gg.domain!=target].logloss.mean(),
            train_p_cell_auc=np.average(gg.auc,weights=gg.train_p_weight)))
macro=pd.DataFrame(macro);macro.to_csv(OUT/'seed_macro_task_metrics.csv',index=False)
pairs=[];summaries=[];cells=[]
for (stage,pop,budget,checkpoint),g in macro.groupby(['stage','population','budget','checkpoint']):
    for candidate,reference in [('T','F'),('HT','F'),('DH','F'),('DH','HT')]:
        a=g[g.arm==candidate].set_index(['seed','task']);b=g[g.arm==reference].set_index(['seed','task'])
        if a.empty or b.empty:continue
        assert set(a.index)==set(b.index)
        contrast=candidate+'-'+reference
        for ix in a.index:
            pairs.append(dict(stage=stage,population=pop,budget=budget,checkpoint=checkpoint,contrast=contrast,
                seed=ix[0],task=ix[1],**{m:float(a.loc[ix,m]-b.loc[ix,m]) for m in (
                    'macro_auc','macro_logloss','target_auc','target_logloss','non_target_auc','non_target_logloss','train_p_cell_auc')}))
        for task in (-1,0,1):
            aa=a.xs(task,level='task');bb=b.xs(task,level='task')
            assert set(aa.index)==set(bb.index)=={2022,2023,2024,2025}
            for metric in ('macro_auc','macro_logloss','target_auc','target_logloss','non_target_auc','non_target_logloss','train_p_cell_auc'):
                summaries.append(dict(stage=stage,population=pop,budget=budget,checkpoint=checkpoint,contrast=contrast,
                    task=task,metric=metric,reference_mean=bb[metric].mean(),candidate_mean=aa[metric].mean(),**interval(aa[metric]-bb[metric])))
        cc=cell[(cell.stage==stage)&(cell.population==pop)&(cell.budget==budget)&(cell.checkpoint==checkpoint)]
        aa=cc[cc.arm==candidate].set_index(['seed','domain','task']);bb=cc[cc.arm==reference].set_index(['seed','domain','task'])
        for ix in aa.index:
            cells.append(dict(stage=stage,population=pop,budget=budget,checkpoint=checkpoint,contrast=contrast,
                seed=ix[0],domain=ix[1],task=ix[2],auc=aa.loc[ix,'auc']-bb.loc[ix,'auc'],
                logloss=aa.loc[ix,'logloss']-bb.loc[ix,'logloss'],baseline_auc=bb.loc[ix,'auc']))
pd.DataFrame(pairs).to_csv(OUT/'paired_seed_deltas.csv',index=False)
pd.DataFrame(cells).to_csv(OUT/'paired_cell_deltas.csv',index=False)
summary=pd.DataFrame(summaries);summary.to_csv(OUT/'paired_summary.csv',index=False)
print(summary[(summary.task==-1)&(summary.checkpoint=='selected')&(summary.metric=='macro_auc')].to_string(index=False))
