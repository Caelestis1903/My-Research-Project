"""Three-population Stage2D comparison with original KR source metrics."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from analyze_core import interval
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'core/AC-2D'
HIST=Path('/home/xzf/Code/phase0-3-20261016-git/research_plans/2026-10-16/stage2d_execution_20261016')
file=HIST/'analysis/test_cell_metrics.csv';f=pd.read_csv(file)
assert not f.duplicated(['seed','arm','checkpoint','domain','task']).any()
source=[]
for seed in (2022,2023,2024,2025):
    for arm in ('SN','SB','EN','EB'):
        p=HIST/'runs'/f'seed{seed}'/arm/'attempt1/config.json'
        source.append(dict(seed=seed,arm=arm,config_path=str(p),config_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            metric_path=str(file),metric_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            evidence='historical metric/config; no claim of old heavy-checkpoint reevaluation'))
rows=[]
for (seed,arm,checkpoint),g in f.groupby(['seed','arm','checkpoint']):
    assert len(g)==6
    rows.append(dict(stage='AC-2D',population='kuairand_natural_historical',budget='full50',seed=seed,arm=arm,
        checkpoint=checkpoint,completed_epoch=int(g.epoch.iloc[0])+1,main_steps=int(g.step_main.iloc[0]),arch_steps=int(g.step_arch.iloc[0]),
        macro_auc=g.auc.mean(),macro_logloss=g.logloss.mean()))
hist=pd.DataFrame(rows);hist.to_csv(OUT/'historical_kr_seed_macro_metrics.csv',index=False)
pd.DataFrame(source).to_csv(OUT/'historical_kr_source_index.csv',index=False)
macro=pd.concat([hist,pd.read_csv(OUT/'seed_macro_metrics.csv')],ignore_index=True)
paired=[]
for (population,budget,checkpoint),g in macro.groupby(['population','budget','checkpoint']):
    for candidate,baseline in [('SB','SN'),('EB','EN'),('EN','SN'),('EB','SB')]:
        a=g[g.arm==candidate].set_index('seed');b=g[g.arm==baseline].set_index('seed')
        assert set(a.index)==set(b.index)=={2022,2023,2024,2025}
        for metric in ('macro_auc','macro_logloss'):
            paired.append(dict(population=population,budget=budget,checkpoint=checkpoint,contrast=candidate+'-'+baseline,
                metric=metric,reference_mean=b[metric].mean(),candidate_mean=a[metric].mean(),**interval(a[metric]-b[metric])))
table=pd.DataFrame(paired);table.to_csv(OUT/'three_population_paired_summary.csv',index=False)
short={'kuairand_natural_historical':'Natural KR, original',
    'movielens_artificial_d2_fixed2022':'Artificial ML, N187500',
    'movielens_natural_matched_n187500':'Natural ML, N187500',
    'movielens_natural':'Natural ML, N800167'}
fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,checkpoint in zip(axs,('selected','final')):
    gg=table[(table.checkpoint==checkpoint)&(table.metric=='macro_auc')&(table.contrast=='SB-SN')].copy()
    gg['label']=[short[p]+(' / 250 steps' if b=='matched250' else '') for p,b in zip(gg.population,gg.budget)]
    for i,row in enumerate(gg.itertuples()):
        ax.errorbar(row.mean,i,xerr=np.array([[row.mean-row.ci95_low],[row.ci95_high-row.mean]]),fmt='o',capsize=3)
    ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(len(gg)),gg.label);ax.set_title(checkpoint)
    ax.set_xlabel('SB - SN macro AUC; descriptive seed t95')
fig.tight_layout();fig.savefig(OUT/'stage2d_population_stage.png',dpi=180);fig.savefig(OUT/'stage2d_population_stage.pdf');plt.close(fig)
print(table[(table.checkpoint=='selected')&(table.metric=='macro_auc')].to_string(index=False))
