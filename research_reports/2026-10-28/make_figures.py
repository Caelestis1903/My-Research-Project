"""Export the observed domain/stage trade-off without fitting a size effect."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent
raw = pd.read_csv(root/'analysis/PAIRED_CELL_DELTAS.csv')
raw = raw[(raw.method=='CLUSTER_RELATION')&(raw.reference=='TARGET_MATCH')&
          (raw.state=='selected')&(raw.metric=='logloss')&raw.role.isin(['Q','E'])]
assert len(raw)==48 and set(raw.seed)=={2022,2023}
profile = pd.read_csv(root/'analysis/DATASET_PROFILES.csv')
fig, axes = plt.subplots(2,2,figsize=(10,7),sharey=True,layout='constrained')
colors = ['#1565c0','#c45a00']
for row, dataset in enumerate(['movielens','kuairand']):
    counts = profile[(profile.dataset==dataset)&(profile.role=='F')&(profile.task==0)].set_index('domain').n
    for col, role in enumerate(['Q','E']):
        ax = axes[row,col]
        for task, marker in [(0,'o'),(1,'s')]:
            selected = raw[(raw.dataset==dataset)&(raw.role==role)&(raw.task==task)]
            grouped = selected.groupby('domain').delta.agg(['mean','std']).reindex([0,1,2])
            position = np.arange(3)+(task-.5)*.14
            ax.errorbar(position, grouped['mean'], yerr=grouped['std'],fmt=marker+'-',
                        capsize=3,color=colors[task],label=f'Task {task}: mean +/- sample SD')
            for domain in range(3):
                values = selected[selected.domain==domain].sort_values('seed').delta
                ax.scatter(position[domain]+np.array([-.025,.025]),values,s=12,color=colors[task],alpha=.45)
        ax.axhline(0,color='#666666',linewidth=.8)
        ax.set_xticks(range(3),[f'd{d}\nF n={int(counts[d]):,}' for d in range(3)])
        ax.set_title(f'{dataset} / {role}')
        ax.set_ylabel('CG - TARGET_MATCH log loss (lower is better)')
        ax.grid(axis='y',alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('Domain effects change between Q and historical E; paired training seeds 2022/2023')
fig.savefig(root/'DOMAIN_STAGE_TRADEOFF.png',dpi=220)
fig.savefig(root/'DOMAIN_STAGE_TRADEOFF.svg')

base = pd.read_csv(root/'analysis/CELL_METRICS.csv')
base = base[(base.method=='TARGET_MATCH')&(base.role=='E')&(base.state=='selected')]
summary = base.groupby(['dataset','domain','task'])[['auc','logloss','positive','n']].mean().reset_index()
delta = raw[raw.role=='E'].groupby(['dataset','domain','task']).delta.mean().rename('delta_logloss')
summary = summary.join(delta,on=['dataset','domain','task'])
auc = pd.read_csv(root/'analysis/PAIRED_CELL_SUMMARY.csv')
auc = auc[(auc.method=='CLUSTER_RELATION')&(auc.reference=='TARGET_MATCH')&
          (auc.role=='E')&(auc.state=='selected')&(auc.metric=='auc')]
summary = summary.merge(auc[['dataset','domain','task','mean']].rename(columns={'mean':'delta_auc'}),
                        on=['dataset','domain','task'],validate='one_to_one')
size = profile[(profile.role=='F')&(profile.task==0)][['dataset','domain','n']].rename(columns={'n':'F_rows'})
summary.merge(size,on=['dataset','domain'],validate='many_to_one').to_csv(root/'analysis/DOMAIN_TASK_COMPARISON.csv',index=False)
print('Exported domain/stage figure and 12-cell comparison table.')
