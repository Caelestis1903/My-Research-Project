"""Completed V9 validation trajectory and prespecified test readout points."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_core import interval

ROOT=Path(__file__).resolve().parent/'core/AC-V9'
validation=pd.read_csv(ROOT/'validation_cell_trajectories.csv')
trajectory=validation.groupby(['arm','seed','completed_epoch'])[['auc','logloss']].mean().reset_index()
test=pd.read_csv(ROOT/'paired_seed_deltas.csv')
fig,axes=plt.subplots(2,2,figsize=(11,7))
colors={'SN':'#2563eb','SB':'#d97706'}
for ax,metric,title in [(axes[0,0],'auc','Full validation macro AUC'),(axes[0,1],'logloss','Full validation macro LogLoss')]:
    for arm in ('SN','SB'):
        table=trajectory[trajectory.arm==arm].pivot(index='completed_epoch',columns='seed',values=metric)
        assert table.shape==(50,4)
        x=table.index.to_numpy()*20;mean=table.mean(axis=1).to_numpy();sd=table.std(axis=1,ddof=1).to_numpy()
        ax.plot(x,mean,label=arm,color=colors[arm],lw=1.7)
        ax.fill_between(x,mean-sd,mean+sd,color=colors[arm],alpha=.12)
    ax.set_title(title);ax.set_xlabel('Actual main updates');ax.legend(frameon=False)
    ax.grid(alpha=.15)
labels=['common09','common19','common29','common39','final','selected']
ticklabels=['E10','E20','E30','E40','E50','Selected']
for ax,metric,title in [(axes[1,0],'macro_auc','Test macro AUC: SB minus SN'),(axes[1,1],'macro_logloss','Test macro LogLoss: SB minus SN')]:
    for index,label in enumerate(labels):
        values=test[test.checkpoint==label].sort_values('seed')[metric].to_numpy()
        assert len(values)==4
        summary=interval(values)
        ax.scatter(index+np.linspace(-.09,.09,4),values,color='#64748b',s=16,alpha=.6,zorder=2)
        ax.errorbar(index,summary['mean'],yerr=[[summary['mean']-summary['ci95_low']],
            [summary['ci95_high']-summary['mean']]],fmt='o',color='#0f766e',capsize=3,zorder=3)
    ax.axhline(0,color='black',lw=.7)
    ax.set_xticks(range(6),ticklabels);ax.set_title(title);ax.grid(axis='y',alpha=.15)
fig.suptitle('AC-V9: fixed M3OE, SGD .3 without momentum; four optimization seeds',fontsize=12)
fig.text(.5,.012,'Top: mean +/- seed SD on full validation (selection uses the original half split).\n'
    'Bottom: seed points and pointwise paired t95 intervals; discrete frozen test readouts, not full-test AULC.\n'
    'Fixed reused population; intervals do not represent independent-population confirmation.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.10,1,.96])
for suffix in ('png','pdf'):fig.savefig(ROOT/('v9_stage_trajectory.'+suffix),dpi=180,bbox_inches='tight')
plt.close(fig)
print(ROOT/'v9_stage_trajectory.png')
