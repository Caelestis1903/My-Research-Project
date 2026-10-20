"""Plot completed frozen H200 test trajectories; no new model evaluation."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
P=Path(__file__).resolve().parent/'n17'
f=pd.read_csv(P/('all_test_cell_metrics.csv' if (P/'all_test_cell_metrics.csv').exists() else 'all_test_cell_metrics.csv.gz'));f=f[f.group=='H200']
s=f.groupby(['arm','seed','epoch'])[['auc','logloss']].mean().reset_index()
s.to_csv(P/'H200_test_trajectory_plot_data.csv',index=False)
fig,axs=plt.subplots(1,3,figsize=(13,3.5),constrained_layout=True)
colors={'SN':'#2563eb','SB':'#dc6b16'}
for arm,g in s.groupby('arm'):
 m=g.groupby('epoch')[['auc','logloss']].agg(['mean','std'])
 for ax,metric,title in zip(axs[:2],['auc','logloss'],['Frozen test macro cell AUC','Frozen test macro cell LogLoss']):
  x=m.index.to_numpy();y=m[(metric,'mean')].to_numpy();sd=m[(metric,'std')].to_numpy()
  ax.plot(x,y,color=colors[arm],label=arm);ax.fill_between(x,y-sd,y+sd,color=colors[arm],alpha=.14);ax.set_title(title)
a=s[s.arm=='SB'].set_index(['seed','epoch']);b=s[s.arm=='SN'].set_index(['seed','epoch']);d=(a.auc-b.auc).groupby('epoch').agg(['mean','std'])
axs[2].plot(d.index,d['mean'],color='#6d28d9');axs[2].fill_between(d.index,d['mean']-d['std'],d['mean']+d['std'],color='#6d28d9',alpha=.14);axs[2].axhline(0,color='.4',lw=.8);axs[2].set_title('Paired SB - SN macro AUC')
for ax in axs:
 ax.axvline(100,color='.35',ls='--',lw=1);ax.set_xlabel('Completed epoch');ax.grid(alpha=.18)
axs[0].legend(frameon=False);fig.suptitle('Natural MovieLens N17, Adam 0.01; mean +/- optimizer-seed SD (n=4)',fontsize=11)
fig.savefig(P/'H200_test_trajectories.png',dpi=190);fig.savefig(P/'H200_test_trajectories.pdf');plt.close(fig)
