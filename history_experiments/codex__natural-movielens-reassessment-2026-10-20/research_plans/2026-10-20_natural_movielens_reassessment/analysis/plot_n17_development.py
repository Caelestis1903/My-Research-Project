"""Frozen H200 development trajectories; no test access or recipe selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'n17_development'
ART=Path('/home_new/xzf_natural_movielens_reassessment')
manifest=json.loads((ART/'N17/H200_manifest.json').read_text())['configurations']
data=[]
for c in manifest:
    run=Path(c['run_dir'])
    f=pd.read_csv(run/'trajectory.csv')
    cells=pd.read_csv(run/'metrics_selection.csv')
    counts=json.loads((ART/'data'/('movielens_natural.json')).read_text())['data']['train_domain_counts']
    cells['w']=cells.domain.map({int(d):n/sum(counts.values())/2 for d,n in counts.items()})
    ll=cells.groupby('epoch').apply(lambda g:np.average(g.logloss,weights=g.w),include_groups=False)
    f['selection_train_p_LL']=f.epoch.map(ll)
    f['arm']=c['arm'];f['seed']=c['seed'];data.append(f)
f=pd.concat(data,ignore_index=True)
fig,axes=plt.subplots(1,3,figsize=(12.5,3.4))
for arm,color in [('SN','#276FBF'),('SB','#D46624')]:
    g=f[f.arm.eq(arm)]
    for ax,column,title in zip(axes,['train_empirical_objective','selection_macro_auc','selection_train_p_LL'],['Train empirical objective J','Selection macro cell AUC','Selection train-p cell LogLoss']):
        for _,s in g.groupby('seed'):ax.plot(s.epoch,s[column],color=color,alpha=.18,lw=.6)
        mean=g.groupby('epoch')[column].mean();ax.plot(mean.index,mean,color=color,label=arm,lw=1.6)
        ax.axvline(100,color='gray',ls='--',lw=.7);ax.set_xlabel('Epoch (H100 shared prefix; continuation to H200)');ax.set_title(title,fontsize=10);ax.grid(alpha=.15)
axes[0].legend(frameon=False)
fig.suptitle('N17 natural MovieLens, Adam lr=0.01: development only, four optimizer seeds',fontsize=11)
fig.tight_layout()
for suffix in ['png','pdf']:fig.savefig(OUT/('H200_development_trajectories.'+suffix),dpi=180,bbox_inches='tight')
f[['arm','seed','epoch','train_empirical_objective','selection_macro_auc','selection_train_p_LL']].to_csv(OUT/'H200_plot_data.csv',index=False)
