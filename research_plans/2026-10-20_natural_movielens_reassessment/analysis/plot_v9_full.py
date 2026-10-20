"""Full frozen V9 curves and preregistered arithmetic AULC windows."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_core import interval

OUT=Path(__file__).resolve().parent/'v9_history'
f=pd.read_csv(OUT/'all_epoch_cell_metrics.csv')
assert set(f.population)=={'movielens_natural','movielens_artificial_historical'}
macro=f.groupby(['population','arm','seed','epoch'])[['auc','logloss']].mean().reset_index()
paired=pd.read_csv(OUT/'paired_seed_deltas.csv')
fig,axes=plt.subplots(2,2,figsize=(12,7.8))
pops=['movielens_artificial_historical','movielens_natural']
for ax,pop,steps,title in zip(axes[0],pops,[5,20],['Historical artificial ML: 5 updates / epoch','Natural ML: 20 updates / epoch']):
    for arm,color in [('SN','#2563eb'),('SB','#d97706')]:
        tab=macro[(macro.population==pop)&(macro.arm==arm)].pivot(index='epoch',columns='seed',values='auc')
        assert tab.shape==(50,4)
        x=tab.index.to_numpy()*steps;m=tab.mean(axis=1).to_numpy();sd=tab.std(axis=1,ddof=1).to_numpy()
        ax.plot(x,m,label=arm,color=color,lw=1.7);ax.fill_between(x,m-sd,m+sd,color=color,alpha=.13)
    ax.set_title(title);ax.set_xlabel('Actual main updates');ax.set_ylabel('Test macro AUC');ax.legend(frameon=False);ax.grid(alpha=.15)
for ax,metric,title in zip(axes[1],['auc','logloss'],['Window mean AUC: SB minus SN','Window mean LogLoss: SB minus SN']):
    for j,(pop,color,label) in enumerate(zip(pops,['#64748b','#0f766e'],['Historical artificial ML','Natural ML'])):
        for i,n in enumerate([10,16,20,50]):
            values=paired[(paired.population==pop)&(paired.readout=='AULC_1_'+str(n))].sort_values('seed')[metric].to_numpy()
            assert len(values)==4
            s=interval(values);x=i+(j-.5)*.22
            ax.scatter(x+np.linspace(-.045,.045,4),values,color=color,s=14,alpha=.45)
            ax.errorbar(x,s['mean'],yerr=[[s['mean']-s['ci95_low']],[s['ci95_high']-s['mean']]],
                color=color,fmt='o',capsize=3,label=label if i==0 else None)
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(4),['1-10','1-16','1-20','1-50'])
    ax.set_xlabel('Prespecified epoch window');ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=8)
fig.suptitle('V9: SGD .3 without momentum, fixed M3OE; complete 50-epoch readout',fontsize=13)
fig.text(.5,.012,'Top bands: optimizer-seed SD; bottom bars: pointwise paired t95 across four seeds.\n'
    'AULC is arithmetic epoch mean. Populations, support and updates per epoch differ; reused test is exploratory.\n'
    'Historical artificial subsets varied with seed; natural support is fixed across optimization seeds.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.10,1,.96])
for suffix in ('png','pdf'):fig.savefig(OUT/('v9_full_trajectory.'+suffix),dpi=180,bbox_inches='tight')
plt.close(fig)
