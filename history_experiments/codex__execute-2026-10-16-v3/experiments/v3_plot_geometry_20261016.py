"""Reproduce V1 full-coordinate fixed-theta geometry figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent.parent/'research_plans/2026-10-16_v3/execution_20261016'

def main():
 data=pd.read_csv(OUT/'analysis/V1_full_coordinate_geometry.csv');keys=['optimizer','seed','epoch','block']
 pair=data[data.arm=='SN'].merge(data[data.arm=='SB'],on=keys,suffixes=('_SN','_SB'),validate='one_to_one')
 metrics=('covariance_trace','effective_rank','anisotropy','top10_explained')
 for metric in metrics:pair[metric+'_ratio']=pair[metric+'_SB']/pair[metric+'_SN']
 pair.to_csv(OUT/'analysis/V1_paired_geometry.csv',index=False)
 fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
 for ax,metric,label in zip(axes.flat,metrics,('Covariance trace SB/SN','Effective rank SB/SN','Anisotropy SB/SN','Top-10 explained fraction SB/SN')):
  for opt,color in (('sgd','#2469ad'),('adam','#c45132')):
   points=pair[(pair.optimizer==opt)&(pair.block=='ALL')];line=points[points.seed==2022].sort_values('epoch')
   ax.plot(line.epoch,line[metric+'_ratio'],color=color,label=opt+' seed2022',marker='o')
   other=points[points.seed!=2022].sort_values('seed');ax.scatter(other.epoch+np.linspace(-.35,.35,len(other)),other[metric+'_ratio'],color=color,marker='x',alpha=.65)
  ax.axhline(1,color='gray',lw=.7,ls='--');ax.set(xlabel='Source checkpoint completed epoch',ylabel=label);ax.legend(frameon=False,fontsize=8)
 fig.suptitle('V1 fixed-theta probes | K=512 per arm, full coordinates\nLines: seed2022 stages; crosses: other seeds at epoch10',fontsize=10)
 for ext in ('png','pdf'):fig.savefig(OUT/'analysis'/('V1_geometry_ratios.'+ext),dpi=190)
 plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
 for column,opt in enumerate(('sgd','adam')):
  raw=json.loads((OUT/'audits'/f'V1__kuairand__{opt}__exact_epoch10__seed2022'/'geometry.json').read_text())
  for arm,color in (('SN','#2469ad'),('SB','#c45132')):
   row=next(r for r in raw if r['arm']==arm and r['block']=='ALL');values=np.asarray(row['eigenvalues'])[:511];x=np.arange(1,len(values)+1);positive=values>0;x=x[positive];values=values[positive]
   axes[0,column].loglog(x,values,color=color,label=arm);axes[1,column].loglog(x,values/row['covariance_trace'],color=color,label=arm)
  axes[0,column].set(title=opt+' | seed2022 epoch10',ylabel='Empirical covariance eigenvalue')
  axes[1,column].set(xlabel='Eigenvalue rank',ylabel='Eigenvalue / covariance trace')
  for ax in axes[:,column]:ax.legend(frameon=False)
 fig.suptitle('Full-coordinate spectra via exact 512 x 512 Gram | rank <=511',fontsize=10)
 for ext in ('png','pdf'):fig.savefig(OUT/'analysis'/('V1_full_coordinate_spectra.'+ext),dpi=190)
 plt.close(fig)
if __name__=='__main__':main()
