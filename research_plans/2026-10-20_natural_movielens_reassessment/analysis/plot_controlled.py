"""Export completed S5/S6 population-stage comparisons with four-seed intervals."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent

def plot(stage):
    f=pd.read_csv(ROOT/'controlled_history/paired_summary.csv')
    f=f[(f.stage==stage)&(f.task==-1)]
    groups=[('kuairand_natural_historical','historical_full50','Historical natural KuaiRand','#8b6c42'),
        ('movielens_artificial_historical','historical_full50','Historical artificial MovieLens','#396ab1'),
        ('movielens_natural','matched250','Natural MovieLens, 250 main updates','#3e9651'),
        ('movielens_natural','full50','Natural MovieLens, 1000 main updates','#9b4f96')]
    groups=[x for x in groups if ((f.population==x[0])&(f.budget==x[1])).any()]
    contrasts=['T-F','HT-F','DH-F','DH-HT'];fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,ck in enumerate(['selected','final']):
        for row,(metric,label) in enumerate([('macro_auc','Delta macro AUC (higher is better)'),('macro_logloss','Delta macro LogLoss (lower is better)')]):
            ax=axes[row,col];ax.axhline(0,color='black',lw=.8)
            for k,(pop,budget,name,color) in enumerate(groups):
                x=[];y=[];lo=[];hi=[]
                for i,contrast in enumerate(contrasts):
                    g=f[(f.population==pop)&(f.budget==budget)&(f.checkpoint==ck)&(f.contrast==contrast)&(f.metric==metric)]
                    if g.empty:continue
                    assert len(g)==1 and g.iloc[0]['n']==4
                    r=g.iloc[0];x.append(i+(k-(len(groups)-1)/2)*.15);y.append(r['mean'])
                    lo.append(r['mean']-r.ci95_low);hi.append(r.ci95_high-r['mean'])
                ax.errorbar(x,y,yerr=[lo,hi],fmt='o',ms=4,capsize=2,color=color,label=name)
            ax.set_xticks(range(4),contrasts);ax.set_ylabel(label);ax.set_title(ck.capitalize());ax.grid(axis='y',alpha=.18)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=9)
    fig.suptitle(stage+' history/current controls: completed cohorts only\nPointwise t95 across four optimizer seeds; populations and targets differ',fontsize=12)
    out=ROOT/'core'/stage;out.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(out/('controlled_population_stage.'+suffix),dpi=180)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['AC-S5','AC-S6'],required=True);plot(p.parse_args().stage)
