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

def plot_cells(stage):
    source=ROOT/'core'/stage/'paired_cell_deltas.csv'
    if not source.exists():return
    f=pd.read_csv(source);f=f[f.checkpoint.isin(['selected','final'])]
    budgets=[b for b in ('matched250','full50') if b in set(f.budget)]
    contrasts=[c for c in ('T-F','HT-F','DH-F','DH-HT') if c in set(f.contrast)]
    cells=[(d,t) for d in range(3) for t in range(2)]
    table=f.groupby(['budget','checkpoint','contrast','domain','task']).auc.agg(['mean',lambda v:int((v>0).sum())])
    bound=max(.01,float(np.max(np.abs(table['mean']))));bound=np.ceil(bound*100)/100
    fig,axes=plt.subplots(len(budgets),2,squeeze=False,figsize=(12,3.5*len(budgets)),layout='constrained')
    for i,budget in enumerate(budgets):
        for j,checkpoint in enumerate(('selected','final')):
            values=np.array([[table.loc[(budget,checkpoint,c,d,t),'mean'] for d,t in cells] for c in contrasts])
            ax=axes[i,j];im=ax.imshow(values,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
            for row,c in enumerate(contrasts):
                for col,(d,t) in enumerate(cells):
                    item=table.loc[(budget,checkpoint,c,d,t)];n=int(item.iloc[1]);value=item['mean']
                    ax.text(col,row,f'{value:+.4f}\n+{n}/4',ha='center',va='center',fontsize=8,
                        color='white' if abs(value)>.6*bound else 'black')
            ax.set_xticks(range(6),[f'd{d}t{t}' for d,t in cells]);ax.set_yticks(range(len(contrasts)),contrasts)
            ax.set_title(budget+' / '+checkpoint)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Mean paired delta AUC',shrink=.85)
    fig.suptitle(stage+' natural MovieLens: cell effects and seed consistency\nSecond line: number of positive paired optimizer seeds / 4; not population replication',fontsize=11)
    for suffix in ('png','pdf'):fig.savefig(ROOT/'core'/stage/('controlled_cell_effects.'+suffix),dpi=180)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['AC-S5','AC-S6'],required=True);stage=p.parse_args().stage;plot(stage);plot_cells(stage)
