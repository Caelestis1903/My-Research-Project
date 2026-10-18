"""Export standalone research figures from frozen aggregate tables."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from n19_data import OUT

PANELS=[('P1','stability','movielens','task_dro','task_mask'),
 ('P2','noise','movielens','joint_axis','nested_axis'),
 ('P3 KR','natural','kuairand','rank_protected','rank_uniform'),
 ('P3 ML','natural','movielens','cell_codebook','domain_codebook'),
 ('P4 KR','natural','kuairand','reliability_interaction','interaction'),
 ('P4 ML','natural','movielens','reliability_interaction','interaction'),
 ('P5 KR','natural','kuairand','coalition_router','utility_router'),
 ('P5 ML','natural','movielens','coalition_router','utility_router'),
 ('P6','sampling','movielens','mask_copula_cv','mask_independent_cv')]

def save(fig,target,name):
    fig.savefig(target/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(target/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def main(phase):
    source=OUT/(phase+'_readout');target=OUT/'figures'/phase;target.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(source/'paired_comparisons.csv');p=p[(p.stage=='selected')&(p.tag.fillna('')=='')]
    fig,axes=plt.subplots(1,2,figsize=(11,6),sharey=True);labels=[]
    for row,(name,suite,dataset,method,control) in enumerate(PANELS):
        labels.append(f'{name}: {method} / {control}')
        for ax,metric,scale,label in zip(axes,['macro_auc','macro_logloss'],[1000,1000],['AUC difference x1000 (right favors candidate)','LogLoss difference x1000 (left favors candidate)']):
            g=p[(p.suite==suite)&(p.dataset==dataset)&(p.method==method)&(p.control==control)&(p.metric==metric)]
            if len(g):
                r=g.iloc[0];mean=r['mean']*scale;lo=r.ci95_low*scale;hi=r.ci95_high*scale
                if np.isfinite(lo) and np.isfinite(hi):ax.errorbar(mean,row,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=3,color='#246b8e')
                else:ax.plot(mean,row,'o',color='#246b8e')
            ax.set_xlabel(label);ax.axvline(0,color='0.55',lw=1);ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(labels)),labels,fontsize=8);axes[0].invert_yaxis()
    fig.suptitle(f'{phase}: selected checkpoints, paired seed means and descriptive 95% t intervals')
    save(fig,target,'paired_effects')
    cells=pd.read_csv(source/'paired_cells.csv');cells=cells[(cells.stage=='selected')&(cells.metric=='auc')&(cells.tag.fillna('')=='')]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    colors={'kuairand':'#246b8e','movielens':'#b65c36'}
    markers={0:'o',1:'^'}
    for _,suite,dataset,method,control in PANELS:
        g=cells[(cells.suite==suite)&(cells.dataset==dataset)&(cells.method==method)&(cells.control==control)]
        if suite!='natural':continue
        for _,r in g.iterrows():
            for ax,x in zip(axes,[r.train_n,r.baseline]):ax.scatter(x,r['mean']*1000,c=colors[dataset],marker=markers[r.task],alpha=.6,s=35)
    axes[0].set_xscale('log');axes[0].set_xlabel('Main-train domain rows (log scale)');axes[1].set_xlabel('Control cell AUC (difficulty proxy)')
    for ax in axes:ax.axhline(0,color='0.5',lw=1);ax.set_ylabel('Paired cell AUC difference x1000');ax.grid(alpha=.2)
    fig.suptitle('Domain size and baseline difficulty are separate descriptors; points are correlated')
    save(fig,target,'size_and_difficulty')
    dyn=OUT/'dynamics'/'training_trajectories.csv'
    if dyn.exists():
        frame=pd.read_csv(dyn);frame=frame[frame.tag.fillna('')==''];fig,axes=plt.subplots(2,3,figsize=(13,7))
        choices=[('P1','stability','movielens',['fixed','task_cf','task_dro','task_mask']),('P2','noise','movielens',['naive','audit_fixed','nested_axis','joint_axis']),('P3','natural','kuairand',['fixed','rank_uniform','rank_protected','cell_codebook']),('P4','natural','kuairand',['fixed','domain_mix','interaction','reliability_interaction']),('P5','natural','kuairand',['full_mix','utility_router','regret_router','coalition_router']),('P6','sampling','movielens',['mask_independent','mask_antithetic','mask_copula','mask_independent_cv','mask_copula_cv'])]
        for ax,(name,suite,dataset,methods) in zip(axes.flat,choices):
            for method in methods:
                g=frame[(frame.suite==suite)&(frame.dataset==dataset)&(frame.method==method)]
                if not len(g):continue
                summary=g.groupby('epoch').development_auc.agg(['mean','std']);x=summary.index.to_numpy();y=summary['mean'].to_numpy();sd=summary['std'].fillna(0).to_numpy()
                ax.plot(x,y,label=method,lw=1.2);ax.fill_between(x,y-sd,y+sd,alpha=.12)
            ax.set_title(f'{name}: {dataset}');ax.set_xlabel('Epoch');ax.set_ylabel('Development macro AUC');ax.legend(fontsize=6);ax.grid(alpha=.2)
        fig.suptitle('Complete training trajectories: means and seed SD, no early stopping');fig.tight_layout()
        save(fig,target,'training_dynamics')
    print('Figures:',target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','test'],default='test');a=p.parse_args();main(a.phase)
