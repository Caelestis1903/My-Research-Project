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
    all_pairs=pd.read_csv(source/'paired_comparisons.csv').fillna({'tag':''})
    p=all_pairs[(all_pairs.stage=='selected')&(all_pairs.tag=='')]
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
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    labels=[]
    for row,(tag,method,control,label) in enumerate([
        ('','profile_axis','audit_fixed','Additive truth: axis / independent composite'),
        ('','profile_joint','audit_joint','Additive truth: axis / independent joint'),
        ('noise_interaction1','profile_axis','audit_fixed','Non-additive truth: axis / independent composite'),
        ('noise_interaction1','profile_joint','audit_joint','Non-additive truth: axis / independent joint')]):
        labels.append(label)
        for ax,metric in zip(axes,['macro_auc','macro_logloss']):
            g=all_pairs[(all_pairs.suite=='noise')&(all_pairs.stage=='selected')&(all_pairs.tag==tag)&(all_pairs.method==method)&(all_pairs.control==control)&(all_pairs.metric==metric)]
            if len(g):
                r=g.iloc[0];mean=r['mean']*1000;lo=r.ci95_low*1000;hi=r.ci95_high*1000
                ax.errorbar(mean,row,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=3,color='#246b8e' if not tag else '#b65c36')
    for ax,label in zip(axes,['AUC difference x1000 (right favors axis)','LogLoss difference x1000 (left favors axis)']):
        ax.axvline(0,color='0.5',lw=1);ax.set_xlabel(label);ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(4),labels,fontsize=8);axes[0].invert_yaxis()
    fig.suptitle(f'{phase}: fixed audit draw, four optimization seeds; axis assumption falsification')
    save(fig,target,'channel_assumption_falsification')
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True);labels=[]
    cases=[('kuairand','','KR Adam'),('kuairand','optimizer_adamw','KR AdamW'),('movielens','','ML Adam')]
    for row,(dataset,tag,label,method,control) in enumerate([
        (d,t,l,m,c) for m,c in [('rank_protected','rank_uniform'),('cell_codebook','domain_codebook')] for d,t,l in cases]):
        labels.append(f'{label}: {method} / {control}')
        for ax,metric in zip(axes,['macro_auc','macro_logloss']):
            g=all_pairs[(all_pairs.suite=='natural')&(all_pairs.dataset==dataset)&(all_pairs.tag==tag)&(all_pairs.stage=='selected')&(all_pairs.method==method)&(all_pairs.control==control)&(all_pairs.metric==metric)&(all_pairs.n_seeds==4)]
            if len(g):
                r=g.iloc[0];mean=r['mean']*1000;lo=r.ci95_low*1000;hi=r.ci95_high*1000
                ax.errorbar(mean,row,xerr=[[mean-lo],[hi-mean]],fmt='s' if tag else 'o',capsize=3,color='#246b8e' if dataset=='kuairand' else '#b65c36')
    axes[0].set_yticks(range(len(labels)),labels,fontsize=8);axes[0].invert_yaxis()
    for ax,label in zip(axes,['AUC difference x1000 (right favors candidate)','LogLoss difference x1000 (left favors candidate)']):
        ax.axvline(0,color='0.5',lw=1);ax.set_xlabel(label);ax.grid(axis='x',alpha=.2)
    fig.suptitle(f'{phase}: capacity mechanisms within optimizer, four-seed descriptive 95% t intervals')
    save(fig,target,'capacity_mechanism_effects')
    finite=OUT/'dynamics'/'rank_finite_removal.csv'
    if finite.exists():
        f=pd.read_csv(finite).fillna({'tag':''})
        f=f[f.stage.isin(['epoch1','epoch5','epoch10','final'])]
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        colors={'fixed':'#246b8e','rank_uniform':'#b65c36','cell_codebook':'#47824a'}
        for col,dataset in enumerate(['kuairand','movielens']):
            for method in colors:
                for tag,style in [('', '-'),('optimizer_adamw','--')]:
                    g=f[(f.dataset==dataset)&(f.method==method)&(f.tag==tag)]
                    if not len(g):continue
                    g=g.groupby('epoch')[['shared_weight_frobenius_norm_float64','probability_absolute_change']].mean()
                    label=method+(' AdamW' if tag else ' Adam')
                    axes[0,col].plot(g.index,np.maximum(g.shared_weight_frobenius_norm_float64,1e-40),style,color=colors[method],label=label,marker='o',markersize=3)
                    axes[1,col].plot(g.index,g.probability_absolute_change,style,color=colors[method],label=label,marker='o',markersize=3)
            axes[0,col].set_yscale('log');axes[0,col].set_title(dataset)
            axes[0,col].set_ylabel('Shared STAR weight norm (floor 1e-40)')
            axes[1,col].set_ylabel('Mean absolute prediction change after slot removal')
            for ax in axes[:,col]:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=6)
        fig.suptitle('Checkpoint capacity activity on fixed train-meta rows; conditional sensitivity, no retraining')
        fig.tight_layout();save(fig,target,'capacity_activity')
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
