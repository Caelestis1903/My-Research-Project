"""Publication artifacts from completed four-training-seed tables only."""
from pathlib import Path
import os
import csv
import numpy as np

REPORT=Path(__file__).resolve().parent
(REPORT/'figures').mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(REPORT/'figures'/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(name):
    return list(csv.DictReader((REPORT/'tables'/name).open()))


def interval(ax,rows,labels,color):
    means=np.array([float(r['mean_delta']) for r in rows])
    low=np.array([float(r['t95_low']) for r in rows])
    high=np.array([float(r['t95_high']) for r in rows])
    y=np.arange(len(rows))
    ax.errorbar(means,y,xerr=np.vstack([means-low,high-means]),fmt='o',color=color,capsize=3)
    ax.axvline(0,color='0.45',lw=.8)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)


def plot_core_roles():
    rows=load('CORE_four_seed_paired_summary.csv')
    pairs=[('PILOT_PROB','H_DIRECT'),('RCF','H_DIRECT'),('RCF','POOLED_TARGET'),
           ('RCF','LOCAL_ACTIVE'),('RCF','NATIVE')]
    roles=['Q_explore','test_exposed','Q_legacy_confirm_exposed']
    colors=['#285b8e','#b65e32','#357563']
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,dataset,title in zip(axes,['movielens','kuairand'],['MovieLens','KuaiRand']):
        for offset,role,color in zip([-.2,0,.2],roles,colors):
            selected=[next(r for r in rows if r['dataset']==dataset and r['candidate']==c and
                r['reference']==ref and r['role']==role and r['training_stage']=='selected' and
                r['level']=='macro' and r['metric']=='logloss') for c,ref in pairs]
            if any(int(r['n_training_seeds'])!=4 for r in selected):raise RuntimeError('Four seeds required')
            mean=np.array([float(r['mean']) for r in selected]);low=np.array([float(r['t95_low']) for r in selected]);high=np.array([float(r['t95_high']) for r in selected])
            ax.errorbar(mean,np.arange(len(pairs))+offset,xerr=[mean-low,high-mean],fmt='o',color=color,
                        capsize=2,label=role.replace('Q_legacy_confirm_exposed','legacy'))
        ax.axvline(0,color='0.45',lw=.8);ax.grid(axis='x',alpha=.2)
        ax.set_yticks(np.arange(len(pairs)),[c+' - '+ref for c,ref in pairs]);ax.set_title(title)
        ax.set_xlabel('Paired Macro LogLoss difference (lower is better)')
    axes[0].invert_yaxis();axes[1].legend(fontsize=8,loc='best')
    fig.suptitle('Frozen policies across roles: four training seeds, paired t95 (df=3)')
    fig.text(.5,.01,'Exposed test/legacy are diagnostics; they do not alter the registered Q decision or model selection.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.95))
    for suffix in ('png','pdf'):fig.savefig(REPORT/'figures'/('CORE_roles_four_seed_intervals.'+suffix),dpi=180)
    plt.close(fig)


def main():
    rows=load('S2_Q_four_seed_paired_summary.csv')
    refs=['H_DIRECT','FULL_SOURCE_ALL','POOLED_TARGET','LOCAL_ACTIVE','PILOT_PROB','NATIVE']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,dataset,title in zip(axes,['movielens','kuairand'],['MovieLens','KuaiRand']):
        selected=[next(r for r in rows if r['dataset']==dataset and r['reference']==ref and r['metric']=='macro_logloss') for ref in refs]
        if any(int(r['n_training_seeds'])!=4 for r in selected):raise RuntimeError('Four training seeds required')
        interval(ax,selected,refs,'#285b8e');ax.axvline(-.0005,color='#b65e32',ls='--',lw=.9)
        ax.set_title(title);ax.set_xlabel('RCF minus reference: Macro LogLoss (lower is better)')
    fig.suptitle('Q_explore: four training seeds; mean and paired t95 (df=3)')
    fig.text(.5,.01,'Dashed line: registered practical mean threshold; all-seed direction and cell/AUC protections also required.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.95))
    for suffix in ('png','pdf'):fig.savefig(REPORT/'figures'/('RCF_Q_four_seed_intervals.'+suffix),dpi=180)
    plt.close(fig)
    if (REPORT/'tables/CORE_four_seed_paired_summary.csv').exists():plot_core_roles()
    scale=load('SCALE_four_seed_paired_summary.csv')
    metrics=['domain0_macro_auc','domain1_macro_auc','domain2_macro_auc','macro_auc','sample_weighted_cell_auc','pooled_task_mean_auc']
    labels=['Domain 0','Domain 1','Domain 2','Macro','Sample-weighted cells','Pooled tasks']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,dataset,title in zip(axes,['movielens','kuairand'],['MovieLens','KuaiRand']):
        selected=[next(r for r in scale if r['dataset']==dataset and r['role']=='test_exposed' and r['candidate']=='Dconst01_NO_AUG' and r['reference']=='B' and r['metric']==metric) for metric in metrics]
        if any(int(r['n_training_seeds'])!=4 for r in selected):raise RuntimeError('Four training seeds required')
        interval(ax,selected,labels,'#357563');ax.set_title(title);ax.set_xlabel('NO_AUG minus B: AUC (higher is better)')
    fig.suptitle('Historical SCALE protocol: exposed test, four training seeds, paired t95')
    fig.text(.5,.01,'Domain size is observational; macro, sample-weighted cell and pooled AUC are distinct estimands.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.95))
    for suffix in ('png','pdf'):fig.savefig(REPORT/'figures'/('SCALE_domain_population_four_seed_intervals.'+suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
