"""Standalone scientific figures from final frozen readout tables."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cm19_train import OUT


def save(fig,name,target):
    target.mkdir(exist_ok=True);fig.savefig(target/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(target/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['development','test'],default='test');args=parser.parse_args()
    rr=OUT/(args.split+'_readout');destination=OUT/('figures' if args.split=='test' else 'development_figures');paths=pd.read_csv(rr/'path_diagnostics.csv.gz');mm=pd.read_csv(rr/'seed_macro_metrics.csv');pairs=pd.read_csv(rr/'paired_comparisons.csv');pc=pd.read_csv(rr/'paired_cell_deltas.csv');cc=pd.read_csv(rr/'seed_cell_metrics.csv')
    designs=[('star_weight_paths',[('fixed','modulation'),('original','modulation'),('original_sample','modulation'),('star0','modulation'),('star1','modulation')]),('other_module_paths',[('pepnet','ep'),('pepnet','pp'),('hamur0','adapter'),('hamur1','adapter'),('mmoe','shared'),('ple','shared')])]
    for name,items in designs:
        fig,axes=plt.subplots(len(items),2,figsize=(10,2.1*len(items)),squeeze=False,sharex=True)
        for row,(model,kind) in enumerate(items):
            for col,dataset in enumerate(['kuairand','movielens']):
                ax=axes[row,col];df=paths[(paths.stage=='G0')&(paths.model==model)&(paths.dataset==dataset)&(paths.intervention==kind)&(paths.probe=='validation')&paths.label.str.startswith('step')]
                # Cell macro average within each seed and checkpoint; seed is replication unit.
                df=df.groupby(['optimizer','seed','step']).delta_p.mean().reset_index()
                for opt,color in [('adam','#1f77b4'),('adamw','#d95f02')]:
                    sub=df[df.optimizer==opt].groupby('step').delta_p.agg(['mean','min','max'])
                    xx=sub.index.to_numpy();yy=sub['mean'].to_numpy();ax.plot(xx,np.maximum(yy,1e-10),label=opt,color=color)
                    ax.fill_between(xx,np.maximum(sub['min'].to_numpy(),1e-10),np.maximum(sub['max'].to_numpy(),1e-10),alpha=.15,color=color)
                ax.set_xscale('symlog',linthresh=10);ax.set_yscale('log');ax.set_title(f'{dataset} / {model} / {kind}',fontsize=10);ax.set_ylabel('mean |p - neutral p|');ax.grid(alpha=.2)
                if row==len(items)-1:ax.set_xlabel('primary optimizer updates')
                if row==0 and col==0:ax.legend(fontsize=8)
        fig.suptitle('Validation probe sensitivity; shade: four-seed range; zeros plotted at 1e-10\nDifferent interventions are not a common cross-model scale',fontsize=11,y=1.01);fig.tight_layout();save(fig,name,destination)
    # Selected versus final optimizer performance, without treating final as deployed selection.
    data=pairs[(pairs.comparison.str.startswith('G0_optimizer/'))&(pairs.metric=='macro_auc')]
    models=['fixed','original','original_sample','star0','star1','pepnet','hamur0','hamur1','mmoe','ple'];fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,ds in zip(axes,['kuairand','movielens']):
        for offset,label,color in [(-.13,'selected','#1f77b4'),(.13,'final','#d95f02')]:
            sub=data[(data.checkpoint==label)&data.comparison.str.contains('/'+ds+'/')].copy();sub['model']=sub.comparison.str.split('/').str[-1];sub=sub.set_index('model').reindex(models);yy=np.arange(len(models))+offset
            ax.errorbar(sub.mean_delta,yy,xerr=np.array([sub.mean_delta-sub.ci95_low,sub.ci95_high-sub.mean_delta]),fmt='o',markersize=4,capsize=2,label=label,color=color)
        ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(models)),models);ax.set_title(ds);ax.set_xlabel('AdamW - Adam macro AUC (descriptive seed CI)');ax.grid(axis='x',alpha=.2);ax.legend()
    fig.suptitle(args.split+' AUC; ML nominal AdamW shrink multiplier rounds to 1.0',fontsize=10);fig.tight_layout();save(fig,'optimizer_selected_final_auc',destination)
    # Mechanism-specific paired differences for all domains/tasks; no collapsed average.
    for ds in ['kuairand','movielens']:
        df=pc[(pc.comparison.str.startswith('G1/'+ds+'/fixed/'))&(pc.checkpoint=='selected')]
        if df.empty:continue
        pivot=df.groupby(['comparison','domain','task']).delta_auc.mean().unstack(['domain','task']);fig,ax=plt.subplots(figsize=(9,4));lim=max(float(np.abs(pivot.to_numpy()).max()),1e-4)
        im=ax.imshow(pivot.to_numpy(),cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(len(pivot.columns)),[f'd{d}/t{t}' for d,t in pivot.columns]);ax.set_yticks(range(len(pivot)),[x.split('/')[5] for x in pivot.index]);ax.set_title(f'{ds}: fixed M3oE selected {args.split} AUC change vs native Adam');fig.colorbar(im,ax=ax,label='paired four-seed mean delta AUC');fig.tight_layout();save(fig,'local_decay_cells_'+ds,destination)
    # Unique-N intervention: matched per-domain exposures and optimizer updates.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ds,target in zip(axes,['kuairand','movielens'],[1,2]):
        for model,color in [('fixed','#1f77b4'),('star0','#d95f02'),('star1','#2ca02c')]:
            df=cc[(cc.dataset==ds)&(cc.model==model)&(cc.optimizer=='adam')&(cc.domain==target)&(cc.checkpoint=='selected')&(cc.intervention=='none')&(((cc.stage=='G4')&(cc.schedule=='matched_updates'))|(cc.stage=='G0'))]
            points=df.groupby(['fraction','seed']).auc.mean().reset_index().groupby('fraction').auc.agg(['mean','std','count']);ax.errorbar(points.index,points['mean'],yerr=3.182446*points['std']/np.sqrt(points['count']),fmt='o-',label={'fixed':'fixed (two-task mean)','star0':'STAR task 0','star1':'STAR task 1'}[model],color=color,capsize=2)
        ax.set_title(f'{ds}: original physical domain {target}');ax.set_xlabel('fraction of unique training rows in target domain');ax.set_ylabel('target-domain selected task-average AUC');ax.grid(alpha=.2);ax.legend()
    fig.suptitle(args.split+': within-model size response; task sets differ across curves\nMatched exposures; one nested subset/fraction; target remains largest',fontsize=10);fig.tight_layout();save(fig,'controlled_unique_domain_size',destination)
    print('Wrote standalone PNG/PDF scientific figures.')
if __name__=='__main__':main()
