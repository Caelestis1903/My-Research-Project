"""Standalone scientific figures from the completed Cycle02 analysis tables."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ARMS = ('NQR-025', 'BRIER-OPPORTUNITY-025', 'MEAN-GAP-025')
LABELS = ('NQR', 'BRIER', 'MEAN-GAP')
DATASETS = ('movielens', 'kuairand')
COLORS = ('#2271b2', '#e69f00', '#8b5fbf')


def read(name):
    with (OUT/name).open(newline='') as f:
        return list(csv.DictReader(f))


def comparison(arm):
    return arm+'/fixed_ema minus CLEAN/fixed_ema'


def main():
    pairs = read('ALL_PAIRED_SEEDS.csv')
    summaries = read('ALL_PAIRED_SUMMARY.csv')
    lookup = {(r['dataset'], r['comparison'], r['endpoint'], r['metric']):r for r in summaries}
    fig, axes = plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for ax,dataset in zip(axes,DATASETS):
        for i,(arm,color) in enumerate(zip(ARMS,COLORS)):
            rows = sorted((r for r in pairs if r['dataset']==dataset and r['comparison']==comparison(arm)
                           and r['endpoint']=='selected' and r['metric']=='macro_auc'),key=lambda r:int(r['seed']))
            assert len(rows)==5
            values = [float(r['delta']) for r in rows]
            ax.scatter([i+(j-2)*.035 for j in range(5)],values,s=26,color=color,alpha=.7)
            r = lookup[dataset,comparison(arm),'selected','macro_auc']
            mean,lo,hi = (float(r[k]) for k in ('mean','ci95_low','ci95_high'))
            ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='D',color=color,capsize=5,markersize=6)
        ax.axhline(0,color='#555555',linewidth=.8)
        ax.set(xticks=range(3),xticklabels=LABELS,title=dataset,ylabel='Delta validation Macro AUC vs CLEAN EMA')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Selected fixed EMA: five paired seeds and descriptive 95% t intervals')
    fig.savefig(OUT/'PAIRED_AUC.png',dpi=200)
    plt.close(fig)
    cells = [f'd{d}t{t}' for d in range(3) for t in range(2)]
    arrays = [np.array([[float(lookup[d,comparison(a),'selected',c+'_auc']['mean']) for c in cells] for a in ARMS]) for d in DATASETS]
    limit = max(float(np.abs(x).max()) for x in arrays)
    fig,axes = plt.subplots(1,2,figsize=(12,3.6),layout='constrained')
    for ax,dataset,values in zip(axes,DATASETS,arrays):
        im = ax.imshow(values,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
        ax.set(xticks=range(6),xticklabels=cells,yticks=range(3),yticklabels=LABELS,title=dataset)
        for i in range(3):
            for j in range(6):
                ax.text(j,i,f'{values[i,j]:+.4f}',ha='center',va='center',fontsize=8,
                        color='white' if abs(values[i,j])>limit*.65 else 'black')
    fig.colorbar(im,ax=axes,label='Mean paired cell AUC delta vs CLEAN EMA',shrink=.8)
    fig.suptitle('Selected fixed EMA: six cells, equal means over five paired seeds')
    fig.savefig(OUT/'CELL_AUC_DELTAS.png',dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
