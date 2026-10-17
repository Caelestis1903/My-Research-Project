"""Final primary N1 figures from globally unfrozen, completed test readouts only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import n17_common as c
from n17_performance import curve


def interval(values):
    values=np.asarray(values,dtype=float);mean=float(values.mean());width=float(t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    return mean,width


def main():
    p=argparse.ArgumentParser();p.add_argument('--comparisons',type=Path,required=True);p.add_argument('--performance',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    receipt=json.loads((a.performance/'receipt.json').read_text());assert receipt['status']=='COMPLETE' and receipt['comparison_sha']==c.sha(a.comparisons)
    pairs=json.loads(a.comparisons.read_text())['comparisons'];raw=pd.read_csv(a.performance/'paired_seed_endpoints.csv')
    a.out.mkdir(parents=True,exist_ok=False);primary={opt:[x for x in pairs if x['group']==f'N1_common_{opt}' and x.get('budget',100)==100] for opt in ('adam','sgd')}
    assert all(len(rows)==8 and len({x['seed'] for x in rows})==8 for rows in primary.values())
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True);cellnames=[f'd{d}t{task}_auc' for d in range(3) for task in range(2)]
    for ax,(opt,pairlist) in zip(axes,primary.items()):
        conf=json.loads((Path(pairlist[0]['SN'])/'config.json').read_text());part=raw[(raw.group==f'N1_common_{opt}')&(raw.readout=='selected')]
        labels=[]
        for index,metric in enumerate(cellnames):
            data=part[part.metric==metric].sort_values('seed');assert len(data)==8;mean,width=interval(data.difference)
            ax.errorbar(mean,index,xerr=width,fmt='o',color=f'C{index//2}',capsize=3)
            d=index//2;labels.append(f'd{d}/t{index%2} (train N={conf["data"]["train_domain_counts"][str(d)]})')
        ax.axvspan(-.003,.003,color='grey',alpha=.12);ax.axvline(0,color='black',lw=.7);ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_title(opt);ax.set_xlabel('SB minus SN AUC; pointwise paired-seed 95% t interval');ax.grid(axis='x',alpha=.2)
    fig.suptitle('H100 selection-chosen models; historical test reused, n=8 seeds');fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(a.out/f'primary_six_cells.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for index,(opt,pairlist) in enumerate(primary.items()):
        for arm,color in [('SN','tab:blue'),('SB','tab:orange')]:
            frames=[]
            for pair in sorted(pairlist,key=lambda x:x['seed']):
                run=Path(pair[arm]);assert json.loads((run/'test_receipt.json').read_text())['status']=='COMPLETE';frames.append(curve(run)[0].loc[1:100])
            for col,metric in enumerate(('macro_auc','train_J')):
                values=np.stack([frame[metric].to_numpy() for frame in frames]);mean=values.mean(0);width=t.ppf(.975,7)*values.std(0,ddof=1)/np.sqrt(8);x=np.arange(1,101);ax=axes[index,col]
                ax.plot(x,mean,label=arm,color=color);ax.fill_between(x,mean-width,mean+width,alpha=.12,color=color);ax.set_title(f'{opt}: '+('test macro AUC' if col==0 else 'full training J'));ax.grid(alpha=.2);ax.legend()
        for ax in axes[index]:ax.set_xlabel('epoch (49 updates per KR epoch)')
    fig.suptitle('Frozen H100 common eta: pointwise seed intervals, not an onset test');fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(a.out/f'primary_budget_curves.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True);weightrows=[]
    for ax,opt in zip(axes,('adam','sgd')):
        part=raw[(raw.group==f'N1_common_{opt}')&(raw.readout=='selected')];curves=[];alpha=np.linspace(0,1,101)
        for seed,sub in part.groupby('seed'):
            macro=float(sub[sub.metric=='macro_auc'].difference.iloc[0]);weighted=float(sub[sub.metric=='evaluation_p_cell_auc'].difference.iloc[0]);values=(1-alpha)*macro+alpha*weighted;curves.append(values);ax.plot(alpha,values,color='grey',alpha=.3,lw=.6)
            weightrows.extend(dict(optimizer=opt,seed=int(seed),alpha=float(x),difference=float(y)) for x,y in zip(alpha,values))
        ax.plot(alpha,np.mean(curves,axis=0),color='black',lw=2,label='seed mean');ax.axhline(0,color='black',lw=.6);ax.set_title(opt);ax.set_xlabel('alpha: uniform cells to test-count cell weights');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('SB minus SN AUC on identical selected predictions');fig.suptitle('Prespecified evaluation-weight sensitivity; primary metric stays macro');fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(a.out/f'primary_weight_sensitivity.{ext}',dpi=160)
    plt.close(fig);pd.DataFrame(weightrows).to_csv(a.out/'primary_weight_sensitivity.csv',index=False)
    c.dump(a.out/'receipt.json',dict(status='COMPLETE',source_sha=c.sha(__file__),comparison_sha=c.sha(a.comparisons),performance_receipt_sha=c.sha(a.performance/'receipt.json'),
        scope='n=8 paired training seeds; descriptive pointwise figures; no new model choice, no causal domain-size regression, no selection of evaluation weights'))


if __name__=='__main__':main()
