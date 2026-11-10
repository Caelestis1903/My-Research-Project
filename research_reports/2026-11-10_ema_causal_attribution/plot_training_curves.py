"""Plot the 20 completed trajectories from saved epoch scalars only."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CONDITIONS=(('movielens','clean'),('kuairand','clean'),('movielens','gaussian_d0'),('kuairand','gaussian_d0'))
SEEDS=tuple(range(2022,2027))
T4=2.7764451052


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/ema_causal_20261110/attempt2')
    parser.add_argument('--dest',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();out=args.out.resolve();dest=args.dest.resolve()
    if dest==out or out in dest.parents:raise ValueError('Output must be outside the experiment attempt')
    jobs=json.loads((out/'PREPARED.json').read_text())['jobs']
    expected={(ds,obj,s) for ds,obj in CONDITIONS for s in SEEDS}
    if len(jobs)!=20 or {(j['dataset'],j['objective'],j['seed']) for j in jobs}!=expected:
        raise ValueError('Exactly four conditions × five seeds required')
    records=[]
    for job in jobs:
        paths=sorted((out/'runs'/job['job_id']/'epochs').glob('E*.json'))
        if [p.name for p in paths]!=[f'E{e:03d}.json' for e in range(1,51)]:
            raise ValueError('Fifty complete epoch files required: '+job['job_id'])
        for epoch,path in enumerate(paths,1):
            raw=path.read_bytes();source=json.loads(raw)
            for key in ('job_id','dataset','objective','seed'):
                if source[key]!=job[key]:raise ValueError('Epoch identity mismatch: '+str(path))
            if source['epoch']!=epoch or source['global_step']!=epoch*job['config']['steps_per_epoch']:
                raise ValueError('Epoch/global step mismatch: '+str(path))
            row={k:source[k] for k in ('dataset','objective','seed','epoch','global_step','training_loss','clean_loss')}
            for state in ('O','F'):
                for metric in ('macro_auc','macro_logloss'):
                    row[f'{state}_validation_{metric}']=source['validation'][state][metric]
            for key,value in row.items():
                if key not in ('dataset','objective') and (not isinstance(value,(int,float)) or not math.isfinite(value)):
                    raise ValueError('Missing/nonfinite scalar: '+str(path)+' '+key)
            row.update(source_record=str(path.relative_to(out)),source_sha256=hashlib.sha256(raw).hexdigest())
            records.append(row)
    assert len(records)==1000
    dest.mkdir(parents=True,exist_ok=True)
    with gzip.open(dest/'TRAINING_CURVES.csv.gz','wt',newline='') as stream:
        writer=csv.DictWriter(stream,list(records[0]));writer.writeheader();writer.writerows(records)
    fig,axes=plt.subplots(4,2,figsize=(15,16),sharex=True,sharey=False)
    epochs=np.arange(1,51)
    for row_number,(dataset,objective) in enumerate(CONDITIONS):
        by={(r['seed'],r['epoch']):r for r in records if r['dataset']==dataset and r['objective']==objective}
        for column,fields in enumerate((('training_loss','clean_loss'),('O_validation_macro_logloss','F_validation_macro_logloss'))):
            ax=axes[row_number,column]
            for field,color,style in zip(fields,('tab:blue','tab:orange'),('-','--')):
                values=np.array([[by[seed,e][field] for e in epochs] for seed in SEEDS],dtype=float)
                mu=values.mean(axis=0);half=T4*values.std(axis=0,ddof=1)/math.sqrt(5)
                ax.plot(epochs,mu,color=color,linestyle=style,label=field,linewidth=1.6)
                ax.fill_between(epochs,mu-half,mu+half,color=color,alpha=.14)
            population='Training population / source objective' if column==0 else 'Validation population / six-cell equal macro'
            ax.set_title(f'{dataset} | {objective}\n{population}',fontsize=11)
            ax.set_ylabel('Saved source loss (unchanged scale)' if column==0 else 'Validation macro logloss')
            ax.grid(alpha=.2);ax.legend(fontsize=9)
            if row_number==3:ax.set_xlabel('Epoch')
    fig.suptitle('Saved training and validation curves | 5 seeds per condition',fontsize=15)
    fig.text(.5,.015,'Different populations and objectives: columns are NOT a common-loss comparison.\n'
             'Raw values; both source-loss identities retained even when equal. Bands: per-epoch 95% t4 CI across five seeds.',
             ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.97));fig.savefig(dest/'training_loss.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(status='EXPORTED',jobs=20,epochs=1000,seeds=list(SEEDS),t4=T4,
        csv=str(dest/'TRAINING_CURVES.csv.gz'),figure=str(dest/'training_loss.png'))))


if __name__=='__main__':main()
