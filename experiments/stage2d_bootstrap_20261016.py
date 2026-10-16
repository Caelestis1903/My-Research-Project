"""Frozen-model row uncertainty, shared row weights across four training seeds."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'research_plans/2026-10-16/stage2d_execution_20261016'
SEEDS=[2022,2023,2024,2025];ARMS=['SN','SB','EN','EB'];CONTRASTS=[('SB','SN'),('EB','EN')]
REPLICATES=1000


def setup(y,p):
    order=np.argsort(p,kind='stable');s=p[order]
    return order,np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1],y[order]


def aucw(w,state,npos,nneg):
    order,starts,y=state;ww=w[order];pos=np.add.reduceat(ww*y,starts);neg=np.add.reduceat(ww*(1-y),starts)
    return np.sum(pos*(np.cumsum(neg)-.5*neg))/(npos*nneg)


def main():
    audit=json.loads((OUT/'analysis/receipt.json').read_text())
    assert audit['status']=='PASS_NUMERICAL_AUDIT_ONLY' and audit['training_runs_complete']==16
    target=OUT/'analysis';rows=[];saved={};rng=np.random.default_rng(20261016)
    started=time.time()
    for split in ['validation','test']:
        predictions={};identity=None
        for seed in SEEDS:
            for arm in ARMS:
                run=OUT/'runs'/f'seed{seed}'/arm/'attempt1'
                filename='validation_selected.npz' if split=='validation' else 'test_selected.npz'
                with np.load(run/filename,allow_pickle=False) as z:
                    ids=z['row_ids'];domains=z['domains'];labels=z['labels']
                    if identity is None:identity=(ids.copy(),domains.copy(),labels.copy())
                    else:assert all(np.array_equal(a,b) for a,b in zip(identity,(ids,domains,labels)))
                    predictions[seed,arm]=z['predictions'].astype(float)
        _,domains,labels=identity
        for d in range(3):
            for t in range(2):
                mask=domains==d;y=labels[mask,t].astype(float);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
                states={k:setup(y,p[mask,t]) for k,p in predictions.items()}
                loss={}
                for k,p in predictions.items():
                    pp=np.clip(p[mask,t],1e-12,1-1e-12)
                    loss[k]=-(y*np.log(pp)+(1-y)*np.log1p(-pp))
                draws={c:np.empty((REPLICATES,4,2)) for c in CONTRASTS}
                for rep in range(REPLICATES):
                    indices=np.r_[pos[rng.integers(len(pos),size=len(pos))],neg[rng.integers(len(neg),size=len(neg))]]
                    w=np.bincount(indices,minlength=len(y));auc={k:aucw(w,state,len(pos),len(neg)) for k,state in states.items()}
                    for candidate,baseline in CONTRASTS:
                        for si,seed in enumerate(SEEDS):
                            draws[candidate,baseline][rep,si,0]=auc[seed,candidate]-auc[seed,baseline]
                            draws[candidate,baseline][rep,si,1]=w@(loss[seed,candidate]-loss[seed,baseline])/len(y)
                for (candidate,baseline),values in draws.items():
                    for qi,quantity in enumerate(['auc','logloss']):
                        v=values[:,:,qi].mean(1);lo,hi=np.quantile(v,[.025,.975])
                        rows.append(dict(split=split,domain=d,task=t,contrast=candidate+'-'+baseline,
                            quantity=quantity,n=len(y),n_pos=len(pos),replicates=REPLICATES,training_models=4,
                            conditional_mean_bootstrap_delta=float(v.mean()),ci95_low=float(lo),ci95_high=float(hi)))
                    saved[f'{split}_d{d}t{t}_{candidate}_{baseline}']=values
                print(json.dumps(dict(split=split,domain=d,task=t,completed=True)),flush=True)
    pd.DataFrame(rows).to_csv(target/'selected_row_bootstrap.csv',index=False)
    np.savez_compressed(target/'selected_row_bootstrap_draws.npz',**saved)
    (target/'bootstrap_receipt.json').write_text(json.dumps(dict(status='COMPLETE_CONDITIONAL_ROW_UNCERTAINTY',
        elapsed_seconds=time.time()-started,replicates=REPLICATES,shared_weights_across_training_seeds=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),independent_review=False,
        limitations=['conditional on four fitted models and frozen selections','not training-seed population uncertainty',
            'tasks stratified separately: no synthetic macro CI','no user/time cluster correction or multiplicity adjustment',
            'validation intervals are not post-selection confidence guarantees','test remains retrospective forensic']),indent=2)+'\n')


if __name__=='__main__':main()
