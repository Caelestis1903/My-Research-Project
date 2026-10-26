"""Conditional validation row bootstrap on preregistered checkpoint grid."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from v2_20261016 import OUT,COMMON,dump,sha


def rank_data(scores,labels):
    order=np.argsort(scores,kind='stable');ss=scores[order];starts=np.r_[0,np.flatnonzero(ss[1:]!=ss[:-1])+1]
    return order,starts,labels[order].astype(float)


def weighted_auc(ranks,weights):
    order,starts,y=ranks;w=weights[order]
    pos=np.add.reduceat(w*y,starts);neg=np.add.reduceat(w*(1-y),starts)
    npos=pos.sum();nneg=neg.sum()
    return float((pos*(np.cumsum(neg)-neg/2)).sum()/(npos*nneg))


def one(run,draws=200):
    config=json.loads((run/'config.json').read_text());seed=config['seed'];dataset=config['dataset']
    epochs=[e for e in COMMON if (run/f'validation_epoch{e:02d}.npz').exists()]
    assert epochs==list(COMMON),'formal checkpoint grid incomplete'
    predictions=[];ids=None;domains=None;labels=None
    for e in epochs:
        with np.load(run/f'validation_epoch{e:02d}.npz',allow_pickle=False) as z:
            if ids is None:ids=z['row_ids'];domains=z['domains'];labels=z['labels']
            else:
                assert np.array_equal(ids,z['row_ids']) and np.array_equal(domains,z['domains']) and np.array_equal(labels,z['labels'])
            predictions.append(z['predictions'].astype(float))
    cell_indices=[np.flatnonzero(domains==d) for d in range(3)]
    ranks=[[[rank_data(pred[idx,t],labels[idx,t]) for t in range(2)] for idx in cell_indices] for pred in predictions]
    strata=[np.flatnonzero((domains==d)&(labels[:,0]==a)&(labels[:,1]==b)) for d in range(3) for a in (0,1) for b in (0,1)]
    strata=[s for s in strata if len(s)>0]
    rng=np.random.default_rng(20261016+seed);scores=np.zeros((draws,len(epochs)));weights=np.empty(len(ids),dtype=float)
    for draw in range(draws):
        for idx in strata:
            # Multinomial row resampling preserves task-label linkage and cell priors.
            weights[idx]=rng.multinomial(len(idx),np.full(len(idx),1/len(idx)))
        for j in range(len(epochs)):
            scores[draw,j]=np.mean([weighted_auc(ranks[j][d][t],weights[idx]) for d,idx in enumerate(cell_indices) for t in range(2)])
    selected=np.asarray(epochs)[scores.argmax(1)]
    target=OUT/'analysis'/'selection_bootstrap';target.mkdir(parents=True,exist_ok=True)
    identity=f"{dataset}_{config['optimizer']}_{config['sampler']}_{config['arm']}_{seed}"
    np.savez_compressed(target/(identity+'.npz'),epoch_grid=epochs,macro_auc=scores,selected_epoch=selected)
    rows=[dict(dataset=dataset,optimizer=config['optimizer'],variant=config['sampler'],arm=config['arm'],seed=seed,
        epoch=e,probability=float(np.mean(selected==e)),draws=draws,run=str(run)) for e in epochs]
    dump(target/(identity+'_receipt.json'),dict(status='COMPLETE',draws=draws,seed=seed,run=str(run),
        source_sha256=sha(__file__),conditional_on_fixed_trained_models=True,
        grid=epochs,full_50_epoch_selection_uncertainty=False,training_uncertainty=False,
        stratification='domain × joint task labels; paired row weights repeat across arms for same seed',test_used=False))
    return rows


def check():
    from sklearn.metrics import roc_auc_score
    rng=np.random.default_rng(23);y=rng.integers(0,2,101);scores=np.round(rng.normal(size=101),1);w=rng.integers(0,5,101)
    error=abs(weighted_auc(rank_data(scores,y),w)-roc_auc_score(y,scores,sample_weight=w));assert error<1e-12
    dump(OUT/'selection_engineering.json',dict(status='PASS',weighted_auc_tie_and_bootstrap_oracle_error=error,source_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--check',action='store_true');args=p.parse_args()
    if args.check:check()
    else:
        rows=one(args.run);target=OUT/'analysis'/'selection_bootstrap';identity=f"{rows[0]['dataset']}_{rows[0]['optimizer']}_{rows[0]['variant']}_{rows[0]['arm']}_{rows[0]['seed']}"
        pd.DataFrame(rows).to_csv(target/(identity+'.csv'),index=False)
        print('Selection bootstrap complete',identity)
