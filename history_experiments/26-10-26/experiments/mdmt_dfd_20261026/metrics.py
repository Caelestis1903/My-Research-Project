"""One evaluation definition for every method and role."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from .state import atomic_json

def losses(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.clip(np.asarray(p,dtype=np.float64),1e-7,1-1e-7)
    return -y*np.log(p)-(1-y)*np.log1p(-p)
def auc(y,p,weights=None):
    if len(y)==0 or len(np.unique(y))<2:return None
    return float(roc_auc_score(y,p,sample_weight=weights))
def evaluate_probabilities(y,p,domain,weights=None):
    y=np.asarray(y);p=np.asarray(p);domain=np.asarray(domain);ll=losses(y,p)
    if y.shape!=p.shape or p.shape[1]!=2 or not np.isfinite(p).all():raise ValueError('Invalid prediction shape or value')
    w=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=np.float64)
    cells=[]
    for q in range(2):
        for d in range(3):
            mask=(domain==d)&(w>0);sw=w[mask];n=int(mask.sum());pos=int(y[mask,q].sum())
            cells.append(dict(cell=q*3+d,domain=d,task=q,n=n,positive=pos,negative=n-pos,auc=auc(y[mask,q],p[mask,q],sw),logloss=float(np.average(ll[mask,q],weights=sw)) if n else None))
    def mean(field,cs):
        a=[c[field] for c in cs];return float(np.mean(a)) if all(v is not None for v in a) else None
    return dict(cells=cells,macro_auc=mean('auc',cells),macro_logloss=mean('logloss',cells),domain=[dict(domain=d,auc=mean('auc',[c for c in cells if c['domain']==d]),logloss=mean('logloss',[c for c in cells if c['domain']==d])) for d in range(3)],task=[dict(task=q,auc=mean('auc',cells[q*3:q*3+3]),logloss=mean('logloss',cells[q*3:q*3+3]),pooled_auc=auc(y[:,q],p[:,q],w)) for q in range(2)],natural_logloss=float(np.average(ll.mean(1),weights=w)),pooled_auc=[auc(y[:,q],p[:,q],w) for q in range(2)],worst_cell_auc=min(c['auc'] for c in cells) if all(c['auc'] is not None for c in cells) else None,clip_rate=float(np.mean((p<1e-7)|(p>1-1e-7))))
compute=evaluate_probabilities

def save_predictions(path,rows,y,p,domain,role,checkpoint_hash,**extra):
    pth=Path(path);pth.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(pth,row_id=np.asarray(rows,dtype=np.int64),domain=np.asarray(domain,dtype=np.int64),y=np.asarray(y),p=np.asarray(p),role=np.asarray(role),checkpoint_hash=np.asarray(checkpoint_hash),**extra)
    m=evaluate_probabilities(y,p,domain);atomic_json(pth.with_suffix('.metrics.json'),dict(m,role=role,checkpoint_hash=checkpoint_hash));return m

def paired_summary(differences):
    x=np.asarray(differences,dtype=np.float64);n=len(x);s=float(x.std(ddof=1)) if n>1 else None
    quant=3.182446 if n==4 else 12.7062047364 if n==2 else None
    return dict(n=n,values=x.tolist(),mean=float(x.mean()),sample_sd=s,positive=int((x>0).sum()),t95=None if quant is None else [float(x.mean()-quant*s/np.sqrt(n)),float(x.mean()+quant*s/np.sqrt(n))])

def paired_bootstrap(y,p_candidate,p_reference,domain,user_ids,rng,repetitions=1000):
    # Raw user clustering where possible; never encoded UNK as cluster.
    n=len(y);clusters=np.arange(n) if user_ids is None else np.asarray(user_ids);unique,inverse=np.unique(clusters,return_inverse=True);out=[];invalid=0
    for _ in range(repetitions):
        count=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique));w=count[inverse]
        a=evaluate_probabilities(y,p_candidate,domain,w);b=evaluate_probabilities(y,p_reference,domain,w)
        if a['macro_auc'] is None or b['macro_auc'] is None:invalid+=1
        out.append([a['macro_logloss']-b['macro_logloss'],np.nan if a['macro_auc'] is None or b['macro_auc'] is None else a['macro_auc']-b['macro_auc']])
    out=np.asarray(out);return dict(cluster='raw_user' if user_ids is not None else 'row_fallback_independence_unverified',replicates=repetitions,invalid_auc_replicates=invalid,delta_ll_percentile95=np.quantile(out[:,0],[.025,.975]).tolist(),delta_auc_percentile95=np.nanquantile(out[:,1],[.025,.975]).tolist(),replicate_deltas=out.tolist())
