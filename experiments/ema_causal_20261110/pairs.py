"""Exact population transitions and bounded fixed-pair diagnostics."""
import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import tempfile
import numpy as np

_KERNEL=None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        source=Path(__file__).with_name('pairs_kernel.cpp')
        identity=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        directory=Path(tempfile.gettempdir())/'ema_causal_code';directory.mkdir(exist_ok=True)
        target=directory/f'pairs_{identity}.so'
        if not target.exists():
            # Compilation produces code only; no predictions or model state enter this directory.
            with tempfile.TemporaryDirectory(dir=directory) as build:
                temporary=Path(build)/'pairs.so'
                subprocess.run(['c++','-std=c++17','-O3','-shared','-fPIC',str(source),'-o',str(temporary)],check=True)
                temporary.replace(target)
        _KERNEL=ctypes.CDLL(str(target)).transitions
        f64=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
        i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        _KERNEL.argtypes=[f64,f64,i64,ctypes.c_int64,i64];_KERNEL.restype=None
    return _KERNEL


def exact_transition_table(y, reference, changed):
    y=np.ascontiguousarray(y,dtype=np.int64)
    a=np.ascontiguousarray(reference,dtype=np.float64);b=np.ascontiguousarray(changed,dtype=np.float64)
    if y.ndim!=1 or a.shape!=y.shape or b.shape!=y.shape: raise ValueError('Population shape mismatch')
    if not np.isin(y,[0,1]).all() or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Invalid labels or nonfinite predictions')
    positives=int(y.sum());negative=len(y)-positives
    if positives*negative>np.iinfo(np.int64).max: raise OverflowError('Pair count exceeds int64')
    result=np.zeros((3,3),dtype=np.int64)
    _kernel()(a,b,y,len(y),result)
    if result.sum()!=positives*negative or (result<0).any(): raise ArithmeticError('Exact transition count failed')
    return result


def transition_metrics(table):
    n=int(table.sum())
    if not n:return dict(pairs=0,auc_reference=None,auc_changed=None,delta_auc=None)
    weights=np.arange(3,dtype=np.float64)/2
    a=float(table.sum(1)@weights/n);b=float(table.sum(0)@weights/n)
    contribution=table*(weights[None,:]-weights[:,None])/n
    return dict(pairs=n,auc_reference=a,auc_changed=b,delta_auc=float(contribution.sum()),
        corrected=float(np.maximum(contribution,0).sum()),broken=float(-np.minimum(contribution,0).sum()),
        closure_error=abs(b-a-contribution.sum()))


def sample_pairs(y, seed, limit=32768):
    y=np.asarray(y);positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
    if len(positive)+len(negative)<=4096 and limit<=32768:
        return _cached_pairs(positive.astype(np.int64,copy=False).tobytes(),
                             negative.astype(np.int64,copy=False).tobytes(),seed,limit)
    return _sample_pairs_uncached(positive,negative,seed,limit)


@lru_cache(maxsize=128)
def _cached_pairs(positive_bytes,negative_bytes,seed,limit):
    # Admission above caps retained arrays at 64 MiB and index keys at 4 MiB.
    pairs=_sample_pairs_uncached(np.frombuffer(positive_bytes,dtype=np.int64),
                                np.frombuffer(negative_bytes,dtype=np.int64),seed,limit)
    pairs.setflags(write=False)
    return pairs


def _sample_pairs_uncached(positive,negative,seed,limit):
    n=len(positive)*len(negative);k=min(limit,n)
    if not k:return np.empty((0,2),dtype=np.int64)
    rng=np.random.default_rng(seed)
    # Floyd sampling uses O(k) memory even for a population with billions of pairs.
    selected=set()
    for upper in range(n-k,n):
        value=int(rng.integers(upper+1));selected.add(upper if value in selected else value)
    ids=np.fromiter(sorted(selected),dtype=np.int64)
    return np.stack((positive[ids//len(negative)],negative[ids%len(negative)]),1)


def panel_pair_statistics(y,reference,changed,pairs,ref_logits=None,new_logits=None,bounds=None):
    pairs=np.asarray(pairs,dtype=np.int64)
    if not len(pairs):return dict(n_draws=0,n_unique_pairs=0,not_estimable=True)
    y=np.asarray(y)
    if not ((y[pairs[:,0]]==1).all() and (y[pairs[:,1]]==0).all()):raise ValueError('Pairs do not align with labels')
    a=np.asarray(reference,dtype=np.float64);b=np.asarray(changed,dtype=np.float64)
    ma=a[pairs[:,0]]-a[pairs[:,1]];mb=b[pairs[:,0]]-b[pairs[:,1]]
    sa=np.sign(ma).astype(np.int64)+1;sb=np.sign(mb).astype(np.int64)+1
    counts=np.bincount(3*sa+sb,minlength=9).reshape(3,3)
    gain=(sb-sa)/2;delta=b-a
    result=dict(n_draws=len(pairs),n_unique_pairs=len(np.unique(pairs,axis=0)),
        transition=counts.tolist(),net=float(gain.mean()),corrected=float(np.maximum(gain,0).mean()),
        broken=float(np.maximum(-gain,0).mean()),margin_delta_mean=float((mb-ma).mean()),
        margin_delta_quantiles=np.quantile(mb-ma,[0,.25,.5,.75,1]).tolist(),
        score_positive_mean=float(delta[y==1].mean()),score_negative_mean=float(delta[y==0].mean()),
        score_positive_quantiles=np.quantile(delta[y==1],[0,.25,.5,.75,1]).tolist(),
        score_negative_quantiles=np.quantile(delta[y==0],[0,.25,.5,.75,1]).tolist())
    if bounds is None: bounds=np.quantile(np.abs(ma),[.25,.5,.75]).tolist()
    strata=np.searchsorted(bounds,np.abs(ma),side='right');result['bounds']=list(bounds);result['strata']=[]
    for k in range(len(bounds)+1):
        mask=strata==k;count=int(mask.sum())
        result['strata'].append(dict(stratum=k,n=count,pair_mass=count/len(pairs),
            net_rate=float(gain[mask].mean()) if count else None,net_contribution=float(gain[mask].sum()/len(pairs))))
    result['rank_strata']=[]
    for k in range(3):
        mask=sa==k;count=int(mask.sum())
        result['rank_strata'].append(dict(reference_rank=k,n=count,pair_mass=count/len(pairs),
            net_rate=float(gain[mask].mean()) if count else None,net_contribution=float(gain[mask].sum()/len(pairs))))
    if ref_logits is not None and new_logits is not None:
        dz=np.asarray(new_logits,dtype=np.float64)-np.asarray(ref_logits,dtype=np.float64)
        result['logit_margin_delta_mean']=float((dz[pairs[:,0]]-dz[pairs[:,1]]).mean())
    return result


def pair_overlap(reference, first, second, pairs):
    def change(value):
        v=np.asarray(value,dtype=np.float64);return (np.sign(v[pairs[:,0]]-v[pairs[:,1]])+1)/2
    base=change(reference);a=change(first)-base;b=change(second)-base
    return dict(corrected_overlap=float(np.minimum(np.maximum(a,0),np.maximum(b,0)).mean()),
        broken_overlap=float(np.minimum(np.maximum(-a,0),np.maximum(-b,0)).mean()),
        first_unique_correction=float((np.maximum(a,0)-np.minimum(np.maximum(a,0),np.maximum(b,0))).mean()),
        second_unique_correction=float((np.maximum(b,0)-np.minimum(np.maximum(a,0),np.maximum(b,0))).mean()))


def source_overlap(past,online,recent,pairs):
    def rank(v):
        v=np.asarray(v,dtype=np.float64);return (np.sign(v[pairs[:,0]]-v[pairs[:,1]])+1)/2
    u=rank(online)-rank(past);v=rank(recent)-rank(online)
    recovered=np.minimum(np.maximum(-u,0),np.maximum(v,0))
    lost=np.minimum(np.maximum(u,0),np.maximum(-v,0))
    return dict(recovered=float(recovered.mean()),lost_recent_benefit=float(lost.mean()),
        correction_not_prior_damage=float((np.maximum(v,0)-recovered).mean()),
        damage_not_prior_benefit=float((np.maximum(-v,0)-lost).mean()),
        joint_sign_counts=np.bincount((np.sign(u).astype(int)+1)*3+np.sign(v).astype(int)+1,minlength=9).reshape(3,3).tolist())
