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
        source=Path(__file__).with_name('b_pairs_kernel.cpp')
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


