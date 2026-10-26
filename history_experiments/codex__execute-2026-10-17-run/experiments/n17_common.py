"""Frozen paths and shared data access for the preregistered N0--N6 study."""
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import v2_20261016 as base

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'research_plans/2026-10-17/execution_20261017'
HISTORY=Path('/home/xzf/Code/phase0-3-20261016-v3-git/research_plans/2026-10-16_v3/execution_20261016')
PYTHON='/home/xzf/miniconda3/envs/phase0-3-env/bin/python'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def bundle(dataset='kuairand',seed=2022):
    # KR population is seed independent. ML subsets are seed dependent.
    popseed=2022 if dataset=='kuairand' else seed
    with (HISTORY/'local_cache'/f'{dataset}_{popseed}.pkl').open('rb') as f:b=pickle.load(f)
    b.seed=seed
    return b

def split(dataset='kuairand',seed=2022):
    popseed=2022 if dataset=='kuairand' else seed
    with np.load(HISTORY/'splits'/f'{dataset}_seed{popseed}.npz') as z:
        return {k:z[k] for k in z.files}

def sources():
    return {**base.sources(),**{str(p.relative_to(ROOT)):sha(p) for p in
        [ROOT/'experiments/v3_train_20261016.py',ROOT/'experiments/n17_common.py',ROOT/'experiments/n17_train.py']}}
