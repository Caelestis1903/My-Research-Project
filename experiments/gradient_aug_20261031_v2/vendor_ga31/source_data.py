"""Frozen train-only preprocessing, natural domains, source row order."""
from pathlib import Path
import hashlib, json, os, pickle, fcntl
import numpy as np
from .audited_data import _load_movielens_base, _load_kuairand_base, array_hash

RAW = {'movielens': '/home/xzf/data/ml-1m/ml-1m.csv', 'kuairand': '/home/xzf/data/kuairand/processed/kuai_rand.csv'}
EXPECTED = {'movielens':'f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4','kuairand':'a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f'}
def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def load_data(dataset, cache_root):
    root=Path(cache_root); root.mkdir(parents=True,exist_ok=True)
    p=root/(dataset+'_trainonly_v1.pkl')
    with open(str(p)+'.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        raw_sha=sha_file(RAW[dataset])
        if raw_sha != EXPECTED[dataset]: raise ValueError('Raw dataset SHA256 mismatch')
        if p.exists():
            with p.open('rb') as f: bundle, manifest=pickle.load(f)
            assert manifest['raw_sha256']==raw_sha
        else:
            fn=_load_movielens_base if dataset=='movielens' else _load_kuairand_base
            bundle=fn(RAW[dataset],'natural',2022)
            manifest=bundle.manifest_fields()
            manifest.update(raw_sha256=raw_sha, protocol='ported-train-only', chronological_provenance='preserved source row order, upstream chronological provenance not independently reverified')
            manifest['array_hashes']={k:array_hash(getattr(bundle,k)) for k in ['train_x','train_y','val_x','val_y','test_x','test_y']}
            if dataset=='kuairand':
                assert bundle.train_counts=={0:172782,1:1743898,2:89452}, bundle.train_counts
            else:
                import pandas as pd
                ts=pd.read_csv(RAW[dataset],usecols=['timestamp'])['timestamp']
                manifest['timestamp_monotonic_increasing']=bool(ts.is_monotonic_increasing)
            tmp=p.with_suffix('.tmp')
            with tmp.open('wb') as f: pickle.dump((bundle,manifest),f,pickle.HIGHEST_PROTOCOL)
            os.replace(tmp,p)
    return bundle,manifest
