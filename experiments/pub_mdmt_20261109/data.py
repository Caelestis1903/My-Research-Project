"""The frozen natural arrays, with Test labels loaded only at final evaluation."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from .common import HERE,read_json,write_json,sha_file
from .vendor.audited_data import FeatureSchema,array_hash,split_row_fingerprint,row_ids_hash,_load_movielens_base,_load_kuairand_base
from .vendor.model import make_model

SOURCE = read_json(HERE/'vendor/resolved_protocol.json')
DEFAULT_CACHE='/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache'
RAW={'movielens':'/home_new/xzf/datasets/ml-1m/ml-1m.csv','kuairand':'/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv'}
ARRAYS=tuple(f'{s}_{f}' for s in ('train','val','test') for f in ('x','y','row_ids'))

def load_bundle(dataset,cache_root=DEFAULT_CACHE,include_test=False):
    folder=Path(cache_root)/dataset; manifest=read_json(folder/'manifest.json')
    arrays={n:np.load(folder/(n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ARRAYS if include_test or not n.startswith('test_')}
    return SimpleNamespace(dataset=dataset,features=FeatureSchema(**manifest['features']),domain_num=3,task_num=2,manifest=manifest,**arrays)

def model_for(bundle): return make_model(bundle,SOURCE['datasets'][bundle.dataset])

def prepare_cache(cache_root=DEFAULT_CACHE):
    result={}
    for dataset in RAW:
        folder=Path(cache_root)/dataset; manifest_path=folder/'manifest.json'; expected=SOURCE['datasets'][dataset]
        if sha_file(RAW[dataset])!=expected['raw_sha256']: raise ValueError('Raw CSV identity changed: '+dataset)
        if not manifest_path.exists():
            loader=_load_movielens_base if dataset=='movielens' else _load_kuairand_base
            bundle=loader(RAW[dataset],'natural',2022); folder.mkdir(parents=True,exist_ok=True)
            manifest={k:expected[k] for k in ('raw_sha256','array_hashes','split_row_fingerprints','split_row_id_hashes','encoder_hash','scaler_hash','split_rows','train_domain_counts')}
            manifest.update(dataset=dataset,features=bundle.features.as_dict(),raw_path=RAW[dataset])
            for n in ARRAYS: np.save(folder/(n+'.npy'),getattr(bundle,n),allow_pickle=False)
            manifest['cache_sha256']={n:sha_file(folder/(n+'.npy')) for n in ARRAYS}; write_json(manifest_path,manifest)
        manifest=read_json(manifest_path)
        for key in ('raw_sha256','array_hashes','split_row_fingerprints','split_row_id_hashes','encoder_hash','scaler_hash','split_rows','train_domain_counts'):
            if manifest[key]!=expected[key]: raise ValueError('Source/cache metadata mismatch: '+dataset+'/'+key)
        if manifest['features']['encoder_hash']!=expected['encoder_hash'] or manifest['features']['scaler_hash']!=expected['scaler_hash']: raise ValueError('Preprocessor identity changed')
        bundle=load_bundle(dataset,cache_root,include_test=True)
        for n in ARRAYS:
            if sha_file(folder/(n+'.npy'))!=manifest['cache_sha256'][n]: raise ValueError('Cache file changed: '+n)
            if n in expected['array_hashes'] and array_hash(getattr(bundle,n))!=expected['array_hashes'][n]: raise ValueError('Array content changed: '+n)
        for split,key in [('train','train'),('val','validation'),('test','test')]:
            x,y,ids=[getattr(bundle,f'{split}_{f}') for f in ('x','y','row_ids')]
            if len(x)!=expected['split_rows'][key] or split_row_fingerprint(x,y,ids)!=expected['split_row_fingerprints'][key] or row_ids_hash(ids)!=expected['split_row_id_hashes'][key]: raise ValueError('Population changed')
        if np.bincount(bundle.train_x[:,-1].astype('int64'),minlength=3).tolist()!=expected['train_domain_counts']: raise ValueError('Domain population changed')
        result[dataset]=manifest
    return result
