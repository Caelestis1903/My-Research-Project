"""The audited natural population, with reusable read-only array caches."""
from pathlib import Path
from types import SimpleNamespace
import copy
import numpy as np
from experiments.mdmt_history_information_20261103 import protocol as history
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.audited_data import FeatureSchema
from .state import read_json,write_json,sha_file,digest

ARRAYS=tuple(f'{split}_{field}' for split in ('train','val','test') for field in ('x','y','row_ids'))
def schema_of(bundle): return bundle.features.as_dict()
def training_view(bundle):
    return SimpleNamespace(**{k:v for k,v in vars(bundle).items() if not k.startswith('test_')})
def load_bundle(dataset,cfg=None):
    source=copy.deepcopy(history.load_protocol())
    if cfg: source['datasets'][dataset]['csv']=cfg['datasets'][dataset]['csv']
    return history.load_dataset(dataset,'original',source)
def prepare_cache(root,p):
    root=Path(root); root.mkdir(parents=True,exist_ok=True); manifests={}
    for dataset in p['datasets']:
        target=root/dataset; meta=target/'manifest.json'
        if meta.exists():
            manifest=read_json(meta)
            if sha_file(p['datasets'][dataset]['csv'])!=manifest['raw_sha256']: raise ValueError('Raw identity changed')
            for name,expected in manifest['cache_sha256'].items():
                if sha_file(target/(name+'.npy'))!=expected: raise ValueError('Data cache changed')
        else:
            bundle,manifest=load_bundle(dataset,p); target.mkdir(parents=True,exist_ok=True)
            for name in ARRAYS: np.save(target/(name+'.npy'),getattr(bundle,name),allow_pickle=False)
            manifest['cache_sha256']={name:sha_file(target/(name+'.npy')) for name in ARRAYS}
            write_json(meta,manifest)
        manifests[dataset]=manifest
    return manifests
def cached_bundle(root,dataset,include_test=True):
    target=Path(root)/dataset; manifest=read_json(target/'manifest.json')
    features=FeatureSchema(**manifest['features'])
    arrays={name:np.load(target/(name+'.npy'),mmap_mode='r',allow_pickle=False)
            for name in ARRAYS if include_test or not name.startswith('test_')}
    return SimpleNamespace(dataset=dataset,features=features,domain_num=3,task_num=2,manifest=manifest,**arrays)

