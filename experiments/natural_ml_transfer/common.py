"""Pinned original producers, common populations and explicit exposure controls."""
from pathlib import Path
import hashlib
import importlib
import json
import pickle
import sys
import numpy as np
from experiments.natural_ml_reassessment.data import load_population

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
REPORT=REPO/'research_plans/2026-10-20_natural_movielens_reassessment'
ART=Path('/home_new/xzf_natural_movielens_reassessment')
ARCHIVE=HERE/'source_archive'
PYTHON='/home/xzf/miniconda3/envs/phase0-3-env/bin/python'
WRAPPERS=('__init__.py','common.py','manifest.py','run.py','evaluate.py','diagnostics.py')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for b in iter(lambda:handle.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)

def ids_sha(ids):
    return hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()

def source_hashes(study):
    result={}
    inventory=json.loads((ARCHIVE/'inventory.json').read_text())
    for name,record in inventory.items():
        if name.startswith(study+'/'):
            actual=sha(ARCHIVE/name)
            if actual!=record['sha256']:raise RuntimeError('pinned source drift: '+name)
            result['archive/'+name]=actual
    for name in WRAPPERS:result['wrapper/'+name]=sha(HERE/name)
    core=HERE.parent/'natural_ml_reassessment'
    for name in ('__init__.py','bootstrap.py','data.py'):result['core/'+name]=sha(core/name)
    for name in ('stage_F0_data.py','stage_S1_data.py','stage_S1_protocol.py'):
        path=core/'source_archive/core/experiments'/name
        result['population_producer/'+name]=sha(path)
    return result

def cache_population(name):
    root=ART/'transfer_population_cache'/name
    path=root/'local_cache/movielens_2022.pkl'
    if path.exists():return root
    b=load_population(name)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as handle:pickle.dump(b,handle,protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    dump(root/'manifest.json',dict(population=name,pickle_sha256=sha(path),data=b.manifest_fields(),schema=b.features.as_dict()))
    return root

def configure(cfg):
    """Inject population/output context before importing original training modules."""
    study=cfg['study'];root=ARCHIVE/study
    for path in (root/'m3oe/examples',root/'m3oe',root/'experiments'):
        sys.path.insert(0,str(path))
    module=importlib.import_module('n19_data' if study=='N19' else 'cm19_legacy_data')
    module.HISTORY=cache_population(cfg['population'])
    module.OUT=ART/study/cfg['scope'];module.ART=module.OUT/'states'
    original_population=module.population
    def population(dataset,suite,noise_interaction=0.):
        b=original_population(dataset,suite,noise_interaction)
        b.metadata=dict(b.metadata,dataset_config=cfg['population'],population_seed=2022,
            reassessment_study=study,synthetic_suite=suite if suite!='natural' else None,
            task_definitions={'task0':'rating>3','task1':'Bernoulli(.2+.6*controlled_Z)' if suite=='stability' else 'rating>4'},
            observation_mechanism='synthetic train-label flips with fixed 2% audit' if suite=='noise' else 'per-visit simulated label mask' if suite=='sampling' else 'fully observed labels',
            main_rows=len(b.main),meta_rows=len(b.meta),
            main_ids_sha256=ids_sha(b.train_ids[b.main]),meta_ids_sha256=ids_sha(b.train_ids[b.meta]),
            selection_ids_sha256=ids_sha(b.val_ids[b.selection]),development_ids_sha256=ids_sha(b.val_ids[b.development]))
        return b
    module.population=population
    return module

class ExposureRNG:
    """Crop only the original full-main permutation, preserving its RNG state."""
    def __init__(self,rng,main,limit):self.rng=rng;self.main=np.asarray(main);self.limit=int(limit)
    def __getattr__(self,name):return getattr(self.rng,name)
    def permutation(self,values):
        order=self.rng.permutation(values)
        if np.array_equal(np.asarray(values),self.main):return order[:self.limit]
        return order

def metadata(cfg,model_data):
    return dict(global_run_id=cfg['run_id'],dataset_config=cfg['population'],budget_mode=cfg['budget'],
        support_main_rows=len(model_data.main),support_main_domain_counts=np.bincount(model_data.train_x[model_data.main,-1].astype(int),minlength=3).tolist(),
        main_exposure_per_macro_epoch=178125 if cfg['budget']=='main178125' else len(model_data.main),
        reassessment_sources=cfg['source_hashes'],historical_run_id=cfg['historical_run_id'],
        original_source_pin='c440350681a7a3e41ab3f05eb1c4d31f419b989f' if cfg['study']=='N19' else 'eab817b3de626caee9ee9aeb1db4dd9aa3ef9e4c',
        exploratory_replication=True,independent_review=False,review='primary under explicitly invoked start.md',
        interruption_policy='archive interrupted attempt; same-seed full-budget restart',
        data_population_metadata=model_data.metadata)
