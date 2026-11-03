"""Frozen protocol, source primitives and canonical scientific identities."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import yaml
from .vendor.v3.protocol import (setup, cpu_copy, equal, rng_state, restore_rng,
    send_memory, receive_memory, write_json, append_csv, update_ema, state_digest,
    make_model, evaluate, metrics)
from .vendor.v3.vendor.ema02.data import load_dataset as source_load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT/'research_reports/2026-11-03_mdmt_history_information'
DEFAULT_ATTEMPT = ROOT/'artifacts/mdmt_history_information_20261103/attempt1'
PLAN = ROOT/'research_plans/26-11-03/experiment_queue.yaml'

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

@lru_cache(maxsize=1)
def load_protocol(plan=PLAN):
    p = yaml.safe_load(Path(plan).read_text())
    source = json.loads((HERE/'vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31/resolved_protocol.json').read_text())
    for name, conf in p['datasets'].items():
        conf['source'] = source['datasets'][name]
    p['study_id'] = p['study']
    return p

def config(arm, dataset, **overrides):
    p = load_protocol(); d = p['datasets'][dataset]
    if arm == 'A0b': arm = 'A0f'
    assert arm in p['arms']
    c = dict(arm=arm, dataset=dataset, lr=.01, weight_decay=d['weight_decay'],
             rho=d['rho'], epochs=50, batch_size=40960, betas=[.9,.999], eps=1e-8)
    c.update(p['arms'][arm])
    c.update(overrides)
    c['weight_decay'] = d['weight_decay']*c.pop('weight_decay_multiplier',1)
    if arm == 'A1': c.update(lambda_aug=.5, kappa=.05 if dataset=='movielens' else .1,
                             target_domain=0 if dataset=='movielens' else 2)
    return c

def load_dataset(dataset, split_id='original', p=None):
    assert split_id == 'original'
    return source_load(dataset, split_id, p or load_protocol())

def source_manifest():
    rows = [dict(path=str(f.relative_to(HERE)),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            for f in sorted(HERE.rglob('*')) if f.suffix in ('.py','.json')]
    return dict(files=rows,sha256=digest(rows))

def job(dataset, arm, seed, stage, **overrides):
    assert seed in range(2022,2027)
    p=load_protocol();c=config(arm,dataset,**overrides)
    identity=dict(dataset=dataset,seed=seed,split_id='original',config=c,
        source_hash=source_manifest()['sha256'],data_identity=p['datasets'][dataset]['source'],
        protocol=p['protocol'],information=p['information'],history=p['history'])
    return dict(dataset=dataset,seed=seed,split_id='original',config=c,stage=stage,
                source_hash=identity['source_hash'],identity=identity,key=digest(identity))

def make_optimizer(model,c):
    from .optimizers import HistoryOptimizer
    return HistoryOptimizer(model,c)
