"""Frozen scientific configuration and dependency-family source identity."""
import copy
import importlib.util
from pathlib import Path
from .state import digest,read_json,sha_file

ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'research_plans/26-11-03_overnight_AB'
SEEDS=tuple(range(2022,2027))
FAMILIES={'HYPERGATE':'A_HYPERGATE','AUTOIFS':'A_AUTOIFS','C_MIX_P01_SERIAL_V1':'B_CP',
          'C_MIX_GAUSSIAN_V1':'B_CG','A0':'NATIVE_REPLAY','C_MIX':'NATIVE_REPLAY','P01':'NATIVE_REPLAY'}
def plan_module():
    spec=importlib.util.spec_from_file_location('overnight_frozen_plan',PLAN/'plan.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
def load_plan(path=None):
    p=read_json(path or PLAN/'protocol.json'); plan_module().compile_plan(p); return p
def source_manifest(family=None):
    here=Path(__file__).parent
    excluded={'A_HYPERGATE':('worker_b.py','objectives.py','optimizers.py'),
              'A_AUTOIFS':('worker_b.py','objectives.py','optimizers.py'),
              'B_CP':('published_models.py','published_training.py'),
              'B_CG':('published_models.py','published_training.py')}
    rows=[]
    for file in sorted(here.rglob('*')):
        if not file.is_file() or file.suffix not in ('.py','.json') or 'tests' in file.parts or '__pycache__' in file.parts: continue
        if family and (file.name in excluded.get(family,()) or
                       (family.startswith('B_') and 'vendor' in file.parts)): continue
        rows.append(dict(path=str(file.relative_to(here)),sha256=sha_file(file)))
    for native,expected in (
        ('experiments.mdmt_history_information_20261103.protocol','7f16a451f5f0e41d9082d18bf548ad29d1d31f6c350175c5fb9f06ddc8813685'),
        ('experiments.conditional_information_momentum_20261102_v3.run','5adc90554898ce9eefe707de2365ef9b13865402d77b360efaee2f255e91db0a')):
        import importlib
        module=importlib.import_module(native); actual=module.source_manifest()['sha256']
        if actual!=expected: raise ValueError('Frozen native source changed: '+native)
        rows.append(dict(path=str(Path(module.__file__).parent),sha256=actual))
    external=here/'vendor/sota2526/SOURCE_MANIFEST.json'
    if external.exists() and (family is None or family.startswith('A_')):
        for entry in read_json(external).get('external_dependency_files',[]):
            path=Path(entry['source_root'])/entry['path']
            if sha_file(path)!=entry['sha256']: raise ValueError('Pinned published source changed: '+str(path))
            rows.append(dict(path=str(path),sha256=entry['sha256']))
    return dict(files=rows,sha256=digest(rows))
def job_config(p,arm,dataset):
    if arm in ('HYPERGATE','AUTOIFS'):
        c=copy.deepcopy(p[arm.lower()])
        if arm=='AUTOIFS': c['rank']=p['datasets'][dataset]['autoifs_rank']
        return c
    d=p['datasets'][dataset]; m=p['m3oe']
    return dict(arm=arm,dataset=dataset,lr=m['lr'],weight_decay=d['weight_decay'],rho=d['rho'],
        epochs=m['epochs'],batch_size=m['batch_size'],eval_batch_size=m['eval_batch_size'],
        betas=m['betas'],eps=m['eps'],c_mix=copy.deepcopy(p['components']['c_mix']),
        p01=copy.deepcopy(p['components']['p01']),gaussian=dict(p['components']['gaussian'],
            target_domain=d['gaussian_target_domain'],kappa=d['gaussian_kappa']))
def make_job(p,row,data_hash):
    arm=row['arm']; ds=row['dataset']; seed=int(row['seed'])
    if seed not in SEEDS or arm not in p['arms']: raise ValueError('Unregistered job')
    family=FAMILIES.get(arm,'NATIVE_REPLAY'); source=source_manifest(family)['sha256']; c=job_config(p,arm,ds)
    identity=dict(dataset=ds,seed=seed,arm=arm,config=c,data_hash=data_hash,source_hash=source,
                  protocol_id=p['protocol_id'],data_protocol=p['data_protocol'])
    return dict(row,job_id=row['job_id'],dataset=ds,seed=seed,arm=arm,method=arm,family=family,
                config=c,protocol=p,config_hash=digest(c),source_hash=source,data_hash=data_hash,
                trajectory_id=digest(identity))

