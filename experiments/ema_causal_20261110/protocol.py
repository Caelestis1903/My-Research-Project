"""Pinned source, natural data, and finite observation schedule."""
import copy
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from experiments.overnight_ab_20261103.state import read_json, write_json, digest, sha_file
from experiments.overnight_ab_20261103.data_bridge import prepare_cache, cached_bundle
from experiments.ema_mechanism_20261106.protocol import verify_data_identity, write_csv
from experiments.ema_mechanism_20261106.panels import build_panels, save_panels
from experiments.aug_history_20261105.model_adapter import build_model
from experiments.aug_history_20261105.group_registry import build_path_groups
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SOURCE_REF='ffa3de067fcda5a1293af83a465d741e069493f1'
PLAN_REF='5cf92af0d3a8ed77688762616248166b9b68266d'
SEEDS=list(range(2022,2027))
CONDITIONS=[('movielens','clean'),('kuairand','clean'),('movielens','gaussian_d0'),('kuairand','gaussian_d0')]
ANCHORS=[5,10,20,35,50]
WINDOWS={'movielens':[20,40,80,160],'kuairand':[69,138,277,553]}
CACHE='/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache'

def implementation_identity():
    files={str(p.relative_to(ROOT)):sha_file(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix in ('.py','.cpp','.sh')}
    return dict(files=files,sha256=digest(files))

def resolve(plan):
    plan=Path(plan).resolve()
    old=read_json(ROOT/'research_plans/26-11-6/experiment_manifest.json')
    source=read_json(ROOT/'research_reports/2026-11-06_ema_mechanism/RESOLVED_PROTOCOL.json')['configs']
    configs=copy.deepcopy(source)
    for ds,c in configs.items():
        c.update(rho_by_physical_shadow={'F':c['rho']},aliases={},target_domain=0)
    jobs=[]
    for ds,obj in CONDITIONS:
        for seed in SEEDS:
            j=dict(dataset=ds,objective=obj,seed=seed,epochs=50,steps_per_epoch=configs[ds]['steps_per_epoch'],config=copy.deepcopy(configs[ds]))
            j['job_id']=f'{ds}__{obj}__seed{seed}';j['online_trajectory_id']=digest(j);jobs.append(j)
    plans={p.name:sha_file(p) for p in sorted(plan.glob('*.md'))}
    return SimpleNamespace(manifest=old,configs=configs,jobs=jobs,plan=plan,
        identity=dict(study_id='EMA_CAUSAL_20261110',source_commit=SOURCE_REF,plan_commit=PLAN_REF,plan_hash=digest(plans),plan_files=plans,implementation_hash=implementation_identity()['sha256']))

def audit(study,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    files={}
    for folder in ('aug_history_20261105','ema_mechanism_20261106','overnight_ab_20261103','mdmt_history_information_20261103'):
        for p in sorted((ROOT/'experiments'/folder).rglob('*.py')):
            rel=str(p.relative_to(ROOT));blob=subprocess.check_output(['git','rev-parse',SOURCE_REF+':'+rel],cwd=ROOT,text=True).strip()
            actual=subprocess.check_output(['git','hash-object',str(p)],cwd=ROOT,text=True).strip()
            if actual!=blob: raise ValueError('Pinned source changed: '+rel)
            files[rel]=dict(git_blob=blob,sha256=sha_file(p))
    receipt=dict(study.identity,source_files=files,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),cuda=torch.cuda.is_available()))
    write_json(out/'SOURCE_RECEIPT.json',receipt);return receipt

def schedule(ds):
    steps=20 if ds=='movielens' else 49
    targets=[e*steps for e in range(1,51)]
    needed=sorted({max(0,t-h) for t in targets for h in WINDOWS[ds]})
    return dict(epoch_end_steps=targets,history_capture_steps=needed,
        pulse_start_steps=[(e-1)*steps for e in range(5,51,5)],pulse_end_steps=[e*steps for e in range(5,51,5)],
        matched_start_steps=[e*steps-WINDOWS[ds][1] for e in (35,50)],matched_end_steps=[e*steps for e in (35,50)],source_presteps=[20*steps-1,35*steps-1])

def prepare(study,out,cache_root=None,ml_csv=None,kr_csv=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);cache=Path(cache_root or CACHE)
    paths={'movielens':ml_csv or '/home_new/xzf/datasets/ml-1m/ml-1m.csv','kuairand':kr_csv or '/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv'}
    if not all((cache/d/'manifest.json').exists() for d in study.configs):
        cache=out/'data_cache';prepare_cache(cache,{'datasets':{d:{'csv':p} for d,p in paths.items()}})
    data=verify_data_identity(study,cache,paths);panels={};groups={}
    for ds in study.configs:
        bundle=cached_bundle(cache,ds,include_test=False)
        model=build_model(bundle,study.configs[ds]);groups[ds]=build_path_groups(model,bundle).as_dict();del model
        panels[ds]={str(seed):save_panels(out,ds,seed,build_panels(bundle,seed,study.manifest['panels'])) for seed in SEEDS}
    result=dict(identity=study.identity,jobs=study.jobs,configs=study.configs,cache_root=str(cache.resolve()),data_identity=data,panels=panels,groups=groups)
    for name,value in [('PREPARED',result),('DATA_MANIFEST',data),('PANEL_MANIFEST',panels),('PROBE_MANIFEST',dict(panels=panels)),('GROUP_SUPPORT',groups),('RESOLVED_PROTOCOL',dict(identity=study.identity,configs=study.configs,jobs=study.jobs)),('OBSERVATION_SCHEDULE',{d:schedule(d) for d in study.configs})]:
        write_json(out/(name+'.json'),value)
    write_json(out/'EXPECTED_OUTPUTS.json',dict(core=['EPOCHS','INTERVENTION_METRICS','FACTORIAL_EFFECTS','TEMPORAL_EFFECTS','PAIR_TRANSITIONS','FUNCTIONAL_EFFECTS','FUSION_EFFECTS','SOURCE_OVERLAP','BRIDGE_FOLDS','BRIDGE_VALUE','ENDPOINTS'],conditional=['H2_MATCH','ADAM_SOURCE','STAR_CROSS'],status='PREPARED'))
    return result


def peak_active_bytes(dataset,parameter_count):
    users={}
    for t in schedule(dataset)['epoch_end_steps']:
        for h in set(min(h,t) for h in WINDOWS[dataset]):users.setdefault(t-h,set()).add(t)
    peak=max(sum(origin<=t<max(ends) for origin,ends in users.items()) for t in range(max(schedule(dataset)['epoch_end_steps'])+1))
    return (8*peak+100)*parameter_count+(32<<20)
