"""Frozen identities and natural data admission for the mechanism study."""
import copy
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from experiments.overnight_ab_20261103.state import read_json, write_json, sha_file, digest, capture_rng, restore_rng
from experiments.overnight_ab_20261103.data_bridge import prepare_cache, cached_bundle

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE_REF = '40705dfe0bb4fe614bdca835c0bae567348c87d0'
PLAN_REF = '9fb59d5a53bf0efa555caeaca03e1ffa3812b686'
SOURCE_PROTOCOL = ROOT/'experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31/resolved_protocol.json'
SEEDS = [2022,2023,2024,2025,2026]
DEFAULT_CACHE = Path('/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache')

@dataclass
class ResolvedStudy:
    manifest: dict
    jobs: list
    configs: dict
    manifest_hash: str
    identity: dict
    paths: dict


def implementation_identity():
    files = {str(f.relative_to(ROOT)):sha_file(f) for f in sorted(HERE.rglob('*.py'))}
    return dict(files=files,sha256=digest(files))


def resolve_study(manifest, paths=None):
    path=Path(manifest).resolve(); m=read_json(path); p=m['protocol']
    if m['seeds'] != SEEDS or len(m['jobs']) != 20 or set(m['datasets']) != {'movielens','kuairand'}:
        raise ValueError('The frozen twenty-job/five-seed matrix changed')
    expected=dict(batch_size=40960,epochs=50,lr=.01,betas=[.9,.999],eps=1e-8,
        foreach=False,fused=False,amsgrad=False,amp=False,tf32=False,early_stopping=False,
        population='FULLTRAIN_NATURAL80_10_10',optimizer='torch.optim.Adam',
        domain_resampling=False,decoupled_weight_decay=False,gradient_clipping=False,
        target_domains=[0],lambda_aug=.5,feedback_to_online=False,adaptive_ema_controller=False)
    if any(p.get(k)!=v for k,v in expected.items()):
        raise ValueError('Source training protocol changed')
    matrix={(j['dataset'],j['objective'],j['seed']) for j in m['jobs']}
    if matrix != {(d,a,s) for d in m['datasets'] for a in ('clean','gaussian_d0') for s in SEEDS}:
        raise ValueError('Scientific job coverage changed')
    if len({j['job_id'] for j in m['jobs']}) != 20 or m['regular_shadows']['half_lives_epochs'] != [1,2,4]:
        raise ValueError('Job identifiers or fixed half-lives changed')
    if m['pulse']['end_epochs'] != list(range(5,51,5)) or m['resource']['max_simultaneous_workers'] != 2:
        raise ValueError('Pulse schedule or concurrency cap changed')
    source=read_json(SOURCE_PROTOCOL)['datasets']; configs={}
    for ds,d in m['datasets'].items():
        frozen=(.05,.0001,20,773470,.9828205985452511) if ds=='movielens' else (.1,.00001,49,1614254,.995)
        if tuple(d[k] for k in ('kappa','weight_decay','steps_per_epoch','parameter_count','rho_F')) != frozen:
            raise ValueError('Frozen dataset training settings changed: '+ds)
        aliases={'F':'H2'} if ds=='movielens' else {}
        rates={f'H{h}':2**(-1/(h*d['steps_per_epoch'])) for h in (1,2,4)}
        if ds=='movielens': rates['H2']=d['rho_F']
        else: rates['F']=d['rho_F']
        if d['aliases'] != aliases or d['rho_by_physical_shadow'] != rates:
            raise ValueError('Physical EMA clock/alias changed: '+ds)
        configs[ds]=dict(source=source[ds],dataset=ds,batch_size=40960,eval_batch_size=40960,
            epochs=50,lr=.01,betas=[.9,.999],eps=1e-8,kappa=d['kappa'],rho=d['rho_F'],
            weight_decay=d['weight_decay'],steps_per_epoch=d['steps_per_epoch'],
            parameter_count=d['parameter_count'],lambda_aug=.5,target_domain=0,
            rho_by_physical_shadow=d['rho_by_physical_shadow'],aliases=d['aliases'])
    source_id=audit_sources(m); impl=implementation_identity()
    plans={f.name:sha_file(f) for f in sorted(path.parent.iterdir()) if f.is_file()}
    identity=dict(study_id=m['study_id'],source_commit=SOURCE_REF,source_files_hash=source_id['sha256'],
        implementation_hash=impl['sha256'],manifest_hash=sha_file(path),plan_hash=digest(plans),
        plan_files=plans,plan_ref=PLAN_REF,
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    jobs=[]
    for raw in m['jobs']:
        job=copy.deepcopy(raw); ds=job['dataset'];job['config']=copy.deepcopy(configs[ds])
        if job['epochs']!=50 or job['steps_per_epoch']!=configs[ds]['steps_per_epoch'] or job['adam_steps']!=50*configs[ds]['steps_per_epoch']:
            raise ValueError('Per-job scientific budget changed')
        if job['regular_readouts']!=['online']+list(configs[ds]['rho_by_physical_shadow']) or job['aliases']!=configs[ds]['aliases']:
            raise ValueError('Per-job regular readouts changed')
        online=dict(dataset=ds,objective=job['objective'],seed=job['seed'],
            config={k:v for k,v in configs[ds].items() if k not in ('rho','rho_by_physical_shadow','aliases')},
            source_files_hash=identity['source_files_hash'])
        job['online_trajectory_id']=digest(online);job['identity']=copy.deepcopy(identity)
        jobs.append(job)
    return ResolvedStudy(m,jobs,configs,sha_file(path),identity,{'manifest':str(path),**(paths or {})})


def audit_sources(m=None):
    if m is None: m=read_json(ROOT/'research_plans/26-11-6/experiment_manifest.json')
    if m['source']['implementation_ref']!=SOURCE_REF:
        raise ValueError('Pinned implementation reference changed')
    files={}
    for row in m['source']['audited_file_digests']:
        actual=sha_file(ROOT/row['path'])
        if actual != row['sha256']: raise ValueError('Pinned source changed: '+row['path'])
        raw=(ROOT/row['path']).read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob!=row['git_blob_sha1']: raise ValueError('Pinned source blob changed: '+row['path'])
        files[row['path']]=actual
    return dict(files=files,sha256=digest(files))


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,keys);writer.writeheader();writer.writerows(rows)


def audit(study,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);inventory=[];factorial=[]
    for name,source in study.manifest['source']['archive_sources'].items():
        if 'path' not in source: continue
        path=ROOT/source['path']; exists=path.exists()
        item=dict(source=name,path=source['path'],commit=source['commit'],
            status='AVAILABLE' if exists else 'MISSING_ARCHIVED_ARTIFACT',
            sha256=sha_file(path) if exists else None,git_blob=source.get('git_blob'))
        if exists and source.get('git_blob'):
            actual=subprocess.check_output(['git','hash-object',str(path)],cwd=ROOT,text=True).strip()
            if actual != source['git_blob']: raise ValueError('Archived blob changed: '+source['path'])
        inventory.append(item)
        if exists and name in ('clean','gaussian_d0'):
            with path.open(newline='') as stream:
                for row in csv.DictReader(stream):
                    keep=(row.get('arm')=='A0' and row.get('model_kind') in ('online','fixed_ema')) if name=='clean' else row.get('candidate_id') in ('online','F')
                    if keep: factorial.append(dict(row,objective=name,readout_id='F' if row.get('model_kind')=='fixed_ema' or row.get('candidate_id')=='F' else 'online',archive_path=source['path'],archive_commit=source['commit']))
    write_csv(out/'EVIDENCE_REUSE.csv',inventory);write_csv(out/'ARCHIVED_FACTORIAL.csv',factorial)
    result=dict(identity=study.identity,sources=inventory,model_parameters_available=False,
        missing_observations=['parameter_path_interventions','bounded_per_row_predictions','M0_M3_future_pulse_features'],
        replay_reason='Collect missing observations; archived baseline performance reused',
        independent_seed_count_per_condition=5)
    write_json(out/'AUDIT_INVENTORY.json',result)
    receipt=dict(study.identity,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source=audit_sources(study.manifest),implementation=implementation_identity())
    write_json(out/'SOURCE_RECEIPT.json',receipt)
    from .analysis import audit_archives
    audit_archives(study,out)
    return result


def _npy_content_hash(path, combined=None):
    # Hash payloads without loading Test arrays into a training/analysis bundle.
    with Path(path).open('rb') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0): shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
        else: raise ValueError('Unsupported source NPY version')
        if fortran or dtype.hasobject: raise ValueError('Non-source cache array layout')
        own=hashlib.sha256()
        for accumulator in (own,combined):
            if accumulator is not None:
                accumulator.update(str(dtype).encode('ascii'));accumulator.update(str(tuple(shape)).encode('ascii'))
        for block in iter(lambda:stream.read(8<<20),b''):
            own.update(block)
            if combined is not None: combined.update(block)
    return own.hexdigest(),shape,dtype


def verify_data_identity(study,cache,paths=None):
    paths=paths or {}; cache=Path(cache);result={}
    source=read_json(SOURCE_PROTOCOL)['datasets']
    for ds,d in study.manifest['datasets'].items():
        raw=Path(paths.get(ds,d['csv']))
        if sha_file(raw)!=d['raw_csv_sha256']: raise ValueError('Raw CSV changed: '+ds)
        meta=read_json(cache/ds/'manifest.json');original=source[ds]
        for key in ('raw_sha256','array_hashes','split_row_fingerprints','split_row_id_hashes','encoder_hash','scaler_hash','train_domain_counts','split_rows'):
            if meta[key]!=original[key]: raise ValueError('Data identity changed: '+ds+'/'+key)
        if meta['preprocessing_fit']!='train_only': raise ValueError('Preprocessor leakage')
        expected_arrays={f'{split}_{field}' for split in ('train','val','test') for field in ('x','y','row_ids')}
        if set(meta['cache_sha256'])!=expected_arrays: raise ValueError('Cache array inventory changed')
        for split,label in (('train','train'),('val','validation'),('test','test')):
            combined=hashlib.sha256()
            for field in ('x','y','row_ids'):
                name=f'{split}_{field}'; path=cache/ds/(name+'.npy')
                if sha_file(path)!=meta['cache_sha256'][name]: raise ValueError('Cache bytes changed: '+ds+'/'+name)
                actual,shape,dtype=_npy_content_hash(path,combined)
                target=original['split_row_id_hashes'][label] if field=='row_ids' else original['array_hashes'][name]
                if actual!=target or shape[0]!=original['split_rows'][label]:
                    raise ValueError('Source array content or row count changed: '+ds+'/'+name)
                if field=='row_ids' and dtype!=np.dtype('int64'): raise ValueError('Original row IDs must be int64')
            if combined.hexdigest()!=original['split_row_fingerprints'][label]:
                raise ValueError('Ordered source split identity changed: '+ds+'/'+label)
        bundle=cached_bundle(cache,ds,include_test=False)
        counts=[int((bundle.train_x[:,-1]==domain).sum()) for domain in range(3)]
        if counts!=d['train_domain_counts']: raise ValueError('Natural domain populations changed: '+ds)
        result[ds]=dict(meta,csv=str(raw),cache_root=str(cache),data_hash=digest(meta))
    return result


def preflight(study,out,cache_root=None,ml_csv=None,kr_csv=None):
    from .panels import build_panels, save_panels
    from experiments.aug_history_20261105.model_adapter import build_model
    from experiments.aug_history_20261105.group_registry import build_path_groups
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cache=Path(cache_root) if cache_root else DEFAULT_CACHE
    if cache_root is None and not cache.exists():
        cache=ROOT/study.manifest['runtime']['default_rebuild_data_cache']
    paths={k:v for k,v in dict(movielens=ml_csv,kuairand=kr_csv).items() if v}
    if not all((cache/ds/'manifest.json').exists() for ds in study.configs):
        if shutil.disk_usage(out).free < (2<<30): raise MemoryError('Insufficient cache disk')
        config=dict(datasets={ds:dict(csv=paths.get(ds,d['csv'])) for ds,d in study.manifest['datasets'].items()})
        for ds,d in config['datasets'].items():
            if sha_file(d['csv'])!=study.manifest['datasets'][ds]['raw_csv_sha256']: raise ValueError('Raw SHA changed')
            target=cache/ds
            if target.exists() and not (target/'manifest.json').exists() and any(target.iterdir()):
                raise ValueError('Unmanifested cache files cannot be overwritten: '+str(target))
        prepare_cache(cache,config)
    data=verify_data_identity(study,cache,paths); probe={};groups={}
    rng=capture_rng('cpu')
    try:
        for ds in study.configs:
            bundle=cached_bundle(cache,ds,include_test=False)
            model=build_model(bundle,study.configs[ds]);reg=build_path_groups(model,bundle)
            groups[ds]=dict(reg.as_dict(),functional_support=study.manifest['parameter_groups']['supported_groups_by_cell'],
                           functional_support_source_hash=study.identity['source_files_hash'])
            del model
            probe[ds]={}
            for seed in SEEDS:
                probe[ds][str(seed)]=save_panels(out,ds,seed,build_panels(bundle,seed,study.manifest['panels']))
    finally:
        restore_rng(rng,'cpu')
    for job in study.jobs:
        ds=job['dataset'];panel_hash=digest(probe[ds][str(job['seed'])])
        job.update(data_hash=data[ds]['data_hash'],layout_hash=groups[ds]['layout_hash'],
            group_hash=groups[ds]['group_hash'],probe_hash=panel_hash)
        job['identity'].update(data_hash=job['data_hash'],split_row_hash=data[ds]['split_row_id_hashes'],
            parameter_layout_hash=job['layout_hash'],group_map_hash=job['group_hash'],probe_hash=panel_hash)
        job['identity_id']=digest(job['identity'])
    study.paths.update(manifest=str(Path(study.paths['manifest']).resolve()),
                       cache_root=str(cache.resolve()),out=str(out.resolve()))
    disk=shutil.disk_usage(out)
    if disk.free<study.manifest['resource']['observation_disk_cap_bytes']:
        raise MemoryError('Insufficient disk for bounded mechanism observations')
    environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,
        platform=platform.platform(),cuda=torch.cuda.is_available(),cuda_runtime=torch.version.cuda,
        devices=torch.cuda.device_count(),disk_free_bytes=disk.free)
    if environment['cuda']:
        environment['gpus']=subprocess.check_output(['nvidia-smi','-L'],text=True).splitlines()
    environment['dependencies']=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()
    prepared=dict(identity=study.identity,manifest_hash=study.manifest_hash,paths=study.paths,
        manifest=study.paths['manifest'],jobs=study.jobs,
        configs=study.configs,cache_root=str(cache.resolve()),data_identity=data,panels=probe,groups=groups,
        environment=environment,
        data_cache_bytes=sum(f.stat().st_size for f in cache.rglob('*') if f.is_file()))
    if (out/'PREPARED.json').exists():
        previous=read_json(out/'PREPARED.json')
        comparable=('identity','manifest_hash','paths','jobs','configs','cache_root','data_identity','panels','groups')
        if any(previous[k]!=prepared[k] for k in comparable): raise ValueError('PREPARED identity conflict')
        prepared=previous
    write_json(out/'PREPARED.json',prepared);write_json(out/'RUN_MANIFEST.json',prepared)
    write_json(out/'DATA_MANIFEST.json',data);write_json(out/'PROBE_MANIFEST.json',dict(panels=probe))
    write_json(out/'GROUP_MAP.json',groups);write_json(out/'RESOLVED_PROTOCOL.json',dict(identity=study.identity,
        manifest=study.manifest,manifest_path=study.paths['manifest'],paths=study.paths,configs=study.configs))
    receipt=dict(study.identity,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 source=audit_sources(study.manifest),implementation=implementation_identity())
    write_json(out/'SOURCE_RECEIPT.json',receipt)
    return prepared
