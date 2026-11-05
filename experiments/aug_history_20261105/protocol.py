"""Resolve only the frozen protocol and verify real data identities."""
import copy
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from experiments.mdmt_history_information_20261103 import protocol as history
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31 import audited_data
from experiments.overnight_ab_20261103.data_bridge import cached_bundle
from experiments.overnight_ab_20261103.inventory import read_csv, scalar_row
from experiments.overnight_ab_20261103.state import read_json, write_json, sha_file, digest

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path('/home/xzf/Code/phase0-3/artifacts/mdmt_overnight_ab_20261103/attempt1/data_cache')
SOURCE_REF = 'c8675ae2923c15b6965884b140b5b646690cd95e'
PLAN_REF = '2645f1f724f52f5ed5dea4b479335c0dc12fabcd'
SEEDS = [2022, 2023, 2024, 2025, 2026]
SOURCE_PROTOCOL = history.HERE/'vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31/resolved_protocol.json'


@dataclass
class ResolvedProtocol:
    manifest: dict
    dataset: str
    config: dict
    candidates: list
    manifest_hash: str


def resolve(manifest, dataset, route='B'):
    path = Path(manifest)
    m = read_json(path)
    if dataset not in ('movielens', 'kuairand') or m['seeds'] != SEEDS:
        raise ValueError('Dataset or five-seed contract changed')
    p = m['protocol']
    expected = dict(batch_size=40960, epochs=50, optimizer='torch.optim.Adam', lr=.01,
        betas=[.9, .999], eps=1e-8, fp32=True, amp=False, tf32=False,
        foreach=False, fused=False, amsgrad=False, decoupled_weight_decay=False,
        gradient_clipping=False, lambda_aug=.5, target_domains=[0], early_stopping=False)
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen parent protocol changed')
    d = m['datasets'][dataset]
    c = dict(dataset=dataset, source=read_json(SOURCE_PROTOCOL)['datasets'][dataset],
        **{k: p[k] for k in ('batch_size','epochs','lr','betas','eps','lambda_aug')},
        kappa=d['kappa'], rho=d['rho0'], weight_decay=d['weight_decay'],
        parameter_count=d['parameter_count'], steps_per_epoch=d['steps_per_epoch'],
        eval_batch_size=40960, target_domain=0)
    if route=='B':
        candidates = copy.deepcopy(m['route_B']['uniform_candidates']+m['route_B']['directed_candidates'])
    elif route=='A':
        candidates=[dict(row,family=family) for key,family in
            (('uniform_candidates','U'),('fixed_group_candidates','A_FIXED'),('adaptive_candidates',None))
            for row in copy.deepcopy(m['route_A'][key])]
        for row in candidates:
            if row['family'] is None:
                row['family']='A_GROUP' if row['mode']=='grouped' else 'A_GLOBAL'
    else:
        raise ValueError('Route C is DESIGN_COMPLETE_NOT_IMPLEMENTED')
    if len(candidates) != 15 or len({row['id'] for row in candidates}) != 15:
        raise ValueError('Each implemented route must contain fifteen distinct estimators')
    return ResolvedProtocol(m, dataset, c, candidates, sha_file(path))


def source_manifest():
    files = []
    for package in ('aug_history_20261105','gaussian_domain_coverage_20261104',
                    'mdmt_history_information_20261103','overnight_ab_20261103'):
        for path in sorted((ROOT/'experiments'/package).rglob('*')):
            if path.suffix in ('.py', '.json'):
                files.append(dict(path=str(path.relative_to(ROOT)), sha256=sha_file(path)))
    path = ROOT/'research_plans/26-11-03/experiment_queue.yaml'
    files.append(dict(path=str(path.relative_to(ROOT)), sha256=sha_file(path)))
    return dict(files=files, sha256=digest(files))


def verify_data(resolved, cache=CACHE):
    ds, source = resolved.dataset, resolved.config['source']
    directory = Path(cache)/ds
    manifest = read_json(directory/'manifest.json')
    if sha_file(resolved.manifest['datasets'][ds]['csv']) != source['raw_sha256']:
        raise ValueError('Raw CSV checksum changed: '+ds)
    for key in ('raw_sha256','array_hashes','split_row_fingerprints','split_row_id_hashes',
                'encoder_hash','scaler_hash','train_domain_counts','split_rows'):
        if manifest[key] != source[key]:
            raise ValueError('Cached identity changed: '+key)
    if manifest['preprocessing_fit'] != 'train_only':
        raise ValueError('Preprocessing must be fitted on training rows')
    for split, label in (('train','train'),('val','validation'),('test','test')):
        arrays = {}
        for field in ('x','y','row_ids'):
            name = split+'_'+field
            path = directory/(name+'.npy')
            if sha_file(path) != manifest['cache_sha256'][name]:
                raise ValueError('Cache bytes changed: '+name)
            arrays[field] = np.load(path, mmap_mode='r', allow_pickle=False)
            if field != 'row_ids' and audited_data.array_hash(arrays[field]) != source['array_hashes'][name]:
                raise ValueError('Cache array content changed: '+name)
        if audited_data.row_ids_hash(arrays['row_ids']) != source['split_row_id_hashes'][label]:
            raise ValueError('Row IDs changed: '+label)
        if audited_data.split_row_fingerprint(arrays['x'],arrays['y'],arrays['row_ids']) != source['split_row_fingerprints'][label]:
            raise ValueError('Ordered population changed: '+label)
        if len(arrays['y']) != source['split_rows'][label]:
            raise ValueError('Population count changed')
    bundle = cached_bundle(cache, ds, include_test=False)
    counts = [int((bundle.train_x[:, -1] == domain).sum()) for domain in range(3)]
    if counts != resolved.manifest['datasets'][ds]['train_domain_counts']:
        raise ValueError('Training domain population changed')
    return manifest


def published_reference():
    path = ROOT/'research_reports/2026-11-03_overnight_AB/ENDPOINTS.csv'
    raw = path.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob != 'a3d7aa3e2035f0b7ff55406e260fc59c880ba254':
        raise ValueError('Published endpoint source blob changed')
    rows = [scalar_row(r) for r in read_csv(path)]
    aliases = {'A0':'M3oE_fixed1','HYPERGATE':'HyperGate','AUTOIFS':'AutoIFS'}
    result = []
    for row in rows:
        if row['arm'] not in aliases or row['estimator'] != 'online':
            continue
        if row['protocol_id'] != 'AUC_SELECTED_V1' or row['data_protocol'] != 'FULLTRAIN_NATURAL80_10_10':
            raise ValueError('Published comparison population/selection differs')
        result.append(dict(row, candidate_id=aliases[row['arm']], family='PUBLISHED',
            source_table_sha256=sha_file(path), source_git_blob=blob,
            hypergate_status='UNVERIFIED_PROPOSAL' if row['arm']=='HYPERGATE' else 'SOURCE_ADAPTED'))
    for ds in ('movielens','kuairand'):
        for name in aliases.values():
            for seed in SEEDS:
                for endpoint in ('selected','E10','E50'):
                    peers = [r for r in result if (r['dataset'],r['candidate_id'],r['seed'],r['checkpoint']) == (ds,name,seed,endpoint)]
                    if len(peers)!=2 or {r['split'] for r in peers}!={'validation','test'} or len({(r['epoch'],r['model_hash']) for r in peers})!=1:
                        raise ValueError('Incomplete published endpoint pairing')
    return result, dict(path=str(path), sha256=sha_file(path), git_blob=blob,
        data_protocol='FULLTRAIN_NATURAL80_10_10', selection='AUC_SELECTED_V1',
        primary_reference=dict(movielens='M3oE_fixed1', kuairand='HyperGate'))


def preflight(manifest, output, route='B'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    protocols = {ds: resolve(manifest,ds,route) for ds in ('movielens','kuairand')}
    source = source_manifest()
    data = {ds: verify_data(p) for ds, p in protocols.items()}
    _, reference = published_reference()
    identity = dict(source_hash=source['sha256'], manifest_hash=sha_file(manifest),
        datasets={ds:digest(row) for ds,row in data.items()})
    if route!='B': identity['route']=route
    old = output/'PREPARED.json'
    if old.exists() and read_json(old)['identity'] != identity:
        raise ValueError('Attempt identity differs; use another output path')
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    receipt = dict(plan_ref=PLAN_REF, source_ref=SOURCE_REF, implementation_commit=commit,
        manifest_hash=sha_file(manifest), source=source, published_reference=reference)
    environment = dict(python=sys.version, executable=sys.executable, pytorch=torch.__version__,
        numpy=np.__version__, cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available(),
        platform=platform.platform(), dependencies=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines())
    if torch.cuda.is_available():
        environment['gpus'] = subprocess.check_output(['nvidia-smi','-L'],text=True).splitlines()
    groups={}
    if route=='A':
        from .group_registry import build_path_groups
        from .model_adapter import build_model
        for ds,p in protocols.items():
            history.setup(2022,'cpu')
            bundle=cached_bundle(CACHE,ds,include_test=False)
            groups[ds]=build_path_groups(build_model(bundle,p.config),bundle).as_dict()
        write_json(output/'GROUP_MAP.json',groups)
    jobs = []
    for ds,p in protocols.items():
        for seed in SEEDS:
            job = dict(dataset=ds, seed=seed, route=route, config=p.config,
                config_hash=digest(p.config), manifest_hash=p.manifest_hash, source_hash=source['sha256'],
                data_hash=digest(data[ds]), layout_hash=None, candidates=p.candidates)
            if route=='A':
                job.update(group_hash=groups[ds]['group_hash'],layout_hash=groups[ds]['layout_hash'])
            key = digest(job)
            job.update(job_id=key, trajectory_id=key)
            jobs.append(job)
    prepared = dict(identity=identity, source_receipt=receipt, environment=environment,
        jobs=jobs, cache_root=str(CACHE), manifest=str(Path(manifest).resolve()))
    if route!='B': prepared['route']=route
    write_json(output/'SOURCE_RECEIPT.json',receipt)
    write_json(output/'RESOLVED_PROTOCOL.json',{ds:p.config for ds,p in protocols.items()})
    write_json(output/'DATA_MANIFEST.json',data)
    write_json(output/'ENVIRONMENT.json',environment)
    write_json(output/'RUN_MANIFEST.json',prepared)
    write_json(old,prepared)
    return prepared
