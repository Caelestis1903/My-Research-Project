"""Cycle04 identities, Test-opaque data, numeric controls and RAM consumer contracts."""
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from experiments.auxiliary_objective_attribution_20261115.protocol import (
    SELECTION, digest, optimizer, load_protocol, make_model, setup, cpu_copy,
    state_digest, rng_state, restore_rng, evaluate, sha_file)
from experiments.conditional_information_momentum_20261102_v3.guardian import request as storage_request
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.audited_data import (
    array_hash, split_row_fingerprint, row_ids_hash)

ROOT = Path('/home/xzf/Code/phase0-3_tvg_rm_20261116')
ORIGINAL = Path('/home/xzf/Code/phase0-3')
REL_ARTIFACT = Path('artifacts/26-11-16_augmentation_gradient_momentum')
A4 = ROOT/REL_ARTIFACT/'astra_cycles/cycle_04'
P4 = ROOT/'research_plans/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04'
ATTEMPT = A4/'attempt2'
HERE = ROOT/'experiments/completion_weight_attribution_20261117'
A3 = ROOT/REL_ARTIFACT/'astra_cycles/cycle_03/attempt1'
DEFAULT_CACHE = Path('/home/xzf/Code/phase0-3-20261115/artifacts/heterogeneity_aware_20261115/attempt1/data_cache')
REFERENCE = A3/'reused_reference_index.json'
CONTROLS_HASH = 'e8d67c6b18ad9ddbfa19b752e35c6ab051e3170a3d28df635b86b27922441fdb'
DESIGN_HASH = '2acde7e9e6764f6243920cba3f538e2c4cd49b36b6ddfa2351addba3b9fc62b7'
RESEARCH_HASH = 'a3648198e25f6929154b628880715c7b4eb7d9749fc8b178a1631722debba01a'
PYTHON = '/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python'
SEEDS = (2022, 2023, 2024, 2025, 2026)
DATASETS = ('movielens', 'kuairand')
ARMS = ('COMPLETION-BCE-025', 'MASS-SCALE-025')
ESTIMATORS = ('online', 'fixed_ema')
CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))
DIAGNOSTIC_EPOCHS = (1, 10, 30, 50)
ROWS = dict(movielens=800167, kuairand=2006132)
VAL_ROWS = dict(movielens=100021, kuairand=250767)
STEPS = dict(movielens=20, kuairand=49)
TAILS = dict(movielens=21927, kuairand=40052)
OPERATIONS = ('training_rows','task_entries','ordinary_forwards','scalar_backwards',
              'live_adam_steps','live_ema_steps','diagnostic_points','extra_autograd_sweeps',
              'virtual_adam_steps','probe_forwards','view_forwards','algorithm_rng_draws')
DIAG_OPERATIONS = ('diagnostic_points','extra_autograd_sweeps','virtual_adam_steps')
ENGINEERING_PLAN = dict(physical_train_calls=8,diagnostic_points=2,extra_autograd_sweeps=6,virtual_adam_steps=8)
SERVER = ROOT/'experiments/task_view_gradient_residual_momentum_20261116/runner.py'
RAW = dict(movielens=(Path('/home_new/xzf/datasets/ml-1m/ml-1m.csv'),
                      'f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4'),
           kuairand=(Path('/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv'),
                     'a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f'))


class TechnicalPause(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise TechnicalPause(reason)


def read_json(path):
    return json.loads(Path(path).read_text())


def new_root(attempt=ATTEMPT):
    p = Path(attempt)
    require(p.is_absolute() and p == ATTEMPT and p.resolve() == ATTEMPT,
            'Only canonical absolute Cycle04 attempt2 is authorized')
    return p


def new_path(path):
    p = Path(path)
    require(p.is_absolute() and p.resolve().is_relative_to(new_root()), 'Write outside Cycle04 attempt2')
    return p


def json_safe(value):
    if torch.is_tensor(value) or isinstance(value, (np.ndarray, bytes, bytearray)):
        raise TechnicalPause('Tensor/array/bytes cannot enter numeric disk records')
    if isinstance(value, dict):
        for item in value.values():
            json_safe(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            json_safe(item)
    elif isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite numeric record')
    elif value is not None:
        require(isinstance(value, (str, int, bool)), 'Unsupported numeric record type')
    return value


def write_json(path, value):
    p = new_path(path); json_safe(value); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def immutable(path, value):
    p = new_path(path)
    if p.exists():
        require(read_json(p) == value, f'Immutable receipt differs: {p.name}')
    else:
        json_safe(value); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def append_json(path, value):
    p = new_path(path); json_safe(value); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        f.write(json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n')


def design_bundle():
    a = read_json(A4/'DESIGN_ACCEPTANCE.json')
    require(a['verdict'] == 'COMPLETE_APPROVED_FOR_IMPLEMENTATION' and not a['blocking_issues'], 'Design not approved')
    names = ('ALGORITHM_DESIGNS.md','CODEX_EXECUTION_PLAN.md','CRITICAL_REVIEW.md',
             'EXPERIMENT_MANIFEST.json','LITERATURE_AND_NOVELTY.md','RESEARCH_EXPLORATION.md')
    mapping = {n: str(P4/n) for n in names}
    mapping['DESIGN_SPECIFICATION.md'] = str(A4/'DESIGN_SPECIFICATION.md')
    require(a['logical_name_to_absolute_path'] == mapping, 'Design path provenance differs')
    files = sorted([[n, sha_file(p)] for n,p in mapping.items()])
    compact = lambda pairs: hashlib.sha256(json.dumps(pairs,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    require(files == a['files'] and compact(files) == a['design_bundle_sha256'] == DESIGN_HASH, 'Seven-design hash differs')
    require(compact([x for x in files if x[0] != 'DESIGN_SPECIFICATION.md']) == a['research_bundle_sha256'] == RESEARCH_HASH, 'Research hash differs')
    handoff = read_json(A4/'ASTRA_HANDOFF.json')
    require(handoff['context_closed'] and handoff['research_bundle_sha256'] == RESEARCH_HASH, 'Astra handoff differs')
    require(sha_file(A4/'ASTRA_HANDOFF.json') == a['astra_handoff_sha256'], 'Astra handoff bytes differ')
    return dict(files=files, logical_name_to_absolute_path=mapping, sha256=DESIGN_HASH,
                research_bundle_sha256=RESEARCH_HASH, astra_handoff_sha256=a['astra_handoff_sha256'])


def source_paths():
    aux = 'experiments/auxiliary_objective_attribution_20261115'
    cond = 'experiments/conditional_information_momentum_20261102_v3'
    g = cond+'/vendor/ema02/vendor/gradient_aug_20261031_v2'; v = g+'/vendor_ga31'
    groups = {
        'experiments/completion_weight_attribution_20261117': ('__init__.py','core.py','checks.py','protocol.py','runner.py','tables.py','README.md'),
        aux: ('__init__.py','core.py','protocol.py','runner.py','stage2/__init__.py','stage2/core.py'),
        cond: ('__init__.py','guardian.py','protocol.py','vendor/__init__.py','vendor/ema02/__init__.py',
               'vendor/ema02/data.py','vendor/ema02/evaluate.py','vendor/ema02/protocol.py','vendor/ema02/vendor/__init__.py'),
        g: ('__init__.py','protocol.py','views.py'),
        v: ('__init__.py','audited_data.py','metrics.py','model.py','protocol.py','reference.py','source_protocol.py',
            'basic/__init__.py','basic/activation.py','basic/features.py','basic/initializers.py','basic/layers.py',
            'utils/__init__.py','utils/data.py','resolved_protocol.json'),
        'experiments/natural_quadratic_ranking_20261116': ('__init__.py','core.py'),
        'experiments/task_view_gradient_residual_momentum_20261116': ('__init__.py','core.py','protocol.py','runner.py','autonomous.py')}
    return sorted([f'{base}/{name}' for base,names in groups.items() for name in names]+['requirements-phase0-3.lock'])


def source_manifest():
    files = [dict(path=p, sha256=sha_file(ROOT/p)) for p in source_paths()]
    adam = Path(inspect.getfile(torch.optim.Adam))
    require(sha_file(adam)=='b099132d262d6cb16eefa48ab4bae789a0655cafb8571e421ef4fe3d2ff7c34c', 'Installed source Adam implementation differs')
    return dict(files=files, sha256=digest(files), external_runtime=[dict(path=str(adam),sha256=sha_file(adam))])


def config(dataset, arm):
    require(dataset in DATASETS and arm in ARMS, 'Unknown dataset/arm')
    d = load_protocol()['datasets'][dataset]
    return dict(cycle_id='cycle_04',dataset=dataset,arm=arm,objective='completion_weight_attribution_v1',
                mixture_lambda=.25,epochs=50,batch_size=40960,precision='float32',lr=.01,betas=[.9,.999],eps=1e-8,
                weight_decay=d['weight_decay'],rho=d['rho'],steps_per_epoch=STEPS[dataset],tail_rows=TAILS[dataset],
                loader_seed_offset=11000,gate_training_detached=True,probe_gate_full_derivative=False,
                algorithm_rng_draws=0,data_hash=digest(d['source']),selection_hash=digest(SELECTION))


def job(dataset, arm, seed, source_hash, purpose='formal', case_id=None):
    require(type(seed) is int and seed in SEEDS, 'Cycle04 requires all five registered seeds')
    require(purpose in ('formal','engineering_smoke'), 'Invalid job purpose')
    require((purpose == 'formal' and case_id is None) or (purpose == 'engineering_smoke' and bool(case_id)), 'Engineering case identity required')
    c = config(dataset,arm)
    j = dict(cycle='cycle_04',dataset=dataset,arm=arm,seed=seed,config=c,config_hash=digest(c),source_hash=source_hash,
             design_bundle_sha256=DESIGN_HASH,controls_index_hash=CONTROLS_HASH,purpose=purpose)
    if case_id is not None:
        j['case_id'] = case_id
    j['key'] = digest(j)
    return j


def _verify_arrays(bundle, manifest, expected, dataset):
    # Only these attributes are dereferenced; Test remains opaque in the pickle container.
    values = {name: getattr(bundle,name) for name in ('train_x','train_y','train_row_ids','val_x','val_y','val_row_ids')}
    for prefix, split in (('train','train'),('val','validation')):
        x,y,ids = (values[prefix+suffix] for suffix in ('_x','_y','_row_ids'))
        for name,arr in ((prefix+'_x',x),(prefix+'_y',y)):
            require(array_hash(arr) == expected['array_hashes'][name] == manifest['array_hashes'][name], f'{dataset}/{name} hash')
        require(split_row_fingerprint(x,y,ids) == expected['split_row_fingerprints'][split] == manifest['split_row_fingerprints'][split], f'{dataset}/{split} row fingerprint')
        require(row_ids_hash(ids) == expected['split_row_id_hashes'][split] == manifest['split_row_id_hashes'][split], f'{dataset}/{split} IDs')
        require(x.shape == (ROWS[dataset] if split=='train' else VAL_ROWS[dataset],len(bundle.features.columns)), 'Input shape')
        require(y.shape == (len(x),2) and ids.shape == (len(x),), 'Label/ID shape')
        require(x.dtype == y.dtype == np.float32 and ids.dtype == np.int64, 'Source array dtype')
        require(np.isfinite(x).all() and np.isin(y,(0,1)).all(), 'Invalid natural arrays')
        require(np.isin(x[:,-1],(0,1,2)).all(), 'Invalid source domains')
        require(len(x) == manifest['split_rows'][split] == expected['split_rows'][split], 'Split rows')
    features = bundle.features.as_dict()
    require(features == manifest['features'], 'Feature schema differs')
    for name in ('encoder_hash','scaler_hash'):
        require(features[name] == expected[name] == manifest[name], f'{name} differs')
    x,y = values['train_x'],values['train_y']
    counts = [int((x[:,-1]==d).sum()) for d in range(3)]
    require(counts == expected['train_domain_counts'] == manifest['train_domain_counts'], 'Natural domain counts differ')
    for d in range(3):
        labels = y[x[:,-1]==d]
        joint = {f'{a}{b}':int(((labels[:,0]==a)&(labels[:,1]==b)).sum()) for a in (0,1) for b in (0,1)}
        require(joint == manifest['train_joint_labels'][str(d)], 'Natural class counts differ')
    return SimpleNamespace(**values,features=bundle.features)


def dataset_cache(dataset, cache_dir=DEFAULT_CACHE):
    require(dataset in DATASETS, 'Invalid dataset')
    path, expected_raw = RAW[dataset]
    require(path.is_file() and sha_file(path) == expected_raw, f'{dataset} raw path/hash missing or mismatched')
    cache = Path(cache_dir)/f'{dataset}.pkl'
    require(cache.is_file(), f'Required prepared cache missing: {cache}')
    with cache.open('rb') as f:
        bundle, manifest = pickle.load(f)
    expected = load_protocol()['datasets'][dataset]['source']
    recorded = read_json(A3/f'{dataset}_data_manifest.json')
    require(manifest == recorded and manifest['raw_sha256'] == expected_raw == expected['raw_sha256'], 'Recorded data/config provenance differs')
    data = _verify_arrays(bundle,manifest,expected,dataset)
    return data, manifest


def load_reference_index():
    require(sha_file(REFERENCE) == CONTROLS_HASH, 'Numeric controls index bytes differ')
    i = read_json(REFERENCE)
    require(i['test_accessed'] is False and i['selection_hash'] == digest(SELECTION), 'Control selection/Test provenance differs')
    require(len(i['entries'])==20 and len(i['initial_hashes'])==10 and len(i['epoch_order_hashes'])==500, 'Incomplete controls index')
    return i


def receipt_json(receipt):
    p = Path(receipt['path'])
    require(p.is_file() and sha_file(p) == receipt['sha256'], f'Missing/mismatched numeric receipt: {p}')
    if 'bytes' in receipt:
        require(p.stat().st_size == receipt['bytes'], 'Numeric receipt byte count differs')
    return read_json(p)


def verify_selection(result, epochs):
    require(set(epochs) == set(range(1,51)), 'Need exact epochs1..50')
    selected = {}; best = {k:-1. for k in ESTIMATORS}
    for e in range(1,51):
        r = epochs[e][1] if isinstance(epochs[e],tuple) else epochs[e]
        require(type(r['epoch']) is int and r['epoch']==e, 'Epoch identity differs')
        for k in ESTIMATORS:
            v = r['validation'][k]['macro_auc']
            require(isinstance(v,(int,float)) and math.isfinite(v), 'Undefined selection AUC')
            if v > best[k]:
                best[k]=v; selected[k]=e
    require(selected == result['selected_epochs'], 'Earliest strict maximum differs')
    for k,e in selected.items():
        r = epochs[e][1] if isinstance(epochs[e],tuple) else epochs[e]
        require(result['selected_validation'][k] == r['validation'][k], 'Selected validation/state metric binding differs')
        require(result['selected_hashes'][k] == r['model_hash' if k=='online' else 'ema_hash'], 'Selected state hash differs')
    return selected


def reference_index():
    i = load_reference_index(); seen = set(); verified=0
    require(sha_file(i['source_index_path'])==i['source_index_sha256'], 'Control source index differs')
    op = i['shared_operator_provenance']
    require(sha_file(op['pilot_source_receipt_path'])==op['pilot_source_receipt_sha256'], 'Shared operator source receipt differs')
    pilot = read_json(op['pilot_source_receipt_path'])
    require(pilot['sha256']==digest(pilot['files']), 'Control operator source recipe differs')
    pilot_manifest=Path(op['pilot_source_receipt_path']).with_name('run_manifest.json')
    require(sha_file(pilot_manifest)==op['pilot_run_manifest_sha256'], 'Control operator run provenance differs')
    pilot_files = {f['path']:f['sha256'] for f in pilot['files']}
    for f in op['shared_conditional_files']:
        require(pilot_files[f['path']]==f['sha256'], 'Shared operator provenance differs')
    for entry in i['entries']:
        d,a,s=entry['dataset'],entry['arm'],entry['seed']
        require(d in DATASETS and a in ('CLEAN','NQR-025') and s in SEEDS and (d,a,s) not in seen, 'Unexpected control coverage')
        seen.add((d,a,s)); r=receipt_json(entry['result']); verified+=1
        require(r['epochs']==50 and r['global_step']==50*STEPS[d] and r['test_accessed'] is False, 'Control completion differs')
        require(r['counts']==dict(optimizer_updates=50*STEPS[d],ema_updates=50*STEPS[d]), 'Control counts differ')
        j=r['job']; c=j['config']; actual=config(d,ARMS[0])
        require(j['dataset']==d and j['seed']==s and j['key']==entry['key'] and c['arm']==a, 'Control job differs')
        for field in ('epochs','batch_size','lr','weight_decay','rho','gate_training_detached'):
            require(c[field]==actual[field], f'Control active config differs: {field}')
        if a=='CLEAN':
            require(j['source_hash']==pilot['sha256'] and c['clean_loss']=='mean_probability_BCE', 'CLEAN source operator differs')
        else:
            require(j['source_hash']=='ddaf5d97b676c4c8ba06c40cb2a7aa2209ae27148950b848329e1dc8bb669bae' and
                    j['config_hash']==digest(c), 'NQR frozen source/config differs')
            for field in ('precision','betas','eps','steps_per_epoch','tail_rows','loader_seed_offset','data_hash','selection_hash'):
                require(c[field]==actual[field],f'NQR active shared field differs: {field}')
            require(c['objective']=='natural_quadratic_ranking_v1' and c['mixture_lambda']==.25 and
                    c['completed_risk']=='Q_R' and c['probability_margin']==1., 'NQR objective differs')
        require(r['initial_hash']==i['initial_hashes'][f'{d}/{s}'], 'Control initialization differs')
        require(len(entry['epochs'])==50 and digest(entry['epochs'])==entry['epoch_receipts_sha256'], 'Control epoch receipts differ')
        epochs={}
        for er in entry['epochs']:
            e=er['epoch']; q=receipt_json(er); verified+=1
            require(e not in epochs and type(e) is int and q['epoch']==e and q['global_step']==e*STEPS[d], 'Control epoch clock differs')
            require(q['epoch_order_hash']==i['epoch_order_hashes'][f'{d}/{s}/{e}'], 'Control source row order differs')
            epochs[e]=q
        verify_selection(r,epochs)
    require(verified==1020, 'Required1020 numeric control files incomplete')
    return i


def stop_requested():
    for root in (ROOT,ORIGINAL):
        if any(p.exists() for p in (root/'STOP_AUTONOMOUS',root/REL_ARTIFACT/'STOP_AUTONOMOUS')):
            return True
        s=read_json(root/REL_ARTIFACT/'AUTONOMOUS_STATE.json')
        if s.get('manual_stop_requested') or any(s.get(k)=='USER_STOPPED' for k in ('status','reason','termination_reason')):
            return True
    return False


def environment():
    require(sys.version.split()[0]=='3.13.5' and torch.__version__=='2.9.0+cu126', 'Environment version differs')
    require(str(Path(sys.executable))==PYTHON, 'Wrong project Python')
    lock=ROOT/'requirements-phase0-3.lock'
    for line in lock.read_text().splitlines():
        if '==' in line and not line.startswith(('#','--')):
            name,version=line.split('==',1)
            require(importlib.metadata.version(name)==version, f'Package differs: {name}')
    return dict(python_executable=PYTHON,python_version=sys.version.split()[0],torch_version=torch.__version__,
                package_lock_sha256=sha_file(lock),cpu_threads=1,autocast=False,cuda_matmul_tf32=False,
                cudnn_tf32=False,cudnn_benchmark=False)


def check_review(path, kind='code', source_hash=None):
    r=read_json(path); source_hash=source_hash or source_manifest()['sha256']
    require(r['verdict'] in ('PASS','PASS WITH FIXES') and not r.get('blocking_issues'), 'Review has blockers')
    require(r['source_hash']==source_hash and r['design_bundle_sha256']==DESIGN_HASH, 'Review source/design differs')
    require(r['test_accessed'] is False, 'Review Test flag differs')
    if kind=='code':
        coders=r['coder_ids']; reviewer=r['reviewer_id']
        require(len(set(coders))>=2 and reviewer not in coders, 'Independent review must exclude every contributing coder')
    else:
        code=read_json(read_json(ATTEMPT/'run_manifest.json')['code_review_path'])
        require(r['reviewer_id'] not in code['coder_ids'] and r['reviewer_id']!=code['reviewer_id'], 'Fresh independent resource/recovery reviewer required')
    return r


def owner_binding(attempt=ATTEMPT):
    a=new_root(attempt); g=read_json(a/'guardian.json'); b=read_json(a/'STORAGE_OWNER.json'); s=read_json(a/'source_receipt.json')
    require((a/'guardian.json').stat().st_mode & 0o777 == 0o600, 'Guardian credentials mode differs')
    endpoint=digest(dict(address=g['address'],pid=g['pid']))
    require(b['pid']==g['pid'] and b['endpoint_hash']==endpoint and b['source_hash']==s['sha256'] and
            b['design_bundle_sha256']==DESIGN_HASH and b['server_file_sha256']==sha_file(SERVER), 'Owner/source/design/server binding differs')
    require(bool(b['tool_session']) and b['pid'] not in (1025015,1570727,1944719,2097722), 'New foreground owner required')
    if 'address' in b:
        require(b['address']==g['address'], 'Local owner endpoint differs')
        require((a/'STORAGE_OWNER.json').stat().st_mode & 0o777 == 0o600, 'Local endpoint binding must be private')
    require(Path(f'/proc/{b["pid"]}/stat').is_file(), 'New RAM owner lost')
    return dict(pid=b['pid'],tool_session=b['tool_session'],endpoint_hash=endpoint,
                source_hash=b['source_hash'],design_bundle_sha256=DESIGN_HASH,server_file_sha256=b['server_file_sha256'])


def request(attempt, command, **payload):
    a=new_root(attempt); owner=owner_binding(a)
    require(command in ('status','register','allocated','commit','get','paused','failed','recover','complete','release'), 'Unsupported storage command')
    if command not in ('status',):
        require(isinstance(payload.get('key'),str) and len(payload['key'])==64 and
                all(c in '0123456789abcdef' for c in payload['key']), 'Invalid storage key')
        j=read_json(a/'jobs'/f'{payload["key"]}.json')
        require(j['source_hash']==owner['source_hash'] and j['design_bundle_sha256']==DESIGN_HASH, 'Storage job identity differs')
        require(j==job(j['dataset'],j['arm'],j['seed'],j['source_hash'],j['purpose'],j.get('case_id')), 'Storage job digest/config differs')
        if command=='register':
            require(payload['job']==j, 'Register job differs')
        if command=='commit':
            validate_epoch_state(payload['state'],None)
            require(payload['state']['job']==j, 'Commit job differs')
            new_path(a/payload['epoch_file'])
        if command=='complete':
            require(j['purpose']=='formal' and payload['result']['epochs']==50, 'Engineering cannot complete')
        if command=='release':
            acceptance=read_json(a/'ENGINEERING_RELEASE_ACCEPTANCE.json')
            require(j['purpose']=='engineering_smoke' and payload['key'] in acceptance['keys'] and
                    len(acceptance['keys'])==4 and acceptance['parent_accepted'] is True, 'Only four accepted engineering keys can be released')
            check_review(acceptance['review_path'],'engineering',owner['source_hash'])
    answer=storage_request(a,command,**payload)
    if command=='status':
        require(answer['pid']==owner['pid'], 'Live owner identity differs')
    if command in ('get','register') and answer.get('state') is not None:
        state=answer['state']
        if 'job' in state:
            validate_epoch_state(state,None)
            require(state['job']==j and state['owner_endpoint_hash']==owner['endpoint_hash'], 'Returned state job/owner differs')
        else:
            require(command=='get' and set(state)=={'selected_models','selected_epochs','selected_validation'} and
                    answer.get('result',{}).get('job')==j, 'Unexpected compacted state')
    return answer


def validate_formal_clock(snapshot, dataset=None):
    d=dataset or snapshot['job']['dataset']; e=snapshot['epoch']; k=STEPS[d]
    require(type(e) is int and 0<=e<=50 and snapshot['purpose']=='formal' and snapshot['clock_kind']=='formal', 'Formal clock kind/epoch')
    require(snapshot['global_step']==e*k and snapshot['next_epoch']==e+1 and snapshot['scientific_epochs_completed']==e, 'Formal whole-epoch clock differs')
    require(snapshot['counts']==dict(optimizer_updates=e*k,ema_updates=e*k), 'Counts must have exactly two compatible keys')
    o=snapshot['operations']; n=sum(x<=e for x in DIAGNOSTIC_EPOCHS)
    expected=dict(training_rows=e*ROWS[d],task_entries=2*e*ROWS[d],ordinary_forwards=e*k,scalar_backwards=e*k,
                  live_adam_steps=e*k,live_ema_steps=e*k,diagnostic_points=n,extra_autograd_sweeps=3*n,
                  virtual_adam_steps=4*n,probe_forwards=0,view_forwards=0,algorithm_rng_draws=0)
    require(o==expected and all(type(v) is int for v in o.values()), 'Formal logical operations differ')
    require(snapshot['job']['config']==config(d,snapshot['job']['arm']), 'Formal configuration differs')
    return True


IDENTITY_FIELDS = ('job','source_hash','design_bundle_sha256','controls_index_hash','owner_endpoint_hash',
                   'model','ema','optimizer','rng','loader','source_modes','selected_models','selected_epochs',
                   'selected_validation','best','common_validation','initial_hash','initial_ema_hash','initial_optimizer_hash',
                   'purpose','clock_kind','epoch','global_step','next_epoch','scientific_epochs_completed',
                   'engineering_commit_ordinal','engineering_state_step','sample_epoch','sample_batches','sample_order',
                   'selection_clock','counts','adam_step_counters','last_parameter_activity')


def scientific_identity(snapshot):
    identity={k:snapshot[k] for k in IDENTITY_FIELDS if k in snapshot}
    if 'branch_counts' in snapshot: identity['branch_counts']=snapshot['branch_counts']
    identity['operations']={k:snapshot['operations'][k] for k in OPERATIONS if k not in DIAG_OPERATIONS}
    return state_digest(identity)


def component_hashes(snapshot):
    return dict(model_hash=state_digest(snapshot['model']),ema_hash=state_digest(snapshot['ema']),
                optimizer_hash=state_digest(snapshot['optimizer']),rng_hash=state_digest(snapshot['rng']),
                loader_hash=state_digest(snapshot['loader']),
                selected_hashes={k:state_digest(v) for k,v in snapshot['selected_models'].items()})


def validate_epoch_state(snapshot, record):
    j=snapshot['job']; require(snapshot['source_hash']==j['source_hash'] and snapshot['design_bundle_sha256']==j['design_bundle_sha256']==DESIGN_HASH and
                             snapshot['controls_index_hash']==j['controls_index_hash']==CONTROLS_HASH, 'Snapshot provenance differs')
    require(set(snapshot['counts'])=={'optimizer_updates','ema_updates'} and all(type(v) is int for v in snapshot['counts'].values()), 'Two-key counts required')
    require(set(snapshot['operations'])==set(OPERATIONS) and all(type(v) is int and v>=0 for v in snapshot['operations'].values()), 'Operation schema differs')
    if j['purpose']=='formal':
        validate_formal_clock(snapshot)
    else:
        require(snapshot['purpose']=='engineering_smoke' and snapshot['clock_kind']=='engineering' and
                snapshot['scientific_epochs_completed']==0 and snapshot['next_epoch'] is None, 'Engineering scientific clock differs')
        ordinal=snapshot['engineering_commit_ordinal']; step=snapshot['engineering_state_step']
        require(ordinal in (1,2) and snapshot['epoch']==ordinal and step==2*ordinal==snapshot['global_step'], 'Engineering storage ordinal differs')
        require(snapshot['counts']==dict(optimizer_updates=step,ema_updates=step), 'Engineering logical counts differ')
    def cpu_only(x):
        if torch.is_tensor(x):
            require(x.device.type=='cpu', 'RAM snapshot must be CPU')
        elif isinstance(x,dict):
            for v in x.values(): cpu_only(v)
        elif isinstance(x,(list,tuple)):
            for v in x: cpu_only(v)
    cpu_only(snapshot)
    if record is not None:
        for k,v in component_hashes(snapshot).items():
            require(record[k]==v, f'{k} differs')
        require(record['continuation_hash']==state_digest(snapshot) and record['scientific_identity_hash']==scientific_identity(snapshot), 'Snapshot identity differs')
        for k in ('job','epoch','global_step','counts','operations','source_hash','design_bundle_sha256','controls_index_hash','owner_endpoint_hash'):
            require(record[k]==snapshot[k], f'Record/snapshot {k} differs')
        require(record['test_accessed'] is False, 'Test sealed')
    return True


def prepare(attempt, review_path):
    a=new_root(attempt); require(not stop_requested(), 'USER_STOPPED')
    design=design_bundle(); source=source_manifest(); check_review(review_path,'code',source['sha256'])
    env=environment(); i=reference_index()
    require(env['package_lock_sha256']==i['environment']['package_lock_receipt']['sha256'], 'Control environment lock differs')
    data={}
    for d in DATASETS:
        _,manifest=dataset_cache(d); data[d]=dict(path=str(A3/f'{d}_data_manifest.json'),sha256=sha_file(A3/f'{d}_data_manifest.json'),verified_splits=['train','validation'])
    a.mkdir(parents=True,exist_ok=True)
    immutable(a/'source_receipt.json',source)
    immutable(a/'run_manifest.json',dict(source_hash=source['sha256'],design_bundle_sha256=design['sha256'],
        research_bundle_sha256=RESEARCH_HASH,astra_handoff_sha256=design['astra_handoff_sha256'],controls_index_hash=CONTROLS_HASH,
        environment=env,selection=SELECTION,selection_hash=digest(SELECTION),data=data,cache_dir=str(DEFAULT_CACHE),
        code_review_path=str(Path(review_path).resolve()),code_review_sha256=sha_file(review_path),
        seeds=list(SEEDS),datasets=list(DATASETS),arms=list(ARMS),new_trajectories=20,reused_trajectories=20,
        epochs=1000,optimizer_updates=34500,ema_updates=34500,training_rows=1403149500,task_entries=2806299000,
        diagnostic_points=80,extra_autograd_sweeps=240,virtual_adam_steps=320,validation_chunks=10000,
        selected_tensor_bytes=191017920,model_storage='RAM_ONLY',test_accessed=False))
    for d in DATASETS:
        immutable(a/f'{d}_data_manifest.json',read_json(A3/f'{d}_data_manifest.json'))
    for d in DATASETS:
        for seed in SEEDS:
            for arm in ARMS:
                j=job(d,arm,seed,source['sha256']); immutable(a/'jobs'/f'{j["key"]}.json',j)
    immutable(a/'CONTROLS_VERIFIED.json',dict(index_path=str(REFERENCE),sha256=CONTROLS_HASH,verified_files=1020,
                                            numeric_only=True,test_accessed=False))
    return dict(status='PREPARED',source_hash=source['sha256'],design_bundle_sha256=DESIGN_HASH,test_accessed=False)


def engineering_physical_counts(job,receipts):
    calls=[]
    for receipt in receipts:
        path=Path(receipt['path']); require(sha_file(path)==receipt['sha256'],'Physical call history changed')
        calls.extend(json.loads(line) for line in path.read_text().splitlines())
    require(all(x['key']==job['key'] and x['source_hash']==job['source_hash'] and x['design_bundle_sha256']==DESIGN_HASH and
                type(x['attempt']) is int and x['attempt']>0 for x in calls),'Physical call provenance differs')
    return dict(physical_train_calls=len(calls),diagnostic_points=sum(x['diagnose'] for x in calls),
        extra_autograd_sweeps=sum(x.get('value',{}).get('extra_autograd_sweeps',0) for x in calls),
        virtual_adam_steps=sum(x.get('value',{}).get('virtual_adam_steps',0) for x in calls))


def validate_attempt(attempt=ATTEMPT, require_engineering=False):
    a=new_root(attempt); m=read_json(a/'run_manifest.json'); source=source_manifest(); design_bundle()
    require(read_json(a/'source_receipt.json')==source and m['source_hash']==source['sha256'] and m['design_bundle_sha256']==DESIGN_HASH and
            m['controls_index_hash']==CONTROLS_HASH, 'Frozen attempt identity differs')
    require(sha_file(m['code_review_path'])==m['code_review_sha256'], 'Code review changed')
    check_review(m['code_review_path'],'code',source['sha256'])
    require(read_json(a/'CONTROLS_VERIFIED.json')['verified_files']==1020 and sha_file(REFERENCE)==CONTROLS_HASH, 'Controls preparation missing')
    if require_engineering:
        signoff=read_json(a/'ENGINEERING_SIGNOFF.json'); check_review(a/'ENGINEERING_SIGNOFF.json','engineering',source['sha256'])
        require(len(signoff['cases'])==4 and signoff['verified_plan_counts']=={k:4*v for k,v in ENGINEERING_PLAN.items()} and
                signoff['scientific_epochs_completed']==0, 'Engineering verified plan/signoff differs')
        for receipt in signoff['cases']+signoff['profiles']:
            receipt_json(receipt)
        cases=[receipt_json(r) for r in signoff['cases']]
        required={(0,'movielens',ARMS[0]),(0,'kuairand',ARMS[1]),(1,'movielens',ARMS[1]),(1,'kuairand',ARMS[0])}
        require({(r['device'],r['dataset'],r['arm']) for r in cases}==required and len(signoff['profiles'])==4, 'Engineering matrix differs')
        for r in cases:
            require(r['status']=='ENGINEERING_VERIFIED_PENDING_REVIEW' and r['verified_plan_counts']==ENGINEERING_PLAN and
                    all(type(r[k]) is int and r[k]>=v for k,v in ENGINEERING_PLAN.items()) and
                    r['scientific_epochs_completed']==0 and r['prefix_immutable'] and r['diagnostics_neutral_exact'] and
                    r['source_lambda_zero_exact'] and r['source_hash']==source['sha256'] and r['design_bundle_sha256']==DESIGN_HASH,
                    'Engineering case identities/accounting differ')
            proof=receipt_json(r['proof'])
            require(proof['job']==r['job'] and proof['suffix_scientific_identity_hash']==r['suffix_scientific_identity_hash'], 'Engineering proof binding differs')
            require(engineering_physical_counts(r['job'],r['physical_call_logs'])=={k:r[k] for k in ENGINEERING_PLAN},
                    'Engineering actual physical accounting differs')
            profile=receipt_json(r['profile'])
            require(profile['source_hash']==source['sha256'] and profile['design_bundle_sha256']==DESIGN_HASH and
                    profile['dataset']==r['dataset'] and profile['arm']==r['arm'] and profile['device']==r['device'] and
                    profile['environment']==m['environment'] and profile['full_process_peak_bytes']>0 and profile['samples'], 'Actual measured case profile required')
        require(all(signoff[k]==sum(r[k] for r in cases) for k in ENGINEERING_PLAN), 'Engineering actual physical totals differ')
        released=read_json(a/'ENGINEERING_RELEASE_ACCEPTANCE.json')
        require(released['parent_accepted'] is True and set(released['keys'])=={receipt_json(r)['key'] for r in signoff['cases']}, 'Engineering release acceptance missing')
        status=request(a,'status')
        require(all(status['jobs'][k].get('state_released') for k in released['keys']), 'Release four accepted engineering keys before formal admission')
    return m


def epoch_files(attempt, result):
    a=new_root(attempt); key=result['job']['key']; i=load_reference_index()
    entry=next((e for e in i['entries'] if e['key']==key),None)
    if entry is not None:
        return {r['epoch']:(Path(r['path']),receipt_json(r)) for r in entry['epochs']}
    epochs={}
    for p in sorted((a/'acks'/key).glob('epoch_*.json')):
        ack=read_json(p); require(ack['ram_get_verified'] and ack['ram_commit_status']=='RAM_COMMITTED' and ack['key']==key, 'Unverified ACK')
        rpath=new_path(a/ack['epoch_file']); require(sha_file(rpath)==ack['epoch_file_sha256'], 'Committed numeric epoch changed')
        transaction_path=new_path(a/ack['transaction_file']); transaction=read_json(transaction_path)
        require(sha_file(transaction_path)==ack['transaction_sha256'] and
                all(transaction[k]==ack[k] for k in ('key','attempt','epoch','epoch_file','epoch_file_sha256','source_hash',
                    'design_bundle_sha256','continuation_hash')), 'ACK physical transaction differs')
        r=read_json(rpath); e=r['epoch']; require(e not in epochs and e==ack['epoch'] and r['continuation_hash']==ack['continuation_hash'], 'Duplicate/conflicting ACK')
        require(ack['source_hash']==result['job']['source_hash'] and ack['design_bundle_sha256']==DESIGN_HASH and
                ack['controls_index_hash']==CONTROLS_HASH and ack['server_journal_row']['key']==key and
                ack['server_journal_row']['epoch']==e and ack['server_journal_row']['epoch_file']==ack['epoch_file'], 'ACK provenance/journal differs')
        require(sha_file(a/r['batch_log'])==ack['batch_log_sha256']==r['batch_log_sha256'], 'Committed batch log changed')
        epochs[e]=(rpath,r)
    require(set(epochs)==set(range(1,len(epochs)+1)), 'Skipped committed epoch')
    return epochs


def finite_checks():
    require(design_bundle()['sha256']==DESIGN_HASH, 'Design fixture')
    source=source_manifest()
    require(source['sha256']==digest(source['files']) and len(source['files'])==len(set(source_paths()))==48,
            'Explicit source closure differs')
    for d in DATASETS:
        require(ROWS[d]==40960*(STEPS[d]-1)+TAILS[d], 'Full/tail fixture')
        j=job(d,ARMS[0],2022,'fixture'); e=job(d,ARMS[0],2022,'fixture','engineering_smoke','case')
        require(j['key']!=e['key'] and j['config']['mixture_lambda']==.25, 'Job disjointness')
        s=dict(job=j,purpose='formal',clock_kind='formal',epoch=1,global_step=STEPS[d],next_epoch=2,
               scientific_epochs_completed=1,counts=dict(optimizer_updates=STEPS[d],ema_updates=STEPS[d]),
               operations=dict(training_rows=ROWS[d],task_entries=2*ROWS[d],ordinary_forwards=STEPS[d],scalar_backwards=STEPS[d],
               live_adam_steps=STEPS[d],live_ema_steps=STEPS[d],diagnostic_points=1,extra_autograd_sweeps=3,
               virtual_adam_steps=4,probe_forwards=0,view_forwards=0,algorithm_rng_draws=0))
        validate_formal_clock(s)
        try: validate_formal_clock({**s,'counts':{**s['counts'],'extra':0}})
        except TechnicalPause: pass
        else: raise AssertionError('Expanded counts accepted')
    try: new_root(ROOT/REL_ARTIFACT/'astra_cycles/cycle_03/attempt1')
    except TechnicalPause: pass
    else: raise AssertionError('Old root accepted')
    i=load_reference_index(); schemas={}
    for arm in ('CLEAN','NQR-025'):
        e=next(x for x in i['entries'] if x['arm']==arm)
        r=receipt_json(e['result']); q=receipt_json(e['epochs'][0])
        require(r['test_accessed'] is False and q['epoch']==1, 'Actual numeric schema fixture')
        schemas[arm]=dict(result_fields=list(r),epoch_fields=list(q),rng_available='rng_hash' in q,
                          rng_null_reason=None if 'rng_hash' in q else 'not_recorded_in_source')
    require(not schemas['CLEAN']['rng_available'] and schemas['NQR-025']['rng_available'], 'Real schema difference')
    # Strict ties and independent streams; hashes/LL remain bound to chosen AUC.
    epochs={e:dict(epoch=e,validation={k:dict(macro_auc=(.7 if e in (2,3) else .6),macro_logloss=e/100) for k in ESTIMATORS},
                   model_hash=f'm{e}',ema_hash=f'e{e}') for e in range(1,51)}
    r=dict(selected_epochs={k:2 for k in ESTIMATORS},selected_validation=epochs[2]['validation'],
           selected_hashes=dict(online='m2',fixed_ema='e2'))
    verify_selection(r,epochs)
    class OpaqueTestFixture:
        def __init__(self,**values): self.__dict__.update(values)
        def __getattr__(self,name):
            raise AssertionError('Unapproved attribute dereference: '+name)
    counts=[155949,318599,325619]; train=np.zeros((ROWS['movielens'],2),np.float32)
    start=0
    for domain,count in enumerate(counts): train[start:start+count,-1]=domain; start+=count
    val=np.zeros((VAL_ROWS['movielens'],2),np.float32)
    values=dict(train_x=train,train_y=np.zeros_like(train),train_row_ids=np.arange(len(train),dtype=np.int64),
                val_x=val,val_y=np.zeros_like(val),val_row_ids=np.arange(len(val),dtype=np.int64))
    features=dict(columns=['feature','domain_indicator'],encoder_hash='fixture',scaler_hash='fixture')
    schema=SimpleNamespace(columns=features['columns'],as_dict=lambda:features)
    manifest=dict(array_hashes={n:array_hash(values[n]) for n in ('train_x','train_y','val_x','val_y')},
        split_row_fingerprints={s:split_row_fingerprint(values[k+'_x'],values[k+'_y'],values[k+'_row_ids']) for k,s in (('train','train'),('val','validation'))},
        split_row_id_hashes={s:row_ids_hash(values[k+'_row_ids']) for k,s in (('train','train'),('val','validation'))},
        features=features,encoder_hash='fixture',scaler_hash='fixture',train_domain_counts=counts,
        split_rows=dict(train=len(train),validation=len(val)),
        train_joint_labels={str(d):{'00':n,'01':0,'10':0,'11':0} for d,n in enumerate(counts)})
    bundle=OpaqueTestFixture(**values,features=schema)
    checked=_verify_arrays(bundle,manifest,manifest,'movielens')
    require(checked.train_x is train and set(checked.__dict__)==set(values)|{'features'}, 'Test-opaque data projection')
    return dict(status='PASS',groups=6,actual_control_schemas=schemas,test_opaque_fixture=True,test_accessed=False)
