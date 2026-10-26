"""Canonical raw-X content reconstruction and a fixed normal-RPC installer.

Only independently encoded H/Z and frozen parameters construct new outputs.
No training cache, stored phi, function.outputs, or stored p is read by the
reconstruction functions. Original modules and numerical reader policies stay.
"""
from __future__ import annotations
import copy
import hashlib
import json
import marshal
import os
from pathlib import Path
import struct
import sys
import time
import types
import numpy as np
import torch
from .state import state_hash,digest

WORKSPACE=Path('/home/xzf/Code/phase0-3')
ARTIFACT_ROOT=WORKSPACE/'artifacts/mdmt_rcf_20261026/v2_four_seed'
SERVICE_PID=2088990
CONFIG_HASH='cefb6d934c774bd78a438e4fba5c560e934f4c56ab09fbb3dfe389f4f4842fd6'
FROZEN_SHA='09def5e2f43786f9c2875e5ae466845a10ec7db47a3c6edd28379827b72e927a'
WORKER_PIDS={'movielens:5201':2089374,'kuairand:5201':2090829,
             'movielens:5202':2091226,'kuairand:5202':2093268,
             'movielens:5203':2093543,'kuairand:5203':2093912,
             'movielens:5204':2094033,'kuairand:5204':2094138}
RECIPE='full_role_pilot40960_delta16384_native_independent_v1'
FINGERPRINT_SCHEME='CODE_SEMANTIC_V1'
_OBSERVER_INSTALLED=False


def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _constant_semantics(value):
    if isinstance(value,types.CodeType):return ['code',_code_semantics(value)]
    if value is None:return ['none']
    if value is Ellipsis:return ['ellipsis']
    if isinstance(value,bool):return ['bool',value]
    if isinstance(value,int):return ['int',str(value)]
    if isinstance(value,float):return ['float64_bits',struct.pack('!d',value).hex()]
    if isinstance(value,complex):return ['complex128_bits',struct.pack('!dd',value.real,value.imag).hex()]
    if isinstance(value,str):return ['str',value]
    if isinstance(value,bytes):return ['bytes',value.hex()]
    if isinstance(value,tuple):return ['tuple',[_constant_semantics(v) for v in value]]
    if isinstance(value,frozenset):
        members=[_constant_semantics(v) for v in value]
        members.sort(key=lambda v:json.dumps(v,sort_keys=True,separators=(',',':')))
        return ['frozenset',members]
    raise TypeError('Unsupported code constant type: '+type(value).__name__)


def _code_semantics(code):
    return dict(name=code.co_name,qualname=code.co_qualname,
        argcount=code.co_argcount,posonlyargcount=code.co_posonlyargcount,
        kwonlyargcount=code.co_kwonlyargcount,nlocals=code.co_nlocals,
        stacksize=code.co_stacksize,flags=code.co_flags,bytecode=code.co_code.hex(),
        exceptiontable=code.co_exceptiontable.hex(),
        constants=[_constant_semantics(v) for v in code.co_consts],
        names=list(code.co_names),varnames=list(code.co_varnames),
        freevars=list(code.co_freevars),cellvars=list(code.co_cellvars))


def semantic_code_fingerprint(code):
    payload=dict(scheme=FINGERPRINT_SCHEME,python_bytecode_version=list(sys.version_info[:2]),code=_code_semantics(code))
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def callable_fingerprint(function):
    return semantic_code_fingerprint(function.__code__)


def _callable_description(function):
    code=function.__code__
    return dict(module=function.__module__,name=function.__name__,qualname=function.__qualname__,
        co_name=code.co_name,co_qualname=code.co_qualname,co_filename=code.co_filename,
        co_firstlineno=code.co_firstlineno,co_flags=code.co_flags,
        semantic_fingerprint=callable_fingerprint(function),
        marshal_fingerprint=hashlib.sha256(marshal.dumps(code)).hexdigest())


def _hidden(head,h):
    # This existing observer primitive has exactly the PilotAccess._hidden
    # Linear/ReLU graph, original head device and full-role 40960 chunks.
    from .baselines import _raw_hidden
    return _raw_hidden(head,h)


def _initial_statistics(bundle,source,task):
    from .baselines import _independent_raw_packets
    from .memory_data import data_hash
    fit=bundle.views['fit'];head=bundle.pilots.initial_bank.head(source,task)
    key=(RECIPE,bundle.identity.get('native_hash'),data_hash(fit.row_id),
         state_hash(head.state_dict()),int(source),int(task))
    cache=getattr(bundle,'_independent_observer_initial_statistics',None)
    if cache is None:cache={};bundle._independent_observer_initial_statistics=cache
    if key not in cache:
        h,_=_independent_raw_packets(bundle,'fit')
        values=_hidden(head,h)[fit.domain==source]
        cache[key]=dict(mean=values.mean(0,dtype=np.float64),
                        std=np.maximum(values.std(0,dtype=np.float64),1e-3),
                        reconstruction_recipe=RECIPE)
    return cache[key]


def _formed_values(bundle,function,h):
    # Clone only in RAM. The actual cache was produced on the formation device,
    # full role, at DeltaFunction.values' 16384 chunk and CPU NumPy scale site.
    module=copy.deepcopy(function.module).to(bundle.device)
    module.eval().requires_grad_(False)
    values=[]
    with torch.no_grad():
        for start in range(0,len(h),16384):
            x=torch.as_tensor(np.array(h[start:start+16384]),dtype=torch.float32,device=bundle.device)
            y=module(x).reshape(-1).cpu().numpy()
            values.append(function.scale*y)
    return np.concatenate(values) if values else np.empty(0,dtype=np.float32)


def raw_source(bundle,h,initial=False,local_replica=None,standardized=False):
    bank=bundle.pilots.initial_bank if initial else bundle.pilots.source_bank if local_replica is None else bundle.pilots.local_banks[local_replica]
    out=np.empty((len(h),3,2,32 if standardized else 1),dtype=np.float32)
    for source in range(3):
        for task in range(2):
            head=bank.head(source,task);hidden=_hidden(head,h)
            if standardized:
                norm=bundle.pilots.phi_normalizers[source,task] if local_replica is None else bundle.pilots.local_normalizers[source,task,local_replica]
                out[:,source,task]=(hidden-norm['mean'])/norm['std']
            else:
                weight=head.output.weight.detach().cpu().numpy().reshape(-1)
                bias=float(head.output.bias.detach().cpu().item())
                out[:,source,task,0]=hidden@weight+bias
    return out if standardized else out[...,0]


def raw_features(bundle,bank,role,h,z,domain):
    method=getattr(bank,'fixed_method',None);values={};hidden_cache={};delta_cache={}
    source=raw_source(bundle,h) if method=='PILOT_LOGIT' else None
    def standardized(kind,owner,task,replica=None):
        key=(kind,owner,task,replica)
        if key not in hidden_cache:
            if kind=='initial':
                head=bundle.pilots.initial_bank.head(owner,task)
                norm=_initial_statistics(bundle,owner,task)
            elif kind=='local':
                head=bundle.pilots.local_banks[replica].head(owner,task)
                norm=bundle.pilots.local_normalizers[owner,task,replica]
            else:
                head=bundle.pilots.source_bank.head(owner,task)
                norm=bundle.pilots.phi_normalizers[owner,task]
            full=_hidden(head,h)
            hidden_cache[key]=((full-norm['mean'])/norm['std']).astype(np.float32)
        return hidden_cache[key]
    for target in range(3):
        mask=domain==target;sources=[s for s in range(3) if s!=target]
        for task in range(2):
            if method=='GROUP_CAL':value=np.empty((int(mask.sum()),0))
            elif method=='PILOT_LOGIT':
                value=np.stack([source[mask,s,task]-source[mask,target,task] for s in sources],1)
            elif method in ('FULL_SOURCE','FULL_SOURCE_ALL','RANDOM_FEATURE'):
                kind='initial' if method=='RANDOM_FEATURE' else 'source'
                value=np.concatenate([standardized(kind,s,task)[mask] for s in sources],1)
            elif method=='FULL_TARGET':
                value=np.concatenate([standardized('local',target,task,k)[mask] for k in range(2)],1)
            elif hasattr(bank,'functions'):
                columns=[]
                for column in bank.columns[target,task]:
                    fid=column['function_id']
                    if fid not in delta_cache:delta_cache[fid]=_formed_values(bundle,bank.functions[fid],h)
                    columns.append(delta_cache[fid][mask])
                value=np.stack(columns,1) if columns else np.empty((int(mask.sum()),0))
            elif hasattr(bank,'projections'):
                parts=[standardized('source',s,task)[mask]@bank.projections[s,task]
                       for s in sources if (s,task) in bank.projections]
                value=np.concatenate(parts,1) if parts else np.empty((int(mask.sum()),0))
            else:raise RuntimeError('Raw-X content reconstruction metadata missing for '+bank.identity.get('method','unknown'))
            values[target,task]=value
    # All output arrays have already been constructed independently. The
    # separate measurement may read retained values solely as an error oracle.
    _postcompute_comparison(bundle,bank,role,h,hidden_cache,delta_cache)
    return values


def _postcompute_comparison(bundle,bank,role,h,hidden_cache,delta_cache):
    if not _OBSERVER_INSTALLED or role!='fit':return
    key=f'{bundle.dataset}:{bundle.seed}'
    if WORKER_PIDS.get(key)!=os.getpid() or os.getppid()!=SERVICE_PID:return
    method=getattr(bank,'fixed_method',bank.identity.get('method'))
    if method not in ('FULL_SOURCE_ALL','RCF'):return
    if _sha(ARTIFACT_ROOT/'FROZEN_MANIFEST.json')!=FROZEN_SHA:
        raise PermissionError('Frozen manifest changed during observer measurement')
    from .artifacts import _json
    from .memory_data import data_hash
    path=ARTIFACT_ROOT/'observer_raw_reconstruction'/'comparisons'/f'{bundle.dataset}_{bundle.seed}_{method}_fit.json'
    source_sha=_sha(__file__)
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('observer_sha256')==source_sha and previous.get('frozen_manifest_sha256')==FROZEN_SHA:return
    result=dict(dataset=bundle.dataset,seed=bundle.seed,method=method,role=role,
        rows=len(h),row_id_hash=data_hash(bundle.views[role].row_id),native_hash=bundle.identity.get('native_hash'),
        frozen_manifest_sha256=FROZEN_SHA,observer_sha256=source_sha,recipe=RECIPE,
        construction_completed_before_oracle_read=True,oracle_used_to_construct_output=False,
        true_logit_error='NOT_MEASURED',final_probability_gate='Separate full-row normal evaluate; unchanged 1e-6',
        labels_used=False,model_files_written=0)
    if method=='FULL_SOURCE_ALL':
        source,task=0,0;oracle=bundle.pilots._cache.get(('source',role,source,task))
        if oracle is None:result.update(status='ORACLE_UNAVAILABLE',kind='source_hidden')
        else:
            head=bundle.pilots.source_bank.head(source,task)
            independently_recomputed=_hidden(head,h)
            norm=bundle.pilots.phi_normalizers[source,task]
            normalized_oracle=((oracle-norm['mean'])/norm['std']).astype(np.float32)
            result.update(status='MEASURED',kind='source_hidden',source=source,task=task,
                canonical_chunk=40960,canonical_device=str(next(head.parameters()).device),
                hidden_max_error=float(np.max(np.abs(independently_recomputed-oracle))),
                standardized_float32_max_error=float(np.max(np.abs(hidden_cache['source',source,task,None]-normalized_oracle))))
    else:
        fid=next(iter(sorted(delta_cache)),None)
        function=None if fid is None else bank.functions[fid]
        oracle=None if function is None else function.outputs.get(role)
        if oracle is None:result.update(status='ORACLE_UNAVAILABLE',kind='formed_delta')
        else:
            target=next(t for t in range(3) if t!=function.source)
            mask=bundle.views[role].domain==target
            old_cpu_target_first=function.values(h[mask])
            result.update(status='MEASURED',kind='formed_delta',function_id=fid,source=function.source,
                task=function.task,canonical_chunk=16384,canonical_device=bundle.device,
                original_module_device=str(next(function.module.parameters()).device),
                new_full_role_max_error=float(np.max(np.abs(delta_cache[fid]-oracle))),
                old_cpu_target_first_max_error=float(np.max(np.abs(old_cpu_target_first-oracle[mask]))),
                old_comparison_target=target,original_scale=float(function.scale))
    _json(path,result)


def _frozen_fingerprint(bundle,predictions,formations):
    from .delivery import _observation_fingerprint,_model_groups
    from .memory_data import data_hash
    result=_observation_fingerprint(bundle,predictions,formations)
    terminal={f'{method}_{key}':model for method,pred in predictions.items()
              for key,model in pred.diagnostics.get('terminal_models',pred.diagnostics.get('_terminal_models',{})).items()}
    result['terminal_models']={key:dict(state=state_hash(model.state_dict()),
        modes={n:bool(m.training) for n,m in model.named_modules()},
        requires_grad={n:bool(p.requires_grad) for n,p in model.named_parameters()},
        gradients=state_hash({n:p.grad for n,p in model.named_parameters()}),
        devices={n:str(p.device) for n,p in model.named_parameters()}) for key,model in terminal.items()}
    result['devices']={key:{n:str(p.device) for n,p in model.named_parameters()}
                       for key,model in _model_groups(bundle,predictions,formations).items()}
    result['selection_and_identity']={m:digest(dict(selection=p.selection,identity=p.identity)) for m,p in predictions.items()}
    result['predictions']={m:{kind:{r:data_hash(a) for r,a in getattr(p,kind).items()}
                              for kind in ('probabilities','terminal_probabilities')} for m,p in predictions.items()}
    result['ram_training_states']=state_hash(dict(
        selected=bundle.diagnostics.get('selected_state'),terminal=bundle.diagnostics.get('native_terminal_complete'),
        pilot_optimizer=bundle.diagnostics['pilot_optimizer'].state_dict(),
        local_optimizers=[o.state_dict() for o in bundle.diagnostics.get('local_optimizers',[])],
        base_optimizers={str(k):v[1].state_dict() for k,v in bundle.base_models.items() if isinstance(v,tuple)},
        functions={m:{fid:f.training_state for fid,f in formed.functions.items()} for m,formed in formations.items()}))
    return result


def install_on_reader_reload():
    """Install exactly two callables only within the existing typed worker RPC."""
    if os.getpid() not in WORKER_PIDS.values():
        return dict(status='NO_OP',reason='Not a registered resident worker')
    campaign=sys.modules.get(__package__+'.campaign')
    if campaign is None:return dict(status='NO_OP',reason='Campaign is not loaded')
    frame=sys._getframe(1);context=None
    try:
        for _ in range(40):
            if frame is None:break
            if frame.f_code is campaign._worker.__code__:
                if frame.f_locals.get('op')=='reload-readers':context=dict(frame.f_locals)
                break
            frame=frame.f_back
    finally:del frame
    if context is None:return dict(status='NO_OP',reason='Outside exact normal reload-readers worker frame')
    from . import baselines
    from .protocol import PrefixBundle
    from .artifacts import _json
    if not isinstance(context.get('bundle'),PrefixBundle) or any(not isinstance(context.get(name),dict) for name in ('predictions','formations','config')):
        raise TypeError('Normal observer reload requires typed PrefixBundle/dict worker state')
    dataset=context['dataset'];seed=int(context['seed']);key=f'{dataset}:{seed}'
    if WORKER_PIDS.get(key)!=os.getpid() or os.getppid()!=SERVICE_PID:
        raise PermissionError('Registered worker identity differs')
    if context['config']['config_hash']!=CONFIG_HASH or Path(context['root']).resolve()!=ARTIFACT_ROOT or not context['frozen']:
        raise PermissionError('Observer installation requires the unchanged frozen campaign')
    frozen=ARTIFACT_ROOT/'FROZEN_MANIFEST.json'
    if _sha(frozen)!=FROZEN_SHA:raise PermissionError('Frozen manifest bytes changed')
    token_path=ARTIFACT_ROOT/'OBSERVER_RAW_RECONSTRUCTION_AUTHORIZATION.json'
    token=json.loads(token_path.read_text())
    expected=dict(service_pid=SERVICE_PID,config_hash=CONFIG_HASH,frozen_manifest_sha256=FROZEN_SHA,worker_pids=WORKER_PIDS)
    if token.get('verdict') not in ('PASS','PASS WITH FIXES') or token.get('blocking_issues',[]):
        raise PermissionError('Independent observer review has not passed')
    if any(token.get(k)!=v for k,v in expected.items()):raise PermissionError('Observer review identity differs')
    if token.get('fingerprint_scheme')!=FINGERPRINT_SCHEME:raise PermissionError('Observer code fingerprint scheme differs')
    for name in ('raw_reconstruction_observer.py','baselines.py','readers.py'):
        path=Path(__file__).with_name(name)
        if token.get('source_sha256',{}).get(name)!=_sha(path):raise PermissionError('Reviewed observer source changed: '+name)
    replacements={'raw_features':raw_features,'_raw_source':raw_source}
    old={name:callable_fingerprint(getattr(baselines,name)) for name in replacements}
    new={name:callable_fingerprint(fn) for name,fn in replacements.items()}
    if token.get('new_callable_fingerprints')!=new or (old!=token.get('old_callable_fingerprints') and old!=new):
        _json(ARTIFACT_ROOT/'observer_raw_reconstruction'/'guard_rejections'/f'{dataset}_{seed}_{time.time_ns()}.json',
            dict(status='REJECTED_BEFORE_ASSIGNMENT',worker_pid=os.getpid(),dataset=dataset,seed=seed,
                 fingerprint_scheme=FINGERPRINT_SCHEME,actual_old={n:_callable_description(getattr(baselines,n)) for n in replacements},
                 actual_new={n:_callable_description(fn) for n,fn in replacements.items()},
                 expected_old_semantic=token.get('old_callable_fingerprints'),expected_new_semantic=token.get('new_callable_fingerprints'),
                 expected_old_marshal=token.get('old_marshal_fingerprints'),expected_new_marshal=token.get('new_marshal_fingerprints'),
                 authorization_sha256=_sha(token_path),frozen_manifest_sha256=FROZEN_SHA,model_files_written=0))
        raise PermissionError('Fixed observer code semantics differ; rejection metadata recorded before assignment')
    training={name:id(getattr(baselines,name)) for name in ('fit_base','fit_direct_baselines','fit_probability_reader','_fit_head','_new_head')}
    before=_frozen_fingerprint(context['bundle'],context['predictions'],context['formations'])
    baselines.raw_features=raw_features
    baselines._raw_source=raw_source
    after=_frozen_fingerprint(context['bundle'],context['predictions'],context['formations'])
    if before!=after or training!={name:id(getattr(baselines,name)) for name in training}:
        raise RuntimeError('Observer installation changed frozen scientific state or training functions')
    global _OBSERVER_INSTALLED
    _OBSERVER_INSTALLED=True
    receipt=dict(status='INSTALLED' if old!=new else 'ALREADY_INSTALLED',dataset=dataset,seed=seed,
        worker_pid=os.getpid(),parent_pid=os.getppid(),frozen_manifest_sha256=FROZEN_SHA,
        authorization_sha256=_sha(token_path),source_sha256=token['source_sha256'],
        fingerprint_scheme=FINGERPRINT_SCHEME,old_callable_fingerprints=old,new_callable_fingerprints=new,recipe=RECIPE,
        before=before,after=after,all_frozen_state_unchanged=True,training_function_objects_unchanged=True,
        model_files_written=0,installed_unix=time.time(),actual_full_parity='PENDING_NORMAL_EVALUATE')
    _json(ARTIFACT_ROOT/'observer_raw_reconstruction'/f'{dataset}_{seed}.json',receipt)
    return {k:v for k,v in receipt.items() if k not in ('before','after')}
