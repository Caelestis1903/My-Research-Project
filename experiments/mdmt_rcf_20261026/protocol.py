"""Frozen RCF configuration, row permissions and inter-module contracts."""
from __future__ import annotations
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any
import json
import numpy as np
from .state import digest,sha_file,array_hash,state_hash,write_json,append_jsonl
from .vendor_snapshot.mdmt_dfd_20261026.data import RoleBundle
from .vendor_snapshot.mdmt_dfd_20261026.metrics import evaluate_probabilities as evaluate
DATASETS=('movielens','kuairand')
OUTER=('test_exposed','Q_legacy_confirm_exposed')
STAGES={'native_init':1,'native_plan':2,'pilot_init':3,'pilot_plan':4,'local_init':5,'local_plan':6,'base_init':10,'base_plan':11,'direct_plan':12,'prob_init':13,'prob_plan':14,'formation_init':20,'formation_query':21,'formation_aux':22,'formation_shuffle':23,'source_label_shuffle':24,'rrr_init':30,'rrr_plan':31,'bootstrap':40}
PERMISSIONS={'fit':{'native','pilot','local_pilot','normalizer','group_cut','source_kl','audit','verify'},'P_form':{'diagnose','base','demand','formation','source_constraint','reader','direct','audit','verify'},'P_read':{'reader','direct','audit','verify'},'P_cal':{'diagnose','select','audit','verify'},'P_update':{'audit'},'Q_explore':{'evaluate','diagnose','audit'},'validation':{'native_select','evaluate','diagnose','audit','verify'},'test_exposed':{'evaluate','diagnose','audit'},'Q_legacy_confirm_exposed':{'evaluate','diagnose','audit'}}

@dataclass
class RoleView:
    row_id:np.ndarray
    domain:np.ndarray
    y:np.ndarray
    h:np.ndarray
    z:np.ndarray
    group:np.ndarray
@dataclass
class PrefixBundle:
    dataset:str
    seed:int
    views:dict[str,RoleView]
    base_logits:dict[str,np.ndarray]
    base_oof:np.ndarray|None
    pilots:Any
    native:Any
    base_models:Any
    normalizers:dict
    identity:dict
    ledger:Any
    data:Any=None
    device:str='cpu'
    output_root:Any=None
    native_terminal:Any=None
    diagnostics:dict=field(default_factory=dict)
@dataclass
class FeatureBank:
    values:dict[tuple[str,int,int],np.ndarray]
    columns:dict[tuple[int,int],list[dict]]
    bounds:dict[tuple[int,int],list[tuple[float|None,float|None]]]
    identity:dict
@dataclass
class PredictionBundle:
    method:str
    probabilities:dict[str,np.ndarray]
    terminal_probabilities:dict[str,np.ndarray]
    selection:dict
    diagnostics:dict
    identity:dict
@dataclass
class FormationResult:
    method:str
    features:FeatureBank
    functions:dict[str,Any]
    requests:Any
    records:list[dict]
    identity:dict
@dataclass
class DemandResult:
    units:dict[tuple[int,int],dict]
    identity:dict

def load_config(path):
    path=Path(path)
    try:cfg=json.loads(path.read_text())
    except json.JSONDecodeError:
        import yaml
        cfg=yaml.safe_load(path.read_text())
    cfg=dict(cfg);cfg['source_config_hash']=sha_file(path);cfg['stage_registry']=dict(STAGES)
    cfg['storage']=dict(cfg['storage'],models='memory_only',optimizer='memory_only',resume='RAM_snapshot_or_same_identity_replay')
    cfg['spec_resolution']='STORAGE_MEMORY_ONLY_V1/BRIDGE_V1';cfg['config_hash']=digest(cfg)
    return cfg

def derive_rng(seed,dataset,stage,source=0,task=0,unit=0,pass_id=0):
    stage_id=STAGES[stage] if isinstance(stage,str) else int(stage)
    d=DATASETS.index(dataset) if isinstance(dataset,str) else int(dataset)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed),d,91000+stage_id,int(source),int(task),int(unit),int(pass_id)])))

def torch_seed(seed,dataset,stage,source=0,task=0,unit=0,pass_id=0):
    return int(derive_rng(seed,dataset,stage,source,task,unit,pass_id).integers(0,2**32,dtype=np.uint32))

class LabelLedger:
    def __init__(self,data,path=None):
        self.data=data;self.path=Path(path) if path else None;self.entries=[];self.unique={};self.outer_authorization=None
    def freeze_outer(self,identity):self.outer_authorization=digest(identity)
    def assert_role(self,role,purpose):
        aliases={'kl':'source_kl','constraint':'source_constraint','base_oof':'base','base_refit':'base','probability_reader':'reader','mechanisms':'diagnose','evaluation':'evaluate'}
        purpose=aliases.get(purpose,purpose)
        if purpose not in PERMISSIONS.get(role,set()):raise PermissionError(f'{purpose} cannot access {role}')
        if role in OUTER and purpose not in ('audit',) and self.outer_authorization is None:raise PermissionError('Global outer manifest not frozen')
        return purpose
    def access(self,method,role,rows=None,task=None,purpose='formation',passes=1,**extra):
        purpose=self.assert_role(role,purpose);ids=np.asarray(self.data.ids(role) if rows is None else rows,dtype=np.int64)
        if not np.isin(ids,self.data.ids(role)).all():raise PermissionError('Requested rows outside role')
        key=(str(method),role,str(task),purpose)
        if key not in self.unique:
            if self.path is None:
                seen=np.zeros(len(self.data.x),dtype=bool)
            else:
                # A label-access mask is research data, never a training state.
                directory=self.path.parent/'row_data_cache'/'ledger'
                directory.mkdir(parents=True,exist_ok=True)
                seen=np.lib.format.open_memmap(directory/(digest(key)+'.npy'),mode='w+',dtype=bool,shape=(len(self.data.x),))
                seen[:]=False
            self.unique[key]=seen
        seen=self.unique[key];new=int((~seen[ids]).sum());seen[ids]=True
        rec=dict(method=str(method),role=role,purpose=purpose,task=task,count=len(ids)*passes,new_unique_rows=new,row_hash=array_hash(ids),passes=passes,**extra)
        self.entries.append(rec)
        if self.path:append_jsonl(self.path,rec)
        return rec
    def summary(self):
        return [dict(method=k[0],role=k[1],task=k[2],purpose=k[3],unique_rows=int(v.sum())) for k,v in self.unique.items()]

def record_access(bundle,method,role,rows=None,task=None,purpose='formation',passes=1,**extra):
    return bundle.ledger.access(method,role,rows=rows,task=task,purpose=purpose,passes=passes,**extra)

def audit_dataset(dataset,config,output_root):
    data=RoleBundle(dataset);out=Path(output_root)/'audit';out.mkdir(parents=True,exist_ok=True)
    report=data.verify(out,full=True)
    if report['raw_sha256']!=config['datasets'][dataset]['raw_sha256']:raise ValueError('raw identity differs from frozen config')
    report.update(spec='RCF-v1',protocol_permissions={k:sorted(v) for k,v in PERMISSIONS.items()},prepared_data_hash=data.data_hash)
    write_json(out/f'{dataset}_DATA_ROLE_MANIFEST.json',report);return report

def folds(data):
    ids=data.ids('P_form');out=np.empty(len(ids),dtype=np.int8)
    for d in range(3):
        pos=np.flatnonzero(data.domain[ids]==d)
        perm=np.random.Generator(np.random.PCG64(20261026+70000+1009*d)).permutation(np.sort(ids[pos]))
        out[np.searchsorted(ids,perm)]=np.arange(len(perm))%2
    return out
