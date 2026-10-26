"""Verified natural rows, fit-only preprocessing and DFD role permissions."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from .vendor.strict_data.data_roles import RoleDataset,assign_roles,_raw_schema,_chunks,_strings,EXPECTED
from .state import atomic_json,array_hash,digest,sha_file,append_jsonl
from .plans import DATASETS
ROOT=Path('/home/xzf/data/mdmt_next_20261024_strict_v1')
ATOMIC=('fit','P_form','P_read','P_cal','P_update','Q_explore','Q_legacy_confirm_exposed','validation','test_exposed')
PURPOSES={'fit':{'native','pilot','formation','normalizer','geometry','audit','semantic_check'},'P_form':{'eta','eta_oof','reader','constant','audit','semantic_check'},'P_read':{'reader','constant','audit','semantic_check'},'P_cal':{'select','audit','semantic_check'},'P_update':{'rhd','audit'},'Q_explore':{'evaluate','audit'},'validation':{'select','evaluate','audit','semantic_check'},'test_exposed':{'evaluate','audit'},'Q_legacy_confirm_exposed':{'evaluate','audit'}}

@dataclass
class Batch:
    x:torch.Tensor
    y:torch.Tensor
    domain:torch.Tensor
    row_id:torch.Tensor
    role:str

class RoleBundle:
    def __init__(self,dataset,root=None,access_log=None):
        self.dataset=dataset;self.dataset_id=DATASETS.index(dataset);self.root=Path(root or ROOT/dataset)
        old=RoleDataset(self.root)
        self.x=old.x;self.y=old._labels;self.domain=np.asarray(self.x[:,-1],dtype=np.int64)
        self.schema=old.manifest['features'];self.features=self.schema
        self.raw_manifest=old.manifest;self.raw_path=Path(old.manifest['raw_path'])
        self.roles={k:v for k,v in old.roles.items() if k not in ('P_label','test','Q_confirm')}
        self.roles['test_exposed']=old.roles['test'];self.roles['Q_legacy_confirm_exposed']=old.roles['Q_confirm']
        pieces=[[],[]]
        for d in range(3):
            ids=old.roles['P_label'][self.domain[old.roles['P_label']]==d]
            order=np.random.Generator(np.random.PCG64(20261025+60000+1009*d)).permutation(np.sort(ids))
            pieces[0].extend(order[:len(order)//2]);pieces[1].extend(order[len(order)//2:])
        for role,ids in zip(('P_form','P_read'),pieces):self.roles[role]=np.sort(np.asarray(ids,dtype=np.int64))
        recreated=assign_roles(self.domain)
        for k,v in recreated.items():
            if not np.array_equal(v,old.roles[k]):raise ValueError('PROTOCOL_MISMATCH '+k)
        combined=np.concatenate([self.roles[k] for k in ATOMIC])
        if not np.array_equal(np.sort(combined),np.arange(len(self.x))):raise ValueError('Atomic partition not exhaustive/disjoint')
        expected=[124759,254879,260495] if dataset=='movielens' else [138225,1395118,71561]
        if np.bincount(self.domain[self.roles['fit']],minlength=3).tolist()!=expected:raise ValueError('Fit count mismatch')
        self.pools={(r,d):ids[self.domain[ids]==d] for r,ids in self.roles.items() for d in range(3)}
        self.manifest=dict(spec='DFD-v4.0',dataset=dataset,parent_data_hash=old.manifest['data_hash'],schema=self.schema,roles={r:dict(n=len(ids),row_id_hash=array_hash(ids)) for r,ids in self.roles.items()},raw_sha256=old.manifest['raw_sha256'],array_hashes=old.manifest['array_hashes'])
        self.data_hash=digest(self.manifest);self.hash=self.data_hash;self.manifest['data_hash']=self.data_hash
        self.access_log=Path(access_log) if access_log else None;self._masks={}
        self.user_ids=None;self.item_ids=None;self.support_reason=None
        if dataset=='movielens':
            ids=pd.read_csv(self.raw_path,usecols=['user_id','movie_id']);self.user_ids=ids.user_id.to_numpy();self.item_ids=ids.movie_id.to_numpy()
        else:self.support_reason='processed raw CSV has no original user/item IDs; trusted row mapping unavailable'
    def ids(self,role,domain=None):return self.roles[role] if domain is None else self.pools[role,int(domain)]
    def _check(self,ids,role,purpose,authorization=None):
        ids=np.asarray(ids,dtype=np.int64)
        if purpose not in PURPOSES.get(role,set()):raise PermissionError(f'{purpose} cannot access {role}')
        if role in ('test_exposed','Q_legacy_confirm_exposed') and purpose!='audit':
            if authorization!=self.data_hash:raise PermissionError('Exposed labels require exact frozen-data authorization')
        if role not in self._masks:
            self._masks[role]=np.zeros(len(self.x),bool);self._masks[role][self.roles[role]]=True
        if (ids<0).any() or (ids>=len(self.x)).any() or not self._masks[role][ids].all():raise PermissionError('Rows outside role')
        if self.access_log:append_jsonl(self.access_log,dict(role=role,purpose=purpose,count=len(ids),rows_hash=array_hash(ids),tasks=[0,1]))
        return ids
    def batch(self,ids,role,device='cpu',purpose='semantic_check',authorization=None):
        ids=self._check(ids,role,purpose,authorization)
        return Batch(torch.as_tensor(np.array(self.x[ids]),device=device),torch.as_tensor(np.array(self.y[ids]),device=device),torch.as_tensor(self.domain[ids],device=device),torch.as_tensor(ids,device=device),role)
    def supports(self,ids):
        ids=np.asarray(ids,dtype=np.int64)
        if self.user_ids is None:return None
        fit=self.roles['fit'];return np.stack((np.isin(self.user_ids[ids],np.unique(self.user_ids[fit])),np.isin(self.item_ids[ids],np.unique(self.item_ids[fit]))),axis=1)
    def verify(self,output_dir,full=True):
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        if sha_file(self.raw_path)!=EXPECTED[self.dataset]:raise ValueError('Raw SHA mismatch')
        raw_domains,_,_= _raw_schema(self.dataset,self.raw_path)
        if not np.array_equal(raw_domains,self.domain):raise ValueError('Raw/prepared domain identity mismatch')
        prep=json.loads((self.root/'PREPROCESSOR.json').read_text())
        if full:
            fitmask=np.zeros(len(self.x),bool);fitmask[self.roles['fit']]=True
            seen={k:set() for k in prep['vocabulary']};offset=0
            for chunk,labels in _chunks(self.dataset,self.raw_path):
                end=offset+len(chunk)
                if not np.array_equal(labels,self.y[offset:end]):raise ValueError('Raw label mismatch')
                local=chunk.loc[fitmask[offset:end]]
                for col in seen:seen[col].update(_strings(local[col]).unique())
                for j,col in enumerate(self.schema['columns']):
                    if col=='domain_indicator':pred=self.domain[offset:end]
                    elif col in self.schema['dense_columns']:
                        sc=prep['scaler'];pred=(chunk[col].to_numpy(np.float32)-sc['age_min'])/max(sc['age_max']-sc['age_min'],1e-12)
                    else:pred=_strings(chunk[col]).map(prep['vocabulary'][col]).fillna(0).to_numpy(np.float32)
                    if not np.array_equal(pred,self.x[offset:end,j]):raise ValueError('Prepared encoding mismatch '+col)
                offset=end
            for col,values in seen.items():
                if {v:i+1 for i,v in enumerate(sorted(values))}!=prep['vocabulary'][col]:raise ValueError('Vocabulary is not fit-only '+col)
            if 'age' in self.schema['dense_columns']:
                ages=pd.read_csv(self.raw_path,usecols=['age']).age.to_numpy();f=ages[self.roles['fit']]
                if prep['scaler']!={'age_min':float(f.min()),'age_max':float(f.max())}:raise ValueError('Scaler not fit-only')
        report=dict(self.manifest,verified_raw=True,verified_array_hashes=True,verified_roles=True,verified_full_fit_only_encoding=full,user_item_support=self.support_reason or 'raw identity available',role_details={})
        for role,ids in self.roles.items():
            cells=[]
            for q in range(2):
                for d in range(3):
                    sub=ids[self.domain[ids]==d];pos=int(self.y[sub,q].sum());cells.append(dict(cell=3*q+d,domain=d,task=q,n=len(sub),positive=pos,negative=len(sub)-pos))
            report['role_details'][role]=dict(cells=cells,support_counts=None if self.supports(ids) is None else {f'{u}_{i}':int(((self.supports(ids)[:,0]==u)&(self.supports(ids)[:,1]==i)).sum()) for u in (False,True) for i in (False,True)})
        atomic_json(out/f'{self.dataset}_DATA_ROLE_MANIFEST.json',report);np.savez_compressed(out/f'{self.dataset}_rows.npz',**self.roles)
        return report
