"""Natural historical rows with direct raw-value checks and label permissions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from experiments.mdmt_relation_basis_v2.vendor_snapshot.mdmt_dfd_20261026.vendor.strict_data.data_roles import (
    assign_roles, _raw_schema, _chunks, _strings)
from .identity import write_json, append_jsonl

ROOT = Path(__file__).resolve().parents[2]
RAW = dict(movielens=Path('/home_new/xzf/datasets/ml-1m/ml-1m.csv'),
           kuairand=Path('/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv'))
PREPARED = dict(movielens=ROOT/'artifacts/mdmt_relation_basis_v2/data/movielens',
                kuairand=ROOT/'artifacts/mdmt_relation_basis_v2/revision1/data/kuairand')
ROLE_MAP = dict(T0=('fit',), F=('P_label',), S=('P_cal','P_update'), Q=('Q_explore',),
                V=('validation',), E=('test',), legacy=('Q_confirm',))
PURPOSES = dict(T0={'native','pilot','normalizer','owner','constraint','audit'},
                F={'adapt','probe','gain','partition','shadow','diagnostic'},
                S={'transaction'}, V={'select'}, Q={'development','diagnostic'},
                E={'final'}, legacy=set())


def verify_prepared(dataset, receipt_path):
    root, raw = PREPARED[dataset], RAW[dataset]
    x, y = np.load(root/'x.npy',mmap_mode='r'), np.load(root/'y.npy',mmap_mode='r')
    manifest=json.loads((root/'DATA_MANIFEST.json').read_text())
    prep=json.loads((root/'PREPROCESSOR.json').read_text())
    domain, columns, dense=_raw_schema(dataset,raw)
    if len(x)!=len(domain) or not np.array_equal(domain,x[:,-1]):
        raise ValueError('Raw/encoded domain mismatch')
    roles=assign_roles(domain)
    for role,ids in roles.items():
        if not np.array_equal(ids,np.load(root/f'rows_{role}.npy')):
            raise ValueError('Historical row plan differs: '+role)
    fit=np.zeros(len(x),bool);fit[roles['fit']]=True
    seen={k:set() for k in prep['vocabulary']}
    low,high,offset=float('inf'),-float('inf'),0
    for chunk, labels in _chunks(dataset,raw):
        end=offset+len(chunk)
        if not np.array_equal(labels,y[offset:end]):
            raise ValueError('Raw label mismatch')
        current=chunk.loc[fit[offset:end]]
        for key in seen:
            seen[key].update(_strings(current[key]).unique())
        if dense and len(current):
            low=min(low,float(current.age.min()));high=max(high,float(current.age.max()))
        for j,key in enumerate(columns):
            if key=='domain_indicator':
                expected=domain[offset:end]
            elif key in dense:
                scaler=prep['scaler']
                expected=(chunk[key].to_numpy(np.float32)-scaler['age_min'])/max(scaler['age_max']-scaler['age_min'],1e-12)
            else:
                expected=_strings(chunk[key]).map(prep['vocabulary'][key]).fillna(0).to_numpy(np.float32)
            if not np.array_equal(expected,x[offset:end,j]):
                raise ValueError('Raw feature mismatch: '+key)
        offset=end
    for key,values in seen.items():
        if {v:i+1 for i,v in enumerate(sorted(values))}!=prep['vocabulary'][key]:
            raise ValueError('Encoder is not T0-only: '+key)
    if dense and prep['scaler']!=dict(age_min=low,age_max=high):
        raise ValueError('Scaler is not T0-only')
    receipt=dict(dataset=dataset,status='PASS_DIRECT_VALUES',raw_path=str(raw),prepared=str(root),
                 rows=len(x),verified_encoding=True,verified_labels=True,verified_fit_only=True,
                 verified_roles=True,comparison='raw values, full ordered row arrays, vocabularies and scaler',
                 raw_identity_provenance='AGENTS.md verified relocation record',
                 role_counts={r:len(ids) for r,ids in roles.items()},features=manifest['features'])
    write_json(receipt_path,receipt)
    return receipt


def _group_partition(ids,group,parts,seed,domain=None):
    domains=np.zeros(len(group),np.int64) if domain is None else domain
    bins=[[] for _ in range(parts)];assigned=set()
    for d in np.unique(domains[ids]):
        values=np.unique(group[ids[domains[ids]==d]])
        values=np.asarray([v for v in values if int(v) not in assigned])
        order=np.random.Generator(np.random.PCG64(seed+1009*int(d))).permutation(values)
        for k in range(parts):
            bins[k].extend(order[k::parts])
        assigned.update(int(v) for v in values)
    return [np.sort(ids[np.isin(group[ids],chosen)]) for chosen in bins]


class Data:
    def __init__(self,dataset,receipt_path,access_log=None):
        receipt=json.loads(Path(receipt_path).read_text())
        if receipt.get('status')!='PASS_DIRECT_VALUES' or receipt.get('dataset')!=dataset:
            raise ValueError('Prepared direct verification required')
        root=PREPARED[dataset]
        original=json.loads((root/'DATA_MANIFEST.json').read_text())
        self.dataset=dataset
        self.x=np.load(root/'x.npy',mmap_mode='r')
        self._y=np.load(root/'y.npy',mmap_mode='r')
        self.domain=np.asarray(self.x[:,-1],np.int64)
        self.schema=original['features']
        old={r:np.load(root/f'rows_{r}.npy') for names in ROLE_MAP.values() for r in names}
        self.roles={r:np.sort(np.concatenate([old[n] for n in names])) for r,names in ROLE_MAP.items()}
        if not np.array_equal(np.sort(np.concatenate(list(self.roles.values()))),np.arange(len(self.x))):
            raise ValueError('Roles must be exhaustive and disjoint')
        if dataset=='movielens':
            self.group=pd.read_csv(RAW[dataset],usecols=['user_id'])['user_id'].to_numpy(np.int64)
            grouping='raw_user_id'
        else:
            self.group=np.arange(len(self.x),dtype=np.int64)
            grouping='row_id_no_reliable_user_id'
        probe=[]
        for d in range(3):
            pool=self.roles['F'][self.domain[self.roles['F']]==d]
            chosen=np.random.Generator(np.random.PCG64(2022+801+1009*d)).permutation(pool)[:min(1024,len(pool))]
            probe.extend(chosen)
        self.probe=np.sort(np.asarray(probe,np.int64))
        self.probe_panels=_group_partition(self.probe,self.group,2,2022+811,self.domain)
        self.folds=np.empty(len(self.probe),np.int64)
        for k,ids in enumerate(self.probe_panels):
            self.folds[np.searchsorted(self.probe,ids)]=k
        self.s_windows=_group_partition(self.roles['S'],self.group,3,2022+821,self.domain)
        self.used_s=set()
        self.read_s=set()
        self.s_consumption={}
        self.final_authorization=None
        self.access_log=Path(access_log) if access_log else None
        self.owner_weights=np.bincount(self.domain[self.roles['T0']],minlength=3)/len(self.roles['T0'])
        def cells(ids):
            result=[]
            for t in range(2):
                for d in range(3):
                    chosen=ids[self.domain[ids]==d]
                    positive=int(self._y[chosen,t].sum())
                    result.append(dict(domain=d,task=t,n=len(chosen),positive=positive,negative=len(chosen)-positive))
            return result
        self.manifest=dict(dataset=dataset,raw_path=str(RAW[dataset]),prepared=str(root),
                           role_map=ROLE_MAP,grouping=grouping,exposed_roles=['E','legacy'],
                           masks='all_observed',features=self.schema,
                           roles={r:dict(n=len(ids),domain_counts=np.bincount(self.domain[ids],minlength=3).tolist(),cells=cells(ids))
                                  for r,ids in self.roles.items()},
                           probe_rows=self.probe.tolist(),probe_panels=[v.tolist() for v in self.probe_panels],
                           S_windows=[v.tolist() for v in self.s_windows],S_window_support=[cells(v) for v in self.s_windows],
                           groups_disjoint_across_S=all(not np.intersect1d(self.group[a],self.group[b]).size
                           for i,a in enumerate(self.s_windows) for b in self.s_windows[i+1:]))

    def _members(self,role,ids):
        pool=self.roles[role]
        chosen=pool if ids is None else np.asarray(ids,np.int64)
        pos=np.searchsorted(pool,chosen)
        if (pos>=len(pool)).any() or not np.array_equal(pool[pos],chosen):
            raise PermissionError('Rows outside '+role)
        return chosen

    def inputs(self,role,ids=None):
        return np.asarray(self.x[self._members(role,ids)])

    def labels(self,role,purpose,ids=None,**context):
        if purpose not in PURPOSES.get(role,set()):
            raise PermissionError(f'{purpose} cannot read {role}')
        if role=='E' and self.final_authorization is None:
            raise PermissionError('Historical E requires a global frozen manifest')
        chosen=self._members(role,ids)
        if role=='S':
            used,read=self.s_state(context.get('method'))
            window=context['window']
            if window not in used:
                raise PermissionError('S requires a registered one-time transaction window')
            if window in read:
                raise PermissionError('S window labels have already been read')
            read.add(window)
        if self.access_log:
            append_jsonl(self.access_log,dict(role=role,purpose=purpose,row_ids=chosen.tolist(),**context))
        y=np.asarray(self._y[chosen])
        return y,np.ones_like(y,dtype=bool)

    def s_state(self,trajectory=None):
        if trajectory is None:
            return self.used_s,self.read_s
        if not hasattr(self,'s_consumption'):
            self.s_consumption={}
        return self.s_consumption.setdefault(trajectory,(set(),set()))

    def restore_s(self,trajectory,used,read):
        if not hasattr(self,'s_consumption'):
            self.s_consumption={}
        self.s_consumption[trajectory]=(set(used),set(read))

    def consume_s(self,window,trajectory=None):
        used,read=self.s_state(trajectory)
        if window not in range(3) or window in used:
            raise PermissionError('S window already used or invalid')
        used.add(window)
        return self.s_windows[window].copy()

    def authorize_final(self,manifest,worker_identity):
        key=f'{self.dataset}/seed{worker_identity["seed"]}'
        if manifest.get('status')!='FROZEN' or manifest.get('workers',{}).get(key)!=worker_identity:
            raise PermissionError('Global freeze differs from resident state')
        self.final_authorization=manifest['version']


def tensor(value,device):
    return torch.as_tensor(np.array(value),device=device)
