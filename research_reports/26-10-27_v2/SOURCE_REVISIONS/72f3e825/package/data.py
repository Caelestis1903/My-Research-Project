"""Pinned natural rows, fit-only preparation and explicit label permissions."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from .vendor_snapshot.mdmt_dfd_20261026.vendor.strict_data.data_roles import (
    RoleDataset, prepare, assign_roles, _raw_schema, _chunks, _strings, EXPECTED)
from .state import array_hash, digest, sha_file, write_json, append_jsonl

RAW = dict(movielens='/home_new/xzf/datasets/ml-1m/ml-1m.csv',
           kuairand='/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv')
ROLE_MAP = dict(T0=('fit',), F=('P_label',), S=('P_cal', 'P_update'),
                Q=('Q_explore',), V=('validation',), E=('test',), legacy=('Q_confirm',))
PURPOSES = dict(T0={'native', 'pilot', 'normalizer', 'audit'}, F={'adapt'},
                S={'diagnostic'}, Q={'development', 'diagnostic'}, V={'select'},
                E={'final'}, legacy=set())


def verify_prepared(dataset, root):
    old = RoleDataset(root)
    raw_path = Path(RAW[dataset])
    if sha_file(raw_path) != EXPECTED[dataset]:
        raise ValueError('Raw identity mismatch')
    domains, columns, dense = _raw_schema(dataset, raw_path)
    if not np.array_equal(domains, old.x[:, -1]):
        raise ValueError('Raw/encoded domain mismatch')
    roles = assign_roles(domains)
    if any(not np.array_equal(roles[k], old.roles[k]) for k in roles):
        raise ValueError('Row-plan mismatch')
    prep = json.loads((Path(root)/'PREPROCESSOR.json').read_text())
    fitmask = np.zeros(len(domains), bool)
    fitmask[roles['fit']] = True
    seen = {key: set() for key in prep['vocabulary']}
    low, high, offset = float('inf'), -float('inf'), 0
    for chunk, labels in _chunks(dataset, raw_path):
        end = offset+len(chunk)
        if not np.array_equal(labels, old._labels[offset:end]):
            raise ValueError('Raw/encoded label mismatch')
        fit = chunk.loc[fitmask[offset:end]]
        for key in seen:
            seen[key].update(_strings(fit[key]).unique())
        if dense and len(fit):
            low, high = min(low, float(fit.age.min())), max(high, float(fit.age.max()))
        for j, key in enumerate(columns):
            if key == 'domain_indicator':
                expected = domains[offset:end]
            elif key in dense:
                scaler = prep['scaler']
                expected = (chunk[key].to_numpy(np.float32)-scaler['age_min'])/max(scaler['age_max']-scaler['age_min'], 1e-12)
            else:
                expected = _strings(chunk[key]).map(prep['vocabulary'][key]).fillna(0).to_numpy(np.float32)
            if not np.array_equal(expected, old.x[offset:end, j]):
                raise ValueError('Prepared value mismatch: '+key)
        offset = end
    for key, values in seen.items():
        if {v:i+1 for i,v in enumerate(sorted(values))} != prep['vocabulary'][key]:
            raise ValueError('Vocabulary is not T0-only: '+key)
    if dense and prep['scaler'] != dict(age_min=low, age_max=high):
        raise ValueError('Scaler is not T0-only')
    receipt = dict(dataset=dataset, raw_sha256=EXPECTED[dataset],
                   data_hash=old.manifest['data_hash'], verified_encoding=True,
                   verified_labels=True, verified_fit_only=True, verified_rows=True,
                   preprocessor_hash=digest(prep))
    write_json(Path(root)/'VALIDATED.json', receipt)
    return receipt


def prepare_data(dataset, artifact_root):
    root = Path(artifact_root)/'data'/dataset
    if not root.exists():
        prepare(dataset, root, raw_path=RAW[dataset])
    return verify_prepared(dataset, root)


class Data:
    def __init__(self, dataset, artifact_root, access_log=None):
        root = Path(artifact_root)/'data'/dataset
        self.old = old = RoleDataset(root)
        self.dataset, self.x, self.y = dataset, old.x, old._labels
        self.domain = np.asarray(self.x[:, -1], dtype=np.int64)
        self.schema = old.manifest['features']
        receipt = json.loads((root/'VALIDATED.json').read_text())
        if receipt['data_hash'] != old.manifest['data_hash'] or receipt['raw_sha256'] != EXPECTED[dataset]:
            raise ValueError('Prepared verification identity mismatch')
        self.roles = {k:np.sort(np.concatenate([old.roles[r] for r in rs])) for k, rs in ROLE_MAP.items()}
        if not np.array_equal(np.sort(np.concatenate(list(self.roles.values()))), np.arange(len(self.x))):
            raise ValueError('Roles not exhaustive/disjoint')
        self.split_roles = {}
        pieces = [[], []]
        for d in range(3):
            pool = self.roles['F'][self.domain[self.roles['F']] == d]
            order = np.random.Generator(np.random.PCG64(20261025+60000+1009*d)).permutation(pool)
            pieces[0].extend(order[:len(order)//2])
            pieces[1].extend(order[len(order)//2:])
        for name, values in zip(('P_form', 'P_read'), pieces):
            self.split_roles[name] = np.sort(np.asarray(values, dtype=np.int64))
        self.manifest = dict(dataset=dataset, parent_data_hash=old.manifest['data_hash'],
                             raw_sha256=EXPECTED[dataset], preprocessor_hash=receipt['preprocessor_hash'],
                             features=self.schema, role_map=ROLE_MAP,
                             roles={r:dict(n=len(ids), row_id_hash=array_hash(ids),
                                            domain_counts=np.bincount(self.domain[ids], minlength=3).tolist())
                                    for r, ids in self.roles.items()},
                             split_roles={r:array_hash(ids) for r, ids in self.split_roles.items()},
                             exposed_roles=['E', 'legacy'], masks='all_observed')
        self.identity = digest(self.manifest)
        self.manifest['data_role_hash'] = self.identity
        self.access_log = access_log
        self.outer_frozen = None

    def authorize_final(self, manifest, worker_identity):
        from .config import DATASETS, SEEDS, METHODS
        required = {f'{d}/seed{s}' for d in DATASETS for s in SEEDS}
        if set(manifest.get('workers', {})) != required or manifest.get('methods') != list(METHODS):
            raise PermissionError('All eight workers and six methods must be frozen')
        if manifest.get('config_hash') != worker_identity['config_hash']:
            raise PermissionError('Frozen configuration differs')
        if manifest.get('source_hash') != worker_identity['source_hash']:
            raise PermissionError('Frozen implementation differs')
        expected = manifest.get('workers', {}).get(f'{self.dataset}/seed{worker_identity["seed"]}')
        if manifest.get('status') != 'FROZEN' or expected != worker_identity:
            raise PermissionError('Global frozen manifest does not match resident worker')
        self.outer_frozen = digest(manifest)

    def labels(self, role, purpose, ids=None, **context):
        if purpose not in PURPOSES.get(role, set()):
            raise PermissionError(f'{purpose} cannot access {role}')
        if role == 'E' and self.outer_frozen is None:
            raise PermissionError('E is locked until global freeze')
        pool = self.roles[role]
        ids = pool if ids is None else np.asarray(ids, dtype=np.int64)
        positions = np.searchsorted(pool, ids)
        if (positions >= len(pool)).any() or not np.array_equal(pool[positions], ids):
            raise PermissionError('Rows outside role')
        if self.access_log:
            append_jsonl(self.access_log, dict(role=role, purpose=purpose, rows=len(ids),
                                               row_id_hash=array_hash(ids), **context))
        return np.asarray(self.y[ids]), np.ones((len(ids), 2), dtype=bool)

    def inputs(self, role, ids=None):
        pool = self.roles[role]
        ids = pool if ids is None else np.asarray(ids, dtype=np.int64)
        return np.asarray(self.x[ids])


def tensor(value, device):
    return torch.as_tensor(np.array(value), device=device)

