"""Raw-identity checked, fit-only preprocessing and auditable label access."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .artifacts import array_hash, digest, sha_file, write_json, append_jsonl

RAW = {'movielens': '/home/xzf/data/ml-1m/ml-1m.csv',
       'kuairand': '/home/xzf/data/kuairand/processed/kuai_rand.csv'}
EXPECTED = {'movielens': 'f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4',
            'kuairand': 'a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f'}
PROTOCOL = 'natural_roworder_fit80_probe10_query10_v1'
SPLIT_SEED = 20261024
ROLES = ('fit', 'P_label', 'P_cal', 'P_update', 'Q_explore', 'Q_confirm', 'validation', 'test')
PURPOSES = {'fit': {'main', 'preprocess', 'initialize', 'support', 'semantic_check'},
            'P_label': {'response', 'direct', 'semantic_check'},
            'P_cal': {'calibrate', 'semantic_check'},
            'P_update': {'update_feedback', 'semantic_check'},
            'Q_explore': {'explore'}, 'Q_confirm': {'confirm'},
            'validation': {'select', 'semantic_check'}, 'test': {'final'}}
ML_COLUMNS = ('user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'cate_id', 'domain_indicator')
KR_SOURCE = ('tab', 'user_active_degree', 'is_live_streamer', 'is_video_author',
             'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range',
             'register_days_range', *(f'onehot_feat{i}' for i in range(18)),
             'video_type', 'upload_dt', 'upload_type', 'server_width', 'server_height',
             'music_type', 'tag', 'counts')


@dataclass
class FeatureSchema:
    dataset: str
    columns: tuple
    dense_columns: tuple
    sparse_columns: tuple
    vocab_sizes: dict
    embedding_dim: int = 16
    domain_column: str = 'domain_indicator'
    encoder_hash: str = ''
    scaler_hash: str = ''
    oov_index: int = 0

    def as_dict(self):
        return asdict(self)


def assign_roles(domains, split_seed=SPLIT_SEED):
    """No labels are accepted by this API. Returned IDs preserve source row order."""
    if int(split_seed) != SPLIT_SEED:
        raise ValueError('New split seed requires a new protocol_id')
    domains = np.asarray(domains, dtype=np.int64)
    if not np.isin(domains, [0, 1, 2]).all():
        raise ValueError('domain outside frozen natural mapping')
    n = len(domains)
    stop, val_stop = (8 * n) // 10, (9 * n) // 10
    result = {role: [] for role in ROLES}
    for d in range(3):
        pool = np.flatnonzero(domains[:stop] == d)
        perm = np.random.Generator(np.random.PCG64(split_seed + 1009*d)).permutation(pool)
        a, b = (8*len(pool))//10, (9*len(pool))//10
        result['fit'].extend(perm[:a])
        probe = np.sort(perm[a:b])
        probe = np.random.Generator(np.random.PCG64(split_seed + 20000 + 1009*d)).permutation(probe)
        c, e = (6*len(probe))//10, (8*len(probe))//10
        for role, ids in zip(('P_label', 'P_cal', 'P_update'), (probe[:c], probe[c:e], probe[e:])):
            result[role].extend(ids)
        query = np.sort(perm[b:])
        query = np.random.Generator(np.random.PCG64(split_seed + 40000 + 1009*d)).permutation(query)
        result['Q_explore'].extend(query[:len(query)//2])
        result['Q_confirm'].extend(query[len(query)//2:])
    result['validation'] = np.arange(stop, val_stop, dtype=np.int64)
    result['test'] = np.arange(val_stop, n, dtype=np.int64)
    result = {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in result.items()}
    combined = np.concatenate(list(result.values()))
    if len(combined) != n or len(np.unique(combined)) != n:
        raise AssertionError('role partition is not exhaustive and disjoint')
    return result


def _strings(series):
    # Matches audited_data._safe_value (including float32 categorical parsing).
    return series.map(lambda v: '<NA>' if pd.isna(v) else str(v))


def _raw_schema(dataset, path):
    if dataset == 'movielens':
        raw = pd.read_csv(path)
        domain = raw['age'].map({1: 0, 18: 0, 25: 1, 35: 2, 45: 2, 50: 2, 56: 2})
        if domain.isna().any():
            raise ValueError('Unknown age in raw MovieLens')
        return domain.to_numpy(np.int64), ML_COLUMNS, ('age',)
    tabs = pd.read_csv(path, usecols=['tab'], dtype={'tab': 'int8'})['tab']
    domain = tabs.map({0: 0, 1: 1, 4: 2})
    if domain.isna().any():
        raise ValueError('Unknown KuaiRand tab')
    return domain.to_numpy(np.int64), tuple(KR_SOURCE[1:]) + ('domain_indicator',), ()


def _chunks(dataset, path, size=250000):
    if dataset == 'movielens':
        for chunk in pd.read_csv(path, chunksize=size):
            chunk['cate_id'] = chunk['genres'].astype(str).str.split('|').str[0]
            labels = np.stack(((chunk['rating'].astype(int) > 3), (chunk['rating'].astype(int) > 4)), axis=1)
            yield chunk, labels.astype(np.float32)
    else:
        strings = {'user_active_degree', 'follow_user_num_range', 'fans_user_num_range',
                   'friend_user_num_range', 'register_days_range', 'video_type',
                   'upload_dt', 'upload_type', 'tag'}
        types = {'tab': 'int8', 'is_click': 'int8', 'long_view': 'int8'}
        for col in KR_SOURCE[1:]:
            types[col] = ('string' if col in strings else 'float32' if col.startswith('onehot_feat')
                          or col in ('server_width', 'server_height', 'music_type') else 'int32')
        for chunk in pd.read_csv(path, usecols=list(KR_SOURCE)+['is_click', 'long_view'],
                                 dtype=types, chunksize=size):
            yield chunk, chunk[['is_click', 'long_view']].to_numpy(np.float32)


class RoleDataset:
    def __init__(self, root, access_log=None):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'DATA_MANIFEST.json').read_text())
        identity = dict(self.manifest)
        recorded = identity.pop('data_hash', None)
        if recorded is None or digest(identity) != recorded:
            raise ValueError('Data manifest identity mismatch')
        for name in ('x', 'y'):
            if sha_file(self.root/f'{name}.npy') != self.manifest['array_hashes'][name]:
                raise ValueError('Prepared array checksum mismatch: '+name)
        preprocessing = json.loads((self.root/'PREPROCESSOR.json').read_text())
        if (digest(preprocessing['vocabulary']) != self.manifest['features']['encoder_hash'] or
                digest(preprocessing['scaler']) != self.manifest['features']['scaler_hash'] or
                preprocessing['fit_row_hash'] != self.manifest['roles']['fit']['row_id_hash']):
            raise ValueError('Preprocessor identity or fit scope mismatch')
        self.features = FeatureSchema(**self.manifest['features'])
        self.x = np.load(self.root/'x.npy', mmap_mode='r')
        self._labels = np.load(self.root/'y.npy', mmap_mode='r')
        self.roles = {role: np.load(self.root/f'rows_{role}.npy') for role in ROLES}
        self.access_log = Path(access_log) if access_log else None
        for role, ids in self.roles.items():
            if array_hash(ids) != self.manifest['roles'][role]['row_id_hash']:
                raise ValueError('Corrupt role row IDs: '+role)
        combined = np.concatenate(list(self.roles.values()))
        if len(combined) != len(self.x) or not np.array_equal(np.sort(combined), np.arange(len(self.x))):
            raise ValueError('Corrupt disjoint role partition')

    @property
    def dataset(self):
        return self.manifest['dataset']

    def rows(self, role, purpose, *, ids=None, authorization=None):
        if role not in PURPOSES or purpose not in PURPOSES[role]:
            raise PermissionError(f'{purpose} cannot access labels from {role}')
        if role in ('test', 'Q_confirm'):
            if authorization is None or not authorization.permits(role, self.manifest['data_hash']):
                raise PermissionError(f'{role} requires a verified matrix authorization')
        pool = self.roles[role]
        selected = pool if ids is None else np.asarray(ids, dtype=np.int64)
        if not np.isin(selected, pool, assume_unique=False).all():
            raise PermissionError('Requested rows outside authorized role')
        if self.access_log:
            append_jsonl(self.access_log, {'role': role, 'purpose': purpose,
                                         'row_hash': array_hash(selected), 'count': len(selected)})
        return np.asarray(self.x[selected]), np.asarray(self._labels[selected]), selected.copy()

    def inputs(self, role, ids=None):
        selected = self.roles[role] if ids is None else np.asarray(ids, dtype=np.int64)
        if not np.isin(selected, self.roles[role]).all():
            raise ValueError('IDs outside input role')
        return np.asarray(self.x[selected]), selected.copy()


def prepare(dataset, destination, raw_path=None, expected_sha=None):
    """Build from raw CSV; no legacy encoded cache is accepted."""
    if dataset not in RAW:
        raise ValueError('dataset must be movielens or kuairand')
    path = Path(raw_path or RAW[dataset])
    if not path.is_file():
        raise FileNotFoundError(f'DATA_UNAVAILABLE: {path}')
    actual = sha_file(path)
    expected = expected_sha or EXPECTED[dataset]
    if actual != expected:
        raise ValueError(f'Raw SHA mismatch: expected {expected}; observed {actual}')
    if expected != EXPECTED[dataset]:
        raise ValueError('Alternate raw identities require a separate protocol')
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    domains, columns, dense = _raw_schema(dataset, path)
    roles = assign_roles(domains)
    fit_mask = np.zeros(len(domains), bool)
    fit_mask[roles['fit']] = True
    sparse = tuple(c for c in columns if c not in dense)
    vocab_sets = {c: set() for c in sparse if c != 'domain_indicator'}
    age_min, age_max = float('inf'), -float('inf')
    offset = 0
    for chunk, _ in _chunks(dataset, path):
        local = chunk.loc[fit_mask[offset:offset+len(chunk)]]
        for col in vocab_sets:
            vocab_sets[col].update(_strings(local[col]).unique().tolist())
        if dense and len(local):
            age_min, age_max = min(age_min, float(local.age.min())), max(age_max, float(local.age.max()))
        offset += len(chunk)
    vocab = {c: {v: i+1 for i, v in enumerate(sorted(values))} for c, values in vocab_sets.items()}
    scaler = {'age_min': age_min, 'age_max': age_max} if dense else {}
    schema = FeatureSchema(dataset, tuple(columns), tuple(dense), sparse,
                           {c: 3 if c == 'domain_indicator' else len(vocab[c])+1 for c in sparse},
                           encoder_hash=digest(vocab), scaler_hash=digest(scaler))
    x = np.lib.format.open_memmap(root/'x.npy', mode='w+', dtype=np.float32, shape=(len(domains), len(columns)))
    y = np.lib.format.open_memmap(root/'y.npy', mode='w+', dtype=np.float32, shape=(len(domains), 2))
    offset = 0
    for chunk, labels in _chunks(dataset, path):
        end = offset+len(chunk)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError('Stored task labels must be binary')
        for j, col in enumerate(columns):
            if col == 'domain_indicator':
                x[offset:end, j] = domains[offset:end]
            elif col in dense:
                x[offset:end, j] = (chunk[col].to_numpy(np.float32)-age_min)/max(age_max-age_min, 1e-12)
            else:
                x[offset:end, j] = _strings(chunk[col]).map(vocab[col]).fillna(0).to_numpy(np.float32)
        y[offset:end] = labels
        offset = end
    x.flush(); y.flush()
    write_json(root/'PREPROCESSOR.json', {'fit_row_hash': array_hash(roles['fit']), 'vocabulary': vocab, 'scaler': scaler})
    role_info = {}
    for role, ids in roles.items():
        np.save(root/f'rows_{role}.npy', ids)
        cells = []
        for q in range(2):
            for d in range(3):
                sub = ids[domains[ids] == d]
                positive = int(y[sub, q].sum())
                cells.append({'cell': q*3+d, 'domain': d, 'task': q,
                              'n': len(sub), 'positive': positive, 'negative': len(sub)-positive})
        role_info[role] = {'n': len(ids), 'row_id_hash': array_hash(ids), 'cells': cells}
    manifest = {'schema_version': 'mdmt-next-v1', 'protocol_id': PROTOCOL,
                'dataset': dataset, 'raw_path': str(path.resolve()), 'raw_sha256': actual,
                'split_seed': SPLIT_SEED, 'preprocess_fit_role': 'fit', 'features': asdict(schema),
                'roles': role_info, 'array_hashes': {'x': sha_file(root/'x.npy'), 'y': sha_file(root/'y.npy')},
                'label_definitions': ['rating>3', 'rating>4'] if dataset == 'movielens' else ['stored_is_click', 'stored_long_view'],
                'provenance': 'natural source row order; no verified user-disjoint or temporal claim'}
    manifest['data_hash'] = digest(manifest)
    write_json(root/'DATA_MANIFEST.json', manifest, exclusive=True)
    return RoleDataset(root)
