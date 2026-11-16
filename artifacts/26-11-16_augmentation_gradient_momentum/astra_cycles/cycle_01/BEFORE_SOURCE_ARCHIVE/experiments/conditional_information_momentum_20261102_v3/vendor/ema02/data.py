"""Natural source identities and the preregistered raw-row W1 adapter."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .protocol import load_protocol
from .vendor.gradient_aug_20261031_v2.vendor_ga31 import audited_data as source


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def window_indices(length):
    ids = np.arange(length, dtype=np.int64)
    a, b, c = (int(length * q) for q in (.6, .7, .8))
    return ids[:a], ids[a:b], ids[b:c]


def joint_counts(x, y):
    return {str(d): {f'{a}{b}': int(((x[:, -1] == d) & (y[:, 0] == a) &
                    (y[:, 1] == b)).sum()) for a in (0, 1) for b in (0, 1)}
            for d in range(3)}


def _window_movielens(path):
    # Reuse the source's complete encoder; only its explicit split dependency changes.
    from types import FunctionType
    fn = source._load_movielens_base
    globals_ = dict(fn.__globals__, _split_indices=window_indices)
    return FunctionType(fn.__code__, globals_, fn.__name__)(path, 'natural', 2022)


def _window_kuairand(path):
    required = list(source.KUAIRAND_SOURCE_COLUMNS) + ['is_click', 'long_view']
    tabs = pd.read_csv(path, usecols=['tab'], dtype='int8')['tab'].to_numpy()
    n = len(tabs); indices = window_indices(n)
    bounds = [(0, int(n * .6)), (int(n * .6), int(n * .7)),
              (int(n * .7), int(n * .8))]
    categorical = [c for c in source.KUAIRAND_SOURCE_COLUMNS if c != 'tab']
    strings = {'user_active_degree', 'follow_user_num_range', 'fans_user_num_range',
               'friend_user_num_range', 'register_days_range', 'video_type',
               'upload_dt', 'upload_type', 'tag'}
    types = {'tab': 'int8', 'is_click': 'int8', 'long_view': 'int8'}
    for c in categorical:
        types[c] = ('string' if c in strings else 'float32' if c.startswith('onehot_feat')
                    or c in ('server_width', 'server_height', 'music_type') else 'int32')
    vocab_sets = {c: set() for c in categorical}; offset = 0
    for chunk in pd.read_csv(path, usecols=required, dtype=types, chunksize=250000):
        stop = min(len(chunk), max(0, bounds[0][1] - offset))
        local = chunk.iloc[:stop]
        for c in categorical:
            vocab_sets[c].update(source._safe_value(v) for v in local[c].tolist())
        offset += len(chunk)
        if offset >= bounds[0][1]: break
    vocab = {c: {v: i + 1 for i, v in enumerate(sorted(values))}
             for c, values in vocab_sets.items()}
    columns = source.KUAIRAND_FEATURE_COLUMNS
    arrays = [np.zeros((stop - start, len(columns)), np.float32) for start, stop in bounds]
    labels = [np.zeros((len(x), 2), np.float32) for x in arrays]
    offset = 0
    for chunk in pd.read_csv(path, usecols=required, dtype=types, chunksize=250000):
        end = offset + len(chunk)
        for i, (start, stop) in enumerate(bounds):
            left, right = max(start, offset), min(stop, end)
            if left >= right: continue
            local = chunk.iloc[left - offset:right - offset]
            target = arrays[i][left - start:right - start]
            for j, c in enumerate(columns):
                target[:, j] = (local['tab'].map({0: 0, 1: 1, 4: 2}).to_numpy()
                                if c == 'domain_indicator' else source._apply_vocab(local[c], vocab[c]))
            labels[i][left - start:right - start] = local[['is_click', 'long_view']].to_numpy(np.float32)
        offset = end
        if offset >= bounds[-1][1]: break
    values = []
    for x, y, ids in zip(arrays, labels, indices):
        mask = np.isin(x[:, -1], (0, 1, 2))
        values.append((x[mask], y[mask], ids[mask]))
    schema = source._feature_schema('kuairand', columns, (), source.KUAIRAND_SPARSE_COLUMNS,
        {c: 3 if c == 'domain_indicator' else len(vocab[c]) + 1
         for c in source.KUAIRAND_SPARSE_COLUMNS}, vocab, {})
    return source.DataBundle('kuairand', 'natural', 2022, schema,
        values[0][0], values[0][1], values[1][0], values[1][1], values[2][0], values[2][1],
        values[0][2], values[1][2], values[2][2])


def load_dataset(dataset, split_id='original', p=None):
    p = p or load_protocol(); conf = p['datasets'][dataset]; identity = conf['source']
    path = conf['csv']; raw_hash = sha_file(path)
    assert raw_hash == identity['raw_sha256'], 'Raw CSV identity mismatch'
    assert split_id in ('original', 'W1')
    fn = (source._load_movielens_base if dataset == 'movielens' else source._load_kuairand_base)
    bundle = (fn(path, 'natural', 2022) if split_id == 'original' else
              (_window_movielens(path) if dataset == 'movielens' else _window_kuairand(path)))
    for y in (bundle.train_y, bundle.val_y):
        assert np.isfinite(y).all() and np.isin(y, (0, 1)).all(), 'Nonbinary/missing labels'
    manifest = dict(dataset=dataset, split_id=split_id, raw_path=str(Path(path).resolve()),
        raw_sha256=raw_hash, array_hashes={k: source.array_hash(getattr(bundle, k))
        for k in identity['array_hashes']}, split_row_fingerprints=bundle.split_fingerprints(),
        split_row_id_hashes=bundle.row_id_fingerprints(), encoder_hash=bundle.features.encoder_hash,
        scaler_hash=bundle.features.scaler_hash, features=bundle.features.as_dict(),
        train_domain_counts=list(bundle.train_counts.values()),
        split_rows=dict(train=len(bundle.train_y), validation=len(bundle.val_y), test=len(bundle.test_y)),
        train_joint_labels=joint_counts(bundle.train_x, bundle.train_y),
        preprocessing_fit='train_only', test_metrics_accessed=False)
    if dataset == 'movielens':
        assert not any(v['01'] for v in manifest['train_joint_labels'].values()), 'ML label nesting mismatch'
    if split_id == 'original':
        for key in ('array_hashes', 'split_row_fingerprints', 'split_row_id_hashes',
                    'encoder_hash', 'scaler_hash', 'train_domain_counts', 'split_rows'):
            assert manifest[key] == identity[key], (dataset, key)
    return bundle, manifest
