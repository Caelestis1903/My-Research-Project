"""Fixed training probes and validation row folds, independent of training RNG."""
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

from experiments.overnight_ab_20261103.state import digest, read_json, sha_file
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.audited_data import array_hash

NAMESPACE = 'EMA_MECHANISM_20261106'
TRAIN_KEYS = ('rank_indices', 'rank_row_ids', 'rank_y', 'rank_pairs',
              'natural_indices', 'natural_row_ids', 'natural_y')
VALIDATION_KEYS = ('indices', 'row_ids', 'y', 'validation_row_side', 'pairs', 'pair_counts')


def row_side(dataset, row_ids, seed=None):
    prefix = f'{dataset}|' if seed is None else f'{NAMESPACE}|train_partition|{dataset}|{seed}|'
    return np.fromiter((hashlib.sha256((prefix+str(int(row))).encode()).digest()[-1] & 1
                        for row in np.asarray(row_ids).ravel()), dtype=np.int8,
                       count=np.asarray(row_ids).size).reshape(np.asarray(row_ids).shape)


def _generator(dataset, seed, panel, role):
    key = f'{NAMESPACE}|probe|{dataset}|{seed}|{panel}|{role}'
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
    return np.random.default_rng(value)


def _sample(pool, count, generator, role):
    if len(pool) < count:
        raise ValueError(f'PANEL_POPULATION_MISMATCH: {role}: need {count}, have {len(pool)}')
    return generator.choice(pool, size=count, replace=False).astype(np.int64)


def _pairs(positive, negative, count, generator):
    available = len(positive)*len(negative)
    chosen = generator.choice(available, size=min(count, available), replace=False)
    if not available:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack((positive[chosen//len(negative)], negative[chosen % len(negative)])).astype(np.int32)


def _identity(panel, dataset, seed, spec):
    keys = TRAIN_KEYS if panel['panel_id'] in ('P_A', 'P_B') else VALIDATION_KEYS
    arrays = {key: array_hash(panel[key]) for key in keys}
    identity = dict(schema_version='1.0', namespace=NAMESPACE, dataset=dataset, seed=int(seed),
                    panel_id=panel['panel_id'], spec=dict(spec), arrays=arrays)
    panel.update(identity=identity, panel_hash=digest(identity),
                 pair_hash=array_hash(panel['rank_pairs'] if panel['panel_id'] != 'V_mech' else panel['pairs']))
    return panel


def _attach_inputs(panel, bundle):
    if panel['panel_id'] == 'V_mech':
        for original, alias in (('indices', 'natural_indices'), ('row_ids', 'natural_row_ids'),
                                ('y', 'natural_y'), ('pairs', 'rank_pairs')):
            panel[alias] = panel[original]
        panel['valid_mask']=panel['indices']>=0
        if bundle is not None:
            valid=panel['valid_mask']
            x=np.zeros((*panel['indices'].shape,bundle.val_x.shape[1]),dtype=bundle.val_x.dtype)
            x[valid]=bundle.val_x[panel['indices'][valid]]
            panel['natural_x']=x
    elif bundle is not None:
        panel['rank_x'] = np.asarray(bundle.train_x[panel['rank_indices']])
        panel['natural_x'] = np.asarray(bundle.train_x[panel['natural_indices']])
    return panel


def build_panels(bundle, seed, spec):
    dataset = bundle.dataset
    positive_n, negative_n = spec['rank_positive_per_cell'], spec['rank_negative_per_cell']
    pair_n = spec['rank_pairs_per_cell']
    partition = row_side(dataset, bundle.train_row_ids, seed)
    train_domains = np.asarray(bundle.train_x[:, -1], dtype=np.int64)
    result = {}
    for side, name in enumerate(('P_A', 'P_B')):
        rank, pair_rows, natural = [], [], []
        for domain in range(3):
            domain_pool = np.flatnonzero((train_domains == domain) & (partition == side))
            natural.append(_sample(domain_pool, spec['natural_rows_per_train_domain'],
                _generator(dataset, seed, name, f'd{domain}_natural'), f'{name}/d{domain}/natural'))
            for task in range(2):
                chosen = []
                for label, count in ((1, positive_n), (0, negative_n)):
                    pool = domain_pool[np.asarray(bundle.train_y[domain_pool, task]) == label]
                    chosen.append(_sample(pool, count,
                        _generator(dataset, seed, name, f'd{domain}t{task}_label{label}'),
                        f'{name}/d{domain}t{task}/label{label}'))
                rank.append(np.concatenate(chosen))
                pair_rows.append(_pairs(np.arange(positive_n), positive_n+np.arange(negative_n),
                    pair_n, _generator(dataset, seed, name, f'd{domain}t{task}_pairs')))
        rank = np.stack(rank); natural = np.stack(natural)
        labels = np.stack([np.asarray(bundle.train_y[rank[c], c % 2]) for c in range(6)])
        panel = dict(panel_id=name, rank_indices=rank, rank_row_ids=np.asarray(bundle.train_row_ids[rank]),
            rank_y=labels, rank_pairs=np.stack(pair_rows), natural_indices=natural,
            natural_row_ids=np.asarray(bundle.train_row_ids[natural]), natural_y=np.asarray(bundle.train_y[natural]))
        result[name] = _attach_inputs(_identity(panel, dataset, seed, spec), bundle)
    domains = np.asarray(bundle.val_x[:, -1], dtype=np.int64)
    natural = [np.flatnonzero(domains == domain) for domain in range(3)]
    natural = [_sample(pool, min(spec['validation_natural_rows_per_domain'], len(pool)),
        _generator(dataset, seed, 'V_mech', f'd{domain}_natural'), f'V_mech/d{domain}')
        for domain, pool in enumerate(natural)]
    width=spec['validation_natural_rows_per_domain']
    indices=np.full((3,width),-1,dtype=np.int64)
    valid=np.zeros((3,width),dtype=bool)
    for domain,selected in enumerate(natural):
        indices[domain,:len(selected)]=selected;valid[domain,:len(selected)]=True
    labels=np.full((3,width,2),-1,dtype=bundle.val_y.dtype)
    labels[valid]=bundle.val_y[indices[valid]]
    pairs_n = spec['validation_pairs_per_cell_for_temporal_stat']
    pairs = np.full((6, pairs_n, 2), -1, dtype=np.int32); counts = []
    for domain in range(3):
        for task in range(2):
            y = labels[domain, :, task]
            selected = _pairs(np.flatnonzero(y == 1), np.flatnonzero(y == 0), pairs_n,
                _generator(dataset, seed, 'V_mech', f'd{domain}t{task}_pairs'))
            cell = 2*domain+task; pairs[cell, :len(selected)] = selected; counts.append(len(selected))
    rows=np.full(indices.shape,-1,dtype=bundle.val_row_ids.dtype)
    rows[valid]=bundle.val_row_ids[indices[valid]]
    sides=np.full(indices.shape,-1,dtype=np.int8)
    sides[valid]=row_side(dataset,rows[valid])
    panel = dict(panel_id='V_mech', indices=indices, row_ids=rows, y=labels,
        validation_row_side=sides, pairs=pairs, pair_counts=np.asarray(counts, dtype=np.int32))
    result['V_mech'] = _attach_inputs(_identity(panel, dataset, seed, spec), bundle)
    a = np.concatenate((result['P_A']['rank_row_ids'].ravel(), result['P_A']['natural_row_ids'].ravel()))
    b = np.concatenate((result['P_B']['rank_row_ids'].ravel(), result['P_B']['natural_row_ids'].ravel()))
    if np.intersect1d(a, b).size:
        raise ValueError('Training observation panels overlap')
    return result


def save_panels(out, dataset, seed, panels):
    out = Path(out); result = {}
    for name, panel in panels.items():
        path = out/'panels'/dataset/f'seed{seed}'/(name+'.npz')
        keys = TRAIN_KEYS if name in ('P_A', 'P_B') else VALIDATION_KEYS
        arrays = {key: panel[key] for key in keys}
        if path.exists():
            old = load_panel_identity(path)
            if set(old) != set(arrays) or any(not np.array_equal(old[k], arrays[k]) for k in arrays):
                raise ValueError('Panel identity conflicts with existing observations: '+str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile('wb',dir=path.parent,delete=False) as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush();os.fsync(stream.fileno());temporary=stream.name
            os.replace(temporary,path)
        result[name] = dict(path=str(path.relative_to(out)), sha256=sha_file(path),
            panel_hash=panel['panel_hash'], pair_hash=panel['pair_hash'], identity=panel['identity'],
            arrays={key:dict(shape=list(value.shape), dtype=str(value.dtype)) for key,value in arrays.items()})
        if name=='V_mech':
            result[name].update(valid_mask_rule='valid_mask=(indices>=0)',
                actual_count=(panel['indices']>=0).sum(axis=1).tolist(),
                representation_version='validation_masked_cap_v1')
    return result


def load_panel_identity(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    if any(value.dtype.hasobject for value in result.values()):
        raise ValueError('Object arrays are forbidden in panel identity')
    return result


def load_panels(out, dataset, seed, bundle=None):
    out = Path(out)
    records = read_json(out/'PROBE_MANIFEST.json')['panels'][dataset][str(seed)]
    result = {}
    for name, receipt in records.items():
        path = out/receipt['path']
        if sha_file(path) != receipt['sha256'] or digest(receipt['identity']) != receipt['panel_hash']:
            raise ValueError('Frozen panel receipt changed: '+name)
        panel = load_panel_identity(path)
        if {key:array_hash(value) for key,value in panel.items()} != receipt['identity']['arrays']:
            raise ValueError('Frozen panel arrays changed: '+name)
        panel.update(panel_id=name, panel_hash=receipt['panel_hash'], pair_hash=receipt['pair_hash'],
                     identity=receipt['identity'])
        result[name] = _attach_inputs(panel, bundle)
    return result
