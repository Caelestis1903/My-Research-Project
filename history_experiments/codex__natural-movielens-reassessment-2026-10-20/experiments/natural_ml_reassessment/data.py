"""One natural F0 population and two prespecified fixed control populations."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from .bootstrap import activate, ROOT, dump, sha
activate()
from stage_F0_data import load_movielens_natural
from stage_S1_data import role_reversal_sample, row_ids_hash, array_hash

POPULATIONS = ('movielens_natural', 'movielens_artificial_d2_fixed2022',
               'movielens_natural_matched_n187500')

def load_population(population='movielens_natural', dataset_path=None):
    if population not in POPULATIONS:
        raise ValueError(population)
    # Population identity never depends on the optimizer seed.
    b = load_movielens_natural(dataset_path, seed=2022)
    x, y, ids = b.train_x, b.train_y, b.train_row_ids
    metadata = dict(b.preprocessing_metadata, dataset_config=population,
                    population_seed=2022, optimizer_seed_changes_population=False)
    if population == POPULATIONS[1]:
        x, y, ids = role_reversal_sample(x, y, 'd2_large', 2022, row_ids=ids)
        metadata['selection'] = 'original role_reversal_sample d2_large seed=2022'
        metadata['training_pool'] = 'fixed_d2_large_subsample'
    elif population == POPULATIONS[2]:
        counts = np.asarray([b.train_counts[d] for d in range(3)], dtype=np.int64)
        exact = 187500 * counts.astype(np.float64)/counts.sum()
        quotas = np.floor(exact).astype(np.int64)
        for d in np.argsort(-(exact-quotas), kind='stable')[:187500-int(quotas.sum())]:
            quotas[d] += 1
        selected = []
        for d in range(3):
            pool = np.flatnonzero(x[:, -1] == d)
            if quotas[d] > len(pool):
                raise ValueError('specified matched N population unavailable')
            selected.extend(np.random.RandomState(20221020+d).choice(pool, quotas[d], replace=False))
        chosen = np.sort(np.asarray(selected, dtype=np.int64))
        x, y, ids = x[chosen], y[chosen], ids[chosen]
        metadata.update(selection='largest remainder natural p; independent RandomState(20221020+d); sorted original rows',
                        quotas=quotas.tolist(), selection_seed_base=20221020,training_pool='natural_proportion_matched_N')
        assert len(ids) == 187500
    metadata.update(selected_train_row_ids_hash=row_ids_hash(ids),selected_train_size=int(len(ids)),
                    base_train_size=int(len(b.train_row_ids)),
                    manifest_legacy_field_note='base_split_row_id_hashes in historical DataBundle.manifest_fields is selected split IDs; use explicit preprocessing_metadata.base_train_row_ids_hash for natural base population')
    return replace(b, train_x=x, train_y=y, train_row_ids=ids,
                   split_row_id_hashes={**b.split_row_id_hashes, 'train': row_ids_hash(ids)},
                   preprocessing_metadata=metadata)

def freeze_data(output=ROOT/'data', dataset_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for population in POPULATIONS:
        b = load_population(population, dataset_path)
        arrays = {}
        hashes = {}
        for split in ('train', 'val', 'test'):
            for suffix in ('x', 'y', 'row_ids'):
                key = split+'_'+suffix
                arrays[key] = getattr(b, key)
                hashes[key] = array_hash(arrays[key])
        path = output/(population+'.npz')
        if not path.exists():
            np.savez_compressed(path, **arrays)
        else:
            with np.load(path) as saved:
                if any(array_hash(saved[k]) != value for k, value in hashes.items()):
                    raise RuntimeError('existing frozen data differs: '+str(path))
        manifest = dict(dataset_config=population, data=b.manifest_fields(),
                        schema=b.features.as_dict(), array_hashes=hashes,
                        archive=str(path), archive_sha256=sha(path))
        dump(output/(population+'.json'), manifest)
        manifests[population] = manifest
    # All schemas and held-out data must be byte-identical.
    baseline = manifests[POPULATIONS[0]]
    for value in manifests.values():
        assert value['schema'] == baseline['schema']
        for key in ('val_x', 'val_y', 'val_row_ids', 'test_x', 'test_y', 'test_row_ids'):
            assert value['array_hashes'][key] == baseline['array_hashes'][key]
    dump(output/'manifest.json', manifests)
    return manifests

if __name__ == '__main__':
    freeze_data()
