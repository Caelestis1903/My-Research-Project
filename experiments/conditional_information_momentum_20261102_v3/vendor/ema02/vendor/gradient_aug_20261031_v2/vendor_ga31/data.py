"""Pinned natural preprocessing; only numerical arrays/identity records reach disk."""
from pathlib import Path
import numpy as np
from .audited_data import _load_movielens_base, _load_kuairand_base, array_hash
from .source_data import sha_file
from .protocol import write_json


def load_dataset(dataset,p,raw_path=None,out=None):
    conf=p['datasets'][dataset]; path=raw_path or conf['raw_path_default']
    assert sha_file(path)==conf['raw_sha256'], 'Raw CSV identity mismatch'
    fn=_load_movielens_base if dataset=='movielens' else _load_kuairand_base
    bundle=fn(path,'natural',2022)
    manifest=bundle.manifest_fields()
    manifest.update(raw_path=str(Path(path).resolve()),raw_sha256=conf['raw_sha256'])
    manifest['array_hashes']={k:array_hash(getattr(bundle,k)) for k in conf['array_hashes']}
    for k in ('array_hashes','split_row_fingerprints','split_row_id_hashes'):
        assert manifest[k]==conf[k], (dataset,k,manifest[k],conf[k])
    for k in ('encoder_hash','scaler_hash'):
        assert getattr(bundle.features,k)==conf[k], (dataset,k)
    assert list(bundle.train_counts.values())==conf['train_domain_counts']
    for split,attribute in [('train','train_x'),('validation','val_x'),('test','test_x')]:
        assert len(getattr(bundle,attribute))==conf['split_rows'][split]
    if out:
        write_json(Path(out)/'data_manifest.json',manifest)
        np.savez_compressed(Path(out)/'split_row_ids.npz',train=bundle.train_row_ids,
                            validation=bundle.val_row_ids,test=bundle.test_row_ids)
    return bundle,manifest
