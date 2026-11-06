"""Source-bound budgets and streaming data identity checks."""
import hashlib

import numpy as np
import pytest

from experiments.overnight_ab_20261103.state import read_json,write_json
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.vendor_ga31.audited_data import array_hash,split_row_fingerprint
from experiments.ema_mechanism_20261106.protocol import ROOT,_npy_content_hash,resolve_study

MANIFEST=ROOT/'research_plans/26-11-6/experiment_manifest.json'


def test_resolved_matrix_source_identity_and_observer_independent_online_ids(tmp_path):
    study=resolve_study(MANIFEST)
    assert len(study.jobs)==20
    assert sum(j['adam_steps'] for j in study.jobs)==34500
    assert len({j['online_trajectory_id'] for j in study.jobs})==20
    assert study.configs['movielens']['aliases']=={'F':'H2'}
    assert 'F' in study.configs['kuairand']['rho_by_physical_shadow']
    assert study.identity['source_commit']=='40705dfe0bb4fe614bdca835c0bae567348c87d0'
    m=read_json(MANIFEST);m['jobs'][0]['seed']=2027
    altered=tmp_path/'bad_manifest.json';write_json(altered,m)
    with pytest.raises(ValueError,match='coverage'):
        resolve_study(altered)
    m=read_json(MANIFEST);m['datasets']['kuairand']['rho_F']=.99;write_json(altered,m)
    with pytest.raises(ValueError,match='training settings'):
        resolve_study(altered)


def test_streamed_npy_hash_matches_source_array_and_split_identity(tmp_path):
    arrays=(np.arange(24,dtype=np.float32).reshape(8,3),
            np.arange(16,dtype=np.float32).reshape(8,2)%2,np.arange(8,dtype=np.int64))
    combined=hashlib.sha256()
    for index,array in enumerate(arrays):
        path=tmp_path/f'{index}.npy';np.save(path,array,allow_pickle=False)
        actual,shape,dtype=_npy_content_hash(path,combined)
        assert actual==array_hash(array) and shape==array.shape and dtype==array.dtype
    assert combined.hexdigest()==split_row_fingerprint(*arrays)
