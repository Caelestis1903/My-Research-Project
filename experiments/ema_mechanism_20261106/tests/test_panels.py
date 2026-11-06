"""Panel row isolation, fixed pair identity and observation persistence."""
import copy
import hashlib
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.overnight_ab_20261103.state import write_json
from experiments.ema_mechanism_20261106.panels import build_panels, load_panels, row_side, save_panels


def panel_fixture():
    domains=np.repeat(np.arange(3),240)
    y=np.stack((np.arange(len(domains)) % 2,(np.arange(len(domains))//2) % 2),axis=1).astype(np.float32)
    bundle=SimpleNamespace(dataset='movielens',
        train_x=np.column_stack((np.arange(len(domains)),domains)).astype(np.float32),
        train_y=y,train_row_ids=np.arange(len(domains),dtype=np.int64)+10000,
        val_x=np.column_stack((np.arange(len(domains)),domains)).astype(np.float32),
        val_y=y.copy(),val_row_ids=np.arange(len(domains),dtype=np.int64)+20000)
    spec=dict(rank_positive_per_cell=16,rank_negative_per_cell=16,rank_pairs_per_cell=64,
        natural_rows_per_train_domain=20,validation_natural_rows_per_domain=40,
        validation_pairs_per_cell_for_temporal_stat=64)
    return bundle,spec


def test_panels_are_disjoint_fixed_and_preserve_training_rng():
    bundle,spec=panel_fixture()
    python=random.getstate(); numpy=np.random.get_state(); cpu=torch.get_rng_state().clone()
    panels=build_panels(bundle,2022,spec)
    assert random.getstate()==python
    actual=np.random.get_state()
    assert actual[0]==numpy[0] and np.array_equal(actual[1],numpy[1]) and actual[2:]==numpy[2:]
    assert torch.equal(cpu,torch.get_rng_state())
    again=build_panels(bundle,2022,spec)
    for name,panel in panels.items():
        assert panel['panel_hash']==again[name]['panel_hash']
    raw=[np.concatenate((panels[name]['rank_row_ids'].ravel(),panels[name]['natural_row_ids'].ravel()))
         for name in ('P_A','P_B')]
    assert not np.intersect1d(*raw).size
    assert np.isin(np.concatenate(raw),bundle.train_row_ids).all()
    assert not np.isin(np.concatenate(raw),bundle.val_row_ids).any()
    for name in ('P_A','P_B'):
        panel=panels[name]
        for cell in range(6):
            pairs=panel['rank_pairs'][cell]
            assert np.unique(pairs,axis=0).shape[0]==64
            assert (panel['rank_y'][cell,pairs[:,0]]==1).all()
            assert (panel['rank_y'][cell,pairs[:,1]]==0).all()
    validation=panels['V_mech']
    assert np.isin(validation['row_ids'],bundle.val_row_ids).all()
    expected=np.array([hashlib.sha256(f'movielens|{row}'.encode()).digest()[-1]&1
                       for row in validation['row_ids'].ravel()]).reshape(3,40)
    assert np.array_equal(validation['validation_row_side'],expected)
    assert np.array_equal(row_side('movielens',bundle.val_row_ids),
                          row_side('movielens',bundle.val_row_ids.copy()))


def test_panel_identity_roundtrip_conflict_and_validation_missing_class(tmp_path):
    bundle,spec=panel_fixture(); panels=build_panels(bundle,2022,spec)
    receipts=save_panels(tmp_path,'movielens',2022,panels)
    assert save_panels(tmp_path,'movielens',2022,panels)==receipts
    write_json(tmp_path/'PROBE_MANIFEST.json',dict(panels={'movielens':{'2022':receipts}}))
    restored=load_panels(tmp_path,'movielens',2022,bundle)
    for name in panels:
        assert np.array_equal(restored[name]['natural_x'],panels[name]['natural_x'])
        assert restored[name]['panel_hash']==panels[name]['panel_hash']
    changed=copy.deepcopy(panels);changed['P_A']['rank_y'][0,0]=0
    with pytest.raises(ValueError,match='conflicts'):
        save_panels(tmp_path,'movielens',2022,changed)
    bundle.val_y[:]=0
    missing=build_panels(bundle,2022,spec)['V_mech']
    assert (missing['pair_counts']==0).all() and (missing['pairs']==-1).all()
    bundle.train_y[:]=0
    with pytest.raises(ValueError,match='PANEL_POPULATION_MISMATCH'):
        build_panels(bundle,2022,spec)
