"""Focused engineering equivalence and timing; no model or research outcomes."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.ema_causal_20261110.pairs import _cached_pairs, sample_pairs


def _original(y, seed, limit=32768):
    y=np.asarray(y);positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
    n=len(positive)*len(negative);k=min(limit,n)
    if not k:return np.empty((0,2),dtype=np.int64)
    rng=np.random.default_rng(seed)
    selected=set()
    for upper in range(n-k,n):
        value=int(rng.integers(upper+1));selected.add(upper if value in selected else value)
    ids=np.fromiter(sorted(selected),dtype=np.int64)
    return np.stack((positive[ids//len(negative)],negative[ids%len(negative)]),1)


def test_pair_cache_equivalence_and_timing():
    root=Path(__file__).resolve().parents[3]
    panel_root=root/'artifacts/ema_causal_20261110/engineering/panels/movielens/seed2022'
    panels={}
    for name in ('P_A','P_B','V_mech'):
        with np.load(panel_root/f'{name}.npz') as data:
            panels[name]={k:data[k] for k in ('rank_y','natural_y','valid_mask','validation_row_side') if k in data}
            if name=='V_mech':
                panels[name].update(natural_y=data['y'],valid_mask=data['indices']>=0)

    def request(population,domain,task):
        panel=panels[population if population in ('P_A','P_B') else 'V_mech']
        if population in ('P_A','P_B'):
            y=panel['rank_y'][2*domain+task]
        else:
            mask=panel['valid_mask'][domain].copy()
            if population in ('V_A','V_B'):mask &= panel['validation_row_side'][domain]==int(population=='V_B')
            y=panel['natural_y'][domain,mask,task]
        seed=int.from_bytes(hashlib.sha256(f'movielens|2022|{population}|{domain}|{task}'.encode()).digest()[:8],'little')
        return y,seed,32768

    actual=request('P_A',0,0);y,seed,limit=actual
    assert (y==1).sum()*(y==0).sum()>limit
    reordered=y[::-1].copy()
    if np.array_equal(y,reordered):reordered=np.roll(y,1)
    assert not np.array_equal(y,reordered)
    cases=[actual,(reordered,seed,limit),(y,seed+1,limit),(y,seed,31),
           (np.array([0,1,0,1]),seed,limit),(np.zeros(4),seed,limit)]
    _cached_pairs.cache_clear()
    for index,args in enumerate(cases):
        expected=_original(*args)
        cold=sample_pairs(*args);warm=sample_pairs(*args)
        assert cold is warm and not cold.flags.writeable
        assert cold.dtype==warm.dtype==expected.dtype==np.int64
        assert np.array_equal(cold,expected) and cold.tobytes()==expected.tobytes()
        assert _cached_pairs.cache_info().misses==index+1
        assert _cached_pairs.cache_info().hits==index+1
    # Both admission limits bypass the LRU; eviction retains exactly 128 entries.
    before=_cached_pairs.cache_info()
    for args in ((np.tile([0,1],2049),seed,2),(y,seed,32769)):
        assert np.array_equal(sample_pairs(*args),_original(*args))
    assert _cached_pairs.cache_info()==before
    _cached_pairs.cache_clear()
    for s in range(129):sample_pairs([0,1],s,1)
    assert _cached_pairs.cache_info().currsize==128
    sample_pairs([0,1],0,1)
    assert _cached_pairs.cache_info().misses==130

    def timed(function,requests):
        start=perf_counter()
        result=[function(*args) for args in requests]
        return perf_counter()-start,result

    _cached_pairs.cache_clear()
    cold_single,_=timed(sample_pairs,[actual]);warm_single,_=timed(sample_pairs,[actual])
    single_info=_cached_pairs.cache_info()._asdict()
    # Match panel_records: one population request and one discovery request per cell.
    sequence=[]
    for population in ('P_A','P_B','V_mech','V_A','V_B'):
        for domain in range(3):
            for task in range(2):
                sequence.extend((request(population,domain,task),
                                 request('P_A' if population in ('P_A','P_B') else 'V_A',domain,task)))
    assert len(sequence)==60
    original_seconds,reference=timed(_original,sequence)
    _cached_pairs.cache_clear()
    cold_seconds,cold=timed(sample_pairs,sequence);cold_info=_cached_pairs.cache_info()._asdict()
    warm_seconds,warm=timed(sample_pairs,sequence);warm_info=_cached_pairs.cache_info()._asdict()
    assert cold_info==dict(hits=30,misses=30,maxsize=128,currsize=30)
    assert warm_info==dict(hits=90,misses=30,maxsize=128,currsize=30)
    assert all(np.array_equal(a,b) and np.array_equal(a,c) and not b.flags.writeable
               for a,b,c in zip(reference,cold,warm))
    print(json.dumps(dict(panel=str(panel_root/'P_A.npz'),pair_count=len(cold[0]),
        single_cold_seconds=cold_single,single_warm_seconds=warm_single,single_cache=single_info,
        original_60_seconds=original_seconds,cold_60_seconds=cold_seconds,warm_60_seconds=warm_seconds,
        cold_cache=cold_info,warm_cache=warm_info,max_pair_bytes=128*32768*2*8,
        max_index_key_bytes=128*4096*8),sort_keys=True))
    _cached_pairs.cache_clear()
