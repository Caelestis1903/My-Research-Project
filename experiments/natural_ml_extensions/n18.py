"""Original N18 finite-population calculations on the frozen natural ML pool."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from .bootstrap import HERE,ART,activate,population,sources as base_sources,sha,dump
OUT=ART/'N18'

def sources():
    return dict(base_sources('n18'),wrapper_n18=sha(__file__))

def mask_identity(b):
    records=[]
    def digest(values):return hashlib.sha256(np.asarray(values,dtype=np.int64).tobytes()).hexdigest()
    for seed in (2022,2023,2024,2025):
        rng=np.random.default_rng(seed)
        for d in range(3):
            ix=np.flatnonzero(b.train_x[:,-1]==d);y0,y1=b.train_y[ix].T.astype(int);n=len(ix)
            for mechanism in ('MAR','MNAR'):
                prob=np.full(n,.4) if mechanism=='MAR' else np.where(y1==1,.7,.2)
                obs=rng.random(n)<prob;pool=np.flatnonzero(~obs);k=min(128,len(pool))
                random_ix=rng.choice(pool,k,False)
                pospool=np.flatnonzero((~obs)&(y0==1));k2=min(k,len(pospool));target_ix=rng.choice(pospool,k2,False)
                records.append(dict(seed=seed,domain=d,mechanism=mechanism,n=n,observed=int(obs.sum()),
                    observed_row_ids_sha256=digest(b.train_row_ids[ix[obs]]),
                    random_audit_row_ids_sha256=digest(b.train_row_ids[ix[random_ix]]),
                    stratified_audit_row_ids_sha256=digest(b.train_row_ids[ix[target_ix]]),
                    random_budget=k,stratified_budget=k2))
    return records

def execute():
    gate=json.loads((OUT/'review_gate.json').read_text());assert gate['verdict']=='PASS' and gate['sources']==sources()
    b=population();activate('n18')
    import n18_common as c
    import n18_theory_data as old
    c.OUT=OUT;c.bundle=lambda *a,**kw:b;old.OUT=OUT;old.bundle=c.bundle
    source=Path(old.__file__).read_text();target="for dataset in ['kuairand','movielens']:"
    assert source.count(target)==1
    exec(compile(source.replace(target,"for dataset in ['movielens']:"),old.__file__,'exec'),old.__dict__)
    old.OUT=OUT;old.bundle=c.bundle
    argv=sys.argv
    try:
        sys.argv=[old.__file__,'--out',str(OUT/'theory_data')];old.main()
    finally:sys.argv=argv
    folder=OUT/'theory_data';identity=mask_identity(b)
    actual=pd.read_csv(folder/'P2_mask_bounds.csv')
    assert actual[['seed','domain','mechanism','n','observed']].to_dict('records')==[
        {k:r[k] for k in ('seed','domain','mechanism','n','observed')} for r in identity]
    dump(folder/'mask_audit_identities.json',identity)
    dump(folder/'reassessment_receipt.json',dict(status='COMPLETE',training_runs=0,test_access=False,
        population='movielens_natural',population_seed=2022,mask_seeds=[2022,2023,2024,2025],
        four_independent_populations=False,data=b.manifest_fields(),sources=sources(),
        artifacts={p.name:dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in folder.iterdir() if p.is_file()}))

if __name__=='__main__':execute()
