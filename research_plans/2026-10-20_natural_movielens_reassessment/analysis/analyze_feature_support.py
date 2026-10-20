"""Frozen-population user/movie embedding coverage, without fitting or labels."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

DATA=Path('/home_new/xzf_natural_movielens_reassessment/data')
OUT=Path(__file__).resolve().parent/'feature_support';OUT.mkdir(exist_ok=True)
rows=[]
for population in ['movielens_natural','movielens_artificial_d2_fixed2022','movielens_natural_matched_n187500']:
    meta_path=DATA/(population+'.json');meta=json.loads(meta_path.read_text());columns=meta['schema']['columns']
    with np.load(DATA/(population+'.npz')) as z:
        train=z['train_x'];train_domain=train[:,-1].astype(int)
        for feature in ['user_id','movie_id']:
            index=columns.index(feature);any_codes=np.unique(train[:,index])
            for d in range(3):
                domain_codes=np.unique(train[train_domain==d,index])
                for split in ['val','test']:
                    heldout=z[split+'_x'];heldout=heldout[heldout[:,-1]==d,index]
                    seen_any=np.isin(heldout,any_codes)&(heldout!=0)
                    seen_domain=np.isin(heldout,domain_codes)&(heldout!=0)
                    rows.append(dict(population=population,feature=feature,domain=d,split=split,
                        train_domain_rows=int((train_domain==d).sum()),unique_codes_any_train=int((any_codes!=0).sum()),
                        unique_codes_same_domain_train=int((domain_codes!=0).sum()),heldout_rows=len(heldout),
                        base_encoder_OOV_rows=int((heldout==0).sum()),base_encoder_OOV_fraction=float((heldout==0).mean()),
                        seen_code_any_train_fraction=float(seen_any.mean()),seen_code_same_domain_fraction=float(seen_domain.mean()),
                        semantics='code0 is OOV, excluded from coverage; coverage is descriptive, not a causal effect on AUC',
                        manifest_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest()))
f=pd.DataFrame(rows);f.to_csv(OUT/'frozen_user_movie_coverage.csv',index=False)
print(f[f.split=='test'][['population','feature','domain','base_encoder_OOV_fraction',
    'seen_code_any_train_fraction','seen_code_same_domain_fraction']].to_string(index=False))
