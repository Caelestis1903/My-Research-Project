"""CPU-only source and loader audit; invoke before experiment launch."""
import sys,json,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from stage_S1_data import load_stage_s1_bundle,DEFAULT_MOVIELENS_PATH,DEFAULT_KUAIRAND_PATH
import numpy as np
DS=sys.argv[1]; out=Path(__file__).resolve().parent
path=Path(DEFAULT_MOVIELENS_PATH if DS=='movielens' else DEFAULT_KUAIRAND_PATH)/('ml-1m.csv' if DS=='movielens' else 'kuai_rand.csv')
assert path.resolve().is_relative_to(Path('/home/xzf/data')) and path.is_file() and os.access(path,os.R_OK)
h=hashlib.sha256()
with path.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
b=load_stage_s1_bundle(DS,None,'d0_large' if DS=='movielens' else 'natural',2022)
report=dict(dataset=DS,resolved_path=str(path.resolve()),size=path.stat().st_size,sha256=h.hexdigest(),seed=2022,setting=b.setting,split_row_id_hashes=b.split_row_id_hashes,base_split_fingerprints=b.base_split_fingerprints,preprocessing=b.preprocessing_metadata,features=b.features.as_dict(),counts={})
for split,x,y,ids in [('train',b.train_x,b.train_y,b.train_row_ids),('validation',b.val_x,b.val_y,b.val_row_ids),('test',b.test_x,b.test_y,b.test_row_ids)]:
 assert len(x)==len(y)==len(ids) and np.isfinite(x).all() and np.isfinite(y).all()
 report['counts'][split]=[dict(domain=d,task=t,n=int((x[:,-1]==d).sum()),n_pos=int(y[x[:,-1]==d,t].sum())) for d in range(3) for t in range(2)]
assert not np.intersect1d(b.train_row_ids,b.val_row_ids).size
assert not np.intersect1d(b.train_row_ids,b.test_row_ids).size
assert not np.intersect1d(b.val_row_ids,b.test_row_ids).size
report['status']='PASS_DATA_PATH_AND_LOADER_ONLY'
(out/f'dataset_preflight_{DS}.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(DS,report['status'],report['resolved_path'],report['sha256'],flush=True)
