"""N17-only dependencies and historical validation split injection."""
from pathlib import Path
import importlib
import json
import numpy as np
from .bootstrap import (HERE,ART,activate,population,dependency_inventory,
                        actual_module_paths,sha,dump)
OUT=ART/'N17'
LOCAL=('n17_bootstrap.py','n17_train.py','n17_manifest.py','n17_diagnostics.py',
       'n17_readout.py','n17_basiccheck.py')

def sources():
    records=dependency_inventory('n17')
    result={key:record['sha256'] for key,record in records.items()}
    import hashlib
    result['dependency_inventory_sha256']=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for name in LOCAL:
        path=HERE/name
        if not path.is_file():raise RuntimeError('N17 dependency not delivered: '+name)
        result['extensions/'+name]=sha(path)
    return result

def prepare(population_name='movielens_natural'):
    b=population(population_name)
    root=activate('n17')
    import n17_common as c
    import v3_audit_20261016 as audit
    # Original row-ID hash/joint-label stratification, held identical across populations.
    audit.OUT=OUT/'history';audit.OUT.mkdir(parents=True,exist_ok=True)
    audit.frozen_split(b,'movielens',2022)
    with np.load(audit.OUT/'splits/movielens_seed2022.npz') as z:
        split={key:z[key] for key in z.files}
    c.HISTORY=audit.OUT;c.OUT=OUT;c.sources=sources
    c.bundle=lambda *args,**kwargs:b
    c.split=lambda *args,**kwargs:split
    c.base.sources=sources
    actual_module_paths('n17')
    return b,split,c,root

def task_path(group,arm,lr,seed):
    return OUT/group/f'lr{lr:g}'/arm/f'seed{seed}'
