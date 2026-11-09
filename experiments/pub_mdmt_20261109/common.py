"""Small shared IO, identities and deterministic execution helpers."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATASETS = ('movielens', 'kuairand')
SEEDS = tuple(range(2022, 2027))
BATCH = 40960

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def read_json(path): return json.loads(Path(path).read_text())

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')

def append_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f: f.write(json.dumps(value,allow_nan=False,default=str)+'\n')

def state_hash(values):
    h=hashlib.sha256()
    for name,value in values.items():
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def cpu_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def setup(seed,device=None):
    if device is not None and str(device).startswith("cuda"): torch.cuda.set_device(device)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def rng_state():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),(torch.cuda.current_device(),torch.cuda.get_rng_state()) if torch.cuda.is_initialized() else None)

def restore_rng(value):
    random.setstate(value[0]); np.random.set_state(value[1]); torch.set_rng_state(value[2]); 
    if value[3] is not None: torch.cuda.set_rng_state(value[3][1],value[3][0])

def implementation_hash():
    return digest({str(p.relative_to(HERE)):sha_file(p) for p in sorted(HERE.rglob('*')) if p.suffix in ('.py','.json') and 'tests' not in p.parts})
