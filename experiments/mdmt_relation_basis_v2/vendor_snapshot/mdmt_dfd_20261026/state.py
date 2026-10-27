"""Atomic complete-state transport; no research decisions."""
from __future__ import annotations
import dataclasses, hashlib, json, math, os, random, tempfile
from pathlib import Path
import numpy as np
import torch


def plain(x):
    if dataclasses.is_dataclass(x): return plain(dataclasses.asdict(x))
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,Path): return str(x)
    if hasattr(x,'tolist'): return plain(x.tolist())
    if isinstance(x,float) and not math.isfinite(x): return None
    return x

def digest(x): return hashlib.sha256(json.dumps(plain(x),sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def array_hash(x):
    x=np.ascontiguousarray(x);h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes());return h.hexdigest()
def state_hash(x):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,dict):
            for k in sorted(v,key=str): h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for a in v:visit(a)
        elif torch.is_tensor(v): h.update(array_hash(v.detach().cpu().numpy()).encode())
        elif isinstance(v,np.ndarray): h.update(array_hash(v).encode())
        else:h.update(str(v).encode())
    visit(x);return h.hexdigest()
def atomic_json(path,value,exclusive=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    s=json.dumps(plain(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if exclusive:
        with p.open('x') as f:f.write(s)
        return
    fd,tmp=tempfile.mkstemp(dir=p.parent,prefix=p.name)
    try:
        with os.fdopen(fd,'w') as f:f.write(s)
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
write_json=atomic_json

def append_jsonl(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(plain(value),ensure_ascii=False,allow_nan=False)+'\n')
def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(0) if torch.cuda.is_initialized() else None)
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'])
    if x.get('cuda') is not None:torch.cuda.set_rng_state(x['cuda'],0)
def save_checkpoint(path,payload):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=p.parent,prefix=p.name);os.close(fd)
    try:
        torch.save(payload,tmp);os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def load_checkpoint(path,expected_identity=None):
    x=torch.load(path,map_location='cpu',weights_only=False)
    if expected_identity:
        for k,v in expected_identity.items():
            if x['identity'].get(k)!=v:raise ValueError('Checkpoint identity mismatch: '+k)
    return x

def module_state(model,opt=None):
    import copy
    return dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()) if opt else None,modes={n:m.training for n,m in model.named_modules()},requires_grad={n:p.requires_grad for n,p in model.named_parameters()},gradients={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in model.named_parameters()})
def restore_module(model,opt,payload):
    model.load_state_dict(payload['model'])
    if opt is not None and payload.get('optimizer') is not None:opt.load_state_dict(payload['optimizer'])
    for n,m in model.named_modules():m.train(payload.get('modes',{}).get(n,False))
    for n,p in model.named_parameters():
        p.requires_grad_(payload.get('requires_grad',{}).get(n,True));g=payload.get('gradients',{}).get(n);p.grad=None if g is None else g.to(p.device)
