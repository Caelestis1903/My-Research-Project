"""RAM snapshots and scalar-only atomic persistence."""
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
from experiments.mdmt_history_information_20261103.protocol import cpu_copy, state_digest

def capture_rng(device=None):
    active=None
    if torch.cuda.is_initialized() and (device is None or torch.device(device).type=='cuda'):
        active=torch.cuda.current_device() if device is None else torch.device(device).index
        if active is None: active=torch.cuda.current_device()
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if active is not None else None,
                active_cuda_device=active,
                active_cuda_state=torch.cuda.get_rng_state(active) if active is not None else None)

def restore_rng(value,device=None):
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'])
    if value.get('cuda') is not None:
        torch.cuda.set_rng_state_all(value['cuda'])
        if device is None or torch.device(device).type=='cuda':
            target=torch.cuda.current_device() if device is None else torch.device(device).index
            if target is None: target=torch.cuda.current_device()
            torch.cuda.set_rng_state(value['active_cuda_state'],target)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload=json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        temporary=stream.name; stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)

def append_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()

def tensor_bytes(value):
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if isinstance(value,np.ndarray): return value.nbytes
    if isinstance(value,dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)): return sum(tensor_bytes(v) for v in value)
    return 0

