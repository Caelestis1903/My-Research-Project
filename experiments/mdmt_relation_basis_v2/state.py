"""Research identities, JSON data and complete RAM-only training state."""
from __future__ import annotations
import copy
import random
from pathlib import Path
import numpy as np
import torch
from .vendor_snapshot.mdmt_dfd_20261026.state import (
    plain, digest, sha_file, array_hash, state_hash, atomic_json, append_jsonl)

write_json = atomic_json


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=[x.clone() for x in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def capture(model, optimizer, identity, **position):
    return dict(model=cpu_copy(model.state_dict()),
                optimizer=cpu_copy(optimizer.state_dict()), scheduler=None,
                training=model.training,
                requires_grad={name:p.requires_grad for name, p in model.named_parameters()},
                rng=capture_rng(), identity=copy.deepcopy(identity),
                position=copy.deepcopy(position), storage='RAM_ONLY')


def restore(snapshot, model, optimizer, identity):
    if snapshot['identity'] != identity:
        raise ValueError('RAM checkpoint identity mismatch')
    if snapshot['storage'] != 'RAM_ONLY' or snapshot['scheduler'] is not None:
        raise ValueError('Unexpected state protocol')
    model.load_state_dict(snapshot['model'])
    model.train(snapshot['training'])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(snapshot['requires_grad'][name])
    optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']))
    restore_rng(snapshot['rng'])
    return copy.deepcopy(snapshot['position'])


def freeze(module):
    module.eval().requires_grad_(False)
    for parameter in module.parameters():
        parameter.grad = None
    return module


def source_identity():
    root = Path(__file__).resolve().parent
    files = {str(p.relative_to(root)): sha_file(p)
             for p in sorted(root.rglob('*.py'))}
    return dict(base_main_commit='8eb461f5e0abc9c06385a4e8239e9621cae87ef8',
                files=files, source_hash=digest(files))


def configure():
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)

