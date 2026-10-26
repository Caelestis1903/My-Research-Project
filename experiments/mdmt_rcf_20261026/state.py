"""Identity utilities and RAM-only complete training snapshots."""
from __future__ import annotations
import copy, random
import numpy as np
import torch
from .vendor_snapshot.mdmt_dfd_20261026.state import (plain,digest,sha_file,array_hash,state_hash,atomic_json,append_jsonl)
write_json=atomic_json

def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),cuda=[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state.get('cuda'):torch.cuda.set_rng_state_all(state['cuda'])

def capture_memory_state(objects:dict,rng_state:dict|None=None,positions:dict|None=None)->dict:
    values={k:copy.deepcopy(v.state_dict() if hasattr(v,'state_dict') else v) for k,v in objects.items()}
    return dict(objects=values,rng=copy.deepcopy(rng_state if rng_state is not None else capture_rng()),positions=copy.deepcopy(positions or {}),storage='RAM_ONLY')

def restore_memory_state(snapshot:dict,objects:dict)->dict:
    for k,v in objects.items():
        if hasattr(v,'load_state_dict'):v.load_state_dict(copy.deepcopy(snapshot['objects'][k]))
        elif isinstance(v,dict):v.clear();v.update(copy.deepcopy(snapshot['objects'][k]))
        else:raise TypeError('RAM restore requires state_dict object or mutable dict: '+k)
    restore_rng(snapshot['rng']);return copy.deepcopy(snapshot['positions'])

def cpu_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}

def freeze(module):
    module.eval();module.requires_grad_(False)
    for p in module.parameters():p.grad=None
    return module
