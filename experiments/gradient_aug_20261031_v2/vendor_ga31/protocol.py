"""Frozen protocol, pure-memory state handling, and lightweight numerical records."""
import copy
import csv
import io
import json
import os
import random
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_PROTOCOL = HERE / 'resolved_protocol.json'
ARMS = ('P0','P1','P2','P3')


def protocol(path=DEFAULT_PROTOCOL):
    p = json.loads(Path(path).read_text())
    assert p['seeds'] == list(range(2022,2027)) and p['scope']['formal_training_runs'] == 40
    assert p['training']['epochs'] == 50 and p['training']['batch_size'] == 40960
    assert all(value is False for value in p['unsupported_flags'].values())
    return p


def setup(seed, device=None):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device is not None and str(device).startswith('cuda'):
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)


def optimizer(model, p):
    o = p['optimizer']
    return torch.optim.Adam(model.parameters(), lr=o['lr'], betas=tuple(o['betas']), eps=o['eps'],
                            weight_decay=o['weight_decay'], amsgrad=False, foreach=False, fused=False)


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def equal(a,b):
    if torch.is_tensor(a): return torch.equal(a.cpu(), b.cpu())
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def rng_state(device=None):
    result = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state())
    if device is not None and str(device).startswith('cuda'):
        result['cuda'] = torch.cuda.get_rng_state(device)
    return result


def restore_rng(state, device=None):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['cpu'])
    if 'cuda' in state: torch.cuda.set_rng_state(state['cuda'], device)


def snapshot(model, opt, loaders, epoch, step, validation, device):
    return dict(model=cpu_copy(model.state_dict()), optimizer=cpu_copy(opt.state_dict()),
                rng=cpu_copy(rng_state(device)), loaders={k:v.get_state().clone() for k,v in loaders.items()},
                epoch=epoch, global_step=step, validation=copy.deepcopy(validation), resume_boundary='epoch')


def restore(model, opt, loaders, state, device, restore_random=True):
    model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
    for name,generator in loaders.items(): generator.set_state(state['loaders'][name])
    if restore_random: restore_rng(state['rng'],device)


def tensor_bytes(value):
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if isinstance(value,dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)): return sum(map(tensor_bytes,value))
    return 0


def send_memory(connection, value):
    buffer = io.BytesIO(); torch.save(value,buffer); connection.send_bytes(buffer.getbuffer())


def receive_memory(connection):
    return torch.load(io.BytesIO(connection.recv_bytes()),map_location='cpu',weights_only=False)


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def append_csv(path, row):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    exists=path.exists()
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if not exists:writer.writeheader()
        writer.writerow(row)


def arm_config(p, arm):
    return next(a for a in p['arms'] if a['id']==arm)
