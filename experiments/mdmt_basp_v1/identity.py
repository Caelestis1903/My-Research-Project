"""Explicit run coordinates, direct parity checks, and complete RAM snapshots."""
from pathlib import Path
import copy
import json
import os
import random
import dataclasses
import numpy as np
import torch


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        raise TypeError('Model/state tensors must remain in RAM; export measurements explicitly')
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(plain(value), ensure_ascii=False, allow_nan=False)+'\n')


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, torch.nn.Module):
        return copy.deepcopy(value).cpu()
    if dataclasses.is_dataclass(value):
        result=type(value)(**{f.name:cpu_copy(getattr(value,f.name)) for f in dataclasses.fields(value)})
        fields={f.name for f in dataclasses.fields(value)}
        for key,item in value.__dict__.items():
            if key not in fields:
                setattr(result,key,cpu_copy(item))
        return result
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
                cuda=[v.clone() for v in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else [])


def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    if value['cuda']:
        torch.cuda.set_rng_state_all(value['cuda'])


def capture(model, optimizer, identity, state):
    optimizer_state=cpu_copy(optimizer.state_dict())
    names={id(p):n for n,p in model.named_parameters()}
    for saved,current in zip(optimizer_state['param_groups'],optimizer.param_groups):
        saved['names']=[names[id(p)] for p in current['params']]
    return dict(model=copy.deepcopy(model).cpu(), optimizer=optimizer_state,
                requires_grad={n:p.requires_grad for n,p in model.named_parameters()},
                training=model.training, rng=capture_rng(), identity=copy.deepcopy(identity),
                state=cpu_copy(state), storage='RAM_ONLY')


def restore(snapshot, device, optimizer_factory):
    model = copy.deepcopy(snapshot['model']).to(device)
    model.train(snapshot['training'])
    for name, p in model.named_parameters():
        p.requires_grad_(snapshot['requires_grad'][name])
    optimizer = optimizer_factory(model)
    named=dict(model.named_parameters())
    optimizer.param_groups.clear()
    optimizer.state.clear()
    for saved in snapshot['optimizer']['param_groups']:
        group={k:copy.deepcopy(v) for k,v in saved.items() if k!='params'}
        group['params']=[named[name] for name in saved['names']]
        optimizer.add_param_group(group)
    optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']))
    restore_rng(snapshot['rng'])
    return model, optimizer, cpu_copy(snapshot['state'])


def directly_equal(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(directly_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(directly_equal(a,b) for a,b in zip(left,right))
    return left == right


def freeze(module):
    module.eval().requires_grad_(False)
    for p in module.parameters():
        p.grad = None
    return module


def configure():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
