"""Stateless row/role keyed SplitMix64 uniforms and Box-Muller normals."""
import hashlib
import numpy as np
import torch


def uniform(context, row_ids, shape, role):
    key = '|'.join(str(context[k]) for k in
        ('study', 'dataset', 'split_id', 'seed', 'epoch', 'global_step'))+'|'+role
    seed = np.uint64(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))
    width = int(np.prod(shape)) if shape else 1
    rows = np.asarray(row_ids, dtype=np.uint64)[:, None]
    coord = np.arange(width, dtype=np.uint64)[None, :]
    with np.errstate(over='ignore'):
        z = seed + rows*np.uint64(0xD2B74407B1CE6E93) + coord*np.uint64(0x9E3779B97F4A7C15)
        z += np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    return ((z >> np.uint64(11)).astype(np.float64)+.5).reshape((len(rows),)+tuple(shape))/(1 << 53)


def normal(context, row_ids, shape, role, reference):
    a = uniform(context, row_ids, shape, role+'|radius')
    b = uniform(context, row_ids, shape, role+'|angle')
    return torch.as_tensor(np.sqrt(-2*np.log(a))*np.cos(2*np.pi*b),
        dtype=reference.dtype, device=reference.device)


def signs(context, row_ids, shape, role, reference):
    return torch.as_tensor(np.where(uniform(context, row_ids, shape, role)<.5, -1., 1.),
        dtype=reference.dtype, device=reference.device)
