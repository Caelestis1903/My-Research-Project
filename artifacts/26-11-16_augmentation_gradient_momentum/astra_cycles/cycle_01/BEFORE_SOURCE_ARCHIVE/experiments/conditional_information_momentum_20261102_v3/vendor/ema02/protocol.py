"""Frozen study configuration; training states remain in RAM."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from .vendor.gradient_aug_20261031_v2.vendor_ga31.protocol import (
    setup, cpu_copy, equal, rng_state, restore_rng, tensor_bytes,
    send_memory, receive_memory, write_json, append_csv)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_protocol():
    p = yaml.safe_load((ROOT/'research_plans/26-11-2_v2/experiment_queue.yaml').read_text())
    source = json.loads((HERE/'vendor/gradient_aug_20261031_v2/vendor_ga31/resolved_protocol.json').read_text())
    for name, conf in p['datasets'].items():
        conf['source'] = source['datasets'][name]
    p['study_id'] = 'EMA02V2'
    p['vendor_source_hash'] = hashlib.sha256((HERE/'vendor/SOURCE_MANIFEST.json').read_bytes()).hexdigest()
    return p


def config(arm, dataset, *, kappa=None):
    p = load_protocol()
    if arm not in p['methods'] or dataset not in p['datasets']:
        raise ValueError((arm, dataset))
    d = p['datasets'][dataset]
    if kappa is not None and kappa not in (d['kappa'], d['kappa_half']):
        raise ValueError('Only registered default or half kappa is allowed')
    return dict(p['methods'][arm], arm=arm, dataset=dataset,
        lambda_aug=.5, kappa=d['kappa'] if kappa is None else kappa,
        target_domain=d['target_domain'], rho=d['rho'], lr=.01,
        weight_decay=d['weight_decay'], epochs=50, batch_size=40960)


def make_optimizer(model, config):
    return torch.optim.Adam(model.parameters(), lr=config['lr'],
        weight_decay=config['weight_decay'], betas=(.9, .999), eps=1e-8,
        foreach=False, fused=False, amsgrad=False)


@torch.no_grad()
def update_ema(ema, model, rho):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.mul_(rho).add_(source, alpha=1-rho)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def selected_update(selected, best, kind, epoch, value):
    if epoch >= 1 and value > best[kind]:
        selected[kind] = epoch; best[kind] = value
        return True
    return False


def state_digest(value):
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(v.tobytes())
        elif isinstance(v, dict):
            for k, item in v.items():
                h.update(str(k).encode()); visit(item)
        elif isinstance(v, (list, tuple)):
            for item in v: visit(item)
        else:
            h.update(str(v).encode())
    visit(value)
    return h.hexdigest()
