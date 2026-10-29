"""N18 frozen-checkpoint access; no historical file is mutated."""
from pathlib import Path
import json
import os
import pickle
import time
import numpy as np
import torch
from stage_S1_models import build_model

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get('N18_OUTPUT_ROOT',ROOT / 'research_plans/2026-10-18/execution'))
HISTORY = Path(os.environ.get('N18_HISTORY_ROOT','/home/xzf/Code/phase0-3-20261016-v3-git/research_plans/2026-10-16_v3/execution_20261016'))

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))

def bundle(dataset='kuairand'):
    with (HISTORY / 'local_cache' / f'{dataset}_2022.pkl').open('rb') as f:
        return pickle.load(f)

class ConstantFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('value', torch.sigmoid(torch.ones(1)))
    def forward(self):
        return self.value

def model_and_state(b, seed, device):
    torch.manual_seed(seed)
    model = build_model(b.features, 3, 2, 'm3oe', seed=seed)
    for key in ('_weight_exp_d','_weight_exp_t','_weight_bal_d','_weight_bal_t'):
        setattr(model, key, ConstantFusion())
    path = HISTORY / 'runs' / f'V4final__kuairand__adam__SN__seed{seed}' / 'epoch010.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    assert state['epoch'] == 10
    assert not any(n.startswith('_weight_') for n,p in model.named_parameters())
    return model.to(device).eval(), state, path

def flat_grad(loss, model, retain_graph=False):
    params = list(model.parameters())
    grad = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain_graph)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for p,g in zip(params,grad)])

def stratified(x, n, rng, pools=None):
    pools = pools or [np.flatnonzero(x[:,-1] == d) for d in range(3)]
    return np.concatenate([rng.choice(pool, n, replace=False) for pool in pools])

def validation_pools(b):
    rng = np.random.default_rng(181018)
    sel, diag = [], []
    for d in range(3):
        ix = rng.permutation(np.flatnonzero(b.val_x[:,-1] == d))
        half = len(ix)//2
        sel.append(ix[:half]); diag.append(ix[half:])
    return sel, diag

def timer(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)
    return time.perf_counter()
