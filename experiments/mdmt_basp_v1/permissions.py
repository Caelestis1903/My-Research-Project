"""Explicit loss permissions and one projected Adam update."""
from __future__ import annotations
import math
import torch


ALLOWED = dict(rec={'native', 'target', 'trunk', 'gain', 'center', 'reader', 'source'},
               gain={'trunk', 'gain', 'center'}, cluster={'trunk', 'gain', 'center'},
               contrast={'trunk', 'gain', 'center'}, owner={'source', 'owner'},
               anchor={'source', 'owner'}, shadow={'reader'})


def make_optimizer(model, cfg=None):
    cfg = cfg or model.cfg
    categories = model.parameter_categories()
    groups = []
    for kind in ('ordinary', 'center', 'native'):
        pairs = [(name, p) for name, p in model.named_parameters() if p.requires_grad and
                 ('native' if categories[name] == 'native' else 'center' if categories[name] == 'center' else 'ordinary') == kind]
        if pairs:
            lr = cfg['training']['full_native_lr'] if kind == 'native' else cfg['training']['center_lr'] if kind == 'center' else cfg['training']['lr']
            groups.append(dict(params=[p for _, p in pairs], names=[n for n, _ in pairs], lr=lr, kind=kind))
    return torch.optim.Adam(groups, betas=tuple(cfg['training']['betas']), eps=cfg['training']['eps'],
                            weight_decay=cfg['training']['weight_decay'])


def add_new_parameters(model, optimizer, cfg=None):
    cfg = cfg or model.cfg
    known = {id(p) for group in optimizer.param_groups for p in group['params']}
    categories = model.parameter_categories()
    added = []
    for kind in ('ordinary', 'center', 'native'):
        pairs = [(name, p) for name, p in model.named_parameters() if p.requires_grad and id(p) not in known and
                 ('native' if categories[name] == 'native' else 'center' if categories[name] == 'center' else 'ordinary') == kind]
        if not pairs:
            continue
        lr = cfg['training']['full_native_lr'] if kind == 'native' else cfg['training']['center_lr'] if kind == 'center' else cfg['training']['lr']
        optimizer.add_param_group(dict(params=[p for _, p in pairs], names=[n for n, _ in pairs], lr=lr, kind=kind))
        added.extend(name for name, _ in pairs)
    return added


def loss_weights(cfg, epoch):
    weights = dict(cfg['loss'])
    if epoch == 1:
        return dict(rec=1., shadow=weights['shadow'])
    if epoch == 2:
        return dict(rec=1., shadow=weights['shadow'], gain=cfg['training']['gain_warmup_weight'])
    return {key: float(weights[key]) for key in ALLOWED}


def apply_gradients(losses, model, optimizer, epoch, cfg=None):
    """Each loss sees only its whitelist; inactive parameters retain grad=None."""
    cfg = cfg or model.cfg
    weights = loss_weights(cfg, epoch)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    categories = model.parameter_categories()
    rec_grads, aux_grads, other_grads = {}, {}, {}
    permissions = {}
    optimizer.zero_grad(set_to_none=True)
    for lossname, weight in weights.items():
        value = losses.get(lossname)
        if value is None or not value.requires_grad or not weight:
            continue
        selected = [(n, p) for n, p in named if categories[n] in ALLOWED[lossname]]
        # Warmup freezes source successors, owner routing and benefit centers.
        if epoch < 3:
            selected = [(n, p) for n, p in selected if categories[n] not in {'source', 'owner', 'center'}]
        if not selected:
            continue
        grads = torch.autograd.grad(value*weight, [p for _, p in selected], retain_graph=True, allow_unused=True)
        active = []
        for (name, p), gradient in zip(selected, grads):
            if gradient is None:
                continue
            if not torch.isfinite(gradient).all():
                raise FloatingPointError('Nonfinite gradient in '+lossname+'/'+name)
            active.append(name)
            bucket = rec_grads if lossname == 'rec' else aux_grads if lossname in ('gain', 'cluster', 'contrast') else other_grads
            bucket[name] = gradient if name not in bucket else bucket[name]+gradient
        permissions[lossname] = active
    trunk_rec = sum((gradient.double().square().sum() for n, gradient in rec_grads.items() if categories[n] == 'trunk'), torch.tensor(0., device=next(model.parameters()).device)).sqrt()
    trunk_aux = sum((gradient.double().square().sum() for n, gradient in aux_grads.items() if categories[n] == 'trunk'), torch.tensor(0., device=next(model.parameters()).device)).sqrt()
    gamma = 1. if epoch == 2 else min(1., cfg['loss']['shared_trunk_aux_ratio_cap'] *
                    max(float(trunk_rec), cfg['loss']['shared_trunk_main_norm_floor'])/max(float(trunk_aux), 1e-12))
    for name, p in named:
        pieces = []
        if name in rec_grads:
            pieces.append(rec_grads[name])
        if name in aux_grads:
            pieces.append(aux_grads[name]*(gamma if categories[name] == 'trunk' else 1.))
        if name in other_grads:
            pieces.append(other_grads[name])
        p.grad = sum(pieces[1:], pieces[0]) if pieces else None
    optimizer.step()
    model.project_readers()
    return dict(aux_gamma=gamma, trunk_rec_norm=float(trunk_rec), trunk_aux_norm=float(trunk_aux),
                active_permissions=permissions,
                effective_write=bool(getattr(model, 'foreign', False) and getattr(model, 'flags', {}).get('SOURCE_WRITE', False) and epoch >= 3))
