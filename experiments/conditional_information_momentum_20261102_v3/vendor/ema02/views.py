"""Registered input-view dispatch and task-specific backup views."""
import torch
from .rng import normal
from .input_probe import embedding, sparse_fields, task_gradients
from .views_joint_label import joint_delta
from .views_ema_field import field_delta


def build_delta(model, ema, x, y, row_ids, config, context, diagnostic=False):
    arm = config['arm']
    if arm in ('J0', 'J1', 'J_SUM'):
        delta, info, count = joint_delta(model, x, y, row_ids, config, context, diagnostic)
        return [delta], info, count
    if arm in ('F0', 'F1', 'FR', 'F_PR'):
        delta, info, count = field_delta(model, ema, x, y, row_ids, config, context, diagnostic)
        return [delta], info, count
    if arm == 'RG':
        h = sparse_fields(model, embedding(model, x)); gradients = None
    else:
        h, gradients = task_gradients(model, x, y)
    radius = config['kappa']*h.flatten(1).norm(dim=1)
    directions = [None, None] if arm == 'RG' else ([gradients[0]+gradients[1]] if arm == 'AG' else gradients)
    deltas = []
    for t, direction in enumerate(directions):
        if direction is None:
            direction = normal(context, row_ids, tuple(h.shape[1:]), 'RG|task'+str(t), h)
        norm = direction.flatten(1).norm(dim=1)
        zero = norm <= 1e-12
        if bool(zero.any()):
            random = normal(context, row_ids, tuple(h.shape[1:]), arm+'|fallback'+str(t), h)
            direction = torch.where(zero[:, None, None], random, direction)
            norm = direction.flatten(1).norm(dim=1)
        deltas.append((direction*(radius/norm.clamp_min(1e-12))[:, None, None]).detach())
    info = dict(task_direction_cosine_mean=float(torch.nn.functional.cosine_similarity(
        gradients[0].flatten(1), gradients[1].flatten(1), dim=1).mean()) if gradients is not None else None)
    return deltas, info, dict(probe_forwards=int(gradients is not None), input_gradient_calls=2*int(gradients is not None))
