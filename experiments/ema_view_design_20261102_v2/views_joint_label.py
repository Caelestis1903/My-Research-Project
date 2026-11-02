"""Two same-domain, joint-label donors with a common local radius."""
import hashlib
import numpy as np
import torch
from .rng import uniform
from .input_probe import embedding, sparse_fields, task_gradients


def donor_indices(labels, row_ids, context):
    labels = np.asarray(labels); row_ids = np.asarray(row_ids)
    result = np.full((len(labels), 2), -1, dtype=np.int64)
    draws = uniform(context, row_ids, (2,), 'J|donors')
    for joint in ((0, 0), (0, 1), (1, 0), (1, 1)):
        pool = np.flatnonzero(np.all(labels == joint, axis=1))
        pool = pool[np.argsort(row_ids[pool], kind='stable')]
        m = len(pool)
        if m < 2: continue
        own = np.arange(m)
        first = np.minimum((draws[pool, 0]*(m-1)).astype(np.int64), m-2)
        result[pool, 0] = pool[first+(first >= own)]
        if m > 2:
            second = np.minimum((draws[pool, 1]*(m-2)).astype(np.int64), m-3)
            second += second >= first
            result[pool, 1] = pool[second+(second >= own)]
    return result


def joint_delta(model, x, y, row_ids, config, context, diagnostic=False):
    arm = config['arm']; n = len(y)
    if arm == 'J0':
        h = sparse_fields(model, embedding(model, x)); gradients = None
    else:
        h, gradients = task_gradients(model, x, y)
    donors = donor_indices(y.detach().cpu().numpy(), row_ids, context)
    indices = torch.as_tensor(donors.clip(min=0), device=h.device)
    difference = (h[indices]-h[:, None]).detach()
    norm = difference.flatten(2).norm(dim=2)
    valid = torch.as_tensor(donors >= 0, device=h.device) & (norm > 1e-12)
    minimum = norm.masked_fill(~valid, float('inf')).amin(1)
    radius = torch.minimum(config['kappa']*h.flatten(1).norm(dim=1), .2*minimum)
    radius = torch.where(valid.any(1), radius, torch.zeros_like(radius))
    candidates = difference*(radius[:, None]/norm.clamp_min(1e-12))[:, :, None, None]
    candidates = candidates*valid[:, :, None, None]
    scores = torch.zeros((n, 2), device=h.device, dtype=h.dtype)
    active = torch.zeros(n, dtype=torch.bool, device=h.device)
    if gradients is not None:
        gs = torch.stack([g.flatten(1) for g in gradients], dim=1)
        gn = gs.norm(dim=2); observed = gn > 1e-12; active = observed.any(1)
        direction = difference.flatten(2)/norm.clamp_min(1e-12)[:, :, None]
        responses = torch.einsum('ntf,ncf->nct', gs/gn.clamp_min(1e-12)[:, :, None], direction)
        if arm == 'J_SUM':
            scores = (responses*observed[:, None]).sum(2)/observed.sum(1).clamp_min(1)[:, None]
        else:
            scores = responses.masked_fill(~observed[:, None], float('inf')).amin(2)
        scores = torch.where(active[:, None], scores, torch.zeros_like(scores))
    scores = scores.masked_fill(~valid, -float('inf'))
    best = scores.amax(1, keepdim=True)
    eligible = valid & ((scores == best) if arm != 'J0' else torch.ones_like(valid))
    draw = torch.as_tensor(uniform(context, row_ids, (), 'J|tie'), device=h.device)
    pick = (draw*eligible.sum(1)).long()
    selected = torch.where(eligible[:, 0] & (pick == 0), 0, 1)
    delta = candidates[torch.arange(n, device=h.device), selected].detach()
    selected_scores = scores[torch.arange(n, device=h.device), selected]
    finite = selected_scores.isfinite()
    diagnostics = dict(donor_covered=int(valid.any(1).sum()), double_donor_rows=int(valid.all(1).sum()),
        no_donor_rows=int((~valid.any(1)).sum()), zero_task_gradient_rows=int((~active).sum()) if gradients is not None else 0,
        radius_sum=float(radius.sum()), selected_score_sum=float(selected_scores[finite].sum()),
        score_rows=int(finite.sum()), donor_identity_hash=hashlib.sha256(
            np.where(donors >= 0, np.asarray(row_ids)[donors.clip(min=0)], -1).tobytes()).hexdigest())
    if diagnostic:
        diagnostics.update(candidate_score_means=[float(scores[:, c][scores[:, c].isfinite()].mean())
            if bool(scores[:, c].isfinite().any()) else None for c in range(2)],
            radius_max=float(radius.max()) if n else 0.,
            interpolation_max=float((radius[:, None]/norm.clamp_min(1e-12)*valid).max()) if n else 0.)
    counts = dict(probe_forwards=int(gradients is not None), input_gradient_calls=2*int(gradients is not None))
    return delta, diagnostics, counts
