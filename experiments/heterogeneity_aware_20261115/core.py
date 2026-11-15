"""First-order embedding views on the unchanged fixed1 training graph."""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from experiments.conditional_information_momentum_20261102_v3.protocol import update_ema


def rng(context, stream):
    key = '|'.join(str(context[k]) for k in ('dataset', 'seed', 'epoch', 'global_step')) + '|' + stream
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))


def forward(model, x, delta=None, probe=False):
    """Return probabilities, domain-selected logits, and the inserted embedding."""
    captured = {}; handles = []
    def embedding_hook(module, args, out):
        value = out if delta is None else out + delta.detach()
        captured['embedding'] = value
        return value
    handles.append(model.embedding.register_forward_hook(embedding_hook))
    for i, tower in enumerate(model.tower):
        handles.append(tower.register_forward_hook(
            lambda module, args, out, i=i: captured.__setitem__(i, out.squeeze(1))))
    if probe:
        handles.append(model.skip_conn.register_forward_hook(
            lambda module, args, out: captured.__setitem__('skip', out)))
        handles.append(model.star_mlp.register_forward_hook(
            lambda module, args, out: captured.__setitem__('star', out)))
        # The source detach masks the numerical function's gate-input derivative.
        for gate in model.gate:
            handles.append(gate.register_forward_pre_hook(
                lambda module, args: (captured['star'] + captured['skip'],)))
    try:
        probabilities = model(x)
        logits = torch.stack([sum(torch.where(x[-1] == d, captured[t*3+d],
                              torch.zeros_like(captured[t*3+d])) for d in range(3))
                              for t in range(2)], dim=1)
        return probabilities, logits, captured['embedding']
    finally:
        for handle in handles:
            handle.remove()


def sample(domain, labels, context, example=False):
    domain = np.asarray(domain); labels = np.asarray(labels)
    assert labels.shape == (len(domain), 2) and np.isin(labels, (0, 1)).all()
    gen = rng(context, 'examples' if example else 'pairs')
    rows = []; tasks = []; pair_cells = []
    for d in range(3):
        available = np.flatnonzero(domain == d)
        for t in range(2):
            pos = available[labels[available, t] == 1]
            neg = available[labels[available, t] == 0]
            k = min(128, len(pos), len(neg))
            if not k:
                continue
            selected = (gen.choice(available, 2*k, replace=False) if example else
                        np.stack((gen.choice(pos, k, replace=False),
                                  gen.choice(neg, k, replace=False)), axis=1).reshape(-1))
            rows.extend(selected.tolist()); tasks.extend([t] * (2*k))
            pair_cells.extend([(d, t)] * k)
    return np.asarray(rows, dtype=np.int64), np.asarray(tasks, dtype=np.int64), pair_cells


def project(g, h):
    gd, hd = g.double(), h.double()
    hh = hd.square().sum(1, keepdim=True)
    q = gd - torch.where(hh > 1e-20, (hd*gd).sum(1, keepdim=True)/hh.clamp_min(1e-20), 0.)*hd
    return q.to(g.dtype)


def sensitivity(model, x):
    with torch.enable_grad():
        _, z, embedding = forward(model, x, probe=True)
        gradients = [torch.autograd.grad(z[:, t].sum(), embedding, retain_graph=t == 0,
                                        create_graph=False)[0].detach() for t in range(2)]
    return embedding.detach(), z.detach(), torch.stack(gradients, dim=1)


def pair_deltas(embedding, z, g, h, alpha, projected):
    q = project(g, h) if projected else g
    norm = q.norm(dim=1); rms = embedding.square().mean(1).sqrt()
    sign = torch.ones(len(g), device=g.device, dtype=g.dtype); sign[::2] = -1
    delta = sign[:, None] * alpha*rms[:, None]*q/(norm[:, None]+1e-12)
    margin = z[::2]-z[1::2]
    product = (g*delta).sum(1)
    change = product[::2]-product[1::2]
    valid = ((norm > 1e-10) & (rms > 1e-12)).reshape(-1, 2).all(1)
    valid &= (margin.abs() > 1e-7) & (change.abs() > 1e-12)
    cap = torch.minimum(torch.ones_like(margin), margin.abs()/(change.abs()+1e-12))
    delta = delta*cap.repeat_interleave(2)[:, None]
    residual = (h.double()*q.double()).sum(1).abs()/(h.double().norm(dim=1)*q.double().norm(dim=1)+1e-30)
    return delta.detach(), valid, dict(q_norm_mean=float(norm.mean()), g_norm_mean=float(g.norm(dim=1).mean()),
        projection_residual_max=float(residual.max()) if projected else 0.,
        capped_pairs=int((cap < 1).sum()), proposed_pairs=len(margin),
        predicted_margin_change_mean=float((change*cap).mean()))


def make_view(model, x, y, arm, alpha, context):
    is_example = arm == 'C'
    rows, tasks_np, cells = sample(x[-1].detach().cpu().numpy(), y.detach().cpu().numpy(), context, is_example)
    diagnostics = dict(proposed_pairs=len(cells), accepted_rows=0, skipped_pairs=0,
                       backtracking_halvings=0, control_fallback=0)
    if not len(rows):
        return None, diagnostics
    selected_x = x[:, rows]; tasks = torch.as_tensor(tasks_np, device=x.device)
    embedding, logits, gradient = sensitivity(model, selected_x)
    arange = torch.arange(len(rows), device=x.device)
    g = gradient[arange, tasks]; h = gradient[arange, 1-tasks]
    target_z = logits[arange, tasks]
    if is_example:
        grad = (target_z.sigmoid()-y[rows, tasks])[:, None]*g
        norm = grad.norm(dim=1); rms = embedding.square().mean(1).sqrt()
        delta = alpha*rms[:, None]*grad/(norm[:, None]+1e-12)
        keep = (norm > 1e-10) & (rms > 1e-12)
    else:
        delta, valid, detail = pair_deltas(embedding, target_z, g, h, alpha, arm != 'B')
        diagnostics.update(detail)
        if arm != 'B':
            with torch.no_grad():
                for attempt in range(3):
                    _, augmented_z, _ = forward(model, selected_x, delta)
                    drift = (augmented_z[arange, 1-tasks]-logits[arange, 1-tasks]).abs()
                    excessive = (drift > 1e-3).reshape(-1, 2).any(1) & valid
                    if attempt == 2:
                        valid &= ~excessive
                    elif excessive.any():
                        delta[excessive.repeat_interleave(2)] *= .5
                        diagnostics['backtracking_halvings'] += int(excessive.sum())
                    else:
                        break
        keep = valid.repeat_interleave(2)
        diagnostics['skipped_pairs'] = int((~valid).sum())
        diagnostics['pairs_by_cell'] = {f'd{d}t{t}': dict(
            proposed=sum(c == (d, t) for c in cells),
            accepted=sum(bool(v) and c == (d, t) for c, v in zip(cells, valid.tolist())))
            for d in range(3) for t in range(2)}
        if arm == 'B_MATCH':
            sign = torch.ones(len(g), device=x.device); sign[::2] = -1
            delta = sign[:, None]*g/(g.norm(dim=1, keepdim=True)+1e-12)*delta.norm(dim=1, keepdim=True)
    if arm in ('RANDOM', 'GAUSSIAN'):
        if not keep.any():
            rows, tasks_np, _ = sample(x[-1].detach().cpu().numpy(), y.detach().cpu().numpy(), context, True)
            if not len(rows):
                return None, diagnostics
            selected_x = x[:, rows]; tasks = torch.as_tensor(tasks_np, device=x.device)
            embedding, logits, _ = sensitivity(model, selected_x)
            norm = alpha*embedding.square().mean(1).sqrt()
            keep = norm > 1e-12
            diagnostics['control_fallback'] = 1
        else:
            norm = delta.norm(dim=1)
        gen = rng(context, arm)
        raw = (gen.choice((-1., 1.), size=embedding.shape) if arm == 'RANDOM' else
               gen.standard_normal(size=embedding.shape))
        direction = torch.as_tensor(raw, device=x.device, dtype=embedding.dtype)
        delta = direction/(direction.norm(dim=1, keepdim=True)+1e-12)*norm[:, None]
    diagnostics['accepted_rows'] = int(keep.sum())
    if not keep.any():
        return None, diagnostics
    rows_tensor = torch.as_tensor(rows, device=x.device)[keep]
    result = dict(x=selected_x[:, keep], y=y[rows_tensor], tasks=tasks[keep],
                  delta=delta[keep].detach(), clean_logits=logits[keep])
    diagnostics['delta_norm_mean'] = float(result['delta'].norm(dim=1).mean())
    diagnostics['delta_norm_max'] = float(result['delta'].norm(dim=1).max())
    return result, diagnostics


def train_step(model, optimizer, ema, x, y, config, context):
    optimizer.zero_grad(set_to_none=True)
    view = None; diagnostics = {}
    if config['arm'] != 'CLEAN' and config['lambda_aug']:
        view, diagnostics = make_view(model, x, y, config['arm'], config['alpha'], context)
    clean = F.binary_cross_entropy(model(x), y)
    auxiliary = clean.new_zeros(())
    if view is not None:
        probability, logits, _ = forward(model, view['x'], view['delta'])
        ids = torch.arange(len(view['tasks']), device=x.device); tasks = view['tasks']
        auxiliary = 2*F.binary_cross_entropy(probability[ids, tasks], view['y'][ids, tasks])
        change = logits.detach()-view['clean_logits']
        diagnostics['non_target_drift_mean'] = float(change[ids, 1-tasks].abs().mean())
        diagnostics['non_target_drift_max'] = float(change[ids, 1-tasks].abs().max())
        diagnostics['target_logit_change_mean'] = float(change[ids, tasks].mean())
        if config['arm'] in ('A', 'B', 'B_MATCH'):
            before = view['clean_logits'][ids, tasks]
            after = logits.detach()[ids, tasks]
            diagnostics['margin_before_mean'] = float((before[::2]-before[1::2]).mean())
            diagnostics['margin_after_mean'] = float((after[::2]-after[1::2]).mean())
    loss = clean+config['lambda_aug']*auxiliary
    assert torch.isfinite(loss), 'Nonfinite training loss'
    loss.backward(); optimizer.step(); update_ema(ema, model, config['rho'])
    return dict(loss=float(loss.detach()), clean_loss=float(clean.detach()),
                auxiliary_loss=float(auxiliary.detach()), diagnostics=diagnostics,
                optimizer_updates=1, ema_updates=1)
