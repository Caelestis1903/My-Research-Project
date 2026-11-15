"""B/C adapters from pilot bf36e454; zero controls change accepted delta only."""
import hashlib
import time
import numpy as np
import torch
import torch.nn.functional as F
from experiments.conditional_information_momentum_20261102_v3.protocol import update_ema


def rng(context, stream):
    key = '|'.join(str(context[k]) for k in ('dataset', 'seed', 'epoch', 'global_step')) + '|' + stream
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))


def forward(model, x, delta=None, probe=False):
    """Preserve ordinary training detach; reconnect gates only for Jacobian probes."""
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


def sensitivity(model, x):
    with torch.enable_grad():
        _, z, embedding = forward(model, x, probe=True)
        gradients = [torch.autograd.grad(z[:, t].sum(), embedding, retain_graph=t == 0,
                                        create_graph=False)[0].detach() for t in range(2)]
    return embedding.detach(), z.detach(), torch.stack(gradients, dim=1)


def make_view(model, x, y, arm, alpha, context, row_ids=None):
    assert arm in ('B', 'C', 'Z-PAIR', 'Z-EXAMPLE')
    example = arm in ('C', 'Z-EXAMPLE')
    rows, tasks_np, cells = sample(x[-1].detach().cpu().numpy(), y.detach().cpu().numpy(), context, example)
    global_ids = np.arange(len(y)) if row_ids is None else np.asarray(row_ids)
    trace = dict(candidate_local_rows=rows.tolist(), candidate_row_ids=global_ids[rows].tolist(),
                 candidate_tasks=tasks_np.tolist(), pair_ids=[] if example else (np.arange(len(rows))//2).tolist(),
                 candidate_domains=x[-1, rows].long().cpu().tolist(),
                 candidate_labels=y[rows, tasks_np].long().cpu().tolist(), keep=[], skip_reason=[],
                 accepted_local_rows=[], accepted_row_ids=[], accepted_tasks=[],
                 raw_delta_norm=[], capped_delta_norm=[], actual_delta_norm=[], cap=[])
    diag = dict(proposed_rows=len(rows), accepted_rows=0, skipped_rows=len(rows))
    if not len(rows):
        return None, diag, trace
    selected_x = x[:, rows]; tasks = torch.as_tensor(tasks_np, device=x.device)
    embedding, logits, gradient = sensitivity(model, selected_x)
    ids = torch.arange(len(rows), device=x.device)
    g = gradient[ids, tasks]; target_z = logits[ids, tasks]
    rms = embedding.square().mean(1).sqrt()
    if example:
        direction = (target_z.sigmoid()-y[rows, tasks])[:, None]*g
        norm = direction.norm(dim=1)
        raw = alpha*rms[:, None]*direction/(norm[:, None]+1e-12)
        keep = (norm > 1e-10) & (rms > 1e-12)
        cap = torch.ones_like(norm); delta = raw
        reasons = torch.where(norm <= 1e-10, 1, torch.where(rms <= 1e-12, 2, 0))
    else:
        norm = g.norm(dim=1)
        sign = torch.ones(len(g), device=g.device, dtype=g.dtype); sign[::2] = -1
        raw = sign[:, None]*alpha*rms[:, None]*g/(norm[:, None]+1e-12)
        margin = target_z[::2]-target_z[1::2]
        product = (g*raw).sum(1); change = product[::2]-product[1::2]
        direction_valid = ((norm > 1e-10) & (rms > 1e-12)).reshape(-1, 2).all(1)
        valid = direction_valid & (margin.abs() > 1e-7) & (change.abs() > 1e-12)
        cap = torch.minimum(torch.ones_like(margin), margin.abs()/(change.abs()+1e-12)).repeat_interleave(2)
        delta = raw*cap[:, None]; keep = valid.repeat_interleave(2)
        pair_reason = torch.where(~direction_valid, 1,
            torch.where(margin.abs() <= 1e-7, 3, torch.where(change.abs() <= 1e-12, 4, 0)))
        reasons = pair_reason.repeat_interleave(2)
        diag.update(proposed_pairs=len(cells), skipped_pairs=int((~valid).sum()),
            capped_pairs=int((cap[::2] < 1).sum()),
            predicted_margin_change_mean=float((change*cap[::2]).mean()))
    trace.update(keep=keep.cpu().tolist(), skip_reason=reasons.cpu().tolist(),
        raw_delta_norm=raw.norm(dim=1).cpu().tolist(), capped_delta_norm=delta.norm(dim=1).cpu().tolist(),
        cap=cap.cpu().tolist(), direction_norm=norm.cpu().tolist(), embedding_rms=rms.cpu().tolist())
    accepted = torch.as_tensor(rows, device=x.device)[keep]
    actual = delta[keep].detach()
    if arm.startswith('Z-'):
        actual = torch.zeros_like(actual)
    trace.update(accepted_local_rows=accepted.cpu().tolist(),
        accepted_row_ids=global_ids[accepted.cpu().numpy()].tolist(), accepted_tasks=tasks[keep].cpu().tolist(),
        actual_delta_norm=actual.norm(dim=1).cpu().tolist())
    diag.update(accepted_rows=int(keep.sum()), skipped_rows=int((~keep).sum()))
    if not keep.any():
        return None, diag, trace
    view = dict(x=selected_x[:, keep], y=y[accepted], tasks=tasks[keep], rows=accepted,
                delta=actual, clean_logits=logits[keep])
    diag.update(delta_norm_mean=float(actual.norm(dim=1).mean()), delta_norm_max=float(actual.norm(dim=1).max()))
    return view, diag, trace


def weight_ledger(x, y, probabilities, view, aux_entries, coefficient):
    """Detached accounting leaves the source BCE reduction path untouched."""
    clean_entries = F.binary_cross_entropy(probabilities.detach(), y, reduction='none')
    batch = len(y); count = 0 if view is None else len(view['tasks'])
    clean_coefficient = 1/(2*batch); aux_coefficient = 2*coefficient/count if count else 0.
    cells = {}
    for d in range(3):
        for t in range(2):
            for label in (0, 1):
                clean_mask = (x[-1] == d) & (y[:, t] == label)
                aux_mask = (torch.zeros(0, device=x.device, dtype=torch.bool) if view is None else
                    ((view['x'][-1] == d) & (view['tasks'] == t) & (view['y'][:, t] == label)))
                clean_sum = float(clean_entries[clean_mask, t].sum())
                aux_sum = 0. if view is None else float(aux_entries[aux_mask].sum())
                cells[f'd{d}t{t}y{label}'] = dict(clean_count=int(clean_mask.sum()), aux_count=int(aux_mask.sum()),
                    clean_numerator=clean_sum, aux_numerator=aux_sum,
                    clean_weight_mass=int(clean_mask.sum())*clean_coefficient,
                    aux_weight_mass=int(aux_mask.sum())*aux_coefficient,
                    clean_contribution=clean_sum*clean_coefficient, aux_contribution=aux_sum*aux_coefficient)
    rows = [] if view is None else view['rows'].cpu().tolist()
    return dict(B=batch, M=count, clean_numerator=float(clean_entries.sum()), clean_denominator=2*batch,
        auxiliary_numerator=float(aux_entries.sum()), auxiliary_denominator=count,
        clean_entry_coefficient=clean_coefficient, auxiliary_entry_coefficient=aux_coefficient,
        auxiliary_to_clean_ratio=aux_coefficient/clean_coefficient,
        total_weight_mass=1+2*coefficient if count else 1., unique_selected_rows=len(set(rows)),
        repeated_row_appearances=count-len(set(rows)), cells=cells)


def gradient_diagnostic(clean, auxiliary, model):
    parameters = tuple(model.parameters())
    g = torch.autograd.grad(clean, parameters, retain_graph=True, allow_unused=True)
    h = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True) if auxiliary.requires_grad else ()
    norm_g = sum(float(a.detach().double().square().sum()) for a in g if a is not None)**.5
    norm_h = sum(float(a.detach().double().square().sum()) for a in h if a is not None)**.5
    dot = sum(float(a.detach().double().mul(b.detach().double()).sum())
              for a, b in zip(g, h) if a is not None and b is not None)
    return dict(clean_gradient_norm=norm_g, auxiliary_gradient_norm=norm_h,
                clean_auxiliary_gradient_cosine=dot/(norm_g*norm_h) if norm_g*norm_h else None)


def train_step(model, optimizer, ema, x, y, config, context, row_ids=None, diagnose=False):
    optimizer.zero_grad(set_to_none=True); started = time.perf_counter()
    view, diagnostics, trace = make_view(model, x, y, config['arm'], config['alpha'], context, row_ids)
    probe_seconds = time.perf_counter()-started; training_started = time.perf_counter()
    probabilities = model(x)
    clean = F.binary_cross_entropy(probabilities, y)
    auxiliary = clean.new_zeros(()); aux_entries = clean.new_empty((0,))
    if view is not None:
        probability, logits, _ = forward(model, view['x'], view['delta'])
        ids = torch.arange(len(view['tasks']), device=x.device); tasks = view['tasks']
        auxiliary = 2*F.binary_cross_entropy(probability[ids, tasks], view['y'][ids, tasks])
        aux_entries = F.binary_cross_entropy(probability.detach()[ids, tasks], view['y'][ids, tasks], reduction='none')
        change = logits.detach()-view['clean_logits']
        diagnostics.update(non_target_drift_mean=float(change[ids, 1-tasks].abs().mean()),
            non_target_drift_max=float(change[ids, 1-tasks].abs().max()),
            target_logit_change_mean=float(change[ids, tasks].mean()))
        if config['arm'] in ('B', 'Z-PAIR'):
            before = view['clean_logits'][ids, tasks]; after = logits.detach()[ids, tasks]
            diagnostics.update(margin_before_mean=float((before[::2]-before[1::2]).mean()),
                               margin_after_mean=float((after[::2]-after[1::2]).mean()))
    loss = clean+config['lambda_aug']*auxiliary
    assert torch.isfinite(loss), 'Nonfinite training loss'
    ledger = weight_ledger(x, y, probabilities, view, aux_entries, config['lambda_aug'])
    if diagnose:
        diagnostics.update(gradient_diagnostic(clean, auxiliary, model))
    loss.backward(); optimizer.step(); update_ema(ema, model, config['rho'])
    return dict(loss=float(loss.detach()), clean_loss=float(clean.detach()), auxiliary_loss=float(auxiliary.detach()),
        diagnostics=diagnostics, ledger=ledger, support=trace, optimizer_updates=1, ema_updates=1,
        probe_seconds=probe_seconds, training_seconds=time.perf_counter()-training_started)
