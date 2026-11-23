"""Subset and two-batch gradient updates using one native Adam."""
import math
import time

import torch
from torch.nn import functional as F

from .source import update_ema

CORE_GROUPS = ("U", "P_CELL", "P_AXIS")
PREFIXES = {
    "U": ("embedding.", "slot_weight.", "slot_bias.", "skip_conn.", "star_mlp.", "expert."),
    "P_CELL": ("gate.", "tower."),
    "P_AXIS": ("domain_expert.", "task_expert."),
}


def groups(model):
    parts = {name: set() for name in CORE_GROUPS}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        found = [group for group, prefixes in PREFIXES.items() if name.startswith(prefixes)]
        if name in ("shared_weight", "shared_bias"):
            found.append("U")
        if len(found) != 1:
            raise ValueError(f"Unclassified or multiply classified trainable parameter: {name}")
        parts[found[0]].add(name)
    parts["P_FULL"] = parts["P_CELL"] | parts["P_AXIS"]
    return {name: frozenset(members) for name, members in parts.items()}


def natural_loss(prediction, labels):
    return F.binary_cross_entropy(prediction, labels, reduction="none").mean(1).mean()


def _range(values):
    return [min(values), max(values)] if values else None


def _clock(opt, parameter):
    state = opt.state.get(parameter)
    return int(state["step"]) if state else 0


@torch.no_grad()
def _snapshot(named, opt):
    result = {}
    for name, p in named:
        if p.grad is None:
            continue
        state = opt.state.get(p)
        result[name] = (
            p.detach().clone(),
            state["exp_avg"].clone() if state else None,
            state["exp_avg_sq"].clone() if state else None,
        )
    return result


@torch.no_grad()
def _summaries(model, named, opt, before, clocks_before, selected, weight_decay):
    quantities = {}
    clocks_after = {}
    device_rows = []
    for name, p in named:
        clocks_after[name] = _clock(opt, p)
        active = name in before
        state = opt.state.get(p)
        zero = p.new_zeros(())
        m_norm = state["exp_avg"].norm() if state else zero
        v_norm = state["exp_avg_sq"].norm() if state else zero
        if active:
            old, old_m, old_v = before[name]
            device_rows.append(torch.stack((
                p.grad.norm(), (p - old).norm(), old.norm() * weight_decay,
                m_norm, v_norm,
                (state["exp_avg"] - old_m).norm() if old_m is not None else m_norm,
                (state["exp_avg_sq"] - old_v).norm() if old_v is not None else v_norm,
                torch.count_nonzero(p.grad).to(dtype=p.dtype),
            )))
        else:
            device_rows.append(torch.stack((zero, zero, zero, m_norm, v_norm, zero, zero, zero)))
    if device_rows:
        quantities = dict(zip((name for name, _ in named), torch.stack(device_rows).cpu().tolist()))

    stats = {}
    for group, names in groups(model).items():
        ordered = sorted(names)
        rows = [quantities[name] for name in ordered]
        stats[group] = dict(
            parameters=len(names),
            included_parameters=len(names & selected),
            updated_parameters=len(names & before.keys()),
            absent_included_gradients=len((names & selected) - before.keys()),
            explicit_zero_gradients=sum(name in before and quantities[name][7] == 0 for name in ordered),
            nonzero_gradients=sum(row[7] > 0 for row in rows),
            gradient_norm=math.sqrt(sum(row[0] ** 2 for row in rows)),
            displacement_norm=math.sqrt(sum(row[1] ** 2 for row in rows)),
            coupled_decay_norm=math.sqrt(sum(row[2] ** 2 for row in rows)),
            exp_avg_norm_range=_range([row[3] for row in rows]),
            exp_avg_sq_norm_range=_range([row[4] for row in rows]),
            exp_avg_delta_norm_range=_range([row[5] for row in rows]),
            exp_avg_sq_delta_norm_range=_range([row[6] for row in rows]),
            step_before_range=_range([clocks_before[name] for name in ordered]),
            step_after_range=_range([clocks_after[name] for name in ordered]),
            step_delta_range=_range([clocks_after[name] - clocks_before[name] for name in ordered]),
        )
    return stats


def step(model, opt, ema, batches, cfg, *, names=None, detailed=True):
    """One Adam update from one or two natural batches and one EMA tick per batch."""
    started = time.perf_counter()
    sizes = tuple(len(y) for _, y in batches)
    if len(sizes) not in (1, 2) or not all(sizes):
        raise ValueError("An update requires one or two nonempty natural batches")
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    trainable = {name for name, _ in named}
    selected = trainable if names is None else frozenset(names)
    if not selected or not selected <= trainable:
        raise ValueError("Subset must contain only trainable parameter names")
    total = sum(sizes)
    losses = []
    first_gradients = {}

    for index, (x, y) in enumerate(batches):
        model.zero_grad(set_to_none=True)
        loss = natural_loss(model(x), y)
        losses.append(loss.detach())
        loss.backward()
        for name, param in named:
            if name not in selected:
                param.grad = None
        if len(sizes) == 2 and index == 0:
            first_gradients = {name: p.grad for name, p in named if p.grad is not None}
            update_ema(ema, model, cfg["rho"])

    if len(sizes) == 2:
        a, b = (size / total for size in sizes)
        for name, param in named:
            earlier, later = first_gradients.get(name), param.grad
            if earlier is None:
                if later is not None:
                    later.mul_(b)
                continue
            earlier.mul_(a)
            if later is not None:
                earlier.add_(later, alpha=b)
            param.grad = earlier

    values = torch.stack(losses).cpu().tolist()
    if not all(map(math.isfinite, values)):
        raise FloatingPointError("Nonfinite supervised loss")
    clocks_before = {name: _clock(opt, p) for name, p in named} if detailed else None
    before = _snapshot(named, opt) if detailed else None
    opt.step()
    update_ema(ema, model, cfg["rho"])
    stats = (_summaries(model, named, opt, before, clocks_before, selected, cfg["weight_decay"])
             if detailed else {})
    device = batches[0][0].device
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    return dict(loss=sum(loss * size for loss, size in zip(values, sizes)) / total,
                rows=total, training_forwards=len(sizes), parameter_backward_calls=len(sizes),
                optimizer_steps=1, ema_updates=len(sizes), groups=stats,
                seconds=time.perf_counter() - started)
