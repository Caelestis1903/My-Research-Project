"""Ordered parameter coordinates for the original and transferred scopes."""

from collections import Counter

import numpy as np
import torch

from experiments.conditional_history_shared_update_20261122.b_virtual import registry as source_registry
from experiments.conditional_information_momentum_20261102_v3.protocol import state_digest

CELLS = tuple(f"d{d}t{t}" for d in range(3) for t in range(2))


def hg_tower_forward(model, function):
    """Capture the very logits consumed by the native objective or predictor."""
    towers = model.network.towers
    names = tuple(model.network.frating)
    outputs = []
    handles = []
    for name in names:
        handles.append(towers[name].register_forward_hook(
            lambda module, args, out, name=name: outputs.append((name, out.squeeze(-1)))))
    try:
        result = function()
    finally:
        for handle in handles:
            handle.remove()
    if tuple(name for name, _ in outputs) != names:
        raise ValueError("Pinned HyperGate tower call order changed")
    return result, tuple(logits for _, logits in outputs)


def _hypergate(model, bundle, named):
    parameters = [p for _, p in named]
    device = parameters[0].device
    domain = bundle.features.columns.index("domain_indicator")
    rows = [np.flatnonzero(bundle.train_x[:, domain] == d)[0] for d in range(3)]
    x = torch.as_tensor(bundle.train_x[rows], device=device)
    parameters_live = [(i, p) for i, p in enumerate(parameters) if p.requires_grad]
    live_indices, live = zip(*parameters_live)
    reached = [set() for _ in parameters]
    mode = model.training
    try:
        model.eval()
        with torch.enable_grad():
            result, towers = hg_tower_forward(model, lambda: model(x))
            if not torch.equal(result, torch.stack(towers, dim=1)):
                raise ValueError("Captured HyperGate logits differ from the pinned output")
            for cell in range(6):
                domain, task = divmod(cell, 2)
                grads = torch.autograd.grad(towers[task][domain], live, allow_unused=True,
                                            retain_graph=cell < 5)
                for i, grad in zip(live_indices, grads):
                    if grad is not None:
                        reached[i].add(CELLS[cell])
    finally:
        model.train(mode)
    embeddings = {id(p) for module in model.modules() if isinstance(module, torch.nn.Embedding)
                  for p in module.parameters()}
    records, scopes, offset = [], {}, 0
    for (name, p), cells in zip(named, reached):
        embedding = name.startswith("network.embedding.") or id(p) in embeddings
        active = not embedding and len(cells) == len(CELLS)
        group = "HG_SHARED_DENSE" if active else "HG_EMBED" if embedding else "HG_NATIVE"
        scopes[name] = "HG_SHARED_DENSE" if active else "NATIVE_ALL"
        records.append(dict(name=name, shape=list(p.shape), dtype=str(p.dtype), group=group,
                            scope=scopes[name], start=offset, end=offset + p.numel(),
                            cells=sorted(cells), reachability="autograd_non_none", mask=None,
                            requires_grad=p.requires_grad, native_step_name=name))
        offset += p.numel()
    if not any(scope == "HG_SHARED_DENSE" for scope in scopes.values()):
        raise ValueError("HyperGate shared dense scope is empty")
    return scopes, records


def registry(model, bundle):
    named = list(model.named_parameters())
    if hasattr(model, "network") and hasattr(model.network, "towers"):
        scopes, records = _hypergate(model, bundle, named)
    elif hasattr(model, "optimizer_groups"):
        scopes, records, offset = {}, [], 0
        for name, p in named:
            scopes[name] = "NATIVE_ALL"
            records.append(dict(name=name, shape=list(p.shape), dtype=str(p.dtype), group="NATIVE_ALL",
                                scope="NATIVE_ALL", start=offset, end=offset + p.numel(),
                                mask=None, requires_grad=p.requires_grad, native_step_name=name))
            offset += p.numel()
    else:
        scopes, native = source_registry(model, bundle)
        records = native["records"]
        domain_name = f"embedding.embed_dict.{bundle.features.columns.index('domain_indicator')}.weight"
        by_name = dict(named)
        for record in records:
            name, group = record["name"], record["group"]
            cells = CELLS
            if group.startswith("DT"):
                cells = (f"d{group[2]}t{group[3]}",)
            elif group.startswith("D"):
                cells = tuple(cell for cell in CELLS if cell[1] == group[1])
            elif group.startswith("T"):
                cells = tuple(cell for cell in CELLS if cell[3] == group[1])
            record.update(cells=list(cells), reachability="fixed_structural_assignment",
                          mask={"axis": 0, "index": int(group[1])} if name == domain_name else None,
                          requires_grad=by_name[name].requires_grad, native_step_name=name)

    counts = Counter()
    for record in records:
        counts[record["group"]] += record["end"] - record["start"]
    if set(scopes) != {name for name, _ in named} or sum(counts.values()) != sum(p.numel() for _, p in named):
        raise ValueError("Parameter registration omitted or duplicated coordinates")
    group_hash = state_digest(records)
    scope_hash = {"NATIVE_ALL": group_hash}
    for scope in set(scopes.values()) - {"NATIVE_ALL"}:
        scope_hash[scope] = state_digest([entry for entry in records if entry["scope"] == scope])
    return scopes, dict(records=records, counts=dict(counts), group_hash=group_hash, scope_hash=scope_hash)
