"""Measure or inject same-batch, two-state shared PAIR gradients."""
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from experiments.conditional_history_shared_update_20261122.history_controls import HistoryOptimizer
from experiments.conditional_information_momentum_20261102_v3 import protocol as v3
from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import CONDITIONS, forward_logits
from experiments.conditional_information_momentum_20261102_v3.objectives import train_step


class _FunctionalModel:
    def __init__(self, model, params, buffers):
        self.model, self.params, self.buffers = model, params, buffers
        self.tower = model.tower

    def __call__(self, x):
        return torch.func.functional_call(self.model, (self.params, self.buffers), (x,), strict=True)


def old_pair_gradient(model, x, pair, previous, buffers, forward_rng):
    """Run the PAIR loss at old shared weights and current private weights."""
    leaves = {name: value.to(x.device).detach().clone().requires_grad_()
              for name, value in previous.items()}
    params = {name: leaves.get(name, value.detach()) for name, value in model.named_parameters()}
    disposable = {name: value.clone() for name, value in buffers.items()}
    immediate_rng = v3.rng_state(x.device)
    try:
        v3.restore_rng(forward_rng, x.device)
        with torch.enable_grad():
            _, logits = forward_logits(_FunctionalModel(model, params, disposable), x)
            pos = torch.as_tensor(pair["pos"], device=x.device)
            neg = torch.as_tensor(pair["neg"], device=x.device)
            loss = F.softplus(-(logits[pos, pair["task"]] - logits[neg, pair["task"]])).mean()
            return list(torch.autograd.grad(loss, tuple(leaves.values())))
    finally:
        v3.restore_rng(immediate_rng, x.device)


def _check_gradients(raw, shared):
    if raw is None or len(raw) != len(shared):
        raise RuntimeError("Missing required S_DENSE PAIR gradients")
    for gradient, (name, parameter) in zip(raw, shared):
        if (gradient is None or gradient.shape != parameter.shape or gradient.device != parameter.device
                or gradient.dtype != torch.float32):
            raise RuntimeError(f"Invalid required S_DENSE gradient for {name}")
    if not bool(torch.stack([torch.isfinite(g).all() for g in raw]).all()):
        raise FloatingPointError("Nonfinite required S_DENSE gradient")


def _measure(raw, old, corrected):
    g, prev, used = [torch.cat([p.flatten() for p in values]).double()
                     for values in (raw, old, corrected)]
    diff = g - prev
    correction = used - g
    norm_g, norm_prev, norm_used, norm_diff, norm_correction = (
        float(item.norm()) for item in (g, prev, used, diff, correction))
    return dict(current_norm=norm_g, previous_norm=norm_prev, corrected_norm=norm_used,
                difference_norm=norm_diff, correction_norm=norm_correction,
                current_previous_cosine=float(g.dot(prev)) / (norm_g * norm_prev) if norm_g and norm_prev else None,
                current_corrected_cosine=float(g.dot(used)) / (norm_g * norm_used) if norm_g and norm_used else None,
                correction_to_current=norm_correction / norm_g if norm_g else None,
                raw_zero_event=norm_g == 0., zero_event=norm_used == 0.)


class FreshnessAdapter:
    """B0 logs the stock step, B1 observes it, and B2 changes its S_DENSE PAIR input."""

    def __init__(self, gamma, observer=False, *, on_probe=None):
        if gamma not in (None, 0., .5) or (observer and gamma != 0.):
            raise ValueError("Unregistered B freshness/observer setting")
        self.gamma = gamma
        self.observer = observer
        self.on_probe = on_probe
        self.snapshot = None
        self.progress = 0

    def state_dict(self):
        return dict(gamma=self.gamma, observer=self.observer, snapshot=self.snapshot, progress=self.progress)

    def load_state_dict(self, state):
        if (state["gamma"], state["observer"]) != (self.gamma, self.observer):
            raise ValueError("Freshness state configuration mismatch")
        snapshot, progress = state["snapshot"], state["progress"]
        if progress < 0 or ((snapshot is not None) != (self.gamma is not None and progress > 0)):
            raise ValueError("Freshness state progress mismatch")
        self.snapshot, self.progress = snapshot, progress

    def step(self, model, optimizer, ema, x, y, row_ids, config, context):
        step = context["global_step"]
        if step != self.progress + 1:
            raise ValueError("Freshness step disagrees with restored progress")
        history = self.gamma is not None
        use_probe = bool(self.gamma) or self.observer
        if history and (not isinstance(optimizer, HistoryOptimizer) or optimizer.mode != "collapsed"
                        or optimizer.on_pre_step is not None):
            raise ValueError("Freshness requires the stock A21 optimizer and its free pre-step hook")
        if not history and optimizer.mode != "adam":
            raise ValueError("B0 requires stock mixed Adam")
        if self.observer and x.is_cuda:
            torch.cuda.synchronize(x.device)
        prepare_started = perf_counter()
        current, previous, buffers, forward_rng = None, None, None, None
        if history:
            named = dict(model.named_parameters())
            if (not optimizer.shared or any(named[name] is not p for name, p in optimizer.shared)
                    or any(p.dtype != torch.float32 for _, p in optimizer.shared)):
                raise RuntimeError("S_DENSE snapshot scope differs from the FP32 manifest")
            current = {name: p.detach().clone() for name, p in optimizer.shared}
            previous = current if self.snapshot is None else self.snapshot
            if tuple(previous) != tuple(current) or any(
                    old.shape != current[n].shape or old.dtype != current[n].dtype for n, old in previous.items()):
                raise RuntimeError("Restored S_DENSE snapshot differs from the manifest")
            if use_probe:
                forward_rng = v3.rng_state(x.device)
                buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        if self.observer and x.is_cuda:
            torch.cuda.synchronize(x.device)
        probe_prepare_seconds = perf_counter() - prepare_started if use_probe else 0.
        info = dict(no_event=True, zero_event=False, probe_forwards=0, probe_autograd_calls=0,
                    probe_seconds=0., observer_seconds=probe_prepare_seconds if self.observer else 0.,
                    current_snapshot_step=step if history else None,
                    previous_snapshot_step=max(1, step - 1) if history else None)
        logits = {}

        def remember(i):
            def hook(_module, _args, output):
                logits[i] = output.detach().squeeze(1)
            return hook

        handles = [tower.register_forward_hook(remember(i)) for i, tower in enumerate(model.tower)]

        def inspect(pair, raw):
            if history:
                _check_gradients([p.grad for _, p in optimizer.shared], optimizer.shared)
            if pair is None:
                return
            domain, task, _, _ = CONDITIONS[pair["selected"]]
            pos = torch.as_tensor(pair["pos"], device=x.device)
            neg = torch.as_tensor(pair["neg"], device=x.device)
            margin = logits[task * 3 + domain][pos] - logits[task * 3 + domain][neg]
            ratio = float(pair["w"][pair["index"]] / pair["q"][pair["index"]])
            info.update(no_event=False, pair_margin=float(margin.mean()), selected_weight_over_q=ratio,
                        unique_positive_rows=int(np.unique(row_ids[pair["pos"]]).size),
                        unique_negative_rows=int(np.unique(row_ids[pair["neg"]]).size))
            if not history:
                info["zero_event"] = None
                return
            _check_gradients(raw, optimizer.shared)
            if not use_probe:
                info["zero_event"] = not any(bool(torch.count_nonzero(g)) for g in raw)
                return
            start = perf_counter()
            equal_state = self.snapshot is None or bool(torch.stack(
                [(old.to(x.device) == current[name]).all() for name, old in previous.items()]).all())
            old = raw if equal_state else old_pair_gradient(model, x, pair, previous, buffers, forward_rng)
            _check_gradients(old, optimizer.shared)
            gamma = .5 if self.observer else self.gamma
            corrected = [g + gamma * (g - before) for g, before in zip(raw, old)]
            _check_gradients(corrected, optimizer.shared)
            nontrivial = not equal_state
            info.update(_measure(raw, old, corrected), probe_forwards=int(nontrivial),
                        probe_autograd_calls=int(nontrivial), probe_seconds=perf_counter() - start)
            if self.observer:
                info["observer_seconds"] += info["probe_seconds"]
                info["observer_zero_event"] = info["zero_event"]
                info["zero_event"] = info["raw_zero_event"]
            if self.on_probe is not None and nontrivial:
                self.on_probe()
            if self.gamma:
                alpha = config["lambda_pair"] * ratio
                changes = [g_hat - g for g_hat, g in zip(corrected, raw)]
                adjusted = [p.grad.add(change, alpha=alpha)
                            for (_, p), change in zip(optimizer.shared, changes)]
                _check_gradients(adjusted, optimizer.shared)
                for k, ((_, p), change) in enumerate(zip(optimizer.shared, changes)):
                    if bool(torch.count_nonzero(change)):
                        p.grad.add_(change, alpha=alpha)
                        raw[k] = corrected[k]

        stock_step = optimizer.step

        def adam_with_receipt(pair, raw):
            inspect(pair, raw)
            stock_step(pair, raw)

        if history:
            optimizer.on_pre_step = inspect
        else:
            optimizer.step = adam_with_receipt
        try:
            result = train_step(model, optimizer, ema, x, y, row_ids, config, context)
        finally:
            if history:
                optimizer.on_pre_step = None
            else:
                del optimizer.step
            for handle in handles:
                handle.remove()
        self.snapshot, self.progress = current, step
        result["diagnostics"].update(info)
        return result
