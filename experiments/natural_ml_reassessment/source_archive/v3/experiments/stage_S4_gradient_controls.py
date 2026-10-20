"""S4 same-state gradient and Adam-move controls (no model forwards)."""

from __future__ import annotations

import math
import copy
from typing import Mapping, Sequence

import torch


TOLERANCE_ABS = 1e-6
TOLERANCE_REL = 1e-4


def vector_norm(values: Sequence[torch.Tensor | None]) -> float:
    total = 0.0
    for value in values:
        if value is not None:
            if not bool(torch.isfinite(value).all().item()):
                raise FloatingPointError("nonfinite gradient or delta")
            total += float(value.detach().double().square().sum().item())
    return math.sqrt(total)


def dot(left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None]) -> float:
    if len(left) != len(right):
        raise ValueError("vector length mismatch")
    return sum(float((a.detach().double() * b.detach().double()).sum().item())
               for a, b in zip(left, right) if a is not None and b is not None)


def cosine(left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None]):
    a, b = vector_norm(left), vector_norm(right)
    return dot(left, right) / (a * b) if a > 0 and b > 0 else None


def scale_for(reference: float, source: float, key: str):
    if not math.isfinite(reference) or not math.isfinite(source):
        raise FloatingPointError("nonfinite norm: " + key)
    if source == 0:
        if reference > 0:
            raise ValueError("undefined direction-preserving scale: " + key)
        return 1.0, "zero_zero"
    if reference == 0:
        return 0.0, "target_zero"
    ratio = reference / source
    if not math.isfinite(ratio):
        raise FloatingPointError("nonfinite scale: " + key)
    return ratio, "regular"


def assert_match(actual: float, reference: float, key: str):
    residual = abs(actual - reference)
    if residual > max(TOLERANCE_ABS, TOLERANCE_REL * reference):
        raise AssertionError("S4 norm mismatch {}: {} != {}".format(key, actual, reference))
    return residual


def assert_vector_match(actual: Sequence[torch.Tensor | None],
                        reference: Sequence[torch.Tensor | None], key: str):
    """Compare applied and predicted vectors with the design's norm-relative tolerance."""
    if len(actual) != len(reference):
        raise AssertionError("S4 vector length mismatch: " + key)
    differences = []
    for observed, expected in zip(actual, reference):
        if observed is None and expected is None:
            differences.append(None)
        elif expected is None and observed is not None:
            if bool(torch.count_nonzero(observed).item()):
                raise AssertionError("S4 parameter changed with None gradient: " + key)
            differences.append(None)
        elif observed is None:
            raise AssertionError("S4 None-gradient/delta mismatch: " + key)
        else:
            differences.append(observed - expected)
    residual = vector_norm(differences)
    baseline = vector_norm(reference)
    if residual > max(TOLERANCE_ABS, TOLERANCE_REL * baseline):
        raise AssertionError("S4 vector mismatch {}: residual={}, reference_norm={}".format(
            key, residual, baseline))
    return residual


def gradients_one_forward(full_loss, target_loss, parameters: Mapping[str, torch.nn.Parameter],
                          shared_names: Sequence[str], private_names: Sequence[str]):
    """One stochastic forward has already occurred; autograd does no forward."""
    shared = [parameters[name] for name in shared_names]
    private = [parameters[name] for name in private_names]
    gf = torch.autograd.grad(full_loss, shared, retain_graph=True, allow_unused=True)
    gt = torch.autograd.grad(target_loss, shared, retain_graph=True, allow_unused=True)
    gp = torch.autograd.grad(full_loss, private, retain_graph=True, allow_unused=True) if private else ()
    for name, p, grad in zip(shared_names, shared, gf):
        p.grad = None if grad is None else grad.detach().clone()
    for name, p, grad in zip(private_names, private, gp):
        p.grad = None if grad is None else grad.detach().clone()
    return gf, gt, gp


def adam_delta(parameter: torch.nn.Parameter, gradient: torch.Tensor | None,
               optimizer: torch.optim.Adam):
    """Read-only torch.optim.Adam delta for the supported frozen S4 settings."""
    if gradient is None:
        return None, None
    if gradient.is_sparse or torch.is_complex(gradient):
        raise ValueError("S4 Adam requires dense real gradients")
    group = next((g for g in optimizer.param_groups if any(p is parameter for p in g["params"])), None)
    if group is None or group.get("amsgrad", False) or group.get("maximize", False) \
            or group.get("foreach", False) or group.get("fused", False) \
            or group.get("capturable", False) or group.get("differentiable", False):
        raise ValueError("unsupported S4 Adam configuration")
    if not all(bool(torch.isfinite(x).all().item()) for x in (parameter, gradient)):
        raise FloatingPointError("nonfinite Adam input")
    state = optimizer.state.get(parameter, {})
    step = int(state["step"].item()) if state and isinstance(state["step"], torch.Tensor) \
        else int(state["step"]) if state else 0
    m = state["exp_avg"] if state else torch.zeros_like(parameter)
    v = state["exp_avg_sq"] if state else torch.zeros_like(parameter)
    if not all(bool(torch.isfinite(x).all().item()) for x in (m, v)):
        raise FloatingPointError("nonfinite Adam moment")
    beta1, beta2 = group["betas"]
    decay = float(group["weight_decay"])
    effective = gradient.detach() + decay * parameter.detach()
    next_m = beta1 * m + (1 - beta1) * effective
    next_v = beta2 * v + (1 - beta2) * effective.square()
    t = step + 1
    correction1, correction2 = 1 - beta1 ** t, 1 - beta2 ** t
    denominator = next_v.sqrt() / math.sqrt(correction2) + float(group["eps"])
    delta = -float(group["lr"]) / correction1 * next_m / denominator
    if not bool(torch.isfinite(delta).all().item()):
        raise FloatingPointError("nonfinite Adam delta")
    return delta, decay * parameter.detach()


def adam_deltas(parameters, names, gradients, optimizer):
    values, decay = [], []
    for name, grad in zip(names, gradients):
        d, w = adam_delta(parameters[name], grad, optimizer)
        values.append(d)
        decay.append(w)
    return values, decay


def validate_adam_delta_against_clone(device=None):
    """Block formal training unless the installed Adam agrees in representative states."""
    device = torch.device(device or "cpu")
    for lr in (0.01, 0.001):
        for warmup in (0, 3):
            for grad_value in (0.0, 0.37):
                named = {
                    "embedding.example": torch.nn.Parameter(torch.tensor([0.4, -0.2], device=device)),
                    "architecture.deep_weights": torch.nn.Parameter(torch.tensor(0.15, device=device)),
                }
                opt = torch.optim.Adam(list(named.values()), lr=lr, betas=(0.9, 0.999),
                                       weight_decay=1e-5)
                for _ in range(warmup):
                    named["embedding.example"].grad = torch.tensor([0.2, -0.4], device=device)
                    named["architecture.deep_weights"].grad = torch.tensor(0.3, device=device)
                    opt.step()
                gradients = {"embedding.example": torch.tensor([grad_value, -grad_value], device=device),
                             "architecture.deep_weights": torch.tensor(grad_value, device=device)}
                before = {name: p.detach().clone() for name, p in named.items()}
                deltas = {name: adam_delta(p, gradients[name], opt)[0]
                          for name, p in named.items()}
                clones = {name: torch.nn.Parameter(p.detach().clone()) for name, p in named.items()}
                disposable = torch.optim.Adam(list(clones.values()), lr=lr,
                                              betas=(0.9, 0.999), weight_decay=1e-5)
                if warmup:
                    disposable.load_state_dict(copy.deepcopy(opt.state_dict()))
                for name, p in clones.items():
                    p.grad = gradients[name].clone()
                disposable.step()
                for name, p in clones.items():
                    observed = p.detach() - before[name]
                    assert_vector_match([observed], [deltas[name]],
                                        "installed Adam clone " + name)
                    if not torch.equal(named[name].detach(), before[name]):
                        raise AssertionError("read-only AdamDelta changed the source parameter")
    return True
