"""Source-separated history recurrences and one native/custom update transaction."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from experiments.conditional_information_momentum_20261102_v3.optimizers import shared_name

_EPS = 1e-8
_COMPLEMENT = {.99: .01, .9: .1}
_H0 = SimpleNamespace(algorithm_id="A21_CANONICAL", zeta=0., alpha=0., beta_pair=.99)


@dataclass(frozen=True)
class Update:
    numerator: torch.Tensor
    denominator: torch.Tensor | None = None

    def direction(self):
        return self.numerator if self.denominator is None else self.numerator / self.denominator

    def fp64_step(self, lr):
        direct = self.numerator.double()
        if self.denominator is not None:
            direct = direct / self.denominator.double()
        return direct * -lr

    def apply_(self, theta, lr):
        if self.denominator is None:
            return theta.add_(self.numerator, alpha=-lr)
        return theta.addcdiv_(self.numerator, self.denominator, value=-lr)

    def delta(self, theta, lr):
        return self.apply_(theta.detach().clone(), lr).sub_(theta.detach())


@torch.no_grad()
def components(grad, theta, raw, ratio, lam, wd):
    mixed = grad.add(theta, alpha=wd)
    return (mixed, mixed, torch.zeros_like(mixed)) if ratio is None or raw is None else (
        mixed, mixed - (lam * ratio) * raw, ratio * raw)


def pair_ratio(pair):
    if pair is None:
        return None
    i = pair["index"]
    return float(pair["w"][i] / pair["q"][i])


def _decayed(state, name, like, beta):
    old = state.get(name)
    return torch.zeros_like(like) if old is None else old.mul(beta)


@torch.no_grad()
def transition(moments, mixed, raw, ratio, spec):
    """Return a prospective per-parameter state and direction without touching the input state."""
    if mixed is None:
        return moments, None
    event = ratio is not None
    lam = spec.lambda_pair
    kind = spec.algorithm_id
    step = moments.get("step", 0) + 1
    k = moments.get("k", 0) + int(event)
    state = dict(moments, step=step, k=k)
    v = _decayed(moments, "v", mixed, .999)
    state["v"] = v
    if kind == "LAPROP_MIXED_ADAPTED":
        v.addcmul_(mixed, mixed, value=.001)
        d = (v / (1 - .999 ** step)).sqrt().add_(_EPS).reciprocal_()
        state["t"] = _decayed(moments, "t", mixed, .9).addcmul_(d, mixed, value=.1)
        return state, Update(state["t"] / (1 - .9 ** step))

    if event:
        if raw is None:
            raw = torch.zeros_like(mixed)
        b = mixed - (lam * ratio) * raw
        h = ratio * raw
    else:
        b = mixed
        h = None
    m = _decayed(moments, "m", mixed, .9).add_(b, alpha=.1)
    state["m"] = m

    if kind == "ADATASK_TWO_SOURCE_ADAPTED":
        v.addcmul_(b, b, value=.001)
        v_p = _decayed(moments, "v_p", mixed, .999)
        if event:
            v_p.addcmul_(h, h, value=.001)
        state["v_p"] = v_p
    elif not spec.zeta or not event:
        v.addcmul_(mixed, mixed, value=.001)
    else:
        separated = b.square().add_(h.mul(lam).square())
        variance_input = separated if spec.zeta == 1. else (
            mixed.square().mul_(1 - spec.zeta).add_(separated, alpha=spec.zeta))
        v.add_(variance_input, alpha=.001)

    denom = (v / (1 - .999 ** step)).sqrt().add_(_EPS)
    numerator = m / (1 - .9 ** step)
    if not event:
        return state, Update(numerator, denom)

    beta = spec.beta_pair
    complement = _COMPLEMENT[beta]
    bias = 1 - beta ** k
    q = _decayed(moments, "q", mixed, beta).add_(raw, alpha=complement * ratio)
    state["q"] = q
    if kind == "ADATASK_TWO_SOURCE_ADAPTED":
        denom_p = (v_p / (1 - .999 ** step)).sqrt().add_(_EPS)
        direction = (numerator / denom).add_(q / denom_p, alpha=lam / bias)
        return state, Update(direction)
    if spec.alpha:
        d = denom.reciprocal()
        u = _decayed(moments, "u", mixed, beta).addcmul_(d, h, value=complement)
        state["u"] = u
        direction = numerator.mul_(d)
        direction.add_(q * d, alpha=lam * (1 - spec.alpha) / bias)
        direction.add_(u, alpha=lam * spec.alpha / bias)
        return state, Update(direction)

    numerator.add_(q, alpha=lam / bias)
    return state, Update(numerator, denom)


def _all_finite(values):
    by_device = {}
    for value in values:
        by_device.setdefault(value.device, []).append(torch.isfinite(value).all())
    return all(bool(torch.stack(checks).all()) for checks in by_device.values())


@torch.no_grad()
def validate_native_inputs(adam):
    for group in adam.param_groups:
        wd = group["weight_decay"]
        mixed = (p.grad.add(p, alpha=wd) if wd else p.grad for p in group["params"] if p.grad is not None)
        if not _all_finite(value.square() for value in mixed):
            raise FloatingPointError("Nonfinite native Adam input or its square")


@torch.no_grad()
def validate_native_outputs(adam):
    values = []
    for group in adam.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            values.append(p)
            values.extend(adam.state[p][key] for key in ("exp_avg", "exp_avg_sq"))
    if not _all_finite(values):
        raise FloatingPointError("Nonfinite native Adam parameter or moment")


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device).clone()
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _checked_recurrence(state, update):
    values = [value for value in state.values() if torch.is_tensor(value)]
    values.append(update.numerator)
    if update.denominator is not None:
        values.append(update.denominator)
    return values


def _group_name(name):
    return "EXPERT" if name.startswith("expert.") else "STAR_SKIP"


@torch.no_grad()
def norm_matched_steps(entries, lr):
    """Match complete candidate step radii on one trajectory in registered order."""
    groups = {"STAR_SKIP": [], "EXPERT": []}
    for name, p, reference, candidate in entries:
        groups[_group_name(name)].append((name, p, reference.fp64_step(lr), candidate.fp64_step(lr)))
    values, receipt = {}, {}
    for group, block in groups.items():
        if not block:
            continue
        reference = torch.stack([r.square().sum() for _, _, r, _ in block]).sum().sqrt()
        target = torch.stack([c.square().sum() for _, _, _, c in block]).sum().sqrt()
        if not _all_finite((reference, target)):
            raise FloatingPointError(f"Nonfinite norm matching radius for {group}")
        zero = not bool(reference)
        scale = 0. if zero else float(target / reference)
        status = ("MATCHED_ZERO" if not bool(target) else "UNMATCHED_ZERO") if zero else "MATCHED"
        errors, actual = [], []
        for name, p, r, _ in block:
            ideal = r * scale
            changed = p.add(ideal.to(dtype=p.dtype))
            values[name] = changed
            displacement = changed.double().sub_(p.double())
            errors.append(displacement.sub(ideal).square().sum())
            actual.append(displacement.square().sum())
        receipt[group] = dict(status=status, scale=scale, reference=float(reference), target=float(target),
                              actual_step_norm=float(torch.stack(actual).sum().sqrt()),
                              fp32_displacement_error=float(torch.stack(errors).sum().sqrt()))
    return values, receipt


class HistoryUpdate:
    """Keep native Adam on the complement and a detached state per custom parameter."""

    def __init__(self, model, spec, scopes):
        spec = spec.optimizer_spec
        self.spec = spec
        self.config = spec.to_dict()
        self.named = list(model.named_parameters())
        self.parameters = [p for _, p in self.named]
        self.shared = [(name, p) for name, p in self.named if scopes[name] == spec.scope]
        native_only = spec.scope == "NATIVE_ALL" or spec.canonical_arm.endswith("B0")
        self.mode = "adam" if native_only or spec.algorithm_id == "LAPROP_MIXED_ADAPTED" else "history_update"
        if native_only:
            self.shared = []
        elif not self.shared:
            raise ValueError(f"Empty registered custom scope: {spec.scope}")
        shared_ids = {id(p) for _, p in self.shared}
        if spec.algorithm_id == "H0_NORM_MATCHED" and (
                spec.scope != "S_DENSE" or not all(shared_name(name) for name, _ in self.shared)):
            raise ValueError("M requires the fixed shared STAR_SKIP and EXPERT coordinates")
        flags = dict(foreach=False, fused=False, amsgrad=False) if spec.backbone == "m3oe" else {}
        self.adam = torch.optim.Adam([p for p in self.parameters if id(p) not in shared_ids],
                                    lr=spec.lr, betas=(.9, .999), eps=_EPS,
                                    weight_decay=spec.weight_decay, **flags)
        self.parameter_manifest = [dict(name=name, shape=list(p.shape), history=id(p) in shared_ids)
                                   for name, p in self.named]
        self.moments = {}
        self.candidate_moments = {}
        self.pair_clock = 0
        self.active, self.selected, self.demand, self.history = {}, {}, {}, {}
        self.last_update = {}
        self.on_pre_step = None

    def zero_grad(self, set_to_none=True):
        for p in self.parameters:
            p.grad = None if set_to_none else torch.zeros_like(p)

    def state_dict(self):
        return dict(version=1, mode=self.mode, config=self.config, parameter_manifest=self.parameter_manifest,
                    adam=self.adam.state_dict(), moments=self.moments, candidate_moments=self.candidate_moments,
                    pair_clock=self.pair_clock, selected=self.selected, active=self.active,
                    demand=self.demand, history=self.history)

    def load_state_dict(self, state):
        if (state["version"], state["mode"], state["config"], state["parameter_manifest"]) != (
                1, self.mode, self.config, self.parameter_manifest):
            raise ValueError("History optimizer identity differs from checkpoint")
        self.adam.load_state_dict(state["adam"])
        for name in ("moments", "candidate_moments", "pair_clock", "selected", "active", "demand", "history"):
            setattr(self, name, _move(state[name], self.parameters[0].device))

    @torch.no_grad()
    def step(self, pair=None, r=None):
        if self.on_pre_step is not None:
            self.on_pre_step(pair, r)
        ratio = pair_ratio(pair)
        spec = self.spec
        is_m = spec.algorithm_id == "H0_NORM_MATCHED"
        if self.mode != "adam" and pair is not None and (r is None or len(r) != len(self.shared)):
            raise ValueError("Raw PAIR gradients must follow the registered custom scope order")
        if is_m:
            reference = SimpleNamespace(**vars(_H0), lambda_pair=spec.lambda_pair)
            candidate = SimpleNamespace(algorithm_id=spec.target_algorithm_id, lambda_pair=spec.lambda_pair,
                                        beta_pair=spec.beta_pair, zeta=spec.zeta, alpha=spec.alpha)
        pending, moments, candidates, values, matched = {}, {}, {}, [], []
        for i, (name, p) in enumerate(self.shared):
            raw = None if r is None else r[i]
            if spec.scope == "S_DENSE" and (p.grad is None or (
                    self.mode != "adam" and pair is not None and raw is None)):
                raise RuntimeError(f"Missing required S_DENSE native/raw gradient: {name}")
            if p.grad is None:
                if pair is not None and raw is not None and bool(torch.count_nonzero(raw)):
                    raise RuntimeError(f"Nonzero raw PAIR gradient with native None: {name}")
                continue
            mixed = p.grad.add(p, alpha=spec.weight_decay)
            moments[name], update = transition(self.moments.get(name, {}), mixed, raw, ratio,
                                                reference if is_m else spec)
            values.extend(_checked_recurrence(moments[name], update))
            if is_m:
                candidates[name], target = transition(self.candidate_moments.get(name, {}), mixed, raw, ratio, candidate)
                values.extend(_checked_recurrence(candidates[name], target))
                matched.append((name, p, update, target))
            else:
                pending[name] = update.apply_(p.detach().clone(), spec.lr)

        receipt = {}
        if is_m:
            pending, receipt = norm_matched_steps(matched, spec.lr)
        if not _all_finite((*values, *pending.values())):
            raise FloatingPointError("Nonfinite custom optimizer state or proposed parameter")
        validate_native_inputs(self.adam)
        self.adam.step()
        validate_native_outputs(self.adam)
        for name, p in self.shared:
            if name in pending:
                p.copy_(pending[name])
        self.moments.update(moments)
        self.candidate_moments.update(candidates)
        if pair is not None:
            condition = pair["selected"]
            self.pair_clock += 1
            self.selected[condition] = self.selected.get(condition, 0) + 1
        self.last_update = receipt
