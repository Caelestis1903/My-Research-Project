"""Original objectives, genuine same-graph PAIR gradients and passive EMA."""

import copy
from functools import cache
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import sources

sources.ensure_shared_sources()

from experiments.conditional_history_shared_update_20261122.history_controls import history_config, make_history_optimizer
from experiments.conditional_information_momentum_20261102_v3 import objectives as v3_objectives
from experiments.conditional_information_momentum_20261102_v3 import protocol as v3
from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import forward_logits, sample
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.objectives import counts
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2 import protocol as gaussian_protocol
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2 import views as gaussian_views
from experiments.overnight_ab_20261103 import published_training as published

from .groups import hg_tower_forward, registry
from .optimizers import HistoryUpdate, _all_finite, pair_ratio, validate_native_inputs, validate_native_outputs
from .protocol import manifest as study_manifest

sources.verify_shared_imports()


@cache
def _published_plan():
    return json.loads(sources.PUBLISHED_ROOT.read_text())


def _published_config(spec):
    plan = _published_plan()
    config = dict(plan[spec.backbone])
    if spec.backbone == "autoifs":
        config["rank"] = plan["datasets"][spec.dataset]["autoifs_rank"]
    return config


def build_training(spec, bundle, device):
    if bundle.dataset != spec.dataset or spec.data_hash and spec.data_hash != bundle.data_hash:
        raise ValueError("Training population differs from RunSpec")
    device = torch.device(device)
    arm = spec.canonical_arm
    if spec.backbone == "m3oe":
        v3.setup(spec.seed, device)
        frozen = v3.load_protocol()["datasets"][spec.dataset]["source"]
        factory = gaussian_views.make_model if spec.base_objective == "GA31_R3_D0" else v3.make_model
        model = factory(bundle, frozen).to(device)
    elif spec.backbone == "autoifs":
        job = dict(dataset=spec.dataset, seed=spec.seed, method="AUTOIFS", config=_published_config(spec))
        session = published.create_session(job, bundle, device)
        model = session["model"]
        optimizer = _AutoIFS(session)
    else:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        published._seed_all(spec.seed)
        model = published.build_published("HYPERGATE", bundle, spec.seed, _published_config(spec)).to(device)
    scopes, manifest = registry(model, bundle)
    if spec.group_hash and spec.group_hash != manifest["scope_hash"][spec.scope]:
        raise ValueError("Parameter coordinate identity differs from RunSpec")
    if arm == "B0":
        optimizer = v3.make_optimizer(model, v3.config("A0", spec.dataset, 0.))
    elif arm == "B1":
        optimizer = v3.make_optimizer(model, v3.config("P00", spec.dataset, .05))
    elif arm == "H0":
        optimizer = make_history_optimizer(model, history_config("A21", spec.dataset, .05))
    elif spec.backbone != "autoifs":
        optimizer = HistoryUpdate(model, spec, scopes)
    ema = None if spec.backbone == "autoifs" or "fixed_ema" not in spec.readouts else copy.deepcopy(model).requires_grad_(False)
    return model, optimizer, ema


def _source_step(model, optimizer, ema, x, y, row_ids, context, observer):
    native_step = optimizer.step
    previous_hook = getattr(optimizer, "on_pre_step", None)
    h0 = optimizer.mode == "collapsed"

    @torch.no_grad()
    def checked(pair=None, raw=None):
        gradients = [p.grad for p in optimizer.parameters if p.grad is not None]
        if not _all_finite(gradients + [r for r in raw or () if r is not None]):
            raise FloatingPointError("Nonfinite source gradient")
        validate_native_inputs(optimizer.adam)
        if h0:
            wd = optimizer.config["weight_decay"]
            if not _all_finite(p.grad.add(p, alpha=wd).square() for _, p in optimizer.shared):
                raise FloatingPointError("Nonfinite source A21 mixed square")
        if observer is not None and not h0:
            observer(context, pair, raw)
        native_step(pair, raw)
        validate_native_outputs(optimizer.adam)
        if h0:
            values = [p for _, p in optimizer.shared]
            values.extend(value for state in optimizer.moments.values()
                          for value in state.values() if torch.is_tensor(value))
            values.extend(optimizer.history.get(-1, ()))
            values.extend(t for state in optimizer.shadow.values() for t in state.values())
            if not _all_finite(values):
                raise FloatingPointError("Nonfinite source A21 parameter or moment")

    def source_observer(pair, raw):
        if previous_hook is not None:
            previous_hook(pair, raw)
        observer(context, pair, raw)

    optimizer.step = checked
    if observer is not None and h0:
        optimizer.on_pre_step = source_observer
    try:
        return v3_objectives.train_step(model, optimizer, ema, x, y, row_ids, optimizer.config, context)
    finally:
        del optimizer.step
        if h0:
            optimizer.on_pre_step = previous_hook


def _m3oe_base(model, x, y, spec, context):
    probability, logits = forward_logits(model, x) if spec.lambda_pair else (model(x), None)
    if spec.base_objective != "GA31_R3_D0":
        loss = F.binary_cross_entropy(probability, y, reduction="mean")
        return loss, probability, logits, {}, counts(len(y), 0)

    real = F.binary_cross_entropy(probability, y, reduction="none")
    mask = x[-1] == 0
    rows = int(mask.sum())
    work, info = counts(len(y), rows), {"gaussian_views": []}
    if not rows:
        return real.mean(1).mean(), probability, logits, info, work
    key = gaussian_protocol.view_key(spec.dataset, spec.seed, "train", context["global_step"], "gaussian", 1)
    view = gaussian_views.draw_view(model, rows, "gaussian", key, x.device)
    kappa = study_manifest()["datasets"][spec.dataset]["gaussian_kappa"]
    augmented = F.binary_cross_entropy(model(x[:, mask], view=view, kappa=kappa), y[mask], reduction="none")
    scalar = (real.sum() + .5 * (augmented.sum() - real[mask].sum())) / real.numel()
    info["gaussian_views"].append(dict(domain=0, rows=rows, key=key, seed=view["seed"], noise_hash=view["hash"]))
    info["clean_bce"] = float(real.detach().mean())
    work.update(extra_view_rows=rows, training_forwards=2, training_forward_rows=len(y) + rows)
    return scalar, probability, logits, info, work


def _published_base(model, x, y, spec):
    mask = torch.ones_like(y, dtype=torch.bool)
    if spec.backbone == "hypergate":
        components, towers = hg_tower_forward(model, lambda: model.objective(x.T, y, mask))
        logits = torch.stack(towers, dim=1)
    else:
        outputs = []
        hook = model.network.register_forward_hook(lambda module, args, out: outputs.append(out))
        try:
            components = model.objective(x.T, y, mask)
        finally:
            hook.remove()
        if len(outputs) != 1:
            raise ValueError("Pinned AutoIFS objective changed its forward count")
        logits = torch.cat((outputs[0][0], outputs[0][1]), dim=1)
    info = {"base_components": {name: float(value.detach()) for name, value in components.items()}}
    return components["loss"], logits.detach().sigmoid(), logits, info, counts(len(y), 0)


def step(model, optimizer, ema, x, y, row_ids, spec, context, pre_step=None):
    context = dict(context, x=x, y=y, row_ids=row_ids, spec=spec)
    if (context.get("dataset"), context.get("seed")) != (spec.dataset, spec.seed) or context.get("global_step", 0) < 1:
        raise ValueError("Step context differs from the fixed PAIR RNG identity")
    if spec.canonical_arm in ("B0", "B1", "H0"):
        return _source_step(model, optimizer, ema, x, y, row_ids, context, pre_step)

    optimizer.zero_grad()
    if spec.backbone == "m3oe":
        base, probability, logits, info, work = _m3oe_base(model, x, y, spec, context)
    else:
        base, probability, logits, info, work = _published_base(model, x, y, spec)
    pair, raw, loss = None, None, base
    if spec.lambda_pair:
        pair = sample(x[-1].detach().cpu().numpy(), y.detach().cpu().numpy(), optimizer.demand,
                      False, context, row_ids)
    if pair is not None:
        pos, neg = (torch.as_tensor(pair[side], device=x.device) for side in ("pos", "neg"))
        margin = logits[pos, pair["task"]] - logits[neg, pair["task"]]
        auxiliary = F.softplus(-margin).mean()
        loss = base + spec.lambda_pair * pair_ratio(pair) * auxiliary
        if optimizer.mode != "adam":
            gs = torch.autograd.grad(auxiliary, [p for _, p in optimizer.shared], retain_graph=True,
                                     allow_unused=spec.scope != "S_DENSE")
            raw = [None if g is None else g.detach() for g in gs]
        info.update({k: pair[k] for k in ("active", "selected", "task", "unique_pair_rows", "pair_index_hash")})
        info.update(w=pair["w"].tolist(), q=pair["q"].tolist(), w_over_q=(pair["w"] / pair["q"]).tolist(),
                    alpha_over_q=(.01 / pair["q"]).tolist(), pair_loss=float(auxiliary.detach()),
                    ranking_error=float(torch.sigmoid(-margin).mean().detach()))
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite supervised loss")
    loss.backward()
    gradients = [p.grad for p in optimizer.parameters if p.grad is not None]
    if not _all_finite(gradients + [r for r in raw or () if r is not None]):
        raise FloatingPointError("Nonfinite native or raw gradient")
    if pre_step is not None:
        with torch.no_grad():
            pre_step(context, pair, raw)
    optimizer.step(pair, raw)
    if ema is not None:
        v3.update_ema(ema, model, spec.rho)
    work.update(pair_rows=4096 if pair is not None else 0, shared_autograd_calls=int(raw is not None),
                optimizer_steps=optimizer.calls if spec.backbone == "autoifs" else 1, ema_updates=int(ema is not None))
    info.update(pair_count=2048 if pair is not None else 0, active_clocks=dict(optimizer.active),
                selected_counts=dict(optimizer.selected), pair_clock=optimizer.pair_clock,
                demand=dict(optimizer.demand), norm_match=optimizer.last_update)
    return dict(loss=float(loss.detach()), real_loss=float(base.detach()), pred=probability.detach(),
                diagnostics=info, counts=work)


def loader_generator(spec):
    return torch.Generator().manual_seed(spec.seed + 11000)


def training_order(spec, bundle, epoch, generator):
    if spec.backbone == "m3oe":
        return torch.randperm(len(bundle.train_y), generator=generator).numpy()
    method = spec.backbone.upper()
    source_epoch = (epoch - 1 if epoch <= 10 else epoch + 90) if method == "AUTOIFS" else epoch
    return published._stage_rng(spec.seed, spec.dataset, method, source_epoch).permutation(len(bundle.train_y))


@torch.no_grad()
def predict(spec, model, x, device):
    started = time.perf_counter()
    predictions = []
    training = model.training
    model.eval()
    try:
        for offset in range(0, len(x), spec.eval_batch_size):
            batch = torch.tensor(np.asarray(x[offset:offset + spec.eval_batch_size]), device=device)
            probability = model(batch.T) if spec.backbone == "m3oe" else model(batch).float().sigmoid()
            predictions.append(probability.detach().cpu().numpy())
    finally:
        model.train(training)
    return np.concatenate(predictions), dict(seconds=time.perf_counter() - started, forwards=len(predictions), rows=len(x))


def evaluate(spec, model, x, y, device):
    predictions, work = predict(spec, model, x, device)
    return v3.metrics(y, predictions, x[:, -1]), work


def inference_meta(model, spec):
    return published._model_meta(model, spec.backbone.upper())


def restore_inference_meta(model, meta):
    published.restore_model_meta(model, meta)


class _AutoIFS:
    """The pinned search/reuse stage transition with a stable live model reference."""

    mode = "adam"
    shared = ()
    pair_clock = 0

    def __init__(self, session):
        self.session = session
        self.model = session["model"]
        self.parameters = list(self.model.parameters())
        self.trainable = {name for name, p in self.model.named_parameters() if p.requires_grad}
        self.active, self.selected, self.demand, self.history, self.last_update = {}, {}, {}, {}, {}
        self.parameter_manifest = [dict(name=n, shape=list(p.shape), history=False)
                                   for n, p in self.model.named_parameters()]

    @property
    def calls(self):
        return len(self.session["optimizers"])

    @property
    def adam(self):
        return self.session["optimizers"]["base"]

    def zero_grad(self, set_to_none=True):
        self.model.zero_grad(set_to_none=set_to_none)

    def step(self, pair=None, raw=None):
        for optimizer in self.session["optimizers"].values():
            validate_native_inputs(optimizer)
        for optimizer in self.session["optimizers"].values():
            optimizer.step()
            validate_native_outputs(optimizer)

    def state_dict(self):
        return dict(version=1, stage=self.session["stage"], model_meta=published._model_meta(self.model, "AUTOIFS"),
                    rewind_state=self.session["rewind_state"],
                    optimizers={name: opt.state_dict() for name, opt in self.session["optimizers"].items()})

    def load_state_dict(self, state):
        if state["version"] != 1 or state["stage"] not in ("search", "reuse"):
            raise ValueError("AutoIFS optimizer stage identity mismatch")
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(name in self.trainable)
        restore_inference_meta(self.model, state["model_meta"])
        stage = state["stage"]
        opts = published._optimizers(self.model, "AUTOIFS", stage, self.session["job"]["config"])
        if opts.keys() != state["optimizers"].keys():
            raise ValueError("AutoIFS checkpoint optimizer ownership changed")
        for name, optimizer in opts.items():
            optimizer.load_state_dict(state["optimizers"][name])
        self.session.update(stage=stage, rewind_state=state["rewind_state"], optimizers=opts)


def prepare_epoch(model, optimizer, ema, spec, epoch):
    if spec.backbone == "autoifs" and epoch <= 10:
        config = optimizer.session["job"]["config"]
        search_epoch = epoch - 1
        if search_epoch == config["rewind_epoch_index"]:
            optimizer.session["rewind_state"] = v3.cpu_copy(model.state_dict())
        if search_epoch:
            model.network.temp *= (config["temperature_end"] / config["temperature_start"]) ** (
                1. / (config["search_epochs"] - 1))


def finish_epoch(model, optimizer, ema, spec, epoch):
    if spec.backbone == "autoifs" and epoch == 10:
        session = optimizer.session
        published._start_reuse(session)
        replacement = session["model"]
        model.load_state_dict(replacement.state_dict())
        restore_inference_meta(model, published._model_meta(replacement, "AUTOIFS"))
        session.update(model=model, optimizers=published._optimizers(
            model, "AUTOIFS", "reuse", session["job"]["config"]))
