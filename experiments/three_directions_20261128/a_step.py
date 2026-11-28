"""Direction A objective and single native Adam/fixed-EMA update."""
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.conditional_information_momentum_20261102_v3.protocol import load_protocol
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.objectives import counts, source_config
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.protocol import make_optimizer, update_ema
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2 import (
    objectives as ga31, views,
)
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.vendor.gradient_aug_20261031_v2.protocol import view_key

from .macro_pair_sampler import PAIR_COUNT, MacroPairSampler, ordered_ids_hash


def a_config(arm, dataset):
    d = load_protocol()["datasets"][dataset]
    gaussian_bce, pair_view, lambda_pair = {
        "B0": (False, "clean", 0.), "P0": (False, "clean", .05),
        "G0": (True, "clean", 0.), "GP0": (True, "independent", .05),
        "GP-shared": (True, "shared", .05),
    }[arm]
    return dict(arm=arm, dataset=dataset, gaussian_bce=gaussian_bce, pair_view=pair_view,
                pair_count=PAIR_COUNT, lambda_pair=lambda_pair, tau=1., lambda_aug=.5,
                target_domain=0, kappa=d["kappa"], rho=d["rho"], weight_decay=d["weight_decay"],
                lr=.01, epochs=50, batch_size=40960)


def make_training_parts(model, job, bundle):
    sampler = MacroPairSampler(bundle.dataset, bundle.train_x[:, -1], bundle.train_y,
                               bundle.train_row_ids, bundle.identity["split_row_fingerprints"]["train"])
    return make_optimizer(model, job.config), DirectionAAdapter(
        bundle.train_x, sampler, trace_warmup_margin_signs=getattr(job, "stage", None) == "C1")


def forward_logits(model, x, *, track_logits=True, **kwargs):
    captured = [None] * len(model.tower)
    handles = []

    def capture(index):
        def hook(_module, _inputs, output):
            captured[index] = output
        return hook

    try:
        for index, tower in enumerate(model.tower):
            handles.append(tower.register_forward_hook(capture(index)))
        pred = model(x, **kwargs)
        with torch.set_grad_enabled(track_logits and torch.is_grad_enabled()):
            domain = x[-1].long()
            rows = torch.arange(len(domain), device=x.device)
            tower_logits = torch.cat(captured, dim=1)
            logits = torch.stack((tower_logits[rows, domain], tower_logits[rows, 3 + domain]), dim=1)
        return pred, logits
    finally:
        for handle in handles:
            handle.remove()


def _gaussian_view(model, rows, x, context, stream):
    key = view_key(context["dataset"], context["seed"], "train", context["global_step"], stream, 1)
    return views.draw_view(model, rows, "gaussian", key, x.device)


def base_gaussian_view(model, x, config, context):
    if config["gaussian_bce"]:
        rows = int((x[-1] == config["target_domain"]).sum())
        if rows:
            return _gaussian_view(model, rows, x, context, "gaussian")
    return None


def _pair_view(model, x, condition, config, context):
    if condition[0] != config["target_domain"] or config["pair_view"] == "clean":
        return None
    view = _gaussian_view(model, x.shape[1], x, context, "gaussian_pairwise")
    if config["pair_view"] == "shared":
        pos, neg = view["noise"].chunk(2)
        neg.copy_(pos)
        view["hash"] = views.digest_tensor(view["noise"])
    return view


def _view_trace(view):
    return {"key": view["key"], "seed": view["seed"], "rows": view["noise"].shape[0],
            "noise_hash": view["hash"]} if view is not None else None


def _gradient_norms(parameters, raw_pair, coefficient):
    total = [p.grad for p in parameters]

    def component(gradient):
        return torch.linalg.vector_norm(gradient, dtype=torch.float64)

    def norm(components):
        return float(torch.stack(components).norm()) if components else 0.

    total_norm = norm([component(g) for g in total if g is not None])
    if raw_pair is None:
        return dict(pair=0., applied_pair=0., total=total_norm, base=total_norm)
    raw_norms, applied_norms, base_norms = [], [], []
    for g, pair in zip(total, raw_pair):
        if pair is None:
            if g is not None:
                base_norms.append(component(g))
            continue
        applied = coefficient * pair
        raw_norms.append(component(pair))
        applied_norms.append(component(applied))
        base_norms.append(component(-applied if g is None else g - applied))
    return dict(pair=norm(raw_norms), applied_pair=norm(applied_norms), total=total_norm, base=norm(base_norms))


class DirectionAAdapter:
    def __init__(self, train_x, sampler, *, trace_warmup_margin_signs=False):
        self._train_x = train_x
        self.sampler = sampler
        self._trace_warmup_margin_signs = trace_warmup_margin_signs

    def state_dict(self):
        return {"sampler": self.sampler.state_dict()}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError("Direction A sampler identity mismatch")

    def step(self, model, optimizer, ema, x, y, row_ids, config, context, *, select_pair=None, reduce_pair=None):
        start = time.perf_counter()
        coefficient = config["lambda_pair"]
        target = config["target_domain"]
        explicit_views = []
        base_logits = []
        optimizer.zero_grad(set_to_none=True)

        def forward(batch, **kwargs):
            pred, logits = forward_logits(model, batch, track_logits=False, **kwargs)
            base_logits.append(logits.detach().mean())
            return pred

        def after_clean():
            nonlocal pair, margin, pair_loss, pair_view
            pair = self.sampler.sample_macro6(context) if select_pair is None else select_pair(context)
            if pair is not None:
                pair_x = torch.as_tensor(self._train_x[pair["offsets"]], device=x.device).T
                pair_view = _pair_view(model, pair_x, pair["condition"], config, context)
                kwargs = {} if pair_view is None else {"view": pair_view, "kappa": config["kappa"]}
                with torch.set_grad_enabled(bool(coefficient)):
                    _, logits = forward_logits(model, pair_x, **kwargs)
                    task = pair["condition"][1]
                    pos, neg = logits[:, task].chunk(2)
                    margin = pos - neg
                    if reduce_pair is None:
                        pair_loss = (pair["w"] / pair["q"]) * F.softplus(-margin / config["tau"]).mean()
                    else:
                        pair_loss = (pair["w"] / pair["q"]) * reduce_pair(margin, pair, config)
            view = base_gaussian_view(model, x, config, context)
            if view is not None:
                explicit_views.append(view)

        pair = margin = pair_loss = pair_view = None
        bce, source = ga31.losses(
            model, x, y, target, source_config(config["gaussian_bce"]),
            context["dataset"], context["seed"], context["global_step"], mix_lambda=config["lambda_aug"],
            kappa=config["kappa"], explicit_views=explicit_views, forward=forward, after_clean=after_clean,
        )
        loss = bce + coefficient * pair_loss if coefficient and pair_loss is not None else bce
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite Direction A training loss")
        params = [p for p in model.parameters() if p.requires_grad]
        raw_pair = (torch.autograd.grad(pair_loss, params, retain_graph=True, allow_unused=True)
                    if coefficient and pair_loss is not None else None)
        loss.backward()
        norms = _gradient_norms(params, raw_pair, coefficient)
        if not np.isfinite(list(norms.values())).all():
            raise FloatingPointError("Nonfinite Direction A parameter gradient or diagnostic VJP")
        optimizer.step()
        update_ema(ema, model, config["rho"])

        target_mask = source["mask"].detach().cpu().numpy()
        rows = np.asarray(row_ids)
        pair_rows = len(pair["ids"]) if pair is not None else 0
        pair_forwards = int(pair is not None)
        gaussian_base_rows = source["extra_views"]
        gaussian_pair_rows = pair_rows if pair_view is not None else 0
        work = counts(len(y), int(target_mask.sum()))
        work.update(training_forwards=source["forwards"] + pair_forwards,
                    training_forward_rows=len(y) + gaussian_base_rows + pair_rows,
                    extra_view_rows=gaussian_base_rows + gaussian_pair_rows,
                    pair_rows=pair_rows, pair_draws=pair_rows // 2, pair_forwards=pair_forwards,
                    gaussian_bce_forwards=len(explicit_views), gaussian_bce_rows=gaussian_base_rows,
                    gaussian_pair_forwards=int(pair_view is not None), gaussian_pair_rows=gaussian_pair_rows,
                    pair_diagnostic_vjp_calls=int(raw_pair is not None))
        real_loss = float(source["real"].detach().mean(1).mean())
        total_loss = float(loss.detach())
        if pair is None:
            pair_diagnostics = dict(condition=None, w=None, q=None, w_over_q=None, pair_index_hash=None,
                                    unique_pairs=0, unique_endpoints=0, pair_requires_grad=False,
                                    pair_loss=0., applied_pair_loss=0., pair_margin_mean=None)
        else:
            pair_diagnostics = dict(
                condition=pair["condition"], w=pair["w"], q=pair["q"], w_over_q=pair["w"] / pair["q"],
                pair_index_hash=pair["pair_index_hash"], unique_pairs=pair["unique_pairs"],
                unique_endpoints=pair["unique_endpoints"], pair_requires_grad=pair_loss.requires_grad,
                pair_loss=float(pair_loss.detach()), applied_pair_loss=float((coefficient * pair_loss.detach())),
                pair_margin_mean=float(margin.detach().mean()))
        diagnostics = dict(
            context={k: context[k] for k in ("dataset", "seed", "epoch", "global_step")},
            train_identity=self.sampler.train_identity, ordered_bce_row_hash=ordered_ids_hash(rows),
            ordered_d0_row_hash=ordered_ids_hash(rows[target_mask]),
            **pair_diagnostics, bce_loss=float(bce.detach()), total_loss=total_loss, grad_norms=norms,
            base_logit_means=[float(value) for value in base_logits],
            gaussian_bce=_view_trace(explicit_views[0]) if explicit_views else None,
            gaussian_pair=_view_trace(pair_view), step_seconds=time.perf_counter() - start,
            cuda_allocated_bytes=torch.cuda.memory_allocated(x.device) if x.is_cuda else None,
            cuda_reserved_bytes=torch.cuda.memory_reserved(x.device) if x.is_cuda else None,
        )
        if self._trace_warmup_margin_signs and reduce_pair is None and margin is not None:
            signed = margin.detach()
            positive, tied = torch.stack(((signed > 0).sum(), (signed == 0).sum())).tolist()
            diagnostics.update(pair_margin_positive=positive, pair_margin_tied=tied,
                               pair_margin_negative=len(signed) - positive - tied)
        return dict(loss=total_loss, real_loss=real_loss, pred=source["pred"].detach(),
                    diagnostics=diagnostics, counts=work)
