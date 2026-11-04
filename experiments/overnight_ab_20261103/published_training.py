"""One-stage-epoch published trainers with complete in-memory continuation."""
from __future__ import annotations

import hashlib
import random
import time

import numpy as np
import torch

from .evaluation import evaluate_model
from .published_models import build_published, restore_model_meta
from .selection import consider
from .state import capture_rng, cpu_copy, restore_rng, state_digest


def _seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _stage_rng(seed, dataset, method, epoch):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([
        int(seed), ("movielens", "kuairand").index(dataset),
        ("M3OE_FIXED1", "HYPERGATE", "AUTOIFS").index(method), 1, int(epoch)])))


def _adam(parameters, cfg, lr, weight_decay):
    parameters = [p for p in parameters if p.requires_grad]
    if not parameters:
        raise ValueError("Empty published optimizer parameter group")
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay,
                            betas=tuple(cfg["betas"]), eps=cfg["eps"])


def _optimizers(model, method, stage, cfg):
    if method == "HYPERGATE":
        return {"base": _adam(model.parameters(), cfg, cfg["lr"], cfg["weight_decay"])}
    groups = model.optimizer_groups()
    ownership = {name: {id(p) for p in values} for name, values in groups.items()}
    all_trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if ownership["base"] & ownership["gate"] or ownership["base"] | ownership["gate"] != all_trainable:
        raise ValueError("AutoIFS optimizer groups must be disjoint and cover trainable parameters")
    if stage == "search":
        return {
            "base": _adam(groups["base"], cfg, cfg["search_lr"], cfg["search_weight_decay"]),
            "gate": _adam(groups["gate"], cfg, cfg["search_lr"] * cfg["gate_lr_ratio"],
                          cfg["gate_weight_decay"]),
        }
    if groups["gate"]:
        raise ValueError("AutoIFS reuse still has trainable gate parameters")
    return {"base": _adam(groups["base"], cfg, cfg["reuse_lr"], cfg["reuse_weight_decay"])}


def _model_meta(model, method):
    if method != "AUTOIFS":
        return {}
    return {"ticket": bool(model.network.ticket), "temp": float(model.network.temp),
            "gate_frozen": all(not p.requires_grad for name, p in model.network.named_parameters()
                               if "gate" in name)}


def create_session(job, bundle, device):
    method = job["method"].upper()
    if method not in ("HYPERGATE", "AUTOIFS"):
        raise ValueError("Published training supports only the two new published adapters")
    if int(job["seed"]) not in range(2022, 2027):
        raise ValueError("Unregistered experimental seed")
    cfg = job["config"]
    if torch.device(device).type == "cuda": torch.cuda.set_device(device)
    _seed_all(job["seed"])
    model = build_published(method, bundle, job["seed"], cfg).to(device)
    stage = "search" if method == "AUTOIFS" else "train"
    if method == "AUTOIFS":
        model.network.temp = float(cfg["temperature_start"])
    return dict(job=job, bundle=bundle, device=str(device), method=method, model=model,
                optimizers=_optimizers(model, method, stage, cfg), stage=stage,
                next_stage_epoch=0 if method == "AUTOIFS" else 1,
                completed_search=0, completed_final=0, commit_index=0, global_step=0,
                rng=capture_rng(device), rewind_state=None, selector_state=None,
                selection={"final_stage": "reuse" if method == "AUTOIFS" else "train",
                           "best_score": -float("inf"), "selected": None},
                snapshots={"online": {}}, cost={}, complete=False)


def train_epoch(session, epoch_argument):
    """Retain native objective, optimizer calls, PCG64 shuffle and tail batch."""
    job, bundle, model = session["job"], session["bundle"], session["model"]
    cfg, device = job["config"], session["device"]
    order = _stage_rng(job["seed"], job["dataset"], session["method"], epoch_argument).permutation(
        len(bundle.train_x))
    model.train()
    losses, exposures, updates = {}, 0, 0
    for begin in range(0, len(order), cfg["batch_size"]):
        rows = order[begin:begin + cfg["batch_size"]]
        x = torch.as_tensor(bundle.train_x[rows], dtype=torch.float32, device=device)
        y = torch.as_tensor(bundle.train_y[rows], dtype=torch.float32, device=device)
        mask = torch.ones_like(y, dtype=torch.bool)
        for optimizer in session["optimizers"].values():
            optimizer.zero_grad(set_to_none=True)
        components = model.objective(x, y, mask)
        loss = components["loss"]
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("Non-finite or non-scalar published objective")
        loss.backward()
        for optimizer in session["optimizers"].values():
            optimizer.step()
        count = y.numel()
        for name, value in components.items():
            losses[name] = losses.get(name, 0.) + float(value.detach()) * count
        exposures += count
        updates += 1
    if not exposures:
        raise ValueError("Empty natural training population")
    row_ids = np.asarray(bundle.train_row_ids)[order]
    return {"train_objective": {name: value / exposures for name, value in losses.items()},
            "steps": updates, "optimizer_calls": updates * len(session["optimizers"]),
            "sample_exposures": len(order), "label_exposures": exposures,
            "data_order_hash": hashlib.sha256(np.ascontiguousarray(row_ids).tobytes()).hexdigest()}


def _selector_matches(model, snapshot):
    for key in ("gate_hypernet", "gate_head0", "gate_head1"):
        actual = getattr(model.network, key).state_dict()
        expected = snapshot[key]
        if actual.keys() != expected.keys() or any(
                not torch.equal(actual[name].detach().cpu(), expected[name]) for name in actual):
            return False
    embedding = model.network.gate_embedding
    return (torch.equal(embedding.embedding.detach().cpu(), snapshot["gate_embedding.embedding"])
            and torch.equal(embedding.dense_vector.detach().cpu(), snapshot["gate_embedding.dense_vector"]))


def _start_reuse(session):
    job, cfg = session["job"], session["job"]["config"]
    if session["rewind_state"] is None:
        raise RuntimeError("AutoIFS missing state entering final search epoch")
    selector = session["model"].export_search_state()
    model = build_published("AUTOIFS", session["bundle"], job["seed"], cfg).to(session["device"])
    model.load_state_dict(session["rewind_state"])
    model.import_selector_state(selector)
    model.network.ticket, model.network.temp = True, float(cfg["temperature_end"])
    model.freeze_gate()
    if not _selector_matches(model, selector):
        raise AssertionError("AutoIFS final selector transplant differs")
    session.update(model=model, optimizers=_optimizers(model, "AUTOIFS", "reuse", cfg),
                   selector_state=selector, stage="reuse", next_stage_epoch=1)


def advance_epoch(session):
    if session["complete"]:
        raise ValueError("Trajectory already completed its fixed budget")
    restore_rng(session["rng"], session["device"])
    started = time.monotonic()
    job, cfg = session["job"], session["job"]["config"]
    stage, epoch = session["stage"], session["next_stage_epoch"]
    model, method = session["model"], session["method"]
    if stage == "search":
        # Rewind is captured before the last search epoch and its temperature increment.
        if epoch == cfg["rewind_epoch_index"]:
            session["rewind_state"] = cpu_copy(model.state_dict())
        if epoch:
            model.network.temp *= (cfg["temperature_end"] / cfg["temperature_start"]) ** (
                1. / (cfg["search_epochs"] - 1))
        epoch_argument = epoch
    else:
        epoch_argument = epoch + 100 if stage == "reuse" else epoch
    train = train_epoch(session, epoch_argument)
    session["global_step"] += train["steps"]
    validation = evaluate_model(model, session["bundle"], "validation", method, cfg["eval_batch_size"])
    model_hash = state_digest(model.state_dict())
    meta = _model_meta(model, method)
    if stage != "search":
        consider(session["selection"], stage, epoch, validation, model.state_dict(), meta)
        session["snapshots"]["online"]["selected"] = session["selection"]["selected"]
        if epoch in (10, 50):
            session["snapshots"]["online"][f"E{epoch}"] = dict(
                model_state=cpu_copy(model.state_dict()), stage=stage, epoch=epoch,
                model_hash=model_hash, validation=validation, model_meta=meta)
        session["completed_final"] += 1
        session["next_stage_epoch"] = epoch + 1
        session["complete"] = session["completed_final"] == 50
    else:
        session["completed_search"] += 1
        session["next_stage_epoch"] = epoch + 1
        if session["completed_search"] == cfg["search_epochs"]:
            _start_reuse(session)
    session["commit_index"] += 1
    elapsed = time.monotonic() - started
    cost = dict(worker_seconds=elapsed, steps=train["steps"], optimizer_calls=train["optimizer_calls"],
                sample_exposures=train["sample_exposures"], label_exposures=train["label_exposures"],
                peak_cuda_bytes=torch.cuda.max_memory_allocated(session["device"])
                if torch.device(session["device"]).type == "cuda" else 0)
    for key, value in cost.items():
        session["cost"][key] = (max(session["cost"].get(key, 0), value) if key == "peak_cuda_bytes"
                                else session["cost"].get(key, 0) + value)
    session["rng"] = capture_rng(session["device"])
    selected = session["selection"]["selected"]
    return dict(job_id=job["job_id"], stage=stage, epoch=epoch, commit_index=session["commit_index"],
                completed_search=session["completed_search"], completed_final=session["completed_final"],
                global_step=session["global_step"], validation=validation, model_hash=model_hash,
                selection_hash=state_digest({"best_score": session["selection"]["best_score"],
                                            "model_hash": selected["model_hash"] if selected else None}),
                train_objective=train["train_objective"], data_order_hash=train["data_order_hash"],
                model_meta=meta, cost=cost, complete=session["complete"])


def export_session(session):
    keys = ("method", "stage", "next_stage_epoch", "completed_search", "completed_final",
            "commit_index", "global_step", "rng", "selection", "snapshots", "cost", "complete",
            "rewind_state", "selector_state")
    payload = {key: session[key] for key in keys}
    payload.update(schema_version=1, job=session["job"], model_state=session["model"].state_dict(),
                   model_meta=_model_meta(session["model"], session["method"]),
                   optimizer_states={name: opt.state_dict() for name, opt in session["optimizers"].items()})
    return cpu_copy(payload)


def restore_session(job, bundle, device, snapshot):
    for key in ("trajectory_id", "config_hash", "source_hash", "data_hash"):
        if snapshot["job"][key] != job[key]:
            raise ValueError(f"Cannot restore different published {key}")
    session = create_session(job, bundle, device)
    restore_model_meta(session["model"], snapshot["model_meta"])
    session["model"].load_state_dict(snapshot["model_state"])
    session["optimizers"] = _optimizers(session["model"], session["method"], snapshot["stage"], job["config"])
    if set(session["optimizers"]) != set(snapshot["optimizer_states"]):
        raise ValueError("Published optimizer stage differs from RAM snapshot")
    for name, optimizer in session["optimizers"].items():
        optimizer.load_state_dict(snapshot["optimizer_states"][name])
    for key in ("stage", "next_stage_epoch", "completed_search", "completed_final", "commit_index",
                "global_step", "rng", "selection", "snapshots", "cost", "complete", "rewind_state", "selector_state"):
        session[key] = cpu_copy(snapshot[key])
    # Constructing adapters consumes RNG; restore only after the full stage exists.
    restore_rng(session["rng"], session["device"])
    return session

