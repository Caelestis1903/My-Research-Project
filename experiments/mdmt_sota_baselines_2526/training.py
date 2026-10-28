"""One-trajectory trainers. This module is never invoked by smoke or audit."""
from __future__ import annotations

import copy
import datetime
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import METHODS, PROTOCOL, ROOT, load_config
from .data import Data, tensor
from .evaluation import evaluate_role, measure_inference
from .models import build_model


def seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def capture_rng():
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _stage_rng(seed, dataset, method, phase, epoch):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([
        int(seed), ("movielens", "kuairand").index(dataset), METHODS.index(method),
        ("development", "formal").index(phase), int(epoch)])))


def _source_roots(config):
    return {name: Path(path) for name, path in config["source_roots"].items()}


def _model_config(config, method, dataset, seed):
    values = dict(config["models"][method])
    values.update(dataset=dataset, seed=int(seed))
    return values


def _objective(model, x, y, mask):
    result = model.objective(x, y, mask)
    if not isinstance(result, dict) or "loss" not in result:
        raise TypeError("Model objective must return a dict containing scalar loss")
    if result["loss"].ndim != 0 or not torch.isfinite(result["loss"]):
        raise FloatingPointError("Non-finite or non-scalar model objective")
    return result


def _make_optimizer(parameters, lr, weight_decay, betas, eps):
    params = [p for p in parameters if p.requires_grad]
    if not params:
        raise ValueError("Optimizer parameter group is empty")
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=tuple(betas), eps=eps)


def _eval(model, data, role, method, seed, device, eval_batch):
    return evaluate_role(model, data, role, method=method, seed=seed, device=device,
                         batch_size=eval_batch, output_dir=None)


def _train_epoch(model, data, ids, *, method, dataset, seed, phase, epoch, batch_size,
                 device, optimizers, loss_log, temperature=None):
    order = _stage_rng(seed, dataset, method, phase, epoch).permutation(ids)
    model.train()
    weighted_loss = 0.0
    exposures = 0
    steps = 0
    for begin in range(0, len(order), batch_size):
        rows = order[begin:begin + batch_size]
        x_np, y_np, mask_np = data.training_batch(rows)
        x = tensor(x_np, device)
        y = tensor(y_np, device)
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        components = _objective(model, x, y, mask)
        components["loss"].backward()
        for optimizer in optimizers:
            optimizer.step()
        weighted_loss += float(components["loss"].detach()) * int(mask.sum().item())
        exposures += int(mask.sum().item())
        steps += 1
        for name, value in components.items():
            if name == "loss":
                continue
            loss_log[name] = loss_log.get(name, 0.0) + float(value.detach()) * int(mask.sum().item())
    return {"train_loss": weighted_loss / max(exposures, 1), "valid_label_exposures": exposures,
            "updates": steps,
            "components": {name: value / max(exposures, 1) for name, value in loss_log.items()},
            "temperature": temperature}


def _checkpoint(model, optimizers, *, run_manifest, epoch, stage, global_steps,
                best_metric, selected_epoch, rewind_state=None, selector_state=None):
    network = getattr(model, "network", None)
    return {
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_states": [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers],
        "scheduler_state": None, "epoch": int(epoch), "stage": stage,
        "global_steps": int(global_steps), "next_epoch": int(epoch) + 1,
        "sampler_position": 0, "rng_state": capture_rng(),
        "model_meta": ({"ticket": bool(network.ticket), "temp": float(network.temp)}
                       if network is not None and hasattr(network, "ticket") else {}),
        "best_metric": best_metric, "selected_epoch": selected_epoch,
        "rewind_state": copy.deepcopy(rewind_state),
        "selector_state": copy.deepcopy(selector_state),
        "run_manifest": copy.deepcopy(run_manifest),
    }


def _write_checkpoint(path, checkpoint):
    torch.save(checkpoint, path)


def _training_manifest(config, data, method, dataset, seed, phase, output_dir):
    return {
        "protocol": PROTOCOL, "status": "TRAINING", "phase": phase,
        "method": method, "dataset": dataset, "seed": int(seed),
        "config": config, "source_commits": config["source_commits"],
        "data_manifest": data.manifest,
        "selection_metric": config["metrics"]["checkpoint_selection"],
        "output_dir": str(Path(output_dir).resolve()),
        "e_policy": "locked_during_training",
    }


def _fit_standard(model, data, *, method, dataset, seed, phase, config, run_manifest, device, log_path):
    cfg = config["models"][method]
    training_pool = data.train_ids
    if len(training_pool) != len(data.roles["T0"]) + len(data.roles["F"]):
        raise AssertionError("T0 ∪ F training rows are not disjoint")
    optimizers = [_make_optimizer(model.parameters(), cfg["lr"], cfg["weight_decay"],
                                 cfg["betas"], cfg["eps"])]
    if method == "M3OE_FIXED1":
        epoch_limit = cfg["epochs"]
        min_epoch = cfg["selection_start_epoch"]
        patience = 0
    else:
        epoch_limit = cfg["max_epochs"]
        min_epoch = 1
        patience = cfg["patience"]
    best = float("inf")
    bad_epochs = 0
    selected = None
    final = None
    total_steps = 0
    epochs_log = []
    for epoch in range(1, epoch_limit + 1):
        started = time.monotonic()
        logs = {}
        train_stats = _train_epoch(model, data, training_pool, method=method, dataset=dataset,
                                   seed=seed, phase=phase, epoch=epoch,
                                   batch_size=cfg["batch_size"], device=device,
                                   optimizers=optimizers, loss_log=logs)
        total_steps += train_stats["updates"]
        validation = _eval(model, data, "V", method, seed, device, cfg.get("eval_batch_size", 4096))
        duration = time.monotonic() - started
        epoch_record = {"stage": "train", "epoch": epoch, **train_stats,
                        "validation": _metric_summary(validation), "seconds": duration}
        epochs_log.append(epoch_record)
        _append_record(log_path, epoch_record)
        score = validation["macro_logloss"]
        if epoch >= min_epoch and score is not None and score < best - config["metrics"]["tie_tolerance"]:
            best = score
            selected = _checkpoint(model, optimizers, run_manifest=run_manifest, epoch=epoch,
                                   stage="train", global_steps=total_steps, best_metric=best,
                                   selected_epoch=epoch)
            bad_epochs = 0
        elif epoch >= min_epoch:
            bad_epochs += 1
        final = _checkpoint(model, optimizers, run_manifest=run_manifest, epoch=epoch,
                            stage="train", global_steps=total_steps, best_metric=best,
                            selected_epoch=selected["epoch"] if selected else None)
        if selected is None:
            continue
        if patience and bad_epochs >= patience:
            break
    if selected is None or final is None:
        raise RuntimeError("No validation checkpoint was produced")
    return selected, final, epochs_log, total_steps


def _fit_autoifs(model, data, *, dataset, seed, phase, config, run_manifest, device, log_path):
    cfg = config["models"]["AUTOIFS"]
    ids = data.train_ids
    groups = model.optimizer_groups()
    if set(groups) != {"base", "gate"} or {id(p) for p in groups["base"]} & {id(p) for p in groups["gate"]}:
        raise ValueError("AutoIFS optimizer groups must be named base/gate and disjoint")
    all_params = {id(p) for p in model.parameters() if p.requires_grad}
    if {id(p) for p in groups["base"]} | {id(p) for p in groups["gate"]} != all_params:
        raise ValueError("AutoIFS optimizer groups must cover every trainable parameter")
    parameter_names = {id(parameter): name for name, parameter in model.network.named_parameters()}
    ownership = {name: [parameter_names[id(parameter)] for parameter in parameters]
                 for name, parameters in groups.items()}
    setup = {"stage": "search_setup", "optimizer_parameter_ownership": ownership,
             "base_parameter_count": sum(parameter.numel() for parameter in groups["base"]),
             "gate_parameter_count": sum(parameter.numel() for parameter in groups["gate"])}
    log = [setup]
    _append_record(log_path, setup)
    search_base = _make_optimizer(groups["base"], cfg["search_lr"], cfg["search_weight_decay"],
                                 config["models"]["M3OE_FIXED1"]["betas"], config["models"]["M3OE_FIXED1"]["eps"])
    search_gate = _make_optimizer(groups["gate"], cfg["search_lr"] * cfg["gate_lr_ratio"], 0.0,
                                 config["models"]["M3OE_FIXED1"]["betas"], config["models"]["M3OE_FIXED1"]["eps"])
    search_opts = [search_base, search_gate]
    rewind = None
    total_steps = 0
    temperature_multiplier = ((cfg["temperature_end"] / cfg["temperature_start"]) **
                              (1.0 / (cfg["search_epochs"] - 1)) if cfg["search_epochs"] > 1 else 1.0)
    for epoch_index in range(cfg["search_epochs"]):
        if epoch_index == 0:
            model.network.temp = float(cfg["temperature_start"])
        if epoch_index == cfg["rewind_epoch_index"]:
            # The rewind snapshot is the state entering the final search epoch.
            rewind = copy.deepcopy(model.state_dict())
            temperature_before_epoch = float(model.network.temp)
            rewind_path = Path(run_manifest["output_dir"]) / f"rewind_epoch{epoch_index}.pt"
            _write_checkpoint(rewind_path, {
                "model_state": rewind, "epoch_index": epoch_index,
                "optimizer_states": [copy.deepcopy(optimizer.state_dict()) for optimizer in search_opts],
                "global_steps": total_steps, "next_epoch": epoch_index, "sampler_position": 0,
                "temperature": temperature_before_epoch,
                "stage": "search_rewind_start", "rng_state": capture_rng(),
                "run_manifest": run_manifest,
            })
            rewind_record = {"stage": "search_rewind", "rewind_epoch_index": epoch_index,
                             "temperature_before_final_search_epoch": temperature_before_epoch,
                             "checkpoint": str(rewind_path.resolve())}
            log.append(rewind_record)
            _append_record(log_path, rewind_record)
        if epoch_index > 0:
            model.network.temp = float(model.network.temp) * temperature_multiplier
        temperature = float(model.network.temp)
        started = time.monotonic()
        train_stats = _train_epoch(model, data, ids, method="AUTOIFS", dataset=dataset, seed=seed,
                                   phase=phase, epoch=epoch_index, batch_size=cfg["batch_size"],
                                   device=device, optimizers=search_opts, loss_log={})
        total_steps += train_stats["updates"]
        validation = _eval(model, data, "V", "AUTOIFS", seed, device, cfg["batch_size"])
        record = {"stage": "search", "epoch": epoch_index, **train_stats,
                  "temperature": temperature, "validation": _metric_summary(validation),
                  "seconds": time.monotonic() - started}
        log.append(record)
        _append_record(log_path, record)
    if rewind is None:
        raise RuntimeError("AutoIFS search did not capture its required rewind state")
    selector_state = model.export_search_state()
    rewind_model = build_model("AUTOIFS", data.schema,
                               _model_config(config, "AUTOIFS", dataset, seed), _source_roots(config)).to(device)
    rewind_model.load_state_dict(rewind)
    rewind_model.import_selector_state(selector_state)
    rewind_model.network.ticket = True
    rewind_model.network.temp = float(cfg["temperature_end"])
    rewind_model.freeze_gate()
    selector_equal = _selector_snapshot_matches(rewind_model, selector_state)
    if not selector_equal:
        raise AssertionError("AutoIFS final selector/gate embedding transplant differs")
    reuse_groups = rewind_model.optimizer_groups()
    if any(p.requires_grad for p in reuse_groups["gate"]):
        raise ValueError("AutoIFS selector and gate embedding must be frozen after pruning")
    if any(not p.requires_grad for p in reuse_groups["base"]):
        raise ValueError("AutoIFS reuse must train all non-gate parameters")
    reuse_optimizer = _make_optimizer(reuse_groups["base"], cfg["reuse_lr"], cfg["reuse_weight_decay"],
                                      config["models"]["M3OE_FIXED1"]["betas"],
                                      config["models"]["M3OE_FIXED1"]["eps"])
    reuse_names = {id(parameter): name for name, parameter in rewind_model.network.named_parameters()}
    reuse_setup = {"stage": "reuse_setup", "rewind_epoch_index": cfg["rewind_epoch_index"],
                   "fresh_model_from_rewind": True, "fresh_base_optimizer": True,
                   "search_optimizer_state_reused": False, "selector_transplant_equal": selector_equal,
                   "ticket": bool(rewind_model.network.ticket), "temperature": float(rewind_model.network.temp),
                   "gate_frozen": all(not p.requires_grad for name, p in rewind_model.network.named_parameters()
                                       if "gate" in name),
                   "optimizer_parameter_ownership": {
                       name: [reuse_names[id(parameter)] for parameter in parameters]
                       for name, parameters in reuse_groups.items()}}
    log.append(reuse_setup)
    _append_record(log_path, reuse_setup)
    best = float("inf")
    selected = None
    final = None
    bad = 0
    selected_epoch = None
    for epoch in range(1, cfg["reuse_max_epochs"] + 1):
        started = time.monotonic()
        train_stats = _train_epoch(rewind_model, data, ids, method="AUTOIFS", dataset=dataset,
                                   seed=seed, phase=phase, epoch=epoch + 100,
                                   batch_size=cfg["batch_size"], device=device,
                                   optimizers=[reuse_optimizer], loss_log={})
        total_steps += train_stats["updates"]
        validation = _eval(rewind_model, data, "V", "AUTOIFS", seed, device, cfg["batch_size"])
        record = {"stage": "reuse", "epoch": epoch, **train_stats,
                  "temperature": float(rewind_model.network.temp), "ticket": True,
                  "validation": _metric_summary(validation), "seconds": time.monotonic() - started}
        log.append(record)
        _append_record(log_path, record)
        score = validation["macro_logloss"]
        if score is not None and score < best - config["metrics"]["tie_tolerance"]:
            best = score
            selected_epoch = epoch
            selected = _checkpoint(rewind_model, [reuse_optimizer], run_manifest=run_manifest,
                                   epoch=epoch, stage="reuse", global_steps=total_steps,
                                   best_metric=best, selected_epoch=epoch, rewind_state=rewind,
                                   selector_state=selector_state)
            bad = 0
        else:
            bad += 1
        final = _checkpoint(rewind_model, [reuse_optimizer], run_manifest=run_manifest,
                             epoch=epoch, stage="reuse", global_steps=total_steps,
                             best_metric=best, selected_epoch=selected_epoch,
                             rewind_state=rewind, selector_state=selector_state)
        if selected is not None and bad >= cfg["reuse_patience"]:
            break
    if selected is None or final is None:
        raise RuntimeError("AutoIFS reuse produced no selected checkpoint")
    pruning = _pruning_summary(rewind_model, data, ids, device, cfg["batch_size"])
    pruning_record = {"stage": "reuse_pruning_summary", "source": "T0_union_F",
                      "selector_dimensions": 4, "cells": pruning}
    log.append(pruning_record)
    _append_record(log_path, pruning_record)
    return selected, final, log, total_steps


def _metric_summary(metrics):
    return {key: metrics[key] for key in ("cells", "macro_auc", "macro_logloss", "natural_auc",
                                           "natural_auc_defined_requests", "macro_auc_defined_cells",
                                           "macro_logloss_defined_cells", "natural_logloss",
                                           "pooled_auc_by_task", "pooled_auc_valid_requests_by_task",
                                           "valid_requests")}


def _selector_snapshot_matches(model, snapshot):
    for key in ("gate_hypernet", "gate_head0", "gate_head1"):
        current = getattr(model.network, key).state_dict()
        expected = snapshot[key]
        if current.keys() != expected.keys() or any(
                not torch.equal(current[name].detach().cpu(), expected[name].detach().cpu())
                for name in current):
            return False
    gate_embedding = model.network.gate_embedding.state_dict()
    return (torch.equal(gate_embedding["embedding"].detach().cpu(), snapshot["gate_embedding.embedding"])
            and ("dense_vector" not in gate_embedding or torch.equal(
                gate_embedding["dense_vector"].detach().cpu(), snapshot["gate_embedding.dense_vector"])))


def _pruning_summary(model, data, ids, device, batch_size):
    counts = np.zeros((3, 2), dtype=np.int64)
    removed = np.zeros((3, 2), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(ids), int(batch_size)):
            rows = ids[begin:begin + int(batch_size)]
            x = tensor(data.x[rows], device)
            model(x, domain=x[:, -1].long())
            row_domains = data.domain[rows]
            for domain in range(3):
                chosen = row_domains == domain
                n = int(chosen.sum())
                if n == 0:
                    continue
                for task in range(2):
                    gates = model.network.weights[task][chosen]
                    gate_counts = _selector_removed_retained(gates)
                    counts[domain, task] += gate_counts["total"]
                    removed[domain, task] += gate_counts["removed"]
    return [{"domain": domain, "task": task, "rows": int(counts[domain, task] // 4),
             "gate_decisions": int(counts[domain, task]),
             "removed": int(removed[domain, task]),
             "retained": int(counts[domain, task] - removed[domain, task]),
             "pruned_fraction": (float(removed[domain, task]) /
                                 int(counts[domain, task]) if counts[domain, task] else None)}
            for domain in range(3) for task in range(2)]


def _selector_removed_retained(gate_logits):
    total = int(gate_logits.numel())
    removed = int((gate_logits > 0).sum().item())
    return {"removed": removed, "retained": total - removed, "total": total,
            "pruned_fraction": removed / total if total else None}


def _append_record(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_run_manifest(run_manifest, output_dir):
    (Path(output_dir) / "run.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _snapshot_implementation(output_dir):
    package = Path(__file__).resolve().parent
    target = Path(output_dir) / "implementation_snapshot"
    sources = sorted(package.glob("*.py")) + [package / "default_config.json"]
    sources += sorted((package / "reference_data").rglob("*.json"))
    sources += [package / "reference_data" / "README.md"]
    files = []
    for source in sources:
        if not source.is_file():
            continue
        relative = source.relative_to(package)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files.append({"path": relative.as_posix(), "size_bytes": destination.stat().st_size})
    return {"kind": "direct_source_snapshot", "snapshot_path": str(target.resolve()), "files": files}


def _environment_record(requested_device):
    return {"python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "torch": torch.__version__, "numpy": np.__version__,
            "pandas": importlib.metadata.version("pandas"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "cuda_available": bool(torch.cuda.is_available()),
            "tf32_matmul": torch.backends.cuda.matmul.fp32_precision,
            "tf32_cudnn_conv": torch.backends.cudnn.conv.fp32_precision,
            "requested_device": str(requested_device)}


def train_one(method, dataset, seed, output_dir, *, phase="development", config=None, device="cuda:0"):
    config = config or load_config()
    if config.get("smoke_only"):
        raise ValueError("A smoke-only configuration cannot be used for real training")
    if method not in METHODS or dataset not in config["datasets"]:
        raise ValueError("Unknown model or dataset")
    if phase not in ("development", "formal"):
        raise ValueError("Training phase must be development or formal")
    permitted = config["development_seeds"] if phase == "development" else config["formal_seeds"]
    if int(seed) not in permitted:
        raise ValueError("Seed is outside the configured phase")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_manifest = {"protocol": PROTOCOL, "status": "STARTED", "phase": phase,
                    "method": method, "dataset": dataset, "seed": int(seed),
                    "config": config, "source_commits": config["source_commits"],
                    "selection_metric": config["metrics"]["checkpoint_selection"],
                    "output_dir": str(output_dir.resolve()), "started_at_utc": started_utc,
                    "e_policy": "locked_during_training",
                    "requested_device": str(device)}
    _write_run_manifest(run_manifest, output_dir)
    failure_stage = "random_seed_setup"
    try:
        seed_all(seed)
        failure_stage = "environment_capture"
        run_manifest["environment"] = _environment_record(device)
        failure_stage = "implementation_snapshot"
        run_manifest["implementation_identity"] = _snapshot_implementation(output_dir)
        _write_run_manifest(run_manifest, output_dir)

        failure_stage = "data_load"
        data = Data(dataset, ROOT / config["prepared_root"])
        run_manifest.update(_training_manifest(config, data, method, dataset, seed, phase, output_dir))
        run_manifest.update(started_at_utc=started_utc,
                            environment=run_manifest["environment"],
                            implementation_identity=run_manifest["implementation_identity"],
                            requested_device=str(device), status="TRAINING")

        failure_stage = "device_setup"
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no GPU is visible")
        if device.type == "cuda":
            run_manifest["environment"]["device_name"] = torch.cuda.get_device_name(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            run_manifest["environment"]["device_name"] = platform.processor() or platform.machine()
        run_manifest["environment"]["device"] = str(device)
        _write_run_manifest(run_manifest, output_dir)

        failure_stage = "model_build"
        model = build_model(method, data.schema, _model_config(config, method, dataset, seed),
                            _source_roots(config)).to(device)
        log_path = output_dir / "epoch_metrics.jsonl"
        training_started = time.monotonic()
        failure_stage = "training"
        if method == "AUTOIFS":
            selected, final, epochs, steps = _fit_autoifs(
                model, data, dataset=dataset, seed=seed, phase=phase, config=config,
                run_manifest=run_manifest, device=device, log_path=log_path)
        else:
            selected, final, epochs, steps = _fit_standard(
                model, data, method=method, dataset=dataset, seed=seed, phase=phase,
                config=config, run_manifest=run_manifest, device=device, log_path=log_path)
        training_elapsed = time.monotonic() - training_started

        failure_stage = "checkpoint_save"
        selected_path = output_dir / "selected.pt"
        final_path = output_dir / "fixed_final.pt"
        _write_checkpoint(selected_path, selected)
        _write_checkpoint(final_path, final)
        model.load_state_dict(selected["model_state"])
        if method == "AUTOIFS":
            model.network.ticket = bool(selected["model_meta"]["ticket"])
            model.network.temp = float(selected["model_meta"]["temp"])
        eval_batch = config["models"][method].get(
            "eval_batch_size", config["models"][method].get("batch_size", 4096))
        failure_stage = "selected_validation"
        validation_selected = _eval(model, data, "V", method, seed, device, eval_batch)
        final_model = build_model(method, data.schema, _model_config(config, method, dataset, seed),
                                  _source_roots(config)).to(device)
        final_model.load_state_dict(final["model_state"])
        if method == "AUTOIFS":
            final_model.network.ticket = bool(final["model_meta"]["ticket"])
            final_model.network.temp = float(final["model_meta"]["temp"])
        failure_stage = "fixed_final_validation"
        validation_final = _eval(final_model, data, "V", method, seed, device, eval_batch)
        failure_stage = "inference_timing"
        timing_batch = min(256, len(data.roles["V"]))
        timing_x = data.x[data.roles["V"][:timing_batch]]
        timing = measure_inference(model, timing_x, device=device, warmup=5, repetitions=25)
        run_manifest.update({
            "status": "TRAIN_COMPLETE_NOT_EVALUATED",
            "selected_checkpoint": str(selected_path.resolve()),
            "fixed_final_checkpoint": str(final_path.resolve()),
            "selected_epoch": selected["epoch"], "final_epoch": final["epoch"],
            "checkpoint_metadata": {
                "selected": {"stage": selected["stage"], "epoch": selected["epoch"],
                             "global_steps": selected["global_steps"], "model_meta": selected["model_meta"]},
                "fixed_final": {"stage": final["stage"], "epoch": final["epoch"],
                                "global_steps": final["global_steps"], "model_meta": final["model_meta"]}},
            "selected_validation": _metric_summary(validation_selected),
            "final_validation": _metric_summary(validation_final),
            "training_seconds": training_elapsed, "total_elapsed_seconds": time.monotonic() - started,
            "global_steps": steps, "parameter_count": sum(p.numel() for p in model.parameters()),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "inference_timing": {"checkpoint_view": "selected", **timing}, "epochs": epochs,
        })
        _write_run_manifest(run_manifest, output_dir)
        return run_manifest
    except Exception as exc:
        run_manifest.update({"status": "FAILED", "failure_stage": failure_stage,
                            "failure_type": type(exc).__name__, "failure_message": str(exc),
                            "failed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "total_elapsed_seconds": time.monotonic() - started,
                            "partial_files": sorted(path.name for path in output_dir.iterdir())})
        _write_run_manifest(run_manifest, output_dir)
        raise

