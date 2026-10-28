"""Mechanical CSV/JSON/Markdown aggregation; seeds are the repeat unit."""
from __future__ import annotations

import csv
import base64
import json
import statistics
from pathlib import Path

import torch

from .config import DATASETS, METHODS, PROTOCOL, SEEDS, load_config


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def formal_run_paths(run_root):
    root = Path(run_root)
    for dataset in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                yield dataset, method, seed, root / "formal" / dataset / method / f"seed{seed}" / "run.json"


def _run_identity(run, run_path):
    run_dir = Path(run_path).resolve().parent
    return {
        "protocol": run.get("protocol"), "phase": run.get("phase"),
        "method": run.get("method"), "dataset": run.get("dataset"), "seed": int(run.get("seed")),
        "run_dir": str(run_dir), "run_manifest": str(Path(run_path).resolve()),
        "selected_checkpoint": str(Path(run["selected_checkpoint"]).resolve()),
        "fixed_final_checkpoint": str(Path(run["fixed_final_checkpoint"]).resolve()),
        "config": run.get("config"), "source_commits": run.get("source_commits"),
        "data_manifest": run.get("data_manifest"), "selection_metric": run.get("selection_metric"),
        "environment": run.get("environment"),
        "implementation_identity": run.get("implementation_identity"),
    }


def _embedded_identity(run):
    fields = ("protocol", "phase", "method", "dataset", "seed", "config", "source_commits",
              "data_manifest", "selection_metric", "environment", "implementation_identity",
              "output_dir", "started_at_utc")
    return {key: run.get(key) for key in fields}


def checkpoint_identity(checkpoint, path):
    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"Checkpoint has no tensor model state: {path}")
    return {
        "path": str(Path(path).resolve()), "stage": checkpoint.get("stage"),
        "epoch": checkpoint.get("epoch"), "selected_epoch": checkpoint.get("selected_epoch"),
        "global_steps": checkpoint.get("global_steps"), "next_epoch": checkpoint.get("next_epoch"),
        "sampler_position": checkpoint.get("sampler_position"),
        "best_metric": checkpoint.get("best_metric"), "model_meta": checkpoint.get("model_meta", {}),
        "optimizer_count": len(checkpoint.get("optimizer_states", [])),
        "model_state_schema": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                               for name, value in sorted(state.items())},
        "embedded_identity": _embedded_identity(checkpoint.get("run_manifest", {})),
    }


def _common_run_identity(run):
    config = run.get("config") or {}
    return {"protocol": run.get("protocol"), "source_commits": run.get("source_commits"),
            "selection_metric": run.get("selection_metric"), "roles": config.get("roles"),
            "metrics": config.get("metrics"), "training": config.get("training")}


def _direct_source_snapshot(implementation_identity):
    """Read every declared snapshot file as bytes and retain a recheckable copy."""
    if not isinstance(implementation_identity, dict) or implementation_identity.get("kind") != "direct_source_snapshot":
        raise ValueError("Formal run has no direct implementation source snapshot")
    root = Path(implementation_identity.get("snapshot_path", "")).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Implementation source snapshot is missing: {root}")
    declared = implementation_identity.get("files")
    if not isinstance(declared, list) or not declared:
        raise ValueError("Implementation source snapshot has no file inventory")
    entries = sorted(declared, key=lambda entry: entry["path"])
    actual_paths = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    if actual_paths != [entry["path"] for entry in entries]:
        raise ValueError(f"Implementation source snapshot inventory differs from its files: {root}")
    contents = {}
    frozen_files = []
    for entry in entries:
        relative = entry["path"]
        content = (root / relative).read_bytes()
        if len(content) != int(entry["size_bytes"]):
            raise ValueError(f"Implementation source snapshot size differs: {root / relative}")
        contents[relative] = content
        frozen_files.append({"path": relative, "size_bytes": len(content),
                             "content_base64": base64.b64encode(content).decode("ascii")})
    return {"kind": "direct_source_snapshot", "snapshot_path": str(root), "files": frozen_files}, contents


def verify_frozen_source_snapshot(implementation_identity, frozen_snapshot):
    try:
        current_snapshot, current_contents = _direct_source_snapshot(implementation_identity)
        expected_contents = {entry["path"]: base64.b64decode(entry["content_base64"], validate=True)
                             for entry in frozen_snapshot["files"]}
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise PermissionError("Frozen implementation source contents are invalid") from exc
    if current_contents != expected_contents:
        raise PermissionError("Current implementation source snapshot differs from the frozen bytes")
    return current_snapshot


def _verify_checkpoint_run(run, checkpoint, run_path):
    embedded = checkpoint.get("run_manifest") or {}
    expected = _embedded_identity(run)
    actual = _embedded_identity(embedded)
    if actual != expected:
        raise ValueError(f"Checkpoint embedded run/config/source/data identity differs: {run_path}")


def build_freeze(run_root, output_path):
    runs = {}
    global_identity = None
    per_dataset_identity = {}
    model_config_by_method = {}
    frozen_source_snapshot = None
    frozen_source_contents = None
    for dataset, method, seed, path in formal_run_paths(run_root):
        key = f"{dataset}/{method}/seed{seed}"
        if not path.is_file():
            raise FileNotFoundError(f"Formal matrix entry missing: {key}")
        run = _read_json(path)
        if run.get("protocol") != PROTOCOL or run.get("phase") != "formal" or run.get("status") != "TRAIN_COMPLETE_NOT_EVALUATED":
            raise ValueError(f"Formal run is not complete and test-locked: {key}")
        if run.get("method") != method or run.get("dataset") != dataset or int(run.get("seed")) != seed:
            raise ValueError(f"Formal run identity mismatch: {key}")
        if Path(run.get("output_dir", "")).resolve() != path.parent.resolve():
            raise ValueError(f"Formal run output directory differs from its matrix location: {key}")
        current_global = _common_run_identity(run)
        current_data = run["data_manifest"]
        method_config = (run.get("config") or {}).get("models", {}).get(method)
        if not isinstance(method_config, dict):
            raise ValueError(f"Formal run has no effective model configuration: {key}")
        if method in model_config_by_method and method_config != model_config_by_method[method]:
            raise ValueError(f"Formal runs have method-specific model configuration drift: {method}")
        model_config_by_method[method] = method_config
        current_snapshot, current_source_contents = _direct_source_snapshot(
            run.get("implementation_identity"))
        if frozen_source_snapshot is None:
            frozen_source_snapshot = current_snapshot
            frozen_source_contents = current_source_contents
        elif current_source_contents != frozen_source_contents:
            raise ValueError("Formal runs have different implementation source snapshot bytes")
        if global_identity is None:
            global_identity = current_global
        elif current_global != global_identity:
            raise ValueError("Formal runs do not share the same frozen config/source/metric identity")
        if dataset in per_dataset_identity and current_data != per_dataset_identity[dataset]:
            raise ValueError(f"Formal runs disagree on prepared {dataset} data identity")
        per_dataset_identity[dataset] = current_data
        identity = _run_identity(run, path)
        checkpoints = {}
        for view, field in (("selected", "selected_checkpoint"), ("fixed_final", "fixed_final_checkpoint")):
            checkpoint_path = Path(run[field]).resolve()
            expected_path = (path.parent / ("selected.pt" if view == "selected" else "fixed_final.pt")).resolve()
            if checkpoint_path != expected_path or not checkpoint_path.is_file():
                raise FileNotFoundError(f"{key} {view} checkpoint is missing or at a different path")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            _verify_checkpoint_run(run, checkpoint, path)
            details = checkpoint_identity(checkpoint, checkpoint_path)
            if details["epoch"] != (run["selected_epoch"] if view == "selected" else run["final_epoch"]):
                raise ValueError(f"{key} {view} checkpoint epoch differs from run manifest")
            recorded_metadata = run.get("checkpoint_metadata", {}).get(view)
            actual_metadata = {field: details[field] for field in ("stage", "epoch", "global_steps", "model_meta")}
            if recorded_metadata != actual_metadata:
                raise ValueError(f"{key} {view} checkpoint metadata differs from run manifest")
            checkpoints[view] = details
        runs[key] = {"status": "COMPLETE", "identity": identity, "checkpoints": checkpoints}
    result = {"protocol": PROTOCOL, "status": "FROZEN", "runs": runs,
              "matrix": {"datasets": list(DATASETS), "methods": list(METHODS), "seeds": list(SEEDS)},
              "identity": {"global": global_identity, "prepared_data_by_dataset": per_dataset_identity,
                           "model_config_by_method": model_config_by_method,
                           "implementation_source_snapshot": frozen_source_snapshot},
              "E_policy": "test role may be evaluated only after this complete 30-run freeze"}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _evaluation_files(run_dir, run_manifest_path):
    found = {}
    for role in ("V", "Q", "E"):
        for view in ("selected", "fixed_final"):
            path = run_dir / f"metrics_{role}_{view}.json"
            if path.is_file():
                found[(role, view)] = path
    index_path = run_dir / "evaluation_index.json"
    if index_path.is_file():
        index = _read_json(index_path)
        for entry in index.get("evaluations", []):
            if entry.get("run_manifest") != str(Path(run_manifest_path).resolve()):
                continue
            path = Path(entry["metrics"])
            if path.is_file():
                found[(entry["role"], entry["view"])] = path
    return found


def _pair_identity(run):
    config = run.get("config") or {}
    return {"protocol": run.get("protocol"), "phase": run.get("phase"),
            "source_commits": run.get("source_commits"),
            "dataset": run.get("dataset"), "data_manifest": run.get("data_manifest"),
            "roles": config.get("roles"), "metrics": config.get("metrics"),
            "training": config.get("training"), "selection_metric": run.get("selection_metric")}


def _rows_from_runs(run_root, *, methods=None, datasets=None, seeds=None):
    rows, costs, statuses = [], [], []
    run_root = Path(run_root)
    methods = set(methods or METHODS)
    datasets = set(datasets or DATASETS)
    seeds = set(int(seed) for seed in (seeds or SEEDS))
    for phase in ("development", "formal"):
        for path in sorted((run_root / phase).glob("*/*/seed*/run.json")):
            run = _read_json(path)
            if (run.get("method") not in methods or run.get("dataset") not in datasets
                    or int(run.get("seed", -1)) not in seeds):
                continue
            eval_files = _evaluation_files(path.parent, path)
            identity = _run_identity(run, path) if all(
                run.get(key) is not None for key in ("selected_checkpoint", "fixed_final_checkpoint")
            ) else {"protocol": run.get("protocol"), "phase": run.get("phase"),
                    "method": run.get("method"), "dataset": run.get("dataset"),
                    "seed": run.get("seed"), "run_manifest": str(path.resolve())}
            statuses.append({"phase": phase, "method": run.get("method"), "dataset": run.get("dataset"),
                             "seed": run.get("seed"), "status": run.get("status"),
                             "run_manifest": str(path.resolve()), "identity": identity,
                             "config": run.get("config"), "source_commits": run.get("source_commits"),
                             "data_manifest": run.get("data_manifest"),
                             "environment": run.get("environment"),
                             "implementation_identity": run.get("implementation_identity"),
                             "checkpoint_metadata": run.get("checkpoint_metadata"),
                             "evaluations": [dict(entry) for entry in _read_json(path.parent / "evaluation_index.json").get("evaluations", [])]
                             if (path.parent / "evaluation_index.json").is_file() else []})
            costs.append({"phase": phase, "method": run.get("method"), "dataset": run.get("dataset"),
                          "seed": run.get("seed"), "training_seconds": run.get("training_seconds"),
                          "total_elapsed_seconds": run.get("total_elapsed_seconds"),
                          "parameter_count": run.get("parameter_count"),
                          "peak_cuda_bytes": run.get("peak_cuda_bytes"),
                          "selected_epoch": run.get("selected_epoch"), "final_epoch": run.get("final_epoch"),
                          "inference_p50_ms": (run.get("inference_timing") or {}).get("p50_ms"),
                          "inference_p95_ms": (run.get("inference_timing") or {}).get("p95_ms"),
                          "inference_batch_size": (run.get("inference_timing") or {}).get("batch_size"),
                          "inference_device": (run.get("inference_timing") or {}).get("device"),
                          "inference_warmup": (run.get("inference_timing") or {}).get("warmup"),
                          "inference_repetitions": (run.get("inference_timing") or {}).get("repetitions"),
                          "environment": json.dumps(run.get("environment"), ensure_ascii=False),
                          "implementation_snapshot": (run.get("implementation_identity") or {}).get("snapshot_path"),
                          "failure_stage": run.get("failure_stage"),
                          "failure_type": run.get("failure_type"),
                          "failure_message": run.get("failure_message"),
                          "status": run.get("status")})
            for (role, view), metric_path in eval_files.items():
                    metrics = _read_json(metric_path)
                    for cell in metrics["cells"]:
                        rows.append({"phase": phase, "method": run["method"], "dataset": run["dataset"],
                                     "seed": int(run["seed"]), "role": role, "view": view,
                                     "domain": cell["domain"], "task": cell["task"], "cell": cell["cell"],
                                     "sample_count": cell["n"], "positive_count": cell["positive"],
                                     "negative_count": cell["negative"], "auc": cell["auc"],
                                     "logloss": cell["logloss"],
                                     "macro_auc": metrics["macro_auc"],
                                     "macro_auc_defined_cells": metrics.get("macro_auc_defined_cells"),
                                     "natural_auc": metrics.get("natural_auc"),
                                     "natural_auc_defined_requests": metrics.get("natural_auc_defined_requests"),
                                     "pooled_auc_task0": metrics.get("pooled_auc_by_task", [None, None])[0],
                                     "pooled_auc_task1": metrics.get("pooled_auc_by_task", [None, None])[1],
                                     "pooled_auc_task0_requests": metrics.get(
                                         "pooled_auc_valid_requests_by_task", [None, None])[0],
                                     "pooled_auc_task1_requests": metrics.get(
                                         "pooled_auc_valid_requests_by_task", [None, None])[1],
                                     "macro_logloss": metrics["macro_logloss"],
                                     "macro_logloss_defined_cells": metrics.get("macro_logloss_defined_cells"),
                                     "natural_logloss": metrics["natural_logloss"],
                                     "valid_requests": metrics.get("valid_requests"),
                                     "__pair_identity": _pair_identity(run),
                                     "run_identity": identity, "run_manifest": str(path.resolve()),
                                     "metric_path": str(metric_path.resolve())})
    return rows, costs, statuses


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_report(run_root, output_dir, *, config=None, methods=None, datasets=None, seeds=None):
    cfg = config or load_config()
    methods = tuple(methods or cfg["methods"])
    datasets = tuple(datasets or cfg["datasets"])
    seeds = tuple(int(seed) for seed in (seeds or cfg["formal_seeds"]))
    if set(methods) - set(cfg["methods"]):
        raise ValueError("Unknown report method filter")
    if set(datasets) - set(cfg["datasets"]):
        raise ValueError("Unknown report dataset filter")
    if set(seeds) - set(cfg["formal_seeds"]):
        raise ValueError("Unknown report seed filter")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, costs, statuses = _rows_from_runs(run_root, methods=methods, datasets=datasets, seeds=seeds)
    model_config_by_method = {}
    for status in statuses:
        method = status["method"]
        config_for_method = (status.get("config") or {}).get("models", {}).get(method)
        key = (status["phase"], method)
        if not isinstance(config_for_method, dict):
            raise ValueError(f"Run has no effective model configuration: {key}")
        if key in model_config_by_method and model_config_by_method[key] != config_for_method:
            raise ValueError(f"Runs have method-specific model configuration drift: {key}")
        model_config_by_method[key] = config_for_method
    per_fields = ["phase", "method", "dataset", "seed", "role", "view", "domain", "task", "cell",
                  "sample_count", "positive_count", "negative_count", "auc", "logloss", "macro_auc",
                  "macro_auc_defined_cells", "natural_auc", "natural_auc_defined_requests",
                  "pooled_auc_task0", "pooled_auc_task1", "pooled_auc_task0_requests",
                  "pooled_auc_task1_requests",
                  "macro_logloss", "macro_logloss_defined_cells", "natural_logloss", "valid_requests"]
    cost_fields = ["phase", "method", "dataset", "seed", "training_seconds", "total_elapsed_seconds",
                   "parameter_count",
                   "peak_cuda_bytes", "selected_epoch", "final_epoch", "inference_p50_ms", "inference_p95_ms",
                   "inference_batch_size", "inference_device", "inference_warmup", "inference_repetitions",
                   "environment", "implementation_snapshot", "failure_stage", "failure_type",
                   "failure_message", "status"]
    _write_csv(output_dir / "per_cell_results.csv", rows, per_fields)
    _write_csv(output_dir / "runtime_cost.csv", costs, cost_fields)

    grouped = {}
    for row in rows:
        key = (row["phase"], row["method"], row["dataset"], row["role"], row["view"])
        grouped.setdefault(key, {}).setdefault(int(row["seed"]), []).append(row)
    aggregates = []
    for (phase, method, dataset, role, view), seed_cells in sorted(grouped.items()):
        seed_values = []
        for seed, cells in sorted(seed_cells.items()):
            if len(cells) != 6:
                continue
            seed_values.append({"seed": seed, "macro_auc": cells[0]["macro_auc"],
                                "macro_auc_defined_cells": cells[0]["macro_auc_defined_cells"],
                                "natural_auc": cells[0]["natural_auc"],
                                "natural_auc_defined_requests": cells[0]["natural_auc_defined_requests"],
                                "pooled_auc_task0": cells[0]["pooled_auc_task0"],
                                "pooled_auc_task1": cells[0]["pooled_auc_task1"],
                                "pooled_auc_task0_requests": cells[0]["pooled_auc_task0_requests"],
                                "pooled_auc_task1_requests": cells[0]["pooled_auc_task1_requests"],
                                "macro_logloss": cells[0]["macro_logloss"],
                                "macro_logloss_defined_cells": cells[0]["macro_logloss_defined_cells"],
                                "natural_logloss": cells[0]["natural_logloss"],
                                "valid_requests": cells[0]["valid_requests"]})
        for metric, count_metric in (("macro_auc", "macro_auc_defined_cells"),
                                     ("natural_auc", "natural_auc_defined_requests"),
                                     ("pooled_auc_task0", "pooled_auc_task0_requests"),
                                     ("pooled_auc_task1", "pooled_auc_task1_requests"),
                                     ("macro_logloss", "macro_logloss_defined_cells"),
                                     ("natural_logloss", "valid_requests")):
            values = [v[metric] for v in seed_values if v[metric] is not None]
            aggregates.append({"phase": phase, "method": method, "dataset": dataset,
                               "role": role, "view": view, "metric": metric,
                               "n_seeds": len(values), "mean": statistics.mean(values) if values else None,
                               "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                               "support_by_seed": ";".join(
                                   f"{v['seed']}:{v[count_metric]}" for v in seed_values),
                               "seeds": ";".join(str(v["seed"]) for v in seed_values)})
    _write_csv(output_dir / "aggregate_results.csv", aggregates,
               ["phase", "method", "dataset", "role", "view", "metric", "n_seeds", "mean", "sample_sd",
                "support_by_seed", "seeds"])

    paired = []
    index = {(r["phase"], r["method"], r["dataset"], r["seed"], r["role"], r["view"], r["domain"], r["task"]): r
             for r in rows}
    for row in rows:
        if row["method"] == "M3OE_FIXED1":
            continue
        key = (row["phase"], "M3OE_FIXED1", row["dataset"], row["seed"], row["role"], row["view"],
               row["domain"], row["task"])
        base = index.get(key)
        if base is None:
            continue
        if row["__pair_identity"] != base["__pair_identity"]:
            raise ValueError("Paired baseline rows have incompatible protocol/data/split/metric identities")
        paired.append({"phase": row["phase"], "method": row["method"], "dataset": row["dataset"],
                       "seed": row["seed"], "role": row["role"], "view": row["view"],
                       "domain": row["domain"], "task": row["task"],
                       "delta_auc": None if row["auc"] is None or base["auc"] is None else row["auc"] - base["auc"],
                       "delta_logloss": None if row["logloss"] is None or base["logloss"] is None else row["logloss"] - base["logloss"]})
    _write_csv(output_dir / "paired_deltas.csv", paired,
               ["phase", "method", "dataset", "seed", "role", "view", "domain", "task", "delta_auc", "delta_logloss"])

    expected = [{"dataset": d, "method": m, "seed": s, "status": "NOT_RUN"}
                for d in datasets for m in methods for s in seeds]
    observed = {(s["dataset"], s["method"], int(s["seed"])): s["status"] for s in statuses
                if s["phase"] == "formal"}
    for item in expected:
        key = (item["dataset"], item["method"], item["seed"])
        if key in observed:
            item["status"] = observed[key]
    (output_dir / "experiment_manifest.json").write_text(json.dumps({
        "protocol": PROTOCOL, "run_status": cfg["run_status"], "formal_seed_matrix": expected,
        "filters": {"methods": methods, "datasets": datasets, "seeds": seeds},
        "observed_runs": statuses, "run_provenance": statuses,
        "record_policy": "Smoke fixtures are not read as empirical runs; only development/formal run manifests are included.",
        "checksum_revalidated": False,
    }, indent=2), encoding="utf-8")

    def metric_for(phase, dataset, method, seed, role, view, metric):
        subset = [row for row in rows if (row["phase"], row["dataset"], row["method"], row["seed"],
                                          row["role"], row["view"]) ==
                  (phase, dataset, method, seed, role, view)]
        if len(subset) != 6:
            return None
        return subset[0].get(metric)

    def fmt(value):
        return "" if value is None else f"{float(value):.6f}"

    def value_table(title, phase, role, view, matrix):
        lines = [f"## {title}", "", "| Dataset | Method | Seed | Status | Macro AUC | Macro LogLoss | Natural AUC | Natural LogLoss |",
                 "|---|---|---:|---|---:|---:|---:|---:|"]
        status_map = {(r["phase"], r["dataset"], r["method"], int(r["seed"])): r["status"]
                      for r in statuses}
        for dataset in datasets:
            for method in methods:
                for seed in matrix:
                    status = status_map.get((phase, dataset, method, seed), "NOT_RUN")
                    vals = [metric_for(phase, dataset, method, seed, role, view, metric)
                            for metric in ("macro_auc", "macro_logloss", "natural_auc", "natural_logloss")]
                    if any(value is not None for value in vals):
                        status = "EVALUATED"
                    lines.append(f"| {dataset} | {method} | {seed} | {status} | " +
                                 " | ".join(fmt(value) for value in vals) + " |")
        return lines

    def summary_table():
        lines = ["## Formal selected E seed summaries", "",
                 "Mean and sample SD use the seed as the unit; incomplete seeds are excluded.", "",
                 "| Dataset | Method | Metric | n seeds | Mean | Sample SD |", "|---|---|---|---:|---:|---:|"]
        for dataset in datasets:
            for method in methods:
                for metric in ("macro_auc", "macro_logloss", "natural_auc", "natural_logloss"):
                    values = [metric_for("formal", dataset, method, seed, "E", "selected", metric)
                              for seed in seeds]
                    values = [float(value) for value in values if value is not None]
                    mean = statistics.mean(values) if values else None
                    sd = statistics.stdev(values) if len(values) > 1 else None
                    lines.append(f"| {dataset} | {method} | {metric} | {len(values)} | {fmt(mean)} | {fmt(sd)} |")
        return lines

    comparison = ["# MDMT baseline comparison", "",
                  "Numerical entries come only from explicit evaluation files. Smoke fixtures are excluded.",
                  "HyperGate settings marked UNVERIFIED_PROPOSAL remain disclosed.", ""]
    comparison += value_table("Formal selected E", "formal", "E", "selected", seeds)
    comparison += [""] + summary_table() + [""]
    comparison += value_table("Formal fixed-final E", "formal", "E", "fixed_final", seeds)
    comparison += [""] + value_table("Development Q selected (not formal evidence)", "development", "Q",
                                        "selected", tuple(s for s in seeds if s in cfg["development_seeds"]))
    comparison += ["", "## Custom method reserved", "",
                   "| Method | Status | Macro AUC | Macro LogLoss |", "|---|---|---:|---:|",
                   "| CUSTOM_METHOD_TBD | NOT_RUN |  |  |"]
    (output_dir / "baseline_comparison.md").write_text("\n".join(comparison) + "\n", encoding="utf-8")
    return {"per_cell_rows": len(rows), "aggregate_rows": len(aggregates), "paired_rows": len(paired),
            "formal_statuses": expected}

