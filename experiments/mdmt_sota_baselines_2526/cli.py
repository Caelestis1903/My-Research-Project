"""audit, prepare-data, smoke, train, evaluate, freeze and report entry points."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from .config import DATASETS, METHODS, ROOT, load_config
from .data import Data, prepare_owned, tensor
from .evaluation import evaluate_role, load_checkpoint
from .report import (build_freeze, build_report, checkpoint_identity, _run_identity)
from .training import _fit_autoifs, _metric_summary, _model_config, _source_roots, seed_all
from .models import build_model


def _write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _source_inventory(config, methods=None):
    manifest_path = ROOT / "research_reports/mdmt_baselines_2526/SOURCE_MANIFEST.json"
    audit_path = ROOT / "research_reports/mdmt_baselines_2526/SOURCE_AUDIT.md"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    static_audit = audit_path.read_text(encoding="utf-8")
    paths = {
        "RecStudio HyperGate implementation": Path(config["source_roots"]["recstudio"]).expanduser().resolve(),
        "AutoIFS": Path(config["source_roots"]["autoifs"]).expanduser().resolve(),
        "M3OE_FIXED1 initialization helper": ROOT,
    }
    rows = []
    owners = {"HyperGate parent repository": {"HYPERGATE"},
              "RecStudio HyperGate implementation": {"HYPERGATE"},
              "AutoIFS": {"AUTOIFS"},
              "M3OE_FIXED1 initialization helper": {"M3OE_FIXED1"}}
    for source in manifest["sources"]:
        name = source["name"]
        if methods is not None and not owners[name] & set(methods):
            continue
        root = (Path(source["source_root"]).expanduser().resolve()
                if name == "HyperGate parent repository" else paths[name])
        expected = source.get("commit") or source.get("reference_commit")
        commit = None
        commit_error = None
        if root.is_dir() and source.get("commit"):
            result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=False)
            commit = result.stdout.strip() if result.returncode == 0 else None
            commit_error = result.stderr.strip() if result.returncode else None
        license_path = root / "LICENSE"
        license_status = "PRESENT" if license_path.is_file() else "ABSENT"
        components = source.get("components", [])
        component_status = []
        for component in components:
            relative = component["path"]
            check_path = root / relative
            component_status.append({"path": relative, "exists": check_path.is_file(),
                                     "git_blob_id": component.get("git_blob_id")})
        if name == "M3OE_FIXED1 initialization helper":
            relative = source["path"].split(":", 1)[0]
            component_status.append({"path": relative, "exists": (root / relative).is_file(),
                                     "git_blob_id": source.get("git_blob_id")})
        if not root.is_dir():
            status = "MISSING"
        elif source.get("commit") and commit is None:
            status = "COMMIT_UNAVAILABLE"
        elif source.get("commit") and commit != expected:
            status = "COMMIT_MISMATCH"
        elif source.get("reference_commit"):
            status = "REFERENCE_ONLY"
        else:
            status = "PASS"
        rows.append({"name": name, "repository": source.get("repository"),
                     "path": str(root), "exists": root.is_dir(), "expected_commit": expected,
                     "actual_commit": commit, "commit_error": commit_error, "status": status,
                     "license": source.get("license"), "license_file": str(license_path),
                     "license_file_status": license_status,
                     "runtime_loaded": source.get("runtime_loaded", False),
                     "components": component_status,
                     "adapter_changes": source.get("adapter_changes", []),
                     "gitlink": source.get("gitlink"), "source_evidence": source.get("source_evidence"),
                     "use": source.get("use"), "delivery": source.get("delivery")})
    return {"manifest_path": str(manifest_path), "identity_policy": manifest.get("identity_policy"),
            "configuration_provenance": manifest.get("configuration_provenance"),
            "inventory": rows, "static_findings_path": str(audit_path),
            "static_findings": static_audit}


def audit(config, *, methods=None, datasets=None, seeds=None):
    methods = tuple(methods or config["methods"])
    datasets = tuple(datasets or config["datasets"])
    seeds = tuple(int(seed) for seed in (seeds or config["formal_seeds"]))
    if set(methods) - set(config["methods"]):
        raise ValueError("Unknown audit method filter")
    if set(datasets) - set(config["datasets"]):
        raise ValueError("Unknown audit dataset filter")
    if set(seeds) - set(config["formal_seeds"]):
        raise ValueError("Unknown audit seed filter")
    raw = {}
    for dataset in datasets:
        path = config["raw_paths"][dataset]
        item = Path(path)
        raw[dataset] = {"path": str(item), "exists": item.is_file(),
                        "bytes": item.stat().st_size if item.is_file() else None}
    historical_reference = {}
    for dataset in datasets:
        relative = config["historical_prepared"][dataset]
        root = ROOT / relative
        historical_reference[dataset] = {
            name: (root / name).is_file() for name in ("DATA_MANIFEST.json", "PREPROCESSOR.json")
        }
    owned_arrays = {}
    for dataset in datasets:
        root = ROOT / config["prepared_root"] / dataset
        expected = ["DATA_MANIFEST.json", "PREPROCESSOR.json", "x.npy", "y.npy",
                    "rows_fit.npy", "rows_P_label.npy", "rows_validation.npy", "rows_test.npy"]
        owned_arrays[dataset] = {name: (root / name).is_file() for name in expected}
    sources = _source_inventory(config, methods)
    gpu = {"nvidia_smi_available": shutil.which("nvidia-smi") is not None,
           "snapshot": None}
    if gpu["nvidia_smi_available"]:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                 "--format=csv,noheader"], capture_output=True, text=True, check=False)
        gpu["snapshot"] = {"returncode": result.returncode, "stdout": result.stdout.strip(),
                          "stderr": result.stderr.strip()}
    return {
        "protocol": config["protocol"], "status": config["run_status"], "checksum_revalidated": False,
        "real_training_started": False, "datasets": raw,
        "historical_reference_json": historical_reference, "owned_prepared_arrays": owned_arrays,
        "official_sources": sources, "gpu": gpu,
        "matrix": {"methods": list(methods), "datasets": list(datasets),
                   "formal_seeds": list(seeds),
                   "formal_trajectories": len(methods) * len(datasets) * len(seeds)},
        "filters": {"methods": methods, "datasets": datasets, "seeds": seeds},
        "hypergate_configuration": config["models"]["HYPERGATE"]["configuration_provenance"],
    }


def smoke(config, output=None, *, method=None, dataset="movielens", seed=2022):
    if method is not None and method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    methods = (method,) if method is not None else METHODS
    seed_all(seed)
    schema = {"dataset": dataset, "columns": ["user_id", "movie_id", "domain_indicator"],
              "dense_columns": [], "sparse_columns": ["user_id", "movie_id", "domain_indicator"],
              "vocab_sizes": {"user_id": 5, "movie_id": 7, "domain_indicator": 3}, "embedding_dim": 16,
              "domain_column": "domain_indicator", "oov_index": 0}
    x_np = np.asarray([[i % 5, i % 7, i % 3] for i in range(12)], dtype=np.float32)
    y_np = np.asarray([[i % 2, (i // 2) % 2] for i in range(12)], dtype=np.float32)
    work = Path(output) if output else Path(tempfile.mkdtemp(prefix="mdmt_baseline_smoke_"))
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"Smoke output is not empty: {work}")
    work.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    results = {}
    for method_name in methods:
        model = build_model(method_name, schema, _model_config(config, method_name, dataset, seed),
                            _source_roots(config)).to(device)
        model.train()
        x = tensor(x_np, device)
        y = tensor(y_np, device)
        mask = torch.ones_like(y, dtype=torch.bool)
        mask[0, 0] = False
        objective = model.objective(x, y, mask)
        if objective["loss"].ndim != 0 or not torch.isfinite(objective["loss"]):
            raise AssertionError(f"{method_name} did not return a finite scalar objective")
        objective["loss"].backward()
        model.eval()
        with torch.inference_mode():
            first = model(x, domain=x[:, -1].long())
            second = model(x, domain=x[:, -1].long())
        if tuple(first.shape) != (len(x_np), 2) or not torch.isfinite(first).all():
            raise AssertionError(f"{method_name} output shape or values are invalid")
        if not torch.equal(first, second):
            raise AssertionError(f"{method_name} eval inference is not deterministic")
        clone = build_model(method_name, schema, _model_config(config, method_name, dataset, seed),
                            _source_roots(config)).to(device)
        clone.load_state_dict(model.state_dict())
        clone.eval()
        with torch.inference_mode():
            restored = clone(x, domain=x[:, -1].long())
        if not torch.equal(first, restored):
            raise AssertionError(f"{method_name} inference checkpoint roundtrip differs")
        results[method_name] = {"synthetic_only": True, "logit_shape": list(first.shape),
                                "finite_loss": True, "backward": "PASS",
                                "deterministic_inference": "PASS", "checkpoint_roundtrip": "PASS"}

    # Exercise the real AutoIFS stage-transition trainer with synthetic rows,
    # shortening only the smoke-local search/reuse epochs.
    tiny = copy.deepcopy(config)
    tiny["models"]["AUTOIFS"]["search_epochs"] = 2
    tiny["models"]["AUTOIFS"]["rewind_epoch_index"] = 1
    tiny["models"]["AUTOIFS"]["reuse_max_epochs"] = 2
    if "AUTOIFS" in methods:
        tiny["models"]["AUTOIFS"]["batch_size"] = 12
        tiny["models"]["AUTOIFS"]["lambda"] = [0.0, 0.0]
    class _SyntheticData:
        def inputs(self, role, row_ids=None):
            ids = self.roles[role] if row_ids is None else np.asarray(row_ids, dtype=np.int64)
            return np.asarray(self.x[ids]), ids.copy()
        def labels(self, role, purpose, row_ids=None):
            allowed = {"T0": "train", "F": "train", "V": "select"}
            if allowed.get(role) != purpose:
                raise PermissionError("Synthetic role denial")
            ids = self.roles[role] if row_ids is None else np.asarray(row_ids, dtype=np.int64)
            return np.asarray(self.y[ids]), np.ones((len(ids), 2), dtype=bool), ids.copy()
        def training_batch(self, row_ids):
            ids = np.asarray(row_ids, dtype=np.int64)
            if not np.isin(ids, self.train_ids).all():
                raise PermissionError("Synthetic train role denial")
            return self.x[ids], self.y[ids], np.ones((len(ids), 2), dtype=bool)
    synthetic = _SyntheticData()
    synthetic.dataset, synthetic.schema = dataset, schema
    synthetic.x, synthetic.y = x_np, y_np
    synthetic.domain = x_np[:, -1].astype(np.int64)
    synthetic.roles = {"T0": np.arange(0, 6), "F": np.arange(6, 12), "V": np.arange(12)}
    synthetic.train_ids = np.arange(12)
    synthetic.manifest = {"status": "SYNTHETIC_FIXTURE", "checksum_revalidated": False}
    if "AUTOIFS" in methods:
        auto = build_model("AUTOIFS", schema, _model_config(tiny, "AUTOIFS", dataset, seed),
                           _source_roots(tiny)).to(device)
        output_dir = work / "synthetic_autoifs_transition"
        output_dir.mkdir()
        identity = {"protocol": config["protocol"], "method": "AUTOIFS", "dataset": dataset,
                    "seed": seed, "phase": "smoke", "output_dir": str(output_dir)}
        selected, final, stage_log, steps = _fit_autoifs(
            auto, synthetic, dataset=dataset, seed=seed, phase="development", config=tiny,
            run_manifest=identity, device=device, log_path=output_dir / "synthetic_epochs.jsonl")
        if selected["model_meta"].get("ticket") is not True or final["stage"] != "reuse":
            raise AssertionError("AutoIFS synthetic search→rewind→prune→reuse did not complete")
        results["AUTOIFS"]["search_rewind_prune_reuse"] = "PASS"
        results["AUTOIFS"]["synthetic_updates"] = steps
    result = {"protocol": config["protocol"], "status": "CPU_SYNTHETIC_SMOKE_PASS",
              "synthetic_only": True, "metrics_are_not_research_results": True,
              "dataset": dataset, "seed": int(seed), "methods": list(methods),
              "output_dir": str(work.resolve()),
              "models": results}
    _write_json(result, work / "smoke_report.json")
    return result


def evaluate_checkpoint(run_dir, role, view, config, freeze_file=None, device="cpu", *,
                        output_dir=None, expected_method=None, expected_dataset=None, expected_seed=None):
    run_dir = Path(run_dir)
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("protocol") != config["protocol"] or run.get("status") != "TRAIN_COMPLETE_NOT_EVALUATED":
        raise ValueError("A completed, test-locked run manifest is required")
    method, dataset, seed = run["method"], run["dataset"], int(run["seed"])
    if expected_method is not None and expected_method != method:
        raise ValueError("--method does not match the run directory")
    if expected_dataset is not None and expected_dataset != dataset:
        raise ValueError("--dataset does not match the run directory")
    if expected_seed is not None and int(expected_seed) != seed:
        raise ValueError("--seed does not match the run directory")
    if run.get("phase") not in ("development", "formal"):
        raise ValueError("Run phase is invalid")
    if Path(run.get("output_dir", "")).resolve() != run_dir.resolve():
        raise ValueError("Run output directory does not match --run-dir")
    if run.get("config") != config or run.get("source_commits") != config["source_commits"]:
        raise ValueError("Run configuration or pinned source identity differs from the active config")
    if role == "E" and not freeze_file:
        raise PermissionError("E evaluation requires --freeze-manifest for the full formal matrix")
    frozen = json.loads(Path(freeze_file).read_text(encoding="utf-8")) if freeze_file else None
    if frozen is not None:
        key = f"{dataset}/{method}/seed{seed}"
        if frozen.get("runs", {}).get(key, {}).get("status") != "COMPLETE":
            raise PermissionError("This run is not part of the frozen formal matrix")
    if role == "E" and run.get("phase") != "formal":
        raise PermissionError("Development and trial trajectories cannot be evaluated on E")
    data = Data(dataset, ROOT / config["prepared_root"])
    if run.get("data_manifest") != data.manifest:
        raise ValueError("Prepared data identity differs from the completed run")
    checkpoint_name = "selected.pt" if view == "selected" else "fixed_final.pt"
    checkpoint_path = run_dir / checkpoint_name
    declared_path = run.get("selected_checkpoint" if view == "selected" else "fixed_final_checkpoint")
    if not declared_path or Path(declared_path).resolve() != checkpoint_path.resolve():
        raise ValueError("Run manifest checkpoint path does not match the selected run directory")
    model = build_model(method, data.schema, _model_config(config, method, dataset, seed),
                        _source_roots(config)).to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, expected_method=method,
                                 expected_dataset=dataset, expected_seed=seed, expected_config=config)
    if checkpoint["run_manifest"].get("data_manifest") != data.manifest:
        raise ValueError("Checkpoint prepared data identity differs from the active data")
    identity = _run_identity(run, run_path)
    checkpoint_details = checkpoint_identity(checkpoint, checkpoint_path)
    if role == "E":
        if frozen is None:
            raise PermissionError("E evaluation requires --freeze-manifest for the complete formal matrix")
        data.authorize_final(frozen, method, seed, phase=run["phase"], run_identity=identity,
                             view=view, checkpoint_identity=checkpoint_details)
    destination = Path(output_dir) if output_dir is not None else run_dir
    result = evaluate_role(model, data, role, method=method, seed=seed, device=device,
                           batch_size=config["models"][method].get("eval_batch_size",
                           config["models"][method].get("batch_size", 4096)),
                           output_dir=destination, final_manifest=frozen, checkpoint_view=view,
                           phase=run["phase"], run_identity=identity,
                           checkpoint_identity=checkpoint_details)
    if destination.resolve() != run_dir.resolve():
        index_path = run_dir / "evaluation_index.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"evaluations": []}
        index["evaluations"] = [entry for entry in index["evaluations"]
                                if (entry["role"], entry["view"]) != (role, view)]
        index["evaluations"].append({"phase": run["phase"], "method": method, "dataset": dataset,
                                     "seed": seed, "role": role, "view": view,
                                     "run_manifest": str(run_path.resolve()),
                                     "checkpoint": str(checkpoint_path.resolve()),
                                     "metrics": str((destination / f"metrics_{role}_{view}.json").resolve()),
                                     "predictions": str((destination / f"predictions_{role}_{view}.csv").resolve())})
        _write_json(index, index_path)
    return result


def make_parser():
    parser = argparse.ArgumentParser(prog="mdmt-sota-baselines")
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--output", type=Path, default=None)
    audit.add_argument("--method", nargs="+", choices=METHODS, default=None)
    audit.add_argument("--dataset", nargs="+", choices=DATASETS, default=None)
    audit.add_argument("--seed", nargs="+", type=int, default=None)
    prep = commands.add_parser("prepare-data")
    prep.add_argument("--dataset", choices=("movielens", "kuairand"), required=True)
    prep.add_argument("--output-root", type=Path, default=None)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output", type=Path, default=None)
    smoke.add_argument("--method", choices=METHODS, default=None)
    smoke.add_argument("--dataset", choices=DATASETS, default="movielens")
    smoke.add_argument("--seed", type=int, default=2022)
    train = commands.add_parser("train")
    train.add_argument("--method", choices=METHODS, required=True)
    train.add_argument("--dataset", choices=("movielens", "kuairand"), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--phase", choices=("development", "formal"), default="development")
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--output", type=Path, default=None)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--run-dir", type=Path, required=True)
    ev.add_argument("--role", choices=("V", "Q", "E"), required=True)
    ev.add_argument("--view", choices=("selected", "fixed_final"), default="selected")
    ev.add_argument("--freeze-manifest", type=Path, default=None)
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--output-dir", type=Path, default=None)
    ev.add_argument("--method", choices=METHODS, default=None)
    ev.add_argument("--dataset", choices=DATASETS, default=None)
    ev.add_argument("--seed", type=int, default=None)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--run-root", type=Path, default=None)
    freeze.add_argument("--output", type=Path, default=None)
    report = commands.add_parser("report")
    report.add_argument("--run-root", type=Path, default=None)
    report.add_argument("--output", type=Path, default=None)
    report.add_argument("--method", nargs="+", choices=METHODS, default=None)
    report.add_argument("--dataset", nargs="+", choices=DATASETS, default=None)
    report.add_argument("--seed", nargs="+", type=int, default=None)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "audit":
        result = audit(config, methods=args.method, datasets=args.dataset, seeds=args.seed)
        if args.output is not None:
            _write_json(result, args.output)
    elif args.command == "prepare-data":
        root = args.output_root or (ROOT / config["prepared_root"])
        result = prepare_owned(args.dataset, root / args.dataset, config=config)
    elif args.command == "smoke":
        result = smoke(config, args.output, method=args.method, dataset=args.dataset, seed=args.seed)
    elif args.command == "train":
        default_root = ROOT / "artifacts/mdmt_baselines_2526/runs"
        phase_root = "development" if args.phase == "development" else "formal"
        output = args.output or default_root / phase_root / args.dataset / args.method / f"seed{args.seed}"
        from .training import train_one
        result = train_one(args.method, args.dataset, args.seed, output, phase=args.phase,
                           config=config, device=args.device)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(args.run_dir, args.role, args.view, config,
                                     args.freeze_manifest, args.device, output_dir=args.output_dir,
                                     expected_method=args.method, expected_dataset=args.dataset,
                                     expected_seed=args.seed)
    elif args.command == "freeze":
        result = build_freeze(args.run_root or ROOT / "artifacts/mdmt_baselines_2526/runs",
                              args.output or ROOT / "artifacts/mdmt_baselines_2526/final_freeze.json")
    else:
        result = build_report(args.run_root or ROOT / "artifacts/mdmt_baselines_2526/runs",
                              args.output or ROOT / "research_reports/mdmt_baselines_2526", config=config,
                              methods=args.method, datasets=args.dataset, seeds=args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0

