import csv
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.mdmt_sota_baselines_2526 import data as data_module
from experiments.mdmt_sota_baselines_2526 import cli as cli_module
from experiments.mdmt_sota_baselines_2526.config import METHODS, PROTOCOL, SEEDS, load_config
from experiments.mdmt_sota_baselines_2526.data import Data, prepare_owned
from experiments.mdmt_sota_baselines_2526.evaluation import evaluate_role
from experiments.mdmt_sota_baselines_2526 import report as report_module
from experiments.mdmt_sota_baselines_2526.report import (
    RUNTIME_IO_AMENDMENT_ID, _evaluation_files, _package_source_contents, build_freeze, build_report,
)
from experiments.mdmt_sota_baselines_2526.training import train_one, measure_inference, _snapshot_implementation
from experiments.mdmt_sota_baselines_2526.training import (
    _checkpoint, _restore_rng, _selector_removed_retained,
)


def _source_fixture(tmp_path, monkeypatch):
    domains = np.tile(np.array([0, 1, 2], dtype=np.int64), 40)
    categories = np.asarray([f"v{i % 7}" for i in range(len(domains))])
    frame = pd.DataFrame({"category": categories})
    labels = np.column_stack((np.arange(len(domains)) % 2,
                              (np.arange(len(domains)) // 2) % 2)).astype(np.float32)
    monkeypatch.setattr(data_module, "_raw_schema",
                        lambda dataset, path: (domains, ("category", "domain_indicator"), ()))
    monkeypatch.setattr(data_module, "_chunks",
                        lambda dataset, path, size=250000: iter(((frame, labels),)))
    cfg = load_config()
    cfg["raw_paths"]["movielens"] = str(tmp_path / "raw.csv")
    cfg["historical_prepared"]["movielens"] = str(tmp_path / "history")
    raw = tmp_path / "raw.csv"
    raw.write_text("source fixture; never a real dataset", encoding="utf-8")
    history = tmp_path / "history"
    history.mkdir()
    roles = data_module.assign_roles(domains)
    fit = np.zeros(len(domains), dtype=bool)
    fit[roles["fit"]] = True
    vocab = {"category": {value: i + 1 for i, value in enumerate(sorted(set(categories[fit])))}}
    counts = {}
    for name, ids in roles.items():
        counts[name] = {"n": len(ids)}
    n_features = {"category": len(vocab["category"]) + 1, "domain_indicator": 3}
    (history / "PREPROCESSOR.json").write_text(json.dumps({"vocabulary": vocab, "scaler": {}}), encoding="utf-8")
    (history / "DATA_MANIFEST.json").write_text(json.dumps({
        "raw_sha256": "registered-provenance-only", "features": {
            "columns": ["category", "domain_indicator"], "dense_columns": [],
            "sparse_columns": ["category", "domain_indicator"], "vocab_sizes": n_features,
            "embedding_dim": 16, "encoder_hash": "old-recorded-value", "scaler_hash": "old-recorded-value",
        }, "roles": counts,
    }), encoding="utf-8")
    return cfg, domains, labels


def _synthetic_formal_matrix(run_root, data_manifests, config=None, source_snapshots=None):
    cfg = config or load_config()
    paths = {}
    for dataset in cfg["datasets"]:
        for method in cfg["methods"]:
            for seed in cfg["formal_seeds"]:
                run_dir = run_root / "formal" / dataset / method / f"seed{seed}"
                run_dir.mkdir(parents=True)
                run_config = copy.deepcopy(cfg)
                if source_snapshots is None:
                    config_key = "lr" if "lr" in run_config["models"][method] else "search_lr"
                    run_config["models"][method][config_key] += (cfg["methods"].index(method) + 1) * 1e-5
                snapshot_dir = run_dir / "implementation_snapshot"
                snapshot_dir.mkdir(parents=True)
                if source_snapshots is None:
                    revision = None
                    snapshot_contents = {"pipeline.py": b"pipeline-source-v1"}
                    (snapshot_dir / "pipeline.py").write_bytes(snapshot_contents["pipeline.py"])
                else:
                    revision_index = (cfg["datasets"].index(dataset) + cfg["methods"].index(method)
                                      + cfg["formal_seeds"].index(seed)) % 2
                    revision = ("R1", "R2")[revision_index]
                    snapshot_contents = source_snapshots[revision]
                    for relative, content in snapshot_contents.items():
                        target = snapshot_dir / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(content)
                implementation_identity = {
                    "kind": "direct_source_snapshot", "snapshot_path": str(snapshot_dir.resolve()),
                    "files": [{"path": relative, "size_bytes": len(content)}
                              for relative, content in sorted(snapshot_contents.items())],
                }
                started = f"2026-10-28T00:00:{seed % 60:02d}+00:00"
                embedded = {
                    "protocol": PROTOCOL, "phase": "formal", "method": method,
                    "dataset": dataset, "seed": seed, "config": run_config,
                    "source_commits": run_config["source_commits"],
                    "data_manifest": data_manifests[dataset],
                    "selection_metric": run_config["metrics"]["checkpoint_selection"],
                    "environment": {"python": "synthetic-3.13"},
                    "implementation_identity": implementation_identity,
                    "output_dir": str(run_dir.resolve()), "started_at_utc": started,
                }
                selected_path = run_dir / "selected.pt"
                final_path = run_dir / "fixed_final.pt"
                run = {**embedded, "status": "TRAIN_COMPLETE_NOT_EVALUATED",
                       "run_dir": str(run_dir.resolve()),
                       "selected_checkpoint": str(selected_path.resolve()),
                       "fixed_final_checkpoint": str(final_path.resolve()),
                       "selected_epoch": 1, "final_epoch": 2,
                       "checkpoint_metadata": {
                           "selected": {"stage": "train", "epoch": 1, "global_steps": 2, "model_meta": {}},
                           "fixed_final": {"stage": "train", "epoch": 2, "global_steps": 4, "model_meta": {}}}}
                (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
                state = {"weight": torch.tensor([float(seed), float(cfg["methods"].index(method))])}
                for path, epoch, selected_epoch in ((selected_path, 1, 1), (final_path, 2, 1)):
                    torch.save({"model_state": state, "optimizer_states": [{"fixture": True}],
                                "epoch": epoch, "stage": "train", "global_steps": epoch * 2,
                                "next_epoch": epoch + 1, "sampler_position": 0,
                                "best_metric": 0.5, "selected_epoch": selected_epoch,
                                "model_meta": {}, "run_manifest": embedded}, path)
                paths[(dataset, method, seed)] = run_dir
    return paths


def test_config_locks_five_seed_protocol_and_pending_status():
    cfg = load_config()
    assert cfg["protocol"] == PROTOCOL
    assert tuple(cfg["formal_seeds"]) == SEEDS == (2022, 2023, 2024, 2025, 2026)
    assert tuple(cfg["methods"]) == METHODS
    assert cfg["run_status"] == "PENDING_APPROVAL"
    assert cfg["audit"]["checksum_revalidated"] is False
    assert all(cfg["models"]["HYPERGATE"]["gates"].values())
    for dataset in cfg["datasets"]:
        root = data_module.ROOT / cfg["historical_prepared"][dataset]
        assert (root / "PREPROCESSOR.json").is_file()
        assert (root / "DATA_MANIFEST.json").is_file()


def test_prepare_owned_rebuilds_directly_and_roles_guard_training_and_test(tmp_path, monkeypatch):
    cfg, domains, labels = _source_fixture(tmp_path, monkeypatch)
    manifest = prepare_owned("movielens", tmp_path / "owned" / "movielens", config=cfg)
    assert manifest["checksum_revalidated"] is False
    assert manifest["checks"]["labels_against_source_formula"] == "PASS_DIRECT_VALUES"
    assert manifest["checks"]["role_rows_disjoint_exhaustive_and_counts"] == "PASS_DIRECT_VALUES"
    data = Data("movielens", tmp_path / "owned")
    assert np.array_equal(data.domain, domains)
    assert np.array_equal(data.y, labels)
    assert len(data.train_ids) == len(data.roles["T0"]) + len(data.roles["F"])
    x, y, mask = data.training_batch(data.train_ids[::9])
    assert x.shape[0] == y.shape[0] == mask.shape[0]
    with pytest.raises(PermissionError):
        data.training_batch(data.roles["V"][:1])
    with pytest.raises(PermissionError):
        data.labels("V", "train")
    with pytest.raises(PermissionError):
        data.labels("E", "final")
    with pytest.raises(PermissionError):
        data.inputs("E")
    with pytest.raises(PermissionError):
        data.labels("S", "transaction")
    with pytest.raises(FileExistsError):
        prepare_owned("movielens", tmp_path / "owned" / "movielens", config=cfg)


def test_report_renders_formal_values_provenance_and_paired_shared_identity(tmp_path):
    run_root = tmp_path / "runs"
    data_manifest = {"dataset": "movielens", "protocol": PROTOCOL,
                     "status": "READY_DIRECT_VALUE_CHECKS", "split": "synthetic"}
    cfg = load_config()
    for method, offset in (("M3OE_FIXED1", 0.0), ("HYPERGATE", 0.01)):
        for phase, seed in (("development", 2022), ("development", 2023),
                            ("formal", 2022), ("formal", 2023)):
            directory = run_root / phase / "movielens" / method / f"seed{seed}"
            directory.mkdir(parents=True)
            run_config = copy.deepcopy(cfg)
            # Model-specific values differ; paired comparison checks shared protocol/data rules only.
            run_config["models"][method]["lr"] += offset
            run = {"protocol": PROTOCOL, "phase": phase, "method": method, "dataset": "movielens",
                   "seed": seed, "status": "TRAIN_COMPLETE_NOT_EVALUATED", "training_seconds": 2.0,
                   "parameter_count": 5, "peak_cuda_bytes": None, "selected_epoch": 2, "final_epoch": 3,
                   "config": run_config, "source_commits": cfg["source_commits"],
                   "data_manifest": data_manifest, "selection_metric": cfg["metrics"]["checkpoint_selection"],
                   "selected_checkpoint": str((directory / "selected.pt").resolve()),
                   "fixed_final_checkpoint": str((directory / "fixed_final.pt").resolve()),
                   "environment": {"python": "synthetic-3.13"},
                   "implementation_identity": {"kind": "synthetic-direct-source-fixture"},
                   "checkpoint_metadata": {"selected": {"epoch": 2}, "fixed_final": {"epoch": 3}},
                   "inference_timing": {"p50_ms": 1.0, "p95_ms": 1.5, "batch_size": 4,
                                        "device": "cpu", "warmup": 1, "repetitions": 3}}
            (directory / "run.json").write_text(json.dumps(run), encoding="utf-8")
            role, views = (("V", ("selected",)) if phase == "development"
                           else ("E", ("selected", "fixed_final")))
            for view in views:
                cells = []
                for task in range(2):
                    for domain in range(3):
                        cells.append({"cell": task * 3 + domain, "domain": domain, "task": task,
                                      "n": 10, "positive": 5, "negative": 5,
                                      "auc": 0.6 + offset + (seed - 2022) * 0.01,
                                      "logloss": 0.5 - offset, "sample_count": 10, "positive_count": 5})
                metrics = {"cells": cells, "macro_auc": np.mean([c["auc"] for c in cells]),
                           "macro_auc_defined_cells": 6, "natural_auc": 0.6 + offset,
                           "natural_auc_defined_requests": 60, "pooled_auc_by_task": [0.7, 0.8],
                           "pooled_auc_valid_requests_by_task": [40, 40],
                           "macro_logloss": 0.5 - offset, "macro_logloss_defined_cells": 6,
                           "natural_logloss": 0.5 - offset, "valid_requests": 60}
                (directory / f"metrics_{role}_{view}.json").write_text(json.dumps(metrics), encoding="utf-8")
    out = tmp_path / "report"
    result = build_report(run_root, out)
    assert result["per_cell_rows"] == 72
    with (out / "paired_deltas.csv").open(encoding="utf-8", newline="") as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 36
    assert {row["seed"] for row in paired} == {"2022", "2023"}
    manifest = json.loads((out / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["formal_seed_matrix"]) == 30
    assert sum(row["status"] == "TRAIN_COMPLETE_NOT_EVALUATED" for row in manifest["formal_seed_matrix"]) == 4
    provenance = manifest["run_provenance"][0]
    assert provenance["data_manifest"] == data_manifest
    assert provenance["source_commits"] == cfg["source_commits"]
    assert provenance["environment"]["python"] == "synthetic-3.13"
    assert provenance["checkpoint_metadata"]["selected"]["epoch"] == 2
    markdown = (out / "baseline_comparison.md").read_text(encoding="utf-8")
    assert "| movielens | M3OE_FIXED1 | 2022 | EVALUATED | 0.600000 | 0.500000" in markdown
    assert "| movielens | HYPERGATE | 2022 | EVALUATED | 0.610000 | 0.490000" in markdown
    assert "0.605000" in markdown and "0.007071" in markdown
    assert "CUSTOM_METHOD_TBD | NOT_RUN" in markdown
    aggregate = (out / "aggregate_results.csv").read_text(encoding="utf-8")
    assert "natural_auc" in aggregate and "support_by_seed" in aggregate
    runtime = (out / "runtime_cost.csv").read_text(encoding="utf-8")
    assert "inference_p50_ms" in runtime and "synthetic-3.13" in runtime

    filtered = tmp_path / "filtered"
    filtered_result = build_report(run_root, filtered, methods=["HYPERGATE"],
                                   datasets=["movielens"], seeds=[2023])
    filtered_manifest = json.loads((filtered / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert len(filtered_result["formal_statuses"]) == 1
    assert filtered_manifest["filters"] == {"methods": ["HYPERGATE"], "datasets": ["movielens"], "seeds": [2023]}

    incompatible_path = run_root / "formal" / "movielens" / "HYPERGATE" / "seed2022" / "run.json"
    incompatible = json.loads(incompatible_path.read_text(encoding="utf-8"))
    incompatible["data_manifest"] = {"dataset": "movielens", "split": "different"}
    incompatible_path.write_text(json.dumps(incompatible), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible protocol/data/split/metric"):
        build_report(run_root, tmp_path / "incompatible", methods=["M3OE_FIXED1", "HYPERGATE"],
                     datasets=["movielens"], seeds=[2022])

    incompatible["data_manifest"] = data_manifest
    incompatible["source_commits"] = {**incompatible["source_commits"], "recstudio": "different-commit"}
    incompatible_path.write_text(json.dumps(incompatible), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible protocol/data/split/metric"):
        build_report(run_root, tmp_path / "incompatible_source", methods=["M3OE_FIXED1", "HYPERGATE"],
                     datasets=["movielens"], seeds=[2022])

    incompatible["source_commits"] = cfg["source_commits"]
    incompatible_path.write_text(json.dumps(incompatible), encoding="utf-8")
    drift_path = run_root / "formal" / "movielens" / "HYPERGATE" / "seed2023" / "run.json"
    drifted = json.loads(drift_path.read_text(encoding="utf-8"))
    drifted["config"]["models"]["HYPERGATE"]["lambda_domain"] = 0.999
    drift_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="model configuration drift"):
        build_report(run_root, tmp_path / "config_drift", methods=["HYPERGATE"],
                     datasets=["movielens"], seeds=[2022, 2023])


def test_evaluation_reports_weighted_natural_auc_and_defined_support():
    domains = np.asarray([0, 0, 0, 1, 1, 2])
    labels = np.asarray([[0, 0], [1, 0], [0, 0], [1, 0], [0, 0], [0, 0]], dtype=np.float32)
    mask = np.zeros_like(labels, dtype=bool)
    mask[:, 0] = True

    class SyntheticData:
        dataset = "movielens"
        roles = {"V": np.arange(6)}
        x = np.column_stack((np.arange(6), domains)).astype(np.float32)
        domain = domains

        def labels(self, role, purpose):
            assert (role, purpose) == ("V", "select")
            return labels, mask, np.arange(6)

    class FixedModel(torch.nn.Module):
        def forward(self, x, domain=None):
            fixed = torch.tensor([[0, 0], [2, 0], [1, 0], [0, 0], [1, 0], [10, 0]], device=x.device)
            return fixed[x[:, 0].long()]

    result = evaluate_role(FixedModel(), SyntheticData(), "V", method="M3OE_FIXED1", seed=2022)
    assert result["macro_auc"] == pytest.approx(0.5)
    assert result["natural_auc"] == pytest.approx(0.6)
    assert result["natural_auc_defined_requests"] == 5
    assert result["macro_auc_defined_cells"] == 2
    assert result["macro_logloss_defined_cells"] == 3


def test_checkpoint_restore_replays_the_next_optimizer_step():
    torch.manual_seed(44)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(0.4), torch.nn.Linear(5, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x, target = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 10, torch.ones((6, 2))

    def step(network, opt):
        opt.zero_grad(set_to_none=True)
        loss = (network(x) - target).square().mean()
        loss.backward()
        opt.step()

    step(model, optimizer)
    identity = {"protocol": PROTOCOL, "dataset": "synthetic", "seed": 2022,
                "config": {"synthetic": True}, "data_manifest": {"status": "SYNTHETIC_FIXTURE"},
                "source_commits": {"synthetic": "fixture"}}
    checkpoint = _checkpoint(model, [optimizer], run_manifest=identity, epoch=2, stage="train",
                             global_steps=7, best_metric=0.5, selected_epoch=2)
    assert checkpoint["next_epoch"] == 3 and checkpoint["sampler_position"] == 0

    _restore_rng(checkpoint["rng_state"])
    step(model, optimizer)
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())

    restored = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(0.4), torch.nn.Linear(5, 2))
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
    restored.load_state_dict(checkpoint["model_state"])
    restored_optimizer.load_state_dict(checkpoint["optimizer_states"][0])
    _restore_rng(checkpoint["rng_state"])
    step(restored, restored_optimizer)
    for name, value in expected_model.items():
        assert torch.equal(value, restored.state_dict()[name])

    def same_tree(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                same_tree(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                same_tree(a, b)
        else:
            assert left == right

    same_tree(expected_optimizer, restored_optimizer.state_dict())


def test_final_freeze_binds_exact_formal_run_and_both_checkpoint_files(tmp_path):
    manifests = {dataset: {"dataset": dataset, "protocol": PROTOCOL,
                           "status": "READY_DIRECT_VALUE_CHECKS", "split": "synthetic"}
                 for dataset in ("movielens", "kuairand")}
    run_root = tmp_path / "runs"
    paths = _synthetic_formal_matrix(run_root, manifests)
    freeze_path = tmp_path / "freeze.json"
    frozen = build_freeze(run_root, freeze_path)
    key = "movielens/M3OE_FIXED1/seed2022"
    entry = frozen["runs"][key]
    assert entry["identity"]["phase"] == "formal"
    assert set(entry["checkpoints"]) == {"selected", "fixed_final"}
    assert all(Path(value["path"]).is_file() for value in entry["checkpoints"].values())

    data = Data.__new__(Data)
    data.dataset = "movielens"
    data.manifest = manifests["movielens"]
    data.authorization = None
    data.authorize_final(frozen, "M3OE_FIXED1", 2022, phase="formal",
                         run_identity=entry["identity"], view="selected",
                         checkpoint_identity=entry["checkpoints"]["selected"])
    assert data.authorization["view"] == "selected"
    data.x = np.asarray([[0, 1, 0]], dtype=np.float32)
    data.y = np.asarray([[1, 0]], dtype=np.float32)
    data.domain = np.asarray([0], dtype=np.int64)
    data.roles = {"E": np.asarray([0], dtype=np.int64)}

    class FixedLogits(torch.nn.Module):
        def forward(self, x, domain=None):
            return x.new_zeros((len(x), 2))

    final_eval = evaluate_role(FixedLogits(), data, "E", method="M3OE_FIXED1", seed=2022,
                               final_manifest=frozen, phase="formal", run_identity=entry["identity"],
                               checkpoint_identity=entry["checkpoints"]["selected"])
    assert final_eval["role"] == "E"
    snapshot_file = Path(entry["identity"]["implementation_identity"]["snapshot_path"]) / "pipeline.py"
    saved_source = snapshot_file.read_bytes()
    snapshot_file.write_bytes(b"pipeline-source-v2")
    with pytest.raises(PermissionError, match="snapshot differs from the frozen bytes"):
        evaluate_role(FixedLogits(), data, "E", method="M3OE_FIXED1", seed=2022,
                      final_manifest=frozen, phase="formal", run_identity=entry["identity"],
                      checkpoint_identity=entry["checkpoints"]["selected"])
    with pytest.raises(PermissionError):
        data.labels("E", "final")
    snapshot_file.write_bytes(saved_source)
    with pytest.raises(PermissionError):
        evaluate_role(FixedLogits(), data, "E", method="M3OE_FIXED1", seed=2022,
                      final_manifest=frozen, phase="development", run_identity=entry["identity"],
                      checkpoint_identity=entry["checkpoints"]["selected"])
    wrong_path_identity = copy.deepcopy(entry["identity"])
    wrong_path_identity["run_dir"] = str(tmp_path / "other-run")
    with pytest.raises(PermissionError):
        data.authorize_final(frozen, "M3OE_FIXED1", 2022, phase="formal",
                             run_identity=wrong_path_identity, view="selected",
                             checkpoint_identity=entry["checkpoints"]["selected"])
    wrong_config_identity = copy.deepcopy(entry["identity"])
    wrong_config_identity["config"]["models"]["M3OE_FIXED1"]["lr"] += 1e-4
    with pytest.raises(PermissionError):
        data.authorize_final(frozen, "M3OE_FIXED1", 2022, phase="formal",
                             run_identity=wrong_config_identity, view="selected",
                             checkpoint_identity=entry["checkpoints"]["selected"])
    wrong_checkpoint = copy.deepcopy(entry["checkpoints"]["selected"])
    wrong_checkpoint["path"] = str(tmp_path / "replacement.pt")
    with pytest.raises(PermissionError):
        data.authorize_final(frozen, "M3OE_FIXED1", 2022, phase="formal",
                             run_identity=entry["identity"], view="selected",
                             checkpoint_identity=wrong_checkpoint)

    selected_path = Path(entry["checkpoints"]["selected"]["path"])
    displaced = selected_path.with_suffix(".hidden")
    selected_path.rename(displaced)
    with pytest.raises(FileNotFoundError):
        build_freeze(run_root, tmp_path / "missing_checkpoint_freeze.json")
    displaced.rename(selected_path)
    saved_checkpoint = selected_path.read_bytes()
    changed_checkpoint = torch.load(selected_path, map_location="cpu", weights_only=False)
    changed_checkpoint["run_manifest"]["source_commits"]["autoifs"] = "different-commit"
    torch.save(changed_checkpoint, selected_path)
    with pytest.raises(ValueError, match="embedded run/config/source/data identity"):
        build_freeze(run_root, tmp_path / "changed_identity_freeze.json")
    selected_path.write_bytes(saved_checkpoint)


def test_formal_freeze_rejects_single_run_model_config_drift_even_when_checkpoints_match(tmp_path):
    manifests = {dataset: {"dataset": dataset, "protocol": PROTOCOL,
                           "status": "READY_DIRECT_VALUE_CHECKS", "split": "synthetic"}
                 for dataset in ("movielens", "kuairand")}
    run_root = tmp_path / "runs"
    paths = _synthetic_formal_matrix(run_root, manifests)
    run_dir = paths[("movielens", "HYPERGATE", 2023)]
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["config"]["models"]["HYPERGATE"]["lambda_domain"] = 0.999
    run_path.write_text(json.dumps(run), encoding="utf-8")
    for checkpoint_name in ("selected.pt", "fixed_final.pt"):
        checkpoint_path = run_dir / checkpoint_name
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["run_manifest"]["config"]["models"]["HYPERGATE"]["lambda_domain"] = 0.999
        torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="method-specific model configuration drift"):
        build_freeze(run_root, tmp_path / "freeze.json")


def test_formal_freeze_compares_source_snapshot_bytes_not_only_sizes(tmp_path):
    manifests = {dataset: {"dataset": dataset, "protocol": PROTOCOL,
                           "status": "READY_DIRECT_VALUE_CHECKS", "split": "synthetic"}
                 for dataset in ("movielens", "kuairand")}
    run_root = tmp_path / "runs"
    paths = _synthetic_formal_matrix(run_root, manifests)
    run_dir = paths[("movielens", "M3OE_FIXED1", 2022)]
    snapshot_file = run_dir / "implementation_snapshot" / "pipeline.py"
    original = snapshot_file.read_bytes()
    replacement = b"pipeline-source-v2"
    assert len(original) == len(replacement)
    snapshot_file.write_bytes(replacement)
    with pytest.raises(ValueError, match="different implementation source snapshot bytes"):
        build_freeze(run_root, tmp_path / "freeze.json")


def _runtime_retry_fixture(tmp_path, name, amendment, source_contents, epoch_records):
    retry_dir = tmp_path / name
    retry_dir.mkdir()
    snapshot_dir = retry_dir / "implementation_snapshot"
    snapshot_dir.mkdir()
    for relative, content in source_contents.items():
        target = snapshot_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    implementation_identity = {
        "kind": "direct_source_snapshot", "snapshot_path": str(snapshot_dir.resolve()),
        "files": [{"path": path, "size_bytes": len(content)}
                  for path, content in sorted(source_contents.items())],
    }
    selected, final = retry_dir / "selected.pt", retry_dir / "fixed_final.pt"
    selected.write_bytes(b"synthetic selected checkpoint fixture")
    final.write_bytes(b"synthetic final checkpoint fixture")
    run = {
        "protocol": PROTOCOL, "phase": "formal", "method": "HYPERGATE",
        "dataset": "kuairand", "seed": 2022, "status": "TRAIN_COMPLETE_NOT_EVALUATED",
        "config": amendment["configuration"], "source_commits": amendment["source_commits"],
        "data_manifest": amendment["prepared_data_manifests"]["kuairand"],
        "selection_metric": amendment["configuration"]["metrics"]["checkpoint_selection"],
        "implementation_identity": implementation_identity,
        "output_dir": str(retry_dir.resolve()),
        "selected_checkpoint": str(selected.resolve()), "fixed_final_checkpoint": str(final.resolve()),
    }
    (retry_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (retry_dir / "epoch_metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in epoch_records), encoding="utf-8")
    return retry_dir


def test_runtime_io_mixed_freeze_requires_exact_review_pair_actual_replay_and_e_r2(tmp_path, monkeypatch):
    import base64
    import difflib

    amendment_path = Path("research_reports/mdmt_baselines_2526/RUNTIME_IO_SOURCE_EQUIVALENCE.json")
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="reviewed and replayed"):
        build_freeze(tmp_path / "not_created", tmp_path / "pending_freeze.json",
                     runtime_io_amendment=amendment_path)
    snapshots = {
        revision: {entry["path"]: base64.b64decode(entry["content_base64"], validate=True)
                   for entry in amendment[f"{revision.lower()}_source_snapshot"]["files"]}
        for revision in ("R1", "R2")
    }
    assert _package_source_contents() == snapshots["R2"]

    old_dir = Path("artifacts/mdmt_baselines_2526/attempts/formal_kuairand_HYPERGATE_seed2022_attempt01_SIGSEGV")
    old_records = [json.loads(line) for line in (old_dir / "epoch_metrics.jsonl").read_text().splitlines()]
    assert len(old_records) >= 5
    retry_dir = _runtime_retry_fixture(tmp_path, "retry_r2", amendment, snapshots["R2"], old_records[:5])
    cpu_receipt = {"amendment_id": RUNTIME_IO_AMENDMENT_ID, "status": "PASS"}
    cpu_path = tmp_path / "cpu_receipt.json"
    cpu_path.write_text(json.dumps(cpu_receipt), encoding="utf-8")
    feature_path = Path("research_reports/mdmt_baselines_2526/RUNTIME_IO_FEATURE_EQUIVALENCE.json")
    feature_receipt = json.loads(feature_path.read_text(encoding="utf-8"))
    reviewed_pair = {key: amendment[key] for key in (
        "r1_source_snapshot", "r2_source_snapshot", "changed_paths", "configuration",
        "source_commits", "prepared_data_manifests")}
    review_receipt = {
        "amendment_id": RUNTIME_IO_AMENDMENT_ID, "status": "PASS", "verdict": "PASS",
        "reviewed_pair": reviewed_pair, "reviewed_cpu_evidence": cpu_receipt,
        "reviewed_feature_evidence": feature_receipt,
    }
    review_path = tmp_path / "review_receipt.json"
    review_path.write_text(json.dumps(review_receipt), encoding="utf-8")
    amendment["status"] = "REVIEW_REPLAY_COMPLETE"
    amendment["review"] = {"status": "PASS", "receipt_path": str(review_path)}
    amendment["cpu_validation"] = {"status": "PASS", "receipt_path": str(cpu_path),
                                   "feature_equivalence_receipt": str(feature_path)}
    amendment["replay"] = {
        "status": "PASS",
        "failed_run_manifest": str((old_dir / "run.json").resolve()),
        "failed_epoch_log": str((old_dir / "epoch_metrics.jsonl").resolve()),
        "failed_crash_receipt": str((old_dir / "NATIVE_CRASH_RECEIPT.json").resolve()),
        "retry_run_manifest": str((retry_dir / "run.json").resolve()),
        "retry_epoch_log": str((retry_dir / "epoch_metrics.jsonl").resolve()),
    }
    completed_amendment = tmp_path / "completed_amendment_fixture.json"
    completed_amendment.write_text(json.dumps(amendment), encoding="utf-8")

    # A PASS summary cannot replace actual files or exact epoch comparisons.
    fake_paths = copy.deepcopy(amendment)
    fake_paths["replay"]["retry_run_manifest"] = str(tmp_path / "nonexistent-run.json")
    fake_paths["replay"]["retry_epoch_log"] = str(tmp_path / "nonexistent-log.jsonl")
    fake_path = tmp_path / "fake_paths.json"
    fake_path.write_text(json.dumps(fake_paths), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="run manifest or epoch log is missing"):
        build_freeze(tmp_path / "not_created", tmp_path / "fake_path_freeze.json",
                     runtime_io_amendment=fake_path)

    four_epoch_dir = _runtime_retry_fixture(tmp_path, "retry_four", amendment,
                                            snapshots["R2"], old_records[:4])
    fewer = copy.deepcopy(amendment)
    fewer["replay"]["retry_run_manifest"] = str((four_epoch_dir / "run.json").resolve())
    fewer["replay"]["retry_epoch_log"] = str((four_epoch_dir / "epoch_metrics.jsonl").resolve())
    fewer_path = tmp_path / "fewer.json"
    fewer_path.write_text(json.dumps(fewer), encoding="utf-8")
    with pytest.raises(ValueError, match="fewer than five complete train records"):
        build_freeze(tmp_path / "not_created", tmp_path / "fewer_freeze.json", runtime_io_amendment=fewer_path)

    changed_records = copy.deepcopy(old_records[:5])
    changed_records[0]["train_loss"] += 1e-6
    changed_dir = _runtime_retry_fixture(tmp_path, "retry_changed", amendment,
                                         snapshots["R2"], changed_records)
    changed = copy.deepcopy(amendment)
    changed["replay"]["retry_run_manifest"] = str((changed_dir / "run.json").resolve())
    changed["replay"]["retry_epoch_log"] = str((changed_dir / "epoch_metrics.jsonl").resolve())
    changed_path = tmp_path / "changed_trajectory.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="first-five scientific trajectory fields differ"):
        build_freeze(tmp_path / "not_created", tmp_path / "changed_freeze.json",
                     runtime_io_amendment=changed_path)

    r1_retry_dir = _runtime_retry_fixture(tmp_path, "retry_non_r2", amendment,
                                          snapshots["R1"], old_records[:5])
    non_r2 = copy.deepcopy(amendment)
    non_r2["replay"]["retry_run_manifest"] = str((r1_retry_dir / "run.json").resolve())
    non_r2["replay"]["retry_epoch_log"] = str((r1_retry_dir / "epoch_metrics.jsonl").resolve())
    non_r2_path = tmp_path / "non_r2.json"
    non_r2_path.write_text(json.dumps(non_r2), encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain the exact approved R2 source snapshot"):
        build_freeze(tmp_path / "not_created", tmp_path / "non_r2_freeze.json",
                     runtime_io_amendment=non_r2_path)

    manifests = amendment["prepared_data_manifests"]
    run_root = tmp_path / "runs"
    paths = _synthetic_formal_matrix(run_root, manifests, source_snapshots=snapshots)
    with pytest.raises(ValueError, match="different implementation source snapshot bytes"):
        build_freeze(run_root, tmp_path / "default_mixed_freeze.json")

    freeze_path = tmp_path / "amended_freeze.json"
    frozen = build_freeze(run_root, freeze_path, runtime_io_amendment=completed_amendment)
    revisions = {entry["source_revision"] for entry in frozen["runs"].values()}
    assert revisions == {"R1", "R2"}
    key = "movielens/M3OE_FIXED1/seed2022"
    entry = frozen["runs"][key]
    assert entry["source_revision"] == "R1"
    assert set(entry["checkpoints"]) == {"selected", "fixed_final"}
    for view in ("selected", "fixed_final"):
        checkpoint = entry["checkpoints"][view]
        expected = Path(paths[("movielens", "M3OE_FIXED1", 2022)]) / (
            "selected.pt" if view == "selected" else "fixed_final.pt")
        assert checkpoint["path"] == str(expected.resolve())
        run_manifest = json.loads((expected.parent / "run.json").read_text(encoding="utf-8"))
        assert checkpoint["embedded_identity"] == report_module._embedded_identity(run_manifest)

    # Same-size mutations, including a currently forbidden eager-Y candidate, cannot reuse the receipt.
    candidate = copy.deepcopy(amendment)
    r2_file = candidate["r2_source_snapshot"]["files"]
    data_entry = next(item for item in r2_file if item["path"] == "data.py")
    r2_data = base64.b64decode(data_entry["content_base64"], validate=True)
    eager = r2_data.replace(b'self.y = np.load(self.root / "y.npy", mmap_mode="r", allow_pickle=False)',
                            b'self.y = np.load(self.root / "y.npy", allow_pickle=False)')
    assert eager != r2_data
    data_entry["size_bytes"] = len(eager)
    data_entry["content_base64"] = base64.b64encode(eager).decode("ascii")
    candidate["changed_files"] = copy.deepcopy(candidate["changed_files"])
    changed_file = next(item for item in candidate["changed_files"] if item["path"] == "data.py")
    changed_file["r2_content_base64"] = base64.b64encode(eager).decode("ascii")
    before = snapshots["R1"]["data.py"].decode().splitlines(keepends=True)
    after = eager.decode().splitlines(keepends=True)
    changed_file["unified_diff"] = "".join(difflib.unified_diff(
        before, after, fromfile="R1/data.py", tofile="R2/data.py"))
    altered_current = dict(snapshots["R2"])
    altered_current["data.py"] = eager
    candidate_path = tmp_path / "altered_pair_with_old_review.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    with monkeypatch.context() as patcher:
        patcher.setattr(report_module, "_package_source_contents", lambda: altered_current)
        with pytest.raises(ValueError, match="exact R1/R2 pair and originals"):
            build_freeze(run_root, tmp_path / "old_receipt_reuse.json",
                         runtime_io_amendment=candidate_path)

    # The amendment's declared original identities are checked against every formal run.
    run_path = paths[("movielens", "M3OE_FIXED1", 2022)] / "run.json"
    original_run = json.loads(run_path.read_text(encoding="utf-8"))
    drifted_run = copy.deepcopy(original_run)
    drifted_run["source_commits"]["recstudio"] = "unapproved-source"
    run_path.write_text(json.dumps(drifted_run), encoding="utf-8")
    with pytest.raises(ValueError, match="source commits differ from the reviewed original"):
        build_freeze(run_root, tmp_path / "original_identity_mismatch.json",
                     runtime_io_amendment=completed_amendment)
    run_path.write_text(json.dumps(original_run), encoding="utf-8")

    data = Data.__new__(Data)
    data.dataset = "movielens"
    data.manifest = manifests["movielens"]
    data.authorization = None
    data.x = np.asarray([[0.0, 0.0]], dtype=np.float32)
    data.y = np.asarray([[0.0, 1.0]], dtype=np.float32)
    data.domain = np.asarray([0], dtype=np.int64)
    data.roles = {"E": np.asarray([0], dtype=np.int64)}
    label_calls = {"n": 0}
    original_labels = Data.labels

    def labels_spy(self, role, purpose, row_ids=None):
        label_calls["n"] += 1
        return original_labels(self, role, purpose, row_ids)

    monkeypatch.setattr(Data, "labels", labels_spy)

    class FixedLogits(torch.nn.Module):
        def forward(self, x, domain=None):
            return x.new_zeros((len(x), 2))

    evaluate_role(FixedLogits(), data, "E", method="M3OE_FIXED1", seed=2022,
                  final_manifest=frozen, phase="formal", run_identity=entry["identity"],
                  checkpoint_identity=entry["checkpoints"]["selected"])
    assert label_calls["n"] == 1

    bad_revision_freeze = copy.deepcopy(frozen)
    bad_revision_freeze["runs"][key]["source_revision"] = "R3"
    data.authorization = None
    label_calls["n"] = 0
    with pytest.raises(PermissionError, match="source-revision binding"):
        evaluate_role(FixedLogits(), data, "E", method="M3OE_FIXED1", seed=2022,
                      final_manifest=bad_revision_freeze, phase="formal", run_identity=entry["identity"],
                      checkpoint_identity=entry["checkpoints"]["selected"])
    assert data.authorization is None and label_calls["n"] == 0

    changed_package = dict(snapshots["R2"])
    changed_data = bytearray(changed_package["data.py"])
    changed_data[0] ^= 1
    changed_package["data.py"] = bytes(changed_data)
    monkeypatch.setattr(report_module, "_package_source_contents", lambda: changed_package)
    with pytest.raises(PermissionError, match="Installed implementation package"):
        evaluate_role(FixedLogits(), data, "E", method="M3OE_FIXED1", seed=2022,
                      final_manifest=frozen, phase="formal", run_identity=entry["identity"],
                      checkpoint_identity=entry["checkpoints"]["selected"])
    assert data.authorization is None and label_calls["n"] == 0


def test_positive_logits_are_removed_and_zero_logits_are_retained():
    all_positive = _selector_removed_retained(torch.ones((2, 4)))
    all_negative = _selector_removed_retained(-torch.ones((2, 4)))
    mixed = _selector_removed_retained(torch.tensor([[1.0, 0.0, -1.0, 2.0]]))
    assert all_positive == {"removed": 8, "retained": 0, "total": 8, "pruned_fraction": 1.0}
    assert all_negative == {"removed": 0, "retained": 8, "total": 8, "pruned_fraction": 0.0}
    assert mixed == {"removed": 2, "retained": 2, "total": 4, "pruned_fraction": 0.5}


def test_train_failure_leaves_started_then_failed_receipt_and_reraises(tmp_path, monkeypatch):
    import experiments.mdmt_sota_baselines_2526.training as training_module

    old_matmul = torch.backends.cuda.matmul.fp32_precision
    old_cudnn = torch.backends.cudnn.conv.fp32_precision

    def missing_data(*args, **kwargs):
        raise FileNotFoundError("synthetic missing prepared data")

    monkeypatch.setattr(training_module, "Data", missing_data)
    output = tmp_path / "failed_run"
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    with pytest.raises(FileNotFoundError, match="synthetic missing prepared data"):
        train_one("HYPERGATE", "movielens", 2022, output, phase="development",
                  config=load_config(), device="cpu")
    assert torch.backends.cuda.matmul.fp32_precision == "ieee"
    assert torch.backends.cudnn.conv.fp32_precision == "ieee"
    receipt = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "FAILED"
    assert receipt["failure_stage"] == "data_load"
    assert receipt["failure_type"] == "FileNotFoundError"
    assert receipt["total_elapsed_seconds"] >= 0
    assert (output / "implementation_snapshot" / "training.py").is_file()
    assert (output / "implementation_snapshot" / "reference_data" / "movielens" / "PREPROCESSOR.json").is_file()
    assert (output / "implementation_snapshot" / "reference_data" / "movielens" / "DATA_MANIFEST.json").is_file()
    assert (output / "implementation_snapshot" / "reference_data" / "kuairand" / "PREPROCESSOR.json").is_file()
    assert (output / "implementation_snapshot" / "reference_data" / "kuairand" / "DATA_MANIFEST.json").is_file()
    assert receipt["environment"]["torch"] == torch.__version__
    assert receipt["environment"]["tf32_matmul"] == "ieee"
    assert receipt["environment"]["tf32_cudnn_conv"] == "ieee"
    torch.backends.cuda.matmul.fp32_precision = old_matmul
    torch.backends.cudnn.conv.fp32_precision = old_cudnn


def test_inference_timing_records_fixed_batch_and_percentiles():
    class SyntheticModel(torch.nn.Module):
        def forward(self, x, domain=None):
            return torch.stack((x[:, 0], x[:, 0] * 2), dim=1)

    timing = measure_inference(SyntheticModel(), np.arange(12, dtype=np.float32).reshape(4, 3),
                               device="cpu", warmup=1, repetitions=5)
    assert timing["batch_size"] == 4
    assert timing["device"] == "cpu"
    assert timing["warmup"] == 1 and timing["repetitions"] == 5
    assert timing["p50_ms"] >= 0 and timing["p95_ms"] >= timing["p50_ms"]


def test_custom_evaluation_output_is_indexed_and_identity_options_are_assertions(tmp_path, monkeypatch):
    cfg = load_config()
    run_root = tmp_path / "runs"
    run_dir = run_root / "development" / "movielens" / "M3OE_FIXED1" / "seed2022"
    run_dir.mkdir(parents=True)
    selected_path, final_path = run_dir / "selected.pt", run_dir / "fixed_final.pt"
    data_manifest = {"dataset": "movielens", "protocol": PROTOCOL, "status": "SYNTHETIC_FIXTURE"}
    started = "2026-10-28T00:00:00+00:00"
    embedded = {"protocol": PROTOCOL, "phase": "development", "method": "M3OE_FIXED1",
                "dataset": "movielens", "seed": 2022, "config": cfg,
                "source_commits": cfg["source_commits"], "data_manifest": data_manifest,
                "selection_metric": cfg["metrics"]["checkpoint_selection"],
                "environment": {"python": "synthetic-3.13"},
                "implementation_identity": {"kind": "synthetic"},
                "output_dir": str(run_dir.resolve()), "started_at_utc": started}
    run = {**embedded, "status": "TRAIN_COMPLETE_NOT_EVALUATED",
           "selected_checkpoint": str(selected_path.resolve()),
           "fixed_final_checkpoint": str(final_path.resolve()), "selected_epoch": 1, "final_epoch": 2}
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    checkpoint = {"model_state": {"logit": torch.tensor([0.2, -0.1])},
                  "run_manifest": embedded, "model_meta": {}}
    torch.save(checkpoint, selected_path)
    torch.save(checkpoint, final_path)

    class FixtureData:
        dataset = "movielens"
        manifest = data_manifest
        x = np.column_stack((np.arange(6), np.tile([0, 1, 2], 2))).astype(np.float32)
        y = np.column_stack((np.arange(6) % 2, (np.arange(6) // 2) % 2)).astype(np.float32)
        domain = x[:, -1].astype(np.int64)
        roles = {"V": np.arange(6)}
        schema = {"dataset": "movielens", "columns": ["user_id", "movie_id", "domain_indicator"],
                  "dense_columns": [], "sparse_columns": ["user_id", "movie_id", "domain_indicator"],
                  "vocab_sizes": {"user_id": 5, "movie_id": 7, "domain_indicator": 3}, "embedding_dim": 16}

        def labels(self, role, purpose):
            assert (role, purpose) == ("V", "select")
            return self.y, np.ones_like(self.y, dtype=bool), self.roles[role]

    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logit = torch.nn.Parameter(torch.zeros(2))

        def forward(self, x, domain=None):
            return self.logit.expand(len(x), 2)

    monkeypatch.setattr(cli_module, "Data", lambda *args, **kwargs: FixtureData())
    monkeypatch.setattr(cli_module, "build_model", lambda *args, **kwargs: FixedModel())
    custom_output = tmp_path / "evaluation_artifacts"
    result = cli_module.evaluate_checkpoint(run_dir, "V", "selected", cfg, device="cpu",
                                            output_dir=custom_output,
                                            expected_method="M3OE_FIXED1", expected_dataset="movielens",
                                            expected_seed=2022)
    assert result["role"] == "V"
    assert (custom_output / "metrics_V_selected.json").is_file()
    index_path = run_dir / "evaluation_index.json"
    assert _evaluation_files(run_dir, run_dir / "run.json")[("V", "selected")] == custom_output / "metrics_V_selected.json"
    indexed = json.loads(index_path.read_text(encoding="utf-8"))["evaluations"][0]
    assert indexed["metrics"] == str((custom_output / "metrics_V_selected.json").resolve())

    reported = build_report(run_root, tmp_path / "custom_report", config=cfg,
                            methods=["M3OE_FIXED1"], datasets=["movielens"], seeds=[2022])
    assert reported["per_cell_rows"] == 6
    with pytest.raises(ValueError, match="--method does not match"):
        cli_module.evaluate_checkpoint(run_dir, "V", "selected", cfg, device="cpu",
                                        output_dir=custom_output, expected_method="HYPERGATE")


def test_audit_and_cli_filters_are_effective(tmp_path, monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(cli_module, "_source_inventory",
                        lambda config, methods=None: {"selected_methods": list(methods)})
    monkeypatch.setattr(cli_module.shutil, "which", lambda _: None)
    parser = cli_module.make_parser()
    parsed = parser.parse_args(["audit", "--method", "AUTOIFS", "--dataset", "kuairand", "--seed", "2025"])
    result = cli_module.audit(cfg, methods=parsed.method, datasets=parsed.dataset, seeds=parsed.seed)
    assert result["filters"] == {"methods": ("AUTOIFS",), "datasets": ("kuairand",), "seeds": (2025,)}
    assert result["matrix"] == {"methods": ["AUTOIFS"], "datasets": ["kuairand"],
                                 "formal_seeds": [2025], "formal_trajectories": 1}
    parsed_evaluate = parser.parse_args(["evaluate", "--run-dir", str(tmp_path / "run"), "--role", "V",
                                         "--method", "M3OE_FIXED1", "--dataset", "movielens", "--seed", "2022",
                                         "--output-dir", str(tmp_path / "eval")])
    assert parsed_evaluate.output_dir == tmp_path / "eval"
    assert (parsed_evaluate.method, parsed_evaluate.dataset, parsed_evaluate.seed) == (
        "M3OE_FIXED1", "movielens", 2022)

