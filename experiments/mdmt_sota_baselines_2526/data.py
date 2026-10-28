"""T0-fit preparation and explicit MDMT label-role permissions.

The existing strict-data module computes checksums in its preparation class.
This module reuses only its row mapping and value-construction functions and
validates by direct values, so it obeys the active hash=deny task boundary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.mdmt_relation_basis_v2.vendor_snapshot.mdmt_dfd_20261026.vendor.strict_data.data_roles import (
    _chunks, _raw_schema, _strings, assign_roles,
)

from .config import DATASETS, METHODS, PROTOCOL, ROOT, SEEDS, load_config

ROLE_MAP = {
    "T0": ("fit",), "F": ("P_label",), "S": ("P_cal", "P_update"),
    "Q": ("Q_explore",), "V": ("validation",), "E": ("test",),
    "legacy": ("Q_confirm",),
}
ROLE_PURPOSES = {"T0": {"train"}, "F": {"train"}, "V": {"select"}, "Q": {"development"}, "E": {"final"}}
_DATASET_INDEX = {"movielens": 0, "kuairand": 1}


def _fit_only_preprocessor(dataset, raw_path, domains, columns, dense, roles):
    fit_mask = np.zeros(len(domains), dtype=bool)
    fit_mask[roles["fit"]] = True
    sparse = tuple(name for name in columns if name not in dense)
    vocabulary_sets = {name: set() for name in sparse if name != "domain_indicator"}
    age_min, age_max, offset = float("inf"), float("-inf"), 0
    for chunk, _ in _chunks(dataset, raw_path):
        end = offset + len(chunk)
        local = chunk.loc[fit_mask[offset:end]]
        for name in vocabulary_sets:
            vocabulary_sets[name].update(_strings(local[name]).unique().tolist())
        if dense and len(local):
            age_min = min(age_min, float(local.age.min()))
            age_max = max(age_max, float(local.age.max()))
        offset = end
    if offset != len(domains):
        raise ValueError("Raw row count changed during T0 preprocessing fit")
    vocab = {name: {value: i + 1 for i, value in enumerate(sorted(values))}
             for name, values in vocabulary_sets.items()}
    scaler = {"age_min": age_min, "age_max": age_max} if dense else {}
    return vocab, scaler, sparse


def prepare_owned(dataset, destination, *, config=None):
    """Rebuild missing arrays at a new destination, checking all values directly."""
    cfg = config or load_config()
    if dataset not in cfg["datasets"]:
        raise ValueError("Unknown dataset")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite prepared output: {destination}")
    raw_path = Path(cfg["raw_paths"][dataset])
    prior = ROOT / cfg["historical_prepared"][dataset]
    for required in (raw_path, prior / "PREPROCESSOR.json", prior / "DATA_MANIFEST.json"):
        if not required.is_file():
            raise FileNotFoundError(f"Required input is missing: {required}")
    historical_manifest = json.loads((prior / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    historical_preprocessor = json.loads((prior / "PREPROCESSOR.json").read_text(encoding="utf-8"))
    domains, columns, dense = _raw_schema(dataset, raw_path)
    roles = assign_roles(domains)
    vocab, scaler, sparse = _fit_only_preprocessor(dataset, raw_path, domains, columns, dense, roles)
    if vocab != historical_preprocessor["vocabulary"] or scaler != historical_preprocessor["scaler"]:
        raise ValueError("T0-only fitted vocabulary/scaler differs from the retained historical preprocessor")
    prior_features = historical_manifest["features"]
    if (list(columns) != prior_features["columns"] or list(dense) != prior_features["dense_columns"]
            or list(sparse) != prior_features["sparse_columns"]):
        raise ValueError("Feature schema differs from the retained historical manifest")
    vocab_sizes = {name: (3 if name == "domain_indicator" else len(vocab[name]) + 1) for name in sparse}
    if vocab_sizes != prior_features["vocab_sizes"]:
        raise ValueError("Feature vocabulary sizes differ from the retained historical manifest")

    # Only now create an owned output. The destination must be absent above.
    destination.mkdir(parents=True, exist_ok=False)
    x = np.lib.format.open_memmap(destination / "x.npy", mode="w+", dtype=np.float32,
                                  shape=(len(domains), len(columns)))
    y = np.lib.format.open_memmap(destination / "y.npy", mode="w+", dtype=np.float32,
                                  shape=(len(domains), 2))
    offset = 0
    for chunk, labels in _chunks(dataset, raw_path):
        end = offset + len(chunk)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Non-binary task label in source rows")
        for col, name in enumerate(columns):
            if name == "domain_indicator":
                values = domains[offset:end]
            elif name in dense:
                values = (chunk[name].to_numpy(np.float32) - scaler["age_min"]) / max(
                    scaler["age_max"] - scaler["age_min"], 1e-12)
            else:
                values = _strings(chunk[name]).map(vocab[name]).fillna(0).to_numpy(np.float32)
            x[offset:end, col] = values
        y[offset:end] = labels
        offset = end
    if offset != len(domains):
        raise ValueError("Raw row count changed during value encoding")
    x.flush()
    y.flush()
    del x, y
    for role, row_ids in roles.items():
        np.save(destination / f"rows_{role}.npy", row_ids, allow_pickle=False)

    # Read back each output and compare its full ordered rows against the source
    # formulas; role arrays are also checked as exhaustive and disjoint.
    x = np.load(destination / "x.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(destination / "y.npy", mmap_mode="r", allow_pickle=False)
    expected_domain, _, _ = _raw_schema(dataset, raw_path)
    if not np.array_equal(expected_domain, x[:, -1]):
        raise ValueError("Encoded domain column differs from source mapping")
    role_arrays = {role: np.load(destination / f"rows_{role}.npy", allow_pickle=False) for role in roles}
    joined = np.concatenate(tuple(role_arrays.values()))
    if len(joined) != len(domains) or not np.array_equal(np.sort(joined), np.arange(len(domains))):
        raise ValueError("Role row plan is not exhaustive and disjoint")
    for role, ids in roles.items():
        if not np.array_equal(role_arrays[role], ids):
            raise ValueError(f"Rebuilt role rows differ: {role}")
        prior_count = historical_manifest["roles"][role]["n"]
        if len(ids) != prior_count:
            raise ValueError(f"Historical role row count differs: {role}")

    # Verify source formulas chunk by chunk against the written full arrays.
    offset = 0
    fit_mask = np.zeros(len(domains), dtype=bool)
    fit_mask[roles["fit"]] = True
    vocabulary_seen = {name: set() for name in vocab}
    low, high = float("inf"), float("-inf")
    for chunk, labels in _chunks(dataset, raw_path):
        end = offset + len(chunk)
        if not np.array_equal(labels, y[offset:end]):
            raise ValueError("Encoded labels differ from source labels")
        local = chunk.loc[fit_mask[offset:end]]
        for name in vocab:
            vocabulary_seen[name].update(_strings(local[name]).unique().tolist())
        if dense and len(local):
            low, high = min(low, float(local.age.min())), max(high, float(local.age.max()))
        for col, name in enumerate(columns):
            if name == "domain_indicator":
                expected = domains[offset:end]
            elif name in dense:
                expected = (chunk[name].to_numpy(np.float32) - scaler["age_min"]) / max(
                    scaler["age_max"] - scaler["age_min"], 1e-12)
            else:
                expected = _strings(chunk[name]).map(vocab[name]).fillna(0).to_numpy(np.float32)
            if not np.array_equal(expected, x[offset:end, col]):
                raise ValueError(f"Encoded values differ from source column: {name}")
        offset = end
    if {name: {value: i + 1 for i, value in enumerate(sorted(values))}
            for name, values in vocabulary_seen.items()} != vocab:
        raise ValueError("T0 vocabulary was not fitted from T0 rows")
    if dense and scaler != {"age_min": low, "age_max": high}:
        raise ValueError("Dense scaler was not fitted from T0 rows")

    features = {
        "dataset": dataset, "columns": list(columns), "dense_columns": list(dense),
        "sparse_columns": list(sparse), "vocab_sizes": vocab_sizes,
        "embedding_dim": int(prior_features.get("embedding_dim", 16)),
        "domain_column": "domain_indicator", "encoder_hash": prior_features["encoder_hash"],
        "scaler_hash": prior_features["scaler_hash"], "oov_index": 0,
    }
    cell_counts = {}
    for role, ids in role_arrays.items():
        cells = []
        for task in range(2):
            for domain in range(3):
                selected = ids[domains[ids] == domain]
                positive = int(y[selected, task].sum())
                cells.append({"domain": domain, "task": task, "n": int(len(selected)),
                              "positive": positive, "negative": int(len(selected) - positive)})
        cell_counts[role] = {"n": int(len(ids)), "domain_counts": np.bincount(domains[ids], minlength=3).tolist(),
                             "cells": cells}
    manifest = {
        "protocol": cfg["protocol"], "status": "READY_DIRECT_VALUE_CHECKS",
        "dataset": dataset, "raw_path": str(raw_path), "prepared_path": str(destination.resolve()),
        "registered_raw_sha256_provenance": cfg["audit"]["registered_raw_sha256_provenance"][dataset],
        "checksum_revalidated": False,
        "historical_manifest_raw_sha256_provenance": historical_manifest.get("raw_sha256"),
        "row_count": int(len(domains)), "features": features, "role_map": ROLE_MAP,
        "roles": cell_counts, "checks": {
            "raw_schema_and_domain_mapping": "PASS_DIRECT_VALUES",
            "labels_against_source_formula": "PASS_DIRECT_VALUES",
            "features_against_source_formula": "PASS_DIRECT_VALUES",
            "fit_vocabulary_and_scaler_against_retained_preprocessor": "PASS_DIRECT_VALUES",
            "role_rows_disjoint_exhaustive_and_counts": "PASS_DIRECT_VALUES",
            "historical_row_array_identity": "NOT_AVAILABLE_ARRAYS_MISSING",
            "checksum": "NOT_RECOMPUTED_HASH_DENIED",
        },
    }
    (destination / "PREPROCESSOR.json").write_text(
        json.dumps({"vocabulary": vocab, "scaler": scaler}, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "DATA_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class Data:
    def __init__(self, dataset, prepared_root):
        if dataset not in _DATASET_INDEX:
            raise ValueError("Unknown dataset")
        self.dataset = dataset
        self.root = Path(prepared_root) / dataset
        self.manifest = json.loads((self.root / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "READY_DIRECT_VALUE_CHECKS" or self.manifest.get("dataset") != dataset:
            raise ValueError("Direct-value preparation receipt is required")
        self.x = np.load(self.root / "x.npy", mmap_mode="r", allow_pickle=False)
        self.y = np.load(self.root / "y.npy", mmap_mode="r", allow_pickle=False)
        self.domain = self.x[:, -1].astype(np.int64)
        self.schema = self.manifest["features"]
        self.roles = {name: np.sort(np.concatenate([
            np.load(self.root / f"rows_{source}.npy", allow_pickle=False)
            for source in components])) for name, components in ROLE_MAP.items()}
        joined = np.concatenate(tuple(self.roles.values()))
        if len(joined) != len(self.x) or not np.array_equal(np.sort(joined), np.arange(len(self.x))):
            raise ValueError("Prepared roles must be exhaustive and disjoint")
        self._train_ids = np.union1d(self.roles["T0"], self.roles["F"])
        if self.x.ndim != 2 or len(self.y) != len(self.x) or self.y.shape[1] != 2:
            raise ValueError("Prepared arrays have invalid shapes")
        if not np.isin(self.domain, [0, 1, 2]).all():
            raise ValueError("Prepared domain values are invalid")
        self.authorization = None

    @property
    def train_ids(self):
        return self._train_ids

    def training_batch(self, row_ids):
        ids = np.asarray(row_ids, dtype=np.int64)
        allowed = self._train_ids
        positions = np.searchsorted(allowed, ids)
        if len(ids) and ((positions >= len(allowed)).any() or not np.array_equal(allowed[positions], ids)):
            raise PermissionError("Optimizer steps may use T0 ∪ F only")
        labels = np.asarray(self.y[ids])
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Training labels must be binary")
        return np.asarray(self.x[ids]), labels, np.ones((len(ids), 2), dtype=bool)

    def _select(self, role, row_ids=None):
        if role not in self.roles:
            raise ValueError(f"Unknown data role: {role}")
        pool = self.roles[role]
        ids = pool if row_ids is None else np.asarray(row_ids, dtype=np.int64)
        positions = np.searchsorted(pool, ids)
        if len(ids) and ((positions >= len(pool)).any() or not np.array_equal(pool[positions], ids)):
            raise PermissionError(f"Rows are outside role {role}")
        return ids

    def inputs(self, role, row_ids=None):
        if role in ("S", "legacy"):
            raise PermissionError(f"Baseline input access to {role} is disabled")
        if role == "E" and self.authorization is None:
            raise PermissionError("E is locked until the exact formal matrix is frozen")
        ids = self._select(role, row_ids)
        return np.asarray(self.x[ids]), ids.copy()

    def labels(self, role, purpose, row_ids=None):
        if role in ("S", "legacy") or purpose not in ROLE_PURPOSES.get(role, set()):
            raise PermissionError(f"{purpose} cannot read labels from {role}")
        if role == "E" and self.authorization is None:
            raise PermissionError("E is locked until the exact formal matrix is frozen")
        ids = self._select(role, row_ids)
        labels = np.asarray(self.y[ids])
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Labels for role {role} must be binary")
        return labels, np.ones((len(ids), 2), dtype=bool), ids.copy()

    def authorize_final(self, manifest, method, seed, *, phase, run_identity, view, checkpoint_identity):
        self.authorization = None
        if manifest.get("status") != "FROZEN" or manifest.get("protocol") != self.manifest.get("protocol"):
            raise PermissionError("Global formal freeze is missing or mismatched")
        expected = {f"{dataset}/{model}/seed{formal_seed}"
                    for dataset in DATASETS for model in METHODS for formal_seed in SEEDS}
        if set(manifest.get("runs", {})) != expected:
            raise PermissionError("The frozen manifest must enumerate the complete 30-run matrix")
        if phase != "formal":
            raise PermissionError("E evaluation is restricted to the exact formal trajectory")
        key = f"{self.dataset}/{method}/seed{int(seed)}"
        entry = manifest["runs"].get(key)
        if entry is None or entry.get("status") != "COMPLETE":
            raise PermissionError("This trajectory is not present in the frozen complete matrix")
        common = {"protocol": run_identity.get("protocol"),
                  "source_commits": run_identity.get("source_commits"),
                  "selection_metric": run_identity.get("selection_metric"),
                  "roles": run_identity.get("config", {}).get("roles"),
                  "metrics": run_identity.get("config", {}).get("metrics"),
                  "training": run_identity.get("config", {}).get("training")}
        if manifest.get("identity", {}).get("global") != common:
            raise PermissionError("Run protocol/source/roles/metrics differ from the frozen identity")
        model_config_by_method = manifest.get("identity", {}).get("model_config_by_method", {})
        run_model_config = (run_identity.get("config") or {}).get("models", {}).get(method)
        if model_config_by_method.get(method) != run_model_config:
            raise PermissionError("Method-specific model configuration differs from the frozen identity")
        if manifest.get("identity", {}).get("prepared_data_by_dataset", {}).get(self.dataset) != self.manifest:
            raise PermissionError("Prepared data identity differs from the frozen dataset")
        if entry.get("identity") != run_identity:
            raise PermissionError("Current run path or run/config/source/data identity differs from freeze")
        frozen_checkpoint = entry.get("checkpoints", {}).get(view)
        if view not in ("selected", "fixed_final") or frozen_checkpoint != checkpoint_identity:
            raise PermissionError("Current checkpoint path/state metadata differs from the frozen entry")
        self.authorization = {"key": key, "view": view, "identity": run_identity}


def tensor(value, device):
    return torch.as_tensor(np.array(value), dtype=torch.float32, device=device)

