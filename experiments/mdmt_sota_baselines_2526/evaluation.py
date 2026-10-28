"""Selected-checkpoint evaluation with a global test-label freeze gate."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.mdmt_relation_basis_v2.metrics import evaluate

from .data import Data, tensor
from .report import verify_frozen_source_snapshot


def evaluate_role(model, data, role, *, method, seed, device="cpu", batch_size=4096,
                  output_dir=None, final_manifest=None, checkpoint_view="selected",
                  phase=None, run_identity=None, checkpoint_identity=None):
    if role not in ("V", "Q", "E"):
        raise ValueError("Evaluation role must be V, Q or E")
    if role == "E":
        data.authorization = None
        if final_manifest is None or run_identity is None or checkpoint_identity is None:
            raise PermissionError("E requires the exact frozen run and checkpoint identity")
        frozen_snapshot = final_manifest.get("identity", {}).get("implementation_source_snapshot")
        verify_frozen_source_snapshot(run_identity.get("implementation_identity"), frozen_snapshot)
        data.authorize_final(final_manifest, method, seed, phase=phase, run_identity=run_identity,
                             view=checkpoint_view, checkpoint_identity=checkpoint_identity)
        purpose = "final"
    else:
        purpose = "select" if role == "V" else "development"
    ids = data.roles[role]
    labels, mask, _ = data.labels(role, purpose)
    domain = data.domain[ids]
    logits_parts = []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(ids), int(batch_size)):
            x = tensor(data.x[ids[begin:begin + int(batch_size)]], device)
            logits_parts.append(model(x, domain=x[:, -1].long()).detach().cpu().numpy())
    logits = np.concatenate(logits_parts, axis=0) if logits_parts else np.empty((0, 2), np.float32)
    result = evaluate(labels, logits, domain, mask)
    defined_auc = [cell for cell in result["cells"] if cell["auc"] is not None]
    auc_requests = sum(cell["n"] for cell in defined_auc)
    result.update({
        "natural_auc": (sum(cell["n"] * cell["auc"] for cell in defined_auc) / auc_requests
                        if auc_requests else None),
        "natural_auc_defined_requests": int(auc_requests),
        "pooled_auc_valid_requests_by_task": [int(mask[:, task].sum()) for task in range(2)],
        "macro_auc_defined_cells": sum(cell["auc"] is not None for cell in result["cells"]),
        "macro_logloss_defined_cells": sum(cell["logloss"] is not None for cell in result["cells"]),
    })
    result.update({"method": method, "dataset": data.dataset, "seed": int(seed),
                   "role": role, "checkpoint_view": checkpoint_view,
                   "checkpoint_id": f"{method}/{data.dataset}/seed{int(seed)}/{checkpoint_view}",
                   "row_ids": ids.tolist()})
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / f"predictions_{role}_{checkpoint_view}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("row_id", "domain", "task", "label", "mask", "logit", "probability",
                             "method", "dataset", "seed", "checkpoint_id", "role"))
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -80, 80)))
            for row, row_id in enumerate(ids):
                for task in range(2):
                    writer.writerow((int(row_id), int(domain[row]), task, float(labels[row, task]),
                                     bool(mask[row, task]), float(logits[row, task]),
                                     float(probabilities[row, task]), method, data.dataset,
                                     int(seed), result["checkpoint_id"], role))
        persisted = {key: value for key, value in result.items() if key != "row_ids"}
        (output_dir / f"metrics_{role}_{checkpoint_view}.json").write_text(
            json.dumps(persisted, indent=2), encoding="utf-8")
    return result


def load_checkpoint(model, path, *, expected_method, expected_dataset, expected_seed, expected_config):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    manifest = checkpoint["run_manifest"]
    if (manifest["method"], manifest["dataset"], int(manifest["seed"])) != (
            expected_method, expected_dataset, int(expected_seed)):
        raise ValueError("Checkpoint method/dataset/seed identity differs")
    if manifest["config"] != expected_config:
        raise ValueError("Checkpoint configuration differs")
    model.load_state_dict(checkpoint["model_state"])
    if expected_method == "AUTOIFS":
        model.network.ticket = bool(checkpoint["model_meta"]["ticket"])
        model.network.temp = float(checkpoint["model_meta"]["temp"])
    return checkpoint


def measure_inference(model, x, *, device="cpu", warmup=5, repetitions=25):
    device = torch.device(device)
    batch = tensor(np.asarray(x), device)
    if batch.ndim != 2 or len(batch) == 0:
        raise ValueError("Inference timing requires a nonempty feature batch")
    domain = batch[:, -1].long()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    model.eval()
    with torch.inference_mode():
        for _ in range(int(warmup)):
            model(batch, domain=domain)
        durations = []
        for _ in range(int(repetitions)):
            synchronize()
            started = time.perf_counter()
            model(batch, domain=domain)
            synchronize()
            durations.append((time.perf_counter() - started) * 1000.0)
    return {"batch_size": int(len(batch)), "device": str(device), "warmup": int(warmup),
            "repetitions": int(repetitions), "p50_ms": float(np.percentile(durations, 50)),
            "p95_ms": float(np.percentile(durations, 95))}

