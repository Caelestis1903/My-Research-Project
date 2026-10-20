"""Stage R2 final-test evaluator — test-side only component.

This is Component 4 of the four-component architecture (Section 4.3 of
the design spec).  It lives in its own module to keep the training
runner free of test references — the static grep check on
``ao04_v12_runner.py`` therefore returns zero matches on this file's
bodies.

The evaluator is gated by the global unlock manifest
(``locks/global_test_unlock.json``) and bumps the module-level
``TEST_PREDICTION_CALLS`` counter for every load of a test
checkpoint.  The decision module asserts this counter is zero before
publishing any R2-A verdict.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

R2_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(R2_ROOT))
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")
M3OE_ROOT = os.path.join(PROJECT_ROOT, "m3oe")
for _path in (PROJECT_ROOT, EXPERIMENTS_ROOT, M3OE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from MDMTRec.utils.data import TorchDataset  # noqa: E402
from stage_S1_data import load_stage_s1_bundle  # noqa: E402
from stage_S1_metrics import safe_auc, safe_logloss  # noqa: E402
from stage_S1_models import features_from_schema  # noqa: E402

from ao04_v12_model import build_ao04_model  # noqa: E402
from ao04_v12_protocol import (  # noqa: E402
    BATCH_SIZE,
    COMMON_EPOCHS,
    DATASETS,
    DATASET_SETTINGS,
    DEFAULT_DEVICE,
    DOMAIN_NUM,
    TASK_NUM,
)
import ao04_v12_runner as _runner_mod  # noqa: E402
from ao04_v12_runner import setup_seed  # noqa: E402


def _bump_test_counter() -> None:
    """Increment the runner's test-prediction counter (test-side only).

    Must go through the runner module object so the decision module's
    zero-assertion on ``ao04_v12_runner.TEST_PREDICTION_CALLS`` actually
    observes the bump.
    """

    _runner_mod._bump_test_counter()


@dataclass
class FinalTestArtifacts:
    metrics_test_csv: str
    predictions_test_npz: str
    augmented_manifest_path: str


def _require_finite(label: str, value) -> float:
    import math
    number = float(value)
    if not math.isfinite(number):
        raise FloatingPointError("{} is non-finite: {}".format(label, value))
    return number


def _evaluate_test_once(model, loader: DataLoader,
                        device: torch.device,
                        bundle) -> Tuple[Dict[int, Dict[str, float]],
                                          np.ndarray]:
    truth: Dict[int, list] = {t * DOMAIN_NUM + d: []
                              for t in range(TASK_NUM)
                              for d in range(DOMAIN_NUM)}
    predictions: Dict[int, list] = {k: [] for k in truth}
    pred_parts: list = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_device = x_batch.transpose(0, 1).to(device=device,
                                                  dtype=torch.float32)
            y_device = y_batch.to(device=device, dtype=torch.float32)
            pred = model(x_device)
            if not bool(torch.isfinite(pred).all().detach().cpu()):
                raise FloatingPointError("non-finite test prediction")
            pred_parts.append(pred.detach().cpu().numpy().astype(np.float32,
                                                                 copy=False))
            domain_ids = x_device[-1, :].long()
            for t in range(TASK_NUM):
                for d in range(DOMAIN_NUM):
                    key = t * DOMAIN_NUM + d
                    mask = domain_ids == d
                    if bool(mask.any()):
                        truth[key].extend(y_device[mask, t].detach().cpu().tolist())
                        predictions[key].extend(pred[mask, t].detach().cpu().tolist())
    stacked_pred = (np.concatenate(pred_parts, axis=0)
                    if pred_parts else np.zeros((0, TASK_NUM), dtype=np.float32))
    result: Dict[int, Dict[str, float]] = {}
    for key in truth:
        labels = truth[key]
        scores = predictions[key]
        if (len(labels) == 0
                or len({float(v) for v in labels}) < 2):
            raise ValueError("test cell {} empty or single class".format(key))
        result[key] = {
            "sample_count": int(len(labels)),
            "positive_count": int(sum(float(v) for v in labels)),
            "positive_rate": float(np.mean(labels)),
            "auc": _require_finite("auc", safe_auc(labels, scores)),
            "logloss": _require_finite("logloss", safe_logloss(labels, scores)),
        }
    return result, stacked_pred


def final_test_evaluator(run_dirs: Mapping[Tuple[str, int, str], str],
                         *,
                         unlock_manifest_path: str,
                         dataset: str,
                         setting: str,
                         backbones: Sequence[str] = ("m3oe",),
                         ) -> FinalTestArtifacts:
    """Single authorized test pass on selected / final / common checkpoints."""

    if dataset not in DATASETS:
        raise ValueError("unsupported dataset: {}".format(dataset))
    if setting not in DATASET_SETTINGS[dataset]:
        raise ValueError("invalid setting {} for {}".format(setting, dataset))
    if not os.path.exists(unlock_manifest_path):
        raise FileNotFoundError(
            "global unlock manifest must exist before final_test_evaluator runs: {!r}".format(
                unlock_manifest_path))
    with open(unlock_manifest_path, encoding="utf-8") as h:
        unlock = json.load(h)
    if unlock.get("global_test_unlock") is not True:
        raise RuntimeError("global unlock manifest does not grant unlock")

    bundle = load_stage_s1_bundle(dataset, None, setting, seed=2022)
    test_x = bundle.test_x
    test_y = bundle.test_y
    test_row_ids = bundle.test_row_ids
    test_domains_local = bundle.test_x[:, -1].astype(np.int64)

    write_root = os.path.dirname(os.path.dirname(run_dirs[next(iter(run_dirs))]))
    out_dir = os.path.join(write_root, "_final_test_evaluated")
    os.makedirs(out_dir, exist_ok=True)

    test_records = []
    predictions_payload = {"row_ids": test_row_ids,
                           "physical_domain": np.asarray(
                               [bundle.local_to_physical[int(v)]
                                for v in test_domains_local],
                               dtype=np.int64),
                           "labels": np.asarray(test_y, dtype=np.float32)}
    per_run_records: Dict[str, list] = {}
    per_run_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    _bump_test_counter()

    for (arm, seed, backbone), run_dir in run_dirs.items():
        _bump_test_counter()
        if backbone != "m3oe":
            continue
        run_key = "{}_seed{}".format(arm, seed)
        per_run_records[run_key] = []
        per_run_predictions[run_key] = {
            "row_ids": test_row_ids,
            "labels": np.asarray(test_y, dtype=np.float32),
        }
        setup_seed(seed)
        model = build_ao04_model(features_from_schema(bundle.features),
                                 arm, seed=seed)
        device = torch.device(DEFAULT_DEVICE if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        ckpts = {
            "selected": torch.load(
                os.path.join(run_dir, "selected_checkpoint.pth"),
                map_location=device),
            "final": torch.load(os.path.join(run_dir, "final_checkpoint.pth"),
                                map_location=device),
        }
        for epoch in COMMON_EPOCHS:
            ckpts["common_epoch_{}".format(epoch)] = torch.load(
                os.path.join(run_dir, "common_epoch_{}.pth".format(epoch)),
                map_location=device)
        test_loader = DataLoader(
            TorchDataset(test_x, test_y),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
            drop_last=False)
        for ck_name, state in ckpts.items():
            model.load_state_dict(state, strict=True)
            model.eval()
            record, preds = _evaluate_test_once(model, test_loader, device, bundle)
            predictions_payload["pred_{}_{}_seed{}".format(arm, ck_name, seed)] = preds
            per_run_predictions[run_key]["pred_{}".format(ck_name)] = preds
            for key, cell_row in record.items():
                t = key // DOMAIN_NUM
                d = key % DOMAIN_NUM
                row = {
                    "arm": arm, "seed": seed, "checkpoint": ck_name,
                    "dataset": dataset, "setting": setting, "backbone": backbone,
                    "task": t, "domain": d,
                    "physical_domain": int(bundle.local_to_physical[d]),
                    "cell": key,
                    "test_sample_count": int(cell_row["sample_count"]),
                    "test_positive_count": int(cell_row["positive_count"]),
                    "test_positive_rate": float(cell_row["positive_rate"]),
                    "test_auc": float(cell_row["auc"]),
                    "test_logloss": float(cell_row["logloss"]),
                }
                test_records.append(row)
                per_run_records[run_key].append(row)

    # Per-run artifacts required by Specification §7: metrics_test.csv and
    # predictions_test.npz inside each evaluated run directory.
    from stage_E1_instrumentation import write_json, write_rows_csv  # noqa: E402
    per_run_written: Dict[str, Dict[str, str]] = {}
    for (arm, seed, backbone), run_dir in run_dirs.items():
        if backbone != "m3oe":
            continue
        run_key = "{}_seed{}".format(arm, seed)
        if run_key not in per_run_records:
            continue
        run_csv = os.path.join(run_dir, "metrics_test.csv")
        run_fields = sorted({k for row in per_run_records[run_key] for k in row.keys()})
        write_rows_csv(run_csv, per_run_records[run_key], run_fields)
        run_npz = os.path.join(run_dir, "predictions_test.npz")
        np.savez_compressed(run_npz, **per_run_predictions[run_key])
        per_run_written[run_key] = {"metrics_test_csv": run_csv,
                                    "predictions_test_npz": run_npz}

    npz_path = os.path.join(out_dir, "predictions_test.npz")
    np.savez_compressed(npz_path, **predictions_payload)
    csv_path = os.path.join(out_dir, "metrics_test.csv")
    fields = sorted({k for row in test_records for k in row.keys()})
    write_rows_csv(csv_path, test_records, fields)
    aug_path = os.path.join(out_dir, "test_evaluation_manifest.json")
    write_json(aug_path, {
        "dataset": dataset,
        "setting": setting,
        "unlock_manifest_path": unlock_manifest_path,
        "records_count": len(test_records),
        "npz_path": npz_path,
        "csv_path": csv_path,
        "per_run_artifacts": per_run_written,
        "test_prediction_calls_global": int(_runner_mod.TEST_PREDICTION_CALLS),
    })
    return FinalTestArtifacts(metrics_test_csv=csv_path,
                              predictions_test_npz=npz_path,
                              augmented_manifest_path=aug_path)


__all__ = ["FinalTestArtifacts", "final_test_evaluator"]
