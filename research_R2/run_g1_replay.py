"""Stage R2 G1 reload-replay validator.

Implements the Specification §9 G1 gate for one B0 run (the queue
invokes it once per seed after both B0 runs of a dataset COMPLETE, and
requires both invocations to PASS before any B1/B2 run starts):

    * G1-1  run status COMPLETE, full epoch schedule completed,
            validation log complete, and the selected epoch re-derived
            offline with ``select_checkpoint_epoch`` (strict >, earliest
            tie) matches the recorded one.
    * G1-2  checkpoint / optimizer / RNG / init / exposure artefacts
            present with SHA256 recorded in this report.
    * G1-3  strict-load ``selected_checkpoint.pth`` and replay the same
            validation split: prediction max_abs <= 1e-6 against the
            runner-recorded ``validation_predictions_selected.npz``;
            six-cell and macro AUC / LogLoss |diff| <= 1e-6 against
            ``metrics_epoch.csv`` at the selected epoch.
    * G1-4  paired-init artefacts (init_state / init_rng / init_manifest)
            exist for this (dataset, seed), the manifest's
            ``paired_init_manifest_sha256`` matches the on-disk file, the
            recorded initial parameter hash matches the init manifest
            (B0), and the sibling-seed B0 run of the same dataset is
            COMPLETE with the same epoch count so the pair is ready for
            B1/B2 replication.
    * G1-5  the test tripwire counter ``TEST_PREDICTION_CALLS`` is still
            0 after the replay and the manifest attests no test
            evaluation / prediction happened during training.

No test array is touched.  Every exit path writes the JSON report.

CLI::

    venv/Scripts/python.exe research_R2/run_g1_replay.py \\
        --config research_R2/configs/kuairand_natural_m3oe.json \\
        --run-dir runs/kuairand/natural/m3oe/B0/seed2022/attempt1 \\
        [--seed 2022]                 # must match the manifest's seed
        [--write-to <acceptance.json>]    # default <run_dir>/g1_replay.json

Hidden flag (orchestration self-test / smoke only)::

    --allow-smoke   accept a run whose epoch count equals its own
                    config's ``epochs`` instead of the formal 50.  The
                    queue never passes this flag.

Exit codes::

    0  - PASS (every gate G1-1..G1-5 passed)
    2  - bad arguments
    3  - ERROR: artefacts missing / unreadable (report still written)
    4  - INVALID: run manifest status not COMPLETE
    5  - FAIL: at least one gate violated (report still written)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
R2_SRC = os.path.join(HERE, "src")
PROJECT_ROOT = os.path.dirname(HERE)
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")
M3OE_ROOT = os.path.join(PROJECT_ROOT, "m3oe")
for p in (R2_SRC, PROJECT_ROOT, EXPERIMENTS_ROOT, M3OE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from MDMTRec.utils.data import TorchDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from stage_S1_data import load_stage_s1_bundle  # noqa: E402
from stage_S1_metrics import safe_auc, safe_logloss  # noqa: E402
from stage_S1_models import features_from_schema  # noqa: E402
from stage_E1_instrumentation import write_json  # noqa: E402

from ao04_v12_model import build_ao04_model  # noqa: E402
from ao04_v12_protocol import (  # noqa: E402
    BATCH_SIZE,
    COMMON_EPOCHS,
    DATASETS,
    DATASET_SETTINGS,
    DEFAULT_DEVICE,
    DOMAIN_NUM,
    EPOCHS,
    SEEDS,
    TASK_NUM,
)
import ao04_v12_runner as _runner_mod  # noqa: E402
from ao04_v12_runner import select_checkpoint_epoch, setup_seed  # noqa: E402


# Numerical acceptance thresholds (Specification §9 G1-3).
PRED_MAX_ABS = 1e-6
METRIC_MAX_ABS = 1e-6

N_CELLS = TASK_NUM * DOMAIN_NUM


# ---------------------------------------------------------------------------
# Small IO helpers.
# ---------------------------------------------------------------------------

def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as h:
            return json.load(h)
    except Exception:
        return {}


def _read_metrics_epoch(metrics_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(metrics_path):
        return []
    with open(metrics_path, encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


# ---------------------------------------------------------------------------
# Validation replay (mirrors the runner's eval loop exactly).
# ---------------------------------------------------------------------------

def _validation_records(bundle, model, device: torch.device) -> Tuple[
        Dict[int, Dict[str, Any]], np.ndarray, np.ndarray]:
    """Run validation forward once; return per-cell metrics + raw arrays."""

    truth: Dict[int, List[float]] = {t * DOMAIN_NUM + d: []
                                     for t in range(TASK_NUM)
                                     for d in range(DOMAIN_NUM)}
    predictions: Dict[int, List[float]] = {k: [] for k in truth}
    pred_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    model.eval()
    val_loader = DataLoader(
        TorchDataset(bundle.val_x, bundle.val_y),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_device = x_batch.transpose(0, 1).to(device=device,
                                                  dtype=torch.float32)
            y_device = y_batch.to(device=device, dtype=torch.float32)
            pred = model(x_device)
            if not bool(torch.isfinite(pred).all().detach().cpu()):
                raise FloatingPointError("non-finite replay prediction")
            pred_parts.append(pred.detach().cpu().numpy().astype(
                np.float32, copy=False))
            label_parts.append(y_batch.detach().cpu().numpy().astype(
                np.float32, copy=False))
            domain_ids = x_device[-1, :].long()
            for t in range(TASK_NUM):
                for d in range(DOMAIN_NUM):
                    key = t * DOMAIN_NUM + d
                    mask = domain_ids == d
                    if bool(mask.any()):
                        truth[key].extend(
                            y_device[mask, t].detach().cpu().tolist())
                        predictions[key].extend(
                            pred[mask, t].detach().cpu().tolist())
    stacked_pred = (np.concatenate(pred_parts, axis=0)
                    if pred_parts else np.zeros((0, TASK_NUM), dtype=np.float32))
    stacked_label = (np.concatenate(label_parts, axis=0)
                     if label_parts else np.zeros((0, TASK_NUM), dtype=np.float32))
    metrics: Dict[int, Dict[str, Any]] = {}
    for key in truth:
        labels = truth[key]
        scores = predictions[key]
        if (len(labels) == 0
                or len({float(v) for v in labels}) < 2):
            raise ValueError("replay cell {} empty or single class".format(key))
        metrics[key] = {
            "sample_count": int(len(labels)),
            "positive_count": int(sum(float(v) for v in labels)),
            "positive_rate": float(np.mean(labels)),
            "auc": float(safe_auc(labels, scores)),
            "logloss": float(safe_logloss(labels, scores)),
        }
    return metrics, stacked_pred, stacked_label


def _per_cell_metrics(rows: List[Mapping[str, Any]], epoch: int
                      ) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for row in rows:
        if int(row["epoch"]) != epoch:
            continue
        cell = int(row["cell"])
        out[cell] = {
            "auc": float(row["validation_auc"]),
            "logloss": float(row["validation_logloss"]),
        }
    return out


def _sibling_b0_run_dir(run_dir: str, seed: int) -> str:
    """Derive the same-dataset B0 run dir for the *other* seed."""

    other = SEEDS[0] if int(seed) == SEEDS[1] else SEEDS[1]
    token = "seed{}".format(int(seed))
    if token not in run_dir:
        return ""
    return run_dir.replace(token, "seed{}".format(int(other)))


# ---------------------------------------------------------------------------
# Main gate.
# ---------------------------------------------------------------------------

def run_replay(config_path: str, run_dir: str, *,
               seed: int, write_to: str,
               allow_smoke: bool = False) -> Dict[str, Any]:
    """Run the full G1 gate for one B0 run and persist the JSON report."""

    os.makedirs(os.path.dirname(os.path.abspath(write_to)) or ".",
                exist_ok=True)

    report: Dict[str, Any] = {
        "status": "ERROR",
        "dataset": None,
        "arm": None,
        "seed": int(seed),
        "run_dir": os.path.abspath(run_dir),
        "config_path": os.path.abspath(config_path),
        "allow_smoke": bool(allow_smoke),
        "gates": {},
        "violations": [],
        "test_evaluated_during_replay": False,
        "test_prediction_calls_after_replay": None,
        "thresholds": {
            "prediction_max_abs": PRED_MAX_ABS,
            "metric_max_abs": METRIC_MAX_ABS,
        },
    }

    def _emit() -> Dict[str, Any]:
        # G1-5 is evaluated last but recorded on every exit path.
        report["test_prediction_calls_after_replay"] = int(
            _runner_mod.TEST_PREDICTION_CALLS)
        write_json(write_to, report)
        return report

    def _fail_early(status: str, reason: str) -> Dict[str, Any]:
        report["status"] = status
        report["reason"] = reason
        return _emit()

    if seed not in (2022, 2023):
        raise ValueError("seed must be 2022 or 2023 (got {})".format(seed))

    cfg = _read_json(config_path)
    if not cfg:
        return _fail_early("ERROR",
                           "config file empty / unreadable: {}".format(config_path))
    dataset = str(cfg.get("dataset", ""))
    if dataset not in DATASETS:
        raise ValueError("unsupported dataset {}".format(dataset))
    report["dataset"] = dataset

    manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
    if not manifest:
        return _fail_early("ERROR", "run_manifest.json missing or unreadable")
    report["arm"] = manifest.get("arm")

    if str(manifest.get("status", "")).upper() != "COMPLETE":
        report["manifest_status"] = manifest.get("status")
        return _fail_early(
            "INVALID", "run status = {}".format(manifest.get("status")))

    arm = str(manifest.get("arm"))
    if arm != "B0":
        # G1 is specified for the B0 pair only; anything else is a
        # caller bug and must not silently pass.
        report["manifest_status"] = manifest.get("status")
        return _fail_early(
            "INVALID", "G1 gate applies to B0 runs only (got {})".format(arm))
    if int(manifest.get("seed", -1)) != int(seed):
        return _fail_early(
            "ERROR", "manifest seed {} != requested seed {}".format(
                manifest.get("seed"), seed))
    if manifest.get("selected_epoch") is None:
        return _fail_early("ERROR", "manifest lacks selected_epoch")

    selected_epoch = int(manifest.get("selected_epoch"))
    violations: List[str] = report["violations"]

    run_cfg = _read_json(os.path.join(run_dir, "config.json"))
    expected_epochs = (int(run_cfg.get("epochs", EPOCHS)) if allow_smoke
                       else int(EPOCHS))

    # ---------------- G1-1: completeness + selection recompute ----------------
    g1_1: Dict[str, Any] = {"expected_epochs": expected_epochs}
    epochs_completed = int(manifest.get("epochs_completed") or -1)
    g1_1["epochs_completed"] = epochs_completed
    if epochs_completed != expected_epochs:
        violations.append(
            "G1-1 epochs_completed={} != expected {}".format(
                epochs_completed, expected_epochs))
    selection = _read_json(os.path.join(run_dir, "selection.json"))
    if not selection:
        violations.append("G1-1 selection.json missing or unreadable")
        g1_1["selection_json"] = "missing"
    else:
        macro_by_epoch = {int(k): float(v) for k, v in
                          selection.get("macro_auc_by_epoch", {}).items()}
        g1_1["selection_epochs"] = len(macro_by_epoch)
        if len(macro_by_epoch) != expected_epochs:
            violations.append(
                "G1-1 selection.json covers {} epochs != {}".format(
                    len(macro_by_epoch), expected_epochs))
        if int(selection.get("selected_epoch", -1)) != selected_epoch:
            violations.append(
                "G1-1 selection.json selected_epoch {} != manifest {}".format(
                    selection.get("selected_epoch"), selected_epoch))
        if macro_by_epoch:
            recomputed = select_checkpoint_epoch(macro_by_epoch)
            g1_1["recomputed_selected_epoch"] = recomputed
            if recomputed != selected_epoch:
                violations.append(
                    "G1-1 select_checkpoint_epoch recompute {} != recorded {}".format(
                        recomputed, selected_epoch))
    metrics_path = os.path.join(run_dir, "metrics_epoch.csv")
    metrics_rows = _read_metrics_epoch(metrics_path)
    g1_1["metrics_rows"] = len(metrics_rows)
    if len(metrics_rows) != expected_epochs * N_CELLS:
        violations.append(
            "G1-1 metrics_epoch.csv rows={} != {} epochs x {} cells".format(
                len(metrics_rows), expected_epochs, N_CELLS))
    original = _per_cell_metrics(metrics_rows, selected_epoch)
    if len(original) != N_CELLS:
        violations.append(
            "G1-1 selected epoch {} has {} cells in metrics_epoch.csv (!= {})".format(
                selected_epoch, len(original), N_CELLS))
    g1_1["pass"] = not any(v.startswith("G1-1") for v in violations)
    report["gates"]["G1-1"] = g1_1

    # ---------------- G1-2: artefact completeness + hashes ----------------
    artefacts = [
        "selected_checkpoint.pth",
        "final_checkpoint.pth",
        "optimizer_state.pth",
        "architecture_optimizer_state.pth",
        "rng_state.pth",
        "metrics_epoch.csv",
        "selection.json",
        "parameter_manifest.json",
        "config.json",
        "run_manifest.json",
        "validation_predictions_selected.npz",
        "last_resume.pt",
    ]
    for common_epoch in COMMON_EPOCHS:
        if common_epoch < expected_epochs:
            artefacts.append("common_epoch_{}.pth".format(common_epoch))
    artefact_hashes: Dict[str, str] = {}
    missing: List[str] = []
    for name in artefacts:
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            artefact_hashes[name] = _file_sha256(path)
        else:
            missing.append(name)
    g1_2: Dict[str, Any] = {"missing": missing,
                            "artefact_sha256": artefact_hashes}
    if missing:
        violations.append("G1-2 missing artefacts: {}".format(missing))
    npz_path = os.path.join(run_dir, "validation_predictions_selected.npz")
    manifest_npz_sha = manifest.get("selected_validation_predictions_sha256")
    if os.path.isfile(npz_path):
        actual_npz_sha = artefact_hashes["validation_predictions_selected.npz"]
        g1_2["manifest_selected_validation_predictions_sha256"] = manifest_npz_sha
        if manifest_npz_sha and manifest_npz_sha != actual_npz_sha:
            violations.append(
                "G1-2 manifest predictions sha {} != file {}".format(
                    manifest_npz_sha, actual_npz_sha))
    g1_2["pass"] = not any(v.startswith("G1-2") for v in violations)
    report["gates"]["G1-2"] = g1_2

    if missing:
        # Cannot continue with the replay without the checkpoints.
        report["gates"]["G1-3"] = {"pass": False, "skipped": "artefacts missing"}
        report["gates"]["G1-4"] = {"pass": False, "skipped": "artefacts missing"}
        report["gates"]["G1-5"] = {
            "pass": int(_runner_mod.TEST_PREDICTION_CALLS) == 0,
            "test_prediction_calls": int(_runner_mod.TEST_PREDICTION_CALLS),
        }
        return _fail_early("ERROR", "required artefacts missing: {}".format(missing))

    # ---------------- G1-3: strict-load replay ----------------
    selected_path = os.path.join(run_dir, "selected_checkpoint.pth")
    g1_3: Dict[str, Any] = {}
    try:
        setup_seed(seed)
        bundle = load_stage_s1_bundle(dataset, cfg.get("dataset_path"),
                                      DATASET_SETTINGS[dataset][0], seed)
        feats = features_from_schema(bundle.features)
        model = build_ao04_model(feats, arm, seed=seed)
        state = torch.load(selected_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        g1_3["pass"] = False
        g1_3["error"] = "strict-load-failed: {}".format(exc)
        report["gates"]["G1-3"] = g1_3
        return _fail_early("ERROR", g1_3["error"])

    # Replay on the same device the run recorded — the 1e-6 tolerance
    # only holds for a same-device reload.
    device_name = str(run_cfg.get("device", DEFAULT_DEVICE))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        g1_3["pass"] = False
        g1_3["error"] = "run device {} unavailable for replay".format(device_name)
        report["gates"]["G1-3"] = g1_3
        return _fail_early("ERROR", g1_3["error"])
    device = torch.device(device_name)
    g1_3["replay_device"] = device_name
    model = model.to(device)

    metrics, replay_pred, replay_label = _validation_records(bundle, model, device)

    stored = np.load(npz_path)
    stored_pred = np.asarray(stored["predictions"], dtype=np.float32)
    stored_label = np.asarray(stored["labels"], dtype=np.float32)
    if stored_pred.shape != replay_pred.shape:
        violations.append(
            "G1-3 prediction shape mismatch: stored {} vs replay {}".format(
                tuple(stored_pred.shape), tuple(replay_pred.shape)))
        max_pred_abs = float("inf")
    else:
        max_pred_abs = float(
            np.abs(replay_pred.astype(np.float64)
                   - stored_pred.astype(np.float64)).max())
        if max_pred_abs > PRED_MAX_ABS:
            violations.append(
                "G1-3 prediction max_abs {:.3e} > {}".format(
                    max_pred_abs, PRED_MAX_ABS))
    labels_equal = bool(stored_label.shape == replay_label.shape
                        and np.array_equal(stored_label, replay_label))
    if not labels_equal:
        violations.append("G1-3 stored validation labels differ from replay labels")
    g1_3["max_prediction_abs_diff"] = max_pred_abs
    g1_3["labels_equal"] = labels_equal

    per_cell_diffs: Dict[str, Dict[str, float]] = {}
    for cell in sorted(set(original) | set(metrics)):
        rec_orig = original.get(cell)
        rec_replay = metrics.get(cell)
        if rec_orig is None or rec_replay is None:
            violations.append(
                "G1-3 cell {} present in only one of original/replay".format(cell))
            continue
        diff_auc = float(rec_orig["auc"] - rec_replay["auc"])
        diff_ll = float(rec_orig["logloss"] - rec_replay["logloss"])
        per_cell_diffs[str(cell)] = {"auc_diff": diff_auc,
                                     "logloss_diff": diff_ll}
        if not _is_finite(diff_auc) or abs(diff_auc) > METRIC_MAX_ABS:
            violations.append(
                "G1-3 cell={} auc_diff={:.3e}".format(cell, diff_auc))
        if not _is_finite(diff_ll) or abs(diff_ll) > METRIC_MAX_ABS:
            violations.append(
                "G1-3 cell={} logloss_diff={:.3e}".format(cell, diff_ll))
    g1_3["per_cell_diffs"] = per_cell_diffs

    macro_auc_replay = float(np.mean([metrics[k]["auc"] for k in metrics]))
    macro_ll_replay = float(np.mean([metrics[k]["logloss"] for k in metrics]))
    macro_auc_orig_raw = manifest.get("selected_validation_macro_auc")
    if macro_auc_orig_raw is None:
        violations.append("G1-3 manifest lacks selected_validation_macro_auc")
        macro_auc_orig = float("nan")
    else:
        macro_auc_orig = float(macro_auc_orig_raw)
    macro_ll_orig = float(np.mean([original[k]["logloss"] for k in original]))
    macro_diff = {
        "macro_auc_diff": float(macro_auc_orig - macro_auc_replay),
        "macro_logloss_diff": float(macro_ll_orig - macro_ll_replay),
    }
    g1_3["macro_diff"] = macro_diff
    if abs(macro_diff["macro_auc_diff"]) > METRIC_MAX_ABS:
        violations.append("G1-3 macro_auc_diff {:.3e} > {}".format(
            macro_diff["macro_auc_diff"], METRIC_MAX_ABS))
    if abs(macro_diff["macro_logloss_diff"]) > METRIC_MAX_ABS:
        violations.append("G1-3 macro_logloss_diff {:.3e} > {}".format(
            macro_diff["macro_logloss_diff"], METRIC_MAX_ABS))
    g1_3["pass"] = not any(v.startswith("G1-3") for v in violations)
    report["gates"]["G1-3"] = g1_3

    # ---------------- G1-4: paired-init / exposure readiness ----------------
    g1_4: Dict[str, Any] = {}
    paired_state_path = str(manifest.get("paired_init_state_path", ""))
    init_dir_for_seed = (os.path.dirname(os.path.abspath(paired_state_path))
                         if paired_state_path else "")
    init_state_path = os.path.join(init_dir_for_seed, "init_state.pt")
    init_rng_path = os.path.join(init_dir_for_seed, "init_rng.pt")
    init_manifest_path = os.path.join(init_dir_for_seed, "init_manifest.json")
    init_manifest = _read_json(init_manifest_path)
    g1_4["init_dir"] = init_dir_for_seed
    for name, path in (("init_state.pt", init_state_path),
                       ("init_rng.pt", init_rng_path),
                       ("init_manifest.json", init_manifest_path)):
        exists = os.path.isfile(path)
        g1_4[name] = {"exists": exists,
                      "sha256": _file_sha256(path) if exists else None}
        if not exists:
            violations.append("G1-4 init artefact missing: {}".format(path))
    if init_manifest:
        recorded = str(manifest.get("paired_init_manifest_sha256", ""))
        actual = _file_sha256(init_manifest_path)
        g1_4["paired_init_manifest_sha256_match"] = bool(recorded == actual)
        if recorded != actual:
            violations.append(
                "G1-4 paired_init_manifest_sha256 mismatch: manifest {} != file {}".format(
                    recorded, actual))
        init_hash = str(init_manifest.get("initial_state_hash", ""))
        run_init_hash = str(manifest.get("initial_parameter_state_hash", ""))
        g1_4["initial_state_hash_match"] = bool(init_hash
                                                and init_hash == run_init_hash)
        if not g1_4["initial_state_hash_match"]:
            violations.append(
                "G1-4 run initial_parameter_state_hash {} != init manifest {}".format(
                    run_init_hash, init_hash))
    # Exposure evidence: batch order hashes must be recorded for B1/B2 to
    # reproduce the same exposure sequence.
    for field_name in ("batch_order_hash", "architecture_batch_order_hash",
                       "batch_fingerprint_count",
                       "architecture_batch_fingerprint_count"):
        value = manifest.get(field_name)
        g1_4[field_name] = value
        if value in (None, ""):
            violations.append("G1-4 exposure field {!r} not recorded".format(
                field_name))
    # Sibling-seed B0 of the same dataset must also be COMPLETE so the
    # pair is ready for B1/B2.
    sibling_dir = _sibling_b0_run_dir(os.path.abspath(run_dir), seed)
    sibling = _read_json(os.path.join(sibling_dir, "run_manifest.json")) \
        if sibling_dir else {}
    g1_4["sibling_run_dir"] = sibling_dir
    g1_4["sibling_status"] = sibling.get("status")
    if str(sibling.get("status", "")).upper() != "COMPLETE":
        violations.append(
            "G1-4 sibling B0 run not COMPLETE: {} (status={})".format(
                sibling_dir, sibling.get("status")))
    elif int(sibling.get("epochs_completed") or -1) != expected_epochs:
        violations.append(
            "G1-4 sibling B0 epochs_completed={} != {}".format(
                sibling.get("epochs_completed"), expected_epochs))
    g1_4["pass"] = not any(v.startswith("G1-4") for v in violations)
    report["gates"]["G1-4"] = g1_4

    # ---------------- G1-5: test tripwire counter ----------------
    calls = int(_runner_mod.TEST_PREDICTION_CALLS)
    g1_5: Dict[str, Any] = {
        "test_prediction_calls": calls,
        "manifest_test_evaluated_during_training":
            manifest.get("test_evaluated_during_training"),
        "manifest_test_predictions_written":
            manifest.get("test_predictions_written"),
    }
    if calls != 0:
        violations.append("G1-5 TEST_PREDICTION_CALLS={} != 0".format(calls))
    if manifest.get("test_evaluated_during_training") is not False:
        violations.append("G1-5 manifest test_evaluated_during_training != False")
    if manifest.get("test_predictions_written") is not False:
        violations.append("G1-5 manifest test_predictions_written != False")
    g1_5["pass"] = not any(v.startswith("G1-5") for v in violations)
    report["gates"]["G1-5"] = g1_5

    report["selected_epoch"] = selected_epoch
    report["selected_checkpoint_path"] = selected_path
    report["status"] = "PASS" if not violations else "FAIL"
    return _emit()


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ao04-g1-replay",
                                description="Stage R2 G1 reload-replay validator.")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seed", type=int, required=True,
                   help="Formal Stage R2 seed (must match the manifest).")
    p.add_argument("--write-to", default=None,
                   help="Output JSON path (default: <run_dir>/g1_replay.json).")
    p.add_argument("--allow-smoke", action="store_true",
                   help="Hidden self-test flag: accept the run's own config "
                        "epoch count instead of the formal 50.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    write_to = args.write_to or os.path.join(args.run_dir, "g1_replay.json")
    report = run_replay(args.config, args.run_dir,
                        seed=args.seed, write_to=write_to,
                        allow_smoke=bool(args.allow_smoke))
    print("[g1_replay] status={} run_dir={} report={}".format(
        report.get("status"), args.run_dir, write_to))
    if report.get("violations"):
        for v in report["violations"]:
            print("[g1_replay] violation:", v)
    if report.get("status") == "ERROR":
        return 3
    if report.get("status") == "INVALID":
        return 4
    return 0 if report.get("status") == "PASS" else 5


if __name__ == "__main__":
    raise SystemExit(main())
