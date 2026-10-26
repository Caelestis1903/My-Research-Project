"""Structural/raw collector for Stage E6.

The collector always enumerates all 24 registered paths and copies raw
seed/cell observations into summary and ablation CSVs.  It does not compare
arms or write a scientific conclusion; that belongs to the main analysis
stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from stage_E6_calibration import E6_ARM_KEYS, E6_FORMAL_SETTING, E6_EXPECTED_SEEDS
from stage_E6_orchestrator import (DEFAULT_RAW_ROOT, REQUIRED_ARTIFACTS, build_tasks,
                                   compare_e6_block, validate_matrix_tasks)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(HERE, "stage_E6_summary.csv")
DEFAULT_ABLATION = os.path.join(HERE, "stage_E6_ablation.csv")
DEFAULT_MANIFEST = os.path.join(HERE, "stage_E6_summary_manifest.json")
DEFAULT_RUN_REPORT = os.path.join(HERE, "reports", "stage_E6_run_report.md")

SUMMARY_FIELDS = [
    "stage", "status", "arm_key", "setting", "seed", "domain", "task", "cell",
    "size_role", "domain_size", "domain_share", "selected_epoch", "stopped_epoch",
    "best_val_auc", "best_val_logloss", "best_test_auc", "best_test_logloss",
    "oracle_auc", "oracle_logloss", "oracle_source", "oracle_coverage", "transfer_auc",
    "transfer_logloss", "overall_auc", "overall_logloss", "worst_domain_auc",
    "worst_cell_auc", "negative_transfer", "shared_raw_gradient_norm",
    "shared_weighted_gradient_norm", "shared_magnitude_share", "shared_signed_contribution",
    "shared_mismatch_index", "target_calibration_error", "coefficient_variation",
    "ema_tracking_error", "gradient_variance", "gradient_SNR", "shared_update_alignment",
    "common_support", "last_common_epoch", "last_common_test_auc",
    "last_common_test_logloss", "last_common_transfer_auc", "last_common_transfer_logloss",
    "censored", "run_dir",
]
ABLATION_FIELDS = [
    "comparison", "stage", "status", "arm_key", "setting", "seed", "domain", "task", "cell",
    "best_test_auc", "best_test_logloss", "overall_auc", "overall_logloss", "worst_domain_auc",
    "worst_cell_auc", "transfer_auc", "transfer_logloss", "negative_transfer",
    "shared_mismatch_index", "target_calibration_error", "coefficient_variation",
    "ema_tracking_error", "gradient_variance", "gradient_SNR", "selected_epoch",
    "stopped_epoch", "last_common_epoch", "last_common_test_auc", "last_common_test_logloss",
    "last_common_transfer_auc", "last_common_transfer_logloss", "run_dir",
]


def _read_json(path: str) -> Dict[str, object]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _read_csv(path: str) -> List[Dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def _write_csv(path: str, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def expected_runs(raw_root: str = DEFAULT_RAW_ROOT) -> List[Dict[str, object]]:
    return build_tasks(raw_root)


def _run_status(run_dir: str, manifest: Mapping[str, object],
                block: Optional[Mapping[str, object]] = None) -> Tuple[str, List[str]]:
    missing = [name for name in REQUIRED_ARTIFACTS
               if not os.path.exists(os.path.join(run_dir, name))]
    if missing:
        return "MISSING", missing
    if manifest.get("stage") != "E6" or manifest.get("status") != "OK":
        return "INVALID", ["manifest_stage_or_status_invalid"]
    if manifest.get("formal_valid") is not True:
        return "INVALID", ["formal_invalid"]
    # A run is reportable only when the fresh six-arm comparator and the
    # persisted block gate both agree.  This prevents incomparable runs from
    # being relabelled OK merely because their local files are complete.
    if block is None or block.get("status") != "VALID" or block.get("valid") is not True:
        return "INVALID", ["block_invalid_or_pending"]
    if manifest.get("block_valid") is not True or manifest.get("block_status") != "VALID":
        return "INVALID", ["manifest_block_gate_invalid"]
    return "OK", []


def _seed_block(raw_root: str, seed: int) -> Dict[str, object]:
    run_dirs = {arm: os.path.join(os.path.abspath(raw_root), arm, E6_FORMAL_SETTING,
                                  "seed{}".format(int(seed))) for arm in E6_ARM_KEYS}
    return compare_e6_block(run_dirs, require_formal=True)


def _as_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _augment_run_summary(run_dir: str, block: Mapping[str, object]) -> None:
    """Write the preregistered common/last-common endpoint into one run file."""

    summary_path = os.path.join(run_dir, "metrics_summary.csv")
    metrics_path = os.path.join(run_dir, "metrics_epoch.csv")
    summary = _read_csv(summary_path)
    metrics = _read_csv(metrics_path)
    common_epochs = {_as_int(value) for value in block.get("common_epochs", [])}
    last_epoch = block.get("last_common_epoch")
    last_epoch_int = _as_int(last_epoch, -1)
    if not summary or last_epoch_int < 0:
        return
    by_cell_last = {
        _as_int(row.get("cell")): row for row in metrics
        if _as_int(row.get("epoch"), -1) == last_epoch_int
        and not _as_int(row.get("stopped_censored"), 0)
    }
    fields: List[str] = []
    for row in summary:
        for field in row:
            if field not in fields:
                fields.append(field)
    for field in ("last_common_epoch", "last_common_test_auc", "last_common_test_logloss",
                  "last_common_transfer_auc", "last_common_transfer_logloss"):
        if field not in fields:
            fields.append(field)
    for row in summary:
        selected_epoch = _as_int(row.get("selected_epoch"), -1)
        cell = _as_int(row.get("cell"), -1)
        endpoint = by_cell_last.get(cell, {})
        row["common_support"] = int(selected_epoch in common_epochs)
        row["last_common_epoch"] = last_epoch_int
        row["last_common_test_auc"] = endpoint.get("test_auc", "")
        row["last_common_test_logloss"] = endpoint.get("test_logloss", "")
        row["last_common_transfer_auc"] = endpoint.get("transfer_auc", "")
        row["last_common_transfer_logloss"] = endpoint.get("transfer_logloss", "")
    _write_csv(summary_path, summary, fields)


def collect(raw_root: str = DEFAULT_RAW_ROOT) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    tasks = expected_runs(raw_root)
    matrix_ok, matrix_reasons = validate_matrix_tasks(tasks)
    blocks = {int(seed): _seed_block(raw_root, int(seed)) for seed in E6_EXPECTED_SEEDS}
    rows: List[Dict[str, object]] = []
    run_records: List[Dict[str, object]] = []
    missing_count = 0
    invalid_count = 0
    for task in tasks:
        run_dir = str(task["run_dir"])
        manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
        block = blocks[int(task["seed"])]
        status, reasons = _run_status(run_dir, manifest, block)
        if status == "MISSING":
            missing_count += 1
        elif status == "INVALID":
            invalid_count += 1
        if status == "OK":
            _augment_run_summary(run_dir, block)
        metrics = _read_csv(os.path.join(run_dir, "metrics_summary.csv"))
        run_record = {
            "arm_key": task["arm_key"], "setting": task["setting"], "seed": task["seed"],
            "run_dir": run_dir, "status": status, "reasons": reasons,
            "block_status": block.get("status", "PENDING"),
            "block_valid": block.get("valid"),
            "common_support_count": block.get("common_support_count", 0),
            "last_common_epoch": block.get("last_common_epoch"),
            "required_artifacts_present": len(REQUIRED_ARTIFACTS) - len(reasons)
            if status == "MISSING" else len(REQUIRED_ARTIFACTS),
            "manifest_formal_valid": manifest.get("formal_valid"),
        }
        run_records.append(run_record)
        if not metrics:
            continue
        for metric in metrics:
            row = {field: metric.get(field, "") for field in SUMMARY_FIELDS}
            row.update({"stage": "E6", "status": status, "arm_key": task["arm_key"],
                        "setting": task["setting"], "seed": task["seed"],
                        "run_dir": run_dir})
            if status != "OK":
                # Do not leave a locally selected endpoint looking comparable
                # when the cross-arm block failed or is still pending.
                row["common_support"] = ""
                row["last_common_epoch"] = ""
                row["last_common_test_auc"] = ""
                row["last_common_test_logloss"] = ""
                row["last_common_transfer_auc"] = ""
                row["last_common_transfer_logloss"] = ""
            rows.append(row)
    manifest = {
        "stage": "E6", "raw_root": os.path.abspath(raw_root),
        "expected_run_count": len(tasks), "expected_arm_count": len(E6_ARM_KEYS),
        "expected_seed_count": len(E6_EXPECTED_SEEDS), "expected_setting": E6_FORMAL_SETTING,
        "matrix_valid": matrix_ok, "matrix_reasons": matrix_reasons,
        "available_run_count": sum(1 for item in run_records if item["status"] == "OK"),
        "missing_run_count": missing_count, "invalid_run_count": invalid_count,
        "block_valid_seed_count": sum(1 for block in blocks.values()
                                      if block.get("status") == "VALID" and block.get("valid") is True),
        "block_results": blocks,
        "summary_row_count": len(rows), "runs": run_records,
        "scope_note": "raw seed/cell structural summary only; no arm comparison or scientific conclusion",
    }
    return rows, manifest


def _ablation_label(arm_key: str) -> str:
    if arm_key == "sample_ema_control":
        return "sample_static_vs_sample_ema_control"
    if arm_key == "adaptive_ema_uniform":
        return "adaptive_ema_vs_adaptive_ema_uniform"
    if arm_key == "adaptive_noema":
        return "adaptive_noema_vs_sample_static"
    if arm_key == "adaptive_ema":
        return "adaptive_ema_vs_sample_static"
    if arm_key == "equal_static":
        return "equal_static_reference"
    return "sample_static_reference"


def make_ablation_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    return [{
        "comparison": _ablation_label(str(row.get("arm_key", ""))),
        **{field: row.get(field, "") for field in ABLATION_FIELDS if field != "comparison"},
    } for row in rows]


def write_summary(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, rows, SUMMARY_FIELDS)


def write_ablation(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, make_ablation_rows(rows), ABLATION_FIELDS)


def write_manifest(path: str, manifest: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)


def write_run_report(path: str, manifest: Mapping[str, object], *, summary_path: str,
                     ablation_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = [
        "# Stage E6 raw run report", "",
        "This report is a structural/raw collection artifact. It contains no scientific comparison or conclusion.", "",
        "- Expected matrix: 6 arms × 1 setting (`d0_large`) × 4 seeds = 24 runs.",
        "- Expected seeds: `2022, 2023, 2024, 2025`.",
        "- Available formal runs: {}".format(manifest.get("available_run_count", 0)),
        "- Missing runs: {}".format(manifest.get("missing_run_count", 0)),
        "- Invalid runs: {}".format(manifest.get("invalid_run_count", 0)),
        "- Summary rows: {}".format(manifest.get("summary_row_count", 0)),
        "", "## Artifacts", "",
        "- Summary: `{}`".format(os.path.abspath(summary_path)),
        "- Ablation raw mapping: `{}`".format(os.path.abspath(ablation_path)),
        "- Manifest: `{}`".format(os.path.abspath(DEFAULT_MANIFEST)),
        "", "## Per-run status", "",
        "| arm | setting | seed | status | reasons |",
        "|---|---|---:|---|---|",
    ]
    for item in manifest.get("runs", []):
        lines.append("| {} | {} | {} | {} | {} |".format(
            item.get("arm_key", ""), item.get("setting", ""), item.get("seed", ""),
            item.get("status", ""), "; ".join(item.get("reasons", []))))
    lines.extend(["", "Raw collection complete; main analysis is a separate workflow stage.", ""])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E6 raw summary collector")
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--ablation", default=DEFAULT_ABLATION)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--run_report", default=DEFAULT_RUN_REPORT)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    rows, manifest = collect(args.raw_root)
    write_summary(args.output, rows)
    write_ablation(args.ablation, rows)
    write_manifest(args.manifest, manifest)
    write_run_report(args.run_report, manifest, summary_path=args.output,
                     ablation_path=args.ablation)
    print("[stage_E6_report] expected:", manifest["expected_run_count"],
          "available:", manifest["available_run_count"],
          "missing:", manifest["missing_run_count"],
          "invalid:", manifest["invalid_run_count"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
