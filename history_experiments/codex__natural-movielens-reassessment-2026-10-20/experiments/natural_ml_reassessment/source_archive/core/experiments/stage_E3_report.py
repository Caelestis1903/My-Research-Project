"""Structured Stage E3 collector.

The report layer enumerates the complete 60-arm joint matrix and preserves
seed-level rows.  It computes only mechanical per-run fields (macro means,
minimums, oracle subtraction and contribution joins); it does not choose an
alpha, test hypotheses or write a scientific conclusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from stage_E3_weighting import (  # noqa: E402
    ALPHA_GRID,
    ALPHA_KEYS,
    EXPECTED_SEEDS,
    EXPECTED_SETTINGS,
    ROLE_MAPS,
    iter_cells,
)


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW_ROOT = os.path.join(HERE, "stage_E3_raw_results")
DEFAULT_OUTPUT = os.path.join(HERE, "stage_E3_summary.csv")
DEFAULT_MANIFEST = os.path.join(HERE, "stage_E3_summary_manifest.json")
DEFAULT_RUN_REPORT = os.path.join(HERE, "reports", "stage_E3_run_report.md")

REPORT_REQUIRED_FILES = (
    "config.json", "stdout.log", "run_manifest.json", "metrics_epoch.csv",
    "metrics_summary.csv", "shared_gradient_step.csv", "shared_gradient_epoch.csv",
    "batch_trace.csv", "nonfinite_inventory.json", "model.pth",
)

SUMMARY_FIELDS = [
    "stage", "alpha", "alpha_key", "setting", "seed", "physical_domain", "size_role",
    "task", "domain", "cell", "domain_size", "domain_share", "selected_epoch", "stopped_epoch", "test_auc",
    "test_logloss", "domain_auc", "overall_auc", "overall_logloss", "worst_domain_auc",
    "worst_cell_auc", "oracle_auc", "oracle_logloss", "oracle_source", "oracle_coverage",
    "transfer_auc", "transfer_logloss", "negative_transfer", "shared_raw_gradient_norm",
    "shared_weighted_gradient_norm", "shared_magnitude_share", "shared_signed_contribution",
    "shared_mismatch_index", "shared_update_alignment", "common_support", "censored",
    "run_status", "artifact_complete",
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


def _as_float(value: object) -> Optional[float]:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _as_int(value: object, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def expected_runs(raw_root: str = DEFAULT_RAW_ROOT) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for key, alpha in zip(ALPHA_KEYS, ALPHA_GRID):
        for setting in EXPECTED_SETTINGS:
            for seed in EXPECTED_SEEDS:
                run_dir = os.path.join(raw_root, key, setting, f"seed{seed}")
                result.append({"alpha": alpha, "alpha_key": key, "setting": setting,
                               "seed": int(seed), "run_dir": run_dir})
    return result


def _cell_rows(metrics: Sequence[Mapping[str, str]], selected_only: bool = False) -> Dict[int, Dict[str, str]]:
    rows: Dict[int, Dict[str, str]] = {}
    for row in metrics:
        if selected_only and _as_int(row.get("selected_checkpoint"), 0) != 1:
            continue
        cell = _as_int(row.get("cell"))
        if cell >= 0:
            rows[cell] = dict(row)
    return rows


def _aggregate(selected: Mapping[int, Mapping[str, str]], domain_num: int = 3,
               task_num: int = 2) -> Dict[str, object]:
    expected = {idx for idx, _, _ in iter_cells(domain_num, task_num)}
    auc = {idx: _as_float(row.get("test_auc", row.get("best_test_auc")))
           for idx, row in selected.items()}
    ll = {idx: _as_float(row.get("test_logloss", row.get("best_test_logloss")))
          for idx, row in selected.items()}
    # A checkpoint macro is meaningful only when all six selected cells are
    # present and finite.  Do not silently average a smaller intersection.
    aggregate_complete = (
        set(selected) == expected
        and all(auc.get(idx) is not None and ll.get(idx) is not None for idx in expected)
    )
    if not aggregate_complete:
        return {
            "domain_auc": {}, "overall_auc": float("nan"),
            "overall_logloss": float("nan"), "worst_domain_auc": float("nan"),
            "worst_cell_auc": float("nan"), "aggregate_complete": False,
        }
    finite_auc = {idx: auc[idx] for idx in expected}
    finite_ll = {idx: ll[idx] for idx in expected}
    domain_auc: Dict[int, float] = {}
    for domain in range(domain_num):
        values = [finite_auc[idx] for idx, d, _ in iter_cells(domain_num, task_num)
                  if d == domain and idx in finite_auc]
        if values:
            domain_auc[domain] = float(np.mean(values))
    return {
        "domain_auc": domain_auc,
        "overall_auc": float(np.mean(list(finite_auc.values()))) if finite_auc else float("nan"),
        "overall_logloss": float(np.mean(list(finite_ll.values()))) if finite_ll else float("nan"),
        "worst_domain_auc": min(domain_auc.values()) if domain_auc else float("nan"),
        "worst_cell_auc": min(finite_auc.values()) if finite_auc else float("nan"),
        "aggregate_complete": True,
    }


def _gradient_for_cell(rows: Sequence[Mapping[str, str]], cell: int, epoch: int) -> Dict[str, str]:
    candidates = [row for row in rows if _as_int(row.get("cell")) == cell and
                  (epoch < 0 or _as_int(row.get("epoch")) == epoch)]
    if not candidates and epoch >= 0:
        candidates = [row for row in rows if _as_int(row.get("cell")) == cell]
    if not candidates:
        return {}
    # Epoch rows already contain probe means.  Step rows are averaged only
    # when no epoch row exists or when the selected epoch is not available.
    if "contribution_mean" in candidates[0]:
        return dict(candidates[-1])
    result = dict(candidates[-1])
    for field in ("raw_gradient_norm", "weighted_gradient_norm", "magnitude_share",
                  "normalized_contribution", "signed_contribution", "mismatch_index",
                  "actual_update_alignment"):
        values = [_as_float(row.get(field)) for row in candidates]
        finite = [value for value in values if value is not None]
        result[field] = float(np.mean(finite)) if finite else ""
    return result


def collect(raw_root: str = DEFAULT_RAW_ROOT) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    manifest: Dict[str, object] = {
        "stage": "E3", "raw_root": os.path.abspath(raw_root),
        "expected_run_count": len(ALPHA_KEYS) * len(EXPECTED_SETTINGS) * len(EXPECTED_SEEDS),
        "available_run_count": 0, "missing_run_count": 0, "missing_files": [],
        "runs": [], "summary_row_count": 0, "oracle_expected_cells": 0,
        "oracle_covered_cells": 0, "scope_note": "structured collection only; no scientific conclusion",
    }
    for expected in expected_runs(raw_root):
        run_dir = str(expected["run_dir"])
        if not os.path.isdir(run_dir):
            missing = list(REPORT_REQUIRED_FILES)
            manifest["missing_run_count"] += 1
            manifest["missing_files"].append({"run_dir": run_dir, "files": missing,
                                               "reason": "run_directory_missing"})
            manifest["runs"].append({**expected, "status": "missing", "missing_files": missing,
                                     "metrics_rows": 0, "gradient_rows": 0})
            continue
        missing = [name for name in REPORT_REQUIRED_FILES
                   if not os.path.exists(os.path.join(run_dir, name))]
        if missing:
            manifest["missing_files"].append({"run_dir": run_dir, "files": missing})
        manifest["available_run_count"] += 1
        run_manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
        metrics = _read_csv(os.path.join(run_dir, "metrics_epoch.csv"))
        summary = _read_csv(os.path.join(run_dir, "metrics_summary.csv"))
        gradient_epoch = _read_csv(os.path.join(run_dir, "shared_gradient_epoch.csv"))
        gradient_step = _read_csv(os.path.join(run_dir, "shared_gradient_step.csv"))
        selected = _cell_rows(metrics, selected_only=True)
        if not selected:
            selected = { _as_int(row.get("cell")): row for row in summary
                         if _as_int(row.get("cell")) >= 0 }
        aggregate = _aggregate(selected)
        selected_epoch = _as_int(run_manifest.get("best_epoch"), -1)
        if selected_epoch < 0 and selected:
            selected_epoch = _as_int(next(iter(selected.values())).get("selected_epoch"), -1)
        oracle_covered = 0
        for idx, domain, task in iter_cells(3, 2):
            selected_row = selected.get(idx, {})
            if _as_int(selected_row.get("oracle_coverage"), 0) == 1:
                oracle_covered += 1
            grad = _gradient_for_cell(gradient_epoch, idx, selected_epoch)
            if not grad:
                grad = _gradient_for_cell(gradient_step, idx, selected_epoch)
            transfer = selected_row.get("transfer_auc", "")
            negative = (1 if _as_float(transfer) is not None and float(transfer) < 0 else
                        0 if _as_float(transfer) is not None else "")
            row = {
                "stage": "E3", "alpha": expected["alpha"], "alpha_key": expected["alpha_key"],
                "setting": expected["setting"], "seed": expected["seed"],
                "physical_domain": domain, "size_role": ROLE_MAPS[str(expected["setting"])][domain],
                "task": task, "domain": domain, "cell": idx,
                "domain_size": run_manifest.get("train_domain_counts", {}).get(str(domain),
                              run_manifest.get("train_domain_counts", {}).get(domain, "")),
                "domain_share": run_manifest.get("domain_shares", [""] * 3)[domain]
                    if isinstance(run_manifest.get("domain_shares"), list)
                    and len(run_manifest.get("domain_shares")) > domain else "",
                "selected_epoch": selected_row.get("selected_epoch", selected_epoch),
                "stopped_epoch": run_manifest.get("stopped_epoch", ""),
                "test_auc": selected_row.get("best_test_auc", selected_row.get("test_auc", "")),
                "test_logloss": selected_row.get("best_test_logloss", selected_row.get("test_logloss", "")),
                "domain_auc": aggregate["domain_auc"].get(domain, float("nan")),
                "overall_auc": aggregate["overall_auc"], "overall_logloss": aggregate["overall_logloss"],
                "worst_domain_auc": aggregate["worst_domain_auc"], "worst_cell_auc": aggregate["worst_cell_auc"],
                "oracle_auc": selected_row.get("oracle_auc", ""),
                "oracle_logloss": selected_row.get("oracle_logloss", ""),
                "oracle_source": selected_row.get("oracle_source", "missing"),
                "oracle_coverage": selected_row.get("oracle_coverage", 0),
                "transfer_auc": transfer, "transfer_logloss": selected_row.get("transfer_logloss", ""),
                "negative_transfer": negative,
                "shared_raw_gradient_norm": grad.get("raw_gradient_norm", ""),
                "shared_weighted_gradient_norm": grad.get("weighted_gradient_norm", ""),
                "shared_magnitude_share": grad.get("magnitude_share", ""),
                "shared_signed_contribution": grad.get("contribution_mean", grad.get("signed_contribution", "")),
                "shared_mismatch_index": grad.get("mismatch_index", ""),
                "shared_update_alignment": grad.get("actual_update_alignment", ""),
                "common_support": int(
                    run_manifest.get("block_valid") is True
                    and _as_int(selected_epoch, -1) in {
                        _as_int(value, -2) for value in
                        run_manifest.get("block_comparison", {}).get("common_epochs", [])
                    }),
                "censored": int(run_manifest.get("censored_after_epoch") not in (None, "")),
                "run_status": run_manifest.get("status", "missing"),
                "artifact_complete": int(not missing),
            }
            rows.append(row)
        manifest["oracle_expected_cells"] += 6
        manifest["oracle_covered_cells"] += oracle_covered
        manifest["runs"].append({
            **expected, "status": run_manifest.get("status", "unknown"),
            "formal_valid": run_manifest.get("formal_valid", False),
            "block_status": run_manifest.get("block_status", "PENDING"),
            "block_valid": run_manifest.get("block_valid"),
            "missing_files": missing, "metrics_rows": len(metrics),
            "summary_rows": len(summary), "gradient_epoch_rows": len(gradient_epoch),
            "gradient_step_rows": len(gradient_step),
        })
    manifest["summary_row_count"] = len(rows)
    manifest["oracle_coverage_fraction"] = (
        float(manifest["oracle_covered_cells"]) / float(manifest["oracle_expected_cells"])
        if manifest["oracle_expected_cells"] else 0.0)
    return rows, manifest


def write_summary(path: str, rows: Sequence[Mapping[str, object]],
                  manifest_path: Optional[str] = None,
                  manifest: Optional[Mapping[str, object]] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    if manifest_path and manifest is not None:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)


def write_run_report(path: str, manifest: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    runs = list(manifest.get("runs", []))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Stage E3 Run Report\n\n")
        handle.write("This is a structured artifact/status report; it contains no scientific conclusion.\n\n")
        handle.write("- Expected joint runs: {}\n".format(manifest.get("expected_run_count", 60)))
        handle.write("- Available run directories: {}\n".format(manifest.get("available_run_count", 0)))
        handle.write("- Missing run directories: {}\n".format(manifest.get("missing_run_count", 0)))
        handle.write("- Oracle coverage cells: {}/{}\n\n".format(
            manifest.get("oracle_covered_cells", 0), manifest.get("oracle_expected_cells", 0)))
        handle.write("## Matrix status\n\n")
        handle.write("| alpha_key | setting | seed | status | block | missing files |\n")
        handle.write("|---|---|---:|---|---|---:|\n")
        for run in runs:
            handle.write("| {alpha_key} | {setting} | {seed} | {status} | {block_status} | {missing} |\n".format(
                alpha_key=run.get("alpha_key", ""), setting=run.get("setting", ""),
                seed=run.get("seed", ""), status=run.get("status", ""),
                block_status=run.get("block_status", ""), missing=len(run.get("missing_files", []))))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E3 structured report")
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--run_report", default=DEFAULT_RUN_REPORT)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    rows, manifest = collect(args.raw_root)
    write_summary(args.output, rows, args.manifest, manifest)
    write_run_report(args.run_report, manifest)
    print("[stage_E3_report] rows:", len(rows), "runs:", manifest["expected_run_count"], flush=True)
    print("[stage_E3_report] output:", os.path.abspath(args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
