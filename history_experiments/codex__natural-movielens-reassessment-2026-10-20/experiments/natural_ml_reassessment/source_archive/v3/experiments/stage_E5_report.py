"""Raw-result collector for Stage E5.

This report intentionally performs schema/completeness joins only.  It does
not compare arms, label effects, or write a GO/STOP conclusion; those actions
belong to the main-analysis stage after the runner has finished.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_E1_instrumentation import write_json  # noqa: E402
from stage_E5_orchestrator import (  # noqa: E402
    DEFAULT_MASTER,
    DEFAULT_RAW_ROOT,
    MASTER_FIELDS,
    REQUIRED_ARTIFACTS,
    _read_csv,
    _read_json,
    validate_run_artifacts,
)
from stage_E5_schedule import (  # noqa: E402
    E5_ARM_KEYS,
    E5_EPOCHS,
    E5_FORMAL_SETTING,
    EXPECTED_SEEDS,
    compare_e5_block,
    expected_matrix_tasks,
)


DEFAULT_SUMMARY = os.path.join(HERE, "stage_E5_summary.csv")
DEFAULT_MANIFEST = os.path.join(HERE, "stage_E5_summary_manifest.json")
DEFAULT_RUN_REPORT = os.path.join(HERE, "reports", "stage_E5_run_report.md")


SUMMARY_FIELDS = [
    "stage", "arm_key", "seed", "setting", "available", "view", "epoch", "domain",
    "physical_domain", "task", "cell", "size_role", "domain_size", "domain_share",
    "train_sample_count", "validation_sample_count", "test_sample_count", "train_auc",
    "train_logloss", "validation_auc", "validation_logloss", "test_auc", "test_logloss",
    "oracle_auc", "oracle_logloss", "oracle_source", "oracle_coverage", "transfer_auc",
    "transfer_logloss", "negative_transfer", "overall_auc", "overall_logloss", "domain_macro_auc",
    "task_macro_auc", "worst_domain_auc", "worst_cell_auc", "peak_epoch", "fixed_final_epoch",
    "selected_epoch", "selected_checkpoint", "common_support", "censored", "alpha_at_epoch",
    "alpha_exposure_total", "coefficient_integral_cell", "step_weighted_coefficient_integral_cell",
    "shared_raw_gradient_norm", "shared_weighted_gradient_norm", "shared_gradient_variance",
    "shared_gradient_snr", "shared_magnitude_share", "shared_signed_contribution",
    "shared_mismatch_index", "shared_update_alignment", "run_dir",
]


def _task_key(task: Mapping[str, object]) -> Tuple[str, int]:
    return str(task["arm_key"]), int(task["seed"])


def _block_for_seed(raw_root: str, seed: int) -> Dict[str, object]:
    dirs = {arm: os.path.join(raw_root, arm, E5_FORMAL_SETTING, "seed{}".format(int(seed)))
            for arm in E5_ARM_KEYS}
    return compare_e5_block(dirs, require_formal=True)


def collect(
    raw_root: str = DEFAULT_RAW_ROOT,
    *,
    dataset_identity: Optional[str] = None,
    device_identity: Optional[str] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Collect one row for each available summary cell/view and audit all 20 paths."""

    tasks = expected_matrix_tasks(raw_root)
    rows: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    invalid: List[Dict[str, object]] = []
    run_records: List[Dict[str, object]] = []
    for task in tasks:
        valid, reasons = validate_run_artifacts(
            task, formal=True, dataset_identity=dataset_identity,
            device_identity=device_identity)
        run_dir = str(task["run_dir"])
        summary_rows = _read_csv(os.path.join(run_dir, "metrics_summary.csv"))
        available = bool(os.path.isdir(run_dir) and valid and summary_rows)
        if not os.path.isdir(run_dir):
            missing.append({"arm_key": task["arm_key"], "seed": task["seed"],
                            "run_dir": run_dir, "reason": "run_directory_missing"})
        elif not valid:
            invalid.append({"arm_key": task["arm_key"], "seed": task["seed"],
                            "run_dir": run_dir, "reasons": reasons})
        manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
        run_records.append({
            "arm_key": task["arm_key"], "seed": int(task["seed"]), "setting": E5_FORMAL_SETTING,
            "run_dir": run_dir, "available": int(available), "valid": bool(valid),
            "formal_valid": manifest.get("formal_valid", False),
            "oracle_coverage": manifest.get("oracle_coverage", ""),
            "block_status": manifest.get("block_status", "PENDING"),
            "block_valid": manifest.get("block_valid"), "reasons": reasons,
        })
        if not available:
            continue
        for source in summary_rows:
            row = {field: source.get(field, "") for field in SUMMARY_FIELDS}
            row.update({"stage": "E5", "arm_key": task["arm_key"], "seed": int(task["seed"]),
                        "setting": E5_FORMAL_SETTING, "available": 1, "run_dir": run_dir})
            rows.append(row)
    blocks = {str(seed): _block_for_seed(raw_root, int(seed)) for seed in EXPECTED_SEEDS}
    block_valid = all(value.get("status") == "VALID" for value in blocks.values())
    manifest = {
        "stage": "E5", "setting": E5_FORMAL_SETTING,
        "expected_matrix": {"arms": list(E5_ARM_KEYS), "seeds": list(EXPECTED_SEEDS),
                             "run_count": len(tasks)},
        "available_run_count": int(sum(int(item["available"]) for item in run_records)),
        "valid_run_count": int(sum(int(item["valid"]) for item in run_records)),
        "complete": bool(len(missing) == 0 and len(invalid) == 0 and len(run_records) == 20),
        "all_blocks_valid": bool(block_valid), "oracle_coverage_complete": bool(
            all(item.get("oracle_coverage") in (1, 1.0, "1", "1.0") for item in run_records)),
        "missing_runs": missing, "invalid_runs": invalid,
        "runs": run_records, "blocks": blocks,
        "summary_row_count": len(rows),
        "scope_note": "structured raw-result inventory; no cross-arm scientific analysis",
    }
    return rows, manifest


def _write_csv(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def write_run_report(path: str, manifest: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    runs = list(manifest.get("runs", []))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Stage E5 Run Report\n\n")
        handle.write("This is a raw artifact and completeness report; it contains no scientific conclusion.\n\n")
        handle.write("- Expected runs: 20 (5 arms × 4 seeds)\n")
        handle.write("- Available valid runs: {}\n".format(manifest.get("valid_run_count", 0)))
        handle.write("- Complete matrix: {}\n".format(manifest.get("complete", False)))
        handle.write("- All seed blocks valid: {}\n".format(manifest.get("all_blocks_valid", False)))
        handle.write("- Oracle coverage complete: {}\n\n".format(manifest.get("oracle_coverage_complete", False)))
        handle.write("## Run inventory\n\n")
        handle.write("| arm | seed | available | valid | formal_valid | block_status |\n")
        handle.write("|---|---:|---:|---:|---:|---|\n")
        for row in runs:
            handle.write("| {} | {} | {} | {} | {} | {} |\n".format(
                row.get("arm_key", ""), row.get("seed", ""), row.get("available", 0),
                row.get("valid", False), row.get("formal_valid", False), row.get("block_status", "PENDING")))
        if manifest.get("missing_runs"):
            handle.write("\n## Missing runs\n\n")
            for item in manifest["missing_runs"]:
                handle.write("- `{}` seed {}: {}\n".format(item.get("arm_key"), item.get("seed"), item.get("run_dir")))
        if manifest.get("invalid_runs"):
            handle.write("\n## Invalid runs\n\n")
            for item in manifest["invalid_runs"]:
                handle.write("- `{}` seed {}: {}\n".format(item.get("arm_key"), item.get("seed"), "; ".join(item.get("reasons", []))))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E5 raw result collector")
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--run_report", default=DEFAULT_RUN_REPORT)
    parser.add_argument("--dataset_identity", default=None)
    parser.add_argument("--device_identity", default=None)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    rows, manifest = collect(
        args.raw_root, dataset_identity=args.dataset_identity,
        device_identity=args.device_identity)
    _write_csv(args.summary, rows)
    write_json(args.manifest, manifest)
    write_run_report(args.run_report, manifest)
    print("[stage_E5] summary rows:", len(rows), flush=True)
    print("[stage_E5] valid runs:", manifest["valid_run_count"], "/ 20", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
