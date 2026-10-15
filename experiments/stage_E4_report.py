"""Structural Stage E4 report writer.

The report enumerates every expected artifact, copies raw selected metrics into
a stable summary schema, and records missing/invalid runs.  It intentionally
does not compare weighting conditions or make scientific conclusions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from stage_E4_orchestrator import (  # noqa: E402
    DEFAULT_MASTER,
    DEFAULT_ORACLE_MASTER,
    DEFAULT_RAW_ROOT,
    build_oracle_tasks,
    build_tasks,
    compare_joint_block,
    validate_joint_artifact,
    validate_oracle_artifact,
)
from stage_E4_weighting import (  # noqa: E402
    E4_DATASET_PATH,
    EXPECTED_SEEDS,
    WEIGHTING_KEYS,
    DatasetBundle,
    load_processed_dataset,
    weighting_alpha,
    weighting_display,
    write_json,
    write_rows_csv,
)


SUMMARY_FIELDS = [
    "stage", "artifact_validity", "weighting", "weighting_display", "alpha", "seed",
    "domain", "physical_domain", "task", "cell", "domain_share", "train_sample_count",
    "validation_sample_count", "test_sample_count", "train_positive_rate",
    "validation_positive_rate", "test_positive_rate", "selected_epoch", "stopped_epoch",
    "best_val_auc", "best_val_logloss", "best_test_auc", "best_test_logloss",
    "oracle_auc", "oracle_logloss", "oracle_source", "oracle_coverage", "transfer_auc",
    "transfer_logloss", "negative_transfer", "domain_macro_auc", "task_macro_auc",
    "overall_auc", "overall_logloss", "worst_domain_auc", "worst_cell_auc", "pooled_auc",
    "pooled_logloss", "shared_raw_gradient_norm", "shared_weighted_gradient_norm",
    "shared_magnitude_share", "shared_signed_contribution", "shared_mismatch_index",
    "shared_update_alignment", "selected_checkpoint", "selected_epoch_in_common_support",
    "common_support", "common_epoch_count", "common_trace_count", "block_status",
    "block_valid", "block_validity", "censored",
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


def _as_float(value: object, default: object = "") -> object:
    try:
        parsed = float(value)
        return parsed if np.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def expected_paths(raw_root: str = DEFAULT_RAW_ROOT,
                   oracle_root: Optional[str] = None) -> Dict[str, List[Dict[str, object]]]:
    oracle_root = oracle_root or os.path.join(raw_root, "oracle_reference")
    return {"joint": build_tasks(raw_root), "oracle": build_oracle_tasks(oracle_root)}


def _joint_artifact_status(valid: bool, manifest: Mapping[str, object],
                           block: Mapping[str, object], reasons: Sequence[str]) -> str:
    """Combine per-arm validation with the same-seed three-arm block state."""

    block_status = str(block.get("status", "PENDING"))
    if block_status == "INVALID":
        return "INVALID_BLOCK"
    if not valid:
        return "MISSING" if "run_directory_missing" in reasons else "INVALID"
    if manifest.get("formal_valid") is True and block_status == "VALID":
        return "FORMAL_VALID"
    if manifest.get("formal_valid") is True and block_status == "PENDING":
        return "BLOCK_PENDING"
    return "SMOKE_ONLY"


def collect(
    raw_root: str = DEFAULT_RAW_ROOT,
    oracle_root: Optional[str] = None,
    dataset_path: str = E4_DATASET_PATH,
    *,
    formal: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Collect fixed-path rows and structural validity metadata."""

    oracle_root = oracle_root or os.path.join(raw_root, "oracle_reference")
    bundle = load_processed_dataset(dataset_path, validate=True)
    paths = expected_paths(raw_root, oracle_root)
    oracle_entries: List[Dict[str, object]] = []
    for task in paths["oracle"]:
        valid, reasons = validate_oracle_artifact(task, bundle, formal=formal)
        artifact_manifest = _read_json(os.path.join(str(task["run_dir"]), "run_manifest.json"))
        status = ("FORMAL_VALID" if valid and artifact_manifest.get("formal_valid") is True
                  else "SMOKE_ONLY" if valid else
                  ("MISSING" if "run_directory_missing" in reasons else "INVALID"))
        oracle_entries.append({
            "stage": "E4_ORACLE", "original_physical_domain": task["original_physical_domain"],
            "seed": task["seed"], "path": task["run_dir"],
            "status": status,
            "reasons": reasons,
        })

    summary_rows: List[Dict[str, object]] = []
    joint_entries: List[Dict[str, object]] = []
    blocks_by_seed: Dict[int, Dict[str, object]] = {}
    for seed in EXPECTED_SEEDS:
        seed_tasks = [item for item in paths["joint"] if int(item["seed"]) == int(seed)]
        blocks_by_seed[int(seed)] = compare_joint_block(
            {str(item["weighting"]): str(item["run_dir"]) for item in seed_tasks},
            bundle, require_formal=formal)
    for task in paths["joint"]:
        valid, reasons = validate_joint_artifact(task, bundle, formal=formal)
        run_dir = str(task["run_dir"])
        raw_rows = _read_csv(os.path.join(run_dir, "metrics_summary.csv"))
        artifact_manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
        block = blocks_by_seed[int(task["seed"])]
        status = _joint_artifact_status(valid, artifact_manifest, block, reasons)
        block_status = str(block.get("status", "PENDING"))
        block_valid = block.get("valid")
        common_epochs = [int(value) for value in block.get("common_epochs", [])]
        common_epoch_count = block.get("common_epoch_count", len(common_epochs))
        common_trace_count = block.get("common_trace_count", 0)
        selected_support = block.get("selected_epoch_in_common_support", {})
        if not isinstance(selected_support, Mapping):
            selected_support = {}
        selected_in_support = bool(selected_support.get(str(task["weighting"]), False))
        joint_entries.append({
            "stage": "E4", "weighting": task["weighting"], "seed": task["seed"],
            "path": run_dir, "status": status, "reasons": reasons,
            "block_status": block_status, "block_valid": block_valid,
            "block_validity": block_status, "common_epochs": common_epochs,
            "common_epoch_count": common_epoch_count,
            "common_trace_count": common_trace_count,
            "selected_epoch_in_common_support": selected_in_support,
        })
        if not raw_rows:
            # Keep one explicit missing row per expected cell, rather than
            # hiding a missing run by emitting no summary rows.
            raw_rows = [{"domain": domain, "task": task_id,
                         "cell": task_id * 3 + domain}
                        for task_id in range(2) for domain in range(3)]
        for raw in raw_rows:
            domain = raw.get("domain", raw.get("physical_domain", ""))
            task_id = raw.get("task", "")
            cell = raw.get("cell", "")
            summary_rows.append({
                "stage": "E4", "artifact_validity": status,
                "weighting": task["weighting"],
                "weighting_display": weighting_display(task["weighting"]),
                "alpha": weighting_alpha(task["weighting"]), "seed": task["seed"],
                "domain": domain, "physical_domain": raw.get("physical_domain", domain),
                "task": task_id, "cell": cell,
                "selected_checkpoint": raw.get(
                    "selected_checkpoint", int(raw.get("selected_epoch", "") not in ("", None))),
                "selected_epoch_in_common_support": (
                    "" if block_status == "PENDING" else int(selected_in_support)),
                "common_support": (common_epoch_count if block_status != "PENDING"
                                    else ""),
                "common_epoch_count": (common_epoch_count if block_status != "PENDING"
                                        else raw.get("common_epoch_count", "")),
                "common_trace_count": (common_trace_count if block_status != "PENDING"
                                        else raw.get("common_trace_count", "")),
                "block_status": block_status,
                "block_valid": "" if block_valid is None else int(bool(block_valid)),
                "block_validity": block_status,
                **{field: raw.get(field, "") for field in SUMMARY_FIELDS
                   if field not in {"stage", "artifact_validity", "weighting", "weighting_display",
                                    "alpha", "seed", "domain", "physical_domain", "task", "cell",
                                    "selected_checkpoint", "selected_epoch_in_common_support",
                                    "common_support", "common_epoch_count", "common_trace_count",
                                    "block_status", "block_valid", "block_validity"}},
            })
    manifest = {
        "stage": "E4", "dataset_identity": bundle.dataset_identity,
        "expected_joint_count": len(paths["joint"]), "expected_oracle_count": len(paths["oracle"]),
        "joint": joint_entries, "oracle": oracle_entries,
        "joint_formal_valid_count": sum(item["status"] == "FORMAL_VALID" for item in joint_entries),
        "oracle_formal_valid_count": sum(item["status"] == "FORMAL_VALID" for item in oracle_entries),
        "joint_smoke_count": sum(item["status"] == "SMOKE_ONLY" for item in joint_entries),
        "oracle_smoke_count": sum(item["status"] == "SMOKE_ONLY" for item in oracle_entries),
        "scope_note": "Structural/raw summary only; no cross-run scientific analysis.",
    }
    return summary_rows, manifest


def write_summary(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    write_rows_csv(path, rows, SUMMARY_FIELDS)


def write_run_report(path: str, manifest: Mapping[str, object]) -> None:
    joint = list(manifest.get("joint", []))
    oracle = list(manifest.get("oracle", []))
    lines = [
        "# Stage E4 Run Report", "",
        "This report records artifact coverage and schema validity only.", "",
        f"- Expected joint runs: {manifest.get('expected_joint_count', 12)}",
        f"- Formal-valid joint runs: {manifest.get('joint_formal_valid_count', 0)}",
        f"- Smoke-only joint runs: {manifest.get('joint_smoke_count', 0)}",
        f"- Expected matched oracle runs: {manifest.get('expected_oracle_count', 12)}",
        f"- Formal-valid matched oracle runs: {manifest.get('oracle_formal_valid_count', 0)}",
        f"- Smoke-only matched oracle runs: {manifest.get('oracle_smoke_count', 0)}", "",
        "## Joint artifact status", "",
        "| weighting | seed | status | block | common epochs/traces | path/reasons |",
        "|---|---:|---|---|---:|---|",
    ]
    for item in joint:
        lines.append(f"| {item.get('weighting','')} | {item.get('seed','')} | "
                     f"{item.get('status','')} | {item.get('block_validity','')} "
                     f"({item.get('block_valid')}) | "
                     f"{item.get('common_epoch_count',0)}/{item.get('common_trace_count',0)} | "
                     f"{item.get('path','')} "
                     f"{'; '.join(item.get('reasons', []))} |")
    lines.extend(["", "## Oracle artifact status", "",
                  "| domain | seed | status | path/reasons |", "|---:|---:|---|---|"])
    for item in oracle:
        lines.append(f"| {item.get('original_physical_domain','')} | {item.get('seed','')} | "
                     f"{item.get('status','')} | {item.get('path','')} "
                     f"{'; '.join(item.get('reasons', []))} |")
    lines.extend(["", manifest.get("scope_note", "")])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E4 structural report")
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--oracle_root", default=None)
    parser.add_argument("--dataset_path", default=E4_DATASET_PATH)
    parser.add_argument("--summary", default=os.path.join(os.path.dirname(DEFAULT_RAW_ROOT), "stage_E4_summary.csv"))
    parser.add_argument("--manifest", default=os.path.join(os.path.dirname(DEFAULT_RAW_ROOT), "stage_E4_summary_manifest.json"))
    parser.add_argument("--run_report", default=os.path.join(HERE, "reports", "stage_E4_run_report.md"))
    parser.add_argument("--formal", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    rows, manifest = collect(args.raw_root, args.oracle_root, args.dataset_path, formal=args.formal)
    write_summary(args.summary, rows)
    write_json(args.manifest, manifest)
    write_run_report(args.run_report, manifest)
    print("[stage_E4_report] expected joint/oracle:",
          manifest["expected_joint_count"], manifest["expected_oracle_count"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
