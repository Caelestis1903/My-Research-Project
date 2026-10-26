"""Schema-preserving Stage E1 raw-output collector.

This report layer deliberately performs no scientific interpretation.  It
joins per-epoch prediction rows with the corresponding per-epoch gradient
rows and writes a long-form CSV retaining seed, role, task, parameter group,
and raw diagnostics.  Any missing run or field is recorded in a manifest;
values are never imputed or converted into conclusions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
DEFAULT_RAW_ROOT = os.path.join(HERE, "stage_E1_raw_results")
DEFAULT_OUTPUT = os.path.join(HERE, "stage_E1_summary.csv")

SUMMARY_FIELDS = [
    "seed", "epoch", "split", "domain", "physical_domain", "task", "cell",
    "weighting", "setting",
    "size_role", "domain_size", "domain_share", "parameter_group", "AUC",
    "LogLoss", "train_auc", "train_logloss", "validation_auc",
    "validation_logloss", "test_auc", "test_logloss", "gradient_norm",
    "weighted_gradient_norm", "gradient_variance", "gradient_SNR",
    "norm_of_mean_gradient", "probe_count", "contribution_mean",
    "contribution_std", "mismatch_index", "variance_scope", "oracle_auc",
    "oracle_logloss", "oracle_source", "transfer_auc", "transfer_logloss",
    "routing_scope", "routing_sample_count", "gate_entropy",
    "effective_expert_count", "top1_expert_frequency",
    "expert_0_mean_probability", "expert_1_mean_probability",
    "expert_2_mean_probability", "expert_3_mean_probability",
    "domain_task_average_gate_entropy",
    "domain_task_average_effective_expert_count",
    "domain_task_average_top1_expert_frequency",
]

EXPECTED_WEIGHTINGS = ("equal", "sample")
EXPECTED_SETTINGS = ("d0_large", "d1_large", "d2_large")
EXPECTED_SEEDS = (2022, 2023, 2024, 2025)
REPORT_REQUIRED_FILES = (
    "config.json", "stdout.log", "run_manifest.json", "metrics_epoch.csv",
    "gradient_step.csv", "gradient_epoch.csv", "gradient_cosine_epoch.csv",
    "gate_epoch.csv", "batch_trace.csv", "metrics_summary.csv",
    "nonfinite_inventory.json",
)


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _as_key(row: Mapping[str, str]) -> Tuple[str, str, str, str, str]:
    return (str(row.get("epoch", "")), str(row.get("domain", "")),
            str(row.get("task", "")), str(row.get("cell", "")),
            str(row.get("parameter_group", "")))


def _run_dirs(raw_root: str) -> List[str]:
    return [item["run_dir"] for item in _expected_runs(raw_root)
            if os.path.isdir(item["run_dir"])]


def _expected_runs(raw_root: str) -> List[Dict[str, object]]:
    """Return the fixed 24-run matrix, including paths that do not exist."""

    expected: List[Dict[str, object]] = []
    for weighting in EXPECTED_WEIGHTINGS:
        for setting in EXPECTED_SETTINGS:
            for seed in EXPECTED_SEEDS:
                expected.append({
                    "weighting": weighting, "setting": setting, "seed": seed,
                    "run_dir": os.path.join(raw_root, weighting, setting, f"seed{seed}"),
                })
    return expected


def collect(raw_root: str = DEFAULT_RAW_ROOT) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Collect all available run rows without aggregation or imputation."""

    output: List[Dict[str, object]] = []
    manifest: Dict[str, object] = {
        "stage": "E1", "raw_root": os.path.abspath(raw_root),
        "expected_run_count": 24, "run_count": 24, "available_run_count": 0,
        "missing_run_count": 0,
        "runs": [], "missing_files": [], "metrics_row_count": 0,
        "gradient_row_count": 0, "summary_row_count": 0,
        "scope_note": "structured collection only; no scientific conclusion",
    }
    for expected in _expected_runs(raw_root):
        run_dir = str(expected["run_dir"])
        if not os.path.isdir(run_dir):
            missing = list(REPORT_REQUIRED_FILES)
            manifest["missing_run_count"] += 1
            manifest["missing_files"].append({"run_dir": run_dir, "files": missing,
                                               "reason": "run_directory_missing"})
            manifest["runs"].append({
                "run_dir": run_dir, "seed": expected["seed"],
                "setting": expected["setting"], "weighting": expected["weighting"],
                "status": "missing", "metrics_rows": 0, "gradient_rows": 0,
                "pair_status": "missing", "pair_valid": None,
                "missing_files": missing,
            })
            continue
        run_manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
        nonfinite_inventory = _read_json(os.path.join(run_dir, "nonfinite_inventory.json"))
        metrics = _read_csv(os.path.join(run_dir, "metrics_epoch.csv"))
        gradient = _read_csv(os.path.join(run_dir, "gradient_epoch.csv"))
        gate = _read_csv(os.path.join(run_dir, "gate_epoch.csv"))
        required = list(REPORT_REQUIRED_FILES)
        missing = [name for name in required if not os.path.exists(os.path.join(run_dir, name))]
        if missing:
            manifest["missing_files"].append({"run_dir": run_dir, "files": missing})
        manifest["available_run_count"] += 1
        manifest["metrics_row_count"] += len(metrics)
        manifest["gradient_row_count"] += len(gradient)
        gradient_by_key = {_as_key(row): row for row in gradient}
        gate_by_key = {}
        gate_average_by_domain = {}
        for gate_row in gate:
            scope = gate_row.get("routing_scope", "cell")
            key = (str(gate_row.get("epoch", "")), str(gate_row.get("domain", "")),
                   str(gate_row.get("task", "")), str(gate_row.get("cell", "")), scope)
            gate_by_key[key] = gate_row
            if scope == "domain_task_average":
                gate_average_by_domain[(key[0], key[1])] = gate_row
        for metric in metrics:
            base = {
                "seed": metric.get("seed", run_manifest.get("seed", "")),
                "epoch": metric.get("epoch", ""), "domain": metric.get("domain", ""),
                "physical_domain": metric.get("physical_domain", metric.get("domain", "")),
                "task": metric.get("task", ""), "cell": metric.get("cell", ""),
                "weighting": metric.get("weighting", run_manifest.get("weighting", "")),
                "setting": metric.get("setting", run_manifest.get("setting", "")),
                "size_role": metric.get("size_role", ""),
                "domain_size": metric.get("domain_size", ""),
                "domain_share": metric.get("domain_share", ""),
            }
            for group in ("shared", "domain_specific", "task_specific", "cell_specific", "architecture"):
                grad = gradient_by_key.get((str(metric.get("epoch", "")),
                                            str(metric.get("domain", "")),
                                            str(metric.get("task", "")),
                                            str(metric.get("cell", "")), group), {})
                cell_gate = gate_by_key.get((str(metric.get("epoch", "")),
                                             str(metric.get("domain", "")),
                                             str(metric.get("task", "")),
                                             str(metric.get("cell", "")), "cell"), {})
                average_gate = gate_average_by_domain.get(
                    (str(metric.get("epoch", "")), str(metric.get("domain", ""))), {})
                for split, auc_field, logloss_field in (
                        ("train", "train_auc", "train_logloss"),
                        ("validation", "validation_auc", "validation_logloss"),
                        ("test", "test_auc", "test_logloss")):
                    row: Dict[str, object] = dict(base)
                    row.update({
                        "split": split,
                        "parameter_group": group,
                        "AUC": metric.get(auc_field, ""),
                        "LogLoss": metric.get(logloss_field, ""),
                        "train_auc": metric.get("train_auc", ""),
                        "train_logloss": metric.get("train_logloss", ""),
                        "validation_auc": metric.get("validation_auc", ""),
                        "validation_logloss": metric.get("validation_logloss", ""),
                        "test_auc": metric.get("test_auc", ""),
                        "test_logloss": metric.get("test_logloss", ""),
                        "gradient_norm": grad.get("gradient_norm", ""),
                        "weighted_gradient_norm": grad.get("weighted_gradient_norm", ""),
                        "gradient_variance": grad.get("gradient_variance", ""),
                        "gradient_SNR": grad.get("gradient_SNR", ""),
                        "norm_of_mean_gradient": grad.get("norm_of_mean_gradient", ""),
                        "probe_count": grad.get("probe_count", ""),
                        "contribution_mean": grad.get("contribution_mean", ""),
                        "contribution_std": grad.get("contribution_std", ""),
                        "mismatch_index": grad.get("mismatch_index", ""),
                        "variance_scope": grad.get("variance_scope", ""),
                        "oracle_auc": metric.get("oracle_auc", ""),
                        "oracle_logloss": metric.get("oracle_logloss", ""),
                        "oracle_source": metric.get("oracle_source", "missing"),
                        "transfer_auc": metric.get("transfer_auc", ""),
                        "transfer_logloss": metric.get("transfer_logloss", ""),
                        "routing_scope": cell_gate.get(
                            "routing_scope", "cell" if cell_gate else ""),
                        "routing_sample_count": cell_gate.get("routing_sample_count", ""),
                        "gate_entropy": cell_gate.get("gate_entropy", ""),
                        "effective_expert_count": cell_gate.get("effective_expert_count", ""),
                        "top1_expert_frequency": cell_gate.get("top1_expert_frequency", ""),
                        "domain_task_average_gate_entropy": average_gate.get("gate_entropy", ""),
                        "domain_task_average_effective_expert_count": average_gate.get(
                            "effective_expert_count", ""),
                        "domain_task_average_top1_expert_frequency": average_gate.get(
                            "top1_expert_frequency", ""),
                    })
                    for expert in range(4):
                        row[f"expert_{expert}_mean_probability"] = cell_gate.get(
                            f"expert_{expert}_mean_probability", "")
                    output.append(row)
        manifest["runs"].append({
            "run_dir": run_dir, "seed": run_manifest.get("seed", expected["seed"]),
            "setting": run_manifest.get("setting", expected["setting"]),
            "weighting": run_manifest.get("weighting", expected["weighting"]),
            "status": run_manifest.get("status", "missing"),
            "pair_status": run_manifest.get("pair_status", ""),
            "pair_valid": run_manifest.get("pair_valid", ""),
            "pair_comparison": run_manifest.get("pair_comparison", {}),
            "metrics_rows": len(metrics), "gradient_rows": len(gradient),
            "gate_rows": len(gate),
            "nonfinite_inventory_count": nonfinite_inventory.get("count", ""),
            "nonfinite_inventory_unexplained_count": nonfinite_inventory.get(
                "unexplained_count", ""),
            "missing_files": missing,
        })
    manifest["summary_row_count"] = len(output)
    return output, manifest


def write_summary(path: str, rows: Sequence[Mapping[str, object]],
                  manifest_path: Optional[str] = None, manifest: Optional[Mapping[str, object]] = None) -> None:
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage E1 raw CSV collector")
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default=None)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    rows, manifest = collect(args.raw_root)
    manifest_path = args.manifest or os.path.splitext(args.output)[0] + "_manifest.json"
    write_summary(args.output, rows, manifest_path, manifest)
    print("[stage_E1_report] rows:", len(rows), "runs:", manifest["run_count"], flush=True)
    print("[stage_E1_report] output:", os.path.abspath(args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
