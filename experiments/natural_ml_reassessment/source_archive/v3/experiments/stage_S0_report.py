"""Mechanical Stage S0 collector and pre-registered paired-summary builder.

The report code never chooses a control with test data.  C_val and D_val are
selected independently inside each (rotation, seed) block from validation
macro-AUC at each arm's already-selected checkpoint.  Outputs are descriptive
tables only; scientific KEEP/STOP interpretation belongs to the main agent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S0_orchestrator import (  # noqa: E402
    ARM_SPECS, DEFAULT_RAW_ROOT, ROTATIONS, SEEDS, audit_all_blocks, build_tasks,
    validate_run, write_json,
)

SUMMARY_FIELDS = [
    "arm", "rotation", "seed", "epoch", "checkpoint_view", "macro_auc", "macro_logloss",
    "sample_weighted_auc", "sample_weighted_logloss", "worst_domain_auc", "worst_cell_auc",
    "raw_auc_nt_rate", "practical_auc_nt_rate", "concordant_practical_nt_rate",
]
PAIR_FIELDS = [
    "comparison", "rotation", "seed", "left_arm", "right_arm", "right_selected_by_validation",
    "epoch_view", "left_epoch", "right_epoch", "macro_auc_difference",
    "macro_logloss_difference", "worst_cell_auc_difference",
]
SUBGROUP_FIELDS = ["comparison", "rotation", "seed", "subgroup_type", "subgroup_value",
                   "auc_gain", "logloss_gain"]
HISTORY_FIELDS = ["arm", "rotation", "seed", "status", "historical_source",
                  "historical_macro_auc", "fresh_macro_auc", "fresh_minus_historical_auc",
                  "included_in_formal_gate", "reason"]
AGG_FIELDS = ["comparison", "unit", "unit_value", "n", "mean", "median", "sample_sd",
              "minimum", "maximum", "positive_count", "direction_denominator"]


def _read_json(path: str) -> Dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_csv(path: str) -> List[Dict[str, str]]:
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _write_csv(path: str, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("required numeric value is missing/non-numeric: {!r}".format(value)) from exc
    if not math.isfinite(result):
        raise ValueError("required numeric value is NaN/Inf: {!r}".format(value))
    return result


def _mean(values: Iterable[object]) -> float:
    numeric = [_float(value) for value in values]
    if not numeric:
        raise ValueError("cannot average an empty required group")
    return sum(numeric) / len(numeric)


def _epoch_rows(rows: Sequence[Mapping[str, str]], epoch: int) -> List[Mapping[str, str]]:
    return [row for row in rows if int(row["epoch"]) == int(epoch)]


def _selected_epoch(rows: Sequence[Mapping[str, str]]) -> int:
    values = {int(row["epoch"]) for row in rows if row.get("selected_checkpoint") == "1"}
    if len(values) != 1:
        raise ValueError("selected checkpoint must be one complete epoch")
    return values.pop()


def validation_score(rows: Sequence[Mapping[str, str]]) -> float:
    chosen = _epoch_rows(rows, _selected_epoch(rows))
    if len(chosen) != 6 or len({(row["domain"], row["task"], row["cell"])
                                for row in chosen}) != 6:
        raise ValueError("validation selector requires six unique finite cells")
    return _mean(row["validation_auc"] for row in chosen)


def summarize_epoch(arm: str, rotation: str, seed: int, rows: Sequence[Mapping[str, str]],
                    epoch: int, view: str) -> Dict[str, object]:
    chosen = _epoch_rows(rows, epoch)
    if len(chosen) != 6:
        raise ValueError("epoch {} is not a complete six-cell endpoint".format(epoch))
    if len({(row["domain"], row["task"], row["cell"]) for row in chosen}) != 6:
        raise ValueError("epoch {} contains duplicate/missing cells".format(epoch))
    shares = {int(row["domain"]): float(row["domain_share"]) for row in chosen}
    domain_auc = {domain: _mean(row["test_auc"] for row in chosen
                                if int(row["domain"]) == domain) for domain in shares}
    sample_auc = sum(shares[int(row["domain"])] * float(row["test_auc"])
                     for row in chosen) / 2.0
    sample_ll = sum(shares[int(row["domain"])] * float(row["test_logloss"])
                    for row in chosen) / 2.0
    covered = [row for row in chosen if row.get("oracle_auc") not in ("", None)]
    return {
        "arm": arm, "rotation": rotation, "seed": seed, "epoch": epoch,
        "checkpoint_view": view, "macro_auc": _mean(row["test_auc"] for row in chosen),
        "macro_logloss": _mean(row["test_logloss"] for row in chosen),
        "sample_weighted_auc": sample_auc, "sample_weighted_logloss": sample_ll,
        "worst_domain_auc": min(domain_auc.values()),
        "worst_cell_auc": min(float(row["test_auc"]) for row in chosen),
        "raw_auc_nt_rate": (_mean(row["raw_auc_nt"] for row in covered) if covered else ""),
        "practical_auc_nt_rate": (_mean(row["practical_auc_nt"] for row in covered) if covered else ""),
        "concordant_practical_nt_rate": (
            _mean(row["concordant_practical_nt"] for row in covered) if covered else ""),
    }


def select_control_family(block_rows: Mapping[str, Sequence[Mapping[str, str]]],
                          family: str) -> str:
    candidates = {
        "C_val": ("uniform_090", "uniform_p_tau", "uniform_eq_tau"),
        "D_val": ("sample_adam", "adam_p_tau", "adam_eq_tau"),
    }[family]
    # Ordered max gives a deterministic first-arm tie break.  No test field is read.
    return max(candidates, key=lambda arm: validation_score(block_rows[arm]))


def historical_reproduction_audit(
    historical_root: str,
    fresh_rows: Mapping[Tuple[str, str, int], Sequence[Mapping[str, str]]],
    block_audits: Mapping[Tuple[str, int], Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Describe legacy A/B availability; never feeds the formal paired gate."""
    rows: List[Dict[str, object]] = []
    root = os.path.abspath(historical_root) if historical_root else ""
    for arm in ("sample_adam", "comp1_domain"):
        for rotation in ROTATIONS:
            for seed in SEEDS:
                fresh = fresh_rows.get((arm, rotation, seed), [])
                fresh_auc: object = ""
                if block_audits.get((rotation, seed), {}).get("valid"):
                    chosen = _epoch_rows(fresh, _selected_epoch(fresh))
                    fresh_auc = _mean(item["test_auc"] for item in chosen)
                path = (os.path.join(root, arm, rotation, f"seed{seed}", "metrics_summary.csv")
                        if root else "")
                historical = _read_csv(path) if path and os.path.isfile(path) else []
                values = []
                for item in historical:
                    value = item.get("best_test_auc", item.get("test_auc", ""))
                    if value not in ("", None):
                        values.append(_float(value))
                if len(values) == 6:
                    historical_auc = _mean(values)
                    status, reason = "AVAILABLE", "descriptive_only"
                    delta = (_float(fresh_auc) - historical_auc) if fresh_auc != "" else ""
                else:
                    historical_auc, delta = "", ""
                    status = "MISSING" if not historical else "INVALID"
                    reason = "expected exactly six finite historical selected-cell metrics"
                rows.append({
                    "arm": arm, "rotation": rotation, "seed": seed, "status": status,
                    "historical_source": path, "historical_macro_auc": historical_auc,
                    "fresh_macro_auc": fresh_auc, "fresh_minus_historical_auc": delta,
                    "included_in_formal_gate": 0, "reason": reason,
                })
    return rows


def collect(raw_root: str, historical_root: str = "") -> Tuple[
        List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]],
        List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    tasks = build_tasks(raw_root)
    all_rows: Dict[Tuple[str, str, int], List[Dict[str, str]]] = {}
    run_entries = []
    for task in tasks:
        valid, reasons = validate_run(task)
        key = (str(task["arm"]), str(task["rotation"]), int(task["seed"]))
        all_rows[key] = _read_csv(os.path.join(str(task["run_dir"]), "metrics_epoch.csv")) if valid else []
        run_entries.append({"formal_run_key": task["formal_run_key"],
                            "status": "VALID" if valid else "INVALID",
                            "reasons": reasons, "run_dir": task["run_dir"]})
    summary_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []
    subgroup_rows: List[Dict[str, object]] = []
    block_audits = {(str(block.get("rotation")), int(block.get("seed", -1))): block
                    for block in audit_all_blocks(tasks)}
    for rotation in ROTATIONS:
        for seed in SEEDS:
            block = {arm: all_rows[(arm, rotation, seed)] for arm in ARM_SPECS}
            if not block_audits.get((rotation, seed), {}).get("valid"):
                continue
            family = {name: select_control_family(block, name) for name in ("C_val", "D_val")}
            endpoint: Dict[Tuple[str, str], Dict[str, object]] = {}
            for arm, rows in block.items():
                selected_epoch = _selected_epoch(rows)
                views = [(selected_epoch, "selected"), (49, "fixed_final")]
                views.extend((epoch, "common_epoch") for epoch in (9, 19, 29, 39, 49))
                for epoch, view in views:
                    item = summarize_epoch(arm, rotation, seed, rows, epoch, view)
                    summary_rows.append(item)
                    endpoint[(arm, f"{view}:{epoch}")] = item
            comparisons = {
                "B_vs_A": "sample_adam", "B_vs_C1": "uniform_p_tau",
                "B_vs_D1": "adam_p_tau", "B_vs_C_val": family["C_val"],
                "B_vs_D_val": family["D_val"],
            }
            for comparison, right_arm in comparisons.items():
                pairs = [("selected", _selected_epoch(block["comp1_domain"]),
                          _selected_epoch(block[right_arm])), ("fixed_final", 49, 49)]
                pairs.extend((f"common_{epoch}", epoch, epoch) for epoch in (9, 19, 29, 39, 49))
                for view, left_epoch, right_epoch in pairs:
                    left = summarize_epoch("comp1_domain", rotation, seed,
                                           block["comp1_domain"], left_epoch, view)
                    right = summarize_epoch(right_arm, rotation, seed,
                                            block[right_arm], right_epoch, view)
                    pair_rows.append({
                        "comparison": comparison, "rotation": rotation, "seed": seed,
                        "left_arm": "comp1_domain", "right_arm": right_arm,
                        "right_selected_by_validation": int(comparison.endswith("_val")),
                        "epoch_view": view, "left_epoch": left_epoch, "right_epoch": right_epoch,
                        "macro_auc_difference": float(left["macro_auc"]) - float(right["macro_auc"]),
                        "macro_logloss_difference": (float(left["macro_logloss"])
                                                     - float(right["macro_logloss"])),
                        "worst_cell_auc_difference": (float(left["worst_cell_auc"])
                                                      - float(right["worst_cell_auc"])),
                    })
            # The pre-registered role redistribution audit is B versus A at selected checkpoints.
            b_rows = _epoch_rows(block["comp1_domain"], _selected_epoch(block["comp1_domain"]))
            a_rows = _epoch_rows(block["sample_adam"], _selected_epoch(block["sample_adam"]))
            subgroup_specs = [
                ("size_role", role, lambda row, value=role: row["size_role"] == value)
                for role in ("L", "M", "S")
            ] + [
                ("task", str(task), lambda row, value=task: int(row["task"]) == value)
                for task in (0, 1)
            ] + [
                ("physical_domain", str(domain),
                 lambda row, value=domain: int(row["physical_domain"]) == value)
                for domain in (0, 1, 2)
            ]
            for subgroup_type, subgroup_value, predicate in subgroup_specs:
                subgroup_rows.append({
                    "comparison": "B_vs_A", "rotation": rotation, "seed": seed,
                    "subgroup_type": subgroup_type, "subgroup_value": subgroup_value,
                    "auc_gain": (_mean(row["test_auc"] for row in b_rows if predicate(row))
                                 - _mean(row["test_auc"] for row in a_rows if predicate(row))),
                    "logloss_gain": (_mean(row["test_logloss"] for row in b_rows if predicate(row))
                                     - _mean(row["test_logloss"] for row in a_rows if predicate(row))),
                })
    aggregate_rows = aggregate_pairs(pair_rows)
    blocks = list(block_audits.values())
    history_rows = historical_reproduction_audit(
        historical_root, all_rows, block_audits)
    manifest = {
        "stage": "S0", "expected_run_count": 84,
        "valid_run_count": sum(entry["status"] == "VALID" for entry in run_entries),
        "complete": all(entry["status"] == "VALID" for entry in run_entries)
                    and all(block["valid"] for block in blocks),
        "runs": run_entries, "block_audits": blocks,
        "selection_guard": "C_val/D_val use validation macro-AUC only within rotation-seed blocks",
        "independence_note": "seed is the repeat; rotations are fixed crossed conditions within seed",
        "scope_note": "descriptive raw aggregation only; no KEEP/STOP conclusion",
    }
    return summary_rows, pair_rows, subgroup_rows, aggregate_rows, history_rows, manifest


def _descriptive(values: Sequence[float]) -> Dict[str, object]:
    finite = [_float(value) for value in values]
    if not finite:
        raise ValueError("descriptive summary received no values")
    return {
        "n": len(finite), "mean": statistics.mean(finite) if finite else "",
        "median": statistics.median(finite) if finite else "",
        "sample_sd": statistics.stdev(finite) if len(finite) > 1 else "",
        "minimum": min(finite) if finite else "", "maximum": max(finite) if finite else "",
        "positive_count": sum(value > 0 for value in finite),
        "direction_denominator": len(finite),
    }


def aggregate_pairs(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    selected = [row for row in rows if row["epoch_view"] == "selected"]
    result = []
    for comparison in sorted({str(row["comparison"]) for row in selected}):
        comp = [row for row in selected if row["comparison"] == comparison]
        # Primary repeat unit: average the three rotations inside each seed.
        seed_values = {seed: _mean(row["macro_auc_difference"] for row in comp
                                   if int(row["seed"]) == seed) for seed in SEEDS}
        item = {"comparison": comparison, "unit": "seed_mean_over_rotations",
                "unit_value": "all_seeds"}
        item.update(_descriptive(list(seed_values.values())))
        result.append(item)
        for rotation in ROTATIONS:
            item = {"comparison": comparison, "unit": "rotation",
                    "unit_value": rotation}
            item.update(_descriptive([float(row["macro_auc_difference"]) for row in comp
                                      if row["rotation"] == rotation]))
            result.append(item)
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage S0 raw report collector")
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--summary", default=os.path.join(HERE, "stage_S0_summary.csv"))
    parser.add_argument("--pairs", default=os.path.join(HERE, "stage_S0_paired.csv"))
    parser.add_argument("--subgroups", default=os.path.join(HERE, "stage_S0_subgroup_gains.csv"))
    parser.add_argument("--historical_root", default=os.path.join(HERE, "history_exp", "first", "results"))
    parser.add_argument("--history_audit", default=os.path.join(HERE, "stage_S0_historical_reproduction_audit.csv"))
    parser.add_argument("--aggregates", default=os.path.join(HERE, "stage_S0_aggregates.csv"))
    parser.add_argument("--manifest", default=os.path.join(HERE, "stage_S0_summary_manifest.json"))
    parser.add_argument("--run_report", default=os.path.join(HERE, "reports", "stage_S0_run_report.md"))
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    summary, pairs, subgroups, aggregates, history, manifest = collect(
        args.raw_root, args.historical_root)
    _write_csv(args.summary, summary, SUMMARY_FIELDS)
    _write_csv(args.pairs, pairs, PAIR_FIELDS)
    _write_csv(args.subgroups, subgroups, SUBGROUP_FIELDS)
    _write_csv(args.aggregates, aggregates, AGG_FIELDS)
    _write_csv(args.history_audit, history, HISTORY_FIELDS)
    write_json(args.manifest, manifest)
    os.makedirs(os.path.dirname(os.path.abspath(args.run_report)), exist_ok=True)
    with open(args.run_report, "w", encoding="utf-8") as handle:
        handle.write("# Stage S0 raw run report\n\n")
        handle.write("- Expected fresh joint runs: 84\n")
        handle.write("- Formal-valid runs: {}\n".format(manifest["valid_run_count"]))
        handle.write("- Complete: {}\n".format(manifest["complete"]))
        handle.write("- C_val/D_val selection: validation macro-AUC only, within each rotation-seed block.\n")
        handle.write("- Repeat unit: seed; rotations are fixed crossed conditions within seed.\n\n")
        handle.write("This is a raw aggregation artifact and contains no scientific conclusion.\n")
    print("[stage_S0_report] valid runs:", manifest["valid_run_count"], "/ 84")
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
