"""Sequential master runner for Stage S8 (Phase 1 + Phase 2).

Uses ``_runner_s8_launch`` (which itself bypasses the orchestrator's
``os.makedirs(..., exist_ok=True)`` bug) to launch one task at a time,
strict serial.  Validates after each task and stops the chain on FAIL.

Task order:

- Phase 1 (base): task-index 0..7 (the script skips the existing
  movielens/seed2022 because the orchestrator returns REFUSED there;
  the script recognises that as "already done, skip").
- Phase 2 (cont): task-index 0..23 (all ML first, then KR).

Phase 3 (cont5) is NOT launched.

Each task record is written to ``stage_S8_base_state.json`` /
``stage_S8_cont_state.json``.  This script also accumulates an
``experiments/stage_S8_runner_summary.json`` with per-task key numbers.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

sys.path.insert(0, HERE)

import stage_S8_protocol as proto  # noqa: E402
from _runner_s8_launch import run_base, run_cont  # noqa: E402

SUMMARY_PATH = os.path.join(HERE, "stage_S8_runner_summary.json")


def _append_summary(record: Dict[str, object]) -> None:
    payload: List[Dict[str, object]] = []
    if os.path.isfile(SUMMARY_PATH):
        try:
            with open(SUMMARY_PATH, encoding="utf-8") as handle:
                decoded = json.load(handle)
                if isinstance(decoded, list):
                    payload = decoded
        except (OSError, ValueError):
            payload = []
    payload.append(record)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def _vbase_key_numbers(base_run_dir: str) -> Dict[str, object]:
    """Return the V-BASE key numbers for one base run dir, or {} on miss."""
    manifest_path = os.path.join(base_run_dir, "run_manifest.json")
    if not os.path.isfile(manifest_path):
        return {}
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {
        "selected_epoch": manifest.get("selected_epoch"),
        "selected_validation_macro_auc": manifest.get("selected_validation_macro_auc"),
        "initial_parameter_state_hash": manifest.get("initial_parameter_state_hash"),
        "vbase_macro_auc_abs": manifest.get("vbase_macro_auc_abs"),
    }


def _cont_key_numbers(out_dir: str) -> Dict[str, object]:
    audit_path = os.path.join(out_dir, "replay_audit.json")
    if not os.path.isfile(audit_path):
        return {}
    with open(audit_path, encoding="utf-8") as handle:
        audit = json.load(handle)
    validity = audit.get("validity", {}) or {}
    return {
        "v0_anchor_macro_auc_diff": (validity.get("v0_anchor", {}) or {}).get("max_macro_auc_diff"),
        "v0_step1_L_full_diff": (validity.get("v0_step1", {}) or {}).get("l_full_abs_diff"),
        "v0_step1_delta_rel_diff": (validity.get("v0_step1", {}) or {}).get("delta_norm_rel_diff"),
        "v3_per_pair_all_equal": audit.get("v3_per_pair_all_equal"),
        "v4_main_fingerprint_matches": audit.get("v4_main_fingerprint_matches"),
        "v4_main_fingerprint_total": audit.get("v4_main_fingerprint_total"),
        "v4_arch_fingerprint_matches": audit.get("v4_arch_fingerprint_matches"),
        "v4_arch_fingerprint_total": audit.get("v4_arch_fingerprint_total"),
        "v5_shared_step_observed": audit.get("shared_step_observed"),
        "v5_shared_step_expected": audit.get("shared_step_expected"),
        "v5_arch_step_observed": audit.get("arch_step_observed"),
        "v5_arch_step_expected": audit.get("arch_step_expected"),
        "private_state_coverage": audit.get("private_state_coverage"),
        "review_flagged": audit.get("review_flagged"),
    }


def _already_done(state_path: str, task_index: int) -> bool:
    """Return True iff ``state_path`` already records an OK entry for the
    given task_index."""
    if not os.path.isfile(state_path):
        return False
    try:
        with open(state_path, encoding="utf-8") as handle:
            decoded = json.load(handle)
    except (OSError, ValueError):
        return False
    if not isinstance(decoded, list):
        return False
    return any(isinstance(entry, dict) and entry.get("task_index") == task_index
               and entry.get("status") == "OK" for entry in decoded)


def main() -> int:
    parser_args: List[str] = sys.argv[1:]
    phase_filter = "all"
    if "--phase" in parser_args:
        idx = parser_args.index("--phase")
        phase_filter = parser_args[idx + 1]
    overall_start = time.time()
    phase1_records: List[Dict[str, object]] = []
    phase2_records: List[Dict[str, object]] = []

    if phase_filter in ("all", "phase1", "base"):
        for idx, task in enumerate(proto.base_tasks()):
            dataset = str(task["dataset"])
            seed = int(task["seed"])
            base_run_dir = str(task["run_dir"])
            print("[master] Phase 1 task-index={} dataset={} seed={} -- launching".format(
                idx, dataset, seed), flush=True)
            if _already_done(proto.STATE_PATH_BASE, idx):
                record = {
                    "phase": "base", "task_index": idx,
                    "dataset": dataset, "seed": seed,
                    "formal_run_key": str(task["formal_run_key"]),
                    "run_dir": base_run_dir,
                    "status": "SKIPPED_ALREADY_OK",
                    "abort_reason": "previous state JSON already has OK entry for this task",
                    "exit_code": 0, "wall_seconds": 0.0,
                    "max_processes": 1, "timestamp": time.time(),
                }
                key_numbers = _vbase_key_numbers(base_run_dir)
            else:
                record = run_base(dataset, seed, base_run_dir,
                                   str(task["formal_run_key"]),
                                   proto.STATE_PATH_BASE, idx)
                key_numbers = _vbase_key_numbers(base_run_dir)
            record_summary = dict(record)
            record_summary["key_numbers"] = key_numbers
            phase1_records.append(record_summary)
            _append_summary({"phase": "base", "task_index": idx, "summary": record_summary})
            print("[master] Phase 1 task-index={} status={} abort={} wall={:.1f}s".format(
                idx, record["status"], record.get("abort_reason", ""),
                float(record.get("wall_seconds", 0.0))), flush=True)
            if record["status"] not in ("OK", "SKIPPED_ALREADY_OK", "REFUSED"):
                print("[master] Phase 1 halted at task-index={} (status={})".format(
                    idx, record["status"]), flush=True)
                break

    if phase_filter in ("all", "phase2", "cont"):
        for idx, task in enumerate(proto.cont_tasks(h_max_epochs=proto.H_MAX_EPOCHS_CORE)):
            dataset = str(task["dataset"])
            seed = int(task["seed"])
            anchor_class = str(task["anchor_class"])
            anchor_epoch = int(task["anchor_epoch"])
            out_dir = str(task["run_dir"])
            print("[master] Phase 2 task-index={} dataset={} seed={} anchor={}/{} -- launching".format(
                idx, dataset, seed, anchor_class, anchor_epoch), flush=True)
            if _already_done(proto.STATE_PATH_CONT, idx):
                record = {
                    "phase": "cont", "task_index": idx,
                    "dataset": dataset, "seed": seed,
                    "anchor_class": anchor_class, "anchor_epoch": anchor_epoch,
                    "h_max_epochs": proto.H_MAX_EPOCHS_CORE,
                    "formal_run_key": str(task["formal_run_key"]),
                    "run_dir": out_dir,
                    "status": "SKIPPED_ALREADY_OK",
                    "abort_reason": "previous state JSON already has OK entry for this task",
                    "exit_code": 0, "wall_seconds": 0.0,
                    "max_processes": 1, "timestamp": time.time(),
                }
                key_numbers = _cont_key_numbers(out_dir)
            else:
                record = run_cont(dataset, seed, anchor_class, out_dir,
                                   str(task["formal_run_key"]), anchor_epoch,
                                   proto.H_MAX_EPOCHS_CORE,
                                   proto.STATE_PATH_CONT, idx)
                key_numbers = _cont_key_numbers(out_dir)
            record_summary = dict(record)
            record_summary["key_numbers"] = key_numbers
            phase2_records.append(record_summary)
            _append_summary({"phase": "cont", "task_index": idx, "summary": record_summary})
            print("[master] Phase 2 task-index={} status={} abort={} wall={:.1f}s".format(
                idx, record["status"], record.get("abort_reason", ""),
                float(record.get("wall_seconds", 0.0))), flush=True)
            if record["status"] not in ("OK", "SKIPPED_ALREADY_OK", "REFUSED"):
                print("[master] Phase 2 halted at task-index={} (status={})".format(
                    idx, record["status"]), flush=True)
                break

    total_wall = time.time() - overall_start
    final = {
        "total_wall_seconds": float(total_wall),
        "phase1_records": phase1_records,
        "phase2_records": phase2_records,
    }
    with open(os.path.join(HERE, "stage_S8_runner_final.json"), "w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print("[master] DONE wall={:.1f}s".format(total_wall), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())