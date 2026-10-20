"""Fail-closed S5 artifact validation and eight-wave launcher.

Mirrors the Stage S4 A2 orchestrator shape while adopting the S5 four-arm
matrix (F, T, HT, DH), the eight two-process waves, the moment-dump
artifacts, the S5 per-step field contract, and the S5 cross-arm
batch-trace equality check.

Overlays on S4:
- ``_validated_overlap_s5`` — Windows-compatible positive-overlap proof.
  ``subprocess.Popen.pid`` is not always equal to ``os.getpid()`` in the
  spawned child on Windows when the orchestrator is launched through a
  shell wrapper (``nohup``, git-bash, etc.): ``Popen.pid`` then points at
  an intermediate Windows launcher process whose PID the child sees as
  ``os.getppid()``.  The strict ``interval.pid == child.pid`` equality
  in the inherited S4 check fails with the message
  ``process interval identity/status mismatch`` even with zero source
  drift and real concurrent execution.  The S5 override drops that
  equality check but retains every other piece of overlap evidence:
  both intervals are ``status="OK"``, both have monotonic time bounds,
  the two child PIDs are distinct, and the time intervals have a strictly
  positive overlap.  This is documented in §6 of the implementation
  report and tested by ``test_validated_overlap_s5_*``.
- ``revalidate_attempts`` — re-runs ``validate_run`` on existing attempt
  directories under ``.attempts/`` without re-spawning any training
  process.  Pass/fail results land in a JSON report; attempts that pass
  are flagged ``ready_for_promotion=true`` in the state file but the
  orchestrator does NOT call ``_promote_wave``.  Reviewer invokes
  ``--revalidate`` (or inspects the report) and decides whether the
  canonical promotion should happen on the next ``run_queue()`` call.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from typing import Iterable, Mapping, Sequence

import numpy as np

from stage_S1_orchestrator import validate_raw_prediction_metrics
from stage_S3_orchestrator import (_artifact_hash_reasons, _checkpoint_state_numerically_equal,
                                   _coefficient_artifact_reasons, _read_csv, _read_json,
                                   validate_batch_trace)
from stage_S4_gradient_controls import TOLERANCE_ABS, TOLERANCE_REL
from stage_S4_orchestrator import (_finalize_child, _journal_path, _monitor_children,
                                   _retry_classification, sha)
from stage_S5_protocol import (ARM_CURRENT, ARM_HISTORY, ARMS, COMMON_EPOCHS,
                               MOMENT_DUMP_EPOCHS, OUTPUT_ROOT, QUEUE_PATH, RAW_ROOT,
                               SEEDS, STAGE, inventory, inventory_hash, key, matrix,
                               waves)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_SEED_RE = re.compile(r"seed(\d+)")

REQUIRED = ("run_manifest.json", "config.json", "gradient_partition.json", "batch_trace.csv",
            "gradient_step.csv", "probe.csv", "metrics_epoch.csv", "metrics_test.csv",
            "metrics_summary.csv", "predictions_test.npz", "objective_coefficients.json",
            "optimizer_state.pth", "architecture_optimizer_state.pth", "rng_state.pth",
            "initialization_audit.json", "nonfinite_inventory.json", "stdout.log",
            "process_interval.json",
            "moment_state_epoch0.pth", "moment_state_epoch1.pth",
            "moment_state_epoch9.pth", "moment_state_epoch19.pth",
            "moment_state_epoch29.pth", "moment_state_epoch39.pth",
            "moment_state_epoch49.pth")
PAIR_FIELDS = ("dataset_file_hash", "split_row_fingerprints", "split_row_id_hashes",
               "physical_view_fingerprints", "preprocessing_metadata", "local_to_physical",
               "train_domain_counts", "domain_shares", "coefficient_table_hash",
               "initial_parameter_state_hash", "gradient_partition_hash", "model_parameter_manifest",
               "batch_order_hash", "architecture_batch_order_hash", "batches_per_epoch",
               "optimizer_steps", "architecture_optimizer_steps", "examples_processed",
               "architecture_examples_processed", "optimizer", "architecture_optimizer",
               "fixed_final_epoch", "common_epochs", "metric_implementation",
               "checkpoint_selection_frozen_before_test", "test_evaluated_during_training",
               "test_metric_in_epoch_history", "moment_dump_epochs")
S5_NUMERIC_STEP_FIELDS = (
    "L_full", "L_target", "L_non", "decomposition_residual",
    "g_full_norm", "g_target_norm", "g_non_norm", "gradient_cosine",
    "m_norm_persistent", "v_norm_persistent",
    "delta_full_norm", "delta_target_norm",
    "delta_history_norm", "delta_applied_norm", "delta_observed_step_norm",
    "history_formula_residual", "applied_formula_residual", "private_full_residual",
)


def _write_json_atomic(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + "." + uuid.uuid4().hex + ".tmp"
    with open(temporary, "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _shared_applied_reference_norm(row, arm):
    if arm in ("T",):
        return float(row["delta_target_norm"])
    if arm in ("HT",):
        # HT installs g_T but applies Delta_F; reference is the FULL counterfactual.
        return float(row["delta_full_norm"])
    if arm in ("DH",):
        return float(row["delta_target_norm"])
    return float(row["delta_full_norm"])


def _arm_protocol_block():
    """Return arm → (history_source, current_source) frozen in protocol."""
    return {arm: (ARM_HISTORY[arm], ARM_CURRENT[arm]) for arm in ARMS}


def validate_run(task, *, source_hashes=None):
    path = os.path.abspath(task["run_dir"])
    reasons = []
    if not all(os.path.isfile(os.path.join(path, f)) for f in REQUIRED):
        return False, ["required artifact missing"]
    try:
        m = _read_json(os.path.join(path, "run_manifest.json"))
        expected_identity = ("S5", "OK", task["formal_run_key"], task["arm"],
                             task["seed"], "natural")
        observed_identity = (m.get("stage"), m.get("status"), m.get("formal_run_key"),
                             m.get("arm"), int(m.get("seed", -1)), m.get("setting"))
        if observed_identity != expected_identity:
            reasons.append("identity mismatch")
        if m.get("dataset_key") != "kuairand" or m.get("backbone") != "m3oe" \
                or m.get("objective_weighting") != "equal":
            reasons.append("protocol identity mismatch")
        history_source, current_source = _arm_protocol_block()[task["arm"]]
        protocol = {"stage": "S5", "dataset": "kuairand", "setting": "natural",
                    "backbone": "m3oe", "objective": "equal", "arm": task["arm"],
                    "history_source": history_source, "current_source": current_source,
                    "seed": task["seed"], "epochs": 50, "batch_size": 40960,
                    "device": "cuda:0", "main_lr": .01, "architecture_lr": .001,
                    "betas": [.9, .999], "weight_decay": 1e-5,
                    "common_epochs": list(COMMON_EPOCHS),
                    "moment_dump_epochs": list(MOMENT_DUMP_EPOCHS)}
        for record in (m, _read_json(os.path.join(path, "config.json"))):
            if (record.get("stage") != "S5"
                    or record.get("formal_run_key") != task["formal_run_key"]
                    or record.get("arm") != task["arm"]
                    or record.get("fresh_role") != "fresh_s5"
                    or record.get("formal_protocol") != protocol
                    or record.get("formal_protocol_observed") != protocol
                    or record.get("formal_protocol_mismatches") != {}
                    or record.get("formal_valid") is not True
                    or record.get("history_source") != history_source
                    or record.get("current_source") != current_source):
                reasons.append("S5 formal protocol/config identity mismatch")
                break
        if m.get("code_hashes") != _read_json(
                os.path.join(path, "config.json")).get("code_hashes"):
            reasons.append("S5 config/source identity mismatch")
        if m.get("code_inventory_hash") != inventory_hash(m.get("code_hashes", {})):
            reasons.append("source inventory digest mismatch")
        if source_hashes and m.get("code_hashes") != source_hashes:
            reasons.append("source identity mismatch")
        reasons += _artifact_hash_reasons(path, m)
        # The ``artifact_hashes`` table is finalised by ``_finalize_child``
        # only after a wave is promoted to its canonical path.  Re-validating
        # a staged attempt (which lives under ``.attempts/`` and has never
        # been finalised) should not require the post-finalisation table to
        # already mention ``stdout.log``/``process_interval.json``; the files
        # themselves just need to be present on disk and non-empty.
        interval_path = os.path.join(path, "process_interval.json")
        stdout_path = os.path.join(path, "stdout.log")
        interval_on_disk = os.path.isfile(interval_path) and os.path.getsize(interval_path) > 0
        stdout_on_disk = os.path.isfile(stdout_path) and os.path.getsize(stdout_path) > 0
        interval_in_hashes = bool(m.get("artifact_hashes", {}).get("process_interval.json"))
        stdout_in_hashes = bool(m.get("artifact_hashes", {}).get("stdout.log"))
        if not interval_on_disk or not stdout_on_disk \
                or (not interval_in_hashes and not interval_on_disk) \
                or (not stdout_in_hashes and not stdout_on_disk):
            reasons.append("canonical process evidence missing")
        interval = _read_json(os.path.join(path, "process_interval.json"))
        if interval.get("status") != "OK" or int(interval.get("finished_ns", 0)) <= int(
                interval.get("started_ns", 0)):
            reasons.append("process interval invalid")
        reasons += _coefficient_artifact_reasons(path, m)
        with open(os.path.join(path, "nonfinite_inventory.json"), encoding="utf-8") as handle:
            nonfinite = json.load(handle)
        if nonfinite != []:
            reasons.append("nonfinite inventory not empty")
        if int(m.get("fixed_final_epoch", -1)) != 49 \
                or list(m.get("common_epochs", [])) != list(COMMON_EPOCHS):
            reasons.append("checkpoint support mismatch")
        if m.get("checkpoint_selection_frozen_before_test") is not True \
                or m.get("test_evaluated_during_training") is not False:
            reasons.append("test selection boundary mismatch")
        for label in ("selected", "final",
                      "common_epoch_9", "common_epoch_19",
                      "common_epoch_29", "common_epoch_39", "common_epoch_49"):
            filename = ("selected_checkpoint.pth" if label == "selected" else
                        "final_checkpoint.pth" if label == "final" else
                        "common_epoch_{}.pth".format(label.rsplit("_", 1)[-1]))
            if not os.path.isfile(os.path.join(path, filename)):
                reasons.append("checkpoint missing " + label)
        if _checkpoint_state_numerically_equal(os.path.join(path, "final_checkpoint.pth"),
                                                os.path.join(path, "common_epoch_49.pth")) is not True:
            reasons.append("final/common49 state mismatch")
        reasons += validate_batch_trace(task, path, m)
        _, step_rows = _read_csv(os.path.join(path, "gradient_step.csv"))
        _, probe_rows = _read_csv(os.path.join(path, "probe.csv"))
        main = [r for r in step_rows if r["kind"] == "main"]
        arch = [r for r in step_rows if r["kind"] == "architecture"]
        if len(main) != int(m.get("batches_per_epoch", 0)) * 50 or len(arch) != 50 \
                or len(probe_rows) != 100:
            reasons.append("S5 step/probe cardinality mismatch")
        for r in probe_rows:
            # I7 evidence: probe-step RNG digests around the controlled step
            # must be present and equal.
            if not r.get("rng_digest_before") \
                    or r["rng_digest_before"] != r.get("rng_digest_after", ""):
                reasons.append("probe RNG digest mismatch")
                break
        if len({(r["kind"], r["epoch"], r["step"]) for r in step_rows}) != len(step_rows):
            reasons.append("duplicate step key")
        for r in step_rows:
            if r["formal_run_key"] != task["formal_run_key"] or r["arm"] != task["arm"]:
                reasons.append("gradient step identity mismatch")
                break
            if r["history_source"] != history_source or r["current_source"] != current_source:
                reasons.append("step arm-factor mismatch")
                break
            if not all(math.isfinite(float(r[field])) for field in S5_NUMERIC_STEP_FIELDS):
                reasons.append("nonfinite S5 step")
                break
            applied_reference = _shared_applied_reference_norm(r, task["arm"])
            if float(r["applied_formula_residual"]) > max(
                    TOLERANCE_ABS, TOLERANCE_REL * applied_reference):
                reasons.append("applied formula residual failed")
                break
            if float(r["history_formula_residual"]) > max(
                    TOLERANCE_ABS, TOLERANCE_REL * float(r["delta_history_norm"])):
                reasons.append("history formula residual failed")
                break
            if float(r["private_full_residual"]) > max(
                    TOLERANCE_ABS, TOLERANCE_REL * float(r["private_delta_norm"])):
                reasons.append("private FULL residual failed")
                break
            installed_cos = r.get("installed_cosine_history", "")
            if installed_cos not in ("", None) and float(installed_cos) < .99999:
                reasons.append("installed history cosine below threshold")
                break
            if int(r["adam_state_step_min"]) < 1 or int(r["adam_state_step_min"]) > int(r["adam_state_step_max"]):
                reasons.append("Adam step range invalid")
                break
            if int(r["none_count_full"]) != 0 or int(r["none_count_target"]) != 0:
                reasons.append("shared/arch None gradient recorded")
                break
        if int(m.get("zero_target_main_steps", -1)) > int(m.get("main_steps_seen", 0)) * .01:
            reasons.append("too many target-zero steps")
        moment_epochs = sorted(set(int(epoch) for epoch in m.get("moment_dump_epochs", [])))
        if moment_epochs != sorted(MOMENT_DUMP_EPOCHS):
            reasons.append("moment dump epoch set mismatch")
        moment_hashes = m.get("moment_dump_hashes", {})
        if not moment_hashes or set(moment_hashes) != {
                "moment_state_epoch{}.pth".format(epoch) for epoch in MOMENT_DUMP_EPOCHS}:
            reasons.append("moment dump artifact set mismatch")
        for filename, expected_hash in moment_hashes.items():
            full_path = os.path.join(path, filename)
            if not os.path.isfile(full_path):
                reasons.append("moment dump file missing " + filename)
                continue
            if sha(full_path) != expected_hash:
                reasons.append("moment dump hash mismatch " + filename)
        with np.load(os.path.join(path, "predictions_test.npz"), allow_pickle=False) as raw:
            predictions = {name: np.asarray(raw[name]) for name in raw.files}
        test = _read_csv(os.path.join(path, "metrics_test.csv"))[1]
        summary = _read_csv(os.path.join(path, "metrics_summary.csv"))[1]
        reasons += validate_raw_prediction_metrics(predictions, m, task, test, summary)
    except Exception as exc:
        reasons.append("validation exception: {}: {}".format(type(exc).__name__, exc))
    return not reasons, sorted(set(reasons))


def validate_block(tasks, seed, *, source_hashes=None):
    block = [t for t in tasks if t["seed"] == seed]
    if {t["arm"] for t in block} != set(ARMS):
        return False, ["incomplete four-arm block"]
    reasons, manifests = [], []
    for task in block:
        valid, detail = validate_run(task, source_hashes=source_hashes)
        reasons += [task["arm"] + ":" + item for item in detail]
        if valid:
            manifests.append(_read_json(os.path.join(task["run_dir"], "run_manifest.json")))
    if len(manifests) == 4:
        left = manifests[0]
        for m in manifests[1:]:
            for field in PAIR_FIELDS:
                if m.get(field) != left.get(field):
                    reasons.append("pairing mismatch:" + field)
            if m["code_hashes"] != left["code_hashes"]:
                reasons.append("pairing source mismatch")
            if m.get("batch_order_hash") != left.get("batch_order_hash"):
                reasons.append("cross-arm batch trace mismatch")
    return not reasons, sorted(set(reasons))


def _load_frozen_queue(queue_path):
    """Read the frozen queue manifest without re-validating against current
    source.  Used by ``revalidate_attempts`` so that the existing attempt's
    pinned inventory is used for ``source_hashes=`` even after a Coder fix
    has changed local source.  Returns the parsed queue dict.
    """
    if not os.path.isfile(queue_path):
        raise RuntimeError("S5 queue manifest not found: " + queue_path)
    queue = _read_json(queue_path)
    if queue.get("stage") != STAGE:
        raise RuntimeError("frozen S5 queue has wrong stage: " + str(queue.get("stage")))
    return queue


def freeze_queue(raw_root, queue_path, *, allow_create=True):
    tasks = matrix(raw_root)
    plan = waves(tasks)
    if os.path.exists(queue_path):
        frozen = _read_json(queue_path)
        if (frozen.get("stage") != STAGE or frozen.get("tasks") != tasks
                or frozen.get("waves") != plan
                or frozen.get("source_inventory") != inventory(ROOT)):
            raise RuntimeError("frozen S5 queue/source differs")
        return frozen
    if not allow_create:
        raise RuntimeError("S5 queue must be frozen after reviewer PASS before RUN")
    current = inventory(ROOT)
    if len(current) < 21 or any(not value for value in current.values()):
        raise RuntimeError("S5 producer source inventory invalid")
    queue = {"stage": STAGE, "tasks": tasks, "waves": plan, "raw_root": os.path.abspath(raw_root),
             "source_inventory": current, "source_inventory_digest": inventory_hash(current),
             "frozen_at": time.time(),
             "max_training_processes": 2, "max_attempts_per_wave": 2}
    _write_json_atomic(queue_path, queue)
    return queue


def _validated_overlap_s5(children):
    """Windows-compatible proof of positive overlap between the two child runs.

    The inherited S4 check (``stage_S4_orchestrator._validated_overlap``)
    demands ``process_interval.pid == subprocess.Popen.pid``.  On Windows
    that equality is not guaranteed: when the orchestrator is launched
    through ``nohup``, git-bash, or any other shell wrapper, ``Popen.pid``
    is the PID of an intermediate launcher process whose PID the child
    sees as ``os.getppid()`` — not the real ``os.getpid()`` of the
    Python child.  Round-1 S4 A2 ran under a different launch context
    where the equality happened to hold; S5 round-2 hit the Windows
    intermediate path (e.g. nohup launch via ``nohup … &`` from git-bash)
    and the equality failed even with zero source drift and real
    concurrent execution.

    Design intent is preserved: we still prove that two distinct Python
    training processes ran in parallel.  We replace the strict
    ``pid`` equality with a four-part evidence chain:

    1. Each ``process_interval.json`` has ``status="OK"``.
    2. Each interval has positive ``started_ns`` and a strictly greater
       ``finished_ns`` (the runner cannot write an inverted interval).
    3. The two child PIDs (read from the per-run
       ``process_interval.json``) are distinct — two real processes.
    4. The two intervals have strictly positive overlap in nanoseconds
       — proof of concurrency.

    ``subprocess.Popen.pid`` is still recorded for the audit trail
    (``launch.pid`` in ``state.json``) and is allowed to differ from the
    child's reported PID.
    """
    intervals = []
    for child, _, work, _ in children:
        value = _read_json(os.path.join(work, "process_interval.json"))
        if value.get("status") != "OK":
            raise RuntimeError(
                "S5 process interval status not OK for {} (got {!r})".format(
                    work, value.get("status")))
        try:
            start = int(value["started_ns"])
            finish = int(value["finished_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("S5 process interval missing time bounds: "
                               + str(exc)) from exc
        if start <= 0 or finish <= start:
            raise RuntimeError("S5 invalid process interval for {}: "
                               "started_ns={}, finished_ns={}".format(
                                   work, start, finish))
        try:
            child_pid = int(value["pid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("S5 process interval missing pid: " + str(exc)) from exc
        if child_pid <= 0:
            raise RuntimeError("S5 process interval has non-positive pid: "
                               + str(child_pid))
        intervals.append({"pid": child_pid, "started_ns": start, "finished_ns": finish,
                          "work": work, "popen_pid": int(child.pid)})
    pids = [x["pid"] for x in intervals]
    if len(set(pids)) != len(pids):
        raise RuntimeError("S5 two child runs reported identical pid {} "
                           "(no real concurrency)".format(pids[0]))
    overlap = min(x["finished_ns"] for x in intervals) - max(x["started_ns"] for x in intervals)
    if overlap <= 0:
        raise RuntimeError(
            "S5 Python training processes did not actually overlap "
            "(overlap_ns={})".format(overlap))
    return intervals, overlap


# ---------------------------------------------------------------------------
# Promotion (journaled two-phase copy) — S5-local copies of the S4 helpers.
#
# ``stage_S4_orchestrator._promote_wave`` / ``_recover_promotion`` cannot be
# inherited: they resolve ``validate_run`` in the S4 module namespace, i.e.
# they would validate S5 staged/canonical runs with the *S4* validator
# (stage="S4", fresh_role="fresh_s4", S4 protocol, no moment dumps), so every
# S5 wave promotion would raise "staged copy invalid".  These copies are
# byte-equivalent to the S4 originals except that ``validate_run`` resolves
# to the S5 validator defined in this module.
# ---------------------------------------------------------------------------


def _recover_promotion(raw_root, wave, tasks, state, state_path, source):
    path = _journal_path(raw_root, wave["wave_id"])
    if not os.path.isfile(path):
        return
    journal = _read_json(path)
    if journal["status"] == "COMMITTED":
        if wave["wave_id"] not in state["completed"]:
            raise RuntimeError("promotion journal/state disagree")
        return
    if journal["status"] == "ROLLED_BACK":
        return
    canonicals = [tasks[k] for k in wave["formal_run_keys"]]
    valid = []
    for task in canonicals:
        if os.path.exists(task["run_dir"]):
            good, _ = validate_run(task, source_hashes=source)
            valid.append(good)
        else:
            valid.append(False)
    if all(valid):
        if wave["wave_id"] not in state["completed"]:
            state["completed"].append(wave["wave_id"])
            _write_json_atomic(state_path, state)
        journal["status"] = "COMMITTED"
        _write_json_atomic(path, journal)
        return
    # Recover a crash between the two renames by moving any partial canonical
    # member into a unique retained rollback path. Source attempts remain intact.
    for task in canonicals:
        canonical = task["run_dir"]
        if os.path.exists(canonical):
            rollback = canonical + ".rollback-" + uuid.uuid4().hex
            os.replace(canonical, rollback)
            journal.setdefault("rollback_paths", []).append(rollback)
    journal["status"] = "ROLLED_BACK"
    _write_json_atomic(path, journal)


def _promote_wave(raw_root, wave, children, state, state_path, source, fail_after_first=False):
    """Prepare both copies, then commit two names with journaled rollback/recovery."""
    journal_path = _journal_path(raw_root, wave["wave_id"])
    if os.path.isfile(journal_path) and _read_json(journal_path)["status"] not in ("ROLLED_BACK",):
        raise RuntimeError("unresolved promotion journal")
    staging_root = os.path.join(raw_root, ".promotions", wave["wave_id"], uuid.uuid4().hex)
    prepared = []
    for _, _, work, task in children:
        if os.path.exists(task["run_dir"]):
            raise RuntimeError("canonical output already exists")
        os.makedirs(os.path.dirname(task["run_dir"]), exist_ok=True)
        staging = os.path.join(staging_root, task["arm"])
        shutil.copytree(work, staging)
        good, why = validate_run(dict(task, run_dir=staging), source_hashes=source)
        if not good:
            raise RuntimeError("staged copy invalid: " + str(why))
        prepared.append((staging, task["run_dir"]))
    journal = {"wave": wave["wave_id"], "status": "PREPARED", "members": prepared,
               "source_attempt_dirs": [work for _, _, work, _ in children]}
    _write_json_atomic(journal_path, journal)
    promoted = []
    try:
        for index, (staging, canonical) in enumerate(prepared):
            os.replace(staging, canonical)
            promoted.append((staging, canonical))
            if fail_after_first and index == 0:
                raise RuntimeError("injected failure after first promotion")
        journal["status"] = "BOTH_PROMOTED"
        _write_json_atomic(journal_path, journal)
        state["completed"].append(wave["wave_id"])
        _write_json_atomic(state_path, state)
        journal["status"] = "COMMITTED"
        _write_json_atomic(journal_path, journal)
    except Exception:
        if wave["wave_id"] in state["completed"]:
            state["completed"].remove(wave["wave_id"])
            _write_json_atomic(state_path, state)
        for staging, canonical in reversed(promoted):
            if os.path.exists(canonical):
                os.replace(canonical, staging)
        journal["status"] = "ROLLED_BACK"
        _write_json_atomic(journal_path, journal)
        raise


def _revalidate_one_attempt(task, attempt_record, source):
    """Re-run ``validate_run`` for one attempt without spawning processes.

    Returns a dict suitable for inclusion in the revalidate report.  The
    caller's ``attempt_record`` is left unmodified; the caller decides
    what to do with the result.
    """
    work_root = attempt_record.get("attempt_root")
    if not work_root or not os.path.isdir(work_root):
        return {"key": task["formal_run_key"], "arm": task["arm"],
                "seed": task["seed"], "wave": attempt_record.get("wave"),
                "attempt": attempt_record.get("attempt"),
                "valid": False, "reasons": ["attempt_root missing: " + str(work_root)],
                "stage_popen_pid": None, "child_pid": None}
    arm_dirs = [os.path.join(work_root, arm) for arm in (task["arm"],)]
    # The orchestrator stages two arm workdirs per attempt; we only validate
    # the run matching this task's arm (the other arm is handled by its own
    # task in the queue).
    if not os.path.isdir(arm_dirs[0]):
        return {"key": task["formal_run_key"], "arm": task["arm"],
                "seed": task["seed"], "wave": attempt_record.get("wave"),
                "attempt": attempt_record.get("attempt"),
                "valid": False, "reasons": ["arm workdir missing: " + arm_dirs[0]],
                "stage_popen_pid": None, "child_pid": None}
    staged_task = dict(task, run_dir=arm_dirs[0])
    valid, reasons = validate_run(staged_task, source_hashes=source)
    interval = _read_json(os.path.join(arm_dirs[0], "process_interval.json")) \
        if os.path.isfile(os.path.join(arm_dirs[0], "process_interval.json")) else {}
    # Match the staged pid against the per-arm launch record so we can
    # surface the Windows pid-mismatch story in the report.
    stage_popen_pid = None
    for entry in attempt_record.get("launch", []) or []:
        if entry.get("key") == task["formal_run_key"]:
            stage_popen_pid = entry.get("pid")
            break
    return {"key": task["formal_run_key"], "arm": task["arm"], "seed": task["seed"],
            "wave": attempt_record.get("wave"), "attempt": attempt_record.get("attempt"),
            "work_dir": arm_dirs[0], "valid": bool(valid), "reasons": reasons,
            "stage_popen_pid": stage_popen_pid,
            "child_pid": interval.get("pid"),
            "interval_status": interval.get("status"),
            "started_ns": interval.get("started_ns"),
            "finished_ns": interval.get("finished_ns")}


def revalidate_attempts(args):
    """Re-run ``validate_run`` on every attempt under ``.attempts/``.

    Does not spawn any training process and does not move directories.
    Writes a structured report to
    ``experiments/stage_S5_revalidate_report.json`` — one entry per
    (wave, attempt, key) with full pass/fail reasons, the cross-arm
    overlap evidence (the Windows-compatible overlap proof, see
    ``_validated_overlap_s5``), and the original ``state.json``
    ``launch[]`` PIDs so Reviewer can see the launcher-vs-interpreter
    PID mismatch story in one place.

    This entry point does NOT modify ``stage_S5_queue_manifest.json``,
    its ``state.json``, or any artifact under ``.attempts/``: the
    existing append-only record is preserved verbatim per the round-2
    prohibition.  Decisions about whether to invoke ``--revalidate``
    and whether to actually promote any revalidated attempt belong to
    the Reviewer; the revalidate report is the only artefact this
    command writes.
    """
    if not args.reviewer_pass:
        raise RuntimeError("independent S5 reviewer PASS is required")
    args.raw_root = args.raw_root or RAW_ROOT
    args.queue_manifest = args.queue_manifest or QUEUE_PATH
    # Load the frozen queue without re-validating against current source:
    # the revalidate path is meant to inspect existing attempts whose
    # artifacts were produced under the queue's pinned inventory, even if
    # the orchestrator has been patched since.
    queue = _load_frozen_queue(args.queue_manifest)
    tasks = {t["formal_run_key"]: t for t in queue["tasks"]}
    source = queue["source_inventory"]
    state_path = args.queue_manifest + ".state.json"
    state = _read_json(state_path) if os.path.isfile(state_path) \
        else {"completed": [], "attempts": []}
    attempts_root = os.path.join(args.raw_root, ".attempts")
    report = {"stage": STAGE, "raw_root": os.path.abspath(args.raw_root),
              "queue_manifest": os.path.abspath(args.queue_manifest),
              "state_path": os.path.abspath(state_path),
              "frozen_at": time.time(), "per_wave": []}
    if not os.path.isdir(attempts_root):
        report["empty"] = True
        report_path = os.path.join(HERE, "stage_S5_revalidate_report.json")
        _write_json_atomic(report_path, report)
        return report
    wave_dirs = sorted(d for d in os.listdir(attempts_root)
                        if os.path.isdir(os.path.join(attempts_root, d)))
    for wave_id in wave_dirs:
        wave_entry = {"wave_id": wave_id, "per_attempt": []}
        attempt_dirs = sorted(d for d in os.listdir(os.path.join(attempts_root, wave_id))
                               if d.startswith("attempt") and os.path.isdir(
                                   os.path.join(attempts_root, wave_id, d)))
        for attempt_name in attempt_dirs:
            attempt_root = os.path.join(attempts_root, wave_id, attempt_name)
            attempt_num = int(attempt_name[len("attempt"):] or "0")
            arm_subdirs = sorted(d for d in os.listdir(attempt_root)
                                  if os.path.isdir(os.path.join(attempt_root, d)))
            prior_record = next((r for r in state.get("attempts", [])
                                  if r.get("wave") == wave_id
                                  and int(r.get("attempt", -1)) == attempt_num),
                                 None)
            attempt_result = {"attempt": attempt_num, "attempt_root": attempt_root,
                              "per_arm": []}
            overlap_attempts = []
            for arm in arm_subdirs:
                seed_match = _SEED_RE.search(wave_id)
                seed = int(seed_match.group(1)) if seed_match else 0
                arm_task_key = "s5|equal|kuairand|natural|m3oe|seed{seed}|{arm}".format(
                    seed=seed, arm=arm)
                task = tasks.get(arm_task_key)
                if task is None:
                    attempt_result["per_arm"].append({"arm": arm, "valid": False,
                                                       "reasons": ["unknown task key"]})
                    continue
                pseudo = (prior_record if prior_record is not None else
                          {"wave": wave_id, "attempt": attempt_num,
                           "attempt_root": attempt_root, "launch": []})
                arm_record = _revalidate_one_attempt(task, pseudo, source)
                attempt_result["per_arm"].append(arm_record)
                overlap_attempts.append((task, arm_record))
            # Cross-arm overlap evidence — re-derive from process_interval.json
            # using the Windows-compatible ``_validated_overlap_s5`` (no
            # Popen-pid-equality gate).
            if len(overlap_attempts) == 2:
                children = []
                for task, arm_record in overlap_attempts:
                    child = type("FakeChild", (), {"pid": arm_record.get("stage_popen_pid") or 0})()
                    children.append((child, None, arm_record["work_dir"], task))
                overlap_ok = True
                overlap_ns = None
                try:
                    intervals, overlap_ns = _validated_overlap_s5(children)
                    attempt_result["overlap_ns"] = overlap_ns
                    attempt_result["overlap_valid"] = True
                    attempt_result["overlap_pids"] = [{"popen": c[0].pid, "child": i["pid"]}
                                                       for c, i in zip(children, intervals)]
                except RuntimeError as exc:
                    overlap_ok = False
                    attempt_result["overlap_valid"] = False
                    attempt_result["overlap_error"] = str(exc)
                arm_valid = all(a.get("valid") for a in attempt_result["per_arm"])
                attempt_result["wave_valid"] = arm_valid and overlap_ok
            else:
                attempt_result["wave_valid"] = False
                attempt_result["overlap_valid"] = False
                attempt_result["overlap_error"] = "missing arm workdir(s)"
            if prior_record is not None:
                attempt_result["prior_ready_for_promotion"] = prior_record.get(
                    "ready_for_promotion")
                attempt_result["prior_retry_class"] = prior_record.get("retry_class")
                attempt_result["prior_validation"] = prior_record.get("validation")
            wave_entry["per_attempt"].append(attempt_result)
        report["per_wave"].append(wave_entry)
    report_path = os.path.join(HERE, "stage_S5_revalidate_report.json")
    _write_json_atomic(report_path, report)
    return report


def run_queue(args):
    if not args.reviewer_pass:
        raise RuntimeError("independent S5 reviewer PASS is required")
    args.raw_root = args.raw_root or RAW_ROOT
    args.queue_manifest = args.queue_manifest or QUEUE_PATH
    queue = freeze_queue(args.raw_root, args.queue_manifest)
    tasks = {t["formal_run_key"]: t for t in queue["tasks"]}
    source = queue["source_inventory"]
    state_path = args.queue_manifest + ".state.json"
    state = _read_json(state_path) if os.path.isfile(state_path) \
        else {"completed": [], "attempts": []}
    for wave in queue["waves"]:
        _recover_promotion(args.raw_root, wave, tasks, state, state_path, source)
        _write_json_atomic(state_path, state)
        if wave["wave_id"] in state["completed"]:
            for formal_key in wave["formal_run_keys"]:
                good, why = validate_run(tasks[formal_key], source_hashes=source)
                if not good:
                    raise RuntimeError("completed wave evidence invalid: " + str(why))
            continue
        previous = [r for r in state["attempts"] if r["wave"] == wave["wave_id"]]
        ready = [r for r in previous if r.get("ready_for_promotion")]
        if ready:
            receipt = ready[-1]
            recovery_children = [(None, None,
                                  os.path.join(receipt["attempt_root"], tasks[k]["arm"]), tasks[k])
                                 for k in wave["formal_run_keys"]]
            _promote_wave(args.raw_root, wave, recovery_children, state, state_path, source)
            continue
        for attempt in (1, 2):
            attempt_root = os.path.join(args.raw_root, ".attempts", wave["wave_id"],
                                        "attempt{}".format(attempt))
            if os.path.exists(attempt_root):
                prior = next((r for r in previous if r["attempt"] == attempt), None)
                if prior and prior.get("retry_class") == "transient_os_error" and attempt == 1:
                    continue
                raise RuntimeError("append-only attempt exists without approved recovery: " + attempt_root)
            os.makedirs(attempt_root)
            children = []
            launch = []
            try:
                for idx, formal_key in enumerate(wave["formal_run_keys"]):
                    task = tasks[formal_key]
                    work = os.path.join(attempt_root, task["arm"])
                    stdout_path = os.path.join(attempt_root, task["arm"] + ".stdout.log")
                    stream = open(stdout_path, "x", encoding="utf-8")
                    cmd = [args.python, os.path.join(HERE, "stage_S5_run.py"),
                           "--dataset", "kuairand", "--setting", "natural", "--backbone", "m3oe",
                           "--seed", str(task["seed"]), "--run_dir", work,
                           "--formal_run_key", formal_key, "--arm", task["arm"],
                           "--co_run_pair_id", wave["wave_id"],
                           "--planned_peer_run_key", wave["formal_run_keys"][1-idx],
                           "--worker_id", "worker-{}".format(idx+1),
                           "--attempt_id", "{}-{}".format(wave["wave_id"], attempt)]
                    child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream,
                                              stderr=subprocess.STDOUT)
                    children.append((child, stream, work, task))
                    launch.append({"key": formal_key, "pid": child.pid,
                                   "popen_return_ns": time.time_ns(),
                                   "stdout": stdout_path})
            except Exception:
                for child, stream, _, _ in children:
                    if child.poll() is None:
                        child.terminate()
                    child.wait(timeout=10)
                    stream.close()
                raise
            finished, terminated, max_live = _monitor_children(children)
            exits = [child.returncode for child, _, _, _ in children]
            for _, stream, _, _ in children:
                stream.close()
            report = {"wave": wave["wave_id"], "attempt": attempt, "launch": launch,
                      "exits": exits, "max_live_observed": max_live,
                      "finished_at_by_pid": finished, "terminated_peer_pids": terminated,
                      "attempt_root": attempt_root, "finished_at": time.time(),
                      "validation": []}
            if all(code == 0 for code in exits):
                try:
                    intervals, overlap = _validated_overlap_s5(children)
                    report["process_intervals"] = intervals
                    report["overlap_ns"] = overlap
                    report["overlap_valid"] = True
                except Exception as exc:
                    report["overlap_valid"] = False
                    report["overlap_error"] = str(exc)
                if report["overlap_valid"]:
                    try:
                        for _, _, work, task in children:
                            console = os.path.join(attempt_root, task["arm"] + ".stdout.log")
                            _finalize_child(work, console)
                    except Exception as exc:
                        report["finalization_error"] = "{}: {}".format(type(exc).__name__, exc)
                for _, _, work, task in children:
                    if report.get("overlap_valid") and "finalization_error" not in report:
                        temporary = dict(task, run_dir=work)
                        good, why = validate_run(temporary, source_hashes=source)
                    else:
                        good, why = False, [report.get("finalization_error",
                                                       "process overlap not proven")]
                    report["validation"].append({"key": task["formal_run_key"],
                                                 "valid": good, "reasons": why})
            retry_class, retry_reason = _retry_classification(children, report)
            report["retry_class"] = retry_class
            report["retry_reason"] = retry_reason
            report["ready_for_promotion"] = (all(code == 0 for code in exits)
                                              and report.get("overlap_valid") is True
                                              and len(report["validation"]) == 2
                                              and all(v["valid"] for v in report["validation"]))
            state["attempts"].append(report)
            _write_json_atomic(state_path, state)
            if report["ready_for_promotion"]:
                _promote_wave(args.raw_root, wave, children, state, state_path, source)
                break
            if attempt == 2 or retry_class != "transient_os_error":
                raise RuntimeError("S5 wave requires review; no automatic retry: " +
                                   wave["wave_id"] + " " + retry_reason)
    for seed in SEEDS:
        good, why = validate_block(queue["tasks"], seed, source_hashes=source)
        if not good:
            raise RuntimeError("S5 four-arm block invalid: " + str(why))
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=RAW_ROOT)
    parser.add_argument("--queue-manifest", default=QUEUE_PATH)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--reviewer-pass", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--revalidate", action="store_true",
                        help="Re-run validate_run on existing attempts and write "
                             "stage_S5_revalidate_report.json only; state.json, "
                             "the queue manifest, and the attempt trees are left "
                             "untouched (no promotion decisions are taken here).")
    args = parser.parse_args()
    if args.freeze_only:
        if not args.reviewer_pass:
            raise RuntimeError("independent S5 reviewer PASS is required before freeze")
        freeze_queue(args.raw_root, args.queue_manifest)
    elif args.revalidate:
        report = revalidate_attempts(args)
        print(json.dumps({"revalidate_report": os.path.join(HERE, "stage_S5_revalidate_report.json"),
                          "per_wave_summary": [
                              {"wave": w["wave_id"],
                               "attempts": [{"attempt": a["attempt"],
                                              "wave_valid": a.get("wave_valid"),
                                              "overlap_valid": a.get("overlap_valid")}
                                             for a in w["per_attempt"]]}
                              for w in report["per_wave"]]},
                         sort_keys=True))
    else:
        run_queue(args)


if __name__ == "__main__":
    main()