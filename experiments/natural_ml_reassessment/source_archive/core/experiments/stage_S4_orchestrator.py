"""Fail-closed S4 artifact validation and fixed two-process wave launcher."""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from typing import Mapping

import numpy as np

from stage_S1_orchestrator import validate_raw_prediction_metrics
from stage_S3_orchestrator import (_artifact_hash_reasons, _checkpoint_state_numerically_equal,
                                   _coefficient_artifact_reasons, _read_csv, _read_json,
                                   validate_batch_trace)
from stage_S4_gradient_controls import TOLERANCE_ABS, TOLERANCE_REL
from stage_S4_run import FIELDS as STEP_FIELDS
from stage_S4_protocol import (A1_QUEUE_PATH, A1_RAW_ROOT, A1_OUTPUT_ROOT, A2_OUTPUT_ROOT,
                               A1_QUEUE_SHA256, A1_STATE_SHA256,
                               A2_QUEUE_PATH, A2_RAW_ROOT, AMENDMENT, ARMS,
                               OLD_ABORT_REPORT_PATH, OLD_ABORT_REPORT_SHA256,
                               EQUAL_TABLE_SHA256, INITIAL_PARAMETER_SHA256,
                               KUAIRAND_DATA_SHA256, OLD_QUEUE_PATH,
                               OLD_QUEUE_SHA256, OLD_RAW_ROOT, OLD_STATE_SHA256,
                               LABELS, MODEL_SOURCE_KEY, MODEL_SOURCE_SHA256,
                               SEEDS, inventory, inventory_hash, key, matrix, waves)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REQUIRED = ("run_manifest.json", "config.json", "gradient_partition.json", "batch_trace.csv",
            "gradient_step.csv", "probe.csv", "metrics_epoch.csv", "metrics_test.csv",
            "metrics_summary.csv", "predictions_test.npz", "objective_coefficients.json",
            "optimizer_state.pth", "architecture_optimizer_state.pth", "rng_state.pth",
            "initialization_audit.json", "nonfinite_inventory.json", "stdout.log",
            "process_interval.json")
PAIR_FIELDS = ("dataset_file_hash", "split_row_fingerprints", "split_row_id_hashes",
               "physical_view_fingerprints", "preprocessing_metadata", "local_to_physical",
               "train_domain_counts", "domain_shares", "coefficient_table_hash",
               "initial_parameter_state_hash", "gradient_partition_hash", "model_parameter_manifest",
               "batch_order_hash", "architecture_batch_order_hash", "batches_per_epoch",
               "optimizer_steps", "architecture_optimizer_steps", "examples_processed",
               "architecture_examples_processed", "optimizer", "architecture_optimizer",
               "fixed_final_epoch", "common_epochs", "metric_implementation",
               "checkpoint_selection_frozen_before_test", "test_evaluated_during_training",
               "test_metric_in_epoch_history")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json_atomic(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + "." + uuid.uuid4().hex + ".tmp"
    with open(temporary, "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _finalize_child(work, console_path):
    """Retain console bytes in the immutable run and update their byte hashes."""
    manifest_path = os.path.join(work, "run_manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError("successful child omitted run manifest")
    with open(console_path, "rb") as handle:
        console = handle.read()
    if not console:
        raise RuntimeError("successful child console is empty")
    stdout_path = os.path.join(work, "stdout.log")
    with open(stdout_path, "ab") as handle:
        handle.write(b"\n--- S4 process console ---\n")
        handle.write(console)
    m = _read_json(manifest_path)
    m.setdefault("artifact_hashes", {})["stdout.log"] = sha(stdout_path)
    m["artifact_hashes"]["process_interval.json"] = sha(os.path.join(work, "process_interval.json"))
    m["process_console_sha256"] = sha(console_path)
    m["artifacts"] = sorted(set(m.get("artifacts", [])) | {"stdout.log", "process_interval.json"})
    _write_json_atomic(manifest_path, m)


def _validated_overlap(children):
    intervals = []
    for child, _, work, _ in children:
        value = _read_json(os.path.join(work, "process_interval.json"))
        if value.get("status") != "OK" or int(value.get("pid", -1)) != child.pid:
            raise RuntimeError("process interval identity/status mismatch")
        start, finish = int(value["started_ns"]), int(value["finished_ns"])
        if start <= 0 or finish <= start:
            raise RuntimeError("invalid process interval")
        intervals.append({"pid": child.pid, "started_ns": start, "finished_ns": finish})
    overlap = min(x["finished_ns"] for x in intervals) - max(x["started_ns"] for x in intervals)
    if overlap <= 0:
        raise RuntimeError("S4 Python training processes did not actually overlap")
    return intervals, overlap


def _monitor_children(children):
    """Poll both children; stop a live peer as soon as one exits unsuccessfully."""
    finished, terminated = {}, []
    maximum_live = 0
    while len(finished) < len(children):
        live = []
        for child, _, _, task in children:
            code = child.poll()
            if code is None:
                live.append(child)
            elif child.pid not in finished:
                finished[child.pid] = {"returncode": code, "observed_at_ns": time.time_ns()}
                if code != 0:
                    for peer, _, _, _ in children:
                        if peer.pid != child.pid and peer.poll() is None:
                            peer.terminate()
                            terminated.append(peer.pid)
        maximum_live = max(maximum_live, len(live))
        if len(finished) < len(children):
            time.sleep(.5)
    for child, _, _, _ in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)
            terminated.append(child.pid)
    return finished, sorted(set(terminated)), maximum_live


def _retry_classification(children, report):
    """Only documented transient OS errors allow one whole-wave retry."""
    if all(code == 0 for code in report["exits"]):
        return "none", "successful processes; validation failures are not transient"
    failures = []
    for _, _, work, _ in children:
        path = os.path.join(work, "failure.json")
        if os.path.isfile(path):
            failures.append(_read_json(path))
    if len(failures) != 1:
        return "blocked", "no single explicit transient child fault receipt"
    fault = failures[0]
    allowed = {errno.EINTR, errno.EAGAIN, errno.ETIMEDOUT, 10060}
    if fault.get("exception_type") == "OSError" and fault.get("exception_errno") in allowed:
        return "transient_os_error", "reviewed errno {}".format(fault["exception_errno"])
    return "blocked", "fault is not a classified transient execution error"


def _journal_path(raw_root, wave_id):
    return os.path.join(raw_root, ".promotions", wave_id + ".json")


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


def validate_probe_trace(step_rows, probe_rows, task):
    reasons = []
    expected = {(kind, epoch, step) for epoch in range(50)
                for kind, step in (("main", 0), ("architecture", -1))}
    step_map = {}
    for row in step_rows:
        try:
            support = (row["kind"], int(row["epoch"]), int(row["step"]))
        except (KeyError, TypeError, ValueError):
            return ["step support invalid"]
        if support in step_map:
            reasons.append("duplicate step key")
        step_map[support] = row
    observed = set()
    for row in probe_rows:
        try:
            support = (row["kind"], int(row["epoch"]), int(row["step"]))
            if support in observed:
                reasons.append("duplicate probe key")
            observed.add(support)
            if support not in expected or support not in step_map:
                reasons.append("unexpected probe key")
                continue
            source = step_map[support]
            if any(row.get(field, "") != source.get(field, "") for field in STEP_FIELDS):
                reasons.append("probe/step content mismatch")
            if row.get("formal_run_key") != task["formal_run_key"] \
                    or row.get("arm") != task["arm"] or int(row.get("seed", -1)) != task["seed"]:
                reasons.append("probe identity mismatch")
            norms = {name: float(row[name]) for name in
                     ("g_full_norm", "g_target_norm", "delta_full_norm", "delta_target_norm")}
            quantities = {name: float(row[name]) for name in
                          ("target_gain_full", "target_gain_target",
                           "target_gain_equal_norm_full", "delta_N_norm", "r_U")}
            if not all(math.isfinite(v) for v in list(norms.values()) + list(quantities.values())) \
                    or quantities["delta_N_norm"] < 0:
                reasons.append("nonfinite probe quantity")
            for field, norm_a, norm_b in (("gradient_cosine", "g_full_norm", "g_target_norm"),
                                          ("delta_cosine", "delta_full_norm", "delta_target_norm")):
                value = row.get(field, "")
                defined = norms[norm_a] > 0 and norms[norm_b] > 0
                if defined:
                    if value == "" or not math.isfinite(float(value)) or abs(float(value)) > 1.00001:
                        reasons.append("invalid defined probe cosine:" + field)
                elif value not in ("", None):
                    reasons.append("undefined probe cosine was populated:" + field)
            gain_full = quantities["target_gain_full"]
            gain_target = quantities["target_gain_target"]
            gain_equal = quantities["target_gain_equal_norm_full"]
            if abs(gain_equal - quantities["r_U"] * gain_full) > max(
                    1e-6, 1e-4 * abs(gain_equal)):
                reasons.append("probe equal-norm identity mismatch")
            if abs(quantities["delta_N_norm"] - float(row["delta_N_predicted_norm"])) > max(
                    TOLERANCE_ABS, TOLERANCE_REL * float(row["delta_N_predicted_norm"])):
                reasons.append("probe predicted N delta norm mismatch")
            if abs(gain_full) > norms["g_target_norm"] * norms["delta_full_norm"] + 1e-5 \
                    or abs(gain_target) > norms["g_target_norm"] * norms["delta_target_norm"] + 1e-5:
                reasons.append("probe directional Cauchy bound failed")
        except (KeyError, TypeError, ValueError, OverflowError):
            reasons.append("probe numeric/support invalid")
    if observed != expected or len(probe_rows) != 100:
        reasons.append("probe support incomplete")
    return sorted(set(reasons))


def _shared_formula_reference_norm(row, arm):
    if arm == "N":
        return float(row["delta_N_predicted_norm"])
    if arm in ("T", "U"):
        return float(row["delta_target_norm"])
    if arm == "F":
        return float(row["delta_full_norm"])
    raise ValueError("unknown S4 arm")


def validate_run(task, *, source_hashes=None):
    path = os.path.abspath(task["run_dir"])
    reasons = []
    if not all(os.path.isfile(os.path.join(path, f)) for f in REQUIRED):
        return False, ["required artifact missing"]
    try:
        m = _read_json(os.path.join(path, "run_manifest.json"))
        if (m.get("stage"), m.get("status"), m.get("formal_run_key"), m.get("arm"),
                int(m.get("seed", -1)), m.get("setting")) != (
                "S4", "OK", task["formal_run_key"], task["arm"], task["seed"], "natural"):
            reasons.append("identity mismatch")
        if m.get("dataset_key") != "kuairand" or m.get("backbone") != "m3oe" \
                or m.get("objective_weighting") != "equal":
            reasons.append("protocol identity mismatch")
        config = _read_json(os.path.join(path, "config.json"))
        protocol = {"stage": "S4", "dataset": "kuairand", "setting": "natural",
                    "backbone": "m3oe", "objective": "equal", "arm": task["arm"],
                    "seed": task["seed"], "epochs": 50, "batch_size": 40960,
                    "device": "cuda:0", "main_lr": .01, "architecture_lr": .001,
                    "betas": [.9, .999], "weight_decay": 1e-5,
                    "common_epochs": [9, 19, 29, 39, 49]}
        for record in (m, config):
            if record.get("stage") != "S4" or record.get("formal_run_key") != task["formal_run_key"] \
                    or record.get("arm") != task["arm"] or record.get("fresh_role") != "fresh_s4" \
                    or record.get("formal_protocol") != protocol \
                    or record.get("formal_protocol_observed") != protocol \
                    or record.get("formal_protocol_mismatches") != {} \
                    or record.get("formal_valid") is not True:
                reasons.append("S4 formal protocol/config identity mismatch")
                break
        if config.get("code_hashes") != m.get("code_hashes") \
                or config.get("forward_model_source_sha256") != m.get("forward_model_source_sha256"):
            reasons.append("S4 config/source identity mismatch")
        if m.get("dataset_file_hash") != KUAIRAND_DATA_SHA256 \
                or m.get("coefficient_table_hash") != EQUAL_TABLE_SHA256:
            reasons.append("frozen data/objective identity mismatch")
        if m.get("initial_parameter_state_hash") != INITIAL_PARAMETER_SHA256[task["seed"]]:
            reasons.append("semantic initialization mismatch")
        hashes = m.get("code_hashes", {})
        expected = source_hashes or inventory(ROOT)
        if not hashes or any(hashes.get(k) != v for k, v in expected.items()) \
                or hashes.get(MODEL_SOURCE_KEY) != MODEL_SOURCE_SHA256 \
                or m.get("forward_model_source_sha256") != MODEL_SOURCE_SHA256:
            reasons.append("source identity mismatch")
        if m.get("code_inventory_hash") != inventory_hash(hashes):
            reasons.append("source inventory digest mismatch")
        reasons += _artifact_hash_reasons(path, m)
        if not m.get("artifact_hashes", {}).get("stdout.log") \
                or not m.get("artifact_hashes", {}).get("process_interval.json") \
                or os.path.getsize(os.path.join(path, "stdout.log")) == 0:
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
                or list(m.get("common_epochs", [])) != [9, 19, 29, 39, 49]:
            reasons.append("checkpoint support mismatch")
        if m.get("checkpoint_selection_frozen_before_test") is not True \
                or m.get("test_evaluated_during_training") is not False:
            reasons.append("test selection boundary mismatch")
        for label in LABELS:
            filename = ("selected_checkpoint.pth" if label == "selected" else
                        "final_checkpoint.pth" if label == "final" else
                        "common_epoch_{}.pth".format(label[6:]))
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
            reasons.append("S4 step/probe cardinality mismatch")
        if len({(r["kind"], r["epoch"], r["step"]) for r in step_rows}) != len(step_rows):
            reasons.append("duplicate step key")
        reasons += validate_probe_trace(step_rows, probe_rows, task)
        for r in step_rows:
            if r["formal_run_key"] != task["formal_run_key"] or r["arm"] != task["arm"]:
                reasons.append("gradient step identity mismatch")
                break
            numeric = ["L_full", "L_target", "L_non", "decomposition_residual",
                       "g_full_norm", "g_target_norm", "g_non_norm", "r_N", "r_U",
                       "actual_gradient_norm", "delta_full_norm", "delta_target_norm",
                       "delta_N_predicted_norm", "delta_actual_norm", "matching_residual",
                       "private_delta_norm", "decay_norm",
                       "shared_formula_residual", "private_full_residual"]
            if not all(math.isfinite(float(r[f])) for f in numeric):
                reasons.append("nonfinite step")
                break
            ref = float(r["g_target_norm"] if task["arm"] == "N" else
                        r["delta_target_norm"] if task["arm"] in ("T", "U") else
                        r["delta_full_norm"])
            observed = float(r["actual_gradient_norm"] if task["arm"] == "N" else
                             r["delta_actual_norm"])
            if abs(observed - ref) > max(TOLERANCE_ABS, TOLERANCE_REL * ref):
                reasons.append("step norm matching failed")
                break
            shared_reference = _shared_formula_reference_norm(r, task["arm"])
            if float(r["shared_formula_residual"]) > max(
                    TOLERANCE_ABS, TOLERANCE_REL * shared_reference):
                reasons.append("shared formula residual failed")
                break
            if float(r["private_full_residual"]) > max(
                    TOLERANCE_ABS, TOLERANCE_REL * float(r["private_delta_norm"])):
                reasons.append("private FULL residual failed")
                break
            if int(r["adam_state_step_min"]) < 1 or int(r["adam_state_step_min"]) > int(r["adam_state_step_max"]):
                reasons.append("Adam step range invalid")
                break
        if int(m.get("zero_target_main_steps", -1)) > int(m.get("main_steps_seen", 0)) * .01:
            reasons.append("too many target-zero steps")
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
    return not reasons, sorted(set(reasons))


def freeze_queue(raw_root, queue_path):
    tasks = matrix(raw_root)
    plan = waves(tasks)
    if os.path.exists(queue_path):
        queue = _read_json(queue_path)
        if queue["tasks"] != tasks or queue["waves"] != plan \
                or queue["source_inventory"] != inventory(ROOT):
            raise RuntimeError("frozen S4 queue/source differs")
        return queue
    queue = {"stage": "S4", "tasks": tasks, "waves": plan,
             "source_inventory": inventory(ROOT), "frozen_at": time.time(),
             "max_training_processes": 2, "max_attempts_per_wave": 2}
    os.makedirs(os.path.dirname(os.path.abspath(queue_path)), exist_ok=True)
    with open(queue_path, "x", encoding="utf-8") as handle:
        json.dump(queue, handle, sort_keys=True, indent=2)
    return queue


def _attempt_inventory(root):
    if not os.path.isdir(root):
        raise RuntimeError("old attempt tree missing")
    result = {}
    for directory, subdirs, files in os.walk(root):
        subdirs.sort()
        for filename in sorted(files):
            path = os.path.join(directory, filename)
            if os.path.islink(path):
                raise RuntimeError("attempt tree contains symlink")
            result[os.path.relpath(path, root).replace("\\", "/")] = sha(path)
    if not result:
        raise RuntimeError("old attempt tree empty")
    return result


def _a1_predecessor_receipt():
    old_state_path = OLD_QUEUE_PATH + ".state.json"
    if sha(OLD_QUEUE_PATH) != OLD_QUEUE_SHA256 or sha(old_state_path) != OLD_STATE_SHA256:
        raise RuntimeError("aborted predecessor queue/state SHA mismatch")
    if sha(OLD_ABORT_REPORT_PATH) != OLD_ABORT_REPORT_SHA256:
        raise RuntimeError("aborted predecessor run report SHA mismatch")
    old_queue, old_state = _read_json(OLD_QUEUE_PATH), _read_json(old_state_path)
    if old_queue.get("stage") != "S4" or old_state.get("completed") != [] \
            or len(old_state.get("attempts", [])) != 1 \
            or old_state["attempts"][0].get("retry_class") != "blocked":
        raise RuntimeError("aborted predecessor state is not the reviewed 0-wave abort")
    old_tasks = old_queue.get("tasks", [])
    if len(old_tasks) != 16 or any(os.path.exists(t["run_dir"]) for t in old_tasks):
        raise RuntimeError("aborted predecessor has canonical S4 evidence")
    # These are legacy root-level products only; A1 products live under A1_OUTPUT_ROOT.
    for name in ("runs", "raw_results", "paired_results", "gradient_steps", "probe",
                 "aggregates", "summary_manifest"):
        suffix = ".json" if name == "summary_manifest" else ".csv"
        if os.path.exists(os.path.join(HERE, "stage_S4_" + name + suffix)):
            raise RuntimeError("aborted predecessor collector output exists")
    tree_root = os.path.join(OLD_RAW_ROOT, ".attempts")
    listing = _attempt_inventory(tree_root)
    digest = hashlib.sha256(json.dumps(listing, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    return old_queue, {"queue_path": os.path.abspath(OLD_QUEUE_PATH),
                       "queue_sha256": OLD_QUEUE_SHA256,
                       "state_path": os.path.abspath(old_state_path),
                       "state_sha256": OLD_STATE_SHA256,
                       "abort_report_path": os.path.abspath(OLD_ABORT_REPORT_PATH),
                       "abort_report_sha256": OLD_ABORT_REPORT_SHA256,
                       "attempt_root": os.path.abspath(tree_root),
                       "attempt_files": listing, "attempt_inventory_sha256": digest,
                       "canonical_count": 0, "accepted_waves": 0,
                       "status": "ABORTED_NOT_FORMAL_EVIDENCE"}


def _normalized_matrix(tasks):
    return [{k: value for k, value in task.items() if k != "run_dir"} for task in tasks]


def freeze_a1_queue(*, allow_create=True):
    old_queue, predecessor = _a1_predecessor_receipt()
    tasks = matrix(A1_RAW_ROOT)
    plan = waves(tasks)
    if _normalized_matrix(tasks) != _normalized_matrix(old_queue.get("tasks", [])) \
            or plan != old_queue.get("waves"):
        raise RuntimeError("A1 scientific matrix/wave differs from predecessor")
    if not os.path.exists(A1_QUEUE_PATH) and os.path.exists(A1_RAW_ROOT) \
            and os.listdir(A1_RAW_ROOT):
        raise RuntimeError("A1 raw root is not empty before freeze")
    current = inventory(ROOT)
    if any(not value for value in current.values()) or current.get(MODEL_SOURCE_KEY) != MODEL_SOURCE_SHA256:
        raise RuntimeError("A1 current producer source inventory invalid")
    queue = {"stage": "S4", "amendment": AMENDMENT, "tasks": tasks, "waves": plan,
             "raw_root": os.path.abspath(A1_RAW_ROOT), "source_inventory": current,
             "source_inventory_digest": inventory_hash(current),
             "aborted_predecessor": predecessor, "frozen_at": time.time(),
             "max_training_processes": 2, "max_attempts_per_wave": 2}
    if os.path.exists(A1_QUEUE_PATH):
        frozen = _read_json(A1_QUEUE_PATH)
        for field in ("stage", "amendment", "tasks", "waves", "raw_root",
                      "source_inventory", "source_inventory_digest", "aborted_predecessor",
                      "max_training_processes", "max_attempts_per_wave"):
            if frozen.get(field) != queue[field]:
                raise RuntimeError("frozen A1 queue/source/predecessor differs: " + field)
        return frozen
    if not allow_create:
        raise RuntimeError("A1 queue must be frozen after reviewer PASS before RUN")
    with open(A1_QUEUE_PATH, "x", encoding="utf-8") as handle:
        json.dump(queue, handle, sort_keys=True, indent=2)
    return queue


def _a2_predecessor_receipt():
    # The A1 run never produced a canonical member or collector product.  Its
    # failed attempts remain byte-bound evidence, never inputs to A2.
    a1_queue, a1_state_path = _read_json(A1_QUEUE_PATH), A1_QUEUE_PATH + ".state.json"
    if sha(A1_QUEUE_PATH) != A1_QUEUE_SHA256 or sha(a1_state_path) != A1_STATE_SHA256:
        raise RuntimeError("frozen A1 queue/state SHA mismatch")
    if a1_queue.get("aborted_predecessor") != _a1_predecessor_receipt()[1]:
        raise RuntimeError("A1 original abort receipt changed")
    state = _read_json(a1_state_path)
    if state.get("completed") != [] or len(state.get("attempts", [])) != 1 \
            or state["attempts"][0].get("retry_class") != "blocked":
        raise RuntimeError("A1 is not the reviewed blocked zero-wave state")
    if a1_queue.get("tasks") != matrix(A1_RAW_ROOT) \
            or a1_queue.get("waves") != waves(a1_queue["tasks"]):
        raise RuntimeError("A1 scientific queue identity changed")
    if any(os.path.exists(t["run_dir"]) for t in a1_queue["tasks"]):
        raise RuntimeError("A1 canonical run exists")
    if os.path.exists(A1_OUTPUT_ROOT) and os.listdir(A1_OUTPUT_ROOT):
        raise RuntimeError("A1 collector product exists")
    tree_root = os.path.join(A1_RAW_ROOT, ".attempts")
    listing = _attempt_inventory(tree_root)
    digest = hashlib.sha256(json.dumps(listing, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    return a1_queue, {"queue_path": os.path.abspath(A1_QUEUE_PATH),
                      "queue_sha256": A1_QUEUE_SHA256,
                      "state_path": os.path.abspath(a1_state_path),
                      "state_sha256": A1_STATE_SHA256,
                      "attempt_root": os.path.abspath(tree_root),
                      "attempt_files": listing, "attempt_inventory_sha256": digest,
                      "aborted_predecessor": a1_queue["aborted_predecessor"],
                      "canonical_count": 0, "accepted_waves": 0,
                      "status": "ABORTED_NOT_FORMAL_EVIDENCE"}


def freeze_a2_queue(*, allow_create=True):
    a1_queue, predecessor = _a2_predecessor_receipt()
    tasks, plan = matrix(A2_RAW_ROOT), waves(matrix(A2_RAW_ROOT))
    if _normalized_matrix(tasks) != _normalized_matrix(a1_queue["tasks"]) \
            or plan != a1_queue["waves"]:
        raise RuntimeError("A2 scientific matrix/wave differs from A1")
    if not os.path.exists(A2_QUEUE_PATH) and os.path.exists(A2_RAW_ROOT) \
            and os.listdir(A2_RAW_ROOT):
        raise RuntimeError("A2 raw root is not empty before freeze")
    if not os.path.exists(A2_QUEUE_PATH) and os.path.exists(A2_OUTPUT_ROOT) \
            and os.listdir(A2_OUTPUT_ROOT):
        raise RuntimeError("A2 output root is not empty before freeze")
    current = inventory(ROOT)
    if len(current) < 21 or any(not v for v in current.values()) \
            or current.get(MODEL_SOURCE_KEY) != MODEL_SOURCE_SHA256:
        raise RuntimeError("A2 producer source inventory invalid")
    queue = {"stage": "S4", "amendment": "A2", "tasks": tasks, "waves": plan,
             "raw_root": os.path.abspath(A2_RAW_ROOT), "source_inventory": current,
             "source_inventory_digest": inventory_hash(current),
             "aborted_predecessor": predecessor, "frozen_at": time.time(),
             "max_training_processes": 2, "max_attempts_per_wave": 2}
    if os.path.exists(A2_QUEUE_PATH):
        frozen = _read_json(A2_QUEUE_PATH)
        for field in ("stage", "amendment", "tasks", "waves", "raw_root",
                      "source_inventory", "source_inventory_digest", "aborted_predecessor",
                      "max_training_processes", "max_attempts_per_wave"):
            if frozen.get(field) != queue[field]:
                raise RuntimeError("frozen A2 queue/source/predecessor differs: " + field)
        return frozen
    if not allow_create:
        raise RuntimeError("A2 queue must be frozen after reviewer PASS before RUN")
    with open(A2_QUEUE_PATH, "x", encoding="utf-8") as handle:
        json.dump(queue, handle, sort_keys=True, indent=2)
    return queue


def run_queue(args):
    if not args.reviewer_pass:
        raise RuntimeError("independent S4 reviewer PASS is required")
    if getattr(args, "a2", False):
        args.raw_root, args.queue_manifest = A2_RAW_ROOT, A2_QUEUE_PATH
        queue = freeze_a2_queue(allow_create=False)
    elif getattr(args, "a1", False):
        args.raw_root, args.queue_manifest = A1_RAW_ROOT, A1_QUEUE_PATH
        queue = freeze_a1_queue(allow_create=False)
    else:
        queue = freeze_queue(args.raw_root, args.queue_manifest)
    tasks = {t["formal_run_key"]: t for t in queue["tasks"]}
    source = queue["source_inventory"]
    state_path = args.queue_manifest + ".state.json"
    state = _read_json(state_path) if os.path.isfile(state_path) else {"completed": [], "attempts": []}
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
                    cmd = [args.python, os.path.join(HERE, "stage_S4_run.py"),
                           "--dataset", "kuairand", "--setting", "natural", "--backbone", "m3oe",
                           "--seed", str(task["seed"]), "--run_dir", work,
                           "--formal_run_key", formal_key, "--arm", task["arm"],
                           "--co_run_pair_id", wave["wave_id"],
                           "--planned_peer_run_key", wave["formal_run_keys"][1-idx],
                           "--worker_id", "worker-{}".format(idx+1),
                           "--attempt_id", "{}-{}".format(wave["wave_id"], attempt)]
                    child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    children.append((child, stream, work, task))
                    launch.append({"key": formal_key, "pid": child.pid, "popen_return_ns": time.time_ns(),
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
                    intervals, overlap = _validated_overlap(children)
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
                        good, why = False, [report.get("finalization_error", "process overlap not proven")]
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
                raise RuntimeError("S4 wave requires review; no automatic retry: " +
                                   wave["wave_id"] + " " + retry_reason)
    for seed in SEEDS:
        good, why = validate_block(queue["tasks"], seed, source_hashes=source)
        if not good:
            raise RuntimeError("S4 four-arm block invalid: " + str(why))
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=os.path.join(HERE, "stage_S4_raw_results"))
    parser.add_argument("--queue-manifest", default=os.path.join(HERE, "stage_S4_queue_manifest.json"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--reviewer-pass", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--a1", action="store_true",
                        help="use the reviewed isolated S4 A1 queue and raw root")
    parser.add_argument("--a2", action="store_true",
                        help="use the reviewed isolated S4 A2 queue and raw root")
    args = parser.parse_args()
    if args.freeze_only:
        if args.a2:
            if not args.reviewer_pass:
                raise RuntimeError("independent S4 A2 reviewer PASS is required before freeze")
            freeze_a2_queue()
        elif args.a1:
            if not args.reviewer_pass:
                raise RuntimeError("independent S4 A1 reviewer PASS is required before freeze")
            freeze_a1_queue()
        else:
            freeze_queue(args.raw_root, args.queue_manifest)
    else:
        run_queue(args)


if __name__ == "__main__":
    main()
