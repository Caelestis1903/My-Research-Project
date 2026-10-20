"""Mechanical S6 collector; scientific interpretation belongs to the main agent.

Mirrors the Stage S5 collector shape while adopting the S6 four-arm matrix
(F, T, HT, DH), the six contrast pairs required by §8 of the design, the
moment-cosine audit table over the seven moment-dump epochs, the Sample
coefficient table pin, the S6-specific cross-phase pairing checks, and the
``stage_S6_config_lock_audit.csv`` S6-new audit (§6.3 of the design).

Phase A materialises only the F/T replication arm pair, so the report
correctly shrinks: 8 runs × 6 cells × 7 labels = 336 raw rows; the full
four-arm collector (after a Phase B promotion) lifts the same code path
back up to 672 raw rows.

All six arm-pair contrasts are still enumerated: in Phase A the contrast
table simply has only the T−F column populated; HT/DH contrasts remain NaN
until Phase B runs.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import statistics

import torch

from stage_S5_orchestrator import sha
from stage_S6_orchestrator import (_read_csv, _read_json, _write_json_atomic,
                                   freeze_queue, validate_block)
from stage_S6_protocol import (ARM_CURRENT, ARM_HISTORY, ARMS, COMMON_EPOCHS,
                               INITIAL_PARAMETER_SHA256, KUAIRAND_DATA_SHA256,
                               MOMENT_DUMP_EPOCHS, OUTPUT_ROOT, PHASE_A_ARMS,
                               PHASE_B_ARMS, QUEUE_PATH, QUEUE_PATH_PHASE_B,
                               RAW_ROOT, SAMPLE_TABLE_SHA256, SEEDS, STAGE,
                               matrix, queue_path_for_phase)

HERE = os.path.dirname(os.path.abspath(__file__))

# All six contrasts from S5 §8; Phase A will populate only T−F.
PAIRS = (("T", "F"), ("HT", "F"), ("DH", "F"),
         ("HT", "T"), ("DH", "T"), ("HT", "DH"))

# Inventory keys that do not affect the training producer.  The orchestrator
# and report modules never execute inside a training child; the protocol
# module is imported by the child, but every value the child consumes from it
# (arm factor maps, frozen dataset/coefficient/initial-parameter hashes, task
# identity, the run key) is re-pinned per run by ``validate_run`` against the
# live module, so any semantically relevant protocol drift fails per-run
# validation.  Reviewed plumbing-only amendments (such as the Phase B
# enablement) therefore change these three file hashes without changing the
# producer code identity.  Every remaining inventory key — the 21 inherited
# S2/S3/S4/m3oe sources plus ``experiments/stage_S6_run.py`` — must be
# byte-identical across phases for the four arms to share one code identity.
_PHASE_PLUMBING_KEYS = frozenset({
    "experiments/stage_S6_orchestrator.py",
    "experiments/stage_S6_report.py",
    "experiments/stage_S6_protocol.py",
})


def _producer_code_hashes(manifest):
    return {k: v for k, v in manifest["code_hashes"].items()
            if k not in _PHASE_PLUMBING_KEYS}


def _load_frozen_phase_queue(raw_root, queue_path, phase):
    """Load a frozen phase queue as a historical record.

    The queue's ``source_inventory`` is deliberately NOT re-validated against
    the live source tree: a completed phase's queue pins the source it was
    executed under, and reviewed plumbing-only amendments after that phase
    completed must not retroactively invalidate it (fail-closed re-freeze
    gating already protected the phase at execution time).  The task/root
    binding is still re-checked against ``matrix()`` so the collector cannot
    be pointed at a queue that does not match the frozen matrix.
    """
    if not os.path.isfile(queue_path):
        raise RuntimeError("S6 queue manifest not found: " + queue_path)
    queue = _read_json(queue_path)
    if queue.get("stage") != STAGE:
        raise RuntimeError("frozen S6 queue has wrong stage: " + str(queue.get("stage")))
    if str(queue.get("phase", "")).upper() != str(phase).upper():
        raise RuntimeError("S6 frozen queue phase mismatch: {} vs {}".format(
            queue.get("phase"), phase))
    expected = matrix(raw_root, phase=phase)
    if queue["tasks"] != expected:
        raise RuntimeError("S6 collector task/root mismatch")
    return queue


def _validate_cross_phase_pairing(tasks, seed):
    """Pairing audit across the Phase A (F/T) and Phase B (HT/DH) manifests.

    Replaces ``validate_block``'s whole-dict ``code_hashes`` equality, which
    cannot hold across phases once a reviewed plumbing-only amendment lands
    between them: the producer-critical inventory subset (all keys except
    ``_PHASE_PLUMBING_KEYS``) is compared instead, alongside the full
    PAIR_FIELDS set and the batch-trace hash, so data identity, batch
    exposure, initialization, and producer code identity are all still
    enforced across the four arms within the seed.
    """
    block = [t for t in tasks if t["seed"] == seed]
    if {t["arm"] for t in block} != set(ARMS):
        raise RuntimeError("S6 cross-phase pairing requires all four arms")
    manifests = [(_read_json(os.path.join(t["run_dir"], "run_manifest.json")))
                 for t in block]
    left = manifests[0]
    left_producer = _producer_code_hashes(left)
    reasons = []
    for m in manifests[1:]:
        for field in ("dataset_file_hash", "split_row_fingerprints", "split_row_id_hashes",
                      "physical_view_fingerprints", "preprocessing_metadata", "local_to_physical",
                      "train_domain_counts", "domain_shares", "coefficient_table_hash",
                      "initial_parameter_state_hash", "gradient_partition_hash",
                      "model_parameter_manifest",
                      "batch_order_hash", "architecture_batch_order_hash",
                      "batches_per_epoch",
                      "optimizer_steps", "architecture_optimizer_steps",
                      "examples_processed", "architecture_examples_processed",
                      "optimizer", "architecture_optimizer",
                      "fixed_final_epoch", "common_epochs", "metric_implementation",
                      "checkpoint_selection_frozen_before_test",
                      "test_evaluated_during_training",
                      "test_metric_in_epoch_history", "moment_dump_epochs"):
            if m.get(field) != left.get(field):
                reasons.append("cross-phase pairing mismatch:" + field)
        if m.get("batch_order_hash") != left.get("batch_order_hash"):
            reasons.append("cross-phase batch trace mismatch")
        if _producer_code_hashes(m) != left_producer:
            reasons.append("cross-phase producer source mismatch")
    if reasons:
        raise RuntimeError("S6 seed {} cross-phase pairing invalid: {}".format(
            seed, sorted(set(reasons))))


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty S6 table")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a is None or b is None:
        return float("nan")
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    na = float(torch.linalg.vector_norm(a.detach().double()).item())
    nb = float(torch.linalg.vector_norm(b.detach().double()).item())
    if na == 0 or nb == 0:
        return float("nan")
    return float(torch.dot(a.detach().double().flatten(),
                            b.detach().double().flatten()).item() / (na * nb))


def _check_arm_present(by_key, seed, left, right, label, domain, task_id):
    if left == right:
        return False
    return (seed, left, label, domain, task_id) in by_key \
        and (seed, right, label, domain, task_id) in by_key


def _build_config_lock_audit(tasks, raw_root, output_root):
    """S6-specific config-lock audit (design §6.3).

    For each S6 run, diff every Field that is locked in §3.2 of the design
    against the corresponding canonical S5 KuaiRand run for the same
    (dataset, seed, arm).  Any locked field mismatch is reported as a FAIL;
    allowed differences (§3.3) are flagged with ``expected_diff=True``.
    """
    import hashlib as _hashlib
    s5_raw_root = os.path.join(os.path.dirname(raw_root), "stage_S5_raw_results")
    rows = []
    locked_fields = [
        "dataset_file_hash", "split_row_fingerprints", "split_row_id_hashes",
        "initial_parameter_state_hash", "gradient_partition_hash",
        "train_domain_counts", "model_parameter_manifest",
        "batches_per_epoch",
    ]
    locked_field_descriptions = {
        "dataset_file_hash": "KuaiRand data hash",
        "split_row_fingerprints": "split row fingerprints",
        "split_row_id_hashes": "split row id hashes",
        "initial_parameter_state_hash": "initial parameter hash",
        "gradient_partition_hash": "gradient partition hash",
        "train_domain_counts": "train domain counts",
        "model_parameter_manifest": "model parameter manifest",
        "batches_per_epoch": "batches per epoch",
    }
    diff_fields = {
        "stage", "objective_weighting", "objective", "objective_name",
        "objective_formula", "coefficient_table_hash", "fresh_role",
        "provenance_type", "source_stage", "formal_run_key",
        "condition", "shared_access",
    }
    diff_field_descriptions = {
        "stage": "stage label",
        "objective_weighting": "objective weighting",
        "objective": "objective name",
        "objective_name": "objective name (alias)",
        "objective_formula": "objective formula",
        "coefficient_table_hash": "coefficient table hash",
        "fresh_role": "fresh role",
        "provenance_type": "provenance type",
        "source_stage": "source stage label",
        "formal_run_key": "formal run key",
        "condition": "condition name",
        "shared_access": "shared_access (logging-only column)",
    }
    train_hp_fields = [
        ("learning_rate", 0.01), ("learning_rate_darts", 0.001),
        ("weight_decay", 1e-5), ("batch_size", 40960),
        ("num_workers", 0), ("device", "cuda:0"),
        ("shadow_patience", 10), ("expert_num", 4),
        ("fcn_dims", [512, 256, 256, 64]),
    ]
    s6_field_to_s5_value = {
        "stage": "S5", "objective_weighting": "equal",
        "objective": "equal_domain_task_mean_bce",
        "objective_name": "equal_domain_task_mean_bce",
        "objective_formula": "mean_i(w[d_i]*mean_t(BCE(p_it,y_it))); w_d=1/(3*p_d)",
        "fresh_role": "fresh_s5", "provenance_type": "fresh_s5",
        "source_stage": "S5", "condition": "{arm}",
        "shared_access": "{arm}",
    }
    for task in tasks:
        seed = task["seed"]
        arm = task["arm"]
        s6_path = task["run_dir"]
        s6_manifest_path = os.path.join(s6_path, "run_manifest.json")
        s6_config_path = os.path.join(s6_path, "config.json")
        s5_path = os.path.join(s5_raw_root, arm, "seed{}".format(seed), "joint")
        s5_manifest_path = os.path.join(s5_path, "run_manifest.json")
        s5_config_path = os.path.join(s5_path, "config.json")
        s6_manifest_exists = os.path.isfile(s6_manifest_path) and os.path.isfile(s6_config_path)
        s5_manifest_exists = os.path.isfile(s5_manifest_path) and os.path.isfile(s5_config_path)
        if not s6_manifest_exists:
            rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                         "field": "run_manifest.json", "expected": "present",
                         "observed": "missing", "expected_diff": False, "status": "FAIL",
                         "note": "S6 run manifest not found"})
            continue
        s6_manifest = _read_json(s6_manifest_path)
        s6_config = _read_json(s6_config_path)
        s5_manifest = _read_json(s5_manifest_path) if s5_manifest_exists else {}
        s5_config = _read_json(s5_config_path) if s5_manifest_exists else {}
        for field in locked_fields:
            expected = s5_manifest.get(field, "<missing>") if s5_manifest_exists else "<no_s5_anchor>"
            observed = s6_manifest.get(field, "<missing>")
            if expected == "<no_s5_anchor>":
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": locked_field_descriptions.get(field, field),
                             "expected": expected, "observed": observed,
                             "expected_diff": False, "status": "INFO",
                             "note": "no S5 anchor for {} (cross-stage comparison only)".format(field)})
            elif expected != observed:
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": locked_field_descriptions.get(field, field),
                             "expected": str(expected), "observed": str(observed),
                             "expected_diff": False, "status": "FAIL",
                             "note": "locked field mismatch"})
        for name, value in train_hp_fields:
            observed = s6_config.get(name, s6_manifest.get(name, "<missing>"))
            expected = s5_config.get(name, s5_manifest.get(name, "<missing>"))
            if expected == "<missing>":
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": "train_hp:" + name, "expected": "<no_s5_anchor>",
                             "observed": str(observed), "expected_diff": False,
                             "status": "INFO",
                             "note": "no S5 anchor for training hyperparameter {}".format(name)})
            elif observed != expected and not (
                    isinstance(observed, list) and isinstance(expected, list)
                    and list(observed) == list(expected)):
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": "train_hp:" + name, "expected": str(expected),
                             "observed": str(observed), "expected_diff": False,
                             "status": "FAIL",
                             "note": "training hyperparameter mismatch"})
        for field in diff_fields:
            if field == "coefficient_table_hash":
                expected = "30c68c9b7b3a0966d1cdeb1e6dce65647d0459dcdf946c57aa735f3b4762aec7"
                observed = s6_manifest.get(field, "<missing>")
                status = "OK" if observed == expected else "FAIL"
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": diff_field_descriptions.get(field, field),
                             "expected": expected, "observed": str(observed),
                             "expected_diff": True, "status": status,
                             "note": "must transition from S5 EQUAL pin to S6 SAMPLE pin"})
                continue
            if field == "objective_weighting":
                expected = "sample"
                observed = s6_manifest.get(field, "<missing>")
                status = "OK" if observed == expected else "FAIL"
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": diff_field_descriptions.get(field, field),
                             "expected": expected, "observed": str(observed),
                             "expected_diff": True, "status": status,
                             "note": "must transition from equal to sample"})
                continue
            if field == "objective":
                expected = "sample_population_mean_task_bce"
                observed = s6_manifest.get(field, "<missing>")
                status = "OK" if observed == expected else "FAIL"
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": diff_field_descriptions.get(field, field),
                             "expected": expected, "observed": str(observed),
                             "expected_diff": True, "status": status,
                             "note": "must transition from equal objective name to sample"})
                continue
            if field == "objective_name":
                expected = "sample_population_mean_task_bce"
                observed = s6_manifest.get(field, "<missing>")
                status = "OK" if observed == expected else "FAIL"
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": diff_field_descriptions.get(field, field),
                             "expected": expected, "observed": str(observed),
                             "expected_diff": True, "status": status,
                             "note": "must transition from equal objective name to sample"})
                continue
            if field == "objective_formula":
                expected = "mean_i(mean_t(BCE(p_it,y_it)))"
                observed = s6_manifest.get(field, "<missing>")
                status = "OK" if observed == expected else "FAIL"
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": diff_field_descriptions.get(field, field),
                             "expected": expected, "observed": str(observed),
                             "expected_diff": True, "status": status,
                             "note": "must transition from equal formula to sample"})
                continue
            if field == "fresh_role":
                expected = "fresh_s6"
                observed = s6_manifest.get(field, "<missing>")
                status = "OK" if observed == expected else "FAIL"
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": diff_field_descriptions.get(field, field),
                             "expected": expected, "observed": str(observed),
                             "expected_diff": True, "status": status,
                             "note": "must be fresh_s6"})
                continue
            if field == "provenance_type":
                expected = "fresh_s6"
                observed = s6_manifest.get(field, "<missing>")
                status = "OK" if observed == expected else "FAIL"
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": diff_field_descriptions.get(field, field),
                             "expected": expected, "observed": str(observed),
                             "expected_diff": True, "status": status,
                             "note": "must be fresh_s6"})
                continue
            if field == "source_stage":
                expected = "S6"
                observed = s6_manifest.get(field, "<missing>")
                status = "OK" if observed == expected else "FAIL"
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": diff_field_descriptions.get(field, field),
                             "expected": expected, "observed": str(observed),
                             "expected_diff": True, "status": status,
                             "note": "stage label must transition from S5 to S6"})
                continue
            if field == "stage":
                expected = "S6"
                observed = s6_manifest.get(field, "<missing>")
                status = "OK" if observed == expected else "FAIL"
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": diff_field_descriptions.get(field, field),
                             "expected": expected, "observed": str(observed),
                             "expected_diff": True, "status": status,
                             "note": "stage label must transition from S5 to S6"})
                continue
            if field == "formal_run_key":
                expected = task["formal_run_key"]
                observed = s6_manifest.get(field, "<missing>")
                status = "OK" if observed == expected else "FAIL"
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": diff_field_descriptions.get(field, field),
                             "expected": expected, "observed": str(observed),
                             "expected_diff": True, "status": status,
                             "note": "run key prefix changes from s5|equal|… to s6|sample|…"})
                continue
            if field == "condition":
                # The on-disk ``condition`` value is the arm letter itself in
                # both S5 and S6: ``S3Runner.write_outputs`` first stamps
                # ``self._condition_name()`` (``"s6_F"`` etc.) into config.json,
                # then the stage runner's ``write_outputs`` overwrites
                # config.json and run_manifest.json with ``self.arm`` — a write
                # path S6 inherits bit-identically from S5, so the S5 anchor
                # carries the same arm letter.  Not an expected diff.
                expected = arm
                observed = s6_manifest.get(field, "<missing>")
                status = "OK" if observed == expected else "FAIL"
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": diff_field_descriptions.get(field, field),
                             "expected": expected, "observed": str(observed),
                             "expected_diff": False, "status": status,
                             "note": "arm letter, identical to the S5 anchor value "
                                     "(_condition_name prefix is overwritten by write_outputs "
                                     "in both S5 and S6)"})
                continue
            if field == "shared_access":
                expected = arm
                observed = s6_manifest.get(field, "<missing>")
                status = "OK" if observed == expected else "FAIL"
                rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                             "field": diff_field_descriptions.get(field, field),
                             "expected": expected, "observed": str(observed),
                             "expected_diff": True, "status": status,
                             "note": "logging-only column copies the arm name"})
                continue
        if not s5_manifest_exists:
            rows.append({"formal_run_key": task["formal_run_key"], "seed": seed, "arm": arm,
                         "field": "s5_anchor", "expected": "present", "observed": "missing",
                         "expected_diff": False, "status": "INFO",
                         "note": "no S5 anchor; cross-stage comparison skipped"})
    return rows


def collect(raw_root, output_root, queue_path=QUEUE_PATH, *, phase="A"):
    queue = _load_frozen_phase_queue(raw_root, queue_path, phase)
    phase_tasks = queue["tasks"]
    queue_sha = sha(queue_path)
    if str(phase).upper() == "B":
        phase_a_path = queue_path_for_phase("A")
        if not os.path.isfile(phase_a_path):
            raise RuntimeError(
                "S6 Phase B collector requires the frozen Phase A queue at "
                + phase_a_path + " (Phase A F/T canonical products must exist)")
        phase_a_queue = _load_frozen_phase_queue(raw_root, phase_a_path, "A")
        tasks_by_key = {t["formal_run_key"]: t for t in phase_a_queue["tasks"]}
        for t in phase_tasks:
            if t["formal_run_key"] in tasks_by_key:
                raise RuntimeError("S6 Phase B task duplicates Phase A key: "
                                   + t["formal_run_key"])
            tasks_by_key[t["formal_run_key"]] = t
        tasks = list(tasks_by_key.values())
    else:
        tasks = list(phase_tasks)
    arms_in_plan = sorted({t["arm"] for t in tasks}, key=lambda a: ARMS.index(a))
    phase_upper = str(phase).upper()
    if phase_upper == "B":
        for seed in SEEDS:
            for arm in ARMS:
                if not any(t["seed"] == seed and t["arm"] == arm for t in tasks):
                    raise RuntimeError(
                        "S6 Phase B collector missing {} seed {} run".format(
                            arm, seed))
        phase_a_tasks = [t for t in tasks if t["arm"] in PHASE_A_ARMS]
        phase_b_tasks = [t for t in tasks if t["arm"] in PHASE_B_ARMS]
        for seed in SEEDS:
            valid, reasons = validate_block(phase_a_tasks, seed)
            if not valid:
                raise RuntimeError("invalid S6 Phase A seed {}: {}".format(
                    seed, reasons))
            valid, reasons = validate_block(phase_b_tasks, seed)
            if not valid:
                raise RuntimeError("invalid S6 Phase B seed {}: {}".format(
                    seed, reasons))
            _validate_cross_phase_pairing(tasks, seed)
    else:
        for seed in SEEDS:
            valid, reasons = validate_block(tasks, seed)
            if not valid:
                raise RuntimeError("invalid S6 seed {}: {}".format(seed, reasons))
    runs, raw, gradient, probes = [], [], [], []
    for task in tasks:
        path = task["run_dir"]
        m = _read_json(os.path.join(path, "run_manifest.json"))
        runs.append({"formal_run_key": task["formal_run_key"], "seed": task["seed"],
                     "arm": task["arm"], "history_source": ARM_HISTORY[task["arm"]],
                     "current_source": ARM_CURRENT[task["arm"]],
                     "run_dir": path, "selected_epoch": m["selected_epoch"],
                     "main_steps": m["optimizer_steps"],
                     "architecture_steps": m["architecture_optimizer_steps"],
                     "source_inventory_digest": m["code_inventory_hash"],
                     "pairing_valid": True})
        for row in _read_csv(os.path.join(path, "metrics_test.csv"))[1]:
            raw.append(dict(row, stage="S6", formal_run_key=task["formal_run_key"],
                            arm=task["arm"], seed=task["seed"]))
        gradient += _read_csv(os.path.join(path, "gradient_step.csv"))[1]
        probes += _read_csv(os.path.join(path, "probe.csv"))[1]
    expected_runs = len(tasks)
    if len(runs) != expected_runs:
        raise RuntimeError("S6 run cardinality mismatch: {} != {}".format(
            len(runs), expected_runs))
    expected_raw = expected_runs * 6 * 7
    if len(raw) != expected_raw:
        raise RuntimeError("S6 raw cardinality mismatch: {} != {}".format(
            len(raw), expected_raw))
    if len(probes) != expected_runs * 100:
        raise RuntimeError("S6 probe cardinality mismatch: {} != {}".format(
            len(probes), expected_runs * 100))
    by_key = {(int(r["seed"]), r["arm"], r["checkpoint"], int(r["domain"]), int(r["task"])): r
              for r in raw}
    if len(by_key) != len(raw):
        raise RuntimeError("duplicate S6 raw key")
    paired = []
    for seed, (left, right), label, domain, task_id in itertools.product(
            SEEDS, PAIRS, ("selected", "final", "common_epoch_9", "common_epoch_19",
                           "common_epoch_29", "common_epoch_39", "common_epoch_49"),
            range(3), range(2)):
        if not _check_arm_present(by_key, seed, left, right, label, domain, task_id):
            continue
        a = by_key[(seed, left, label, domain, task_id)]
        b = by_key[(seed, right, label, domain, task_id)]
        paired.append({"seed": seed, "contrast": left + "-" + right,
                       "checkpoint": label, "domain": domain, "task": task_id,
                       "physical_domain": a["physical_domain_identity"],
                       "auc_left": a["auc"], "auc_right": b["auc"],
                       "delta_auc": float(a["auc"]) - float(b["auc"]),
                       "logloss_left": a["logloss"], "logloss_right": b["logloss"],
                       "delta_logloss": float(a["logloss"]) - float(b["logloss"]),
                       "left_key": a["formal_run_key"], "right_key": b["formal_run_key"],
                       "pairing_valid": True})
    aggregate = []
    for contrast, label, domain, task_id in itertools.product(
        (a + "-" + b for a, b in PAIRS),
        ("selected", "final", "common_epoch_9", "common_epoch_19",
         "common_epoch_29", "common_epoch_39", "common_epoch_49"), range(3), range(2)):
        rows = [r for r in paired if (r["contrast"], r["checkpoint"], r["domain"], r["task"])
                == (contrast, label, domain, task_id)]
        if not rows:
            continue
        auc = [r["delta_auc"] for r in rows]
        loss = [r["delta_logloss"] for r in rows]
        aggregate.append({"contrast": contrast, "checkpoint": label, "domain": domain,
                          "task": task_id, "mean_delta_auc": statistics.mean(auc),
                          "sd_delta_auc": statistics.stdev(auc),
                          "positive_seed_count": sum(x > 0 for x in auc),
                          "worst_seed_delta_auc": min(auc),
                          "mean_delta_logloss": statistics.mean(loss),
                          "sd_delta_logloss": statistics.stdev(loss)})
    moment_audit = []
    moment_gate = {}
    for seed in SEEDS:
        moment_gate[seed] = {}
        for dump_epoch in MOMENT_DUMP_EPOCHS:
            states = {}
            for arm in arms_in_plan:
                task = next(t for t in tasks if t["seed"] == seed and t["arm"] == arm)
                path = os.path.join(task["run_dir"],
                                    "moment_state_epoch{}.pth".format(dump_epoch))
                states[arm] = torch.load(path, map_location="cpu")
            m_vecs, v_vecs = {}, {}
            row = {"seed": seed, "dump_epoch": int(dump_epoch)}
            for name, state in states.items():
                if not state or not all(
                        {"exp_avg", "exp_avg_sq", "step"} <= set(state[key])
                        for key in state):
                    raise RuntimeError("S6 moment dump schema mismatch: " + name)
                m_vecs[name] = torch.cat([state[key]["exp_avg"].detach().cpu()
                                          .double().flatten() for key in sorted(state)])
                v_vecs[name] = torch.cat([state[key]["exp_avg_sq"].detach().cpu()
                                          .double().flatten() for key in sorted(state)])
                row["m_norm_" + name] = float(torch.linalg.vector_norm(m_vecs[name]).item())
                row["v_norm_" + name] = float(torch.linalg.vector_norm(v_vecs[name]).item())
                row["adam_step_min_" + name] = min(int(state[key]["step"]) for key in state)
                row["adam_step_max_" + name] = max(int(state[key]["step"]) for key in state)
            for pair in PAIRS:
                left, right = pair
                if left not in states or right not in states:
                    row["cos_m_{}_{}".format(left, right)] = float("nan")
                    row["cos_v_{}_{}".format(left, right)] = float("nan")
                    continue
                row["cos_m_{}_{}".format(left, right)] = _safe_cosine(m_vecs[left], m_vecs[right])
                row["cos_v_{}_{}".format(left, right)] = _safe_cosine(v_vecs[left], v_vecs[right])
            if int(dump_epoch) in (0, 1) and len(arms_in_plan) == len(ARMS):
                gate = {
                    "m_pass": bool(row["cos_m_HT_T"] > row["cos_m_HT_F"]
                                   and row["cos_m_DH_F"] > row["cos_m_DH_T"]),
                    "v_pass": bool(row["cos_v_HT_T"] > row["cos_v_HT_F"]
                                   and row["cos_v_DH_F"] > row["cos_v_DH_T"]),
                }
                gate["pass"] = gate["m_pass"] and gate["v_pass"]
                row["moment_gate_m_pass"] = gate["m_pass"]
                row["moment_gate_v_pass"] = gate["v_pass"]
                moment_gate[seed][int(dump_epoch)] = gate
            moment_audit.append(row)
    os.makedirs(output_root, exist_ok=True)
    outputs = {"runs": runs, "raw_results": raw, "paired_results": paired,
               "gradient_steps": gradient, "probe": probes, "aggregates": aggregate,
               "moment_audit": moment_audit}
    paths = {}
    for name, rows in outputs.items():
        path = os.path.join(output_root, "stage_S6_{}.csv".format(name))
        write_csv(path, rows)
        paths[name] = path
    config_lock_rows = _build_config_lock_audit(tasks, raw_root, output_root)
    config_lock_path = os.path.join(output_root, "stage_S6_config_lock_audit.csv")
    write_csv(config_lock_path, config_lock_rows)
    config_lock_failures = [r for r in config_lock_rows if r["status"] == "FAIL"]
    manifest = {"stage": "S6", "status": "STRUCTURALLY_VALIDATED",
                "phase": phase_upper,
                "cardinalities": {k: len(v) for k, v in outputs.items()}, "paths": paths,
                "config_lock_audit": config_lock_path,
                "config_lock_failures": len(config_lock_failures),
                "config_lock_passed": not config_lock_failures,
                "arms_in_plan": arms_in_plan, "pairing_valid": True,
                "scientific_interpretation": None,
                "moment_gate": moment_gate,
                "moment_gate_overall_pass": bool(
                    all(gate["pass"] for per_seed in moment_gate.values()
                        for gate in per_seed.values())) \
                    if moment_gate else None,
                "queue_manifest": os.path.abspath(queue_path), "queue_sha256": queue_sha,
                "raw_root": os.path.abspath(raw_root), "output_root": os.path.abspath(output_root),
                "dataset": "kuairand", "setting": "natural", "backbone": "m3oe",
                "objective": "sample"}
    if phase_upper == "B":
        phase_a_path = queue_path_for_phase("A")
        manifest["queue_manifest_phase_a"] = os.path.abspath(phase_a_path)
        manifest["queue_sha256_phase_a"] = sha(phase_a_path)
        manifest["cross_phase_pairing_audit"] = (
            "PAIR_FIELDS + batch_order_hash + producer-critical code-identity "
            "subset (all inventory keys except the three S6 plumbing files) "
            "verified equal across F/T/HT/DH within every seed")
    _write_json_atomic(os.path.join(output_root, "stage_S6_summary_manifest.json"), manifest)
    report_path = os.path.join(output_root, "reports", "stage_S6_run_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Stage S6 raw execution report\n\n")
        handle.write("Status: {}/{} fresh runs and 4/4 strict {} seed blocks "
                     "structurally validated.\n\n".format(
                         len(runs), len(SEEDS) * len(arms_in_plan), "two-arm" if phase_upper == "A" else "four-arm"))
        handle.write("Phase: {} (arms={}).\n".format(phase_upper, ",".join(arms_in_plan)))
        handle.write("Queue SHA-256: `{}`; raw root: `{}`.\n\n".format(
            queue_sha, os.path.abspath(raw_root)))
        handle.write("Config-lock audit: `{}` (FAIL count: {})\n\n".format(
            os.path.basename(config_lock_path), len(config_lock_failures)))
        for name, rows in outputs.items():
            handle.write("- `{}`: {} rows\n".format(os.path.basename(paths[name]), len(rows)))
        handle.write("\nNo scientific interpretation is included.\n")
    if config_lock_failures:
        # Design §6.3: any locked-field FAIL blocks the collection.  All
        # evidence artifacts (the audit CSV, the output tables, the summary
        # manifest with ``config_lock_passed=false``) are already on disk;
        # raising here makes the FAIL fail-closed instead of advisory.
        raise RuntimeError(
            "S6 config-lock audit FAIL: {} locked-field mismatch row(s); "
            "see {}".format(len(config_lock_failures), config_lock_path))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=RAW_ROOT)
    parser.add_argument("--queue-manifest", default=None,
                        help="Default is auto-derived from --phase: "
                             "Phase A -> stage_S6_queue_manifest.json; "
                             "Phase B -> stage_S6_queue_manifest_phaseB.json.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--phase", choices=("A", "B"), default="A")
    args = parser.parse_args()
    if not args.queue_manifest:
        args.queue_manifest = queue_path_for_phase(args.phase)
    print(json.dumps(collect(args.raw_root, args.output_root,
                             args.queue_manifest, phase=args.phase)["cardinalities"],
                     sort_keys=True))


if __name__ == "__main__":
    main()