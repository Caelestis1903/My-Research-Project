"""Mechanical Stage S6ML collector; scientific interpretation belongs to the main agent.

Mirrors the Stage S5ML collector shape while substituting the MovieLens
``d2_large`` dataset identity, the Sample objective identity (equal → sample),
and the S6ML arm factor graph (bit-identical to S6's).  Phase A materialises
only the F/T replication arm pair, so the report correctly shrinks: 8 runs ×
6 cells × 7 labels = 336 raw rows; the full four-arm collector (after a
Phase B promotion) lifts the same code path back up to 672 raw rows.

All six arm-pair contrasts are still enumerated: in Phase A the contrast
table simply has only the T−F column populated; HT/DH contrasts remain NaN
until Phase B runs.

Phase B unions its own HT/DH queue with the frozen Phase A F/T queue.  Frozen
queues are loaded as historical records (``_load_frozen_phase_queue``): their
pinned source inventory is the record of what the phase executed under and is
not re-validated against the live source tree, so a reviewed plumbing-only
amendment between phases (e.g. the Phase B enablement itself) cannot
retroactively invalidate a completed phase.  Cross-phase code identity is
instead enforced by ``_validate_cross_phase_pairing`` on the
producer-critical inventory subset.
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
from stage_S6ML_orchestrator import (PAIR_FIELDS, _load_frozen_queue,
                                      _read_csv, _read_json,
                                      validate_block, _write_json_atomic)
from stage_S6ML_protocol import (ARM_CURRENT, ARM_HISTORY, ARMS, BACKBONE,
                                 COMMON_EPOCHS, DATASET, MOMENT_DUMP_EPOCHS,
                                 OBJECTIVE_WEIGHTING, OUTPUT_ROOT, PHASE_A_ARMS,
                                 PHASE_B_ARMS, QUEUE_PATH, RAW_ROOT, SEEDS,
                                 SETTING, STAGE, matrix, waves,
                                 queue_path_for_phase)

HERE = os.path.dirname(os.path.abspath(__file__))

# All six contrasts from S5 §8; Phase A will populate only T−F.
PAIRS = (("T", "F"), ("HT", "F"), ("DH", "F"),
         ("HT", "T"), ("DH", "T"), ("HT", "DH"))

# Inventory keys that do not affect the training producer.  Mirrors S5ML.
_PHASE_PLUMBING_KEYS = frozenset({
    "experiments/stage_S6ML_orchestrator.py",
    "experiments/stage_S6ML_report.py",
    "experiments/stage_S6ML_protocol.py",
})


def _producer_code_hashes(manifest):
    return {k: v for k, v in manifest["code_hashes"].items()
            if k not in _PHASE_PLUMBING_KEYS}


def _load_frozen_phase_queue(raw_root, queue_path, phase):
    """Load a frozen phase queue as a historical record."""
    queue = _load_frozen_queue(queue_path)
    if str(queue.get("phase", "")).upper() != str(phase).upper():
        raise RuntimeError("S6ML frozen queue phase mismatch: {} vs {}".format(
            queue.get("phase"), phase))
    expected = matrix(raw_root, phase=phase)
    if queue["tasks"] != expected:
        raise RuntimeError("S6ML collector task/root mismatch")
    return queue


def _validate_cross_phase_pairing(tasks, seed):
    """Pairing audit across the Phase A (F/T) and Phase B (HT/DH) manifests."""
    block = [t for t in tasks if t["seed"] == seed]
    if {t["arm"] for t in block} != set(ARMS):
        raise RuntimeError("S6ML cross-phase pairing requires all four arms")
    manifests = [(_read_json(os.path.join(t["run_dir"], "run_manifest.json")))
                 for t in block]
    left = manifests[0]
    left_producer = _producer_code_hashes(left)
    reasons = []
    for m in manifests[1:]:
        for field in PAIR_FIELDS:
            if m.get(field) != left.get(field):
                reasons.append("cross-phase pairing mismatch:" + field)
        if m.get("batch_order_hash") != left.get("batch_order_hash"):
            reasons.append("cross-phase batch trace mismatch")
        if _producer_code_hashes(m) != left_producer:
            reasons.append("cross-phase producer source mismatch")
    if reasons:
        raise RuntimeError("S6ML seed {} cross-phase pairing invalid: {}".format(
            seed, sorted(set(reasons))))


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty S6ML table")
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


def collect(raw_root, output_root, queue_path=QUEUE_PATH, *, phase="A"):
    queue = _load_frozen_phase_queue(raw_root, queue_path, phase)
    phase_tasks = queue["tasks"]
    queue_sha = sha(queue_path)
    phase_upper = str(phase).upper()
    if phase_upper == "B":
        phase_a_path = queue_path_for_phase("A")
        if not os.path.isfile(phase_a_path):
            raise RuntimeError(
                "S6ML Phase B collector requires the frozen Phase A queue at "
                + phase_a_path + " (Phase A F/T canonical products must exist)")
        phase_a_queue = _load_frozen_phase_queue(raw_root, phase_a_path, "A")
        tasks_by_key = {t["formal_run_key"]: t for t in phase_a_queue["tasks"]}
        for t in phase_tasks:
            if t["formal_run_key"] in tasks_by_key:
                raise RuntimeError("S6ML Phase B task duplicates Phase A key: "
                                   + t["formal_run_key"])
            tasks_by_key[t["formal_run_key"]] = t
        tasks = list(tasks_by_key.values())
    else:
        tasks = list(phase_tasks)
    arms_in_plan = sorted({t["arm"] for t in tasks}, key=lambda a: ARMS.index(a))
    if phase_upper == "B":
        for seed in SEEDS:
            for arm in ARMS:
                if not any(t["seed"] == seed and t["arm"] == arm for t in tasks):
                    raise RuntimeError(
                        "S6ML Phase B collector missing {} seed {} run".format(
                            arm, seed))
        phase_a_tasks = [t for t in tasks if t["arm"] in PHASE_A_ARMS]
        phase_b_tasks = [t for t in tasks if t["arm"] in PHASE_B_ARMS]
        for seed in SEEDS:
            valid, reasons = validate_block(phase_a_tasks, seed)
            if not valid:
                raise RuntimeError("invalid S6ML Phase A seed {}: {}".format(
                    seed, reasons))
            valid, reasons = validate_block(phase_b_tasks, seed)
            if not valid:
                raise RuntimeError("invalid S6ML Phase B seed {}: {}".format(
                    seed, reasons))
            _validate_cross_phase_pairing(tasks, seed)
    else:
        for seed in SEEDS:
            valid, reasons = validate_block(tasks, seed)
            if not valid:
                raise RuntimeError("invalid S6ML seed {}: {}".format(seed, reasons))
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
            raw.append(dict(row, stage=STAGE, formal_run_key=task["formal_run_key"],
                            arm=task["arm"], seed=task["seed"]))
        gradient += _read_csv(os.path.join(path, "gradient_step.csv"))[1]
        probes += _read_csv(os.path.join(path, "probe.csv"))[1]
    expected_runs = len(tasks)
    if len(runs) != expected_runs:
        raise RuntimeError("S6ML run cardinality mismatch: {} != {}".format(
            len(runs), expected_runs))
    expected_raw = expected_runs * 6 * 7
    if len(raw) != expected_raw:
        raise RuntimeError("S6ML raw cardinality mismatch: {} != {}".format(
            len(raw), expected_raw))
    if len(probes) != expected_runs * 100:
        raise RuntimeError("S6ML probe cardinality mismatch: {} != {}".format(
            len(probes), expected_runs * 100))
    by_key = {(int(r["seed"]), r["arm"], r["checkpoint"], int(r["domain"]), int(r["task"])): r
              for r in raw}
    if len(by_key) != len(raw):
        raise RuntimeError("duplicate S6ML raw key")
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
                    raise RuntimeError("S6ML moment dump schema mismatch: " + name)
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
        path = os.path.join(output_root, "stage_S6ML_{}.csv".format(name))
        write_csv(path, rows)
        paths[name] = path
    manifest = {"stage": "S6ML", "status": "STRUCTURALLY_VALIDATED",
                "phase": phase_upper,
                "cardinalities": {k: len(v) for k, v in outputs.items()}, "paths": paths,
                "arms_in_plan": arms_in_plan, "pairing_valid": True,
                "scientific_interpretation": None,
                "moment_gate": moment_gate,
                "moment_gate_overall_pass": bool(
                    all(gate["pass"] for per_seed in moment_gate.values()
                        for gate in per_seed.values())) \
                    if moment_gate else None,
                "queue_manifest": os.path.abspath(queue_path), "queue_sha256": queue_sha,
                "raw_root": os.path.abspath(raw_root), "output_root": os.path.abspath(output_root),
                "dataset": DATASET, "setting": SETTING, "backbone": BACKBONE,
                "objective": OBJECTIVE_WEIGHTING}
    if phase_upper == "B":
        phase_a_path = queue_path_for_phase("A")
        manifest["queue_manifest_phase_a"] = os.path.abspath(phase_a_path)
        manifest["queue_sha256_phase_a"] = sha(phase_a_path)
        manifest["cross_phase_pairing_audit"] = (
            "PAIR_FIELDS + batch_order_hash + producer-critical code-identity "
            "subset (all inventory keys except the three S6ML plumbing files) "
            "verified equal across F/T/HT/DH within every seed")
    _write_json_atomic(os.path.join(output_root, "stage_S6ML_summary_manifest.json"), manifest)
    report_path = os.path.join(output_root, "reports", "stage_S6ML_run_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Stage S6ML raw execution report\n\n")
        handle.write("Status: {}/{} fresh runs and 4/4 strict {} seed blocks "
                     "structurally validated.\n\n".format(
                         len(runs), len(SEEDS) * len(arms_in_plan), "two-arm" if phase_upper == "A" else "four-arm"))
        handle.write("Phase: {} (arms={}).\n".format(phase_upper, ",".join(arms_in_plan)))
        handle.write("Queue SHA-256: `{}`; raw root: `{}`.\n\n".format(
            queue_sha, os.path.abspath(raw_root)))
        for name, rows in outputs.items():
            handle.write("- `{}`: {} rows\n".format(os.path.basename(paths[name]), len(rows)))
        handle.write("\nNo scientific interpretation is included.\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=RAW_ROOT)
    parser.add_argument("--queue-manifest", default=None,
                        help="Default is auto-derived from --phase: "
                             "Phase A -> stage_S6ML_queue_manifest.json; "
                             "Phase B -> stage_S6ML_queue_manifest_phaseB.json.")
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