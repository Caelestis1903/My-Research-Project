"""Mechanical S5 collector; scientific interpretation belongs to the main agent.

Mirrors the Stage S4 collector shape while adopting the S5 four-arm matrix
(F, T, HT, DH), the six contrast pairs required by §8 of the design, and the
moment-cosine audit table over the seven moment-dump epochs.
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

from stage_S5_orchestrator import (_read_csv, _read_json, freeze_queue, sha,
                                   validate_block, _write_json_atomic)
from stage_S5_protocol import (ARM_CURRENT, ARM_HISTORY, ARMS, COMMON_EPOCHS,
                               MOMENT_DUMP_EPOCHS, OUTPUT_ROOT, QUEUE_PATH, RAW_ROOT,
                               SEEDS, matrix)

HERE = os.path.dirname(os.path.abspath(__file__))

PAIRS = (("T", "F"), ("HT", "F"), ("DH", "F"),
         ("HT", "T"), ("DH", "T"), ("HT", "DH"))


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty S5 table")
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


def collect(raw_root, output_root, queue_path=QUEUE_PATH):
    queue = freeze_queue(raw_root, queue_path, allow_create=False)
    tasks = queue["tasks"]
    if tasks != matrix(raw_root):
        raise RuntimeError("S5 collector task/root mismatch")
    queue_sha = sha(queue_path)
    for seed in SEEDS:
        valid, reasons = validate_block(tasks, seed)
        if not valid:
            raise RuntimeError("invalid S5 seed {}: {}".format(seed, reasons))
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
            raw.append(dict(row, stage="S5", formal_run_key=task["formal_run_key"],
                            arm=task["arm"], seed=task["seed"]))
        gradient += _read_csv(os.path.join(path, "gradient_step.csv"))[1]
        probes += _read_csv(os.path.join(path, "probe.csv"))[1]
    if len(runs) != 16 or len(raw) != 672 or len(probes) != 1600:
        raise RuntimeError("S5 run/raw/probe cardinality mismatch")
    by_key = {(int(r["seed"]), r["arm"], r["checkpoint"], int(r["domain"]), int(r["task"])): r
              for r in raw}
    if len(by_key) != len(raw):
        raise RuntimeError("duplicate S5 raw key")
    paired = []
    for seed, (left, right), label, domain, task_id in itertools.product(
            SEEDS, PAIRS, ("selected", "final", "common_epoch_9", "common_epoch_19",
                           "common_epoch_29", "common_epoch_39", "common_epoch_49"),
            range(3), range(2)):
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
    if len(paired) != 1008:
        raise RuntimeError("S5 paired cardinality mismatch")
    aggregate = []
    for contrast, label, domain, task_id in itertools.product(
            (a + "-" + b for a, b in PAIRS),
            ("selected", "final", "common_epoch_9", "common_epoch_19",
             "common_epoch_29", "common_epoch_39", "common_epoch_49"), range(3), range(2)):
        rows = [r for r in paired if (r["contrast"], r["checkpoint"], r["domain"], r["task"])
                == (contrast, label, domain, task_id)]
        auc = [r["delta_auc"] for r in rows]
        loss = [r["delta_logloss"] for r in rows]
        aggregate.append({"contrast": contrast, "checkpoint": label, "domain": domain,
                          "task": task_id, "mean_delta_auc": statistics.mean(auc),
                          "sd_delta_auc": statistics.stdev(auc),
                          "positive_seed_count": sum(x > 0 for x in auc),
                          "worst_seed_delta_auc": min(auc),
                          "mean_delta_logloss": statistics.mean(loss),
                          "sd_delta_logloss": statistics.stdev(loss)})
    # Cross-run moment-cosine audit table: per seed, per dump epoch.  Each
    # dump file maps parameter name -> {"exp_avg", "exp_avg_sq", "step"} for
    # shared + architecture parameters (design §7), so the m- and v-stream
    # cosines are computed from the actual persistent Adam moments.
    moment_audit = []
    moment_gate = {}
    for seed in SEEDS:
        moment_gate[seed] = {}
        for dump_epoch in MOMENT_DUMP_EPOCHS:
            states = {}
            for arm in ARMS:
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
                    raise RuntimeError("S5 moment dump schema mismatch: " + name)
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
                row["cos_m_{}_{}".format(left, right)] = _safe_cosine(m_vecs[left], m_vecs[right])
                row["cos_v_{}_{}".format(left, right)] = _safe_cosine(v_vecs[left], v_vecs[right])
            # §9 cross-run moment gate (hard at dump epochs 0 and 1): the
            # history factor must dominate — HT closer to T than to F, and
            # DH closer to F than to T — for both m and v.
            if int(dump_epoch) in (0, 1):
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
        path = os.path.join(output_root, "stage_S5_{}.csv".format(name))
        write_csv(path, rows)
        paths[name] = path
    manifest = {"stage": "S5", "status": "STRUCTURALLY_VALIDATED",
                "cardinalities": {k: len(v) for k, v in outputs.items()}, "paths": paths,
                "four_arm_blocks": 4, "pairing_valid": True, "scientific_interpretation": None,
                "moment_gate": moment_gate,
                "moment_gate_overall_pass": all(
                    gate["pass"] for per_seed in moment_gate.values()
                    for gate in per_seed.values()),
                "queue_manifest": os.path.abspath(queue_path), "queue_sha256": queue_sha,
                "raw_root": os.path.abspath(raw_root), "output_root": os.path.abspath(output_root)}
    _write_json_atomic(os.path.join(output_root, "stage_S5_summary_manifest.json"), manifest)
    report_path = os.path.join(output_root, "reports", "stage_S5_run_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Stage S5 raw execution report\n\n")
        handle.write("Status: 16/16 fresh runs and 4/4 strict four-arm blocks structurally validated.\n\n")
        handle.write("Queue SHA-256: `{}`; raw root: `{}`.\n\n".format(
            queue_sha, os.path.abspath(raw_root)))
        for name, rows in outputs.items():
            handle.write("- `{}`: {} rows\n".format(os.path.basename(paths[name]), len(rows)))
        handle.write("\nNo scientific interpretation is included.\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=RAW_ROOT)
    parser.add_argument("--queue-manifest", default=QUEUE_PATH)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(collect(args.raw_root, args.output_root,
                             args.queue_manifest)["cardinalities"], sort_keys=True))


if __name__ == "__main__":
    main()