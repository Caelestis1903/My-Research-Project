"""Mechanical S4 collector; scientific interpretation belongs to the main agent."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import statistics

from stage_S4_orchestrator import (_read_csv, _read_json, freeze_a1_queue, freeze_a2_queue, sha,
                                   validate_block, validate_run)
from stage_S4_protocol import (A1_OUTPUT_ROOT, A1_QUEUE_PATH, A1_RAW_ROOT,
                               A2_OUTPUT_ROOT, A2_QUEUE_PATH, A2_RAW_ROOT,
                               ARMS, LABELS, SEEDS, matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = (("T", "F"), ("N", "F"), ("U", "F"),
         ("T", "N"), ("T", "U"), ("N", "U"))


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty S4 table")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(raw_root, output_root, queue_path=A2_QUEUE_PATH):
    actual = tuple(map(os.path.abspath, (raw_root, queue_path, output_root)))
    a2 = tuple(map(os.path.abspath, (A2_RAW_ROOT, A2_QUEUE_PATH, A2_OUTPUT_ROOT)))
    a1 = tuple(map(os.path.abspath, (A1_RAW_ROOT, A1_QUEUE_PATH, A1_OUTPUT_ROOT)))
    if actual == a2:
        queue = freeze_a2_queue(allow_create=False)
    elif actual == a1:
        queue = freeze_a1_queue(allow_create=False)
    else:
        raise RuntimeError("S4 collector requires an isolated matched queue, raw and output root")
    amendment = queue["amendment"]
    tasks = queue["tasks"]
    if tasks != matrix(raw_root):
        raise RuntimeError("S4 collector task/root mismatch")
    queue_sha = sha(queue_path)
    for seed in SEEDS:
        valid, reasons = validate_block(tasks, seed)
        if not valid:
            raise RuntimeError("invalid S4 seed {}: {}".format(seed, reasons))
    runs, raw, gradient, probes = [], [], [], []
    for task in tasks:
        path = task["run_dir"]
        m = _read_json(os.path.join(path, "run_manifest.json"))
        runs.append({"formal_run_key": task["formal_run_key"], "seed": task["seed"],
                     "arm": task["arm"], "run_dir": path, "selected_epoch": m["selected_epoch"],
                     "main_steps": m["optimizer_steps"],
                     "architecture_steps": m["architecture_optimizer_steps"],
                     "source_inventory_digest": m["code_inventory_hash"],
                     "pairing_valid": True})
        for row in _read_csv(os.path.join(path, "metrics_test.csv"))[1]:
            raw.append(dict(row, stage="S4", formal_run_key=task["formal_run_key"],
                            arm=task["arm"], seed=task["seed"]))
        gradient += _read_csv(os.path.join(path, "gradient_step.csv"))[1]
        probes += _read_csv(os.path.join(path, "probe.csv"))[1]
    if len(runs) != 16 or len(raw) != 672 or len(probes) != 1600:
        raise RuntimeError("S4 run/raw/probe cardinality mismatch")
    by_key = {(int(r["seed"]), r["arm"], r["checkpoint"], int(r["domain"]), int(r["task"])): r
              for r in raw}
    if len(by_key) != len(raw):
        raise RuntimeError("duplicate S4 raw key")
    paired = []
    for seed, (left, right), label, domain, task_id in itertools.product(
            SEEDS, PAIRS, ("selected", "final", "common_epoch_9", "common_epoch_19",
                           "common_epoch_29", "common_epoch_39", "common_epoch_49"),
            range(3), range(2)):
        a, b = by_key[(seed, left, label, domain, task_id)], by_key[(seed, right, label, domain, task_id)]
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
        raise RuntimeError("S4 paired cardinality mismatch")
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
    os.makedirs(output_root, exist_ok=True)
    outputs = {"runs": runs, "raw_results": raw, "paired_results": paired,
               "gradient_steps": gradient, "probe": probes, "aggregates": aggregate}
    paths = {}
    for name, rows in outputs.items():
        path = os.path.join(output_root, "stage_S4_{}.csv".format(name))
        write_csv(path, rows)
        paths[name] = path
    manifest = {"stage": "S4", "amendment": amendment, "status": "STRUCTURALLY_VALIDATED",
                "cardinalities": {k: len(v) for k, v in outputs.items()}, "paths": paths,
                "four_arm_blocks": 4, "pairing_valid": True, "scientific_interpretation": None,
                "queue_manifest": os.path.abspath(queue_path), "queue_sha256": queue_sha,
                "raw_root": os.path.abspath(raw_root),
                "aborted_predecessor_sha256": queue["aborted_predecessor"]["queue_sha256"],
                "aborted_attempts_excluded": True}
    with open(os.path.join(output_root, "stage_S4_summary_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    report_path = os.path.join(output_root, "reports", "stage_S4_{}_run_report.md".format(amendment))
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Stage S4 raw execution report\n\n")
        handle.write("Status: 16/16 fresh runs and 4/4 strict four-arm blocks structurally validated.\n\n")
        handle.write("{} queue SHA-256: `{}`; raw root: `{}`.\n\n".format(
            amendment, queue_sha, os.path.abspath(raw_root)))
        handle.write("Original and predecessor aborted queues and attempts were excluded.\n\n")
        for name, rows in outputs.items():
            handle.write("- `{}`: {} rows\n".format(os.path.basename(paths[name]), len(rows)))
        handle.write("\nNo scientific interpretation is included.\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=A2_RAW_ROOT)
    parser.add_argument("--queue-manifest", default=A2_QUEUE_PATH)
    parser.add_argument("--output-root", default=A2_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(collect(args.raw_root, args.output_root,
                             args.queue_manifest)["cardinalities"], sort_keys=True))


if __name__ == "__main__":
    main()
