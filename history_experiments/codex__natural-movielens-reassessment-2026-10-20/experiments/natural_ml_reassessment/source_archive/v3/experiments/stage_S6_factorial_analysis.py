"""Main-agent descriptive analysis of completed S6; never trains a model."""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "stage_S6_analysis_tables"
ARMS = ("F", "T", "HT", "DH")
CHECKPOINTS = ("selected", "common9", "common19", "common29", "common39", "common49", "final")
SEEDS = (2022, 2023, 2024, 2025)
SPECS = (("kuairand", "equal", "S5", 1), ("movielens", "equal", "S5ML", 2),
         ("kuairand", "sample", "S6", 1), ("movielens", "sample", "S6ML", 2))
COEFS = {
    "T-F": {"T": 1, "F": -1}, "HT-F": {"HT": 1, "F": -1},
    "DH-F": {"DH": 1, "F": -1}, "HT-T": {"HT": 1, "T": -1},
    "DH-T": {"DH": 1, "T": -1}, "HT-DH": {"HT": 1, "DH": -1},
    "history_at_full_current": {"HT": 1, "F": -1},
    "history_at_target_current": {"T": 1, "DH": -1},
    "current_at_full_history": {"DH": 1, "F": -1},
    "current_at_target_history": {"T": 1, "HT": -1},
    "history_main": {"HT": .5, "T": .5, "F": -.5, "DH": -.5},
    "current_main": {"DH": .5, "T": .5, "F": -.5, "HT": -.5},
    "interaction": {"T": 1, "HT": -1, "DH": -1, "F": 1},
}


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(name, rows):
    rows = list(rows)
    assert rows, name
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def stats(values, metric=None, delta=False):
    return {"n": len(values), "mean": st.mean(values), "sd": st.stdev(values),
            "min": min(values), "max": max(values),
            "positive": sum(v > 0 for v in values), "negative": sum(v < 0 for v in values),
            "auc_ge_0p003_seeds": sum(v >= .003 for v in values) if metric == "auc" and delta else "",
            "auc_le_minus0p003_seeds": sum(v <= -.003 for v in values) if metric == "auc" and delta else ""}


def main():
    OUT.mkdir(exist_ok=True)
    # Absolute raw values include every seed/cell/checkpoint; add per-run aggregate outcomes.
    outcomes = []
    validation = []
    for dataset, objective, stage, target in SPECS:
        rows = read(ROOT / f"stage_{stage}_outputs" / f"stage_{stage}_raw_results.csv")
        assert len(rows) == 672, (stage, len(rows))
        seen = set()
        blocks = defaultdict(list)
        for row in rows:
            seed, domain, task = (int(row[k]) for k in ("seed", "domain", "task"))
            cp, arm = row["checkpoint"].replace("common_epoch_", "common"), row["arm"]
            identity = (seed, cp, domain, task, arm)
            assert identity not in seen
            seen.add(identity)
            blocks[seed, cp, arm].append(row)
            for metric in ("auc", "logloss"):
                outcomes.append(dict(dataset=dataset, objective=objective, stage=stage,
                    seed=seed, checkpoint=cp, cell=f"d{domain}t{task}", target_cell=domain == target,
                    arm=arm, metric=metric, value=float(row[metric])))
        assert len(seen) == 4 * 7 * 6 * 4
        assert {x[0] for x in seen} == set(SEEDS)
        assert {x[1] for x in seen} == set(CHECKPOINTS)
        assert {x[4] for x in seen} == set(ARMS)
        for (seed, cp, arm), block in blocks.items():
            assert len(block) == 6
            for metric in ("auc", "logloss"):
                values = [float(r[metric]) for r in block]
                per_domain = [st.mean(float(r[metric]) for r in block if int(r["domain"]) == d) for d in range(3)]
                per_task = [st.mean(float(r[metric]) for r in block if int(r["task"]) == t) for t in range(2)]
                worst = min if metric == "auc" else max
                aggregates = {"macro": st.mean(values), "worst_cell": worst(values),
                              "worst_domain": worst(per_domain),
                              "target_mean": st.mean(float(r[metric]) for r in block if int(r["domain"]) == target),
                              "non_target_mean": st.mean(float(r[metric]) for r in block if int(r["domain"]) != target)}
                aggregates.update({f"domain{d}_mean": v for d, v in enumerate(per_domain)})
                aggregates.update({f"task{t}_mean": v for t, v in enumerate(per_task)})
                for cell, value in aggregates.items():
                    outcomes.append(dict(dataset=dataset, objective=objective, stage=stage,
                        seed=seed, checkpoint=cp, cell=cell, target_cell=cell == "target_mean",
                        arm=arm, metric=metric, value=value))
        lookup = {(r["seed"], r["checkpoint"], r["cell"], r["arm"], r["metric"]): r["value"]
                  for r in outcomes if r["stage"] == stage}
        for (seed, cp, cell, arm, metric), value in lookup.items():
            if cp == "final":
                assert value == lookup[seed, "common49", cell, arm, metric]
        paired = read(ROOT / f"stage_{stage}_outputs" / f"stage_{stage}_paired_results.csv")
        assert len(paired) == 1008
        for row in paired:
            cp = row["checkpoint"].replace("common_epoch_", "common")
            seed, cell = int(row["seed"]), f"d{row['domain']}t{row['task']}"
            for metric in ("auc", "logloss"):
                observed = sum(lookup[seed, cp, cell, arm, metric] * coef
                               for arm, coef in COEFS[row["contrast"]].items())
                assert abs(observed - float(row["delta_" + metric])) < 1e-12
        if stage in ("S6", "S6ML"):
            snapshot = ROOT / f"stage_{stage}_outputs_pre_phaseB_20261007" / f"stage_{stage}_raw_results.csv"
            old = read(snapshot)
            old_lookup = {(int(r["seed"]), r["checkpoint"].replace("common_epoch_", "common"), f"d{r['domain']}t{r['task']}", r["arm"], m): float(r[m])
                          for r in old for m in ("auc", "logloss")}
            assert len(old) == 336
            assert all(lookup[k] == v for k, v in old_lookup.items())
            validation.append(dict(stage=stage, rows=len(rows), unique_keys=len(seen),
                phase_a_rows=len(old), phase_a_metrics_unchanged=True, final_equals_common49=True,
                original_paired_contrasts_reproduced=True))
    write("absolute_seed_outcomes.csv", outcomes)
    write("data_integrity.csv", validation)
    absolutes = defaultdict(list)
    cells = defaultdict(dict)
    for row in outcomes:
        absolutes[row["dataset"], row["objective"], row["stage"], row["checkpoint"], row["cell"], row["arm"], row["metric"]].append(row["value"])
        cells[row["dataset"], row["objective"], row["stage"], row["seed"], row["checkpoint"], row["cell"], row["metric"]][row["arm"]] = row["value"]
    absolute_summary = []
    for key, values in absolutes.items():
        base = dict(zip(("dataset", "objective", "stage", "checkpoint", "cell", "arm", "metric"), key))
        absolute_summary.append(base | stats(values, metric=base["metric"]))
    write("absolute_summary.csv", absolute_summary)
    contrasts = []
    for key, values in cells.items():
        assert set(values) == set(ARMS)
        base = dict(zip(("dataset", "objective", "stage", "seed", "checkpoint", "cell", "metric"), key))
        for name, coef in COEFS.items():
            contrasts.append(base | {"contrast": name, "value": sum(values[a] * c for a, c in coef.items())})
    write("factorial_seed_contrasts.csv", contrasts)
    grouped = defaultdict(list)
    for row in contrasts:
        grouped[row["dataset"], row["objective"], row["stage"], row["checkpoint"], row["cell"], row["metric"], row["contrast"]].append(row["value"])
    summary = []
    for key, values in grouped.items():
        base = dict(zip(("dataset", "objective", "stage", "checkpoint", "cell", "metric", "contrast"), key))
        summary.append(base | stats(values, metric=base["metric"], delta=True))
    write("factorial_summary.csv", summary)
    index = {(r["dataset"], r["objective"], r["checkpoint"], r["cell"], r["metric"], r["contrast"]): r for r in summary}
    recovery = []
    for r in summary:
        if r["contrast"] != "T-F" or r["metric"] != "auc":
            continue
        k = (r["dataset"], r["objective"], r["checkpoint"], r["cell"], "auc")
        denom = r["mean"]
        recovery.append({key: r[key] for key in ("dataset", "objective", "stage", "checkpoint", "cell")} |
            dict(denominator_mean_T_F=denom, denominator_positive_seeds=r["positive"],
                 denominator_sd=r["sd"], HR=index[k + ("HT-F",)]["mean"] / denom if denom else "",
                 CR=index[k + ("DH-F",)]["mean"] / denom if denom else "",
                 interpretable_positive_rescue=denom >= .003 and r["positive"] >= 3))
    write("recovery_ratios.csv", recovery)
    objective_contrasts = []
    by_objective = defaultdict(dict)
    for row in contrasts:
        by_objective[row["dataset"], row["seed"], row["checkpoint"], row["cell"], row["metric"], row["contrast"]][row["objective"]] = row["value"]
    for key, vals in by_objective.items():
        assert set(vals) == {"equal", "sample"}
        objective_contrasts.append(dict(zip(("dataset", "seed", "checkpoint", "cell", "metric", "contrast"), key)) |
            dict(equal=vals["equal"], sample=vals["sample"], sample_minus_equal=vals["sample"] - vals["equal"]))
    write("objective_seed_comparison.csv", objective_contrasts)
    og = defaultdict(list)
    for r in objective_contrasts:
        og[r["dataset"], r["checkpoint"], r["cell"], r["metric"], r["contrast"]].append(r)
    o_summary = []
    for key, rows in og.items():
        base = dict(zip(("dataset", "checkpoint", "cell", "metric", "contrast"), key))
        o_summary.append(base | dict(equal_mean=st.mean(r["equal"] for r in rows),
            sample_mean=st.mean(r["sample"] for r in rows)) |
            {"objective_interaction_" + k: v for k, v in stats([r["sample_minus_equal"] for r in rows], metric=base["metric"], delta=True).items()})
    write("objective_comparison_summary.csv", o_summary)
    lock_rows = []
    lock_fields = ("backbone", "dataset", "setting", "seed", "arm", "history_source", "current_source",
        "dataset_file_hash", "initial_parameter_state_hash", "forward_model_source_sha256",
        "split_row_fingerprints", "split_row_id_hashes", "batch_order_hash", "architecture_batch_order_hash",
        "gradient_partition_hash", "train_domain_counts", "domain_shares", "observed_epochs", "fixed_final_epoch",
        "target_local_domain", "target_physical_domain", "target_task_names", "batches_per_epoch",
        "optimizer", "architecture_optimizer", "common_epochs", "moment_dump_epochs", "parameter_count",
        "checkpoint_selection_frozen_before_test", "test_evaluated_during_training")
    for dataset, equal_stage, sample_stage in (("kuairand", "S5", "S6"), ("movielens", "S5ML", "S6ML")):
        for seed in SEEDS:
            for arm in ARMS:
                paths = [ROOT / f"stage_{s}_raw_results" / arm / f"seed{seed}" / "joint" / "run_manifest.json"
                         for s in (equal_stage, sample_stage)]
                a, b = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
                for field in lock_fields:
                    # parameter_count is in the metric/model manifest rather than top-level for some stages.
                    if field not in a and field not in b:
                        continue
                    lock_rows.append(dict(dataset=dataset, seed=seed, arm=arm, field=field,
                        equal_present=field in a, sample_present=field in b,
                        match=field in a and field in b and a[field] == b[field]))
                for field in sorted(set(a["formal_protocol_observed"]) | set(b["formal_protocol_observed"])):
                    if field in ("objective", "stage"):
                        continue
                    lock_rows.append(dict(dataset=dataset, seed=seed, arm=arm, field="formal_protocol_observed." + field,
                        equal_present=field in a["formal_protocol_observed"], sample_present=field in b["formal_protocol_observed"],
                        match=a["formal_protocol_observed"].get(field) == b["formal_protocol_observed"].get(field)))
    write("equal_sample_manifest_locks.csv", lock_rows)
    print("CROSS-OBJECTIVE LOCKS", len(lock_rows), "FAIL", sum(not r["match"] for r in lock_rows))
    # A compact stdout view for the main agent; complete tables remain on disk.
    focus = [r for r in summary if r["objective"] == "sample" and r["metric"] == "auc" and
             r["checkpoint"] in ("selected", "final") and r["cell"] in ("d1t0", "d1t1", "d2t0", "d2t1", "macro", "worst_cell")
             and r["contrast"] in ("T-F", "HT-F", "DH-F", "HT-T", "history_main", "current_main", "interaction")]
    for r in focus:
        if r["cell"].startswith("d") and not r["cell"].startswith("d1" if r["dataset"] == "kuairand" else "d2"):
            continue
        print(r["dataset"], r["checkpoint"], r["cell"], r["contrast"],
              f"{r['mean']:+.6f}", f"sd={r['sd']:.6f}", f"positive={r['positive']}/4")
    print("OUTPUT", OUT)


if __name__ == "__main__":
    main()
