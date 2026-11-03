#!/usr/bin/env python3
"""Compile the frozen overnight matrix; no training or external service access.

Usage: python plan.py self-test
       python plan.py materialize --out /tmp/mdmt-overnight-ab-plan
This is a plan compiler, not an implementation of the GPU runner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SEEDS = (2022, 2023, 2024, 2025, 2026)
DATASETS = ("movielens", "kuairand")
CP = "C_MIX_P01_SERIAL_V1"
CG = "C_MIX_GAUSSIAN_V1"
CELLS = tuple(f"d{d}t{t}" for d in range(3) for t in range(2))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate(p: dict[str, Any]) -> None:
    require(tuple(p["seeds"]) == SEEDS, "Frozen seeds must be 2022--2026")
    require(tuple(p["datasets"]) == DATASETS, "Unexpected dataset order or identity")
    require(p["protocol_id"] == "AUC_SELECTED_V1", "Wrong selection protocol")
    require(p["data_protocol"] == "FULLTRAIN_NATURAL80_10_10", "Wrong population")
    s, d = p["selection"], p["decision"]
    require(tuple(s["cells"]) == CELLS and s["metric"] == "macro_auc", "Six-cell AUC required")
    require(s["mode"] == "strict_max_earliest_tie", "Wrong tie rule")
    require((s["first_epoch"], s["last_epoch"]) == (1, 50), "Wrong selection window")
    require(not s["use_test_for_selection"] and not s["use_logloss_for_selection"], "Selection leak")
    require(s["exclude_search"] and s["exclude_initialization"], "Invalid candidate stage")
    require(d["minimum_paired_seeds"] == 5 and d["macro_auc_gain"] == .003 and
            d["minimum_cell_auc_delta"] == -.001, "Decision thresholds changed")
    require(not d["require_all_seeds_positive"] and not d["ci_is_gate"] and
            not d["validation_test_agreement_is_gate"], "Unauthorized performance gate")
    require(p["datasets"]["movielens"]["rho"] == .9828205985452511 and
            p["datasets"]["kuairand"]["rho"] == .995, "Wrong fixed EMA")
    require(p["m3oe"]["epochs"] == 50 and p["hypergate"]["epochs"] == 50 and
            (p["autoifs"]["search_epochs"], p["autoifs"]["reuse_epochs"]) == (10, 50),
            "Training budget changed")
    require(p["components"]["c_mix"]["kappa"] == 1 and
            p["components"]["p01"]["lambda_pair"] == .05, "Component definition changed")
    require(not p["common"]["early_stopping"] and not p["common"]["hyperparameter_search"],
            "No stopping/search allowed")
    require(p["common"]["checkpoint_storage"] == "RAM_ONLY", "Storage policy changed")
    require(not p["fallback_activation"]["performance_trigger_allowed"] and
            not p["scheduler"]["performance_gates"], "Performance-based routing forbidden")
    expected = {"A0", "HYPERGATE", "AUTOIFS", "C_MIX", "P01", CP, "GAUSSIAN_A1", CG}
    require(set(p["arms"]) == expected, "Unexpected or missing arm")
    require(p["factorials"] == {
        "CP": {"00": "A0", "10": "C_MIX", "01": "P01", "11": CP},
        "CG": {"00": "A0", "10": "C_MIX", "01": "GAUSSIAN_A1", "11": CG}},
        "Factorial cells changed")


def job_id(dataset: str, arm: str, seed: int) -> str:
    return f"{dataset}.{arm}.s{seed}"


def compile_plan(p: dict[str, Any]) -> dict[str, Any]:
    validate(p)
    jobs, published, factorial, queue = [], [], [], []
    for dataset in DATASETS:
        for arm, cfg in p["arms"].items():
            for seed in SEEDS:
                jobs.append(dict(job_id=job_id(dataset, arm, seed), dataset=dataset,
                    arm=arm, seed=seed, lane=cfg["lane"], core=cfg["core"],
                    default_action=cfg["default_action"], estimator=cfg["estimator"],
                    fixed_ema_rho=p["datasets"][dataset]["rho"] if cfg["estimator"] != "online" else "",
                    final_epochs=50, search_epochs=10 if arm == "AUTOIFS" else 0))
        for method, arm in (("M3OE_FIXED1_ADAPTED", "A0"), ("HYPERGATE", "HYPERGATE"),
                            ("AUTOIFS", "AUTOIFS")):
            for seed in SEEDS:
                published.append(dict(dataset=dataset, method=method, seed=seed,
                    job_id=job_id(dataset, arm, seed), estimator="online", endpoint="selected",
                    role="PUBLISHED_BASELINE"))
        for name, cells in p["factorials"].items():
            for cell, arm in cells.items():
                for seed in SEEDS:
                    factorial.append(dict(dataset=dataset, factorial=name, cell=cell,
                        arm=arm, seed=seed, job_id=job_id(dataset, arm, seed),
                        estimator="fixed_ema", endpoint="selected", default_enabled=name == "CP"))
    # Job priority only: the real scheduler alternates A/B at epoch boundaries.
    for seed in SEEDS:
        for dataset, arm in (("kuairand", "HYPERGATE"), ("kuairand", CP),
                             ("movielens", "HYPERGATE"), ("movielens", CP),
                             ("kuairand", "AUTOIFS"), ("movielens", "AUTOIFS")):
            queue.append(job_id(dataset, arm, seed))
    core = [j for j in jobs if j["core"]]
    new = [j for j in core if j["default_action"] == "RUN_NEW_AFTER_AUDIT"]
    require(len(jobs) == 80 and len({j["job_id"] for j in jobs}) == 80, "Matrix duplicate")
    require(len(core) == 60 and len(new) == 30, "Wrong core/new count")
    require(len(published) == 30 and len(factorial) == 80, "Wrong endpoint slot count")
    require(set(queue) == {j["job_id"] for j in new}, "Queue does not cover new core jobs")
    return dict(plan_sha256=canonical_hash(p), jobs=jobs, published=published,
                factorial=factorial, core_job_priority=queue,
                counts=dict(all_unique_trajectories=80, core_unique_trajectories=60,
                            expected_new_core=30, published_slots=30,
                            primary_factorial_slots=40, fallback_factorial_slots=40))


def selected_epoch(records: list[dict[str, Any]], stage: str = "train") -> int:
    eligible = [r for r in records if r["stage"] == stage and 1 <= r["epoch"] <= 50]
    require(bool(eligible), "No eligible validation record")
    seen: set[int] = set()
    best, epoch = -math.inf, -1
    for r in sorted(eligible, key=lambda x: x["epoch"]):
        require(r["epoch"] not in seen, "Duplicate epoch record")
        seen.add(r["epoch"])
        require(len(r["cell_auc"]) == 6 and all(v is not None and math.isfinite(v)
                for v in r["cell_auc"]), "Undefined validation cell")
        score = math.fsum(r["cell_auc"]) / 6
        if score > best:
            best, epoch = score, r["epoch"]
    return epoch


def decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(len({r["seed"] for r in rows}) == len(rows), "Duplicate seed")
    require(set(r["seed"] for r in rows) <= set(SEEDS), "Unregistered seed")
    if len(rows) < 5:
        return dict(status="INCOMPLETE", n=len(rows))
    for r in rows:
        require(len(r["cell_delta"]) == 6 and all(math.isfinite(x) for x in
            [r["macro_delta"], r["ll_delta"], *r["cell_delta"]]), "Nonfinite endpoint")
    delta = math.fsum(r["macro_delta"] for r in rows) / 5
    cells = [math.fsum(r["cell_delta"][i] for r in rows) / 5 for i in range(6)]
    ll = math.fsum(r["ll_delta"] for r in rows) / 5
    macro_pass = delta >= .003
    guards = [x >= -.001 for x in cells]
    status = ("AUC_PASS_WITH_CELL_PROTECTION" if macro_pass and all(guards) else
              "AUC_MACRO_PASS_CELL_PROTECTION_FAIL" if macro_pass else "AUC_MACRO_BELOW_THRESHOLD")
    return dict(status=status, n=5, macro_auc_delta=delta, cell_auc_deltas=cells,
                macro_auc_pass=macro_pass, cell_protection_pass=all(guards),
                cell_pass=guards, logloss_delta=ll, logloss_improved=ll < 0)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                         delete=False) as f:
            name = f.name
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), "Refuse to write an empty matrix")
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, buf.getvalue())


def materialize(p: dict[str, Any], out: Path) -> dict[str, Any]:
    x = compile_plan(p)
    receipt = out / "PLAN_COMPILE_RECEIPT.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        require(old["plan_sha256"] == x["plan_sha256"], "Output belongs to a different frozen plan")
    write_csv(out / "TRAINING_MATRIX.csv", x["jobs"])
    write_csv(out / "A_ENDPOINT_MATRIX.csv", x["published"])
    write_csv(out / "B_FACTORIAL_MATRIX.csv", x["factorial"])
    atomic_write(out / "PLAN_EXPANDED.json", json.dumps(x, ensure_ascii=False, indent=2) + "\n")
    atomic_write(out / "CORE_QUEUE.json", json.dumps(dict(kind="job_priority_not_full_run_serial_order",
        epoch_fairness_required=True, job_ids=x["core_job_priority"]), indent=2) + "\n")
    metadata = dict(kind="PLAN_ONLY_NOT_TRAINING_RESULTS", plan_sha256=x["plan_sha256"], counts=x["counts"])
    atomic_write(receipt, json.dumps(metadata, indent=2) + "\n")
    return metadata


def self_test(p: dict[str, Any]) -> dict[str, Any]:
    x = compile_plan(p)
    require(Counter(r["factorial"] for r in x["factorial"]) == {"CP": 40, "CG": 40}, "Factorial incomplete")
    def row(seed: int, value: float, ll: float = .01) -> dict[str, Any]:
        return dict(seed=seed, macro_delta=value, cell_delta=[value]*6, ll_delta=ll)
    require(decision([row(s, .003) for s in SEEDS])["macro_auc_pass"], "Inclusive macro threshold failed")
    negative = decision([row(s, v) for s, v in zip(SEEDS, [-.001, .004, .004, .004, .004])])
    require(negative["macro_auc_pass"] and not negative["logloss_improved"], "Negative seed or LL became a gate")
    boundary = [row(s, .003) for s in SEEDS]
    for r in boundary:
        r["cell_delta"] = [-.001] + [.0038]*5
    require(decision(boundary)["cell_protection_pass"], "Inclusive cell threshold failed")
    for r in boundary:
        r["cell_delta"][0] = -.00101
    require(not decision(boundary)["cell_protection_pass"] and decision(boundary)["macro_auc_pass"], "Cell flag coupled to macro")
    require(decision([row(s, 0, -.001) for s in SEEDS])["logloss_improved"], "LL independence failed")
    require(decision([row(s, 1) for s in SEEDS[:4]])["status"] == "INCOMPLETE", "Four-seed verdict forbidden")
    try:
        decision([row(2022, .003), row(2022, .003)])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate seed was accepted")
    records = [dict(stage="search", epoch=1, cell_auc=[.99]*6),
               dict(stage="reuse", epoch=0, cell_auc=[1.]*6),
               dict(stage="reuse", epoch=1, cell_auc=[.7]*6),
               dict(stage="reuse", epoch=2, cell_auc=[.8]*6),
               dict(stage="reuse", epoch=3, cell_auc=[.8]*6)]
    require(selected_epoch(records, "reuse") == 2, "Stage/argmax/tie selection failed")
    try:
        selected_epoch([dict(stage="train", epoch=1, cell_auc=[.7]*5+[None])])
    except ValueError:
        pass
    else:
        raise AssertionError("Undefined cell was ignored")
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp)
        materialize(p, out)
        before = {f.name: f.read_bytes() for f in out.iterdir()}
        materialize(p, out)
        require(before == {f.name: f.read_bytes() for f in out.iterdir()}, "Compiler is not idempotent")
    return dict(status="PASS", scope="plan_compiler_and_decision_contract_only", counts=x["counts"],
                training_executed=False, gpu_tests_executed=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("self-test", "materialize", "validate"))
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        p = json.loads(args.protocol.read_text(encoding="utf-8"))
        if args.command == "self-test":
            result = self_test(p)
        elif args.command == "materialize":
            require(args.out is not None, "materialize requires --out")
            result = materialize(p, args.out)
        else:
            result = dict(status="VALID", counts=compile_plan(p)["counts"])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Plan error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
