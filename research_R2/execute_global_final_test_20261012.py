from __future__ import annotations
import csv, hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UNLOCK_PATH = ROOT / "locks/global_test_unlock.json"
FREEZE_PATH = ROOT / "locks/global_selection_freeze.json"
EVALUATOR = ROOT / "src/ao04_v12_test_evaluator.py"
OUT = ROOT / "preflight/global_final_test_execution_20261012.json"
ACCEPTED_EVALUATOR_SHA = "bb49baacb2d005ff93b1d510a1d085b1f2267692b04e80d34eeff6a98dbf8a06"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

freeze = json.loads(FREEZE_PATH.read_text())
unlock = json.loads(UNLOCK_PATH.read_text())
freeze_sha = sha(FREEZE_PATH)
assert unlock.get("global_test_unlock") is True
assert unlock.get("freeze_manifest_sha256") == freeze_sha
assert sha(EVALUATOR) == ACCEPTED_EVALUATOR_SHA, "reviewed evaluator hash drift"
runs = unlock["valid_complete_runs"]
assert len(runs) == 9
assert all(r["status"] == "COMPLETE" and r["epochs_completed"] == 50 for r in runs)
assert not any(r["dataset"] == "kuairand" and r["arm"] == "B1" and r["seed"] == 2023 for r in runs)
assert {(r["dataset"], r["arm"], r["seed"]) for r in runs} == {
    ("kuairand", "B0", 2022), ("kuairand", "B0", 2023), ("kuairand", "B1", 2022),
    *(("movielens", a, s) for a in ("B0", "B1", "B2") for s in (2022, 2023)),
}
# Verify exact frozen file inventory and prohibit a pre-existing test output.
for run in runs:
    run_dir = Path(run["run_dir"])
    assert run_dir.is_dir()
    frozen = run.get("files", {})
    for name, meta in frozen.items():
        f = run_dir / name
        assert f.is_file() and sha(f) == meta["sha256"], f"frozen file drift: {f}"
    for name in ("metrics_test.csv", "predictions_test.npz"):
        assert not (run_dir / name).exists(), f"pre-existing test output: {run_dir/name}"

runtime = {
    "kuairand": ROOT / "configs/runtime_20261012/kuairand_natural_m3oe.json",
    "movielens": ROOT / "configs/runtime_20261012/movielens_d0_large_m3oe.json",
}
expected_groups = {"kuairand": ("natural", 3), "movielens": ("d0_large", 6)}
for ds, cfg_path in runtime.items():
    cfg = json.loads(cfg_path.read_text())
    data_path = Path(cfg["dataset_path"])
    assert data_path.is_file() and sha(data_path) == cfg["dataset_sha256"], f"dataset hash mismatch: {ds}"
    assert sum(r["dataset"] == ds for r in runs) == expected_groups[ds][1]

# The evaluator derives the shared aggregate output path from the first run.
# Confirm there is no existing aggregate so the one-call-per-dataset output is unique.
for ds in runtime:
    ds_runs = [r for r in runs if r["dataset"] == ds]
    first = Path(ds_runs[0]["run_dir"])
    aggregate = first.parents[1] / "_final_test_evaluated"
    assert not aggregate.exists(), f"pre-existing aggregate output: {aggregate}"

# Write a preflight receipt before test access. This also establishes exact scope.
preflight = {
    "status": "PREFLIGHT_PASS", "freeze_sha256": freeze_sha,
    "unlock_sha256": sha(UNLOCK_PATH), "evaluator_sha256": sha(EVALUATOR),
    "runs": [{"dataset": r["dataset"], "setting": r["setting"], "arm": r["arm"],
              "seed": r["seed"], "run_dir": r["run_dir"]} for r in runs],
    "endpoints_expected": 63, "dataset_run_counts": {"kuairand": 3, "movielens": 6},
    "counter_semantics": "one increment per dataset bundle + one per run; not per checkpoint; expected final counter 11",
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
preflight_path = ROOT / "preflight/global_final_test_preflight_20261012.json"
assert not preflight_path.exists(), f"refusing overwrite: {preflight_path}"
preflight_path.write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n")

# Imports occur after preflight and the global unlock checks.
sys.path.insert(0, str(ROOT / "src"))
import ao04_v12_runner as runner
from ao04_v12_test_evaluator import final_test_evaluator
runner.TEST_PREDICTION_CALLS = 0
artifacts = {}
for ds, (setting, _) in expected_groups.items():
    ds_runs = [r for r in runs if r["dataset"] == ds]
    run_dirs = {(r["arm"], int(r["seed"]), r["backbone"]): r["run_dir"] for r in ds_runs}
    cfg = json.loads(runtime[ds].read_text())
    print(f"START dataset={ds} runs={len(run_dirs)} endpoints={len(run_dirs)*7} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    art = final_test_evaluator(run_dirs, unlock_manifest_path=str(UNLOCK_PATH),
        dataset=ds, setting=setting, backbones=("m3oe",), dataset_path=cfg["dataset_path"])
    artifacts[ds] = {"metrics_test_csv": art.metrics_test_csv,
        "predictions_test_npz": art.predictions_test_npz,
        "augmented_manifest_path": art.augmented_manifest_path}
    print(f"DONE dataset={ds} artifacts={artifacts[ds]} counter={runner.TEST_PREDICTION_CALLS}", flush=True)

assert runner.TEST_PREDICTION_CALLS == 11, runner.TEST_PREDICTION_CALLS
# Integrity-only validation: seven checkpoint labels x six domain-task cells per run.
for r in runs:
    run_dir = Path(r["run_dir"])
    csv_path = run_dir / "metrics_test.csv"
    npz_path = run_dir / "predictions_test.npz"
    assert csv_path.is_file() and npz_path.is_file()
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 42, (csv_path, len(rows))
    assert len({row["checkpoint"] for row in rows}) == 7
    assert len({row["cell"] for row in rows}) == 6
    assert len({(row["checkpoint"], row["cell"]) for row in rows}) == 42

for ds, art in artifacts.items():
    manifest = json.loads(Path(art["augmented_manifest_path"]).read_text())
    assert manifest["records_count"] == expected_groups[ds][1] * 42
    assert manifest["test_prediction_calls_global"] == (4 if ds == "kuairand" else 11)

receipt = {
    **preflight, "status": "COMPLETE", "artifacts": artifacts,
    "test_prediction_calls_final": int(runner.TEST_PREDICTION_CALLS),
    "per_run_metrics_rows": 42, "total_per_run_rows": 9 * 42,
    "endpoints_evaluated": 63,
    "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "outputs": {},
}
paths = []
for r in runs:
    rd = Path(r["run_dir"])
    paths += [rd/"metrics_test.csv", rd/"predictions_test.npz"]
for art in artifacts.values():
    paths += [Path(art["metrics_test_csv"]), Path(art["predictions_test_npz"]), Path(art["augmented_manifest_path"])]
for p in paths:
    receipt["outputs"][str(p)] = {"sha256": sha(p), "bytes": p.stat().st_size}
assert not OUT.exists(), f"refusing overwrite: {OUT}"
OUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print("FINAL_TEST_COMPLETE", json.dumps({"calls": runner.TEST_PREDICTION_CALLS,
    "endpoints": 63, "runs": 9, "receipt": str(OUT)}, ensure_ascii=False), flush=True)
