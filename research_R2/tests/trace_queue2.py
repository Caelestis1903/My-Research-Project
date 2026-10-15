"""Trace _paired_run directly."""
import json
import os, sys
sys.path.insert(0, ".")
sys.path.insert(0, "research_R2/src")
sys.path.insert(0, "experiments")
sys.path.insert(0, "m3oe")

from research_R2.run_queue import (
    QueueKey, _spawn_run_formal, _paired_run, _wait_pair,
)
import time

PY_ROOT = os.getcwd()
TMP = os.path.join(os.environ["TEMP"], "r2_orch_trace2")
SRC = os.path.join(TMP, "research_R2_root")
runs_root = os.path.join(SRC, "runs")
init_dir = os.path.join(SRC, "init")

os.makedirs(runs_root, exist_ok=True)
os.makedirs(init_dir, exist_ok=True)

# Test basic spawn
key_a = QueueKey(dataset="kuairand", arm="B0", seed=2022)
key_b = QueueKey(dataset="movielens", arm="B0", seed=2022)

# Make a real stub-formal in a known spot
stub_formal = os.path.join(TMP, "stub.py")
with open(stub_formal, "w", encoding="utf-8") as f:
    f.write('''import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--seed", required=True, type=int)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--init-dir", required=False)
ap.add_argument("--epochs", required=False, type=int)
args = ap.parse_args()
os.makedirs(args.run_dir, exist_ok=True)
manifest = {
    "stage": "R2", "design": "ao04_v12",
    "status": os.environ.get("R2_STUB_STATUS", "COMPLETE"),
    "arm": args.arm, "seed": args.seed,
    "dataset": "kuairand" if "kuairand" in args.config else "movielens",
    "backbone": "m3oe",
    "run_dir": os.path.abspath(args.run_dir),
    "selected_epoch": 0, "selected_validation_macro_auc": 0.5,
    "finished_at": "2026-10-12T00:00:00+00:00",
    "queued_at": "2026-10-12T00:00:00+00:00",
    "epochs_completed": 1, "smoke_completed": True,
    "test_evaluated_during_training": False, "test_predictions_written": False,
}
with open(os.path.join(args.run_dir, "run_manifest.json"), "w", encoding="utf-8") as h:
    json.dump(manifest, h, indent=2)
sys.exit(int(os.environ.get("R2_STUB_RC", "0")))
''')

os.environ["R2_STUB_FORMAL"] = stub_formal
os.environ["R2_STUB_STATUS"] = "COMPLETE"
os.environ["R2_STUB_RC"] = "0"
os.environ["R2_SELF_TEST"] = "1"

print("key_a config_path:", key_a.config_path)
print("key_a config exists?", os.path.exists(key_a.config_path))
print("key_a relative_run_dir:", key_a.relative_run_dir(runs_root))

proc_a = _spawn_run_formal(key_a, runs_root=runs_root, init_dir=init_dir)
proc_b = _spawn_run_formal(key_b, runs_root=runs_root, init_dir=init_dir)
print("spawned:", proc_a.pid, proc_b.pid)
print("wait_pair...")
rc_a, rc_b = _wait_pair(proc_a, proc_b)
print("rc_a=", rc_a, "rc_b=", rc_b)

# Check run manifest exists
manifest_a = os.path.join(key_a.relative_run_dir(runs_root), "run_manifest.json")
print("manifest_a path:", manifest_a, "exists:", os.path.exists(manifest_a))
if os.path.exists(manifest_a):
    with open(manifest_a, encoding="utf-8") as h:
        m = json.load(h)
    print("status=", m.get("status"))

from research_R2.run_queue import _refresh_run_status
queue_state = {"phase": "r2a", "status": "RUNNING", "per_run": {}}

ok_a = _refresh_run_status(key_a, rc_a, queue_state, runs_root)
ok_b = _refresh_run_status(key_b, rc_b, queue_state, runs_root)
print("ok_a=", ok_a, "ok_b=", ok_b)
