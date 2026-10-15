"""Quick direct trace: run_queue r2a both on stub-formal."""
import os, sys, shutil, subprocess

PY_ROOT = os.getcwd()
TMP = os.path.join(os.environ["TEMP"], "r2_orch_trace")
SRC = os.path.join(TMP, "research_R2_root")
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(SRC)
os.makedirs(os.path.join(SRC, "configs"))
shutil.copyfile(os.path.join(PY_ROOT, "research_R2", "configs", "kuairand_natural_m3oe.json"),
                os.path.join(SRC, "configs", "kuairand_natural_m3oe.json"))
shutil.copyfile(os.path.join(PY_ROOT, "research_R2", "configs", "movielens_d0_large_m3oe.json"),
                os.path.join(SRC, "configs", "movielens_d0_large_m3oe.json"))
os.makedirs(os.path.join(SRC, "init"))
os.makedirs(os.path.join(SRC, "runs"))
os.makedirs(os.path.join(SRC, "decisions"))
stub_formal = os.path.join(TMP, "run_formal_stub.py")
_STUB = '''#!/usr/bin/env python
import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--seed", required=True, type=int)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--init-dir", required=False)
ap.add_argument("--epochs", required=False, type=int)
args = ap.parse_args()
os.makedirs(args.run_dir, exist_ok=True)
dataset = "kuairand" if "kuairand" in args.config else "movielens"
manifest = {
    "stage": "R2", "design": "ao04_v12",
    "status": os.environ.get("R2_STUB_STATUS", "COMPLETE"),
    "arm": args.arm, "seed": args.seed, "dataset": dataset,
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
'''
with open(stub_formal, "w", encoding="utf-8") as h:
    h.write(_STUB)
env = dict(os.environ)
env["R2_STUB_FORMAL"] = stub_formal
env["R2_STUB_STATUS"] = "COMPLETE"
env["R2_STUB_RC"] = "0"
env["R2_SELF_TEST"] = "1"
env["PYTHONPATH"] = PY_ROOT
proc = subprocess.run([
    sys.executable, os.path.join(PY_ROOT, "research_R2", "run_queue.py"),
    "r2a", "--dataset", "both",
    "--runs-root", os.path.join(SRC, "runs"),
    "--init-dir", os.path.join(SRC, "init"),
], cwd=PY_ROOT, env=env, capture_output=True, text=True)
print("rc =", proc.returncode)
print("STDOUT ---")
print(proc.stdout)
print("STDERR ---")
print(proc.stderr)
