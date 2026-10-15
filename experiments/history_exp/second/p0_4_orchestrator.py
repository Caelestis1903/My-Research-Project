"""P0-4 orchestrator: 36-run joint baseline sweep over loss-weighting arms.

Per spec (designer output) P0-4 tests whether the P0-1/2/3 pattern — the
fixed-N max-domain being systematically sacrificed under joint training
(Δ_transfer ≈ −0.015) — originates from the training objective (6-term
equal-weight BCE) or from checkpoint selection (6-cell equal-weight mean val
AUC). Three weighting arms x three role settings x four seeds, fixed ratio
50 (L/M/S = 150000/34500/3000), M3oE only (mdmoe backbone), strong-ratio.

Task key: kind|ratio|setting|weighting|method_or_domain|seed
Execution order: weighting -> setting -> seed -> baseline joint.

Master CSV: experiments/P0_4_loss_weighting/P0_4_raw.csv
            (RAW_FIELDS = P0-2 fields + `loss_weighting` column)

Run count: 3 weightings x 3 settings x 4 seeds x 1 arm (baseline joint) = 36.

No comp1, no oracle — oracle is reused from P0-2 by the report layer. All
P0-4 runs pass --fix_optim_grouping 1 and the same epoch=50, bs=40960,
lr=1e-2, earlystop=10 contract as P0-2 ratio_50 baseline.

Per-run directory:
  experiments/P0_4_loss_weighting/w_<w>/setting_<X>/baseline/seed<s>/
    config.json
    stdout.log          (written by this orchestrator)
    metrics_epoch.csv   (written by p4_run.py --run_dir; now includes
                         test_auc column per epoch)
    metrics_summary.csv

Concurrency cap is 2 (8GB VRAM laptop); stagger 30s; resume-safe (only OK
rows in master CSV count as done). A PID lockfile is written at startup
and removed on graceful exit to prevent dual-orchestrator collisions.

Usage:
  python -u experiments/p0_4_orchestrator.py [--max_runs N]
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
P4 = os.path.join(BASE, "p4_run.py")
CWD = os.path.join(ROOT, "m3oe", "examples")

P0_4_BASE = os.path.abspath(os.path.join(BASE, "P0_4_loss_weighting"))
RAW = os.path.join(P0_4_BASE, "P0_4_raw.csv")
LOGDIR = os.path.join(BASE, "logs")
LOCKFILE = os.path.join(LOGDIR, "p0_4_orchestrator.lock")

STAGGER_SEC = 30
WEIGHTINGS = ["equal", "sample", "sqrt"]
SETTINGS = ["d0_large", "d1_large", "d2_large"]
SEEDS = [2022, 2023, 2024, 2025]

# Fixed ratio 50 (strong): L = 150000, M = 34500, S = 3000. Same as the P0-2
# ratio_50 baseline arms, so the joint x_tr is byte-identical (modulo the
# loss weighting) and the report can directly re-use the ratio_50 oracle rows
# from P0-2_raw.csv.
RATIO = 50
ROLE_SIZES = "150000,34500,3000"

RAW_FIELDS = ["kind", "ratio", "setting", "loss_weighting", "method_or_domain", "seed",
              "status", "fit_sec", "peak_mem_mb", "wall_sec", "per_dt_test_auc",
              "best_epoch", "stopped_epoch", "best_val_auc", "note"]


def pid_alive(pid):
    """Return True iff pid is a live process on this OS."""
    if pid is None:
        return False
    try:
        if os.name == "nt":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            h = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False
            try:
                code = ctypes.c_ulong()
                ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
                return bool(ok) and code.value == STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(h)
        else:
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            return True
    except Exception:
        return False


def acquire_lock():
    """Raise SystemExit(1) if another live orchestrator holds the lock;
    otherwise write our PID and return."""
    os.makedirs(LOGDIR, exist_ok=True)
    if os.path.exists(LOCKFILE):
        try:
            with open(LOCKFILE, "r", encoding="utf-8") as f:
                old_pid = int(f.read().strip() or "0")
        except (OSError, ValueError):
            old_pid = 0
        if pid_alive(old_pid):
            print(f"[p0_4] another orchestrator is alive (pid={old_pid}); refusing "
                  f"to start. Lockfile: {LOCKFILE}", flush=True)
            sys.exit(1)
        try:
            os.remove(LOCKFILE)
        except OSError:
            pass
    with open(LOCKFILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def release_lock():
    try:
        if os.path.exists(LOCKFILE):
            os.remove(LOCKFILE)
    except OSError:
        pass


def done_set():
    done = set()
    if os.path.exists(RAW):
        for r in csv.DictReader(open(RAW, newline="")):
            if r["status"] == "OK":
                done.add((r["kind"], r["ratio"], r["setting"], r["loss_weighting"],
                          r["method_or_domain"], int(r["seed"])))
    return done


def build_tasks():
    tasks = []
    done = done_set()
    # Spec: weighting -> setting -> seed -> baseline (joint). All 3 weightings
    # x 3 settings x 4 seeds = 36 joint baseline runs; no comp1, no oracle.
    for w in WEIGHTINGS:
        for setting in SETTINGS:
            for seed in SEEDS:
                key = ("joint", str(RATIO), setting, w, "baseline", seed)
                if key not in done:
                    tasks.append({"kind": "joint", "ratio": RATIO,
                                  "setting": setting, "loss_weighting": w,
                                  "role_sizes": ROLE_SIZES,
                                  "method_or_domain": "baseline", "seed": seed})
    return tasks


def run_dir(t):
    return os.path.join(P0_4_BASE, f"w_{t['loss_weighting']}",
                        f"setting_{t['setting']}", t["method_or_domain"],
                        f"seed{t['seed']}")


def task_cmd(t):
    # P0-4 contract: identical training hparams to P0-2 ratio_50 baseline.
    common = ["--seed", str(t["seed"]), "--epoch", "50",
              "--batch_size", "40960",
              "--fix_optim_grouping", "1",
              "--role_setting", t["setting"],
              "--role_sizes", t["role_sizes"],
              "--run_dir", run_dir(t),
              "--stage_label", "P0-4",
              "--experiment_label", "loss_weighting",
              "--loss_weighting", t["loss_weighting"]]
    cmd = [PY, "-u", P4] + common
    name = f"p0_4_w{t['loss_weighting']}_{t['setting']}_{t['method_or_domain']}_seed{t['seed']}"
    return cmd, name


def write_config(t):
    d = run_dir(t)
    os.makedirs(d, exist_ok=True)
    cfg = {
        "kind": t["kind"],
        "ratio": t["ratio"],
        "setting": t["setting"],
        "loss_weighting": t["loss_weighting"],
        "method_or_domain": t["method_or_domain"],
        "seed": t["seed"],
        "role_sizes": t["role_sizes"],
        "momentum_mode": "none",
        "loss_weighting_arg": t["loss_weighting"],
        "fix_optim_grouping": 1,
        "epoch": 50,
        "batch_size": 40960,
        "learning_rate": 1e-2,
        "learning_rate_darts": 1e-3,
        "weight_decay": 1e-5,
        "earlystop_patience": 10,
    }
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def append_row(path, fields, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def parse_summary(log_text):
    """Parse stdout/stderr of one p4_run invocation for fit time, peak mem,
    and the FINAL per-(d,t) test auc list."""
    fit = peak = ""
    m = re.search(r"fit wall time: ([\d.]+)s \| peak cuda mem: ([\d.]+)", log_text)
    if m:
        fit, peak = m.group(1), m.group(2)
    per_dt = ""
    m = re.search(r"FINAL per-\(d,t\) test auc: \[(.*?)\]", log_text)
    if m:
        per_dt = m.group(1)
    if not per_dt:
        m = re.search(r"per-\(d,t\)\s*test\s*auc:\s*\[(.*?)\]", log_text)
        if m:
            per_dt = m.group(1)
    return fit, peak, per_dt


def parse_summary_csv(t):
    """Read metrics_summary.csv from the run dir for best_epoch / val_auc."""
    p = os.path.join(run_dir(t), "metrics_summary.csv")
    if not os.path.exists(p):
        return None, None, None
    try:
        r = next(csv.DictReader(open(p, newline="")))
    except StopIteration:
        return None, None, None
    return r.get("best_epoch"), r.get("stopped_epoch"), r.get("best_val_auc")


def finalize(t, rc, log_path, wall):
    out = open(log_path, encoding="utf-8", errors="replace").read()
    note = out.strip().splitlines()[-1][:200] if rc != 0 and out.strip() else ""
    fit, peak, per_dt = parse_summary(out)
    best_ep, stopped_ep, best_val = parse_summary_csv(t)
    ok = (rc == 0 and per_dt != "")
    append_row(RAW, RAW_FIELDS, {
        "kind": t["kind"],
        "ratio": t["ratio"],
        "setting": t["setting"],
        "loss_weighting": t["loss_weighting"],
        "method_or_domain": t["method_or_domain"],
        "seed": t["seed"],
        "status": "OK" if ok else "FAILED",
        "fit_sec": fit,
        "peak_mem_mb": peak,
        "wall_sec": f"{wall:.0f}",
        "per_dt_test_auc": per_dt,
        "best_epoch": best_ep if best_ep is not None else "",
        "stopped_epoch": stopped_ep if stopped_ep is not None else "",
        "best_val_auc": best_val if best_val is not None else "",
        "note": note,
    })
    return (f"w{t['loss_weighting']} {t['kind']} {t['setting']} "
            f"{t['method_or_domain']} seed{t['seed']}: "
            f"{'OK' if ok else 'FAILED'} auc={per_dt or 'NA'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_runs", type=int, default=10 ** 9)
    args = ap.parse_args()
    CONCURRENCY = 2
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(P0_4_BASE, exist_ok=True)
    acquire_lock()

    try:
        queue = build_tasks()[:args.max_runs]
        by_w = {}
        for t in queue:
            by_w[t["loss_weighting"]] = by_w.get(t["loss_weighting"], 0) + 1
        for w in sorted(by_w):
            print(f"  weighting {w}: {by_w[w]} pending runs", flush=True)
        print(f"queued {len(queue)} tasks (P0-4, total 36 joint baseline)", flush=True)

        slots = []
        last_launch = 0.0
        launched = 0
        while queue or slots:
            while queue and len(slots) < CONCURRENCY:
                wait = STAGGER_SEC - (time.time() - last_launch)
                if launched > 0 and wait > 0:
                    time.sleep(wait)
                t = queue.pop(0)
                write_config(t)
                cmd, name = task_cmd(t)
                log_path = os.path.join(LOGDIR, f"p0_4_{name}.log")
                fh = open(log_path, "w", encoding="utf-8", errors="replace")
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=CWD)
                slots.append({"task": t, "proc": proc, "fh": fh,
                              "log": log_path, "t0": time.time()})
                last_launch = time.time()
                launched += 1
                print(f"[launch] {name} (pid {proc.pid})", flush=True)
            time.sleep(5)
            for s in list(slots):
                rc = s["proc"].poll()
                if rc is None:
                    continue
                s["fh"].close()
                wall = time.time() - s["t0"]
                print(f"[done] {finalize(s['task'], rc, s['log'], wall)} "
                      f"wall={wall:.0f}s", flush=True)
                slots.remove(s)
        print("P0_4_ORCHESTRATOR DONE", flush=True)
    finally:
        release_lock()


if __name__ == "__main__":
    main()
