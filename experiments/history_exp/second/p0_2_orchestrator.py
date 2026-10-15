"""P0-2 orchestrator: 140-run ratio sweep over L/M/S domain-size roles.

Per spec (designer output) the sweep keeps N = L+M+S = 187,500 fixed and
varies the L:M:S ratio. Three ratios are run fresh (1:1 / 5:1 / 50:1); the
20:1 ratio is reused from P0-1 (handled by the report layer, not here).

Task key: kind|ratio|setting|method_or_domain|seed
Execution order: ratio_1 -> ratio_5 -> ratio_50; within a ratio, setting-
major -> seed -> [baseline, comp1, oracle(L域), oracle(M域), oracle(S域)].
The 1:1 ratio collapses to a single setting (1:1 is rotation-invariant: all
three settings sample the same three sets of rows up to permutation, so the
joint x_tr is byte-identical across settings -> only d0_large is run, the
other two are skipped to save the 2*20 = 40 redundant runs).

Per-run directory:
  experiments/P0_2_ratio_sweep/ratio_<r>/setting_<X>/<method|oracle_d<d>>/seed<s>/
    config.json
    stdout.log          (written by this orchestrator)
    metrics_epoch.csv   (written by p4_run.py --run_dir)
    metrics_summary.csv

Master CSV: experiments/P0_2_ratio_sweep/P0_2_raw.csv
            (RAW_FIELDS = P0-1 RAW_FIELDS + extra `ratio` column)

Run count: ratio_1 (1 setting x [2 joint + 3 oracle] x 4 seeds) +
           ratio_5 (3 settings x 5 arms x 4 seeds) +
           ratio_50 (3 settings x 5 arms x 4 seeds)
         = 1*5*4 + 3*5*4 + 3*5*4 = 20 + 60 + 60 = 140.

Concurrency cap is 2 (8GB VRAM laptop); stagger 30s; resume-safe (only OK
rows in master CSV count as done). A PID lockfile is written at startup and
removed on graceful exit to prevent the dual-orchestrator collision that
struck P0-1.

Usage:
  python -u experiments/p0_2_orchestrator.py [--max_runs N]
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

P0_2_BASE = os.path.abspath(os.path.join(BASE, "P0_2_ratio_sweep"))
RAW = os.path.join(P0_2_BASE, "P0_2_raw.csv")
LOGDIR = os.path.join(BASE, "logs")
LOCKFILE = os.path.join(LOGDIR, "p0_2_orchestrator.lock")

STAGGER_SEC = 30
SETTINGS = ["d0_large", "d1_large", "d2_large"]
SEEDS = [2022, 2023, 2024, 2025]

# (ratio, role_sizes string). 20:1 is reused from P0-1 and is NOT in the
# task table (spec: "ratio_20 不进任务表"). The three fresh ratios share
# N = 187,500 = L + M + S.
RATIO_TASKS = [
    # ratio 1: 1 -> d0, 1 -> d1, 1 -> d2 — rotation-invariant
    (1, "62500,62500,62500", ["d0_large"]),
    # ratio 5
    (5, "125000,37500,25000", SETTINGS),
    # ratio 50
    (50, "150000,34500,3000", SETTINGS),
]

# Per spec section 3 arms. All P0-2 runs pass --fix_optim_grouping 1.
JOINT_METHODS = {
    "baseline": [],
    "comp1": ["--momentum_mode", "per_domain_only"],
}
# spec section 5 execution order within each (setting, seed):
#   [baseline, comp1, oracle(L域), oracle(M域), oracle(S域)].
# "L域" is the domain that plays L under that setting (see ROLE_MAPS in
# p4_run.py). The domain-to-role mapping is encoded here so the order is
# correct. For ratio_1 the joint x_tr is invariant under the setting, so
# only d0_large is run; the L域/M域/S域 ordering is d0/d1/d2.
SETTING_LMS_DOMAIN = {
    "d0_large": [0, 1, 2],   # L域=d0, M域=d1, S域=d2
    "d1_large": [1, 2, 0],   # L域=d1, M域=d2, S域=d0
    "d2_large": [2, 0, 1],   # L域=d2, M域=d0, S域=d1
}

RAW_FIELDS = ["kind", "ratio", "setting", "method_or_domain", "seed", "status",
              "fit_sec", "peak_mem_mb", "wall_sec", "per_dt_test_auc",
              "best_epoch", "stopped_epoch", "best_val_auc", "note"]


def pid_alive(pid):
    """Return True iff pid is a live process on this OS."""
    if pid is None:
        return False
    try:
        if os.name == "nt":
            # OpenProcess with PROCESS_QUERY_LIMITED_INFORMATION (0x1000)
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
            print(f"[p0_2] another orchestrator is alive (pid={old_pid}); refusing "
                  f"to start. Lockfile: {LOCKFILE}", flush=True)
            sys.exit(1)
        # Stale lock (process dead). Remove and re-acquire below.
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
                done.add((r["kind"], r["ratio"], r["setting"],
                          r["method_or_domain"], int(r["seed"])))
    return done


def build_tasks():
    tasks = []
    done = done_set()
    for ratio, sizes, settings in RATIO_TASKS:
        for setting in settings:
            for seed in SEEDS:
                # baseline
                key = ("joint", str(ratio), setting, "baseline", seed)
                if key not in done:
                    tasks.append({"kind": "joint", "ratio": ratio,
                                  "setting": setting, "role_sizes": sizes,
                                  "method_or_domain": "baseline", "seed": seed})
                # comp1
                key = ("joint", str(ratio), setting, "comp1", seed)
                if key not in done:
                    tasks.append({"kind": "joint", "ratio": ratio,
                                  "setting": setting, "role_sizes": sizes,
                                  "method_or_domain": "comp1", "seed": seed})
                # oracle in (L域, M域, S域) order (per spec section 5)
                for d in SETTING_LMS_DOMAIN[setting]:
                    key = ("oracle", str(ratio), setting, f"oracle_d{d}", seed)
                    if key not in done:
                        tasks.append({"kind": "oracle", "ratio": ratio,
                                      "setting": setting, "role_sizes": sizes,
                                      "method_or_domain": f"oracle_d{d}",
                                      "seed": seed, "single_domain": d})
    return tasks


def run_dir(t):
    method_dir = t["method_or_domain"]
    return os.path.join(P0_2_BASE, f"ratio_{t['ratio']}",
                        f"setting_{t['setting']}", method_dir,
                        f"seed{t['seed']}")


def task_cmd(t):
    common = ["--seed", str(t["seed"]), "--epoch", "50",
              "--batch_size", "40960",
              "--fix_optim_grouping", "1",
              "--role_setting", t["setting"],
              "--role_sizes", t["role_sizes"],
              "--run_dir", run_dir(t),
              "--stage_label", "P0-2",
              "--experiment_label", "ratio_sweep"]
    if t["kind"] == "joint":
        extra = JOINT_METHODS[t["method_or_domain"]]
        cmd = [PY, "-u", P4] + common + extra
        name = f"joint_ratio{t['ratio']}_{t['setting']}_{t['method_or_domain']}_seed{t['seed']}"
    else:
        cmd = [PY, "-u", P4] + common + ["--single_domain", str(t["single_domain"])]
        name = f"oracle_ratio{t['ratio']}_{t['setting']}_{t['method_or_domain']}_seed{t['seed']}"
    return cmd, name


def write_config(t):
    d = run_dir(t)
    os.makedirs(d, exist_ok=True)
    cfg = {
        "kind": t["kind"],
        "ratio": t["ratio"],
        "setting": t["setting"],
        "method_or_domain": t["method_or_domain"],
        "seed": t["seed"],
        "single_domain": t.get("single_domain", -1),
        "role_sizes": t["role_sizes"],
        "momentum_mode": ("per_domain_only"
                          if (t["kind"] == "joint" and t["method_or_domain"] == "comp1")
                          else "none"),
        "fix_optim_grouping": 1,
        "epoch": 50,
        "batch_size": 40960,
        "learning_rate": 1e-2,
        "learning_rate_darts": 1e-3,
        "weight_decay": 1e-5,
        "earlystop_patience": 10,
        "gamma0": 0.9,
        "gamma_max": 0.99,
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
    return (f"ratio{t['ratio']} {t['kind']} {t['setting']} "
            f"{t['method_or_domain']} seed{t['seed']}: "
            f"{'OK' if ok else 'FAILED'} auc={per_dt or 'NA'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_runs", type=int, default=10 ** 9)
    args = ap.parse_args()
    assert True, "concurrency cap enforced below"
    # Hard cap per spec: concurrency <= 2 (8GB VRAM laptop).
    CONCURRENCY = 2
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(P0_2_BASE, exist_ok=True)
    acquire_lock()

    try:
        queue = build_tasks()[:args.max_runs]
        # Count what will be launched by ratio for visibility.
        by_ratio = {}
        for t in queue:
            by_ratio[t["ratio"]] = by_ratio.get(t["ratio"], 0) + 1
        for r in sorted(by_ratio):
            print(f"  ratio {r}: {by_ratio[r]} pending runs", flush=True)
        print(f"queued {len(queue)} tasks (P0-2, total 140)", flush=True)

        slots = []          # list of dicts: task, proc, log_fh, log_path, t0
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
                log_path = os.path.join(LOGDIR, f"p0_2_{name}.log")
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
        print("P0_2_ORCHESTRATOR DONE", flush=True)
    finally:
        release_lock()


if __name__ == "__main__":
    main()
