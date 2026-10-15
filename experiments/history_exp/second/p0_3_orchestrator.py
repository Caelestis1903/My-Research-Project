"""P0-3 orchestrator: 80-run MMoE transfer re-run of the P0-2 ratio_1 + ratio_50
cells, on the same cached data, so P0-3 can pair M3OE (MDMTRec) and MMoE rows
cell-by-cell (per (档, setting, domain, task, seed)) in the report.

Differences from p0_2_orchestrator.py:
- Base dir:  experiments/P0_3_mmoe_transfer/   (instead of P0_2_ratio_sweep/)
- Raw CSV:   P0_3_raw.csv                      (RAW_FIELDS = P0-2 RAW + backbone col)
- Only two fresh ratios: ratio_1 (balanced, d0_large only) and ratio_50
  (strong, 3 settings). 80 runs total. (ratio_5 is dropped per the spec;
  ratio_20 is not re-run from P0-1 either.)
- Every run appends --backbone mmoe --stage_label P0-3 --experiment_label
  mmoe_transfer, and uses the same --fix_optim_grouping 1 (with --momentum_mode
  per_domain_only for the comp1 arm, none for baseline, per P0-2 contract).
- Oracle arms pass --single_domain <d>; balance and seed arrays match P0-2.

Run count: balanced (1 setting x [2 joint + 3 oracle] x 4 seeds)
         + strong (3 settings x [2 joint + 3 oracle] x 4 seeds)
         = 1*5*4 + 3*5*4 = 20 + 60 = 80.

Concurrency cap 2; 30s stagger; resume-safe (only OK rows count as done);
PID lockfile. Run dirs mirror P0-2 (so P0-3 report can pair on
{档, setting, method_or_domain, seed} using the same directory layout).

Usage:
  python -u experiments/p0_3_orchestrator.py [--max_runs N]
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

P0_3_BASE = os.path.abspath(os.path.join(BASE, "P0_3_mmoe_transfer"))
RAW = os.path.join(P0_3_BASE, "P0_3_raw.csv")
LOGDIR = os.path.join(BASE, "logs")
LOCKFILE = os.path.join(LOGDIR, "p0_3_orchestrator.lock")

STAGGER_SEC = 30
SETTINGS = ["d0_large", "d1_large", "d2_large"]
SEEDS = [2022, 2023, 2024, 2025]

# Two fresh ratio tiers; ratio_5 is not re-run for P0-3 (the spec only asks
# for the two extreme cases to keep 80 runs tractable). The two tiers are
# labelled `档` (tier) in the run-dir layout and report: "balanced" = 1:1,
# "strong" = 50:1. Same N = L+M+S as P0-2.
TIER_TASKS = [
    # (tier_name, role_sizes, settings). "ratio" column in the CSV carries
    # the underlying imbalance ratio (1, 50) so the report can join P0-2
    # rows on ratio.
    ("balanced", "62500,62500,62500",  1, ["d0_large"]),       # 1:1 tier
    ("strong",   "150000,34500,3000", 50, SETTINGS),            # 50:1 tier
]

# Per spec section 3 arms. All P0-3 runs pass --fix_optim_grouping 1.
JOINT_METHODS = {
    "baseline": [],
    "comp1": ["--momentum_mode", "per_domain_only"],
}
# L域/M域/S域 -> (d0,d1,d2) under each setting. Copied verbatim from
# p0_2_orchestrator.py:85-89 so the orchestrator order matches.
SETTING_LMS_DOMAIN = {
    "d0_large": [0, 1, 2],
    "d1_large": [1, 2, 0],
    "d2_large": [2, 0, 1],
}

# Same as P0-2 plus a leading `backbone` column so the report can group
# rows by backbone without scanning args.
RAW_FIELDS = ["backbone", "tier", "ratio", "setting", "method_or_domain", "seed",
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
    os.makedirs(LOGDIR, exist_ok=True)
    if os.path.exists(LOCKFILE):
        try:
            with open(LOCKFILE, "r", encoding="utf-8") as f:
                old_pid = int(f.read().strip() or "0")
        except (OSError, ValueError):
            old_pid = 0
        if pid_alive(old_pid):
            print(f"[p0_3] another orchestrator is alive (pid={old_pid}); refusing "
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
    """Read P0_3_raw.csv and return the set of (tier, ratio, setting,
    method_or_domain, seed) tuples already marked OK. Used to skip finished
    runs on resume."""
    done = set()
    if os.path.exists(RAW):
        for r in csv.DictReader(open(RAW, newline="")):
            if r["status"] == "OK":
                done.add((r["tier"], int(r["ratio"]), r["setting"],
                          r["method_or_domain"], int(r["seed"])))
    return done


def build_tasks():
    tasks = []
    done = done_set()
    for tier, sizes, ratio, settings in TIER_TASKS:
        for setting in settings:
            for seed in SEEDS:
                # baseline
                key = (tier, ratio, setting, "baseline", seed)
                if key not in done:
                    tasks.append({"tier": tier, "ratio": ratio,
                                  "setting": setting, "role_sizes": sizes,
                                  "method_or_domain": "baseline", "seed": seed})
                # comp1
                key = (tier, ratio, setting, "comp1", seed)
                if key not in done:
                    tasks.append({"tier": tier, "ratio": ratio,
                                  "setting": setting, "role_sizes": sizes,
                                  "method_or_domain": "comp1", "seed": seed})
                # oracle in (L域, M域, S域) order (matches P0-2).
                for d in SETTING_LMS_DOMAIN[setting]:
                    key = (tier, ratio, setting, f"oracle_d{d}", seed)
                    if key not in done:
                        tasks.append({"tier": tier, "ratio": ratio,
                                      "setting": setting, "role_sizes": sizes,
                                      "method_or_domain": f"oracle_d{d}",
                                      "seed": seed, "single_domain": d})
    return tasks


def run_dir(t):
    """Per-run directory layout: P0_3_mmoe_transfer/<tier>/setting_<X>/<method>/seed<s>/.
    The tier directory matches the P0-2 `ratio_<r>` dir so the pairing report
    can key on (tier, setting, method, seed) directly."""
    method_dir = t["method_or_domain"]
    return os.path.join(P0_3_BASE, t["tier"],
                        f"setting_{t['setting']}", method_dir,
                        f"seed{t['seed']}")


def task_cmd(t):
    common = ["--seed", str(t["seed"]), "--epoch", "50",
              "--batch_size", "40960",
              "--fix_optim_grouping", "1",
              "--role_setting", t["setting"],
              "--role_sizes", t["role_sizes"],
              "--run_dir", run_dir(t),
              "--stage_label", "P0-3",
              "--experiment_label", "mmoe_transfer",
              "--backbone", "mmoe"]
    if t["kind"] == "joint":
        extra = JOINT_METHODS[t["method_or_domain"]]
        cmd = [PY, "-u", P4] + common + extra
        name = f"mmoe_{t['tier']}_{t['setting']}_{t['method_or_domain']}_seed{t['seed']}"
    else:
        cmd = [PY, "-u", P4] + common + ["--single_domain", str(t["single_domain"])]
        name = f"mmoe_{t['tier']}_{t['setting']}_{t['method_or_domain']}_seed{t['seed']}"
    return cmd, name


def write_config(t):
    d = run_dir(t)
    os.makedirs(d, exist_ok=True)
    cfg = {
        "backbone": "mmoe",
        "tier": t["tier"],
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
        "backbone": "mmoe",
        "tier": t["tier"],
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
    return (f"{t['tier']}(ratio{t['ratio']}) {t['setting']} "
            f"{t['method_or_domain']} seed{t['seed']}: "
            f"{'OK' if ok else 'FAILED'} auc={per_dt or 'NA'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_runs", type=int, default=10 ** 9)
    args = ap.parse_args()
    CONCURRENCY = 2
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(P0_3_BASE, exist_ok=True)
    acquire_lock()

    try:
        # Inject the `kind` field the rest of the pipeline relies on (joint vs
        # oracle). The TIER_TASKS list doesn't carry it because the per-arm
        # build loop only generates joint vs oracle separately.
        pending = build_tasks()
        for t in pending:
            if t["method_or_domain"].startswith("oracle_d"):
                t["kind"] = "oracle"
            else:
                t["kind"] = "joint"
        queue = pending[:args.max_runs]
        # Count what will be launched by tier for visibility.
        by_tier = {}
        for t in queue:
            by_tier[t["tier"]] = by_tier.get(t["tier"], 0) + 1
        for tier in sorted(by_tier):
            print(f"  tier {tier}: {by_tier[tier]} pending runs", flush=True)
        print(f"queued {len(queue)} tasks (P0-3, total 80)", flush=True)

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
                log_path = os.path.join(LOGDIR, f"p0_3_{name}.log")
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
        print("P0_3_ORCHESTRATOR DONE", flush=True)
    finally:
        release_lock()


if __name__ == "__main__":
    main()
