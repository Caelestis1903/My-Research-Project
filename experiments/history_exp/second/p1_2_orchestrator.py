"""P1-2 orchestrator: Curriculum (comp2) Life-or-Death, corrected口径.

Re-tests the comp2 (domain-conditioned curriculum on fusion weights) keep /
drop verdict under the corrected sample-weighting (P0-4 w=3p) loss formula.
6 arms x 3 settings x 4 seeds = 72 cells. Layout:

  P1_2_curriculum/
    P1_2_raw.csv
    logs/
    {A0,B0,C0,A1,B1,C1}/setting_<X>/seed<s>/     <- run dirs

Arm definitions (curriculum_mode x comp1, all under --loss_weighting sample):
  A0: curriculum none, comp1 off     -> reuse P0-4 w_sample/baseline
  B0: domain_conditioned kappa0=0.5, comp1 off -> 12 fresh
  C0: domain_conditioned kappa0=-0.5, comp1 off -> 12 fresh
  A1: curriculum none, comp1 on      -> reuse P0-5 w_sample/comp1
  B1: domain_conditioned kappa0=0.5, comp1 on  -> 12 fresh
  C1: domain_conditioned kappa0=-0.5, comp1 on -> 12 fresh

Reuse contract: read the source run dir's config.json and verify the
relevant hparams match what A0/A1 require (curriculum_mode defaults to
"none", which is consistent with both P0-4 and P0-5 source rows).
Data-cache hit is inferred: P0-4/P0-5 used the same (seed, role_setting,
role_sizes) triple, so the deterministic preprocessing cache file is
identical for the A0/A1 arms. If reuse passes the check, symlink the
source dir under P1_2_curriculum/A<x>/... so downstream tools find the
run under its new canonical name without rerunning training. If the
reuse check fails, the orchestrator records the mismatch in the raw CSV
and queues a fresh run.

grad_norm_epoch.csv: P0-4 / P0-5 did not produce it, so the A arms lack
the grad norm trajectory. By default the orchestrator reuses without
re-running; if --include_A_grad_norm=1 is passed, the A arms are
re-launched with --log_grad_norm 1 (48 fresh runs instead of 24).

Run cmd line:
  python -u experiments/p1_2_orchestrator.py [--dry_run]
  python -u experiments/p1_2_orchestrator.py --include_A_grad_norm 1 --dry_run
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

P1_2_BASE = os.path.abspath(os.path.join(BASE, "P1_2_curriculum"))
RAW = os.path.join(P1_2_BASE, "P1_2_raw.csv")
LOGDIR = os.path.join(BASE, "logs")
LOCKFILE = os.path.join(LOGDIR, "p1_2_orchestrator.lock")

# P0-4 / P0-5 source dirs (referenced for reuse)
P0_4_BASE = os.path.join(BASE, "P0_4_loss_weighting")
P0_5_BASE = os.path.join(BASE, "P0_5_comp1_x_weighting")

STAGGER_SEC = 30
SETTINGS = ["d0_large", "d1_large", "d2_large"]
SEEDS = [2022, 2023, 2024, 2025]

# Same contract as P0-2 ratio_50 baseline / comp1.
RATIO = 50
ROLE_SIZES = "150000,34500,3000"

# Arm spec: name -> (momentum_mode, curriculum_mode, kappa0, source)
# `source` is the label written to the raw CSV's "source" column and the
# path template used by `source_path()` for reuse detection. `source` may
# be None for fresh-only arms.
ARMS = {
    "A0": {"momentum_mode": "none",
           "curriculum_mode": "none", "kappa0": 0.0,
           "loss_weighting": "sample", "gamma0": 0.9, "gamma_max": 0.99,
           "source": "p0_4_sample_baseline"},
    "B0": {"momentum_mode": "none",
           "curriculum_mode": "domain_conditioned", "kappa0": 0.5,
           "loss_weighting": "sample", "gamma0": 0.9, "gamma_max": 0.99,
           "source": None},
    "C0": {"momentum_mode": "none",
           "curriculum_mode": "domain_conditioned", "kappa0": -0.5,
           "loss_weighting": "sample", "gamma0": 0.9, "gamma_max": 0.99,
           "source": None},
    "A1": {"momentum_mode": "per_domain_only",
           "curriculum_mode": "none", "kappa0": 0.0,
           "loss_weighting": "sample", "gamma0": 0.9, "gamma_max": 0.99,
           "source": "p0_5_sample_comp1"},
    "B1": {"momentum_mode": "per_domain_only",
           "curriculum_mode": "domain_conditioned", "kappa0": 0.5,
           "loss_weighting": "sample", "gamma0": 0.9, "gamma_max": 0.99,
           "source": None},
    "C1": {"momentum_mode": "per_domain_only",
           "curriculum_mode": "domain_conditioned", "kappa0": -0.5,
           "loss_weighting": "sample", "gamma0": 0.9, "gamma_max": 0.99,
           "source": None},
}

RAW_FIELDS = ["kind", "ratio", "setting", "arm", "comp1", "curriculum_mode", "kappa0",
              "loss_weighting", "method_or_domain", "seed",
              "status", "source", "reuse_check", "fit_sec", "peak_mem_mb",
              "wall_sec", "per_dt_test_auc", "best_epoch", "stopped_epoch",
              "best_val_auc", "note"]


# ----------------------------------------------------------------------------
# PID lock (mirrors p0_5_orchestrator.py)
# ----------------------------------------------------------------------------

def pid_alive(pid):
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
            print(f"[p1_2] another orchestrator is alive (pid={old_pid}); refusing "
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
                done.add((r["arm"], r["setting"], int(r["seed"])))
    return done


# ----------------------------------------------------------------------------
# Run dir / source path
# ----------------------------------------------------------------------------

def run_dir(t):
    return os.path.join(P1_2_BASE, t["arm"], f"setting_{t['setting']}",
                        f"seed{t['seed']}")


def source_path(t):
    """Path to the source run dir if t has a source. None for fresh arms."""
    src = ARMS[t["arm"]]["source"]
    if src is None:
        return None
    if src == "p0_4_sample_baseline":
        return os.path.join(P0_4_BASE, "w_sample",
                            f"setting_{t['setting']}", "baseline",
                            f"seed{t['seed']}")
    if src == "p0_5_sample_comp1":
        return os.path.join(P0_5_BASE, "w_sample",
                            f"setting_{t['setting']}", "comp1",
                            f"seed{t['seed']}")
    raise ValueError(f"unknown source label: {src}")


# ----------------------------------------------------------------------------
# Reuse check: compare source config.json against the A-arm hparam contract
# ----------------------------------------------------------------------------

def _load_source_cfg(path):
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def reuse_check(t):
    """Verify a source run dir matches the A-arm hparam contract bit-for-bit
    on every field we depend on. Returns (ok: bool, reason: str)."""
    src = source_path(t)
    if src is None:
        return False, "no source label"
    if not os.path.exists(src):
        return False, f"source dir missing: {src}"
    if not os.path.exists(os.path.join(src, "metrics_summary.csv")):
        return False, "source missing metrics_summary.csv"
    cfg = _load_source_cfg(src)
    if cfg is None:
        return False, "source missing config.json"
    arm_spec = ARMS[t["arm"]]
    # The fields the contract requires. curriculum_mode is NOT recorded in
    # P0-4/P0-5 config.json (the switch was off in those rounds), but
    # default="none" makes that consistent with both.
    required = {
        "loss_weighting": arm_spec["loss_weighting"],
        "momentum_mode": arm_spec["momentum_mode"],
        "fix_optim_grouping": 1,
        "role_sizes": ROLE_SIZES,
        "epoch": 50,
        "batch_size": 40960,
        "learning_rate": 0.01,
        "learning_rate_darts": 0.001,
        "weight_decay": 1e-05,
        "earlystop_patience": 10,
        "seed": t["seed"],
        "setting": t["setting"],
    }
    if arm_spec["momentum_mode"] == "per_domain_only":
        required["gamma0"] = arm_spec["gamma0"]
        required["gamma_max"] = arm_spec["gamma_max"]
    mismatches = []
    for k, v in required.items():
        if k not in cfg:
            mismatches.append(f"missing:{k}")
            continue
        # numeric fields: float compare with tolerance for 1e-05
        if isinstance(v, float):
            try:
                cv = float(cfg[k])
                if abs(cv - v) > 1e-9:
                    mismatches.append(f"{k}={cfg[k]}!={v}")
            except (TypeError, ValueError):
                mismatches.append(f"{k}={cfg[k]}!={v}")
        else:
            if cfg[k] != v:
                mismatches.append(f"{k}={cfg[k]}!={v}")
    if mismatches:
        return False, "; ".join(mismatches)
    return True, "ok"


def link_or_copy(src, dst):
    """Create a junction / symlink so the A-arm run dir appears under
    P1_2_curriculum without duplicating files. Falls back to copy on
    platforms where symlinks are restricted."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if os.name == "nt":
        # Use junction for directories (no admin required). If junction
        # fails (e.g. on this Windows version), fall back to copy.
        try:
            subprocess.run(["cmd", "/c", "mklink", "/J", dst, src],
                           check=True, capture_output=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    # symlink (POSIX) or fallback copy
    try:
        os.symlink(src, dst)
        return
    except (OSError, NotImplementedError):
        pass
    import shutil
    shutil.copytree(src, dst)


# ----------------------------------------------------------------------------
# Task construction + command
# ----------------------------------------------------------------------------

def build_tasks(include_A_grad_norm):
    tasks = []
    done = done_set()
    for arm in ("A0", "B0", "C0", "A1", "B1", "C1"):
        spec = ARMS[arm]
        for setting in SETTINGS:
            for seed in SEEDS:
                key = (arm, setting, seed)
                if key in done:
                    continue
                fresh_needed = (spec["source"] is None) or (
                    include_A_grad_norm and arm in ("A0", "A1"))
                tasks.append({
                    "kind": "joint", "ratio": RATIO,
                    "setting": setting, "arm": arm,
                    "seed": seed,
                    "role_sizes": ROLE_SIZES,
                    "momentum_mode": spec["momentum_mode"],
                    "curriculum_mode": spec["curriculum_mode"],
                    "kappa0": spec["kappa0"],
                    "loss_weighting": spec["loss_weighting"],
                    "gamma0": spec["gamma0"],
                    "gamma_max": spec["gamma_max"],
                    "fresh": fresh_needed,
                    "comp1": spec["momentum_mode"],
                    "method_or_domain": ("comp1" if spec["momentum_mode"] == "per_domain_only"
                                          else "baseline"),
                })
    return tasks


def task_cmd(t):
    common = ["--seed", str(t["seed"]), "--epoch", "50",
              "--batch_size", "40960",
              "--fix_optim_grouping", "1",
              "--role_setting", t["setting"],
              "--role_sizes", t["role_sizes"],
              "--run_dir", run_dir(t),
              "--stage_label", "P1-2",
              "--experiment_label", "curriculum_lod",
              "--loss_weighting", t["loss_weighting"],
              "--momentum_mode", t["momentum_mode"],
              "--curriculum_mode", t["curriculum_mode"],
              "--kappa0", str(t["kappa0"]),
              "--log_grad_norm", "1"]
    if t["momentum_mode"] == "per_domain_only":
        common += ["--gamma0", str(t["gamma0"]), "--gamma_max", str(t["gamma_max"])]
    cmd = [PY, "-u", P4] + common
    name = f"p1_2_{t['arm']}_{t['setting']}_seed{t['seed']}"
    return cmd, name


def write_config(t):
    d = run_dir(t)
    os.makedirs(d, exist_ok=True)
    cfg = {
        "kind": t["kind"],
        "ratio": t["ratio"],
        "setting": t["setting"],
        "arm": t["arm"],
        "comp1": t["comp1"],
        "curriculum_mode": t["curriculum_mode"],
        "kappa0": t["kappa0"],
        "loss_weighting": t["loss_weighting"],
        "method_or_domain": t["method_or_domain"],
        "seed": t["seed"],
        "role_sizes": t["role_sizes"],
        "momentum_mode": t["momentum_mode"],
        "loss_weighting_arg": t["loss_weighting"],
        "fix_optim_grouping": 1,
        "epoch": 50,
        "batch_size": 40960,
        "learning_rate": 0.01,
        "learning_rate_darts": 0.001,
        "weight_decay": 1e-05,
        "earlystop_patience": 10,
        "gamma0": t["gamma0"],
        "gamma_max": t["gamma_max"],
        "log_grad_norm": 1,
        "fresh_run": bool(t.get("fresh", False)),
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
        "kind": t["kind"],
        "ratio": t["ratio"],
        "setting": t["setting"],
        "arm": t["arm"],
        "comp1": t["comp1"],
        "curriculum_mode": t["curriculum_mode"],
        "kappa0": t["kappa0"],
        "loss_weighting": t["loss_weighting"],
        "method_or_domain": t["method_or_domain"],
        "seed": t["seed"],
        "status": "OK" if ok else "FAILED",
        "source": ARMS[t["arm"]]["source"] or "fresh",
        "reuse_check": "",
        "fit_sec": fit,
        "peak_mem_mb": peak,
        "wall_sec": f"{wall:.0f}",
        "per_dt_test_auc": per_dt,
        "best_epoch": best_ep if best_ep is not None else "",
        "stopped_epoch": stopped_ep if stopped_ep is not None else "",
        "best_val_auc": best_val if best_val is not None else "",
        "note": note,
    })
    return (f"{t['arm']} {t['setting']} seed{t['seed']}: "
            f"{'OK' if ok else 'FAILED'} auc={per_dt or 'NA'}")


def reuse_finalize(t, ok, reason):
    """Write an OK row for a reused A-arm run, copying the summary metrics
    from the source dir so downstream reports can pick it up."""
    src = source_path(t)
    if not ok:
        append_row(RAW, RAW_FIELDS, {
            "kind": t["kind"], "ratio": t["ratio"],
            "setting": t["setting"], "arm": t["arm"],
            "comp1": t["comp1"], "curriculum_mode": t["curriculum_mode"],
            "kappa0": t["kappa0"], "loss_weighting": t["loss_weighting"],
            "method_or_domain": t["method_or_domain"], "seed": t["seed"],
            "status": "REUSE_FAIL", "source": ARMS[t["arm"]]["source"] or "fresh",
            "reuse_check": reason, "fit_sec": "", "peak_mem_mb": "",
            "wall_sec": "", "per_dt_test_auc": "", "best_epoch": "",
            "stopped_epoch": "", "best_val_auc": "", "note": ""})
        return f"{t['arm']} {t['setting']} seed{t['seed']}: REUSE_FAIL ({reason})"
    dst = run_dir(t)
    link_or_copy(src, dst)
    best_ep, stopped_ep, best_val = parse_summary_csv(t)
    append_row(RAW, RAW_FIELDS, {
        "kind": t["kind"], "ratio": t["ratio"],
        "setting": t["setting"], "arm": t["arm"],
        "comp1": t["comp1"], "curriculum_mode": t["curriculum_mode"],
        "kappa0": t["kappa0"], "loss_weighting": t["loss_weighting"],
        "method_or_domain": t["method_or_domain"], "seed": t["seed"],
        "status": "OK_REUSED", "source": ARMS[t["arm"]]["source"] or "fresh",
        "reuse_check": reason, "fit_sec": "", "peak_mem_mb": "",
        "wall_sec": "", "per_dt_test_auc": "",
        "best_epoch": best_ep if best_ep is not None else "",
        "stopped_epoch": stopped_ep if stopped_ep is not None else "",
        "best_val_auc": best_val if best_val is not None else "",
        "note": "linked from source run dir (no retrain)"})
    return (f"{t['arm']} {t['setting']} seed{t['seed']}: REUSED "
            f"(linked {src})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_runs", type=int, default=10 ** 9)
    ap.add_argument("--dry_run", action="store_true",
                    help="print the task list and exit without launching")
    ap.add_argument("--include_A_grad_norm", type=int, default=0,
                    help="if 1, also re-run A0/A1 with --log_grad_norm 1 "
                         "so the grad_norm analysis covers all 6 arms. "
                         "Default 0: reuse A0/A1 from P0-4 / P0-5 (no "
                         "grad_norm for the A arms).")
    args = ap.parse_args()
    include_A = bool(args.include_A_grad_norm)
    CONCURRENCY = 2
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(P1_2_BASE, exist_ok=True)

    if args.dry_run:
        queue = build_tasks(include_A)
        print(f"dry-run: {len(queue)} P1-2 tasks queued "
              f"(include_A_grad_norm={include_A})", flush=True)
        # Report reuse decisions
        for t in queue:
            if not t["fresh"] and ARMS[t["arm"]]["source"] is not None:
                ok, reason = reuse_check(t)
                print(f"  reuse_check {t['arm']} {t['setting']} seed{t['seed']} "
                      f"-> {'OK' if ok else 'FAIL: '+reason}", flush=True)
        for i, t in enumerate(queue):
            d = run_dir(t)
            tag = "reuse" if not t["fresh"] else "fresh"
            print(f"  [{i:2d}] {tag:>5s} {t['arm']} {t['setting']:>9s} "
                  f"kappa0={t['kappa0']} comp1={t['comp1']} "
                  f"seed{t['seed']} -> {d}", flush=True)
        return

    acquire_lock()
    try:
        # Step 1: try reuse for A arms (skip if --include_A_grad_norm=1).
        for arm in ("A0", "A1"):
            spec = ARMS[arm]
            for setting in SETTINGS:
                for seed in SEEDS:
                    t = {
                        "kind": "joint", "ratio": RATIO, "arm": arm,
                        "setting": setting, "seed": seed,
                        "role_sizes": ROLE_SIZES,
                        "momentum_mode": spec["momentum_mode"],
                        "curriculum_mode": spec["curriculum_mode"],
                        "kappa0": spec["kappa0"],
                        "loss_weighting": spec["loss_weighting"],
                        "gamma0": spec["gamma0"], "gamma_max": spec["gamma_max"],
                        "comp1": spec["momentum_mode"],
                        "method_or_domain": ("comp1" if spec["momentum_mode"] == "per_domain_only"
                                              else "baseline"),
                    }
                    key = (arm, setting, seed)
                    if key in done_set():
                        continue
                    if include_A:
                        continue  # queued as fresh below
                    ok, reason = reuse_check(t)
                    if ok:
                        print(reuse_finalize(t, True, reason), flush=True)
                    else:
                        # Reuse fails -> schedule a fresh run
                        print(f"[p1_2] reuse fail for {arm} {setting} seed{seed}: "
                              f"{reason}; will queue fresh run", flush=True)

        # Step 2: queue remaining tasks (fresh runs + reuse-failed A arms)
        queue = build_tasks(include_A)[:args.max_runs]
        print(f"queued {len(queue)} fresh P1-2 tasks (include_A_grad_norm={include_A})",
              flush=True)

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
                log_path = os.path.join(LOGDIR, f"p1_2_{name}.log")
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
        print("P1_2_ORCHESTRATOR DONE", flush=True)
    finally:
        release_lock()


if __name__ == "__main__":
    main()