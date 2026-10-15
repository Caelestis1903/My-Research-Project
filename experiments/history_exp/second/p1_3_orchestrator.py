"""P1-3 orchestrator: 24 fresh runs for the Matched Momentum Control design.

Per spec: H1 vs H2 disambiguation. Four arms (n=12=3 setting x 4 seed,
strong ratio=50, 150000/34500/3000, seeds 2022-2025, all
--loss_weighting sample):

  A: baseline w_sample      -> reuse P0-4 w_sample/baseline runs
  B: comp1 (domain γ_d)    -> reuse P0-5 w_sample/comp1 runs
  C: comp1 uniform γ*=0.9432 -> 12 fresh runs
  D: Adam β1*=0.9432        -> 12 fresh runs (structural control; not in
                              H1/H2 verdict)

Run dir layout (only C and D are produced here; A and B are reuse-only):
  experiments/P1_3_matched_momentum/
    P1_3_raw.csv
    logs/
    C/setting_<X>/seed<s>/
    D/setting_<X>/seed<s>/
      config.json
      stdout.log            (written by this orchestrator's run_logs/ dir)
      metrics_epoch.csv     (written by p4_run.py --run_dir; carries
                              test_auc per epoch, matches P0-4/P0-5 schema)
      metrics_summary.csv

Reuse contract: read P0-4/P0-5 source run dirs' config.json and verify
the hparams match the A/B arm contract. Verify (per spec §0a) that the
EMA update frequency in PerDomainMomentum.backward_accumulate is "every
step, all domains", which makes γ*=0.9432 the correct matched value
(frequency-weighted Σ f_d·τ_d = 17.6).

C arm contract: --loss_weighting sample --momentum_mode per_domain_only
  --gamma_source uniform --gamma_uniform 0.9432 (GAMMA_STAR).
D arm contract: --loss_weighting sample --momentum_mode none
  --beta1_shared 0.9432 (BETA1_STAR).
All other hparams mirror P0-5 task_cmd bit-for-bit.

Concurrency cap 2 (8GB VRAM laptop); stagger 30s; resume-safe (only OK
rows in master CSV count as done). A PID lockfile is written at startup
and removed on graceful exit.

Usage:
  python -u experiments/p1_3_orchestrator.py [--dry_run] [--max_runs N]
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

P1_3_BASE = os.path.abspath(os.path.join(BASE, "P1_3_matched_momentum"))
RAW = os.path.join(P1_3_BASE, "P1_3_raw.csv")
LOGDIR = os.path.join(BASE, "logs")
LOCKFILE = os.path.join(LOGDIR, "p1_3_orchestrator.lock")

# P0-4 / P0-5 source dirs (referenced for reuse verification)
P0_4_BASE = os.path.join(BASE, "P0_4_loss_weighting")
P0_5_BASE = os.path.join(BASE, "P0_5_comp1_x_weighting")

STAGGER_SEC = 30
SETTINGS = ["d0_large", "d1_large", "d2_large"]
SEEDS = [2022, 2023, 2024, 2025]

# Same contract as P0-2 ratio_50 / P0-4 / P0-5.
RATIO = 50
ROLE_SIZES = "150000,34500,3000"

# P1-3 matched-momentum constants (mirror p4_run module constants).
# γ* derivation: τ_d = 1/(1-γ_d) with γ_d from P0-5 (= [0.9, 0.977, 0.99]
# formula, p_d=[0.8,0.184,0.016]). Per-step update frequency f_d=1 for
# every d (backward_accumulate loops D times every step regardless of
# which domain's samples appear in the current batch). Then
# τ* = Σ f_d·τ_d = 0.8·10 + 0.184·43.478 + 0.016·100 = 17.6 steps, and
# γ* = 1 - 1/τ* = 0.9431818182. Adam β1* = γ* (matched EMA half-life).
GAMMA_STAR = 1.0 - 1.0 / 17.6   # ≈ 0.9431818182
BETA1_STAR = GAMMA_STAR
TAU_STAR = 17.6

# Arm spec: name -> (momentum_mode, gamma_source, gamma_uniform,
#   beta1_shared, source_label)
# `source_label` is None for fresh-only arms, or a reuse source key.
ARMS = {
    "A": {"momentum_mode": "none",
          "gamma_source": "formula", "gamma_uniform": None,
          "beta1_shared": None,
          "loss_weighting": "sample",
          "source": "p0_4_sample_baseline"},
    "B": {"momentum_mode": "per_domain_only",
          "gamma_source": "formula", "gamma_uniform": None,
          "beta1_shared": None,
          "loss_weighting": "sample",
          "source": "p0_5_sample_comp1"},
    "C": {"momentum_mode": "per_domain_only",
          "gamma_source": "uniform", "gamma_uniform": GAMMA_STAR,
          "beta1_shared": None,
          "loss_weighting": "sample",
          "source": None},
    "D": {"momentum_mode": "none",
          "gamma_source": "formula", "gamma_uniform": None,
          "beta1_shared": BETA1_STAR,
          "loss_weighting": "sample",
          "source": None},
}

RAW_FIELDS = ["kind", "ratio", "setting", "arm", "momentum_mode", "gamma_source",
              "gamma_uniform", "beta1_shared", "loss_weighting",
              "method_or_domain", "seed",
              "status", "source", "reuse_check", "fit_sec", "peak_mem_mb",
              "wall_sec", "per_dt_test_auc", "best_epoch", "stopped_epoch",
              "best_val_auc", "note"]


# ----------------------------------------------------------------------------
# PID lock (mirrors p0_5 / p1_2 orchestrators)
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
            print(f"[p1_3] another orchestrator is alive (pid={old_pid}); refusing "
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
            if r["status"] in ("OK", "OK_REUSED"):
                done.add((r["arm"], r["setting"], int(r["seed"])))
    return done


# ----------------------------------------------------------------------------
# Run dir / source path / reuse check
# ----------------------------------------------------------------------------

def run_dir(t):
    return os.path.join(P1_3_BASE, t["arm"], f"setting_{t['setting']}",
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


def _load_source_cfg(path):
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def reuse_check(t):
    """Verify a source run dir matches the A/B arm hparam contract
    bit-for-bit. Returns (ok: bool, reason: str)."""
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
        required["gamma0"] = 0.9
        required["gamma_max"] = 0.99
    mismatches = []
    for k, v in required.items():
        if k not in cfg:
            mismatches.append(f"missing:{k}")
            continue
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
    """Junction / symlink so the A/B-arm run dir appears under P1_3
    without duplicating files."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if os.name == "nt":
        try:
            subprocess.run(["cmd", "/c", "mklink", "/J", dst, src],
                           check=True, capture_output=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
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

def build_tasks():
    tasks = []
    done = done_set()
    for arm in ("A", "B", "C", "D"):
        spec = ARMS[arm]
        for setting in SETTINGS:
            for seed in SEEDS:
                key = (arm, setting, seed)
                if key in done:
                    continue
                fresh_needed = (spec["source"] is None)
                tasks.append({
                    "kind": "joint", "ratio": RATIO,
                    "setting": setting, "arm": arm, "seed": seed,
                    "role_sizes": ROLE_SIZES,
                    "momentum_mode": spec["momentum_mode"],
                    "gamma_source": spec["gamma_source"],
                    "gamma_uniform": spec["gamma_uniform"],
                    "beta1_shared": spec["beta1_shared"],
                    "loss_weighting": spec["loss_weighting"],
                    "fresh": fresh_needed,
                    "method_or_domain": ("comp1" if spec["momentum_mode"] == "per_domain_only"
                                          else "baseline"),
                })
    return tasks


def task_cmd(t):
    spec = ARMS[t["arm"]]
    common = ["--seed", str(t["seed"]), "--epoch", "50",
              "--batch_size", "40960",
              "--fix_optim_grouping", "1",
              "--role_setting", t["setting"],
              "--role_sizes", t["role_sizes"],
              "--run_dir", run_dir(t),
              "--stage_label", "P1-3",
              "--experiment_label", "matched_momentum",
              "--loss_weighting", t["loss_weighting"],
              "--momentum_mode", t["momentum_mode"]]
    if spec["momentum_mode"] == "per_domain_only":
        # C arm uses --gamma_source uniform (sets every domain to GAMMA_STAR).
        # B arm uses default --gamma_source formula (P0-5 path).
        common += ["--gamma_source", t["gamma_source"],
                   "--gamma0", "0.9", "--gamma_max", "0.99"]
        if t["gamma_source"] == "uniform":
            common += ["--gamma_uniform", f"{GAMMA_STAR:.6f}"]
    else:
        # A and D arms use momentum_mode=none.
        if t["beta1_shared"] is not None:
            common += ["--beta1_shared", f"{BETA1_STAR:.6f}"]
    cmd = [PY, "-u", P4] + common
    name = f"p1_3_{t['arm']}_{t['setting']}_seed{t['seed']}"
    return cmd, name


def write_config(t):
    d = run_dir(t)
    os.makedirs(d, exist_ok=True)
    spec = ARMS[t["arm"]]
    cfg = {
        "kind": t["kind"],
        "ratio": t["ratio"],
        "setting": t["setting"],
        "arm": t["arm"],
        "momentum_mode": t["momentum_mode"],
        "gamma_source": t["gamma_source"],
        "gamma_uniform": (float(t["gamma_uniform"]) if t["gamma_uniform"] is not None else None),
        "beta1_shared": (float(t["beta1_shared"]) if t["beta1_shared"] is not None else None),
        "loss_weighting": t["loss_weighting"],
        "method_or_domain": t["method_or_domain"],
        "seed": t["seed"],
        "role_sizes": t["role_sizes"],
        "fix_optim_grouping": 1,
        "epoch": 50,
        "batch_size": 40960,
        "learning_rate": 0.01,
        "learning_rate_darts": 0.001,
        "weight_decay": 1e-05,
        "earlystop_patience": 10,
        "gamma0": 0.9,
        "gamma_max": 0.99,
        "fresh_run": bool(t.get("fresh", False)),
        # P1-3 matched-momentum constants (mirror p4_run module globals so
        # downstream tools can audit the derivation without re-importing).
        "gamma_star": GAMMA_STAR,
        "beta1_star": BETA1_STAR,
        "tau_star": TAU_STAR,
        "stage_label": "P1-3",
        "experiment_label": "matched_momentum",
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
        "momentum_mode": t["momentum_mode"],
        "gamma_source": t["gamma_source"],
        "gamma_uniform": (f"{float(t['gamma_uniform']):.6f}"
                          if t["gamma_uniform"] is not None else ""),
        "beta1_shared": (f"{float(t['beta1_shared']):.6f}"
                          if t["beta1_shared"] is not None else ""),
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
    """Write an OK_REUSED row for an A/B-arm run, copying the summary
    metrics from the source dir so downstream reports can pick it up."""
    src = source_path(t)
    if not ok:
        append_row(RAW, RAW_FIELDS, {
            "kind": t["kind"], "ratio": t["ratio"],
            "setting": t["setting"], "arm": t["arm"],
            "momentum_mode": t["momentum_mode"],
            "gamma_source": t["gamma_source"],
            "gamma_uniform": (f"{float(t['gamma_uniform']):.6f}"
                              if t["gamma_uniform"] is not None else ""),
            "beta1_shared": (f"{float(t['beta1_shared']):.6f}"
                             if t["beta1_shared"] is not None else ""),
            "loss_weighting": t["loss_weighting"],
            "method_or_domain": t["method_or_domain"], "seed": t["seed"],
            "status": "REUSE_FAIL",
            "source": ARMS[t["arm"]]["source"] or "fresh",
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
        "momentum_mode": t["momentum_mode"],
        "gamma_source": t["gamma_source"],
        "gamma_uniform": (f"{float(t['gamma_uniform']):.6f}"
                          if t["gamma_uniform"] is not None else ""),
        "beta1_shared": (f"{float(t['beta1_shared']):.6f}"
                         if t["beta1_shared"] is not None else ""),
        "loss_weighting": t["loss_weighting"],
        "method_or_domain": t["method_or_domain"], "seed": t["seed"],
        "status": "OK_REUSED",
        "source": ARMS[t["arm"]]["source"] or "fresh",
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
    args = ap.parse_args()
    CONCURRENCY = 2
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(P1_3_BASE, exist_ok=True)

    if args.dry_run:
        queue = build_tasks()
        print(f"dry-run: {len(queue)} P1-3 tasks queued "
              f"(12 fresh C + 12 fresh D + reuse-only A/B)", flush=True)
        for i, t in enumerate(queue):
            d = run_dir(t)
            tag = "reuse" if not t["fresh"] else "fresh"
            extra = ""
            if t["arm"] == "C":
                extra = (f" gamma_source=uniform gamma_uniform="
                         f"{float(t['gamma_uniform']):.6f}")
            elif t["arm"] == "D":
                extra = f" beta1_shared={float(t['beta1_shared']):.6f}"
            print(f"  [{i:2d}] {tag:>5s} {t['arm']} {t['setting']:>9s} "
                  f"mm={t['momentum_mode']:>16s} seed{t['seed']:>4d}{extra}"
                  f"\n        -> {d}", flush=True)
        return

    acquire_lock()
    try:
        # Step 1: try reuse for A/B arms.
        for arm in ("A", "B"):
            spec = ARMS[arm]
            for setting in SETTINGS:
                for seed in SEEDS:
                    t = {
                        "kind": "joint", "ratio": RATIO, "arm": arm,
                        "setting": setting, "seed": seed,
                        "role_sizes": ROLE_SIZES,
                        "momentum_mode": spec["momentum_mode"],
                        "gamma_source": spec["gamma_source"],
                        "gamma_uniform": spec["gamma_uniform"],
                        "beta1_shared": spec["beta1_shared"],
                        "loss_weighting": spec["loss_weighting"],
                        "method_or_domain": ("comp1" if spec["momentum_mode"] == "per_domain_only"
                                              else "baseline"),
                    }
                    key = (arm, setting, seed)
                    if key in done_set():
                        continue
                    ok, reason = reuse_check(t)
                    if ok:
                        print(reuse_finalize(t, True, reason), flush=True)
                    else:
                        print(f"[p1_3] reuse fail for {arm} {setting} seed{seed}: "
                              f"{reason}; will queue fresh run", flush=True)

        # Step 2: queue remaining tasks (C, D fresh arms; reuse-failed A/B).
        queue = build_tasks()[:args.max_runs]
        n_fresh_c = sum(1 for t in queue if t["arm"] == "C" and t["fresh"])
        n_fresh_d = sum(1 for t in queue if t["arm"] == "D" and t["fresh"])
        print(f"queued {len(queue)} P1-3 tasks "
              f"({n_fresh_c} fresh C + {n_fresh_d} fresh D)", flush=True)

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
                log_path = os.path.join(LOGDIR, f"p1_3_{name}.log")
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
        print("P1_3_ORCHESTRATOR DONE", flush=True)
    finally:
        release_lock()


if __name__ == "__main__":
    main()
