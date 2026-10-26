"""P6 ablation orchestrator (7 groups + Adam-momentum comparison).

Runs on ONE data config (default ml_orig; pass --config to change).
Variants not already covered by the P5 matrix are run here at 5 seeds.
Resume-safe via results/p6/P6_raw.csv.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
RUN = os.path.join(BASE, "p4_run.py")
OUTDIR = os.path.join(BASE, "results", "p6")
LOGDIR = os.path.join(OUTDIR, "logs")
RAW = os.path.join(OUTDIR, "P6_raw.csv")
os.makedirs(LOGDIR, exist_ok=True)

FG = ["--fix_optim_grouping", "1"]
C2 = ["--curriculum_mode", "domain_conditioned", "--kappa0", "0.5"] + FG
C1 = ["--momentum_mode", "per_domain_only"] + FG

# (name, p4_run flags) — P5-covered arms (baseline, fixed073, comp1, comp2,
# comp1_comp2, comp123) are NOT re-run here; the report joins them from P5.
VARIANTS = [
    # group 1: momentum tiering
    ("g1_uniform_gamma", ["--momentum_mode", "per_domain_only", "--gamma_uniform", "1"] + FG),
    # group 2: gamma source
    ("g2_manual_gamma", ["--momentum_mode", "per_domain_only", "--gamma_source", "manual",
                         "--gamma_manual", "0.95,0.90,0.90"] + FG),
    ("g2_snr_gamma", ["--momentum_mode", "per_domain_only", "--gamma_source", "snr"] + FG),
    # group 3/4: beta granularity, curriculum off (per-domain beta, kappa=0)
    ("g3_beta_perdomain_nocur", ["--curriculum_mode", "domain_conditioned", "--kappa0", "0.0"] + FG),
    # group 5: reversed curriculum
    ("g5_reverse_curriculum", ["--curriculum_mode", "domain_conditioned", "--kappa0", "-0.5"] + FG),
    # group 6: orthogonality (comp3 arms)
    ("g6_comp3", ["--arch_grad_ema", "1", "--arch_update_every", "5"] + FG),
    ("g6_comp1_comp3", C1 + ["--arch_grad_ema", "1", "--arch_update_every", "5"]),
    ("g6_comp2_comp3", C2 + ["--arch_grad_ema", "1", "--arch_update_every", "5"]),
    # Adam momentum relation
    ("g8_stacked", ["--momentum_mode", "per_domain_stacked"] + FG),
    # group 7: kappa0 sensitivity (0.5 = P5 comp1_comp2 reference)
    ("g7_kappa0_0.1", ["--curriculum_mode", "domain_conditioned", "--kappa0", "0.1"] + C1),
    ("g7_kappa0_0.25", ["--curriculum_mode", "domain_conditioned", "--kappa0", "0.25"] + C1),
    ("g7_kappa0_1.0", ["--curriculum_mode", "domain_conditioned", "--kappa0", "1.0"] + C1),
    ("g7_kappa0_2.0", ["--curriculum_mode", "domain_conditioned", "--kappa0", "2.0"] + C1),
    # group 7: gamma0 sensitivity (0.9 = P5 comp1 reference)
    ("g7_gamma0_0.8", ["--momentum_mode", "per_domain_only", "--gamma0", "0.8"] + FG),
    ("g7_gamma0_0.85", ["--momentum_mode", "per_domain_only", "--gamma0", "0.85"] + FG),
    ("g7_gamma0_0.95", ["--momentum_mode", "per_domain_only", "--gamma0", "0.95"] + FG),
    ("g7_gamma0_0.99", ["--momentum_mode", "per_domain_only", "--gamma0", "0.99"] + FG),
]

CONFIG_FLAGS = {
    "ml_orig": ["--downsample_ratio", "0"],
    "ml_r30": ["--downsample_ratio", "30.0"],
    "ml_r100": ["--downsample_ratio", "100.0"],
}


def existing_keys():
    done = set()
    if os.path.exists(RAW):
        for r in csv.DictReader(open(RAW)):
            if r["status"] == "OK":
                done.add((r["variant"], int(r["seed"])))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml_orig", choices=list(CONFIG_FLAGS))
    ap.add_argument("--seeds", default="2022,2023,2024,2025,2026")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    done = existing_keys()

    for name, flags in VARIANTS:
        for seed in seeds:
            if (name, seed) in done:
                continue
            cmd = [PY, RUN, "--seed", str(seed), "--epoch", "50",
                   "--tag", f"p6_{name}", "--outdir", OUTDIR] + CONFIG_FLAGS[args.config] + flags
            t0 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 cwd=os.path.join(ROOT, "m3oe", "examples"))
            wall = time.time() - t0
            out = res.stdout + res.stderr
            with open(os.path.join(LOGDIR, f"{name}_{seed}.log"), "w", encoding="utf-8",
                      errors="replace") as f:
                f.write(out)
            m = re.search(r"FINAL test auc: ([\d.]+)", out)
            m2 = re.search(r"FINAL per-\(d,t\) test auc: \[(.*?)\]", out)
            m3 = re.search(r"fit wall time: ([\d.]+)s \| peak cuda mem: ([\d.]+)", out)
            status = "OK" if res.returncode == 0 else "FAILED"
            exists = os.path.exists(RAW)
            with open(RAW, "a", newline="") as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(["variant", "seed", "status", "test_auc_mean", "per_dt",
                                "fit_sec", "peak_mem_mb", "wall_sec", "note"])
                w.writerow([name, seed, status,
                            m.group(1) if m else "",
                            m2.group(1) if m2 else "",
                            m3.group(1) if m3 else "", m3.group(2) if m3 else "",
                            f"{wall:.0f}",
                            out.strip().splitlines()[-1][:150] if res.returncode != 0 and out.strip() else ""])
            print(f"{name} {seed}: {status} auc={m.group(1) if m else 'NA'}", flush=True)
    print("P6_ORCHESTRATOR_DONE")


if __name__ == "__main__":
    main()
