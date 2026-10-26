"""P5 single-domain oracle runs: per domain d, train the same backbone (MDMTRec)
on domain d's data only (as-published single-domain path, domain_num=1),
5 seeds. Written to results/p5/P5_oracle.csv. Resume-safe.
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
OUTDIR = os.path.join(BASE, "results", "p5")
LOGDIR = os.path.join(OUTDIR, "logs")
RAW = os.path.join(OUTDIR, "P5_oracle.csv")
os.makedirs(LOGDIR, exist_ok=True)

SEEDS = [2022, 2023, 2024, 2025, 2026]
CONFIGS = {
    "ml_orig": ["--downsample_ratio", "0"],
    "ml_r30": ["--downsample_ratio", "30.0"],
    "ml_r100": ["--downsample_ratio", "100.0"],
    "kuairand": ["--kuairand", "1", "--dataset_path", "D:/CODE/data/kuairand/processed"],
}


def existing():
    done = set()
    if os.path.exists(RAW):
        for r in csv.DictReader(open(RAW)):
            if r["status"] == "OK":
                done.add((r["config"], int(r["domain"]), int(r["seed"])))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="ml_orig,ml_r30,ml_r100,kuairand")
    args = ap.parse_args()
    configs = args.configs.split(",")
    done = existing()
    for config in configs:
        for domain in range(3):
            for seed in SEEDS:
                if (config, domain, seed) in done:
                    continue
                # KuaiRand oracle uses the single_domain flag only for
                # MovieLens (run_kuaishou also supports --domain, but the
                # kuairand path in p4_run uses single_domain for ML only);
                # for kuairand we train on the pre-filtered domain CSV via
                # the same as-published --domain mechanism of run_movielens
                # equivalent: here we reuse p4_run --single_domain for ML.
                if config == "kuairand":
                    # single-domain KuaiRand oracle: run_kuaishou.py's native
                    # as-published --domain mechanism (domain_num=1)
                    cmd = [PY, os.path.join(ROOT, "m3oe", "examples", "run_kuaishou.py"),
                           "--dataset_path", "D:/CODE/data/kuairand/processed",
                           "--seed", str(seed), "--epoch", "50", "--domain", str(domain)]
                    t0 = time.time()
                    res = subprocess.run(cmd, capture_output=True, text=True,
                                         cwd=os.path.join(ROOT, "m3oe", "examples"))
                    wall = time.time() - t0
                    out = res.stdout + res.stderr
                    with open(os.path.join(LOGDIR, f"oracle_{config}_d{domain}_{seed}.log"),
                              "w", encoding="utf-8", errors="replace") as f:
                        f.write(out)
                    dt = re.findall(r"domain:\s*0,\s*task:\s*(\d),\s*auc:\s*([\d.]+)", out)
                    status = "OK" if res.returncode == 0 and len(dt) == 2 else "FAILED"
                    exists = os.path.exists(RAW)
                    with open(RAW, "a", newline="") as f:
                        w = csv.writer(f)
                        if not exists:
                            w.writerow(["config", "domain", "seed", "status", "auc_d0t0",
                                        "auc_d0t1", "wall_sec", "note"])
                        vals = ["", ""]
                        for t_, a_ in dt:
                            vals[int(t_)] = a_
                        w.writerow([config, domain, seed, status, vals[0], vals[1],
                                    f"{wall:.0f}",
                                    out.strip().splitlines()[-1][:150]
                                    if res.returncode != 0 and out.strip() else ""])
                    print(f"oracle {config} d{domain} {seed}: {status}", flush=True)
                    continue
                cmd = [PY, RUN, "--seed", str(seed), "--epoch", "50",
                       "--single_domain", str(domain),
                       "--tag", f"p5_oracle_{config}_d{domain}", "--outdir", OUTDIR] \
                    + CONFIGS[config]
                t0 = time.time()
                res = subprocess.run(cmd, capture_output=True, text=True,
                                     cwd=os.path.join(ROOT, "m3oe", "examples"))
                wall = time.time() - t0
                out = res.stdout + res.stderr
                with open(os.path.join(LOGDIR, f"oracle_{config}_d{domain}_{seed}.log"),
                          "w", encoding="utf-8", errors="replace") as f:
                    f.write(out)
                m = re.search(r"FINAL per-\(d,t\) test auc: \[(.*?)\]", out)
                m2 = re.search(r"FINAL test auc: ([\d.]+)", out)
                status = "OK" if res.returncode == 0 else "FAILED"
                exists = os.path.exists(RAW)
                with open(RAW, "a", newline="") as f:
                    w = csv.writer(f)
                    if not exists:
                        w.writerow(["config", "domain", "seed", "status", "auc_d0t0",
                                    "auc_d0t1", "wall_sec", "note"])
                    vals = ["", ""]
                    if m:
                        parts = [x.replace("np.float64(", "").replace(")", "")
                                 for x in m.group(1).split(",")]
                        vals = [parts[0], parts[1]]
                    w.writerow([config, domain, seed, status, vals[0], vals[1],
                                f"{wall:.0f}",
                                out.strip().splitlines()[-1][:150]
                                if res.returncode != 0 and out.strip() else ""])
                print(f"oracle {config} d{domain} {seed}: {status} "
                      f"auc={m2.group(1) if m2 else 'NA'}", flush=True)
    print("ORACLE_DONE")


if __name__ == "__main__":
    main()
