"""P1 equivalence evidence:
1) instrument=0 run vs as-published P0 log  : per-epoch val AUC stream + final test AUC
2) instrument=1 run vs instrument=0 run     : per-epoch val AUC stream + final test AUC
3) per-step loss streams (0 vs 1)           : max abs / rel deviation
"""
import re
import sys

BASE = "D:/CODE/phase0-3/experiments"


def val_auc_stream(path):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"epoch:\s*(\d+)\s+validation:\s+auc:\s*([\d.]+)", line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def final_test_auc(path):
    auc = ll = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"test auc:\s*([\d.]+)\s*\|\s*test logloss:\s*([\d.]+)", line)
        if m:
            auc, ll = float(m.group(1)), float(m.group(2))
    return auc, ll


def read_stream(p):
    return [float(x) for x in open(p) if x.strip()]


p0_log = f"{BASE}/p0_seed2022.log"
eq0_log = f"{BASE}/results/p1/eq0.log"
eq1_log = f"{BASE}/results/p1/s2022_b40960.log"

ok = True

# --- 1) eq0 vs P0 ---
a, b = val_auc_stream(p0_log), val_auc_stream(eq0_log)
n = min(len(a), len(b))
match = len(a) == len(b) and all(x[1] == y[1] for x, y in zip(a[:n], b[:n]))
ta, tll = final_test_auc(p0_log)
tb, tllb = final_test_auc(eq0_log)
print(f"[1] instrument=0 vs P0 log: epochs {len(a)} vs {len(b)}, "
      f"val AUC streams identical: {match}; final test AUC {ta} vs {tb} identical: {ta == tb}")
ok &= match and (ta == tb)

# --- 2) eq1 vs eq0 ---
a, b = val_auc_stream(eq0_log), val_auc_stream(eq1_log)
n = min(len(a), len(b))
match = len(a) == len(b) and all(x[1] == y[1] for x, y in zip(a[:n], b[:n]))
ta, _ = final_test_auc(eq0_log)
tb, _ = final_test_auc(eq1_log)
print(f"[2] instrument=1 vs instrument=0: epochs {len(a)} vs {len(b)}, "
      f"val AUC streams identical: {match}; final test AUC {ta} vs {tb} identical: {ta == tb}")
ok &= match and (ta == tb)

# --- 3) loss streams ---
try:
    s0 = read_stream(f"{BASE}/results/p1/loss_stream_eq0.txt")
    s1 = read_stream(f"{BASE}/results/p1/loss_stream_s2022_b40960.txt")
    n = min(len(s0), len(s1))
    max_abs = max(abs(x - y) for x, y in zip(s0[:n], s1[:n]))
    max_rel = max(abs(x - y) / max(abs(x), 1e-12) for x, y in zip(s0[:n], s1[:n]))
    print(f"[3] per-step loss stream: n0={len(s0)} n1={len(s1)} "
          f"max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e} "
          f"(tolerance 1e-6: eq0 sums 6 float32 items in float64, eq1 reads the float32 sum -> "
          f"float32 rounding of the summation itself is ~1e-7)")
    ok &= n == min(len(s0), len(s1)) and max_rel < 1e-6
except FileNotFoundError as e:
    print(f"[3] loss stream missing: {e}")
    ok = False

print("EQUIVALENCE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
