"""Extract proper 5x2 per-(d,t) AUC matrices from P7_backbones.csv.

The CSV was created by DictWriter using the first row's fieldnames, so D=5
rows have 15 physical fields but the header only knows 11 columns. The extra
fields (cols 12-15) hold auc_d3t0, auc_d3t1, auc_d4t0, auc_d4t1.
"""
import csv
import os
from collections import defaultdict

CSV = r"D:\CODE\phase0-3\experiments\results\p7\P7_backbones.csv"
OUT = r"D:\CODE\phase0-3\experiments\results\p7\P7_d5_per_dt_matrix.csv"

# Read raw rows manually because DictReader would only see 11 fields
rows_by_key = defaultdict(list)
with open(CSV, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    # header: backbone, domain_version, momentum, seed, test_auc_mean, auc_d0t0..auc_d2t1
    for r in reader:
        if len(r) < 11:
            continue
        bb, dv, mom, sd, mean_auc = r[0], r[1], r[2], r[3], r[4]
        if dv != "five":
            continue
        # cols 5..10 (0-indexed) = auc_d0t0..auc_d2t1, cols 11..14 = auc_d3t0..auc_d4t1
        try:
            cells = [float(x) for x in r[5:15]]
        except ValueError:
            continue
        if len(cells) < 10:
            continue
        rows_by_key[(bb, mom)].append((int(sd), cells))

# For each (backbone, momentum), compute per-(d,t) mean across seeds
out_rows = []
for (bb, mom), sds in sorted(rows_by_key.items()):
    print(f"\n{bb} {mom} (n={len(sds)})")
    means = []
    for dt in range(10):
        vals = [cells[dt] for _, cells in sds]
        m = sum(vals) / len(vals)
        means.append(m)
    cells_str = "  ".join(f"{m:.6f}" for m in means)
    print(f"  d0t0 d0t1 d1t0 d1t1 d2t0 d2t1 d3t0 d3t1 d4t0 d4t1")
    print(f"  {cells_str}")
    out_rows.append({"backbone": bb, "momentum": mom, "n": len(sds),
                     **{f"auc_d{d}t{t}": means[d*2+t] for d in range(5) for t in range(2)}})

# write a tidy CSV
with open(OUT, "w", newline="") as f:
    if out_rows:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
print(f"\n[csv] wrote {OUT}")
