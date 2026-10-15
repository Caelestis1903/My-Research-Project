# Raw Experiment Report — P0-3

## 1. Experiment metadata

- Stage: P0-3 (MMoE transfer re-run of the P0-2 ratio_1 + ratio_50 cells; fresh runs on the same cached data so P0-3 pairs M3OE (MDMTRec) and MMoE rows cell-by-cell)
- Dataset: ml-1m (MDMTRec domain configuration: d0=d1=d2 in ml-1m; mappings rotate per setting)
- Model: MMoE backbone (from m3oe/examples via p4_run.py with `--backbone mmoe`)
- Method: joint (baseline, comp1) and oracle (single_domain=0/1/2); all P0-3 runs pass `--fix_optim_grouping 1`
- comp1 extra: `--momentum_mode per_domain_only`
- Seeds: 2022, 2023, 2024, 2025 (4 seeds per arm)
- Joint arms: ratio_1 (balanced, 1 setting) + ratio_50 (strong, 3 settings) = 4 (setting, ratio) combos × 2 methods × 4 seeds = 32 joint runs
- Oracle arms: same 4 combos × 3 domains × 4 seeds = 48 oracle runs
- Fresh runs: 32 + 48 = 80 (no reuse from P0-1 / P0-2; only M3OE backbone differs and is reused at aggregator stage)
- Total rows in P0_3_raw.csv: 80 (MMoE only; aggregator joins M3OE rows from P0-2 on the fly)
- Ratio role_sizes contracts (joint totals = 187,500):
  - balanced (ratio=1): 62500,62500,62500 (L,M,S)
  - strong (ratio=50): 150000,34500,3000
- Code: experiments/p0_3_orchestrator.py + experiments/p4_run.py (orchestrator); experiments/p0_3_report.py (aggregator)
- GPU: NVIDIA GeForce RTX 4060 Laptop (8 GiB)
- Start time: 2026-09-09 04:27 (orchestrator launch; lock mtime)
- End time: 2026-09-09 05:29 (last OK row written; P0_3_ORCHESTRATOR DONE)
- Wall-clock total: ~62 minutes (sum of per-run wall_sec = 7025 s; sum of fit_sec = 6518 s; mean fit 81.5 s; peak cuda mem mean 670 MiB, max 1000 MiB)
- Concurrency: max 2 p4_run children simultaneously; stagger 30s
- Resume-safe: orchestrator reads existing P0_3_raw.csv on start and skips (ratio,setting,method_or_domain,seed) tuples already marked OK

## 2. Completion status

All 80 unique tasks completed OK (no FAIL/CRASH rows in master CSV).

Master CSV `P0_3_raw.csv` has 80 OK rows (one per unique task).

| run | status | checkpoint | source | log |
|---|---|---|---|---|
| mmoe_balanced_d0_large_baseline_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_baseline_seed2022.log |
| mmoe_balanced_d0_large_baseline_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_baseline_seed2023.log |
| mmoe_balanced_d0_large_baseline_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_baseline_seed2024.log |
| mmoe_balanced_d0_large_baseline_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_baseline_seed2025.log |
| mmoe_balanced_d0_large_comp1_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_comp1_seed2022.log |
| mmoe_balanced_d0_large_comp1_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_comp1_seed2023.log |
| mmoe_balanced_d0_large_comp1_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_comp1_seed2024.log |
| mmoe_balanced_d0_large_comp1_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_comp1_seed2025.log |
| mmoe_balanced_d0_large_oracle_d0_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_oracle_d0_seed2022.log |
| mmoe_balanced_d0_large_oracle_d0_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_oracle_d0_seed2023.log |
| mmoe_balanced_d0_large_oracle_d0_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_oracle_d0_seed2024.log |
| mmoe_balanced_d0_large_oracle_d0_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_oracle_d0_seed2025.log |
| mmoe_balanced_d0_large_oracle_d1_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_oracle_d1_seed2022.log |
| mmoe_balanced_d0_large_oracle_d1_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_oracle_d1_seed2023.log |
| mmoe_balanced_d0_large_oracle_d1_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_oracle_d1_seed2024.log |
| mmoe_balanced_d0_large_oracle_d1_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_oracle_d1_seed2025.log |
| mmoe_balanced_d0_large_oracle_d2_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_oracle_d2_seed2022.log |
| mmoe_balanced_d0_large_oracle_d2_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_oracle_d2_seed2023.log |
| mmoe_balanced_d0_large_oracle_d2_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_oracle_d2_seed2024.log |
| mmoe_balanced_d0_large_oracle_d2_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_balanced_d0_large_oracle_d2_seed2025.log |
| mmoe_strong_d0_large_baseline_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_baseline_seed2022.log |
| mmoe_strong_d0_large_baseline_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_baseline_seed2023.log |
| mmoe_strong_d0_large_baseline_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_baseline_seed2024.log |
| mmoe_strong_d0_large_baseline_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_baseline_seed2025.log |
| mmoe_strong_d0_large_comp1_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_comp1_seed2022.log |
| mmoe_strong_d0_large_comp1_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_comp1_seed2023.log |
| mmoe_strong_d0_large_comp1_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_comp1_seed2024.log |
| mmoe_strong_d0_large_comp1_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_comp1_seed2025.log |
| mmoe_strong_d0_large_oracle_d0_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_oracle_d0_seed2022.log |
| mmoe_strong_d0_large_oracle_d0_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_oracle_d0_seed2023.log |
| mmoe_strong_d0_large_oracle_d0_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_oracle_d0_seed2024.log |
| mmoe_strong_d0_large_oracle_d0_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_oracle_d0_seed2025.log |
| mmoe_strong_d0_large_oracle_d1_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_oracle_d1_seed2022.log |
| mmoe_strong_d0_large_oracle_d1_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_oracle_d1_seed2023.log |
| mmoe_strong_d0_large_oracle_d1_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_oracle_d1_seed2024.log |
| mmoe_strong_d0_large_oracle_d1_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_oracle_d1_seed2025.log |
| mmoe_strong_d0_large_oracle_d2_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_oracle_d2_seed2022.log |
| mmoe_strong_d0_large_oracle_d2_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_oracle_d2_seed2023.log |
| mmoe_strong_d0_large_oracle_d2_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_oracle_d2_seed2024.log |
| mmoe_strong_d0_large_oracle_d2_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d0_large_oracle_d2_seed2025.log |
| mmoe_strong_d1_large_baseline_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_baseline_seed2022.log |
| mmoe_strong_d1_large_baseline_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_baseline_seed2023.log |
| mmoe_strong_d1_large_baseline_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_baseline_seed2024.log |
| mmoe_strong_d1_large_baseline_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_baseline_seed2025.log |
| mmoe_strong_d1_large_comp1_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_comp1_seed2022.log |
| mmoe_strong_d1_large_comp1_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_comp1_seed2023.log |
| mmoe_strong_d1_large_comp1_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_comp1_seed2024.log |
| mmoe_strong_d1_large_comp1_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_comp1_seed2025.log |
| mmoe_strong_d1_large_oracle_d0_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_oracle_d0_seed2022.log |
| mmoe_strong_d1_large_oracle_d0_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_oracle_d0_seed2023.log |
| mmoe_strong_d1_large_oracle_d0_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_oracle_d0_seed2024.log |
| mmoe_strong_d1_large_oracle_d0_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_oracle_d0_seed2025.log |
| mmoe_strong_d1_large_oracle_d1_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_oracle_d1_seed2022.log |
| mmoe_strong_d1_large_oracle_d1_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_oracle_d1_seed2023.log |
| mmoe_strong_d1_large_oracle_d1_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_oracle_d1_seed2024.log |
| mmoe_strong_d1_large_oracle_d1_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_oracle_d1_seed2025.log |
| mmoe_strong_d1_large_oracle_d2_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_oracle_d2_seed2022.log |
| mmoe_strong_d1_large_oracle_d2_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_oracle_d2_seed2023.log |
| mmoe_strong_d1_large_oracle_d2_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_oracle_d2_seed2024.log |
| mmoe_strong_d1_large_oracle_d2_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d1_large_oracle_d2_seed2025.log |
| mmoe_strong_d2_large_baseline_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_baseline_seed2022.log |
| mmoe_strong_d2_large_baseline_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_baseline_seed2023.log |
| mmoe_strong_d2_large_baseline_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_baseline_seed2024.log |
| mmoe_strong_d2_large_baseline_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_baseline_seed2025.log |
| mmoe_strong_d2_large_comp1_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_comp1_seed2022.log |
| mmoe_strong_d2_large_comp1_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_comp1_seed2023.log |
| mmoe_strong_d2_large_comp1_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_comp1_seed2024.log |
| mmoe_strong_d2_large_comp1_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_comp1_seed2025.log |
| mmoe_strong_d2_large_oracle_d0_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_oracle_d0_seed2022.log |
| mmoe_strong_d2_large_oracle_d0_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_oracle_d0_seed2023.log |
| mmoe_strong_d2_large_oracle_d0_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_oracle_d0_seed2024.log |
| mmoe_strong_d2_large_oracle_d0_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_oracle_d0_seed2025.log |
| mmoe_strong_d2_large_oracle_d1_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_oracle_d1_seed2022.log |
| mmoe_strong_d2_large_oracle_d1_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_oracle_d1_seed2023.log |
| mmoe_strong_d2_large_oracle_d1_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_oracle_d1_seed2024.log |
| mmoe_strong_d2_large_oracle_d1_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_oracle_d1_seed2025.log |
| mmoe_strong_d2_large_oracle_d2_seed2022 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_oracle_d2_seed2022.log |
| mmoe_strong_d2_large_oracle_d2_seed2023 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_oracle_d2_seed2023.log |
| mmoe_strong_d2_large_oracle_d2_seed2024 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_oracle_d2_seed2024.log |
| mmoe_strong_d2_large_oracle_d2_seed2025 | OK | metrics_summary.csv | P0-3 | experiments/logs/p0_3_mmoe_strong_d2_large_oracle_d2_seed2025.log |

## 3. Raw metrics

Each cell value is the FINAL per-(domain,task) test AUC at best_epoch.
`per_dt_test_auc` layout: `[d0t0, d1t0, d2t0, d0t1, d1t1, d2t1]` (task-major, idx = t*3 + d) for joint runs; `[t0, t1]` for oracle runs.

### 3.1 Joint runs (32 total)

| ratio | setting | arm | seed | d0t0 | d1t0 | d2t0 | d0t1 | d1t1 | d2t1 | best_epoch | best_val_auc | fit_sec | peak_mem_mb | wall_sec | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | d0_large | baseline | 2022 | 0.738467 | 0.740493 | 0.703907 | 0.767468 | 0.762744 | 0.723205 | 6 | 0.737209 | 125.8 | 824 | 135 | P0-3 |
| 1 | d0_large | baseline | 2023 | 0.735413 | 0.744304 | 0.714783 | 0.752360 | 0.764402 | 0.734726 | 6 | 0.735519 | 125.6 | 824 | 135 | P0-3 |
| 1 | d0_large | baseline | 2024 | 0.753284 | 0.750454 | 0.706142 | 0.762722 | 0.770215 | 0.727276 | 5 | 0.738809 | 118.6 | 824 | 125 | P0-3 |
| 1 | d0_large | baseline | 2025 | 0.740566 | 0.747470 | 0.704895 | 0.751061 | 0.768966 | 0.725355 | 4 | 0.737948 | 110.0 | 824 | 115 | P0-3 |
| 1 | d0_large | comp1 | 2022 | 0.738342 | 0.744316 | 0.708246 | 0.762712 | 0.755925 | 0.729416 | 5 | 0.736903 | 129.6 | 999 | 135 | P0-3 |
| 1 | d0_large | comp1 | 2023 | 0.738639 | 0.745316 | 0.714964 | 0.758702 | 0.763618 | 0.733338 | 5 | 0.736024 | 129.7 | 999 | 135 | P0-3 |
| 1 | d0_large | comp1 | 2024 | 0.753705 | 0.752320 | 0.707790 | 0.762523 | 0.775799 | 0.729981 | 4 | 0.740885 | 121.1 | 999 | 130 | P0-3 |
| 1 | d0_large | comp1 | 2025 | 0.739194 | 0.749437 | 0.701001 | 0.756165 | 0.769402 | 0.725419 | 4 | 0.739030 | 120.1 | 999 | 125 | P0-3 |
| 50 | d0_large | baseline | 2022 | 0.740493 | 0.735291 | 0.658125 | 0.756113 | 0.749369 | 0.644223 | 3 | 0.711145 | 103.8 | 824 | 110 | P0-3 |
| 50 | d0_large | baseline | 2023 | 0.744964 | 0.724421 | 0.650898 | 0.759049 | 0.747560 | 0.670250 | 3 | 0.712893 | 103.1 | 824 | 110 | P0-3 |
| 50 | d0_large | baseline | 2024 | 0.738897 | 0.727569 | 0.647793 | 0.748987 | 0.749386 | 0.683263 | 2 | 0.719541 | 96.2 | 824 | 105 | P0-3 |
| 50 | d0_large | baseline | 2025 | 0.744334 | 0.717543 | 0.670640 | 0.756756 | 0.755370 | 0.687260 | 2 | 0.722908 | 96.1 | 824 | 105 | P0-3 |
| 50 | d0_large | comp1 | 2022 | 0.748052 | 0.728883 | 0.654356 | 0.764379 | 0.754897 | 0.673346 | 4 | 0.714720 | 121.0 | 1000 | 130 | P0-3 |
| 50 | d0_large | comp1 | 2023 | 0.747321 | 0.734677 | 0.664393 | 0.761627 | 0.749349 | 0.684077 | 3 | 0.718625 | 112.1 | 1000 | 120 | P0-3 |
| 50 | d0_large | comp1 | 2024 | 0.745167 | 0.732630 | 0.668716 | 0.757802 | 0.743922 | 0.686675 | 3 | 0.723586 | 112.4 | 1000 | 120 | P0-3 |
| 50 | d0_large | comp1 | 2025 | 0.747747 | 0.735562 | 0.673976 | 0.764970 | 0.759228 | 0.687711 | 4 | 0.724949 | 120.7 | 1000 | 130 | P0-3 |
| 50 | d1_large | baseline | 2022 | 0.668501 | 0.738246 | 0.684285 | 0.680415 | 0.764915 | 0.706436 | 5 | 0.699804 | 118.6 | 824 | 125 | P0-3 |
| 50 | d1_large | baseline | 2023 | 0.686496 | 0.727044 | 0.700190 | 0.713661 | 0.767726 | 0.715722 | 2 | 0.709876 | 95.3 | 824 | 100 | P0-3 |
| 50 | d1_large | baseline | 2024 | 0.649674 | 0.752975 | 0.687502 | 0.678721 | 0.773451 | 0.710864 | 6 | 0.710627 | 125.0 | 824 | 130 | P0-3 |
| 50 | d1_large | baseline | 2025 | 0.670566 | 0.739898 | 0.684719 | 0.699137 | 0.764346 | 0.719592 | 2 | 0.713001 | 96.4 | 824 | 105 | P0-3 |
| 50 | d1_large | comp1 | 2022 | 0.686964 | 0.750304 | 0.695063 | 0.721152 | 0.773694 | 0.719261 | 4 | 0.712502 | 121.2 | 999 | 130 | P0-3 |
| 50 | d1_large | comp1 | 2023 | 0.691995 | 0.757903 | 0.717428 | 0.719151 | 0.776307 | 0.739981 | 4 | 0.717893 | 119.6 | 999 | 125 | P0-3 |
| 50 | d1_large | comp1 | 2024 | 0.680110 | 0.756262 | 0.702204 | 0.689254 | 0.776100 | 0.717211 | 4 | 0.713365 | 120.8 | 999 | 130 | P0-3 |
| 50 | d1_large | comp1 | 2025 | 0.690457 | 0.752524 | 0.704956 | 0.701072 | 0.771614 | 0.726629 | 3 | 0.717037 | 112.3 | 999 | 120 | P0-3 |
| 50 | d2_large | baseline | 2022 | 0.721941 | 0.650207 | 0.697848 | 0.744075 | 0.704219 | 0.726355 | 5 | 0.700619 | 118.4 | 824 | 125 | P0-3 |
| 50 | d2_large | baseline | 2023 | 0.725269 | 0.658944 | 0.718907 | 0.739534 | 0.715225 | 0.734599 | 5 | 0.710022 | 117.7 | 824 | 125 | P0-3 |
| 50 | d2_large | baseline | 2024 | 0.732711 | 0.672099 | 0.718302 | 0.738431 | 0.689488 | 0.729124 | 5 | 0.704655 | 117.9 | 824 | 125 | P0-3 |
| 50 | d2_large | baseline | 2025 | 0.723169 | 0.671208 | 0.695393 | 0.733978 | 0.707658 | 0.720295 | 3 | 0.706795 | 103.9 | 824 | 110 | P0-3 |
| 50 | d2_large | comp1 | 2022 | 0.724437 | 0.681911 | 0.722690 | 0.745805 | 0.718539 | 0.737865 | 4 | 0.708427 | 120.8 | 999 | 130 | P0-3 |
| 50 | d2_large | comp1 | 2023 | 0.733993 | 0.695737 | 0.729096 | 0.745168 | 0.725711 | 0.745129 | 4 | 0.717002 | 122.0 | 999 | 130 | P0-3 |
| 50 | d2_large | comp1 | 2024 | 0.730795 | 0.701613 | 0.719910 | 0.745870 | 0.709694 | 0.739516 | 4 | 0.710528 | 120.8 | 999 | 130 | P0-3 |
| 50 | d2_large | comp1 | 2025 | 0.730748 | 0.678854 | 0.718938 | 0.738827 | 0.721220 | 0.745177 | 4 | 0.713811 | 120.8 | 999 | 130 | P0-3 |

### 3.2 Oracle runs (48 total)

| ratio | setting | oracle_domain | seed | t0 | t1 | best_epoch | best_val_auc | fit_sec | peak_mem_mb | wall_sec | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | d0_large | oracle_d0 | 2022 | 0.727746 | 0.749153 | 15 | 0.734909 | 64.0 | 664 | 70 | P0-3 |
| 1 | d0_large | oracle_d0 | 2023 | 0.728577 | 0.742533 | 14 | 0.735037 | 61.9 | 664 | 65 | P0-3 |
| 1 | d0_large | oracle_d0 | 2024 | 0.735414 | 0.748935 | 10 | 0.740006 | 51.7 | 664 | 55 | P0-3 |
| 1 | d0_large | oracle_d0 | 2025 | 0.725228 | 0.746984 | 11 | 0.732925 | 53.6 | 664 | 60 | P0-3 |
| 1 | d0_large | oracle_d1 | 2022 | 0.720859 | 0.739939 | 18 | 0.727977 | 84.2 | 664 | 90 | P0-3 |
| 1 | d0_large | oracle_d1 | 2023 | 0.729451 | 0.749317 | 14 | 0.735432 | 71.6 | 664 | 75 | P0-3 |
| 1 | d0_large | oracle_d1 | 2024 | 0.740910 | 0.748396 | 12 | 0.740995 | 66.0 | 664 | 70 | P0-3 |
| 1 | d0_large | oracle_d1 | 2025 | 0.734930 | 0.757098 | 15 | 0.744799 | 75.0 | 664 | 80 | P0-3 |
| 1 | d0_large | oracle_d2 | 2022 | 0.701622 | 0.718821 | 14 | 0.708111 | 67.8 | 664 | 75 | P0-3 |
| 1 | d0_large | oracle_d2 | 2023 | 0.706660 | 0.729153 | 10 | 0.711263 | 56.7 | 664 | 60 | P0-3 |
| 1 | d0_large | oracle_d2 | 2024 | 0.696526 | 0.714169 | 11 | 0.700869 | 57.8 | 664 | 65 | P0-3 |
| 1 | d0_large | oracle_d2 | 2025 | 0.697968 | 0.718564 | 6 | 0.706091 | 46.3 | 664 | 50 | P0-3 |
| 50 | d0_large | oracle_d0 | 2022 | 0.751636 | 0.767068 | 14 | 0.758768 | 94.8 | 664 | 100 | P0-3 |
| 50 | d0_large | oracle_d0 | 2023 | 0.752393 | 0.763832 | 9 | 0.759497 | 75.9 | 664 | 80 | P0-3 |
| 50 | d0_large | oracle_d0 | 2024 | 0.754840 | 0.763940 | 8 | 0.760105 | 72.9 | 664 | 80 | P0-3 |
| 50 | d0_large | oracle_d0 | 2025 | 0.748319 | 0.764887 | 4 | 0.758176 | 56.2 | 664 | 60 | P0-3 |
| 50 | d0_large | oracle_d1 | 2022 | 0.697716 | 0.718667 | 27 | 0.704965 | 94.8 | 567 | 100 | P0-3 |
| 50 | d0_large | oracle_d1 | 2023 | 0.722928 | 0.738522 | 9 | 0.730935 | 50.3 | 567 | 55 | P0-3 |
| 50 | d0_large | oracle_d1 | 2024 | 0.710031 | 0.719736 | 21 | 0.713502 | 80.0 | 567 | 85 | P0-3 |
| 50 | d0_large | oracle_d1 | 2025 | 0.698325 | 0.726434 | 20 | 0.708576 | 78.0 | 567 | 85 | P0-3 |
| 50 | d0_large | oracle_d2 | 2022 | 0.594623 | 0.588183 | 11 | 0.599395 | 31.0 | 146 | 35 | P0-3 |
| 50 | d0_large | oracle_d2 | 2023 | 0.610271 | 0.621668 | 3 | 0.617680 | 20.5 | 146 | 35 | P0-3 |
| 50 | d0_large | oracle_d2 | 2024 | 0.585594 | 0.606989 | 5 | 0.602617 | 22.7 | 146 | 30 | P0-3 |
| 50 | d0_large | oracle_d2 | 2025 | 0.607831 | 0.622072 | 8 | 0.612099 | 26.7 | 146 | 30 | P0-3 |
| 50 | d1_large | oracle_d0 | 2022 | 0.581844 | 0.605830 | 14 | 0.591760 | 31.4 | 117 | 35 | P0-3 |
| 50 | d1_large | oracle_d0 | 2023 | 0.617735 | 0.620650 | 8 | 0.624801 | 22.9 | 117 | 35 | P0-3 |
| 50 | d1_large | oracle_d0 | 2024 | 0.592683 | 0.625175 | 8 | 0.600973 | 24.1 | 117 | 35 | P0-3 |
| 50 | d1_large | oracle_d0 | 2025 | 0.612001 | 0.620986 | 9 | 0.616693 | 23.6 | 117 | 30 | P0-3 |
| 50 | d1_large | oracle_d1 | 2022 | 0.751504 | 0.771216 | 10 | 0.762089 | 88.9 | 664 | 95 | P0-3 |
| 50 | d1_large | oracle_d1 | 2023 | 0.750395 | 0.771490 | 6 | 0.761086 | 69.8 | 664 | 75 | P0-3 |
| 50 | d1_large | oracle_d1 | 2024 | 0.755484 | 0.777434 | 8 | 0.766695 | 80.1 | 664 | 85 | P0-3 |
| 50 | d1_large | oracle_d1 | 2025 | 0.754895 | 0.774085 | 5 | 0.765550 | 66.7 | 664 | 70 | P0-3 |
| 50 | d1_large | oracle_d2 | 2022 | 0.677942 | 0.692125 | 20 | 0.684456 | 73.0 | 567 | 80 | P0-3 |
| 50 | d1_large | oracle_d2 | 2023 | 0.693474 | 0.713663 | 11 | 0.702416 | 51.7 | 567 | 55 | P0-3 |
| 50 | d1_large | oracle_d2 | 2024 | 0.668240 | 0.699758 | 10 | 0.682551 | 49.5 | 567 | 55 | P0-3 |
| 50 | d1_large | oracle_d2 | 2025 | 0.684439 | 0.709269 | 12 | 0.695984 | 54.1 | 567 | 60 | P0-3 |
| 50 | d2_large | oracle_d0 | 2022 | 0.717634 | 0.737770 | 34 | 0.724151 | 95.3 | 567 | 100 | P0-3 |
| 50 | d2_large | oracle_d0 | 2023 | 0.704336 | 0.723035 | 17 | 0.712586 | 58.7 | 567 | 65 | P0-3 |
| 50 | d2_large | oracle_d0 | 2024 | 0.720208 | 0.737856 | 21 | 0.728304 | 68.1 | 567 | 75 | P0-3 |
| 50 | d2_large | oracle_d0 | 2025 | 0.709059 | 0.722974 | 20 | 0.716175 | 66.0 | 567 | 70 | P0-3 |
| 50 | d2_large | oracle_d1 | 2022 | 0.608039 | 0.624990 | 13 | 0.615030 | 37.9 | 157 | 45 | P0-3 |
| 50 | d2_large | oracle_d1 | 2023 | 0.619282 | 0.637882 | 8 | 0.627157 | 31.0 | 157 | 35 | P0-3 |
| 50 | d2_large | oracle_d1 | 2024 | 0.604143 | 0.614468 | 12 | 0.611001 | 36.6 | 157 | 40 | P0-3 |
| 50 | d2_large | oracle_d1 | 2025 | 0.604183 | 0.638979 | 9 | 0.616775 | 31.6 | 157 | 35 | P0-3 |
| 50 | d2_large | oracle_d2 | 2022 | 0.722532 | 0.733646 | 8 | 0.728946 | 76.8 | 664 | 80 | P0-3 |
| 50 | d2_large | oracle_d2 | 2023 | 0.727650 | 0.747986 | 8 | 0.736872 | 76.9 | 664 | 85 | P0-3 |
| 50 | d2_large | oracle_d2 | 2024 | 0.722662 | 0.742036 | 7 | 0.731864 | 72.9 | 664 | 80 | P0-3 |
| 50 | d2_large | oracle_d2 | 2025 | 0.720011 | 0.746780 | 7 | 0.732423 | 72.4 | 664 | 80 | P0-3 |

## 4. Aggregated metrics (mean/std/min/max across 4 seeds)

### 4.1 Joint per-cell mean/std/min/max

| ratio | setting | arm | cell | mean | std | min | max | n_seeds |
|---|---|---|---|---|---|---|---|---|
| 1 | d0_large | baseline | d0t0 | 0.741932 | 0.006805 | 0.735413 | 0.753284 | 4 |
| 1 | d0_large | baseline | d0t1 | 0.758403 | 0.006915 | 0.751061 | 0.767468 | 4 |
| 1 | d0_large | baseline | d1t0 | 0.745680 | 0.003701 | 0.740493 | 0.750454 | 4 |
| 1 | d0_large | baseline | d1t1 | 0.766582 | 0.003097 | 0.762744 | 0.770215 | 4 |
| 1 | d0_large | baseline | d2t0 | 0.707432 | 0.004318 | 0.703907 | 0.714783 | 4 |
| 1 | d0_large | baseline | d2t1 | 0.727641 | 0.004337 | 0.723205 | 0.734726 | 4 |
| 1 | d0_large | comp1 | d0t0 | 0.742470 | 0.006494 | 0.738342 | 0.753705 | 4 |
| 1 | d0_large | comp1 | d0t1 | 0.760026 | 0.002744 | 0.756165 | 0.762712 | 4 |
| 1 | d0_large | comp1 | d1t0 | 0.747847 | 0.003218 | 0.744316 | 0.752320 | 4 |
| 1 | d0_large | comp1 | d1t1 | 0.766186 | 0.007325 | 0.755925 | 0.775799 | 4 |
| 1 | d0_large | comp1 | d2t0 | 0.708000 | 0.004939 | 0.701001 | 0.714964 | 4 |
| 1 | d0_large | comp1 | d2t1 | 0.729538 | 0.002811 | 0.725419 | 0.733338 | 4 |
| 50 | d0_large | baseline | d0t0 | 0.742172 | 0.002550 | 0.738897 | 0.744964 | 4 |
| 50 | d0_large | baseline | d0t1 | 0.755226 | 0.003764 | 0.748987 | 0.759049 | 4 |
| 50 | d0_large | baseline | d1t0 | 0.726206 | 0.006376 | 0.717543 | 0.735291 | 4 |
| 50 | d0_large | baseline | d1t1 | 0.750421 | 0.002952 | 0.747560 | 0.755370 | 4 |
| 50 | d0_large | baseline | d2t0 | 0.656864 | 0.008793 | 0.647793 | 0.670640 | 4 |
| 50 | d0_large | baseline | d2t1 | 0.671249 | 0.016823 | 0.644223 | 0.687260 | 4 |
| 50 | d0_large | comp1 | d0t0 | 0.747072 | 0.001130 | 0.745167 | 0.748052 | 4 |
| 50 | d0_large | comp1 | d0t1 | 0.762194 | 0.002832 | 0.757802 | 0.764970 | 4 |
| 50 | d0_large | comp1 | d1t0 | 0.732938 | 0.002571 | 0.728883 | 0.735562 | 4 |
| 50 | d0_large | comp1 | d1t1 | 0.751849 | 0.005763 | 0.743922 | 0.759228 | 4 |
| 50 | d0_large | comp1 | d2t0 | 0.665360 | 0.007203 | 0.654356 | 0.673976 | 4 |
| 50 | d0_large | comp1 | d2t1 | 0.682952 | 0.005702 | 0.673346 | 0.687711 | 4 |
| 50 | d1_large | baseline | d0t0 | 0.668809 | 0.013059 | 0.649674 | 0.686496 | 4 |
| 50 | d1_large | baseline | d0t1 | 0.692983 | 0.014377 | 0.678721 | 0.713661 | 4 |
| 50 | d1_large | baseline | d1t0 | 0.739541 | 0.009199 | 0.727044 | 0.752975 | 4 |
| 50 | d1_large | baseline | d1t1 | 0.767610 | 0.003607 | 0.764346 | 0.773451 | 4 |
| 50 | d1_large | baseline | d2t0 | 0.689174 | 0.006479 | 0.684285 | 0.700190 | 4 |
| 50 | d1_large | baseline | d2t1 | 0.713153 | 0.004960 | 0.706436 | 0.719592 | 4 |
| 50 | d1_large | comp1 | d0t0 | 0.687381 | 0.004577 | 0.680110 | 0.691995 | 4 |
| 50 | d1_large | comp1 | d0t1 | 0.707657 | 0.013193 | 0.689254 | 0.721152 | 4 |
| 50 | d1_large | comp1 | d1t0 | 0.754248 | 0.002998 | 0.750304 | 0.757903 | 4 |
| 50 | d1_large | comp1 | d1t1 | 0.774429 | 0.001922 | 0.771614 | 0.776307 | 4 |
| 50 | d1_large | comp1 | d2t0 | 0.704913 | 0.008078 | 0.695063 | 0.717428 | 4 |
| 50 | d1_large | comp1 | d2t1 | 0.725770 | 0.008921 | 0.717211 | 0.739981 | 4 |
| 50 | d2_large | baseline | d0t0 | 0.725773 | 0.004179 | 0.721941 | 0.732711 | 4 |
| 50 | d2_large | baseline | d0t1 | 0.739005 | 0.003591 | 0.733978 | 0.744075 | 4 |
| 50 | d2_large | baseline | d1t0 | 0.663115 | 0.009086 | 0.650207 | 0.672099 | 4 |
| 50 | d2_large | baseline | d1t1 | 0.704148 | 0.009353 | 0.689488 | 0.715225 | 4 |
| 50 | d2_large | baseline | d2t0 | 0.707612 | 0.011028 | 0.695393 | 0.718907 | 4 |
| 50 | d2_large | baseline | d2t1 | 0.727593 | 0.005153 | 0.720295 | 0.734599 | 4 |
| 50 | d2_large | comp1 | d0t0 | 0.729993 | 0.003467 | 0.724437 | 0.733993 | 4 |
| 50 | d2_large | comp1 | d0t1 | 0.743918 | 0.002952 | 0.738827 | 0.745870 | 4 |
| 50 | d2_large | comp1 | d1t0 | 0.689529 | 0.009441 | 0.678854 | 0.701613 | 4 |
| 50 | d2_large | comp1 | d1t1 | 0.718791 | 0.005844 | 0.709694 | 0.725711 | 4 |
| 50 | d2_large | comp1 | d2t0 | 0.722658 | 0.003964 | 0.718938 | 0.729096 | 4 |
| 50 | d2_large | comp1 | d2t1 | 0.741922 | 0.003284 | 0.737865 | 0.745177 | 4 |

### 4.2 Oracle per-(ratio, setting, oracle_domain, task): mean/std/min/max across 4 seeds

| ratio | setting | oracle_domain | task | mean | std | min | max | n_seeds |
|---|---|---|---|---|---|---|---|---|
| 1 | d0_large | oracle_d0 | t0 | 0.729241 | 0.003771 | 0.725228 | 0.735414 | 4 |
| 1 | d0_large | oracle_d0 | t1 | 0.746901 | 0.002660 | 0.742533 | 0.749153 | 4 |
| 1 | d0_large | oracle_d1 | t0 | 0.731537 | 0.007378 | 0.720859 | 0.740910 | 4 |
| 1 | d0_large | oracle_d1 | t1 | 0.748687 | 0.006078 | 0.739939 | 0.757098 | 4 |
| 1 | d0_large | oracle_d2 | t0 | 0.700694 | 0.003913 | 0.696526 | 0.706660 | 4 |
| 1 | d0_large | oracle_d2 | t1 | 0.720177 | 0.005502 | 0.714169 | 0.729153 | 4 |
| 50 | d0_large | oracle_d0 | t0 | 0.751797 | 0.002331 | 0.748319 | 0.754840 | 4 |
| 50 | d0_large | oracle_d0 | t1 | 0.764932 | 0.001300 | 0.763832 | 0.767068 | 4 |
| 50 | d0_large | oracle_d1 | t0 | 0.707250 | 0.010297 | 0.697716 | 0.722928 | 4 |
| 50 | d0_large | oracle_d1 | t1 | 0.725840 | 0.007904 | 0.718667 | 0.738522 | 4 |
| 50 | d0_large | oracle_d2 | t0 | 0.599580 | 0.010032 | 0.585594 | 0.610271 | 4 |
| 50 | d0_large | oracle_d2 | t1 | 0.609728 | 0.013844 | 0.588183 | 0.622072 | 4 |
| 50 | d1_large | oracle_d0 | t0 | 0.601066 | 0.014467 | 0.581844 | 0.617735 | 4 |
| 50 | d1_large | oracle_d0 | t1 | 0.618160 | 0.007339 | 0.605830 | 0.625175 | 4 |
| 50 | d1_large | oracle_d1 | t0 | 0.753070 | 0.002166 | 0.750395 | 0.755484 | 4 |
| 50 | d1_large | oracle_d1 | t1 | 0.773556 | 0.002503 | 0.771216 | 0.777434 | 4 |
| 50 | d1_large | oracle_d2 | t0 | 0.681024 | 0.009214 | 0.668240 | 0.693474 | 4 |
| 50 | d1_large | oracle_d2 | t1 | 0.703704 | 0.008364 | 0.692125 | 0.713663 | 4 |
| 50 | d2_large | oracle_d0 | t0 | 0.712809 | 0.006401 | 0.704336 | 0.720208 | 4 |
| 50 | d2_large | oracle_d0 | t1 | 0.730409 | 0.007404 | 0.722974 | 0.737856 | 4 |
| 50 | d2_large | oracle_d1 | t0 | 0.608912 | 0.006193 | 0.604143 | 0.619282 | 4 |
| 50 | d2_large | oracle_d1 | t1 | 0.629080 | 0.010071 | 0.614468 | 0.638979 | 4 |
| 50 | d2_large | oracle_d2 | t0 | 0.723214 | 0.002771 | 0.720011 | 0.727650 | 4 |
| 50 | d2_large | oracle_d2 | t1 | 0.742612 | 0.005634 | 0.733646 | 0.747986 | 4 |

### 4.3 Δ_transfer per cell: baseline (1 cell) − oracle (1 cell) at best_epoch, per (tier, role)

Cells produced by `experiments/p0_3_report.py` (cell-pairing: joint baseline dXtY matched against oracle_dX tY of the same seed). MMoE rows only, n = 4 seeds × 2 tasks × {1 setting for balanced, 3 settings for strong}.

Per (tier, role):

| tier | role | n_cells | delta_mean | delta_std | delta_min | delta_max | comp1_gain_mean | comp1_gain_std | comp1_gain_min | comp1_gain_max |
|---|---|---|---|---|---|---|---|---|---|---|
| balanced | L | 8 | 0.012096 | 0.004801 | 0.004077 | 0.018315 | 0.001080 | 0.003393 | -0.004756 | 0.006342 |
| balanced | M | 8 | 0.016019 | 0.004559 | 0.009544 | 0.022805 | 0.000886 | 0.003450 | -0.006819 | 0.005584 |
| balanced | S | 8 | 0.007101 | 0.003092 | 0.002285 | 0.013107 | 0.001233 | 0.003008 | -0.003894 | 0.006211 |
| strong | L | 24 | -0.011571 | 0.007062 | -0.026485 | -0.002509 | 0.010462 | 0.007730 | 0.001608 | 0.030859 |
| strong | M | 24 | 0.013783 | 0.010028 | 0.000280 | 0.037575 | 0.007608 | 0.007429 | -0.006408 | 0.024259 |
| strong | S | 24 | 0.065107 | 0.013693 | 0.039662 | 0.093011 | 0.015751 | 0.011964 | -0.003769 | 0.040737 |

Per (tier, setting, role):

| tier | setting | role | n_cells | delta_mean | delta_std | delta_min | delta_max | comp1_gain_mean | comp1_gain_std | comp1_gain_min | comp1_gain_max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| balanced | d0_large | L | 8 | 0.012096 | 0.004801 | 0.004077 | 0.018315 | 0.001080 | 0.003393 | -0.004756 | 0.006342 |
| balanced | d0_large | M | 8 | 0.016019 | 0.004559 | 0.009544 | 0.022805 | 0.000886 | 0.003450 | -0.006819 | 0.005584 |
| balanced | d0_large | S | 8 | 0.007101 | 0.003092 | 0.002285 | 0.013107 | 0.001233 | 0.003008 | -0.003894 | 0.006211 |
| strong | d0_large | L | 8 | -0.009665 | 0.004103 | -0.015943 | -0.003985 | 0.005934 | 0.002552 | 0.002357 | 0.008815 |
| strong | d0_large | M | 8 | 0.021769 | 0.011406 | 0.001493 | 0.037575 | 0.004080 | 0.007432 | -0.006408 | 0.018019 |
| strong | d0_large | S | 8 | 0.059403 | 0.010218 | 0.040627 | 0.076274 | 0.010100 | 0.010490 | -0.003769 | 0.029123 |
| strong | d1_large | L | 8 | -0.009738 | 0.006692 | -0.023351 | -0.002509 | 0.010763 | 0.008304 | 0.002649 | 0.030859 |
| strong | d1_large | M | 8 | 0.008800 | 0.005872 | 0.000280 | 0.019262 | 0.014178 | 0.005849 | 0.006347 | 0.024259 |
| strong | d1_large | S | 8 | 0.071283 | 0.013479 | 0.053546 | 0.093011 | 0.016623 | 0.012683 | 0.001935 | 0.040737 |
| strong | d2_large | L | 8 | -0.015310 | 0.008187 | -0.026485 | -0.004360 | 0.014687 | 0.008083 | 0.001608 | 0.024882 |
| strong | d2_large | M | 8 | 0.010779 | 0.006285 | 0.000575 | 0.020933 | 0.004567 | 0.003358 | -0.001916 | 0.008724 |
| strong | d2_large | S | 8 | 0.064635 | 0.014335 | 0.039662 | 0.079229 | 0.020529 | 0.010150 | 0.007646 | 0.036793 |

### 4.4 Summary per (tier, method) — MMoE only (mirror of P0_3_summary.csv, mmoe rows)

| tier | method | mean_auc | worst_domain_auc | comp1_gain_overall | comp1_gain_role_L | comp1_gain_role_M | comp1_gain_role_S |
|---|---|---|---|---|---|---|---|
| balanced | baseline | 0.741278 | 0.707432 |  |  |  |  |
| balanced | comp1 | 0.742345 | 0.708000 | 0.001066 | degenerate | degenerate | degenerate |
| balanced | oracle | 0.729540 | 0.700694 |  |  |  |  |
| strong | baseline | 0.713370 | 0.684550 |  |  |  |  |
| strong | comp1 | 0.724643 | 0.697644 | 0.011273 | 0.010462 | 0.007608 | 0.015751 |
| strong | oracle | 0.690930 | 0.667939 |  |  |  |  |

### 4.5 Quick-look: joint 6-cell mean per (tier, setting, method), averaged across 4 seeds

| tier | setting | method | joint_6cell_mean |
|---|---|---|---|
| balanced | d0_large | baseline | 0.741278 |
| balanced | d0_large | comp1 | 0.742345 |
| strong | d0_large | baseline | 0.717033 |
| strong | d0_large | comp1 | 0.723728 |
| strong | d1_large | baseline | 0.708545 |
| strong | d1_large | comp1 | 0.725733 |
| strong | d2_large | baseline | 0.713533 |
| strong | d2_large | comp1 | 0.724470 |

## 5. Failed runs

No failed runs. All 80 unique fresh tasks reached status=OK.

Operational notes:

- The orchestrator used a PID lockfile at `experiments/logs/p0_3_orchestrator.lock`; the launch wrote pid and removed it on graceful exit (no second orchestrator was ever started).
- The orchestrator process was started via `nohup` and detached from the runner's session; the master CSV was consulted on every monitor checkpoint (T1=25min → 35/80 OK; T2=50min → 67/80 OK; T3=75min → completed 80/80).
- Per-run stdout logs are at `experiments/logs/p0_3_<name>.log` (one per fresh task; 80 logs total, plus `p0_3_mmoe_transfer/orchestrator.log` for orchestrator progress).

## 6. Missing metrics

No missing metrics.

- All 80 fresh run directories under `experiments/P0_3_mmoe_transfer/{balanced,strong}/` contain `config.json`, `metrics_epoch.csv`, `metrics_summary.csv`.
- Joint total_train_samples: all 32 joint runs have role_sizes sum = 187,500 (matches contract per tier).
- Oracle per-domain training sample count: each oracle_d run's single_domain role size matches the expected (L, M, S) value for its (tier, setting):
  - balanced: 62500 / 62500 / 62500 (L,M,S all equal)
  - strong: 150000 / 34500 / 3000

## 7. Aggregator outputs

`experiments/p0_3_report.py` reads `P0_3_raw.csv` (MMoE, 80 rows) and joins with `P0_2_raw.csv` filtered to `ratio in {1, 50}` (M3OE, 80 rows), producing:

- `experiments/P0_3_mmoe_transfer/P0_3_cells.csv` (192 rows; 96 cells per backbone × 2 backbones)
- `experiments/P0_3_mmoe_transfer/P0_3_transfer_matrix.csv` (46 rows; backbone × tier × scope)
- `experiments/P0_3_mmoe_transfer/P0_3_summary.csv` (12 rows; backbone × tier × method)