# Raw Experiment Report — P0-2

## 1. Experiment metadata

- Stage: P0-2 (L:M:S role-ratio sweep, fixed total N = 187,500 samples across 3 domains)
- Dataset: ml-1m (MDMTRec domain configuration: d0=d1=d2 in ml-1m; mappings rotate per setting)
- Model: MDMTRec (from m3oe/examples)
- Method: joint (baseline, comp1) and oracle (single_domain=0/1/2); all P0-2 runs pass `--fix_optim_grouping 1`
- comp1 extra: `--momentum_mode per_domain_only`
- Seeds: 2022, 2023, 2024, 2025 (4 seeds per arm)
- Joint arms: ratio_1 (1 setting) + ratio_5 (3 settings) + ratio_50 (3 settings) = 7 (setting, ratio) combos × 2 methods × 4 seeds = 56 runs
- Oracle arms: same 7 combos × 3 domains × 4 seeds = 84 runs
- Fresh runs: 56 + 84 = 140
- Reused from P0-1 (20:1, ratio=20): 60 runs (3 settings × 5 arms × 4 seeds); source=P0-1_reuse
- Total rows in P0_2_all.csv: 200 (140 fresh + 60 reused)
- Ratio role_sizes contracts (joint totals = 187,500):
  - ratio=1: 62500,62500,62500 (L,M,S)
  - ratio=5: 125000,37500,25000
  - ratio=20: 150000,30000,7500 (from P0-1, role sizes match contract)
  - ratio=50: 150000,34500,3000
- Code: experiments/p0_2_orchestrator.py + experiments/p4_run.py (orchestrator); experiments/p0_2_report.py (aggregator)
- GPU: NVIDIA GeForce RTX 4060 Laptop (8 GiB)
- Start time: 2026-09-09 01:26 (orchestrator launch)
- End time: 2026-09-09 03:46 (last OK row written; P0_2_ORCHESTRATOR DONE)
- Wall-clock total: ~269 minutes (sum of per-run wall_sec)
- Total fit time: 15267s; mean fit 109s; peak cuda mem mean 791 MiB, max 1158 MiB
- Concurrency: max 2 p4_run children simultaneously; stagger 30s
- Resume-safe: orchestrator reads existing P0_2_raw.csv on start and skips (kind,ratio,setting,method_or_domain,seed) tuples already marked OK

## 2. Completion status

All 140 unique tasks completed OK (no FAIL/CRASH rows in master CSV).

Master CSV `P0_2_raw.csv` has 140 OK rows (one per unique task). The 20:1 rows were never re-run; they live in `P0_1_raw_dedup.csv` and are referenced by the aggregator as `source=P0-1_reuse`.

| run | status | checkpoint | source | log |
|---|---|---|---|---|
| joint_ratio1_d0_large_baseline_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio1_d0_large_baseline_seed2022.log |
| joint_ratio1_d0_large_baseline_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio1_d0_large_baseline_seed2023.log |
| joint_ratio1_d0_large_baseline_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio1_d0_large_baseline_seed2024.log |
| joint_ratio1_d0_large_baseline_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio1_d0_large_baseline_seed2025.log |
| joint_ratio1_d0_large_comp1_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio1_d0_large_comp1_seed2022.log |
| joint_ratio1_d0_large_comp1_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio1_d0_large_comp1_seed2023.log |
| joint_ratio1_d0_large_comp1_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio1_d0_large_comp1_seed2024.log |
| joint_ratio1_d0_large_comp1_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio1_d0_large_comp1_seed2025.log |
| oracle_ratio1_d0_large_oracle_d0_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio1_d0_large_oracle_d0_seed2022.log |
| oracle_ratio1_d0_large_oracle_d0_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio1_d0_large_oracle_d0_seed2023.log |
| oracle_ratio1_d0_large_oracle_d0_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio1_d0_large_oracle_d0_seed2024.log |
| oracle_ratio1_d0_large_oracle_d0_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio1_d0_large_oracle_d0_seed2025.log |
| oracle_ratio1_d0_large_oracle_d1_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio1_d0_large_oracle_d1_seed2022.log |
| oracle_ratio1_d0_large_oracle_d1_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio1_d0_large_oracle_d1_seed2023.log |
| oracle_ratio1_d0_large_oracle_d1_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio1_d0_large_oracle_d1_seed2024.log |
| oracle_ratio1_d0_large_oracle_d1_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio1_d0_large_oracle_d1_seed2025.log |
| oracle_ratio1_d0_large_oracle_d2_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio1_d0_large_oracle_d2_seed2022.log |
| oracle_ratio1_d0_large_oracle_d2_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio1_d0_large_oracle_d2_seed2023.log |
| oracle_ratio1_d0_large_oracle_d2_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio1_d0_large_oracle_d2_seed2024.log |
| oracle_ratio1_d0_large_oracle_d2_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio1_d0_large_oracle_d2_seed2025.log |
| joint_ratio5_d0_large_baseline_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d0_large_baseline_seed2022.log |
| joint_ratio5_d0_large_baseline_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d0_large_baseline_seed2023.log |
| joint_ratio5_d0_large_baseline_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d0_large_baseline_seed2024.log |
| joint_ratio5_d0_large_baseline_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d0_large_baseline_seed2025.log |
| joint_ratio5_d0_large_comp1_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d0_large_comp1_seed2022.log |
| joint_ratio5_d0_large_comp1_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d0_large_comp1_seed2023.log |
| joint_ratio5_d0_large_comp1_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d0_large_comp1_seed2024.log |
| joint_ratio5_d0_large_comp1_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d0_large_comp1_seed2025.log |
| oracle_ratio5_d0_large_oracle_d0_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d0_large_oracle_d0_seed2022.log |
| oracle_ratio5_d0_large_oracle_d0_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d0_large_oracle_d0_seed2023.log |
| oracle_ratio5_d0_large_oracle_d0_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d0_large_oracle_d0_seed2024.log |
| oracle_ratio5_d0_large_oracle_d0_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d0_large_oracle_d0_seed2025.log |
| oracle_ratio5_d0_large_oracle_d1_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d0_large_oracle_d1_seed2022.log |
| oracle_ratio5_d0_large_oracle_d1_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d0_large_oracle_d1_seed2023.log |
| oracle_ratio5_d0_large_oracle_d1_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d0_large_oracle_d1_seed2024.log |
| oracle_ratio5_d0_large_oracle_d1_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d0_large_oracle_d1_seed2025.log |
| oracle_ratio5_d0_large_oracle_d2_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d0_large_oracle_d2_seed2022.log |
| oracle_ratio5_d0_large_oracle_d2_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d0_large_oracle_d2_seed2023.log |
| oracle_ratio5_d0_large_oracle_d2_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d0_large_oracle_d2_seed2024.log |
| oracle_ratio5_d0_large_oracle_d2_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d0_large_oracle_d2_seed2025.log |
| joint_ratio5_d1_large_baseline_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d1_large_baseline_seed2022.log |
| joint_ratio5_d1_large_baseline_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d1_large_baseline_seed2023.log |
| joint_ratio5_d1_large_baseline_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d1_large_baseline_seed2024.log |
| joint_ratio5_d1_large_baseline_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d1_large_baseline_seed2025.log |
| joint_ratio5_d1_large_comp1_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d1_large_comp1_seed2022.log |
| joint_ratio5_d1_large_comp1_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d1_large_comp1_seed2023.log |
| joint_ratio5_d1_large_comp1_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d1_large_comp1_seed2024.log |
| joint_ratio5_d1_large_comp1_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d1_large_comp1_seed2025.log |
| oracle_ratio5_d1_large_oracle_d0_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d1_large_oracle_d0_seed2022.log |
| oracle_ratio5_d1_large_oracle_d0_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d1_large_oracle_d0_seed2023.log |
| oracle_ratio5_d1_large_oracle_d0_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d1_large_oracle_d0_seed2024.log |
| oracle_ratio5_d1_large_oracle_d0_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d1_large_oracle_d0_seed2025.log |
| oracle_ratio5_d1_large_oracle_d1_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d1_large_oracle_d1_seed2022.log |
| oracle_ratio5_d1_large_oracle_d1_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d1_large_oracle_d1_seed2023.log |
| oracle_ratio5_d1_large_oracle_d1_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d1_large_oracle_d1_seed2024.log |
| oracle_ratio5_d1_large_oracle_d1_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d1_large_oracle_d1_seed2025.log |
| oracle_ratio5_d1_large_oracle_d2_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d1_large_oracle_d2_seed2022.log |
| oracle_ratio5_d1_large_oracle_d2_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d1_large_oracle_d2_seed2023.log |
| oracle_ratio5_d1_large_oracle_d2_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d1_large_oracle_d2_seed2024.log |
| oracle_ratio5_d1_large_oracle_d2_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d1_large_oracle_d2_seed2025.log |
| joint_ratio5_d2_large_baseline_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d2_large_baseline_seed2022.log |
| joint_ratio5_d2_large_baseline_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d2_large_baseline_seed2023.log |
| joint_ratio5_d2_large_baseline_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d2_large_baseline_seed2024.log |
| joint_ratio5_d2_large_baseline_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d2_large_baseline_seed2025.log |
| joint_ratio5_d2_large_comp1_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d2_large_comp1_seed2022.log |
| joint_ratio5_d2_large_comp1_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d2_large_comp1_seed2023.log |
| joint_ratio5_d2_large_comp1_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d2_large_comp1_seed2024.log |
| joint_ratio5_d2_large_comp1_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio5_d2_large_comp1_seed2025.log |
| oracle_ratio5_d2_large_oracle_d0_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d2_large_oracle_d0_seed2022.log |
| oracle_ratio5_d2_large_oracle_d0_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d2_large_oracle_d0_seed2023.log |
| oracle_ratio5_d2_large_oracle_d0_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d2_large_oracle_d0_seed2024.log |
| oracle_ratio5_d2_large_oracle_d0_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d2_large_oracle_d0_seed2025.log |
| oracle_ratio5_d2_large_oracle_d1_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d2_large_oracle_d1_seed2022.log |
| oracle_ratio5_d2_large_oracle_d1_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d2_large_oracle_d1_seed2023.log |
| oracle_ratio5_d2_large_oracle_d1_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d2_large_oracle_d1_seed2024.log |
| oracle_ratio5_d2_large_oracle_d1_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d2_large_oracle_d1_seed2025.log |
| oracle_ratio5_d2_large_oracle_d2_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d2_large_oracle_d2_seed2022.log |
| oracle_ratio5_d2_large_oracle_d2_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d2_large_oracle_d2_seed2023.log |
| oracle_ratio5_d2_large_oracle_d2_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d2_large_oracle_d2_seed2024.log |
| oracle_ratio5_d2_large_oracle_d2_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio5_d2_large_oracle_d2_seed2025.log |
| joint_ratio50_d0_large_baseline_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d0_large_baseline_seed2022.log |
| joint_ratio50_d0_large_baseline_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d0_large_baseline_seed2023.log |
| joint_ratio50_d0_large_baseline_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d0_large_baseline_seed2024.log |
| joint_ratio50_d0_large_baseline_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d0_large_baseline_seed2025.log |
| joint_ratio50_d0_large_comp1_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d0_large_comp1_seed2022.log |
| joint_ratio50_d0_large_comp1_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d0_large_comp1_seed2023.log |
| joint_ratio50_d0_large_comp1_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d0_large_comp1_seed2024.log |
| joint_ratio50_d0_large_comp1_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d0_large_comp1_seed2025.log |
| oracle_ratio50_d0_large_oracle_d0_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d0_large_oracle_d0_seed2022.log |
| oracle_ratio50_d0_large_oracle_d0_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d0_large_oracle_d0_seed2023.log |
| oracle_ratio50_d0_large_oracle_d0_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d0_large_oracle_d0_seed2024.log |
| oracle_ratio50_d0_large_oracle_d0_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d0_large_oracle_d0_seed2025.log |
| oracle_ratio50_d0_large_oracle_d1_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d0_large_oracle_d1_seed2022.log |
| oracle_ratio50_d0_large_oracle_d1_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d0_large_oracle_d1_seed2023.log |
| oracle_ratio50_d0_large_oracle_d1_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d0_large_oracle_d1_seed2024.log |
| oracle_ratio50_d0_large_oracle_d1_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d0_large_oracle_d1_seed2025.log |
| oracle_ratio50_d0_large_oracle_d2_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d0_large_oracle_d2_seed2022.log |
| oracle_ratio50_d0_large_oracle_d2_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d0_large_oracle_d2_seed2023.log |
| oracle_ratio50_d0_large_oracle_d2_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d0_large_oracle_d2_seed2024.log |
| oracle_ratio50_d0_large_oracle_d2_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d0_large_oracle_d2_seed2025.log |
| joint_ratio50_d1_large_baseline_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d1_large_baseline_seed2022.log |
| joint_ratio50_d1_large_baseline_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d1_large_baseline_seed2023.log |
| joint_ratio50_d1_large_baseline_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d1_large_baseline_seed2024.log |
| joint_ratio50_d1_large_baseline_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d1_large_baseline_seed2025.log |
| joint_ratio50_d1_large_comp1_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d1_large_comp1_seed2022.log |
| joint_ratio50_d1_large_comp1_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d1_large_comp1_seed2023.log |
| joint_ratio50_d1_large_comp1_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d1_large_comp1_seed2024.log |
| joint_ratio50_d1_large_comp1_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d1_large_comp1_seed2025.log |
| oracle_ratio50_d1_large_oracle_d0_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d1_large_oracle_d0_seed2022.log |
| oracle_ratio50_d1_large_oracle_d0_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d1_large_oracle_d0_seed2023.log |
| oracle_ratio50_d1_large_oracle_d0_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d1_large_oracle_d0_seed2024.log |
| oracle_ratio50_d1_large_oracle_d0_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d1_large_oracle_d0_seed2025.log |
| oracle_ratio50_d1_large_oracle_d1_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d1_large_oracle_d1_seed2022.log |
| oracle_ratio50_d1_large_oracle_d1_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d1_large_oracle_d1_seed2023.log |
| oracle_ratio50_d1_large_oracle_d1_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d1_large_oracle_d1_seed2024.log |
| oracle_ratio50_d1_large_oracle_d1_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d1_large_oracle_d1_seed2025.log |
| oracle_ratio50_d1_large_oracle_d2_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d1_large_oracle_d2_seed2022.log |
| oracle_ratio50_d1_large_oracle_d2_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d1_large_oracle_d2_seed2023.log |
| oracle_ratio50_d1_large_oracle_d2_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d1_large_oracle_d2_seed2024.log |
| oracle_ratio50_d1_large_oracle_d2_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d1_large_oracle_d2_seed2025.log |
| joint_ratio50_d2_large_baseline_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d2_large_baseline_seed2022.log |
| joint_ratio50_d2_large_baseline_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d2_large_baseline_seed2023.log |
| joint_ratio50_d2_large_baseline_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d2_large_baseline_seed2024.log |
| joint_ratio50_d2_large_baseline_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d2_large_baseline_seed2025.log |
| joint_ratio50_d2_large_comp1_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d2_large_comp1_seed2022.log |
| joint_ratio50_d2_large_comp1_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d2_large_comp1_seed2023.log |
| joint_ratio50_d2_large_comp1_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d2_large_comp1_seed2024.log |
| joint_ratio50_d2_large_comp1_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_joint_ratio50_d2_large_comp1_seed2025.log |
| oracle_ratio50_d2_large_oracle_d0_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d2_large_oracle_d0_seed2022.log |
| oracle_ratio50_d2_large_oracle_d0_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d2_large_oracle_d0_seed2023.log |
| oracle_ratio50_d2_large_oracle_d0_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d2_large_oracle_d0_seed2024.log |
| oracle_ratio50_d2_large_oracle_d0_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d2_large_oracle_d0_seed2025.log |
| oracle_ratio50_d2_large_oracle_d1_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d2_large_oracle_d1_seed2022.log |
| oracle_ratio50_d2_large_oracle_d1_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d2_large_oracle_d1_seed2023.log |
| oracle_ratio50_d2_large_oracle_d1_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d2_large_oracle_d1_seed2024.log |
| oracle_ratio50_d2_large_oracle_d1_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d2_large_oracle_d1_seed2025.log |
| oracle_ratio50_d2_large_oracle_d2_seed2022 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d2_large_oracle_d2_seed2022.log |
| oracle_ratio50_d2_large_oracle_d2_seed2023 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d2_large_oracle_d2_seed2023.log |
| oracle_ratio50_d2_large_oracle_d2_seed2024 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d2_large_oracle_d2_seed2024.log |
| oracle_ratio50_d2_large_oracle_d2_seed2025 | OK | metrics_summary.csv | P0-2 | experiments/logs/p0_2_oracle_ratio50_d2_large_oracle_d2_seed2025.log |
| joint_ratio20_d0_large_baseline_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd0_large__d0_large_baseline_seed2022.log |
| joint_ratio20_d0_large_comp1_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd0_large__d0_large_comp1_seed2022.log |
| oracle_ratio20_d0_large_oracle_d1_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd0_large__d0_large_oracle_d1_seed2022.log |
| oracle_ratio20_d0_large_oracle_d0_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd0_large__d0_large_oracle_d0_seed2022.log |
| oracle_ratio20_d0_large_oracle_d2_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd0_large__d0_large_oracle_d2_seed2022.log |
| joint_ratio20_d0_large_baseline_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd0_large__d0_large_baseline_seed2023.log |
| joint_ratio20_d0_large_comp1_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd0_large__d0_large_comp1_seed2023.log |
| oracle_ratio20_d0_large_oracle_d0_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd0_large__d0_large_oracle_d0_seed2023.log |
| oracle_ratio20_d0_large_oracle_d1_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd0_large__d0_large_oracle_d1_seed2023.log |
| oracle_ratio20_d0_large_oracle_d2_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd0_large__d0_large_oracle_d2_seed2023.log |
| joint_ratio20_d0_large_baseline_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd0_large__d0_large_baseline_seed2024.log |
| joint_ratio20_d0_large_comp1_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd0_large__d0_large_comp1_seed2024.log |
| oracle_ratio20_d0_large_oracle_d0_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd0_large__d0_large_oracle_d0_seed2024.log |
| oracle_ratio20_d0_large_oracle_d2_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd0_large__d0_large_oracle_d2_seed2024.log |
| oracle_ratio20_d0_large_oracle_d1_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd0_large__d0_large_oracle_d1_seed2024.log |
| joint_ratio20_d0_large_baseline_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd0_large__d0_large_baseline_seed2025.log |
| oracle_ratio20_d0_large_oracle_d0_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd0_large__d0_large_oracle_d0_seed2025.log |
| joint_ratio20_d0_large_comp1_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd0_large__d0_large_comp1_seed2025.log |
| oracle_ratio20_d0_large_oracle_d2_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd0_large__d0_large_oracle_d2_seed2025.log |
| oracle_ratio20_d0_large_oracle_d1_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd0_large__d0_large_oracle_d1_seed2025.log |
| joint_ratio20_d1_large_baseline_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd1_large__d1_large_baseline_seed2022.log |
| joint_ratio20_d1_large_comp1_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd1_large__d1_large_comp1_seed2022.log |
| oracle_ratio20_d1_large_oracle_d1_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd1_large__d1_large_oracle_d1_seed2022.log |
| oracle_ratio20_d1_large_oracle_d2_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd1_large__d1_large_oracle_d2_seed2022.log |
| oracle_ratio20_d1_large_oracle_d0_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd1_large__d1_large_oracle_d0_seed2022.log |
| joint_ratio20_d1_large_baseline_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd1_large__d1_large_baseline_seed2023.log |
| joint_ratio20_d1_large_comp1_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd1_large__d1_large_comp1_seed2023.log |
| oracle_ratio20_d1_large_oracle_d1_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd1_large__d1_large_oracle_d1_seed2023.log |
| oracle_ratio20_d1_large_oracle_d2_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd1_large__d1_large_oracle_d2_seed2023.log |
| oracle_ratio20_d1_large_oracle_d0_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd1_large__d1_large_oracle_d0_seed2023.log |
| joint_ratio20_d1_large_baseline_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd1_large__d1_large_baseline_seed2024.log |
| joint_ratio20_d1_large_comp1_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd1_large__d1_large_comp1_seed2024.log |
| oracle_ratio20_d1_large_oracle_d1_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd1_large__d1_large_oracle_d1_seed2024.log |
| oracle_ratio20_d1_large_oracle_d2_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd1_large__d1_large_oracle_d2_seed2024.log |
| oracle_ratio20_d1_large_oracle_d0_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd1_large__d1_large_oracle_d0_seed2024.log |
| joint_ratio20_d1_large_baseline_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd1_large__d1_large_baseline_seed2025.log |
| joint_ratio20_d1_large_comp1_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd1_large__d1_large_comp1_seed2025.log |
| oracle_ratio20_d1_large_oracle_d1_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd1_large__d1_large_oracle_d1_seed2025.log |
| oracle_ratio20_d1_large_oracle_d2_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd1_large__d1_large_oracle_d2_seed2025.log |
| oracle_ratio20_d1_large_oracle_d0_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd1_large__d1_large_oracle_d0_seed2025.log |
| joint_ratio20_d2_large_baseline_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd2_large__d2_large_baseline_seed2022.log |
| joint_ratio20_d2_large_comp1_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd2_large__d2_large_comp1_seed2022.log |
| oracle_ratio20_d2_large_oracle_d2_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd2_large__d2_large_oracle_d2_seed2022.log |
| oracle_ratio20_d2_large_oracle_d0_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd2_large__d2_large_oracle_d0_seed2022.log |
| oracle_ratio20_d2_large_oracle_d1_seed2022 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd2_large__d2_large_oracle_d1_seed2022.log |
| joint_ratio20_d2_large_baseline_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd2_large__d2_large_baseline_seed2023.log |
| joint_ratio20_d2_large_comp1_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd2_large__d2_large_comp1_seed2023.log |
| oracle_ratio20_d2_large_oracle_d2_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd2_large__d2_large_oracle_d2_seed2023.log |
| oracle_ratio20_d2_large_oracle_d0_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd2_large__d2_large_oracle_d0_seed2023.log |
| oracle_ratio20_d2_large_oracle_d1_seed2023 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd2_large__d2_large_oracle_d1_seed2023.log |
| joint_ratio20_d2_large_baseline_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd2_large__d2_large_baseline_seed2024.log |
| joint_ratio20_d2_large_comp1_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd2_large__d2_large_comp1_seed2024.log |
| oracle_ratio20_d2_large_oracle_d2_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd2_large__d2_large_oracle_d2_seed2024.log |
| oracle_ratio20_d2_large_oracle_d0_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd2_large__d2_large_oracle_d0_seed2024.log |
| oracle_ratio20_d2_large_oracle_d1_seed2024 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd2_large__d2_large_oracle_d1_seed2024.log |
| joint_ratio20_d2_large_baseline_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd2_large__d2_large_baseline_seed2025.log |
| joint_ratio20_d2_large_comp1_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_joint_dd2_large__d2_large_comp1_seed2025.log |
| oracle_ratio20_d2_large_oracle_d2_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd2_large__d2_large_oracle_d2_seed2025.log |
| oracle_ratio20_d2_large_oracle_d0_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd2_large__d2_large_oracle_d0_seed2025.log |
| oracle_ratio20_d2_large_oracle_d1_seed2025 | OK | metrics_summary.csv | P0-1_reuse | experiments/logs/p0_1_oracle_dd2_large__d2_large_oracle_d1_seed2025.log |

## 3. Raw metrics

Each cell value is the FINAL per-(domain,task) test AUC at best_epoch.
`per_dt_test_auc` layout: `[d0t0, d1t0, d2t0, d0t1, d1t1, d2t1]` (task-major, idx = t*3 + d) for joint runs; `[t0, t1]` for oracle runs.

### 3.1 Joint runs (56 fresh + 24 reused = 80 total)

| setting | method | ratio | seed | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | best_epoch | best_val_auc | fit_sec | peak_mem_mb | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d0_large | baseline | 1 | 2022 | 0.740425 | 0.753100 | 0.743716 | 0.757100 | 0.701889 | 0.720219 | 8 | 0.732642 | 149.0 | 942 | P0-2 |
| d0_large | baseline | 1 | 2023 | 0.745140 | 0.757233 | 0.746681 | 0.763788 | 0.715803 | 0.728868 | 5 | 0.731735 | 129.1 | 942 | P0-2 |
| d0_large | baseline | 1 | 2024 | 0.748478 | 0.762974 | 0.757333 | 0.778510 | 0.702703 | 0.723512 | 9 | 0.736692 | 164.3 | 942 | P0-2 |
| d0_large | baseline | 1 | 2025 | 0.740677 | 0.747884 | 0.746139 | 0.777235 | 0.696276 | 0.716359 | 9 | 0.734905 | 162.6 | 942 | P0-2 |
| d0_large | comp1 | 1 | 2022 | 0.743665 | 0.756338 | 0.747197 | 0.765877 | 0.711726 | 0.724561 | 7 | 0.734908 | 160.1 | 1158 | P0-2 |
| d0_large | comp1 | 1 | 2023 | 0.746716 | 0.757330 | 0.746362 | 0.765254 | 0.718524 | 0.733786 | 6 | 0.732018 | 154.6 | 1158 | P0-2 |
| d0_large | comp1 | 1 | 2024 | 0.751866 | 0.759351 | 0.755045 | 0.776827 | 0.704560 | 0.725839 | 6 | 0.735308 | 153.1 | 1158 | P0-2 |
| d0_large | comp1 | 1 | 2025 | 0.744126 | 0.751200 | 0.750846 | 0.773298 | 0.707495 | 0.727105 | 7 | 0.738127 | 161.5 | 1158 | P0-2 |
| d0_large | baseline | 5 | 2022 | 0.745281 | 0.756139 | 0.730717 | 0.750290 | 0.680084 | 0.704780 | 8 | 0.727977 | 152.2 | 942 | P0-2 |
| d0_large | baseline | 5 | 2023 | 0.747543 | 0.753833 | 0.741853 | 0.757156 | 0.709176 | 0.725342 | 5 | 0.728800 | 128.2 | 942 | P0-2 |
| d0_large | baseline | 5 | 2024 | 0.747835 | 0.753956 | 0.742321 | 0.757187 | 0.686955 | 0.713622 | 7 | 0.728401 | 147.4 | 942 | P0-2 |
| d0_large | baseline | 5 | 2025 | 0.745082 | 0.755546 | 0.740182 | 0.760795 | 0.686216 | 0.710076 | 8 | 0.729441 | 156.9 | 942 | P0-2 |
| d0_large | comp1 | 5 | 2022 | 0.748101 | 0.759768 | 0.728123 | 0.747532 | 0.679132 | 0.704429 | 11 | 0.730326 | 198.5 | 1158 | P0-2 |
| d0_large | comp1 | 5 | 2023 | 0.756208 | 0.763054 | 0.741072 | 0.762561 | 0.710458 | 0.725418 | 8 | 0.731864 | 171.0 | 1158 | P0-2 |
| d0_large | comp1 | 5 | 2024 | 0.753890 | 0.764480 | 0.746173 | 0.763186 | 0.689300 | 0.711377 | 7 | 0.731770 | 162.9 | 1158 | P0-2 |
| d0_large | comp1 | 5 | 2025 | 0.753036 | 0.762969 | 0.743245 | 0.761800 | 0.698583 | 0.722145 | 9 | 0.733606 | 181.7 | 1158 | P0-2 |
| d1_large | baseline | 5 | 2022 | 0.725481 | 0.738632 | 0.746366 | 0.765513 | 0.686375 | 0.703225 | 10 | 0.728028 | 166.6 | 942 | P0-2 |
| d1_large | baseline | 5 | 2023 | 0.727271 | 0.740918 | 0.743039 | 0.772034 | 0.713682 | 0.726424 | 5 | 0.730714 | 126.5 | 942 | P0-2 |
| d1_large | baseline | 5 | 2024 | 0.735924 | 0.741481 | 0.748088 | 0.768206 | 0.693547 | 0.718044 | 7 | 0.731554 | 148.2 | 942 | P0-2 |
| d1_large | baseline | 5 | 2025 | 0.730510 | 0.741567 | 0.750148 | 0.766781 | 0.694187 | 0.716278 | 8 | 0.729771 | 156.6 | 942 | P0-2 |
| d1_large | comp1 | 5 | 2022 | 0.739199 | 0.750757 | 0.752164 | 0.766771 | 0.704208 | 0.723045 | 9 | 0.734354 | 181.3 | 1158 | P0-2 |
| d1_large | comp1 | 5 | 2023 | 0.735075 | 0.749391 | 0.750794 | 0.772482 | 0.715663 | 0.734659 | 6 | 0.733068 | 152.8 | 1158 | P0-2 |
| d1_large | comp1 | 5 | 2024 | 0.744686 | 0.758558 | 0.755048 | 0.780083 | 0.702433 | 0.720134 | 8 | 0.736189 | 170.7 | 1158 | P0-2 |
| d1_large | comp1 | 5 | 2025 | 0.741215 | 0.749990 | 0.758812 | 0.776762 | 0.705206 | 0.726049 | 8 | 0.734210 | 170.1 | 1158 | P0-2 |
| d2_large | baseline | 5 | 2022 | 0.736343 | 0.752332 | 0.730330 | 0.746978 | 0.706984 | 0.721973 | 8 | 0.730667 | 150.6 | 942 | P0-2 |
| d2_large | baseline | 5 | 2023 | 0.732775 | 0.741524 | 0.734813 | 0.755827 | 0.716823 | 0.725708 | 5 | 0.726668 | 127.7 | 942 | P0-2 |
| d2_large | baseline | 5 | 2024 | 0.743743 | 0.757910 | 0.731061 | 0.750084 | 0.710432 | 0.737373 | 8 | 0.728901 | 156.2 | 942 | P0-2 |
| d2_large | baseline | 5 | 2025 | 0.728719 | 0.740732 | 0.737954 | 0.759036 | 0.707006 | 0.732056 | 8 | 0.728872 | 155.4 | 942 | P0-2 |
| d2_large | comp1 | 5 | 2022 | 0.743674 | 0.759097 | 0.735256 | 0.756048 | 0.715353 | 0.737115 | 7 | 0.734305 | 162.4 | 1158 | P0-2 |
| d2_large | comp1 | 5 | 2023 | 0.742919 | 0.757473 | 0.735947 | 0.757485 | 0.721024 | 0.741774 | 6 | 0.727841 | 153.0 | 1158 | P0-2 |
| d2_large | comp1 | 5 | 2024 | 0.744456 | 0.759063 | 0.741813 | 0.761806 | 0.719589 | 0.740847 | 9 | 0.735332 | 181.6 | 1158 | P0-2 |
| d2_large | comp1 | 5 | 2025 | 0.737175 | 0.743270 | 0.740885 | 0.757445 | 0.719788 | 0.739463 | 7 | 0.733668 | 162.4 | 1158 | P0-2 |
| d0_large | baseline | 50 | 2022 | 0.742451 | 0.759712 | 0.729731 | 0.751636 | 0.593696 | 0.665226 | 6 | 0.703398 | 134.2 | 942 | P0-2 |
| d0_large | baseline | 50 | 2023 | 0.738298 | 0.746986 | 0.724376 | 0.742837 | 0.644277 | 0.677056 | 5 | 0.711049 | 127.2 | 942 | P0-2 |
| d0_large | baseline | 50 | 2024 | 0.748197 | 0.758929 | 0.726061 | 0.750921 | 0.662026 | 0.700791 | 5 | 0.720019 | 132.5 | 942 | P0-2 |
| d0_large | baseline | 50 | 2025 | 0.738381 | 0.747552 | 0.725394 | 0.742865 | 0.670099 | 0.683284 | 7 | 0.714317 | 146.4 | 942 | P0-2 |
| d0_large | comp1 | 50 | 2022 | 0.746566 | 0.760855 | 0.732876 | 0.753273 | 0.667734 | 0.677386 | 6 | 0.723018 | 153.6 | 1158 | P0-2 |
| d0_large | comp1 | 50 | 2023 | 0.748490 | 0.762759 | 0.735868 | 0.752157 | 0.676684 | 0.694747 | 5 | 0.723189 | 144.9 | 1158 | P0-2 |
| d0_large | comp1 | 50 | 2024 | 0.746164 | 0.763924 | 0.734598 | 0.752786 | 0.677103 | 0.703116 | 4 | 0.723429 | 135.0 | 1158 | P0-2 |
| d0_large | comp1 | 50 | 2025 | 0.749702 | 0.759701 | 0.732747 | 0.755689 | 0.678653 | 0.685911 | 10 | 0.723770 | 186.2 | 1158 | P0-2 |
| d1_large | baseline | 50 | 2022 | 0.670082 | 0.669884 | 0.714973 | 0.733097 | 0.673407 | 0.688616 | 8 | 0.692016 | 151.4 | 942 | P0-2 |
| d1_large | baseline | 50 | 2023 | 0.679943 | 0.684304 | 0.737243 | 0.764433 | 0.704071 | 0.721772 | 5 | 0.711255 | 128.3 | 942 | P0-2 |
| d1_large | baseline | 50 | 2024 | 0.657808 | 0.669204 | 0.752155 | 0.774132 | 0.696247 | 0.715842 | 9 | 0.711900 | 164.2 | 942 | P0-2 |
| d1_large | baseline | 50 | 2025 | 0.674107 | 0.657300 | 0.730230 | 0.745426 | 0.683004 | 0.703589 | 11 | 0.698322 | 181.1 | 942 | P0-2 |
| d1_large | comp1 | 50 | 2022 | 0.701921 | 0.711804 | 0.751384 | 0.774962 | 0.705621 | 0.723966 | 9 | 0.728167 | 180.2 | 1158 | P0-2 |
| d1_large | comp1 | 50 | 2023 | 0.692237 | 0.702239 | 0.747103 | 0.766496 | 0.713226 | 0.740060 | 7 | 0.723884 | 162.5 | 1158 | P0-2 |
| d1_large | comp1 | 50 | 2024 | 0.697562 | 0.708934 | 0.753127 | 0.773444 | 0.701188 | 0.723661 | 8 | 0.727570 | 172.5 | 1158 | P0-2 |
| d1_large | comp1 | 50 | 2025 | 0.707748 | 0.715197 | 0.748880 | 0.771097 | 0.700756 | 0.725464 | 8 | 0.728484 | 172.1 | 1158 | P0-2 |
| d2_large | baseline | 50 | 2022 | 0.728085 | 0.740598 | 0.634127 | 0.681153 | 0.703884 | 0.710698 | 10 | 0.704192 | 166.4 | 942 | P0-2 |
| d2_large | baseline | 50 | 2023 | 0.720398 | 0.740634 | 0.667516 | 0.686272 | 0.714509 | 0.737429 | 10 | 0.708847 | 168.5 | 942 | P0-2 |
| d2_large | baseline | 50 | 2024 | 0.731686 | 0.743303 | 0.692487 | 0.719951 | 0.703993 | 0.723525 | 6 | 0.713424 | 140.8 | 942 | P0-2 |
| d2_large | baseline | 50 | 2025 | 0.716377 | 0.727744 | 0.684753 | 0.706441 | 0.693327 | 0.722188 | 8 | 0.713974 | 155.7 | 942 | P0-2 |
| d2_large | comp1 | 50 | 2022 | 0.738498 | 0.751864 | 0.704250 | 0.724092 | 0.715794 | 0.733486 | 10 | 0.725630 | 188.0 | 1158 | P0-2 |
| d2_large | comp1 | 50 | 2023 | 0.736904 | 0.751331 | 0.689999 | 0.709991 | 0.710261 | 0.735058 | 7 | 0.717957 | 164.0 | 1158 | P0-2 |
| d2_large | comp1 | 50 | 2024 | 0.736614 | 0.753030 | 0.710431 | 0.733251 | 0.718232 | 0.737006 | 7 | 0.726458 | 162.6 | 1158 | P0-2 |
| d2_large | comp1 | 50 | 2025 | 0.730530 | 0.743473 | 0.700862 | 0.721482 | 0.716426 | 0.742174 | 6 | 0.725169 | 153.1 | 1158 | P0-2 |

Plus 24 reused P0-1 20:1 joint rows:

| setting | method | ratio | seed | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | best_epoch | best_val_auc | fit_sec | peak_mem_mb | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d0_large | baseline | 20 | 2022 | 0.744805 | 0.760342 | 0.715168 | 0.759552 | 0.667817 | 0.709900 | 5 | 0.716812 | 132.9 | 942 | P0-1_reuse |
| d0_large | baseline | 20 | 2023 | 0.742450 | 0.752645 | 0.729762 | 0.750325 | 0.680681 | 0.691520 | 5 | 0.718517 | 130.8 | 942 | P0-1_reuse |
| d0_large | baseline | 20 | 2024 | 0.743929 | 0.752081 | 0.727216 | 0.740114 | 0.670028 | 0.684037 | 6 | 0.719863 | 148.3 | 942 | P0-1_reuse |
| d0_large | baseline | 20 | 2025 | 0.737954 | 0.753651 | 0.730519 | 0.756387 | 0.677076 | 0.695968 | 5 | 0.721704 | 132.6 | 942 | P0-1_reuse |
| d0_large | comp1 | 20 | 2022 | 0.752109 | 0.766892 | 0.733562 | 0.756295 | 0.685938 | 0.701321 | 5 | 0.727913 | 158.7 | 1158 | P0-1_reuse |
| d0_large | comp1 | 20 | 2023 | 0.746009 | 0.760200 | 0.734325 | 0.753279 | 0.687966 | 0.698825 | 6 | 0.724639 | 164.7 | 1158 | P0-1_reuse |
| d0_large | comp1 | 20 | 2024 | 0.753419 | 0.763851 | 0.734786 | 0.752203 | 0.681045 | 0.700727 | 7 | 0.728874 | 180.8 | 1158 | P0-1_reuse |
| d0_large | comp1 | 20 | 2025 | 0.748254 | 0.753689 | 0.735436 | 0.751116 | 0.680037 | 0.697726 | 9 | 0.729329 | 197.5 | 1158 | P0-1_reuse |
| d1_large | baseline | 20 | 2022 | 0.680397 | 0.692427 | 0.736124 | 0.754247 | 0.679137 | 0.696439 | 10 | 0.708076 | 174.0 | 942 | P0-1_reuse |
| d1_large | baseline | 20 | 2023 | 0.704687 | 0.700349 | 0.734728 | 0.762332 | 0.700723 | 0.719340 | 4 | 0.713998 | 124.4 | 942 | P0-1_reuse |
| d1_large | baseline | 20 | 2024 | 0.689675 | 0.715012 | 0.744734 | 0.767010 | 0.679503 | 0.698190 | 9 | 0.716466 | 172.8 | 942 | P0-1_reuse |
| d1_large | baseline | 20 | 2025 | 0.710226 | 0.716606 | 0.736400 | 0.752812 | 0.685774 | 0.705327 | 8 | 0.714746 | 162.7 | 942 | P0-1_reuse |
| d1_large | comp1 | 20 | 2022 | 0.716135 | 0.735508 | 0.754369 | 0.776698 | 0.702839 | 0.727588 | 10 | 0.729606 | 226.3 | 1158 | P0-1_reuse |
| d1_large | comp1 | 20 | 2023 | 0.718923 | 0.728640 | 0.750012 | 0.773408 | 0.712431 | 0.736244 | 4 | 0.728636 | 151.5 | 1158 | P0-1_reuse |
| d1_large | comp1 | 20 | 2024 | 0.716839 | 0.733946 | 0.755866 | 0.781193 | 0.695390 | 0.718077 | 8 | 0.733783 | 188.8 | 1158 | P0-1_reuse |
| d1_large | comp1 | 20 | 2025 | 0.723033 | 0.735340 | 0.748689 | 0.774464 | 0.701278 | 0.729412 | 9 | 0.727815 | 194.8 | 1158 | P0-1_reuse |
| d2_large | baseline | 20 | 2022 | 0.724800 | 0.739181 | 0.689988 | 0.723038 | 0.699584 | 0.707858 | 7 | 0.714147 | 149.1 | 942 | P0-1_reuse |
| d2_large | baseline | 20 | 2023 | 0.722816 | 0.733242 | 0.709592 | 0.728265 | 0.702806 | 0.713985 | 5 | 0.714562 | 130.1 | 942 | P0-1_reuse |
| d2_large | baseline | 20 | 2024 | 0.712105 | 0.727859 | 0.688021 | 0.714728 | 0.705571 | 0.727365 | 10 | 0.712735 | 179.6 | 942 | P0-1_reuse |
| d2_large | baseline | 20 | 2025 | 0.722838 | 0.735542 | 0.705428 | 0.731413 | 0.703678 | 0.728359 | 7 | 0.718395 | 150.2 | 942 | P0-1_reuse |
| d2_large | comp1 | 20 | 2022 | 0.733294 | 0.748323 | 0.723888 | 0.749299 | 0.717826 | 0.734498 | 7 | 0.729776 | 177.7 | 1158 | P0-1_reuse |
| d2_large | comp1 | 20 | 2023 | 0.738126 | 0.753693 | 0.723510 | 0.742229 | 0.722881 | 0.739083 | 7 | 0.728601 | 178.9 | 1158 | P0-1_reuse |
| d2_large | comp1 | 20 | 2024 | 0.737152 | 0.752705 | 0.720068 | 0.739627 | 0.724294 | 0.744391 | 9 | 0.732595 | 206.2 | 1158 | P0-1_reuse |
| d2_large | comp1 | 20 | 2025 | 0.734999 | 0.746929 | 0.718087 | 0.731852 | 0.717119 | 0.737570 | 6 | 0.728609 | 153.3 | 1158 | P0-1_reuse |

### 3.2 Oracle runs (84 fresh + 36 reused = 120 total)

| setting | oracle_domain | ratio | seed | t0 | t1 | best_epoch | best_val_auc | fit_sec | peak_mem_mb | source |
|---|---|---|---|---|---|---|---|---|---|---|
| d0_large | oracle_d0 | 1 | 2022 | 0.728181 | 0.737738 | 12 | 0.734841 | 63.8 | 732 | P0-2 |
| d0_large | oracle_d0 | 1 | 2023 | 0.725611 | 0.738946 | 22 | 0.734562 | 88.8 | 732 | P0-2 |
| d0_large | oracle_d0 | 1 | 2024 | 0.729189 | 0.739514 | 23 | 0.734800 | 92.4 | 732 | P0-2 |
| d0_large | oracle_d0 | 1 | 2025 | 0.728127 | 0.741765 | 16 | 0.734040 | 74.5 | 732 | P0-2 |
| d0_large | oracle_d1 | 1 | 2022 | 0.720155 | 0.726044 | 30 | 0.726103 | 128.1 | 732 | P0-2 |
| d0_large | oracle_d1 | 1 | 2023 | 0.726770 | 0.746380 | 22 | 0.719399 | 102.7 | 732 | P0-2 |
| d0_large | oracle_d1 | 1 | 2024 | 0.735421 | 0.754162 | 17 | 0.723908 | 87.4 | 732 | P0-2 |
| d0_large | oracle_d1 | 1 | 2025 | 0.733629 | 0.758218 | 13 | 0.728138 | 77.1 | 732 | P0-2 |
| d0_large | oracle_d2 | 1 | 2022 | 0.697849 | 0.715093 | 17 | 0.722019 | 84.2 | 732 | P0-2 |
| d0_large | oracle_d2 | 1 | 2023 | 0.709567 | 0.729829 | 18 | 0.721816 | 86.4 | 732 | P0-2 |
| d0_large | oracle_d2 | 1 | 2024 | 0.689078 | 0.712786 | 17 | 0.721718 | 83.0 | 732 | P0-2 |
| d0_large | oracle_d2 | 1 | 2025 | 0.688336 | 0.703627 | 11 | 0.724845 | 66.6 | 732 | P0-2 |
| d0_large | oracle_d0 | 5 | 2022 | 0.740537 | 0.755959 | 12 | 0.746113 | 81.2 | 733 | P0-2 |
| d0_large | oracle_d0 | 5 | 2023 | 0.740073 | 0.755422 | 34 | 0.750387 | 162.4 | 733 | P0-2 |
| d0_large | oracle_d0 | 5 | 2024 | 0.748887 | 0.759045 | 29 | 0.745567 | 148.1 | 733 | P0-2 |
| d0_large | oracle_d0 | 5 | 2025 | 0.749456 | 0.763409 | 28 | 0.747148 | 141.5 | 733 | P0-2 |
| d0_large | oracle_d1 | 5 | 2022 | 0.709494 | 0.731010 | 14 | 0.710074 | 68.0 | 673 | P0-2 |
| d0_large | oracle_d1 | 5 | 2023 | 0.705465 | 0.722806 | 27 | 0.701910 | 106.5 | 673 | P0-2 |
| d0_large | oracle_d1 | 5 | 2024 | 0.711872 | 0.724575 | 34 | 0.705837 | 126.3 | 673 | P0-2 |
| d0_large | oracle_d1 | 5 | 2025 | 0.718893 | 0.742497 | 7 | 0.707477 | 52.5 | 673 | P0-2 |
| d0_large | oracle_d2 | 5 | 2022 | 0.673053 | 0.696111 | 12 | 0.697603 | 50.5 | 459 | P0-2 |
| d0_large | oracle_d2 | 5 | 2023 | 0.679159 | 0.698425 | 15 | 0.698900 | 62.5 | 459 | P0-2 |
| d0_large | oracle_d2 | 5 | 2024 | 0.652747 | 0.682589 | 15 | 0.698794 | 62.1 | 459 | P0-2 |
| d0_large | oracle_d2 | 5 | 2025 | 0.663923 | 0.699643 | 12 | 0.697896 | 55.4 | 459 | P0-2 |
| d1_large | oracle_d0 | 5 | 2022 | 0.692509 | 0.705462 | 15 | 0.714509 | 50.1 | 459 | P0-2 |
| d1_large | oracle_d0 | 5 | 2023 | 0.697250 | 0.715117 | 4 | 0.710556 | 33.8 | 459 | P0-2 |
| d1_large | oracle_d0 | 5 | 2024 | 0.675865 | 0.690843 | 7 | 0.700395 | 40.2 | 459 | P0-2 |
| d1_large | oracle_d0 | 5 | 2025 | 0.702981 | 0.710253 | 11 | 0.712715 | 48.2 | 459 | P0-2 |
| d1_large | oracle_d1 | 5 | 2022 | 0.745601 | 0.764411 | 21 | 0.736078 | 126.7 | 733 | P0-2 |
| d1_large | oracle_d1 | 5 | 2023 | 0.742032 | 0.769242 | 15 | 0.731176 | 107.3 | 733 | P0-2 |
| d1_large | oracle_d1 | 5 | 2024 | 0.749460 | 0.771170 | 10 | 0.734503 | 87.2 | 733 | P0-2 |
| d1_large | oracle_d1 | 5 | 2025 | 0.747072 | 0.770975 | 22 | 0.738087 | 133.6 | 733 | P0-2 |
| d1_large | oracle_d2 | 5 | 2022 | 0.682275 | 0.703276 | 13 | 0.706987 | 60.5 | 673 | P0-2 |
| d1_large | oracle_d2 | 5 | 2023 | 0.696239 | 0.718739 | 28 | 0.708385 | 103.2 | 673 | P0-2 |
| d1_large | oracle_d2 | 5 | 2024 | 0.678416 | 0.691152 | 9 | 0.707982 | 55.4 | 673 | P0-2 |
| d1_large | oracle_d2 | 5 | 2025 | 0.675266 | 0.697049 | 20 | 0.708214 | 82.3 | 673 | P0-2 |
| d2_large | oracle_d0 | 5 | 2022 | 0.718064 | 0.729954 | 21 | 0.728421 | 73.0 | 673 | P0-2 |
| d2_large | oracle_d0 | 5 | 2023 | 0.708633 | 0.725244 | 5 | 0.721669 | 41.7 | 673 | P0-2 |
| d2_large | oracle_d0 | 5 | 2024 | 0.689417 | 0.704127 | 7 | 0.707776 | 46.2 | 673 | P0-2 |
| d2_large | oracle_d0 | 5 | 2025 | 0.702710 | 0.714621 | 9 | 0.718677 | 50.6 | 673 | P0-2 |
| d2_large | oracle_d1 | 5 | 2022 | 0.694943 | 0.714173 | 14 | 0.696026 | 58.9 | 459 | P0-2 |
| d2_large | oracle_d1 | 5 | 2023 | 0.697188 | 0.724101 | 13 | 0.690772 | 61.7 | 459 | P0-2 |
| d2_large | oracle_d1 | 5 | 2024 | 0.693055 | 0.721827 | 17 | 0.690699 | 71.4 | 459 | P0-2 |
| d2_large | oracle_d1 | 5 | 2025 | 0.703715 | 0.729190 | 12 | 0.697048 | 59.1 | 459 | P0-2 |
| d2_large | oracle_d2 | 5 | 2022 | 0.709169 | 0.728520 | 8 | 0.733721 | 70.9 | 733 | P0-2 |
| d2_large | oracle_d2 | 5 | 2023 | 0.715711 | 0.739593 | 16 | 0.733621 | 106.4 | 733 | P0-2 |
| d2_large | oracle_d2 | 5 | 2024 | 0.721768 | 0.739699 | 19 | 0.738002 | 117.8 | 733 | P0-2 |
| d2_large | oracle_d2 | 5 | 2025 | 0.709934 | 0.729181 | 18 | 0.734976 | 113.6 | 733 | P0-2 |
| d0_large | oracle_d0 | 50 | 2022 | 0.750812 | 0.765480 | 26 | 0.747680 | 146.6 | 733 | P0-2 |
| d0_large | oracle_d0 | 50 | 2023 | 0.751578 | 0.762610 | 20 | 0.749300 | 127.8 | 733 | P0-2 |
| d0_large | oracle_d0 | 50 | 2024 | 0.747599 | 0.757183 | 10 | 0.748298 | 88.8 | 733 | P0-2 |
| d0_large | oracle_d0 | 50 | 2025 | 0.751081 | 0.763873 | 8 | 0.749503 | 80.5 | 733 | P0-2 |
| d0_large | oracle_d1 | 50 | 2022 | 0.706298 | 0.727423 | 14 | 0.705128 | 70.4 | 624 | P0-2 |
| d0_large | oracle_d1 | 50 | 2023 | 0.701176 | 0.734068 | 14 | 0.696278 | 69.9 | 624 | P0-2 |
| d0_large | oracle_d1 | 50 | 2024 | 0.707244 | 0.724003 | 33 | 0.705276 | 120.1 | 624 | P0-2 |
| d0_large | oracle_d1 | 50 | 2025 | 0.698913 | 0.715489 | 25 | 0.706539 | 99.1 | 624 | P0-2 |
| d0_large | oracle_d2 | 50 | 2022 | 0.610259 | 0.612607 | 8 | 0.623286 | 33.7 | 391 | P0-2 |
| d0_large | oracle_d2 | 50 | 2023 | 0.603663 | 0.617540 | 10 | 0.633422 | 37.3 | 391 | P0-2 |
| d0_large | oracle_d2 | 50 | 2024 | 0.613099 | 0.620468 | 6 | 0.627605 | 30.5 | 391 | P0-2 |
| d0_large | oracle_d2 | 50 | 2025 | 0.606141 | 0.621258 | 4 | 0.623601 | 26.9 | 391 | P0-2 |
| d1_large | oracle_d0 | 50 | 2022 | 0.584222 | 0.601662 | 10 | 0.646756 | 25.6 | 297 | P0-2 |
| d1_large | oracle_d0 | 50 | 2023 | 0.618183 | 0.637969 | 2 | 0.636599 | 21.0 | 297 | P0-2 |
| d1_large | oracle_d0 | 50 | 2024 | 0.588531 | 0.600971 | 9 | 0.624271 | 31.4 | 297 | P0-2 |
| d1_large | oracle_d0 | 50 | 2025 | 0.606043 | 0.616698 | 9 | 0.642478 | 31.2 | 297 | P0-2 |
| d1_large | oracle_d1 | 50 | 2022 | 0.752541 | 0.773333 | 10 | 0.739886 | 97.1 | 733 | P0-2 |
| d1_large | oracle_d1 | 50 | 2023 | 0.751134 | 0.774599 | 13 | 0.731556 | 111.4 | 733 | P0-2 |
| d1_large | oracle_d1 | 50 | 2024 | 0.753325 | 0.774757 | 9 | 0.737806 | 93.4 | 733 | P0-2 |
| d1_large | oracle_d1 | 50 | 2025 | 0.752387 | 0.771814 | 9 | 0.741349 | 93.6 | 733 | P0-2 |
| d1_large | oracle_d2 | 50 | 2022 | 0.681414 | 0.704292 | 13 | 0.706211 | 63.0 | 624 | P0-2 |
| d1_large | oracle_d2 | 50 | 2023 | 0.685622 | 0.713101 | 28 | 0.704114 | 99.7 | 624 | P0-2 |
| d1_large | oracle_d2 | 50 | 2024 | 0.670413 | 0.685483 | 9 | 0.703821 | 54.0 | 624 | P0-2 |
| d1_large | oracle_d2 | 50 | 2025 | 0.669484 | 0.694634 | 20 | 0.706636 | 80.8 | 624 | P0-2 |
| d2_large | oracle_d0 | 50 | 2022 | 0.715709 | 0.728684 | 23 | 0.725106 | 79.9 | 624 | P0-2 |
| d2_large | oracle_d0 | 50 | 2023 | 0.705732 | 0.722606 | 6 | 0.717822 | 42.1 | 624 | P0-2 |
| d2_large | oracle_d0 | 50 | 2024 | 0.667951 | 0.702976 | 5 | 0.709840 | 40.4 | 624 | P0-2 |
| d2_large | oracle_d0 | 50 | 2025 | 0.702888 | 0.716117 | 12 | 0.720902 | 55.4 | 624 | P0-2 |
| d2_large | oracle_d1 | 50 | 2022 | 0.601201 | 0.619916 | 14 | 0.620948 | 45.9 | 423 | P0-2 |
| d2_large | oracle_d1 | 50 | 2023 | 0.610246 | 0.625104 | 12 | 0.618078 | 43.2 | 423 | P0-2 |
| d2_large | oracle_d1 | 50 | 2024 | 0.605671 | 0.627791 | 12 | 0.601420 | 43.2 | 423 | P0-2 |
| d2_large | oracle_d1 | 50 | 2025 | 0.614774 | 0.632876 | 13 | 0.609185 | 46.1 | 423 | P0-2 |
| d2_large | oracle_d2 | 50 | 2022 | 0.719238 | 0.739353 | 9 | 0.738843 | 89.3 | 733 | P0-2 |
| d2_large | oracle_d2 | 50 | 2023 | 0.726436 | 0.746738 | 11 | 0.739139 | 97.6 | 733 | P0-2 |
| d2_large | oracle_d2 | 50 | 2024 | 0.723728 | 0.743415 | 10 | 0.741820 | 94.2 | 733 | P0-2 |
| d2_large | oracle_d2 | 50 | 2025 | 0.720659 | 0.741145 | 6 | 0.736681 | 77.2 | 733 | P0-2 |

Plus 36 reused P0-1 20:1 oracle rows:

| setting | oracle_domain | ratio | seed | t0 | t1 | best_epoch | best_val_auc | fit_sec | peak_mem_mb | source |
|---|---|---|---|---|---|---|---|---|---|---|
| d0_large | oracle_d0 | 20 | 2022 | 0.750812 | 0.765480 | 26 | 0.747680 | 146.2 | 733 | P0-1_reuse |
| d0_large | oracle_d0 | 20 | 2023 | 0.751578 | 0.762610 | 20 | 0.749300 | 128.0 | 733 | P0-1_reuse |
| d0_large | oracle_d0 | 20 | 2024 | 0.747599 | 0.757183 | 10 | 0.748298 | 88.0 | 733 | P0-1_reuse |
| d0_large | oracle_d0 | 20 | 2025 | 0.751081 | 0.763873 | 8 | 0.749503 | 83.5 | 733 | P0-1_reuse |
| d0_large | oracle_d1 | 20 | 2022 | 0.706154 | 0.729746 | 13 | 0.703626 | 58.8 | 547 | P0-1_reuse |
| d0_large | oracle_d1 | 20 | 2023 | 0.695967 | 0.723419 | 15 | 0.694361 | 62.7 | 547 | P0-1_reuse |
| d0_large | oracle_d1 | 20 | 2024 | 0.698988 | 0.711619 | 31 | 0.702684 | 108.6 | 547 | P0-1_reuse |
| d0_large | oracle_d1 | 20 | 2025 | 0.706204 | 0.735206 | 13 | 0.701590 | 65.2 | 547 | P0-1_reuse |
| d0_large | oracle_d2 | 20 | 2022 | 0.637804 | 0.652942 | 13 | 0.658153 | 39.0 | 391 | P0-1_reuse |
| d0_large | oracle_d2 | 20 | 2023 | 0.634457 | 0.636952 | 11 | 0.654765 | 40.6 | 391 | P0-1_reuse |
| d0_large | oracle_d2 | 20 | 2024 | 0.640219 | 0.644213 | 6 | 0.657019 | 32.7 | 391 | P0-1_reuse |
| d0_large | oracle_d2 | 20 | 2025 | 0.634128 | 0.652421 | 6 | 0.658630 | 27.9 | 391 | P0-1_reuse |
| d1_large | oracle_d0 | 20 | 2022 | 0.636472 | 0.662045 | 13 | 0.671383 | 32.3 | 297 | P0-1_reuse |
| d1_large | oracle_d0 | 20 | 2023 | 0.656473 | 0.658416 | 5 | 0.670251 | 26.8 | 297 | P0-1_reuse |
| d1_large | oracle_d0 | 20 | 2024 | 0.638067 | 0.667203 | 15 | 0.663175 | 40.6 | 297 | P0-1_reuse |
| d1_large | oracle_d0 | 20 | 2025 | 0.640806 | 0.658699 | 10 | 0.662840 | 34.2 | 297 | P0-1_reuse |
| d1_large | oracle_d1 | 20 | 2022 | 0.752541 | 0.773333 | 10 | 0.739886 | 95.4 | 733 | P0-1_reuse |
| d1_large | oracle_d1 | 20 | 2023 | 0.751134 | 0.774599 | 13 | 0.731556 | 111.5 | 733 | P0-1_reuse |
| d1_large | oracle_d1 | 20 | 2024 | 0.753325 | 0.774757 | 9 | 0.737806 | 93.4 | 733 | P0-1_reuse |
| d1_large | oracle_d1 | 20 | 2025 | 0.752387 | 0.771814 | 9 | 0.741349 | 93.0 | 733 | P0-1_reuse |
| d1_large | oracle_d2 | 20 | 2022 | 0.682215 | 0.701624 | 12 | 0.702910 | 53.5 | 547 | P0-1_reuse |
| d1_large | oracle_d2 | 20 | 2023 | 0.677873 | 0.706023 | 23 | 0.700562 | 83.0 | 547 | P0-1_reuse |
| d1_large | oracle_d2 | 20 | 2024 | 0.669775 | 0.687019 | 9 | 0.703211 | 51.1 | 547 | P0-1_reuse |
| d1_large | oracle_d2 | 20 | 2025 | 0.668455 | 0.700159 | 11 | 0.705257 | 56.1 | 547 | P0-1_reuse |
| d2_large | oracle_d0 | 20 | 2022 | 0.694421 | 0.713797 | 13 | 0.715668 | 49.1 | 547 | P0-1_reuse |
| d2_large | oracle_d0 | 20 | 2023 | 0.702204 | 0.717521 | 4 | 0.712556 | 36.0 | 547 | P0-1_reuse |
| d2_large | oracle_d0 | 20 | 2024 | 0.684251 | 0.704708 | 7 | 0.704760 | 42.0 | 547 | P0-1_reuse |
| d2_large | oracle_d0 | 20 | 2025 | 0.697780 | 0.710745 | 12 | 0.719825 | 53.1 | 547 | P0-1_reuse |
| d2_large | oracle_d1 | 20 | 2022 | 0.654050 | 0.671056 | 11 | 0.660189 | 42.6 | 423 | P0-1_reuse |
| d2_large | oracle_d1 | 20 | 2023 | 0.660158 | 0.673999 | 3 | 0.644423 | 32.4 | 423 | P0-1_reuse |
| d2_large | oracle_d1 | 20 | 2024 | 0.638062 | 0.655520 | 28 | 0.643312 | 78.3 | 423 | P0-1_reuse |
| d2_large | oracle_d1 | 20 | 2025 | 0.640342 | 0.663728 | 13 | 0.643341 | 51.4 | 423 | P0-1_reuse |
| d2_large | oracle_d2 | 20 | 2022 | 0.719238 | 0.739353 | 9 | 0.738843 | 86.2 | 733 | P0-1_reuse |
| d2_large | oracle_d2 | 20 | 2023 | 0.726436 | 0.746738 | 11 | 0.739139 | 99.7 | 733 | P0-1_reuse |
| d2_large | oracle_d2 | 20 | 2024 | 0.723728 | 0.743415 | 10 | 0.741820 | 95.4 | 733 | P0-1_reuse |
| d2_large | oracle_d2 | 20 | 2025 | 0.720659 | 0.741145 | 6 | 0.736681 | 76.9 | 733 | P0-1_reuse |

## 4. Mean/std (mathematical only)

### 4.1 Joint per-(ratio, setting, method, cell): mean/std/min/max across 4 seeds

Fresh P0-2 rows (ratios 1, 5, 50):

| ratio | setting | method | cell | mean | std | min | max | n_seeds |
|---|---|---|---|---|---|---|---|---|
| 1 | d0_large | baseline | d0t0 | 0.743680 | 0.003345 | 0.740425 | 0.748478 | 4 |
| 1 | d0_large | baseline | d0t1 | 0.755298 | 0.005533 | 0.747884 | 0.762974 | 4 |
| 1 | d0_large | baseline | d1t0 | 0.748467 | 0.005239 | 0.743716 | 0.757333 | 4 |
| 1 | d0_large | baseline | d1t1 | 0.769158 | 0.009041 | 0.757100 | 0.778510 | 4 |
| 1 | d0_large | baseline | d2t0 | 0.704168 | 0.007159 | 0.696276 | 0.715803 | 4 |
| 1 | d0_large | baseline | d2t1 | 0.722240 | 0.004589 | 0.716359 | 0.728868 | 4 |
| 1 | d0_large | comp1 | d0t0 | 0.746593 | 0.003259 | 0.743665 | 0.751866 | 4 |
| 1 | d0_large | comp1 | d0t1 | 0.756055 | 0.003006 | 0.751200 | 0.759351 | 4 |
| 1 | d0_large | comp1 | d1t0 | 0.749862 | 0.003435 | 0.746362 | 0.755045 | 4 |
| 1 | d0_large | comp1 | d1t1 | 0.770314 | 0.004915 | 0.765254 | 0.776827 | 4 |
| 1 | d0_large | comp1 | d2t0 | 0.710576 | 0.005248 | 0.704560 | 0.718524 | 4 |
| 1 | d0_large | comp1 | d2t1 | 0.727823 | 0.003558 | 0.724561 | 0.733786 | 4 |
| 5 | d0_large | baseline | d0t0 | 0.746435 | 0.001260 | 0.745082 | 0.747835 | 4 |
| 5 | d0_large | baseline | d0t1 | 0.754868 | 0.000997 | 0.753833 | 0.756139 | 4 |
| 5 | d0_large | baseline | d1t0 | 0.738768 | 0.004716 | 0.730717 | 0.742321 | 4 |
| 5 | d0_large | baseline | d1t1 | 0.756357 | 0.003802 | 0.750290 | 0.760795 | 4 |
| 5 | d0_large | baseline | d2t0 | 0.690608 | 0.011047 | 0.680084 | 0.709176 | 4 |
| 5 | d0_large | baseline | d2t1 | 0.713455 | 0.007550 | 0.704780 | 0.725342 | 4 |
| 5 | d0_large | comp1 | d0t0 | 0.752809 | 0.002955 | 0.748101 | 0.756208 | 4 |
| 5 | d0_large | comp1 | d0t1 | 0.762568 | 0.001724 | 0.759768 | 0.764480 | 4 |
| 5 | d0_large | comp1 | d1t0 | 0.739653 | 0.006899 | 0.728123 | 0.746173 | 4 |
| 5 | d0_large | comp1 | d1t1 | 0.758770 | 0.006507 | 0.747532 | 0.763186 | 4 |
| 5 | d0_large | comp1 | d2t0 | 0.694368 | 0.011559 | 0.679132 | 0.710458 | 4 |
| 5 | d0_large | comp1 | d2t1 | 0.715842 | 0.008391 | 0.704429 | 0.725418 | 4 |
| 5 | d1_large | baseline | d0t0 | 0.729796 | 0.003970 | 0.725481 | 0.735924 | 4 |
| 5 | d1_large | baseline | d0t1 | 0.740649 | 0.001191 | 0.738632 | 0.741567 | 4 |
| 5 | d1_large | baseline | d1t0 | 0.746910 | 0.002605 | 0.743039 | 0.750148 | 4 |
| 5 | d1_large | baseline | d1t1 | 0.768134 | 0.002445 | 0.765513 | 0.772034 | 4 |
| 5 | d1_large | baseline | d2t0 | 0.696948 | 0.010137 | 0.686375 | 0.713682 | 4 |
| 5 | d1_large | baseline | d2t1 | 0.715993 | 0.008308 | 0.703225 | 0.726424 | 4 |
| 5 | d1_large | comp1 | d0t0 | 0.740044 | 0.003476 | 0.735075 | 0.744686 | 4 |
| 5 | d1_large | comp1 | d0t1 | 0.752174 | 0.003717 | 0.749391 | 0.758558 | 4 |
| 5 | d1_large | comp1 | d1t0 | 0.754205 | 0.003071 | 0.750794 | 0.758812 | 4 |
| 5 | d1_large | comp1 | d1t1 | 0.774025 | 0.004980 | 0.766771 | 0.780083 | 4 |
| 5 | d1_large | comp1 | d2t0 | 0.706878 | 0.005169 | 0.702433 | 0.715663 | 4 |
| 5 | d1_large | comp1 | d2t1 | 0.725972 | 0.005434 | 0.720134 | 0.734659 | 4 |
| 5 | d2_large | baseline | d0t0 | 0.735395 | 0.005523 | 0.728719 | 0.743743 | 4 |
| 5 | d2_large | baseline | d0t1 | 0.748124 | 0.007275 | 0.740732 | 0.757910 | 4 |
| 5 | d2_large | baseline | d1t0 | 0.733540 | 0.003064 | 0.730330 | 0.737954 | 4 |
| 5 | d2_large | baseline | d1t1 | 0.752981 | 0.004722 | 0.746978 | 0.759036 | 4 |
| 5 | d2_large | baseline | d2t0 | 0.710311 | 0.004013 | 0.706984 | 0.716823 | 4 |
| 5 | d2_large | baseline | d2t1 | 0.729278 | 0.005902 | 0.721973 | 0.737373 | 4 |
| 5 | d2_large | comp1 | d0t0 | 0.742056 | 0.002870 | 0.737175 | 0.744456 | 4 |
| 5 | d2_large | comp1 | d0t1 | 0.754726 | 0.006646 | 0.743270 | 0.759097 | 4 |
| 5 | d2_large | comp1 | d1t0 | 0.738475 | 0.002903 | 0.735256 | 0.741813 | 4 |
| 5 | d2_large | comp1 | d1t1 | 0.758196 | 0.002163 | 0.756048 | 0.761806 | 4 |
| 5 | d2_large | comp1 | d2t0 | 0.718939 | 0.002142 | 0.715353 | 0.721024 | 4 |
| 5 | d2_large | comp1 | d2t1 | 0.739800 | 0.001755 | 0.737115 | 0.741774 | 4 |
| 50 | d0_large | baseline | d0t0 | 0.741832 | 0.004040 | 0.738298 | 0.748197 | 4 |
| 50 | d0_large | baseline | d0t1 | 0.753295 | 0.006035 | 0.746986 | 0.759712 | 4 |
| 50 | d0_large | baseline | d1t0 | 0.726390 | 0.002020 | 0.724376 | 0.729731 | 4 |
| 50 | d0_large | baseline | d1t1 | 0.747065 | 0.004221 | 0.742837 | 0.751636 | 4 |
| 50 | d0_large | baseline | d2t0 | 0.642524 | 0.029698 | 0.593696 | 0.670099 | 4 |
| 50 | d0_large | baseline | d2t1 | 0.681589 | 0.012844 | 0.665226 | 0.700791 | 4 |
| 50 | d0_large | comp1 | d0t0 | 0.747730 | 0.001438 | 0.746164 | 0.749702 | 4 |
| 50 | d0_large | comp1 | d0t1 | 0.761810 | 0.001638 | 0.759701 | 0.763924 | 4 |
| 50 | d0_large | comp1 | d1t0 | 0.734022 | 0.001292 | 0.732747 | 0.735868 | 4 |
| 50 | d0_large | comp1 | d1t1 | 0.753476 | 0.001337 | 0.752157 | 0.755689 | 4 |
| 50 | d0_large | comp1 | d2t0 | 0.675044 | 0.004283 | 0.667734 | 0.678653 | 4 |
| 50 | d0_large | comp1 | d2t1 | 0.690290 | 0.009618 | 0.677386 | 0.703116 | 4 |
| 50 | d1_large | baseline | d0t0 | 0.670485 | 0.008115 | 0.657808 | 0.679943 | 4 |
| 50 | d1_large | baseline | d0t1 | 0.670173 | 0.009571 | 0.657300 | 0.684304 | 4 |
| 50 | d1_large | baseline | d1t0 | 0.733650 | 0.013378 | 0.714973 | 0.752155 | 4 |
| 50 | d1_large | baseline | d1t1 | 0.754272 | 0.016002 | 0.733097 | 0.774132 | 4 |
| 50 | d1_large | baseline | d2t0 | 0.689182 | 0.011818 | 0.673407 | 0.704071 | 4 |
| 50 | d1_large | baseline | d2t1 | 0.707455 | 0.012700 | 0.688616 | 0.721772 | 4 |
| 50 | d1_large | comp1 | d0t0 | 0.699867 | 0.005698 | 0.692237 | 0.707748 | 4 |
| 50 | d1_large | comp1 | d0t1 | 0.709543 | 0.004764 | 0.702239 | 0.715197 | 4 |
| 50 | d1_large | comp1 | d1t0 | 0.750123 | 0.002306 | 0.747103 | 0.753127 | 4 |
| 50 | d1_large | comp1 | d1t1 | 0.771500 | 0.003200 | 0.766496 | 0.774962 | 4 |
| 50 | d1_large | comp1 | d2t0 | 0.705198 | 0.005011 | 0.700756 | 0.713226 | 4 |
| 50 | d1_large | comp1 | d2t1 | 0.728288 | 0.006831 | 0.723661 | 0.740060 | 4 |
| 50 | d2_large | baseline | d0t0 | 0.724136 | 0.006057 | 0.716377 | 0.731686 | 4 |
| 50 | d2_large | baseline | d0t1 | 0.738070 | 0.006062 | 0.727744 | 0.743303 | 4 |
| 50 | d2_large | baseline | d1t0 | 0.669721 | 0.022450 | 0.634127 | 0.692487 | 4 |
| 50 | d2_large | baseline | d1t1 | 0.698454 | 0.015602 | 0.681153 | 0.719951 | 4 |
| 50 | d2_large | baseline | d2t0 | 0.703928 | 0.007489 | 0.693327 | 0.714509 | 4 |
| 50 | d2_large | baseline | d2t1 | 0.723460 | 0.009482 | 0.710698 | 0.737429 | 4 |
| 50 | d2_large | comp1 | d0t0 | 0.735637 | 0.003034 | 0.730530 | 0.738498 | 4 |
| 50 | d2_large | comp1 | d0t1 | 0.749924 | 0.003775 | 0.743473 | 0.753030 | 4 |
| 50 | d2_large | comp1 | d1t0 | 0.701385 | 0.007415 | 0.689999 | 0.710431 | 4 |
| 50 | d2_large | comp1 | d1t1 | 0.722204 | 0.008296 | 0.709991 | 0.733251 | 4 |
| 50 | d2_large | comp1 | d2t0 | 0.715178 | 0.002977 | 0.710261 | 0.718232 | 4 |
| 50 | d2_large | comp1 | d2t1 | 0.736931 | 0.003274 | 0.733486 | 0.742174 | 4 |

Reused P0-1 rows (ratio 20):

| ratio | setting | method | cell | mean | std | min | max | n_seeds |
|---|---|---|---|---|---|---|---|---|
| 20 | d0_large | baseline | d0t0 | 0.742285 | 0.002638 | 0.737954 | 0.744805 | 4 |
| 20 | d0_large | baseline | d0t1 | 0.754680 | 0.003317 | 0.752081 | 0.760342 | 4 |
| 20 | d0_large | baseline | d1t0 | 0.725666 | 0.006183 | 0.715168 | 0.730519 | 4 |
| 20 | d0_large | baseline | d1t1 | 0.751595 | 0.007411 | 0.740114 | 0.759552 | 4 |
| 20 | d0_large | baseline | d2t0 | 0.673901 | 0.005198 | 0.667817 | 0.680681 | 4 |
| 20 | d0_large | baseline | d2t1 | 0.695356 | 0.009417 | 0.684037 | 0.709900 | 4 |
| 20 | d0_large | comp1 | d0t0 | 0.749948 | 0.002962 | 0.746009 | 0.753419 | 4 |
| 20 | d0_large | comp1 | d0t1 | 0.761158 | 0.004920 | 0.753689 | 0.766892 | 4 |
| 20 | d0_large | comp1 | d1t0 | 0.734527 | 0.000683 | 0.733562 | 0.735436 | 4 |
| 20 | d0_large | comp1 | d1t1 | 0.753223 | 0.001931 | 0.751116 | 0.756295 | 4 |
| 20 | d0_large | comp1 | d2t0 | 0.683747 | 0.003304 | 0.680037 | 0.687966 | 4 |
| 20 | d0_large | comp1 | d2t1 | 0.699650 | 0.001443 | 0.697726 | 0.701321 | 4 |
| 20 | d1_large | baseline | d0t0 | 0.696246 | 0.011843 | 0.680397 | 0.710226 | 4 |
| 20 | d1_large | baseline | d0t1 | 0.706098 | 0.010122 | 0.692427 | 0.716606 | 4 |
| 20 | d1_large | baseline | d1t0 | 0.737997 | 0.003941 | 0.734728 | 0.744734 | 4 |
| 20 | d1_large | baseline | d1t1 | 0.759100 | 0.005833 | 0.752812 | 0.767010 | 4 |
| 20 | d1_large | baseline | d2t0 | 0.686284 | 0.008744 | 0.679137 | 0.700723 | 4 |
| 20 | d1_large | baseline | d2t1 | 0.704824 | 0.009018 | 0.696439 | 0.719340 | 4 |
| 20 | d1_large | comp1 | d0t0 | 0.718732 | 0.002686 | 0.716135 | 0.723033 | 4 |
| 20 | d1_large | comp1 | d0t1 | 0.733359 | 0.002791 | 0.728640 | 0.735508 | 4 |
| 20 | d1_large | comp1 | d1t0 | 0.752234 | 0.002969 | 0.748689 | 0.755866 | 4 |
| 20 | d1_large | comp1 | d1t1 | 0.776441 | 0.002990 | 0.773408 | 0.781193 | 4 |
| 20 | d1_large | comp1 | d2t0 | 0.702985 | 0.006121 | 0.695390 | 0.712431 | 4 |
| 20 | d1_large | comp1 | d2t1 | 0.727830 | 0.006490 | 0.718077 | 0.736244 | 4 |
| 20 | d2_large | baseline | d0t0 | 0.720640 | 0.004993 | 0.712105 | 0.724800 | 4 |
| 20 | d2_large | baseline | d0t1 | 0.733956 | 0.004108 | 0.727859 | 0.739181 | 4 |
| 20 | d2_large | baseline | d1t0 | 0.698257 | 0.009395 | 0.688021 | 0.709592 | 4 |
| 20 | d2_large | baseline | d1t1 | 0.724361 | 0.006315 | 0.714728 | 0.731413 | 4 |
| 20 | d2_large | baseline | d2t0 | 0.702910 | 0.002165 | 0.699584 | 0.705571 | 4 |
| 20 | d2_large | baseline | d2t1 | 0.719392 | 0.008750 | 0.707858 | 0.728359 | 4 |
| 20 | d2_large | comp1 | d0t0 | 0.735893 | 0.001879 | 0.733294 | 0.738126 | 4 |
| 20 | d2_large | comp1 | d0t1 | 0.750412 | 0.002851 | 0.746929 | 0.753693 | 4 |
| 20 | d2_large | comp1 | d1t0 | 0.721388 | 0.002418 | 0.718087 | 0.723888 | 4 |
| 20 | d2_large | comp1 | d1t1 | 0.740752 | 0.006239 | 0.731852 | 0.749299 | 4 |
| 20 | d2_large | comp1 | d2t0 | 0.720530 | 0.003108 | 0.717119 | 0.724294 | 4 |
| 20 | d2_large | comp1 | d2t1 | 0.738885 | 0.003582 | 0.734498 | 0.744391 | 4 |

### 4.2 Oracle per-(ratio, setting, oracle_domain, cell): mean/std/min/max across 4 seeds

Fresh P0-2 rows (ratios 1, 5, 50):

| ratio | setting | oracle_domain | cell | mean | std | min | max | n_seeds |
|---|---|---|---|---|---|---|---|---|
| 1 | d0_large | oracle_d0 | t0 | 0.727777 | 0.001320 | 0.725611 | 0.729189 | 4 |
| 1 | d0_large | oracle_d0 | t1 | 0.739491 | 0.001461 | 0.737738 | 0.741765 | 4 |
| 1 | d0_large | oracle_d1 | t0 | 0.728994 | 0.006039 | 0.720155 | 0.735421 | 4 |
| 1 | d0_large | oracle_d1 | t1 | 0.746201 | 0.012391 | 0.726044 | 0.758218 | 4 |
| 1 | d0_large | oracle_d2 | t0 | 0.696207 | 0.008573 | 0.688336 | 0.709567 | 4 |
| 1 | d0_large | oracle_d2 | t1 | 0.715334 | 0.009404 | 0.703627 | 0.729829 | 4 |
| 5 | d0_large | oracle_d0 | t0 | 0.744738 | 0.004441 | 0.740073 | 0.749456 | 4 |
| 5 | d0_large | oracle_d0 | t1 | 0.758459 | 0.003175 | 0.755422 | 0.763409 | 4 |
| 5 | d0_large | oracle_d1 | t0 | 0.711431 | 0.004879 | 0.705465 | 0.718893 | 4 |
| 5 | d0_large | oracle_d1 | t1 | 0.730222 | 0.007717 | 0.722806 | 0.742497 | 4 |
| 5 | d0_large | oracle_d2 | t0 | 0.667220 | 0.009961 | 0.652747 | 0.679159 | 4 |
| 5 | d0_large | oracle_d2 | t1 | 0.694192 | 0.006818 | 0.682589 | 0.699643 | 4 |
| 5 | d1_large | oracle_d0 | t0 | 0.692151 | 0.010108 | 0.675865 | 0.702981 | 4 |
| 5 | d1_large | oracle_d0 | t1 | 0.705419 | 0.009081 | 0.690843 | 0.715117 | 4 |
| 5 | d1_large | oracle_d1 | t0 | 0.746041 | 0.002693 | 0.742032 | 0.749460 | 4 |
| 5 | d1_large | oracle_d1 | t1 | 0.768949 | 0.002726 | 0.764411 | 0.771170 | 4 |
| 5 | d1_large | oracle_d2 | t0 | 0.683049 | 0.008010 | 0.675266 | 0.696239 | 4 |
| 5 | d1_large | oracle_d2 | t1 | 0.702554 | 0.010281 | 0.691152 | 0.718739 | 4 |
| 5 | d2_large | oracle_d0 | t0 | 0.704706 | 0.010387 | 0.689417 | 0.718064 | 4 |
| 5 | d2_large | oracle_d0 | t1 | 0.718487 | 0.009979 | 0.704127 | 0.729954 | 4 |
| 5 | d2_large | oracle_d1 | t0 | 0.697225 | 0.004022 | 0.693055 | 0.703715 | 4 |
| 5 | d2_large | oracle_d1 | t1 | 0.722323 | 0.005408 | 0.714173 | 0.729190 | 4 |
| 5 | d2_large | oracle_d2 | t0 | 0.714145 | 0.005076 | 0.709169 | 0.721768 | 4 |
| 5 | d2_large | oracle_d2 | t1 | 0.734248 | 0.005403 | 0.728520 | 0.739699 | 4 |
| 50 | d0_large | oracle_d0 | t0 | 0.750267 | 0.001565 | 0.747599 | 0.751578 | 4 |
| 50 | d0_large | oracle_d0 | t1 | 0.762287 | 0.003117 | 0.757183 | 0.765480 | 4 |
| 50 | d0_large | oracle_d1 | t0 | 0.703408 | 0.003473 | 0.698913 | 0.707244 | 4 |
| 50 | d0_large | oracle_d1 | t1 | 0.725246 | 0.006695 | 0.715489 | 0.734068 | 4 |
| 50 | d0_large | oracle_d2 | t0 | 0.608290 | 0.003641 | 0.603663 | 0.613099 | 4 |
| 50 | d0_large | oracle_d2 | t1 | 0.617968 | 0.003391 | 0.612607 | 0.621258 | 4 |
| 50 | d1_large | oracle_d0 | t0 | 0.599245 | 0.013650 | 0.584222 | 0.618183 | 4 |
| 50 | d1_large | oracle_d0 | t1 | 0.614325 | 0.015028 | 0.600971 | 0.637969 | 4 |
| 50 | d1_large | oracle_d1 | t0 | 0.752347 | 0.000785 | 0.751134 | 0.753325 | 4 |
| 50 | d1_large | oracle_d1 | t1 | 0.773626 | 0.001183 | 0.771814 | 0.774757 | 4 |
| 50 | d1_large | oracle_d2 | t0 | 0.676733 | 0.006954 | 0.669484 | 0.685622 | 4 |
| 50 | d1_large | oracle_d2 | t1 | 0.699377 | 0.010345 | 0.685483 | 0.713101 | 4 |
| 50 | d2_large | oracle_d0 | t0 | 0.698070 | 0.018029 | 0.667951 | 0.715709 | 4 |
| 50 | d2_large | oracle_d0 | t1 | 0.717596 | 0.009539 | 0.702976 | 0.728684 | 4 |
| 50 | d2_large | oracle_d1 | t0 | 0.607973 | 0.005064 | 0.601201 | 0.614774 | 4 |
| 50 | d2_large | oracle_d1 | t1 | 0.626422 | 0.004680 | 0.619916 | 0.632876 | 4 |
| 50 | d2_large | oracle_d2 | t0 | 0.722515 | 0.002785 | 0.719238 | 0.726436 | 4 |
| 50 | d2_large | oracle_d2 | t1 | 0.742663 | 0.002758 | 0.739353 | 0.746738 | 4 |

Reused P0-1 rows (ratio 20):

| ratio | setting | oracle_domain | cell | mean | std | min | max | n_seeds |
|---|---|---|---|---|---|---|---|---|
| 20 | d0_large | oracle_d0 | t0 | 0.750267 | 0.001565 | 0.747599 | 0.751578 | 4 |
| 20 | d0_large | oracle_d0 | t1 | 0.762287 | 0.003117 | 0.757183 | 0.765480 | 4 |
| 20 | d0_large | oracle_d1 | t0 | 0.701828 | 0.004480 | 0.695967 | 0.706204 | 4 |
| 20 | d0_large | oracle_d1 | t1 | 0.724997 | 0.008778 | 0.711619 | 0.735206 | 4 |
| 20 | d0_large | oracle_d2 | t0 | 0.636652 | 0.002512 | 0.634128 | 0.640219 | 4 |
| 20 | d0_large | oracle_d2 | t1 | 0.646632 | 0.006574 | 0.636952 | 0.652942 | 4 |
| 20 | d1_large | oracle_d0 | t0 | 0.642954 | 0.007957 | 0.636472 | 0.656473 | 4 |
| 20 | d1_large | oracle_d0 | t1 | 0.661591 | 0.003541 | 0.658416 | 0.667203 | 4 |
| 20 | d1_large | oracle_d1 | t0 | 0.752347 | 0.000785 | 0.751134 | 0.753325 | 4 |
| 20 | d1_large | oracle_d1 | t1 | 0.773626 | 0.001183 | 0.771814 | 0.774757 | 4 |
| 20 | d1_large | oracle_d2 | t0 | 0.674579 | 0.005695 | 0.668455 | 0.682215 | 4 |
| 20 | d1_large | oracle_d2 | t1 | 0.698706 | 0.007084 | 0.687019 | 0.706023 | 4 |
| 20 | d2_large | oracle_d0 | t0 | 0.694664 | 0.006615 | 0.684251 | 0.702204 | 4 |
| 20 | d2_large | oracle_d0 | t1 | 0.711693 | 0.004693 | 0.704708 | 0.717521 | 4 |
| 20 | d2_large | oracle_d1 | t0 | 0.648153 | 0.009243 | 0.638062 | 0.660158 | 4 |
| 20 | d2_large | oracle_d1 | t1 | 0.666076 | 0.007150 | 0.655520 | 0.673999 | 4 |
| 20 | d2_large | oracle_d2 | t0 | 0.722515 | 0.002785 | 0.719238 | 0.726436 | 4 |
| 20 | d2_large | oracle_d2 | t1 | 0.742663 | 0.002758 | 0.739353 | 0.746738 | 4 |

### 4.3 Δ_transfer per cell: baseline (1 cell) − oracle (1 cell) at best_epoch, per (ratio, role)

For ratio=1 all three role columns hold the same set of cell-pairings by construction; the per-role table treats each role as a separate bucket for completeness. comp1_gain is reported alongside.

Per (ratio, role):

| ratio | role | n_cells | delta_mean | delta_std | delta_min | delta_max | comp1_gain_mean | comp1_gain_std | comp1_gain_min | comp1_gain_max |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | L | 8 | 0.015855 | 0.005099 | 0.006119 | 0.023460 | 0.001835 | 0.002346 | -0.003623 | 0.003449 |
| 1 | M | 8 | 0.021215 | 0.005117 | 0.012510 | 0.031056 | 0.001275 | 0.003939 | -0.003937 | 0.008777 |
| 1 | S | 8 | 0.007433 | 0.004559 | -0.000961 | 0.013625 | 0.005996 | 0.003705 | 0.001857 | 0.011219 |
| 20 | L | 24 | -0.014557 | 0.007439 | -0.032753 | -0.003670 | 0.013806 | 0.006601 | 0.000038 | 0.026640 |
| 20 | M | 24 | 0.019416 | 0.010232 | -0.005185 | 0.033795 | 0.013651 | 0.008933 | -0.005271 | 0.031149 |
| 20 | S | 24 | 0.048694 | 0.010853 | 0.029809 | 0.069420 | 0.017235 | 0.012203 | -0.008579 | 0.043081 |
| 5 | L | 24 | -0.001774 | 0.004805 | -0.013885 | 0.007470 | 0.007735 | 0.003874 | 0.000448 | 0.016066 |
| 5 | M | 24 | 0.023523 | 0.012533 | -0.000051 | 0.054326 | 0.006078 | 0.005909 | -0.002758 | 0.019820 |
| 5 | S | 24 | 0.030417 | 0.011191 | 0.007031 | 0.060059 | 0.006345 | 0.005561 | -0.002245 | 0.017077 |
| 50 | L | 24 | -0.015545 | 0.010894 | -0.040236 | 0.001746 | 0.012139 | 0.011778 | -0.004248 | 0.041865 |
| 50 | M | 24 | 0.018645 | 0.014937 | -0.015676 | 0.063735 | 0.012374 | 0.008229 | 0.001637 | 0.035350 |
| 50 | S | 24 | 0.059787 | 0.021553 | -0.016563 | 0.092160 | 0.027564 | 0.019116 | 0.002325 | 0.074038 |

Per (ratio, setting, role):

| ratio | setting | role | n_cells | delta_mean | delta_std | delta_min | delta_max | comp1_gain_mean | comp1_gain_std | comp1_gain_min | comp1_gain_max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | d0_large | L | 8 | 0.015855 | 0.005099 | 0.006119 | 0.023460 | 0.001835 | 0.002346 | -0.003623 | 0.003449 |
| 1 | d0_large | M | 8 | 0.021215 | 0.005117 | 0.012510 | 0.031056 | 0.001275 | 0.003939 | -0.003937 | 0.008777 |
| 1 | d0_large | S | 8 | 0.007433 | 0.004559 | -0.000961 | 0.013625 | 0.005996 | 0.003705 | 0.001857 | 0.011219 |
| 20 | d0_large | L | 8 | -0.007795 | 0.003069 | -0.013127 | -0.003670 | 0.007071 | 0.003548 | 0.000038 | 0.011770 |
| 20 | d0_large | M | 8 | 0.025218 | 0.007044 | 0.009014 | 0.033795 | 0.005245 | 0.007192 | -0.005271 | 0.018394 |
| 20 | d0_large | S | 8 | 0.042986 | 0.009301 | 0.029809 | 0.056958 | 0.007070 | 0.008065 | -0.008579 | 0.018121 |
| 20 | d1_large | L | 8 | -0.014438 | 0.004128 | -0.019086 | -0.007747 | 0.015789 | 0.004239 | 0.011076 | 0.022451 |
| 20 | d1_large | M | 8 | 0.008911 | 0.008990 | -0.005185 | 0.022850 | 0.019853 | 0.005810 | 0.011708 | 0.031149 |
| 20 | d1_large | S | 8 | 0.048900 | 0.010767 | 0.030382 | 0.069420 | 0.024873 | 0.009989 | 0.012807 | 0.043081 |
| 20 | d2_large | L | 8 | -0.021438 | 0.006816 | -0.032753 | -0.012786 | 0.018557 | 0.005320 | 0.009211 | 0.026640 |
| 20 | d2_large | M | 8 | 0.024120 | 0.004181 | 0.015721 | 0.030379 | 0.015855 | 0.006321 | 0.008494 | 0.025047 |
| 20 | d2_large | S | 8 | 0.054195 | 0.009385 | 0.035938 | 0.067685 | 0.019761 | 0.010671 | 0.000439 | 0.033900 |
| 5 | d0_large | L | 8 | -0.000947 | 0.004765 | -0.007863 | 0.007470 | 0.007036 | 0.002523 | 0.002820 | 0.010524 |
| 5 | d0_large | M | 8 | 0.026736 | 0.006949 | 0.018298 | 0.036388 | 0.001649 | 0.003233 | -0.002758 | 0.005999 |
| 5 | d0_large | S | 8 | 0.021325 | 0.010314 | 0.007031 | 0.034208 | 0.003074 | 0.005434 | -0.002245 | 0.012367 |
| 5 | d1_large | L | 8 | 0.000027 | 0.002458 | -0.004194 | 0.003076 | 0.006593 | 0.003742 | 0.000448 | 0.011877 |
| 5 | d1_large | M | 8 | 0.013669 | 0.008406 | -0.000051 | 0.026892 | 0.009954 | 0.006022 | 0.001981 | 0.019820 |
| 5 | d1_large | S | 8 | 0.036438 | 0.011414 | 0.025801 | 0.060059 | 0.010886 | 0.003038 | 0.007804 | 0.017077 |
| 5 | d2_large | L | 8 | -0.004402 | 0.005447 | -0.013885 | 0.002875 | 0.009575 | 0.004414 | 0.003474 | 0.016066 |
| 5 | d2_large | M | 8 | 0.030164 | 0.014170 | 0.016280 | 0.054326 | 0.006631 | 0.004806 | 0.000713 | 0.015949 |
| 5 | d2_large | S | 8 | 0.033486 | 0.003277 | 0.028257 | 0.038006 | 0.005075 | 0.004591 | -0.001591 | 0.011722 |
| 50 | d0_large | L | 8 | -0.008714 | 0.006589 | -0.016321 | 0.001746 | 0.007207 | 0.005703 | -0.002033 | 0.015773 |
| 50 | d0_large | M | 8 | 0.022401 | 0.005760 | 0.008769 | 0.027376 | 0.007022 | 0.004061 | 0.001637 | 0.012824 |
| 50 | d0_large | S | 8 | 0.048928 | 0.027081 | -0.016563 | 0.080323 | 0.020610 | 0.022100 | 0.002325 | 0.074038 |
| 50 | d1_large | L | 8 | -0.019025 | 0.014230 | -0.040236 | -0.000625 | 0.016850 | 0.015478 | -0.000688 | 0.041865 |
| 50 | d1_large | M | 8 | 0.010263 | 0.014726 | -0.015676 | 0.030359 | 0.018424 | 0.010424 | 0.004941 | 0.035350 |
| 50 | d1_large | S | 8 | 0.063544 | 0.013327 | 0.040602 | 0.085860 | 0.034376 | 0.013382 | 0.012294 | 0.057897 |
| 50 | d2_large | L | 8 | -0.018895 | 0.006336 | -0.028655 | -0.009309 | 0.012361 | 0.009874 | -0.004248 | 0.023099 |
| 50 | d2_large | M | 8 | 0.023270 | 0.017709 | 0.011627 | 0.063735 | 0.011677 | 0.003499 | 0.004928 | 0.016506 |
| 50 | d2_large | S | 8 | 0.066890 | 0.017322 | 0.032926 | 0.092160 | 0.027707 | 0.018276 | 0.013300 | 0.070123 |

### 4.4 Summary per (ratio, method)

Mirror of P0_2_summary_ratio.csv (read-only):

| ratio | method | mean_auc | worst_domain_auc | comp1_gain_overall | comp1_gain_role_L | comp1_gain_role_M | comp1_gain_role_S |
|---|---|---|---|---|---|---|---|
| 1 | baseline | 0.740502 | 0.704168 |  |  |  |  |
| 1 | comp1 | 0.743537 | 0.710576 | 0.003036 |  |  |  |
| 1 | oracle | 0.725667 | 0.696207 |  |  |  |  |
| 5 | baseline | 0.733808 | 0.699289 |  |  |  |  |
| 5 | comp1 | 0.740528 | 0.706728 | 0.006719 | 0.007735 | 0.006078 | 0.006345 |
| 5 | oracle | 0.716420 | 0.688138 |  |  |  |  |
| 20 | baseline | 0.718530 | 0.687698 |  |  |  |  |
| 20 | comp1 | 0.733427 | 0.702420 | 0.014897 | 0.013806 | 0.013651 | 0.017235 |
| 20 | oracle | 0.700679 | 0.677916 |  |  |  |  |
| 50 | baseline | 0.709760 | 0.678545 |  |  |  |  |
| 50 | comp1 | 0.727120 | 0.698473 | 0.017359 | 0.012139 | 0.012374 | 0.027564 |
| 50 | oracle | 0.688798 | 0.669180 |  |  |  |  |

### 4.5 Quick-look: joint 6-cell mean per (ratio, setting, method), averaged across 4 seeds

| ratio | setting | method | joint_6cell_mean |
|---|---|---|---|
| 1 | d0_large | baseline | 0.740502 |
| 1 | d0_large | comp1 | 0.743537 |
| 5 | d0_large | baseline | 0.733415 |
| 5 | d0_large | comp1 | 0.737335 |
| 5 | d1_large | baseline | 0.733072 |
| 5 | d1_large | comp1 | 0.742216 |
| 5 | d2_large | baseline | 0.734938 |
| 5 | d2_large | comp1 | 0.742032 |
| 20 | d0_large | baseline | 0.723914 |
| 20 | d0_large | comp1 | 0.730375 |
| 20 | d1_large | baseline | 0.715092 |
| 20 | d1_large | comp1 | 0.735263 |
| 20 | d2_large | baseline | 0.716586 |
| 20 | d2_large | comp1 | 0.734643 |
| 50 | d0_large | baseline | 0.715449 |
| 50 | d0_large | comp1 | 0.727062 |
| 50 | d1_large | baseline | 0.704203 |
| 50 | d1_large | comp1 | 0.727420 |
| 50 | d2_large | baseline | 0.709628 |
| 50 | d2_large | comp1 | 0.726877 |

## 5. Failed runs

No failed runs. All 140 unique fresh tasks reached status=OK. All 60 P0-1 reused tasks had status=OK in P0_1_raw_dedup.csv at the time the aggregator ran.

Operational notes:

- The orchestrator used a PID lockfile at `experiments/logs/p0_2_orchestrator.lock`; the launch wrote pid 29340 and removed it on graceful exit (no second orchestrator was ever started).
- The run was interrupted twice by the runner's session (2-hour limit, then a 529 transient). The orchestrator process was started via `nohup` and survived both interruptions; the master CSV was consulted on every monitor checkpoint to confirm progress.
- Per-run stdout logs are at `experiments/logs/p0_2_<name>.log` (one per fresh task; 140 logs total, plus `p0_2_orchestrator_main.log`).

## 6. Missing metrics

No missing metrics.

- All 140 fresh run directories under `experiments/P0_2_ratio_sweep/` contain `config.json`, `metrics_epoch.csv`, `metrics_summary.csv`.
- Joint total_train_samples: all 56 joint runs have role_sizes sum = 187,500 (matches contract per ratio).
- Oracle per-domain training sample count: each oracle_d run's single_domain role size matches the expected (L, M, S) value for its (ratio, setting):
  - ratio=1: 62500 / 62500 / 62500 (L,M,S all equal)
  - ratio=5: 125000 / 37500 / 25000
  - ratio=20: 150000 / 30000 / 7500 (P0-1 reuse; config.json spot-checked)
  - ratio=50: 150000 / 34500 / 3000
- Per-run `per_dt_test_auc` parses successfully (6 floats per joint row, 2 floats per oracle row).
