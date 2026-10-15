# Raw Experiment Report — P1-2

## 1. Experiment metadata

- Stage: P1-2 (Curriculum Life-or-Death: domain-conditioned curriculum on fusion weights, corrected sample-weighting)
- Dataset: ml-1m (MDMTRec domain configuration: d0=d1=d2 in ml-1m; mappings rotate per setting)
- Model: MDMTRec (from m3oe/examples)
- Method: joint sample-weighted × {comp1 off, comp1 on} × {curriculum none, curriculum forward κ0=+0.5, curriculum reverse κ0=−0.5}; all P1-2 runs pass `--fix_optim_grouping 1`
- Arm definitions (curriculum_mode × comp1, all `--loss_weighting sample`):
  - A0: curriculum none, comp1 off
  - B0: domain_conditioned κ0=+0.5, comp1 off
  - C0: domain_conditioned κ0=−0.5, comp1 off
  - A1: curriculum none, comp1 on
  - B1: domain_conditioned κ0=+0.5, comp1 on
  - C1: domain_conditioned κ0=−0.5, comp1 on
- Seed: 2022, 2023, 2024, 2025
- Settings: d0_large, d1_large, d2_large
- Layout: 6 arms × 3 settings × 4 seeds = 72 cells
- Ratio role_sizes contract: ratio=50, L/M/S = 150000/34500/3000
- Code: experiments/p1_2_orchestrator.py + experiments/p4_run.py (orchestrator); experiments/p1_2_report.py (aggregator); experiments/p4_run.py invoked with `--log_grad_norm 1` for A0/A1 fresh runs (so grad_norm covers all 6 arms)
- GPU: NVIDIA GeForce RTX 4060 Laptop (8 GiB)
- Concurrency: maximum 2 p4_run children
- Stagger: 30 seconds
- Start time: 2026-09-09 09:59 (orchestrator launch)
- End time: 2026-09-09 11:44 (last raw row written; `P1_2_ORCHESTRATOR DONE`)
- Wall-clock total: approximately 105 minutes
- Total fit time: 13770s; mean fit 191.26s; min fit 139.5s; max fit 292.1s
- Total recorded wall time: 14305s; mean recorded wall time 198.68s; min 145s; max 300s
- Peak CUDA memory: mean 1052 MiB; min 944 MiB; max 1160 MiB
- Resume contract: orchestrator reads P1_2_raw.csv at startup and skips rows already marked `OK`
- Per-run stdout logs: experiments/logs/p1_2_p1_2_<arm>_<setting>_seed<seed>.log

## 2. Completion status

All 72 unique tasks completed `OK`. `P1_2_curriculum/P1_2_raw.csv` contains 72 data rows: `OK=72`, `FAILED=0`.

Arm-level row counts:

| arm | rows |
|---|---:|
| A0 | 12 |
| A1 | 12 |
| B0 | 12 |
| B1 | 12 |
| C0 | 12 |
| C1 | 12 |

Per-run table (arm × setting × seed → status, log path):

| run | status | checkpoint | source | log |
|---|---|---|---|---|
| A0 d0_large seed2022 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A0_d0_large_seed2022.log |
| A0 d0_large seed2023 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A0_d0_large_seed2023.log |
| A0 d0_large seed2024 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A0_d0_large_seed2024.log |
| A0 d0_large seed2025 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A0_d0_large_seed2025.log |
| A0 d1_large seed2022 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A0_d1_large_seed2022.log |
| A0 d1_large seed2023 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A0_d1_large_seed2023.log |
| A0 d1_large seed2024 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A0_d1_large_seed2024.log |
| A0 d1_large seed2025 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A0_d1_large_seed2025.log |
| A0 d2_large seed2022 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A0_d2_large_seed2022.log |
| A0 d2_large seed2023 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A0_d2_large_seed2023.log |
| A0 d2_large seed2024 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A0_d2_large_seed2024.log |
| A0 d2_large seed2025 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A0_d2_large_seed2025.log |
| B0 d0_large seed2022 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B0_d0_large_seed2022.log |
| B0 d0_large seed2023 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B0_d0_large_seed2023.log |
| B0 d0_large seed2024 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B0_d0_large_seed2024.log |
| B0 d0_large seed2025 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B0_d0_large_seed2025.log |
| B0 d1_large seed2022 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B0_d1_large_seed2022.log |
| B0 d1_large seed2023 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B0_d1_large_seed2023.log |
| B0 d1_large seed2024 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B0_d1_large_seed2024.log |
| B0 d1_large seed2025 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B0_d1_large_seed2025.log |
| B0 d2_large seed2022 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B0_d2_large_seed2022.log |
| B0 d2_large seed2023 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B0_d2_large_seed2023.log |
| B0 d2_large seed2024 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B0_d2_large_seed2024.log |
| B0 d2_large seed2025 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B0_d2_large_seed2025.log |
| C0 d0_large seed2022 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C0_d0_large_seed2022.log |
| C0 d0_large seed2023 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C0_d0_large_seed2023.log |
| C0 d0_large seed2024 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C0_d0_large_seed2024.log |
| C0 d0_large seed2025 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C0_d0_large_seed2025.log |
| C0 d1_large seed2022 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C0_d1_large_seed2022.log |
| C0 d1_large seed2023 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C0_d1_large_seed2023.log |
| C0 d1_large seed2024 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C0_d1_large_seed2024.log |
| C0 d1_large seed2025 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C0_d1_large_seed2025.log |
| C0 d2_large seed2022 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C0_d2_large_seed2022.log |
| C0 d2_large seed2023 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C0_d2_large_seed2023.log |
| C0 d2_large seed2024 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C0_d2_large_seed2024.log |
| C0 d2_large seed2025 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C0_d2_large_seed2025.log |
| A1 d0_large seed2022 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A1_d0_large_seed2022.log |
| A1 d0_large seed2023 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A1_d0_large_seed2023.log |
| A1 d0_large seed2024 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A1_d0_large_seed2024.log |
| A1 d0_large seed2025 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A1_d0_large_seed2025.log |
| A1 d1_large seed2022 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A1_d1_large_seed2022.log |
| A1 d1_large seed2023 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A1_d1_large_seed2023.log |
| A1 d1_large seed2024 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A1_d1_large_seed2024.log |
| A1 d1_large seed2025 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A1_d1_large_seed2025.log |
| A1 d2_large seed2022 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A1_d2_large_seed2022.log |
| A1 d2_large seed2023 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A1_d2_large_seed2023.log |
| A1 d2_large seed2024 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A1_d2_large_seed2024.log |
| A1 d2_large seed2025 | OK | metrics_summary.csv | fresh (--log_grad_norm 1) | experiments/logs/p1_2_p1_2_A1_d2_large_seed2025.log |
| B1 d0_large seed2022 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B1_d0_large_seed2022.log |
| B1 d0_large seed2023 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B1_d0_large_seed2023.log |
| B1 d0_large seed2024 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B1_d0_large_seed2024.log |
| B1 d0_large seed2025 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B1_d0_large_seed2025.log |
| B1 d1_large seed2022 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B1_d1_large_seed2022.log |
| B1 d1_large seed2023 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B1_d1_large_seed2023.log |
| B1 d1_large seed2024 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B1_d1_large_seed2024.log |
| B1 d1_large seed2025 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B1_d1_large_seed2025.log |
| B1 d2_large seed2022 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B1_d2_large_seed2022.log |
| B1 d2_large seed2023 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B1_d2_large_seed2023.log |
| B1 d2_large seed2024 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B1_d2_large_seed2024.log |
| B1 d2_large seed2025 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_B1_d2_large_seed2025.log |
| C1 d0_large seed2022 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C1_d0_large_seed2022.log |
| C1 d0_large seed2023 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C1_d0_large_seed2023.log |
| C1 d0_large seed2024 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C1_d0_large_seed2024.log |
| C1 d0_large seed2025 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C1_d0_large_seed2025.log |
| C1 d1_large seed2022 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C1_d1_large_seed2022.log |
| C1 d1_large seed2023 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C1_d1_large_seed2023.log |
| C1 d1_large seed2024 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C1_d1_large_seed2024.log |
| C1 d1_large seed2025 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C1_d1_large_seed2025.log |
| C1 d2_large seed2022 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C1_d2_large_seed2022.log |
| C1 d2_large seed2023 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C1_d2_large_seed2023.log |
| C1 d2_large seed2024 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C1_d2_large_seed2024.log |
| C1 d2_large seed2025 | OK | metrics_summary.csv | fresh | experiments/logs/p1_2_p1_2_C1_d2_large_seed2025.log |

## 3. Regression anchor (P0-4 w_sample, P0-5 comp1)

A0 fresh runs (curriculum none, comp1 off, sample weighting) compared field-by-field with `experiments/P0_4_loss_weighting/w_sample/<setting>/baseline/seed<s>` (12 cells).
A1 fresh runs (curriculum none, comp1 on, sample weighting) compared field-by-field with `experiments/P0_5_comp1_x_weighting/w_sample/<setting>/comp1/seed<s>` (12 cells).

`experiments/tmp_kr/p1_2_anchor_check.py` output:

```
=== P1-2 anchor check (A0 vs P0-4 w_sample, A1 vs P0-5) ===
  OK=24 MISMATCH=0 MISSING=0 (of 24 A cells)
P1_2_ANCHOR_CHECK PASS: A0/A1 fresh runs are bit-identical to P0-4/P0-5 on all non-label columns (config + summary + epoch)
```

- Anchored cells: 24 (12 A0 + 12 A1)
- Field comparisons: 24
- Differences: none
- Result: `P1_2_ANCHOR_CHECK PASS`

## 4. Report outputs

The report command completed successfully.

| file | data rows |
|---|---:|
| experiments/P1_2_curriculum/P1_2_paired.csv | 96 |
| experiments/P1_2_curriculum/P1_2_per_domain.csv | 1608 |
| experiments/P1_2_curriculum/P1_2_peak_epoch.csv | 252 |
| experiments/P1_2_curriculum/P1_2_grad_norm.csv | 1680 |
| experiments/P1_2_curriculum/P1_2_rho_sanity.csv | 6 |

`P1_2_paired.csv` contains 6 paired-comparison rows (each: per-seed × 3 settings + median/iqr/sign/n summary block) = 96 rows.

`P1_2_per_domain.csv` contains 72 runs × 3 (per-d,t) slices × d=2 + 6 arm summary × per-d aggregates = 1608 rows.

`P1_2_peak_epoch.csv` contains 72 runs × 3 domain roles + 6 arm summaries × 3 roles + 8 paired-diff summaries × 3 roles = 252 rows.

`P1_2_grad_norm.csv` contains per-epoch × per-domain mean/min/max grad_norm across arms, settings, domains = 1680 rows.

`P1_2_rho_sanity.csv` contains 6 arm-level rho sanity summaries = 6 rows.

## 5. Paired comparison (P1_2_paired.csv)

Pair definitions: B/A refers to the same `comp1` setting; comparison is `curriculum_off (A) vs curriculum_on (B or C)`.

| pair | metric | median | iqr | sign-consistency | n |
|---|---|---:|---|---:|---:|
| B0-A0 | delta_overall | 0.000759 | [-0.000537, 0.002384] | 0.5833 | 12 |
| B0-A0 | delta_worst | 0.001226 | [-0.000329, 0.002337] | 0.6667 | 12 |
| C0-A0 | delta_overall | 0.001066 | [-0.000329, 0.001712] | 0.6667 | 12 |
| C0-A0 | delta_worst | 0.001229 | [-0.000072, 0.002362] | 0.6667 | 12 |
| C0-B0 | delta_overall | 0.000416 | [-0.000076, 0.000971] | 0.7500 | 12 |
| C0-B0 | delta_worst | 0.000348 | [-0.001012, 0.001543] | 0.5000 | 12 |
| B1-A1 | delta_overall | 0.000320 | [-0.001691, 0.001099] | 0.5833 | 12 |
| B1-A1 | delta_worst | -0.000909 | [-0.002123, 0.000306] | 0.2500 | 12 |
| C1-A1 | delta_overall | 0.000390 | [-0.001659, 0.001388] | 0.5833 | 12 |
| C1-A1 | delta_worst | -0.000059 | [-0.003218, 0.001742] | 0.5000 | 12 |
| (B1-A1)-(B0-A0) | delta_overall | -0.001015 | [-0.001865, 0.001025] | 0.4167 | 12 |
| (B1-A1)-(B0-A0) | delta_worst | -0.001511 | [-0.004347, 0.000248] | 0.3333 | 12 |

## 6. Peak epoch (P1_2_peak_epoch.csv summary block)

Arm summary by domain role (peak_epoch over 12 runs per arm):

| arm | role | mean | median | min | max |
|---|---|---:|---:|---:|---:|
| A0 | L | 8.83 | 8.0 | 4 | 22 |
| A0 | M | 8.00 | 7.5 | 4 | 12 |
| A0 | S | 7.08 | 7.0 | 4 | 11 |
| B0 | L | 9.67 | 8.0 | 5 | 22 |
| B0 | M | 7.92 | 7.5 | 4 | 12 |
| B0 | S | 7.17 | 7.0 | 5 | 11 |
| C0 | L | 10.25 | 8.5 | 4 | 22 |
| C0 | M | 8.83 | 8.5 | 5 | 15 |
| C0 | S | 7.33 | 7.0 | 6 | 11 |
| A1 | L | 11.83 | 11.0 | 6 | 20 |
| A1 | M | 13.50 | 14.0 | 7 | 20 |
| A1 | S | 8.50 | 8.5 | 6 | 11 |
| B1 | L | 12.50 | 12.5 | 8 | 18 |
| B1 | M | 12.83 | 12.5 | 8 | 18 |
| B1 | S | 8.58 | 8.5 | 6 | 13 |
| C1 | L | 12.50 | 12.5 | 6 | 20 |
| C1 | M | 12.58 | 12.0 | 6 | 21 |
| C1 | S | 8.67 | 8.5 | 6 | 15 |

Paired-difference summary (peak_diff, per role, n=12):

| pair | role | mean_diff | median_diff |
|---|---|---:|---:|
| B0-A0 | L | 0.83 | 0.0 |
| B0-A0 | M | -0.08 | 0.0 |
| B0-A0 | S | 0.08 | 0.0 |
| C0-A0 | L | 1.42 | 0.0 |
| C0-A0 | M | 0.83 | 0.5 |
| C0-A0 | S | 0.25 | 0.0 |
| C0-B0 | L | 0.58 | 0.0 |
| C0-B0 | M | 0.92 | 0.5 |
| C0-B0 | S | 0.17 | 0.0 |
| B1-A1 | L | 0.67 | 1.0 |
| B1-A1 | M | -0.67 | -1.0 |
| B1-A1 | S | 0.08 | 0.0 |
| C1-A1 | L | 0.67 | 0.0 |
| C1-A1 | M | -0.92 | -1.0 |
| C1-A1 | S | 0.17 | 0.0 |
| C1-B1 | L | 0.00 | 0.0 |
| C1-B1 | M | -0.25 | 0.0 |
| C1-B1 | S | 0.08 | 0.0 |

## 7. grad_norm trajectory (P1_2_grad_norm.csv; representative sample)

Per-arm representative epoch points (arm=A0,setting=d0_large,domain=shared,phase=rho=1) selected from rows of `P1_2_grad_norm.csv`:

| epoch | mean_grad_norm | min_grad_norm | max_grad_norm | mean_rho | n_seeds |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.626035e-02 | 2.367328e-02 | 9.335388e-02 | 1.000000 | 4 |
| 5 | 1.657235e-01 | 9.463923e-02 | 3.300073e-01 | 1.000000 | 4 |
| 10 | 1.524112e-01 | 1.197596e-01 | 1.888572e-01 | 1.000000 | 4 |
| 15 | 1.217023e-01 | 1.020140e-01 | 1.507847e-01 | 1.000000 | 3 |
| 20 | 1.106202e-01 | 1.106202e-01 | 1.106202e-01 | 1.000000 | 1 |

Per-arm representative epoch points (arm=B0,setting=d0_large,domain=shared,phase=rho<1):

| epoch | mean_grad_norm | mean_rho | n_seeds |
|---:|---:|---:|---:|
| 0 | 5.541902e-02 | 0.000000 | 4 |
| 5 | 1.576007e-01 | 0.200000 | 4 |
| 10 | 1.162341e-01 | 0.400000 | 4 |
| 15 | 1.307344e-01 | 0.600000 | 4 |
| 20 | 8.449284e-02 | 0.800000 | 1 |

Full per-arm × per-setting × per-domain × per-epoch trajectory in `experiments/P1_2_curriculum/P1_2_grad_norm.csv` (1680 rows).

## 8. rho sanity (P1_2_rho_sanity.csv)

| arm | n_rho_obs | rho_min | rho_median | rho_max | kappa_d_range |
|---|---:|---:|---:|---:|---|
| A0 | 900 | 1.000000 | 1.000000 | 1.000000 | n/a |
| B0 | 884 | 0.000000 | 0.360000 | 0.880000 | kappa0=0.5 kappa_d[pmax]=0.000 kappa_d[pmin]=0.500 sign_at_rho0=+ |
| C0 | 900 | 0.000000 | 0.360000 | 0.880000 | kappa0=-0.5 kappa_d[pmax]=-0.000 kappa_d[pmin]=-0.500 sign_at_rho0=- |
| A1 | 952 | 1.000000 | 1.000000 | 1.000000 | n/a |
| B1 | 984 | 0.000000 | 0.400000 | 0.920000 | kappa0=0.5 kappa_d[pmax]=0.000 kappa_d[pmin]=0.500 sign_at_rho0=+ |
| C1 | 980 | 0.000000 | 0.400000 | 0.920000 | kappa0=-0.5 kappa_d[pmax]=-0.000 kappa_d[pmin]=-0.500 sign_at_rho0=- |

## 9. Failed runs and missing metrics

No failed runs.

No missing metrics.

- 72 run directories under `experiments/P1_2_curriculum/<arm>/setting_<setting>/seed<s>/`
- 72 run directories contain `config.json`, `metrics_epoch.csv`, `metrics_summary.csv`, and `grad_norm_epoch.csv` (the latter for the A arms; produced because `--log_grad_norm 1` was passed for A0/A1 fresh runs)
- 72 per-run logs under `experiments/logs/p1_2_p1_2_*.log`
- P1-2 orchestrator lock removed after graceful exit (`P1_2_ORCHESTRATOR DONE` written)
- No Python orchestrator process remained after completion
- Error scan of `orchestrator.log` and per-run logs: no `traceback`, `error`, `exception`, `FAILED`, `cuda out of memory`, or `killed` matches