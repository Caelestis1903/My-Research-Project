# Stage E5 Run Report

This is a raw artifact and completeness report; it contains no scientific conclusion.

- Expected runs: 20 (5 arms × 4 seeds)
- Available valid runs: 20
- Complete matrix: True
- All seed blocks valid: True
- Oracle coverage complete: True

## Run inventory

| arm | seed | available | valid | formal_valid | block_status |
|---|---:|---:|---:|---:|---|
| static_sample | 2022 | 1 | True | True | VALID |
| static_sample | 2023 | 1 | True | True | VALID |
| static_sample | 2024 | 1 | True | True | VALID |
| static_sample | 2025 | 1 | True | True | VALID |
| static_interior | 2022 | 1 | True | True | VALID |
| static_interior | 2023 | 1 | True | True | VALID |
| static_interior | 2024 | 1 | True | True | VALID |
| static_interior | 2025 | 1 | True | True | VALID |
| static_matched | 2022 | 1 | True | True | VALID |
| static_matched | 2023 | 1 | True | True | VALID |
| static_matched | 2024 | 1 | True | True | VALID |
| static_matched | 2025 | 1 | True | True | VALID |
| forward | 2022 | 1 | True | True | VALID |
| forward | 2023 | 1 | True | True | VALID |
| forward | 2024 | 1 | True | True | VALID |
| forward | 2025 | 1 | True | True | VALID |
| reverse | 2022 | 1 | True | True | VALID |
| reverse | 2023 | 1 | True | True | VALID |
| reverse | 2024 | 1 | True | True | VALID |
| reverse | 2025 | 1 | True | True | VALID |

## Execution record

Raw execution and completeness record only; no scientific interpretation.

- Experiment ID: Stage E5, formal `d0_large`
- Protocol: fixed 18 epochs; 5 arms × 4 seeds = 20 runs
- Seeds: `2022`, `2023`, `2024`, `2025`
- Dataset: `D:/CODE/phase0-3/m3oe/examples/data/ml-1m`
- Device: `cuda:0`
- Batch size: `40960`; main LR `0.01`; architecture LR `0.001`; weight decay `1e-05`
- Expert count `4`; probe interval `1`; instrumentation `1`; patience field `10`
- Role sizes: `150000,34500,3000`
- Oracle root: `experiments/stage_E3_raw_results/oracle_reference`
- Legacy oracle root: `experiments/P0_2_ratio_sweep`

Formal launch command:

```text
$env:PYTHONPATH='D:\\CODE\\.venv\\Lib\\site-packages'; python experiments/stage_E5_orchestrator.py --python D:\\Programs\\anaconda3\\python.exe
```

Git state unavailable: `D:\\CODE\\phase0-3` has no `.git` metadata.

## Five-arm block and batch identity

All four seed blocks were `VALID`; each arm observed epochs `0..17`.

| seed | arms | train steps/arm | architecture steps/arm | shared epoch rows/arm | shared step rows/arm | block |
|---:|---:|---:|---:|---:|---:|---|
| 2022 | 5 | 90 | 18 | 108 | 540 | VALID |
| 2023 | 5 | 90 | 18 | 108 | 540 | VALID |
| 2024 | 5 | 90 | 18 | 108 | 540 | VALID |
| 2025 | 5 | 90 | 18 | 108 | 540 | VALID |

For each seed, the five arms had 90 common train `(epoch, step)` supports, zero mismatched groups, and identical batch fingerprints at each common support. Each seed had 90 unique train fingerprints across those supports.

## Integrated exposure

Domain order is `d0,d1,d2`; step-weighted values use the 90 main optimizer steps. Values are raw schedule/config fields (the schedule is identical by arm across seeds).

| arm | alpha exposure | coefficient integral d0,d1,d2 | step-weighted coefficient integral d0,d1,d2 |
|---|---:|---|---|
| static_sample | 18.000000 | 43.200000, 9.936000, 0.864000 | 216.000000, 49.680000, 4.320000 |
| static_interior | 13.500000 | 38.980625, 12.946270, 2.073105 | 194.903123, 64.731352, 10.365525 |
| static_matched | 15.750000 | 41.252955, 11.401629, 1.345415 | 206.264777, 57.008147, 6.727076 |
| forward | 15.750000 | 41.192357, 11.416491, 1.391152 | 205.961785, 57.082456, 6.955759 |
| reverse | 15.750000 | 41.192357, 11.416491, 1.391152 | 205.961785, 57.082456, 6.955759 |

Each arm has 18 schedule rows; each epoch coefficient sum is 6 within recorded floating-point precision.

## Oracle coverage

- E3 frozen oracle audit: 12/12 expected references reused.
- Per-run oracle coverage: 1.0 for all 20 runs.
- Oracle coverage complete: `true`.
- Oracle retraining: none.

## Raw schema and counts

Required artifacts: 11 per run; present artifacts: 220/220.

| artifact | rows/run | total |
|---|---:|---:|
| `config.json` | JSON | 20 files |
| `stdout.log` | log | 20 files |
| `run_manifest.json` | JSON | 20 files |
| `schedule_epoch.csv` | 18 | 360 |
| `batch_trace.csv` | 108 | 2160 |
| `shared_gradient_epoch.csv` | 108 | 2160 |
| `shared_gradient_step.csv` | 540 | 10800 |
| `metrics_epoch.csv` | 108 | 2160 |
| `metrics_summary.csv` | 12 | 240 |
| `nonfinite_inventory.json` | JSON | 20 files |
| `model.pth` | binary | 20 files |

Metric CSV schemas retain train/validation/test AUC and LogLoss, domain/task/cell identifiers, and selected/fixed-final fields. Instrumentation schemas retain schedule coefficients, batch fingerprints, raw/weighted gradient fields, contribution fields, update alignment, and reconstruction error.

## Logs, errors, and missing runs

- Raw run root: `D:\\CODE\\phase0-3\\experiments\\stage_E5_raw_results\\<arm>\\d0_large\\seed<seed>`
- Per-run logs: each run directory contains `stdout.log`.
- Run inventory: `D:\\CODE\\phase0-3\\experiments\\stage_E5_runs.csv`
- Summary: `D:\\CODE\\phase0-3\\experiments\\stage_E5_summary.csv`
- Manifest: `D:\\CODE\\phase0-3\\experiments\\stage_E5_summary_manifest.json`
- Pre-launch environment issue: system Python lacked `torch`; configured `D:\\CODE\\phase0-3\\venv\\Scripts\\python.exe` was unusable.
- Resolution: `D:\\Programs\\anaconda3\\python.exe` with `PYTHONPATH=D:\\CODE\\.venv\\Lib\\site-packages`.
- Formal run errors: none; retries: none.
- Missing runs: none; invalid runs: none; unexplained non-finite entries: 0.
