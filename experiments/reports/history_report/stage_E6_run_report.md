# Stage E6 raw run report

This report is a structural/raw collection artifact. It contains no scientific comparison or conclusion.

- Expected matrix: 6 arms × 1 setting (`d0_large`) × 4 seeds = 24 runs.
- Expected seeds: `2022, 2023, 2024, 2025`.
- Available formal runs: 24
- Missing runs: 0
- Invalid runs: 0
- Summary rows: 144

## Artifacts

- Summary: `D:\CODE\phase0-3\experiments\stage_E6_summary.csv`
- Ablation raw mapping: `D:\CODE\phase0-3\experiments\stage_E6_ablation.csv`
- Manifest: `D:\CODE\phase0-3\experiments\stage_E6_summary_manifest.json`

## Per-run status

| arm | setting | seed | status | reasons |
|---|---|---:|---|---|
| equal_static | d0_large | 2022 | OK |  |
| equal_static | d0_large | 2023 | OK |  |
| equal_static | d0_large | 2024 | OK |  |
| equal_static | d0_large | 2025 | OK |  |
| sample_static | d0_large | 2022 | OK |  |
| sample_static | d0_large | 2023 | OK |  |
| sample_static | d0_large | 2024 | OK |  |
| sample_static | d0_large | 2025 | OK |  |
| sample_ema_control | d0_large | 2022 | OK |  |
| sample_ema_control | d0_large | 2023 | OK |  |
| sample_ema_control | d0_large | 2024 | OK |  |
| sample_ema_control | d0_large | 2025 | OK |  |
| adaptive_noema | d0_large | 2022 | OK |  |
| adaptive_noema | d0_large | 2023 | OK |  |
| adaptive_noema | d0_large | 2024 | OK |  |
| adaptive_noema | d0_large | 2025 | OK |  |
| adaptive_ema | d0_large | 2022 | OK |  |
| adaptive_ema | d0_large | 2023 | OK |  |
| adaptive_ema | d0_large | 2024 | OK |  |
| adaptive_ema | d0_large | 2025 | OK |  |
| adaptive_ema_uniform | d0_large | 2022 | OK |  |
| adaptive_ema_uniform | d0_large | 2023 | OK |  |
| adaptive_ema_uniform | d0_large | 2024 | OK |  |
| adaptive_ema_uniform | d0_large | 2025 | OK |  |

Raw collection complete; main analysis is a separate workflow stage.

## Formal execution record

- Command: `$env:PYTHONPATH='D:\\CODE\\.venv\\Lib\\site-packages'; D:\\Programs\\anaconda3\\python.exe experiments/stage_E6_orchestrator.py --python D:\\Programs\\anaconda3\\python.exe`
- Working directory: `D:\\CODE\\phase0-3`
- Runtime: Anaconda Python with `torch 2.6.0+cu126`; CUDA device `cuda:0` available.
- Dataset: `D:/CODE/phase0-3/m3oe/examples/data/ml-1m`; batch size `40960`; epoch budget `50`; early-stop patience `10`; expert count `4`; learning rates `0.01 / 0.001`; weight decay `1e-5`.
- E3 frozen oracle: audited and reused; 12 references, 6 cells × 4 seeds; per-run oracle coverage `1.0`.

## Matrix and block status

All 24 registered runs are `OK`; missing runs `0`; invalid runs `0`; failed runs `0`; retries `0`. The six arms are `equal_static`, `sample_static`, `sample_ema_control`, `adaptive_noema`, `adaptive_ema`, and `adaptive_ema_uniform`, each with seeds `2022, 2023, 2024, 2025`.

| seed | six-arm block | common epochs | last common epoch | common trace rows | batch/controller/shared-step identity |
|---:|---|---|---:|---:|---|
| 2022 | VALID | 0–15 | 15 | 96 | VALID |
| 2023 | VALID | 0–14 | 14 | 90 | VALID |
| 2024 | VALID | 0–15 | 15 | 96 | VALID |
| 2025 | VALID | 0–17 | 17 | 108 | VALID |

All four seed blocks have `common_support_valid=true` and no comparator reasons. Unequal suffix lengths were censored only after the common support.

## Raw schema and counts

- Required artifacts per run: `config.json`, `run_manifest.json`, `stdout.log`, `metrics_epoch.csv`, `metrics_summary.csv`, `controller_step.csv`, `controller_epoch.csv`, `shared_gradient_step.csv`, `shared_gradient_epoch.csv`, `batch_trace.csv`, `nonfinite_inventory.json`, `model.pth`.
- Artifact completeness: `12/12` required files × `24/24` runs.
- Raw summary: `144` rows (`24` runs × `6` cells); each run has `300` `metrics_epoch` rows and `6` `metrics_summary` rows.
- Aggregate trace counts: `controller_step=2070`, `shared_gradient_step=12420`, `batch_trace=2484`, `metrics_epoch=7200`.
- Each run's `controller_epoch` and `shared_gradient_epoch` counts are preserved in `stage_E6_summary_manifest.json` and `stage_E6_runs.csv`.
- Nonfinite inventory: unexplained count `0` across all runs.
- Summary schema includes raw AUC/LogLoss, per-domain/per-task cell fields, oracle/transfer fields, controller magnitude/contribution/calibration fields, variance/SNR/alignment diagnostics, common-support fields, censoring, and run path.

## Paths and errors

- Raw run root: `D:\\CODE\\phase0-3\\experiments\\stage_E6_raw_results`
- Master status: `D:\\CODE\\phase0-3\\experiments\\stage_E6_runs.csv`
- Summary: `D:\\CODE\\phase0-3\\experiments\\stage_E6_summary.csv`
- Raw ablation mapping: `D:\\CODE\\phase0-3\\experiments\\stage_E6_ablation.csv`
- Structural manifest: `D:\\CODE\\phase0-3\\experiments\\stage_E6_summary_manifest.json`
- Per-run stdout logs: `<raw run root>/<arm>/d0_large/seed<seed>/stdout.log`
- Runtime errors: none recorded; all process return codes were `0`.
