# Raw Experiment Report — P0-4

## 1. Experiment metadata

- Stage: P0-4 (loss-weighting sweep, fixed ratio 50)
- Dataset: ml-1m (MDMTRec domain configuration: d0=d1=d2 in ml-1m; mappings rotate per setting)
- Model: MDMTRec (from m3oe/examples)
- Method: joint baseline, 3 loss-weighting arms (equal, sample, sqrt), 3 settings (d0_large, d1_large, d2_large)
- Seed: 2022, 2023, 2024, 2025
- Total fresh tasks: 36 (3 weightings × 3 settings × 4 seeds × 1 joint baseline arm)
- Ratio role_sizes contract: ratio=50, L/M/S = 150000/34500/3000
- Code: experiments/p0_4_orchestrator.py + experiments/p4_run.py (orchestrator); experiments/p0_4_report.py (aggregator)
- GPU: NVIDIA GeForce RTX 4060 Laptop (8 GiB)
- Concurrency: maximum 2 p4_run children
- Stagger: 30 seconds
- Start time: 2026-09-09 06:10 (orchestrator launch)
- End time: 2026-09-09 06:56 (last raw row written; `P0_4_ORCHESTRATOR DONE`)
- Wall-clock total: approximately 45 minutes
- Total fit time: 5186.3s; mean fit time 144.064s; maximum fit time 195.1s
- Total recorded wall time: 5435s; mean recorded wall time 150.972s; maximum recorded wall time 200s
- Peak CUDA memory: mean 942 MiB; maximum 942 MiB
- Resume contract: orchestrator reads P0_4_raw.csv at startup and skips rows already marked `OK`
- Per-run stdout logs: experiments/logs/p0_4_<weighting>_<setting>_baseline_seed<seed>.log

## 2. Completion status

All 36 unique tasks completed `OK`. `P0_4_raw.csv` contains 36 data rows: `OK=36`, `FAILED=0`.

| run | status | checkpoint | source | log |
|---|---|---|---|---|
| wequal d0_large baseline seed2022 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wequal_d0_large_baseline_seed2022.log |
| wequal d0_large baseline seed2023 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wequal_d0_large_baseline_seed2023.log |
| wequal d0_large baseline seed2024 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wequal_d0_large_baseline_seed2024.log |
| wequal d0_large baseline seed2025 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wequal_d0_large_baseline_seed2025.log |
| wequal d1_large baseline seed2022 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wequal_d1_large_baseline_seed2022.log |
| wequal d1_large baseline seed2023 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wequal_d1_large_baseline_seed2023.log |
| wequal d1_large baseline seed2024 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wequal_d1_large_baseline_seed2024.log |
| wequal d1_large baseline seed2025 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wequal_d1_large_baseline_seed2025.log |
| wequal d2_large baseline seed2022 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wequal_d2_large_baseline_seed2022.log |
| wequal d2_large baseline seed2023 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wequal_d2_large_baseline_seed2023.log |
| wequal d2_large baseline seed2024 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wequal_d2_large_baseline_seed2024.log |
| wequal d2_large baseline seed2025 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wequal_d2_large_baseline_seed2025.log |
| wsample d0_large baseline seed2022 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsample_d0_large_baseline_seed2022.log |
| wsample d0_large baseline seed2023 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsample_d0_large_baseline_seed2023.log |
| wsample d0_large baseline seed2024 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsample_d0_large_baseline_seed2024.log |
| wsample d0_large baseline seed2025 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsample_d0_large_baseline_seed2025.log |
| wsample d1_large baseline seed2022 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsample_d1_large_baseline_seed2022.log |
| wsample d1_large baseline seed2023 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsample_d1_large_baseline_seed2023.log |
| wsample d1_large baseline seed2024 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsample_d1_large_baseline_seed2024.log |
| wsample d1_large baseline seed2025 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsample_d1_large_baseline_seed2025.log |
| wsample d2_large baseline seed2022 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsample_d2_large_baseline_seed2022.log |
| wsample d2_large baseline seed2023 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsample_d2_large_baseline_seed2023.log |
| wsample d2_large baseline seed2024 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsample_d2_large_baseline_seed2024.log |
| wsample d2_large baseline seed2025 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsample_d2_large_baseline_seed2025.log |
| wsqrt d0_large baseline seed2022 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsqrt_d0_large_baseline_seed2022.log |
| wsqrt d0_large baseline seed2023 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsqrt_d0_large_baseline_seed2023.log |
| wsqrt d0_large baseline seed2024 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsqrt_d0_large_baseline_seed2024.log |
| wsqrt d0_large baseline seed2025 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsqrt_d0_large_baseline_seed2025.log |
| wsqrt d1_large baseline seed2022 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsqrt_d1_large_baseline_seed2022.log |
| wsqrt d1_large baseline seed2023 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsqrt_d1_large_baseline_seed2023.log |
| wsqrt d1_large baseline seed2024 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsqrt_d1_large_baseline_seed2024.log |
| wsqrt d1_large baseline seed2025 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsqrt_d1_large_baseline_seed2025.log |
| wsqrt d2_large baseline seed2022 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsqrt_d2_large_baseline_seed2022.log |
| wsqrt d2_large baseline seed2023 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsqrt_d2_large_baseline_seed2023.log |
| wsqrt d2_large baseline seed2024 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsqrt_d2_large_baseline_seed2024.log |
| wsqrt d2_large baseline seed2025 | OK | metrics_summary.csv | P0-4 | experiments/logs/p0_4_wsqrt_d2_large_baseline_seed2025.log |

## 3. Regression anchor

The 12 `w_equal` rows were compared field-by-field with the 12 `joint, ratio=50, baseline` rows in `experiments/P0_2_ratio_sweep/P0_2_raw.csv`.

- Anchored runs: 12
- Field comparisons: 36 (per_dt_test_auc, best_epoch, stopped_epoch for each run)
- `per_dt_test_auc`: 12 rows, 6 decimal places, identical
- `best_epoch`: 12 rows, identical
- `stopped_epoch`: 12 rows, identical
- Differences: none
- Result: `P0-4_ANCHOR_CHECK PASS`

## 4. Report outputs

The report command completed successfully.

| file | data rows |
|---|---:|
| experiments/P0_4_loss_weighting/P0_4_cells.csv | 432 |
| experiments/P0_4_loss_weighting/P0_4_summary.csv | 6 |
| experiments/P0_4_loss_weighting/P0_4_headline.csv | 6 |

- `P0_4_cells.csv` MISMATCH count: 0
- `P0_4_cells.csv` censored count: 18
- censored rows: selection=`sample`, count=18
- `P0_4_summary.csv` rows: 6
- `P0_4_headline.csv` rows: 6

## 5. Failed runs and missing metrics

No failed runs.

No missing metrics.

- 36 run directories under `experiments/P0_4_loss_weighting/`
- 36 run directories contain `config.json`, `metrics_epoch.csv`, and `metrics_summary.csv`
- 36 per-run logs under `experiments/logs/`
- P0-4 orchestrator lock removed after graceful exit
- No Python orchestrator process remained after completion
- Error scan of `orchestrator.log` and per-run logs: no `traceback`, `error`, `exception`, `FAILED`, `cuda out of memory`, or `killed` matches
