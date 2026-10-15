# Raw Experiment Report — P0-5

## 1. Experiment metadata

- Stage: P0-5 (sample loss-weighting × comp1 joint, fixed ratio 50)
- Dataset: ml-1m (MDMTRec domain configuration: d0=d1=d2 in ml-1m; mappings rotate per setting)
- Model: MDMTRec (from m3oe/examples)
- Method: joint sample-weighted × comp1, 3 settings (d0_large, d1_large, d2_large)
- Seed: 2022, 2023, 2024, 2025
- Total fresh tasks: 12 (1 weighting × 3 settings × 4 seeds × 1 joint comp1 arm)
- Ratio role_sizes contract: ratio=50, L/M/S = 150000/34500/3000
- Code: experiments/p0_5_orchestrator.py + experiments/p4_run.py (orchestrator); experiments/p0_5_report.py (aggregator)
- GPU: NVIDIA GeForce RTX 4060 Laptop (8 GiB)
- Concurrency: maximum 2 p4_run children
- Stagger: 30 seconds
- Start time: 2026-09-09 07:52 (orchestrator launch)
- End time: 2026-09-09 08:12 (last raw row written; `P0_5_ORCHESTRATOR DONE`)
- Wall-clock total: approximately 20 minutes
- Total fit time: 2210.8s; mean fit time 184.23s; maximum fit time 225.1s
- Total recorded wall time: 2290s; mean recorded wall time 190.83s; maximum recorded wall time 230s
- Peak CUDA memory: 1158 MiB (uniform across all 12 runs)
- Resume contract: orchestrator reads P0_5_raw.csv at startup and skips rows already marked `OK`
- Per-run stdout logs: experiments/logs/p0_5_p0_5_<setting>_comp1_seed<seed>.log

## 2. Completion status

All 12 unique tasks completed `OK`. `P0_5_raw.csv` contains 12 data rows: `OK=12`, `FAILED=0`.

| run | status | checkpoint | source | log |
|---|---|---|---|---|
| sample joint d0_large comp1 seed2022 | OK | metrics_summary.csv | P0-5 | experiments/logs/p0_5_p0_5_d0_large_comp1_seed2022.log |
| sample joint d0_large comp1 seed2023 | OK | metrics_summary.csv | P0-5 | experiments/logs/p0_5_p0_5_d0_large_comp1_seed2023.log |
| sample joint d0_large comp1 seed2024 | OK | metrics_summary.csv | P0-5 | experiments/logs/p0_5_p0_5_d0_large_comp1_seed2024.log |
| sample joint d0_large comp1 seed2025 | OK | metrics_summary.csv | P0-5 | experiments/logs/p0_5_p0_5_d0_large_comp1_seed2025.log |
| sample joint d1_large comp1 seed2022 | OK | metrics_summary.csv | P0-5 | experiments/logs/p0_5_p0_5_d1_large_comp1_seed2022.log |
| sample joint d1_large comp1 seed2023 | OK | metrics_summary.csv | P0-5 | experiments/logs/p0_5_p0_5_d1_large_comp1_seed2023.log |
| sample joint d1_large comp1 seed2024 | OK | metrics_summary.csv | P0-5 | experiments/logs/p0_5_p0_5_d1_large_comp1_seed2024.log |
| sample joint d1_large comp1 seed2025 | OK | metrics_summary.csv | P0-5 | experiments/logs/p0_5_p0_5_d1_large_comp1_seed2025.log |
| sample joint d2_large comp1 seed2022 | OK | metrics_summary.csv | P0-5 | experiments/logs/p0_5_p0_5_d2_large_comp1_seed2022.log |
| sample joint d2_large comp1 seed2023 | OK | metrics_summary.csv | P0-5 | experiments/logs/p0_5_p0_5_d2_large_comp1_seed2023.log |
| sample joint d2_large comp1 seed2024 | OK | metrics_summary.csv | P0-5 | experiments/logs/p0_5_p0_5_d2_large_comp1_seed2024.log |
| sample joint d2_large comp1 seed2025 | OK | metrics_summary.csv | P0-5 | experiments/logs/p0_5_p0_5_d2_large_comp1_seed2025.log |

## 3. Regression anchor

The 12 P0-5 sample/comp1 rows were compared field-by-field with the 12 `joint, ratio=50, comp1` rows in `experiments/P0_2_ratio_sweep/P0_2_raw.csv`.

`experiments/tmp_kr/p0_5_anchor_check.py` output:

```
=== P0-4 sample/off vs P0-2 equal/baseline ===
  OK=12 MISMATCH=0 MISSING=0
=== P0-5 sample/comp1 vs P0-2 equal/comp1 ===
  OK=12 MISMATCH=0 MISSING=0
P0_5_ANCHOR_CHECK PASS: P0-5 sample/comp1 configs match P0-2 equal/comp1 (allowed diffs only); P0-4 sample/off vs P0-2 equal/baseline match too.
```

- Anchored runs: 12
- Field comparisons: 36 (per_dt_test_auc, best_epoch, stopped_epoch for each run)
- `per_dt_test_auc`: 12 rows, 6 decimal places, identical
- `best_epoch`: 12 rows, identical
- `stopped_epoch`: 12 rows, identical
- Differences: none
- Result: `P0_5_ANCHOR_CHECK PASS`

## 4. Report outputs

The report command completed successfully.

| file | data rows |
|---|---:|
| experiments/P0_5_comp1_x_weighting/P0_5_cells.csv | 288 |
| experiments/P0_5_comp1_x_weighting/P0_5_summary.csv | 4 |
| experiments/P0_5_comp1_x_weighting/P0_5_headline.csv | 4 |

- `P0_5_cells.csv` MISMATCH count: 0 (all 288 rows mismatch=False)
- `P0_5_summary.csv` rows: 4
- `P0_5_headline.csv` rows: 4

## 5. Failed runs and missing metrics

No failed runs.

No missing metrics.

- 12 run directories under `experiments/P0_5_comp1_x_weighting/w_sample/<setting>/comp1/`
- 12 run directories contain `config.json`, `metrics_epoch.csv`, and `metrics_summary.csv`
- 12 per-run logs under `experiments/logs/p0_5_p0_5_*.log`
- P0-5 orchestrator lock removed after graceful exit (`P0_5_ORCHESTRATOR DONE` written)
- No Python orchestrator process remained after completion
- Error scan of `orchestrator.log` and per-run logs: no `traceback`, `error`, `exception`, `FAILED`, `cuda out of memory`, or `killed` matches