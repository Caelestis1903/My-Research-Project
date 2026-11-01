# Failed Attempt Runner Report

## Identity and configuration

- Experiment: `gradient_aug_time_select_20261102`
- Attempt: `attempt_20261102_0005`
- Reviewed execution commit: `ac2ec5cc70be7dcb880a7dde1015e4634726fd96`
- Source manifest hash: `ff5136395313436fa8b96f7bf585245eb0c6e7f2607c1caad40f4d4013bea130`
- Runtime: `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`; requested full mode, devices `0,1`, batch `40960`, `50` epochs, FP32.
- This report records the failed attempt. No retry was started after the coordinator error; its attempt directory and files were retained.

## Final run status

The trial ledger has 20 unique `(dataset, trial, seed)` keys after deduplication by each key's latest state. All 20 latest states are `COMPLETE`, with 50/50 epochs recorded per run. The breakdown is:

| Phase / trial | Unique completed keys |
|---|---:|
| Calibration A0 | 2 |
| Development A0 | 2 |
| Development A1 | 4 |
| Development B1 | 4 |
| Development B2 | 4 |
| Development B3 | 4 |
| **Total** | **20** |

Failure summary: training-run failures `0`; fatal coordinator errors `1`; wrapper startup rejections `1`. The coordinator did not complete the full queue. It exited with status 1 after development B3 and before confirmation and later queue stages. No confirmation runs for seeds 2023, 2025, or 2026 were dispatched. No `conditional_design_request`, `selection_frozen` receipt, or `state_release` receipt was produced.

## Coordinator error and state

`completion.json` records `FAILED`, `KeyError('encoder_hash')`. The traceback identifies `historical_baseline_verification` in `experiments/gradient_aug_time_select_20261102/run.py:260`, accessing `data[d]['encoder_hash']`. At exit, `progress.json` recorded development completed `4/4`, active `0`, pending `0`, failed `0`, and `test_accessed=false`.

The coordinator process exited, so its RAM-only coordinator state was no longer resident. No unified frozen test was executed and no test metrics were accessed. The attempt recorded `test_accessed=false`; Stage0 recorded `test_evaluated=false` for both datasets.

## Raw runtime and resource accounting

The cost table uses exactly one `runtime_stats.json` per unique `(dataset, trial, seed)` key, for 20 rows. A0 rows use the `primary` record; augmented trial rows use the combined record at the attempt root for that run. The 52 runtime files include component-level primary/replay records alongside combined records; they are retained as raw artifacts and are not summed together. Per-key source paths and all included raw counters are in `failed_attempt_costs.csv`.

| Record | Field | Value |
|---|---|---:|
| `budget_calibration.json` | development worker seconds | 21367.9183922371 |
| `budget_calibration.json` | predicted worker seconds | 17870.690975639038 |
| `budget_calibration.json` | cost ratio | 1.1956962616255529 |
| 20 unique-key records, summed | training / worker seconds | 21367.9183922371 |
| 20 unique-key records, summed | original exposures | 2525669100 |
| 20 unique-key records, summed | extra views | 191412780 |
| 20 unique-key records, summed | view draws | 53820 |
| 20 unique-key records, summed | training passes | 36 |
| 20 unique-key records, summed | training forwards | 115920 |
| 20 unique-key records, summed | gradient calls | 117300 |
| 20 unique-key records, summed | optimizer steps | 62100 |
| 20 unique-key records, summed | validation forwards | 19300 |

The throughput calibration record reports MovieLens `180.66079627798172` seconds, peak `983312384` bytes, and KuaiRand `563.9513277069782` seconds, peak `1092386816` bytes. Stage0 reports MovieLens peak allocated `2079853568` bytes / peak reserved `2298478592` bytes and KuaiRand peak allocated `2220430336` bytes / peak reserved `2589982720` bytes; each dataset reports 13 optimizer updates and PASS. Stage0 receipt: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005_check/stage0_checks.json`.

## Logs and artifacts

- Run artifacts and ledgers: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005/`
- Full-run stdout: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005.stdout.log`
- Full-run stderr and final traceback: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005.stderr.log`
- Stage0 artifacts: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005_check/`
- The first wrapper start attempt exited with `FileExistsError` because its run directory already contained the output logs and dispatched no experiment. Its original stderr was overwritten. `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005_initial_wrapper_error.stderr.log` is a transcription of a contemporaneous status message, not the original raw traceback.
- Per-run stdout/stderr, validation records, traces, CSV/JSON outputs, all 52 raw runtime summary files, ledgers, and completion/progress records remain in the attempt artifacts. The corrected unique-key cost export is `research_reports/2026-11-02_gradient_aug_time_select/failed_attempt_costs.csv`.

## Not run

The confirmation stage, conditional queue, frozen test, selection freeze, and state release were not reached. No additional run was started after the coordinator error. The experiment's absolute task-time budget remains anchored to the first formal training start; it was not reset for this report.
