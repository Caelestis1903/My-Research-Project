# Execution cost inventory (raw)

Updated: 2026-11-11 03:52:11 +08:00. Counts come from committed epoch records and block completion receipts. Durations below are recorded duration fields; they are not interchangeable with end-to-end wall time.

## Formal training

| Attempt | Source hash | Committed epochs | Committed Adam updates | State |
|---|---|---:|---:|---|
| attempt1 | `575fb1c9bc7f87e900bf56007e3fb21728c92e4e95756c68965313a5818d8b36` | 250 | 5,000 | `ABORTED_ENGINEERING_PERFORMANCE`; completed ML-clean seed trajectories are discarded and excluded from attempt2 totals. |
| attempt2 | `4411eec43005a8e432f0de81d426c08a30f09960e84f7b1302e92b29ae4212e5` | 1,000 | 34,500 | All 20 jobs × 50 epochs completed, five seeds (2022–2026), four blocks complete. |

Across formal attempts, 39,500 Adam updates were executed; the final attempt contributes 34,500 and the discarded attempt contributes 5,000. The attempt1 abort record reports 2h09m parent wall time including 8m10s for completed training, and 2h00m19s of discarded attribution time. See `attempt1/EXECUTION_ABORT_RECORD.md` and each attempt's `PREPARED.json` and `epoch_commit_ledger.jsonl`.

Attempt2 block receipt `duration_seconds` values (attribution durations) are:

| Block | Duration seconds |
|---|---:|
| `movielens__clean` | 5,821.589034 |
| `kuairand__clean` | 10,455.121223 |
| `movielens__gaussian_d0` | 5,176.990139 |
| `kuairand__gaussian_d0` | 7,715.063751 |
| **Sum** | **29,168.764147** |

The sum is the four `BLOCK_COMPLETE.json` duration fields; it is not the attempt's end-to-end elapsed time. The 1,000 ordinary epoch records contain `cost.wall_seconds` totaling **29,280.573718 seconds**. This is the sum of worker epoch wall times across jobs and is not overall wall-clock duration.

## Engineering smokes

- Current attempt2: one PASS invocation per dataset (two total), source hash `4411eec43005a8e432f0de81d426c08a30f09960e84f7b1302e92b29ae4212e5`. Logs and receipts: `smoke_movielens.log`, `SMOKE_MOVIELENS.json`, `smoke_kuairand.log`, `SMOKE_KUAIRAND.json`. Source-defined true-Adam counts are 85 for MovieLens (`4*20+5`) and 201 for KuaiRand (`4*49+5`), 286 combined. Recorded per-objective `smoke_seconds` sum to 94.040s for MovieLens and 194.629s for KuaiRand (288.669s combined); these are not end-to-end CLI wall times.
- Earlier engineering area: six PASS command logs are retained under `engineering/` (three per dataset: initial, final, review-fixes). The retained full receipt pair is hash `575fb1c9bc7f87e900bf56007e3fb21728c92e4e95756c68965313a5818d8b36`, with `smoke_seconds` totals 21.857s (MovieLens) and 46.310s (KuaiRand). `smoke_ml.log` also retains one earlier PASS output with hash `9f9530f9785bc93b5339c15fc2f92c2bf277486a2442116a145184e898d0f43c` and 13.933 recorded smoke seconds. Other older PASS logs lack per-invocation source hashes or full receipts. Their exact end-to-end wall time and Adam costs are not recorded.
- Copies of the earlier receipts under `attempt1/engineering/` and `attempt1/SMOKE_*.json` are not additional invocations and are not double-counted.

The smoke runner uses `4*steps_per_epoch+5` true Adam updates per dataset invocation: two objectives, each with baseline and observed epochs, two recovery next-step comparisons, and one block-calibration source step. Raw receipt paths: `attempt2/SMOKE_*.json`, `attempt1/engineering/SMOKE_*.json`.

## Raw aggregation outputs

`analyze --stage 1`, `2`, `3`, and `4` each returned `COMPLETE` for 20 jobs, seeds 2022–2026, and 34,500 real Adam updates. Output schemas are committed as `VALIDATION_CURVES.csv`, `SAME_TIME_SEED_EFFECTS.csv`, and `STAGE_SEED_SUMMARY.csv`; stage receipts are `STAGE1_COMPLETE.json` through `STAGE4_COMPLETE.json`. These stages aggregate committed observations only. No freeze, Test evaluation, report generation, or release was run as part of this inventory update.
