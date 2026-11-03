# Experiment Execution Report

## Experiment and execution

- Experiment ID: `mdmt_five_seed_completion_20261103`
- Review gate: PASS; run authorized after independent review.
- Reviewed package source hash: `db8d3385f3acabb11e2f10207e6adafb88cc949b51b7b9553a5117434366b7d6`
- Run command: `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_five_seed_completion_20261103.run run --devices 0,1`
- Summarize command: `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_five_seed_completion_20261103.run summarize`
- Attempt: `artifacts/mdmt_five_seed_completion_20261103/attempt1`
- Status: `COMPLETE`; summarize reported 30 new runs, 180 new test endpoints, and 20 decisions.

## Configuration and run accounting

- Prepared matrix: 30 new training runs and 10 exact-identity reused runs (`PREPARED.json`).
- New run files: 24 `history` and 6 `pair` jobs. Seeds present in new job records: 2023, 2024, 2025, 2026. The reused runs remain recorded by the preparation/identity artifacts.
- Training budget: 50 epochs per run; result artifacts record 1,500 total epochs and 51,750 optimizer steps across the 30 new runs.
- Batch size: 40,960. Early stopping: disabled. Configuration and selection freeze records are in the attempt root and each family directory.
- Run command `progress.json`: complete 30/30; active 0; pending 0; failed 0; recorded wall time 5,931.878 seconds.
- End-to-end interval recorded in `COMPLETE.json`: 6,236.235 seconds (`finished - started`).
- Sum of the 30 `worker_seconds` values in the new `result.json` files: 46,800.278 seconds. All 30 results record attempt 1.

## Test evaluation and raw metrics

- `history/final_evaluation.json`: COMPLETE, 144 endpoints and 144 raw rows.
- `pair/final_evaluation.json`: COMPLETE, 36 endpoints and 36 raw rows.
- Total: 180 newly evaluated endpoints; `COMPLETE.json` and summarize both record 180.
- Raw per-endpoint AUC, log loss, domain/task cells, counts, and evaluation cost are preserved in the `metrics` and `cost` fields of those 180 rows. Per-run training losses, validation records, and selected epochs are in each family's `runs/*/result.json` and `epoch_commit_ledger.jsonl` artifacts.
- Summarize stdout: `{"status":"COMPLETE","endpoint_rows":660,"decisions":20}`. The 660 rows include the summary's broader endpoint collection; the 180 new endpoints are identified in `COMPLETE.json` and the family `final_evaluation.json` files.

## Failures, retries, and runtime notes

- Failed new runs: 0.
- Recovered/retried new runs: 0. All 30 result files have `attempt: 1`; the 30 launch-ledger records have `previous_attempts: 0`.
- Runtime error: none recorded. `runner.log` contains a PyTorch TF32 legacy-API deprecation warning.
- Full stdout/stderr log: `artifacts/mdmt_five_seed_completion_20261103/attempt1/runner.log`; per-job logs are in `attempt1/launch_logs/`.

## Guardian shutdown

- `history/SHUTDOWN.json`: `SHUTDOWN`.
- `pair/SHUTDOWN.json`: `SHUTDOWN`.
- Both family `final_evaluation.json` files report `COMPLETE`; the attempt-root `COMPLETE.json` reports `COMPLETE` with no pending runs.
