# EXPERIMENT EXECUTION REPORT

## 1. Experiment ID

`gradient_aug_time_select_20261102`, attempt `attempt_20261102_0005`.

## 2. Git/worktree and execution identity

The shared workspace `/home/xzf/Code/phase0-3` has no `.git` directory. The runner used the reviewed execution identity recorded in the Stage0 receipt and run manifests:

- Execution commit: `ac2ec5cc70be7dcb880a7dde1015e4634726fd96`
- Source manifest hash: `ff5136395313436fa8b96f7bf585245eb0c6e7f2607c1caad40f4d4013bea130`
- Stage0 receipt identity matched both requested values.
- Full attempt identity/configuration: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005/source_manifest.json`, `resolved_config.json`, `environment.json`.

## 3. Config

- Python: `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`
- Mode requested: `all`; devices: `0,1`.
- Batch size: 40960; epochs: 50; precision: FP32.
- Trial plan seeds: 2022, 2023, 2024, 2025, 2026.
- Development dispatches observed: seed 2022 and 2024 for arms A0, A1, B1, B2, B3 across MovieLens and KuaiRand, with A0 calibration at seed 2022 on both datasets.
- Stage0 command and full run command/arguments are represented in the attempt manifest and parent execution record. Full-run stdout/stderr: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005.stdout.log` and `.stderr.log`.
- First wrapper startup error is represented by a transcription of the contemporaneous status message at `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005_initial_wrapper_error.stderr.log`; this is not the original raw traceback, which was overwritten.

## 4. Seeds

Seeds recorded as completed in the run ledger: 2022 and 2024. Seeds 2023, 2025, and 2026 were in the plan but no confirmation-stage run was dispatched before coordinator failure.

## 5. Run status

- Stage0: PASS for MovieLens and KuaiRand.
- Completed unique runs: 20 (2 calibration A0, 2 development A0, 4 development A1, 4 development B1, 4 development B2, 4 development B3). Each completed run recorded 50/50 epochs.
- Unique planned runs in the attempt ledger: 20 completed; no unique run remains RUNNING or PLANNED in that ledger snapshot. This is not full-queue completion: later confirmation/conditional phases were not dispatched.
- `completion.json`: `FAILED`; `progress.json`: development completed 4/4, active 0, pending 0, failed 0, `test_accessed=false`.
- Coordinator exited with status 1 after B3 development and before historical baseline verification completed.
- No seed confirmation runs, conditional-design request, selection-frozen receipt, or state-release receipt was produced. The coordinator process exited; the run-level completion file contains no state-release record.

## 6. Raw metrics

Per-run raw logs and metric artifacts are retained under `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005/runs/`. Each run directory contains its own `stdout.log`, `stderr.log`, validation epoch metrics, batch traces, and applicable model-specific CSV/JSON outputs. No test metrics were accessed; Stage0 also records `test_evaluated=false`.

## 7. Log locations

- Full run stdout: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005.stdout.log`
- Full run stderr/final traceback: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005.stderr.log`
- Initial wrapper startup stderr: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005_initial_wrapper_error.stderr.log`
- Trial status/config/hash ledger: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005/trial_ledger.csv`
- Launch/resource ledger: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005/launch_ledger.csv`
- Failure ledger: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005/failure_ledger.csv` (header only; no recorded run failures)
- Coordinator completion/progress: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005/completion.json`, `progress.json`

## 8. Runtime errors

The final coordinator error occurred in `historical_baseline_verification` at `experiments/gradient_aug_time_select_20261102/run.py:260`:

`KeyError: 'encoder_hash'`

It occurred while accessing `data[d]['encoder_hash']` after all four B3 development runs had completed. The complete traceback is in the full-run stderr log and `completion.json`.

An earlier startup attempt exited with `FileExistsError: Use a new attempt: artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005` because the run directory contained the runner log files. That attempt dispatched no experiment and was retried with stdout/stderr logs outside the run directory. The original stderr was overwritten; the separate initial-wrapper file is only a transcription of a contemporaneous status message, not a preserved raw traceback.

## 9. Missing runs

No confirmation stage was dispatched. The ledger contains no runs for planned seed 2023, 2025, or 2026. No full-queue completion, conditional-design request, selection-frozen record, or state-release record exists. Raw artifacts for all 20 completed runs remain available for the parent agent.
