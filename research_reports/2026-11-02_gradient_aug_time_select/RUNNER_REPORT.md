# EXPERIMENT EXECUTION REPORT

## 1. Experiment ID

- Experiment: `gradient_aug_time_select_20261102` (GATS02)
- Final attempt: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0440`
- Stage0 attempt: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0440_check`

## 2. Git/worktree state and execution identity

The shared worktree has no `.git` directory. Execution used the reviewed source identity recorded by the runner receipts:

- Execution commit: `be0227e510f00864b8c7ec7565def0b5a813be4f`
- Source manifest hash: `7f7a0eecb052ca0be1f5f7ffc07433e85d5e2d73bcbfd7e2f6f0ff4efac328a1`
- `cross_attempt_identity.json`: `PASS`
- Stage0 receipt matched the execution commit and source hash.

## 3. Config

- Python: `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`
- Mode: `all`; requested devices: `0,1`
- Datasets: MovieLens and KuaiRand
- Batch size: 40960; epochs: 50; precision: FP32
- Training/checkpoint states were RAM-only until frozen unified test and table generation; runner then released the states.
- Full-run stdout/stderr: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0440.stdout.log` and `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0440.stderr.log`.

## 4. Seeds and dispatched trajectories

The official `completion.json` records 31 complete unique scientific trajectories:

- Calibration A0: 2 trajectories (both datasets, seed 2022).
- Development: 18 trajectories (A0 seed 2024; A1, B1, B2, B3 seeds 2022 and 2024 on both datasets).
- Matched controls: 8 trajectories (R1: 2; R2: 4; R3: 2), with seeds and dataset keys in `trial_ledger.csv`.
- Confirmation: 3 trajectories (KuaiRand B2, seeds 2023, 2025, 2026).

Every planned/running/completed row is retained in `trial_ledger.csv`; all 31 current-attempt unique trajectories ended `COMPLETE` at 50/50 epochs. The R3 native-failure retry for KuaiRand B2 seed 2022 was dispatched as attempt 2 with serial retry metadata; it completed in this attempt.

## 5. Run status

- Stage0: `PASS` for MovieLens and KuaiRand; details are in [STAGE0_REPORT.md](STAGE0_REPORT.md).
- Development: A0/A1/B1/B2/B3 completed for the planned dataset/seed keys.
- Matched controls: R1/R2/R3 completed for the planned dataset/seed keys.
- Confirmation: KuaiRand B2 seeds 2023/2025/2026 completed.
- Unified frozen test: evaluated; `test_evaluated=true` in `completion.json`.
- Selection freeze: `selection_frozen.json` was produced before unified test.
- State release: `state_release.json` records `released=true`, `storage=RAM_ONLY`, after frozen unified test and tables.
- Completion: `COMPLETE`; 31 unique trajectories, 58 current-attempt complete physical passes, and 1 inherited partial physical pass. Coordinator exit code: 0.
- `failure_ledger.csv` for the final attempt contains its header only. No new failed runs were recorded in this R3 attempt.

## 6. Raw metrics and physical cost

Raw validation, test, selected, and per-trajectory cost rows are retained in `validation_metrics.csv`, `test_metrics.csv`, `selected_metrics.csv`, and `costs.csv`. These files contain metric values; this report does not interpret them.

The official `execution_cost_continuity.json` reports the following combined totals across the referenced attempts. Its listed lower-bound fields remain lower bounds; pass costs are sourced through the official combined physical-cost accounting, without adding combined and child rows together.

| Field | Combined total |
|---|---:|
| Complete physical training passes | 113 |
| Inherited partial physical passes | 1 |
| Training worker seconds | 63039.43954395433 (lower bound) |
| Total epochs | 5650 |
| Optimizer steps | 194836 |
| Original exposures | 7922751724 |
| Extra views | 615497907 (lower bound) |
| View draws | 170976 (lower bound) |
| Training forwards | 365812 (lower bound) |
| Gradient calls | 431580 |
| Shadow updates | 468482 (lower bound) |
| Validation forwards | 60444 (lower bound) |
| Validation wall seconds | 4144.641697684245 (lower bound) |
| Training GPU event milliseconds | 53614962.756923676 (lower bound) |
| Validation GPU event milliseconds | 3081924.8648290634 (lower bound) |
| Selection checks | 12172 (lower bound) |
| Task wall seconds from original task start | 29992.620189433044 |

The current attempt contributed 58 complete passes and 31808.93607985397 training worker seconds. Cost details and source rows are in `physical_cost_ledger.csv`, `physical_cost_sources.json`, `execution_cost_continuity.json`, and `prior_physical_cost_ledger.csv`.

## 7. Log and receipt locations

- Current full-run stdout/stderr: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0440.stdout.log`, `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0440.stderr.log`
- Current trial/launch/failure ledgers: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0440/trial_ledger.csv`, `launch_ledger.csv`, `failure_ledger.csv`
- Current completion/cost/release receipts: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0440/completion.json`, `execution_cost_continuity.json`, `physical_cost_ledger.csv`, `physical_cost_sources.json`, `state_release.json`
- Frozen selections/configuration: `selection_frozen.json`, `configuration_frozen.json`, and per-control `*_frozen.json` in the same attempt directory
- Raw run logs and metrics: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0440/runs/`
- Stage0 receipt: `artifacts/gradient_aug_time_select_20261102/attempt_20261102_recovery_0440_check/stage0_checks.json`

## 8. Runtime errors and prior failed attempts

- Initial attempt `attempt_20261102_0005`: after its completed training runs, the coordinator stopped at historical baseline verification with `KeyError: 'encoder_hash'`. This was a coordinator error, not a recorded training-run failure; see `FAILED_ATTEMPT_RUNNER_REPORT.md` and its retained attempt artifacts.
- Recovery attempt `attempt_20261102_recovery_0315`: KuaiRand B2 seed 2022 had a native process exit `-11` during replay. The native cause was not established. The failed/partial artifacts and raw diagnostics are recorded in `FAILED_RECOVERY_RUNNER_REPORT.md`.
- Current attempt `attempt_20261102_recovery_0440`: failure ledger empty; no new failed training runs. The B2 seed 2022 retry completed as attempt 2.
- An earlier wrapper invocation was refused because the target run directory already existed. It dispatched no training. Its record is in the prior runner reports and outer logs.

## 9. Missing runs

No missing runs are recorded in the final attempt ledger for the dispatched final plan. `completion.json` records all 31 unique trajectories complete and the unified test evaluated. The final queue produced `selection_frozen.json` and `state_release.json`.
