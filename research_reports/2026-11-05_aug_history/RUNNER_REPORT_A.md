# EXPERIMENT EXECUTION REPORT — Route A

## 1. Experiment ID

`AUG_HISTORY_20261105`, route `A`, attempt `A1`.

## 2. Git and worktree state

- Worktree: `/home/xzf/Code/phase0-3_aug_history_20261105`
- Branch: `26-11-05`
- HEAD: `f00b5ce6b9e6476ca689b781667a5d698742aa70`
- Source hash: `afecc8dcef7f1687d61383af0ca69176eb7b7ec26d0700eb43836a827b5e2ff9`
- Manifest hash: `76345194a8d09c8cadf3544e39c35174a452abb3827f071250da7f259c45749d`
- Attempt output: `artifacts/aug_history_20261105/attempt_A1`
- At runner handoff, the A scalar artifacts awaited the orchestrator's result commit.

## 3. Config

Plan: `research_plans/26-11-5/experiment_manifest.json`. Route A; datasets `movielens`, `kuairand`; batch size 40960; 50 epochs; five seeds per dataset; devices `0 1`; resume enabled. Prepared identities and per-job configs are in `PREPARED.json`. The A selection was frozen on validation before any A test evaluation.

## 4. Seeds

`2022`, `2023`, `2024`, `2025`, `2026` for each dataset.

## 5. Run status

All ten prepared jobs completed 50/50 epochs (500 committed epochs total). Exact job-ID epoch counts were obtained from each `PREPARED.json` job's corresponding `runs/<job_id>/epochs/E*.json` files:

| Dataset | Seed 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|
| MovieLens | 50 | 50 | 50 | 50 | 50 |
| KuaiRand | 50 | 50 | 50 | 50 | 50 |

The same `train ... --resume` command was run once after the original coordinator had exited and all 500 epoch files were present. It exited 0 with `TRAIN_COMPLETE`, `jobs: 10`.

Frozen validation selection: `HPO_FROZEN`, selection hash `ba672ccb017266847085f3de6ccd72cbec8115c93489b4c89b84517b21d8a4c2`.

Validation-only decision: command exited 0; `test_consulted: false`. Raw actions recorded by the CLI:

- MovieLens selected candidate `F`; `next_action: RUN_C`; C action is `C_FULL`, five seeds.
- KuaiRand selected candidate `A_FIXED_S_FAST`; `next_action: RETAIN_A_OR_B_AND_REPORT`; C deferred.

Frozen test evaluation: exited 0 with `EVALUATED`; 360 endpoint rows; endpoint file SHA-256 `0c8d113e88214c60e3555b92e809c238d8d8bd274e8bd872b7052f38d377e226`.

Report command exited 0: 540 endpoint rows, 39,600 paired rows, 7,920 summary rows.

Continuation command exited 0 with `CONTINUATION_RESOLVED`; it emitted the MovieLens C command and specification path. No C code or run was started by this runner.

## 6. Raw metrics and output files

Raw validation and test metrics are preserved in:

- `artifacts/aug_history_20261105/attempt_A1/A/ENDPOINTS.csv`
- `artifacts/aug_history_20261105/attempt_A1/A/PAIRED_SEED.csv`
- `artifacts/aug_history_20261105/attempt_A1/A/PAIRED_SUMMARY.csv`
- `artifacts/aug_history_20261105/attempt_A1/A/HPO_SUMMARY.csv`
- `artifacts/aug_history_20261105/attempt_A1/A/VALIDATION_CURVES.csv`
- `artifacts/aug_history_20261105/attempt_A1/A/COSTS.csv`
- Frozen records: `A/selection_frozen.json`, `A/DECISION.json`, and `A/EVALUATED.json`.

## 7. Logs and receipts

- Run manifest/admissions/retries/epoch ledger: `RUN_MANIFEST.json`, `ADMISSIONS.jsonl`, `RETRIES.jsonl`, `epoch_commit_ledger.jsonl`
- Per-job logs: `runs/<job_id>/worker.log`
- Coordinator/guardian log: `guardian.log`
- Preflight/acceptance evidence: `SOURCE_RECEIPT.json`, `STAGE0_CHECKS.json`, `GPU_SMOKE.json`, `validation_tests.log`
- Selection, evaluation and continuation records: `A/selection_frozen.json`, `A/EVALUATED.json`, `CONTINUATION.json`

The live guardian PID 65260 remains running to retain RAM state. Its IPC authentication material is intentionally not included here.

## 8. Runtime errors and execution notes

- The frozen evaluation emitted a NumPy non-writable-array warning from `experiments/overnight_ab_20261103/evaluation.py:15`; evaluation completed with exit 0.
- Three worker rotations occurred: authorized KuaiRand 2023 E39→E40; authorized KuaiRand 2024 E33→E34; and an accidental repeat of the KuaiRand 2024 replacement PID 80144 at E39/40 with approximately 29% RAM available. Coordinators returned RuntimeError after the intentional SIGTERM events, and execution resumed from retained RAM commits. The repeated rotation was unnecessary. The final supervisor restored KuaiRand 2024 at E40. All committed epochs were retained; partial uncommitted work has no complete timing record.

## 9. Missing runs

None for route A: all 10/10 jobs and 500/500 epochs completed; frozen test evaluation and report completed. The continuation receipt resolves a future MovieLens C_FULL run, but C is outside this completed A execution and was not started.

## A lifecycle release

After frozen evaluation, scalar reports, and continuation resolution were complete, the authorized CLI command

```text
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.aug_history_20261105.run release --route A --manifest research_plans/26-11-5/experiment_manifest.json --output artifacts/aug_history_20261105/attempt_A1
```

ran once and exited 0 with `RELEASED`. A subsequent `ps -p 65260` check returned no process row; guardian PID 65260 had exited. `A/EVALUATED.json` still reports `EVALUATED`.
