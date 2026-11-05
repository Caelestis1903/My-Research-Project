# Experiment Execution Report — AUG_HISTORY_20261105, Route B

Mechanical run record only. This report contains execution status, configuration identifiers, counts, hashes, warnings, recoveries, and artifact paths; it makes no scientific interpretation.

## Run identity

- Worktree: `/home/xzf/Code/phase0-3_aug_history_20261105`
- Branch and implementation commit: `26-11-05`, `79690772dedefafa24b80d1817aca8c61907be5b`
- Plan: `research_plans/26-11-5/experiment_manifest.json`
- Run directory: `artifacts/aug_history_20261105/attempt1`
- Route: B
- Source hash: `9baac157d4574ccc2ec02a9db42cd365ec918346c32b1b3ace2dcc6f65893f2a`
- Manifest hash: `76345194a8d09c8cadf3544e39c35174a452abb3827f071250da7f259c45749d`
- Execution environment: `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`

## Configuration and coverage

- Datasets: MovieLens and KuaiRand.
- Seeds per dataset: 2022, 2023, 2024, 2025, 2026.
- Prepared jobs: 10 total; every prepared `job_id` has `E001.json` through `E050.json` (500 committed epoch records total).
- Epochs: 50 per job; batch size: 40,960; precision: FP32; original full validation cadence and 15 non-online validation candidate keys retained.
- Validation audit: 15 candidate records × 10 jobs × 50 epochs = 7,500 candidate records; online records = 10 × 50 = 500. Each candidate key occurs 500 times overall and covers five seeds in each dataset. No missing or incomplete epoch records.
- The exact dataset, seed, job ID, and epoch mapping is in `artifacts/aug_history_20261105/attempt1/PREPARED.json` and each job’s `runs/<job_id>/epochs/` directory.

## Command and stage record

Commands used the environment Python path above and were run from the worktree root. Paths below are relative to that root.

| Stage | Command / action | Outcome | Recorded time (Asia/Shanghai, 2026-11-05) |
|---|---|---|---|
| Preflight | `python -m experiments.aug_history_20261105.run preflight --manifest research_plans/26-11-5/experiment_manifest.json --output artifacts/aug_history_20261105/attempt1` | PASS; exit 0; source and manifest matched the reviewed receipt | Before 18:25 |
| Validation | `python -m experiments.aug_history_20261105.run validate --manifest research_plans/26-11-5/experiment_manifest.json --output artifacts/aug_history_20261105/attempt1 --suite all --gpu-smoke --devices 0 1` | PASS; exit 0; 11 passed, 2 warnings, 12.72 s; GPU smoke passed on CUDA devices 0 and 1 | 18:25:24 |
| Training invocation 1 | `python -m experiments.aug_history_20261105.run train --manifest research_plans/26-11-5/experiment_manifest.json --route B --datasets movielens kuairand --seeds 2022 2023 2024 2025 2026 --output artifacts/aug_history_20261105/attempt1 --devices 0 1 --resume` | Worker PID 42945 was SIGTERM’d under parent-authorized RAM recovery at its last committed seed-2023 epoch 27. Supervisor reported a worker failure; numeric process exit code was not retained by the runner terminal session. | First GPU admission 18:25:56; recovery 19:16:42 |
| Training invocation 2 | Same `train ... --resume` command as above | Completed all 10 jobs; terminal returned `{"status":"TRAIN_COMPLETE","jobs":10}`. The supervisor PID exited; its progress JSON was stale, so final counts were audited directly from `PREPARED.json` job IDs and epoch files. | Resume after 19:16; final job completion recorded by 19:49 |
| Validation selection | `python -m experiments.aug_history_20261105.run select --manifest research_plans/26-11-5/experiment_manifest.json --output artifacts/aug_history_20261105/attempt1 --route B --split validation` | PASS; exit 0; `HPO_FROZEN` | 19:50:51 |
| Decision | `python -m experiments.aug_history_20261105.run decide --manifest research_plans/26-11-5/experiment_manifest.json --output artifacts/aug_history_20261105/attempt1 --route B --basis validation` | PASS; exit 0; decision saved before test evaluation | 19:51:05 |
| Frozen evaluation | `python -m experiments.aug_history_20261105.run evaluate --manifest research_plans/26-11-5/experiment_manifest.json --output artifacts/aug_history_20261105/attempt1 --route B --frozen artifacts/aug_history_20261105/attempt1/B/selection_frozen.json --device cuda:1` | PASS; exit 0; `EVALUATED`; 420 new endpoint rows | 19:57:07 |
| Continue | `python -m experiments.aug_history_20261105.run continue --manifest research_plans/26-11-5/experiment_manifest.json --output artifacts/aug_history_20261105/attempt1 --route B --decision artifacts/aug_history_20261105/attempt1/B/DECISION.json --resume` | PASS; exit 0; `AWAITING_TRIGGERED_IMPLEMENTATION`; no A/C training launched by this runner | 19:57:26 |
| Report generation | `python -m experiments.aug_history_20261105.run report --manifest research_plans/26-11-5/experiment_manifest.json --output artifacts/aug_history_20261105/attempt1 --route B` | PASS; exit 0; `B_REPORTED`, 600 endpoint rows, 36,630 paired rows, 7,326 summary rows | 19:57:45 |

The training invocation timestamps are based on admission, retry, and completion records. The first invocation’s numeric exit code was not captured; the worker failure and recovery are retained in `RETRIES.jsonl` and the worker logs. No partial-epoch GPU seconds are estimated.

## RAM recoveries and partial-work accounting

Four parent-authorized RAM recoveries were performed. In each case, only the identified KuaiRand worker was sent SIGTERM; the RAM guardian and other jobs were retained. The resume supervisor restored the last committed state and replaced the worker.

| Dataset / seed | Recycled worker | Last committed state before recycle | Resume state observed |
|---|---:|---|---|
| KuaiRand / 2023 | PID 42945 | E027, global step 1323 | Replacement resumed at E027, then advanced |
| KuaiRand / 2022 | PID 42717 | E035, global step 1715 | Replacement resumed at E035, then advanced |
| KuaiRand / 2024 | PID 43143 | E039, global step 1911 | Replacement resumed from E039 and advanced through E040 |
| KuaiRand / 2025 | PID 43665 | E040, global step 1960 | Replacement resumed at E040, then advanced |

`RETRIES.jsonl` records four `PAUSED_RAM` records, each with retry count 1 and the corresponding committed epoch index. The amount of uncommitted work at each signal and its GPU time are not separately recorded; partial-epoch cost is unknown. No model, optimizer, Q, or parameter checkpoints were written to disk. RAM guardian PID 37876 remains active; keep it until the orchestrator explicitly releases it.

## Frozen selection IDs

The following IDs and values are copied from `B/DECISION.json` and `B/HPO_SELECTION.json`; they are configuration records only.

| Dataset | Family | Selected candidate | Candidate values |
|---|---|---|---|
| MovieLens | D | `D_H05_K10` | half-life ratio 0.5; refresh steps 10 |
| MovieLens | U | `F` | half-life ratio 1.0; reference candidate |
| MovieLens | M | `M_H05_K20` | half-life ratio 0.5; refresh steps 20 |
| MovieLens | S | `S_H20_K20` | half-life ratio 2.0; refresh steps 20 |
| KuaiRand | D | `D_H05_K20` | half-life ratio 0.5; refresh steps 20 |
| KuaiRand | U | `U_FAST` | half-life ratio 0.5 |
| KuaiRand | M | `M_H20_K10` | half-life ratio 2.0; refresh steps 10 |
| KuaiRand | S | `S_H05_K20` | half-life ratio 0.5; refresh steps 20 |

Decision next actions are two `A_FULL` matrices, one per dataset, with seeds 2022–2026. The continuation status is awaiting triggered implementation.

## Hashes and output files

- Selection hash: `fce00de767c5e95b0cb627f53eb5851d43a6c1e64a8c4786db73e57c0ff202a4`
- HPO selection file SHA-256: `58fa76316c0482bd8c8bd1093fe081a58ae04442cfb214a998fbeccc594c614f2`
- New endpoint file SHA-256: `6d9b9cb7b1c188c732018b416b946386170834aa664b19f3ed5180976bc2b935`
- Endpoint CSV: `artifacts/aug_history_20261105/attempt1/B/ENDPOINTS.csv` (600 rows: 420 new and 180 published reference rows)
- B report: `artifacts/aug_history_20261105/attempt1/B/REPORT.md`
- Cost records: `artifacts/aug_history_20261105/attempt1/B/COSTS.csv`, `artifacts/aug_history_20261105/attempt1/IPC_COSTS.jsonl`, `artifacts/aug_history_20261105/attempt1/epoch_commit_ledger.jsonl`
- Other raw report tables: `artifacts/aug_history_20261105/attempt1/B/PAIRED_SEED.csv`, `artifacts/aug_history_20261105/attempt1/B/PAIRED_SUMMARY.csv`
- Runner / protocol records: `PREPARED.json`, `ADMISSIONS.jsonl`, `RETRIES.jsonl`, `RUN_MANIFEST.json`, `B/DECISION.json`, `B/EVALUATED.json`, and `CONTINUATION.json` in the run directory.

## Warnings and runtime errors

- Validation emitted two warnings: PyTorch TF32-setting deprecation and conversion of a requires-grad tensor to a scalar.
- Frozen evaluation emitted a PyTorch warning that a NumPy array was non-writable when converted to a tensor.
- Four worker failures were deliberate SIGTERM recoveries under the parent-authorized RAM procedure; all four resumed from committed epochs. No OOM was recorded.
- A transient progress-report mapping error occurred during monitoring; final counts above were audited by joining every `PREPARED.json` job ID to its epoch files. Final audit: 10/10 jobs × 50/50 epochs, 7,500 candidate-validation records, 500 online records.
- After B evaluation, report generation, and scalar outputs were complete, the orchestrator authorized `python -m experiments.aug_history_20261105.run release --manifest research_plans/26-11-5/experiment_manifest.json --output artifacts/aug_history_20261105/attempt1 --route B`; it exited 0 with `RELEASED`. Guardian PID 37876 is no longer present. A targeted scan of the attempt output found no `.pt`, `.pth`, `.ckpt`, `.safetensors`, or `.onnx` parameter files.
