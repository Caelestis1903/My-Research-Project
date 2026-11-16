# EXPERIMENT EXECUTION REPORT — Cycle03 four-case engineering smoke

## 1. Experiment ID
Cycle03 approved four-case engineering smoke.

## 2. Git/worktree state
- Workspace: `/home/xzf/Code/phase0-3_tvg_rm_20261116`
- HEAD: `ed34eb243099dc2693f06ba566fc689c7657f641` (detached HEAD).
- Other concurrent/untracked artifacts were present; this run modified only its smoke outputs and this report.

## 3. Config
- Command:
```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.nqr_temporal_exposure_20261116.runner smoke --devices 0,1 --review /home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/IMPLEMENTATION_REVIEW.json
```
- Seeds: `2022` in all four engineering cases.
- Datasets: MovieLens and KuaiRand; arms: NQR-EARLY10 and NQR-LATE10; opposite arm placements on devices 0 and 1 for each dataset.
- Full batch: 40,960 rows; tail rows: MovieLens 21,927, KuaiRand 40,052; FP32.
- Each case: 4 logical engineering steps, 2 RAM commits, 10 physical train calls; engineering virtual clock; no scientific epoch completed.

## 4. Seeds
2022. No other seeds were part of this four-case smoke.

## 5. Run status
- Command exit code: 0; stdout: `{"status":"PASS","cases":4,"physical_train_calls":40,"test_accessed":false}`.
- Cases passed: 4/4. Test access: false.
- Parent RAM owner remained alive after smoke; it was not started, stopped, or contacted by this runner.
- All four case profile PIDs were absent after the run; no smoke worker process remained.
- Parent supervision acceptance: `ACCEPTED`; all workers dead, retained states `0`, Test access false (`PARENT_SMOKE_SUPERVISION_ACCEPTANCE.json`).

## 6. Raw metrics
No scientific AUC, loss, per-domain, or per-task metrics were emitted by this engineering smoke. The following are execution observations:

| Dataset | Arm | Seed | GPU | Full/tail rows | Logical steps | Physical calls | Peak full-process bytes | Peak host RAM fraction | Seconds | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| movielens | NQR-LATE10 | 2022 | 1 | 40960/21927 | 4 | 10 | 1384120320 | 0.165784 | 3.466 | PASS |
| movielens | NQR-EARLY10 | 2022 | 0 | 40960/21927 | 4 | 10 | 1384120320 | 0.165269 | 2.943 | PASS |
| kuairand | NQR-LATE10 | 2022 | 0 | 40960/40052 | 4 | 10 | 1749024768 | 0.173194 | 5.752 | PASS |
| kuairand | NQR-EARLY10 | 2022 | 1 | 40960/40052 | 4 | 10 | 1749024768 | 0.173289 | 6.814 | PASS |

- Totals: 16 logical steps, 8 engineering commits, 40 physical calls, 0 failed training batches.

## 7. Log locations
Complete worker logs were retained at:

- `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/attempt1/smoke/1d8b48d7bfe726502316b8b627ff2dfb171fa9a74e766f42330df78bbae9f7ff/worker.log` (SHA-256 `1274ca429495efd3e4852684856ec4b83fcea4d93d1da82c8f47009c3f64fa0f`; 15853 bytes).
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/attempt1/smoke/32928b102f2e5cf30e5bdde0c0db36c7b3887a6c4cdaf430fc908b57fa05d6dc/worker.log` (SHA-256 `0ad9f6b2044b1d754c8ce0b64797d1911fbae9cc5926dccda323c3613b597cca`; 15864 bytes).
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/attempt1/smoke/3390db6e8523252657a8d5ba5d8d831dc712bd672845deab65fdfe760f8add10/worker.log` (SHA-256 `3b1e848ba5ef1d58928c48a03811e1bdd103e480e98dee43b199ee8e51ee3ffe`; 15869 bytes).
- `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/attempt1/smoke/eb6ee4622dda384514cd76095e3092d5604eda1026d9fdc1dfdc1fc5d6db3538/worker.log` (SHA-256 `3ab486e91a0979a9af49376ab625293d4cb1e8630323f5c714f5d1f71806a441`; 15864 bytes).

## 8. Runtime errors
None reported in case records or worker logs; no failed training batches.

## 9. Missing runs
None. All four approved cases completed.

## Process/storage records
- Owner receipt: `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/attempt1/STORAGE_OWNER.json`; owner PID 2097722 remained alive at report time.
- Smoke worker processes and profile PIDs were absent after completion.
- No `.pt`, `.pth`, `.ckpt`, or `.safetensors` files were found under the attempt root.
- Case records: `attempt1/smoke/<case-key>/case.json`; job configs: `attempt1/smoke/<case-key>/job.json`; admission observations: `attempt1/smoke_admission.jsonl`.

## Report files
- Markdown: `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/C3_SMOKE_RUNNER_REPORT.md`
- JSON: `/home/xzf/Code/phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_03/C3_SMOKE_RUNNER_REPORT.json`
