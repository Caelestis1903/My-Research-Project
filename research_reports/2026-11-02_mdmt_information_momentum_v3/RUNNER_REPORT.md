# EXPERIMENT EXECUTION REPORT — MDMT_INFORMATION_MOMENTUM_V3

## Execution identity and environment

- Reviewed implementation: PASS, source hash `5adc90554898ce9eefe707de2365ef9b13865402d77b360efaee2f255e91db0a`.
- Interpreter: `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`; Python 3.13.5, PyTorch 2.9.0+cu126.
- Devices: two NVIDIA GeForce RTX 3090. S0 ran on `cuda:1`; S1 scheduler uses devices 0 and 1.
- Attempt: `/home/xzf/Code/phase0-3/artifacts/conditional_information_momentum_20261102_v3/attempt1`.

## Stage0

- Status: PASS; worker time 51.8277629119 seconds; peak reserved memory 1,633,681,408 bytes.
- Dataset source/array/split/encoder/scaler manifests were checked. Full-batch smoke arms A0, A1, P11, CV and PCOLL completed; lambda0/A0 comparison and RAM restore checks passed.
- Test metrics accessed: false. Formal trajectories: 0.
- Logs: `artifacts/conditional_information_momentum_20261102_v3/attempt1/logs/stage0.stdout.log`, `stage0.stderr.log`.

## Stage S1

- Planned matrix: 24 trajectories; MovieLens and KuaiRand; A0 and A1; P00 and P11 at lambda 0.05 and 0.2; seeds 2022 and 2024; 50 epochs; batch size 40960.
- Dispatch: `artifacts/conditional_information_momentum_20261102_v3/attempt1/S1_DISPATCH.json`.
- Schedule logs: `artifacts/conditional_information_momentum_20261102_v3/attempt1/logs/S1_schedule.stdout.log`, `S1_schedule.stderr.log`.
- Current progress is recorded in `artifacts/conditional_information_momentum_20261102_v3/attempt1/progress.json`; launch and per-epoch records are in `launch_ledger.jsonl`, `runs/*/attempt*/batch_trace.jsonl` and `epoch_*.json`.
- Status: 24/24 complete; 0 active, 0 pending, 0 failed; scheduler exit code 0; schedule wall time 2380.0266 seconds.
- The final test evaluation completed 144/144 endpoints: 24 trajectories × online/fixed EMA × selected/E10/E50. All endpoint rows have status `COMPLETE`. The interval between the freeze stdout log mtime (23:28:36.932) and final-evaluation stdout log mtime (23:32:23.661 local time) was 227 seconds; it includes command/process startup gaps and is not a measured evaluation-process wall duration. Summed endpoint time was 132.999 seconds (mean 0.924, median 0.714; range 0.348–41.606 seconds). The long endpoint includes first-time MovieLens/KR loading and is retained as recorded.
- S2/S3 were NOT_TRIGGERED per the signed final decision. The RAM guardian shut down successfully after all states were released.
- Final evaluation output: `artifacts/conditional_information_momentum_20261102_v3/attempt1/final_evaluation.json`; per-endpoint rows: `test_endpoints/`; logs: `logs/final_evaluate.stdout.log`, `logs/final_evaluate.stderr.log`, `logs/shutdown.stdout.log`.


## Supplement 2026-11-03, Stage S2 (running)

- Authorization: reviewer PASS supplement entry bound to unchanged source hash `5adc90554898ce9eefe707de2365ef9b13865402d77b360efaee2f255e91db0a`; orchestrator-signed fixed dispatch. Prior Stage0 reused unchanged; no CPU/S0 rerun.
- Matrix: 8 trajectories, P01 and P10 × MovieLens/KuaiRand × seeds 2022/2024; lambda 0.05; 50 epochs; batch 40960.
- Attempt: `artifacts/conditional_information_momentum_20261102_v3/supplement_20261103`; dispatch `S2_DISPATCH.json`.
- Current progress: see attempt `progress.json`; test not accessed. No test evaluation until a separate signed final decision.
- Logs: `S2_schedule.stdout.log`, `S2_schedule.stderr.log`; per-run raw metrics/traces in attempt `runs/`.

- Final S2 status: 8/8 trajectories complete; 0 active, 0 pending, 0 failed; scheduler exit code 0; queue wall time 2320.4988 seconds. Stage0 was reused unchanged. `progress.json` records `test_accessed: false`. All eight result files are present; the RAM guardian remains active holding states while final evaluation awaits a separate orchestrator decision.

- Final S2 supplement evaluation: 48/48 test endpoints completed across 8 trajectories × online/fixed EMA × selected/E10/E50. Evaluation process exit code 0; actual process wall 297.998 seconds, UTC 2026-11-03T08:06:11.858211+00:00 to 2026-11-03T08:11:09.855927+00:00 (record: `artifacts/conditional_information_momentum_20261102_v3/supplement_20261103/evaluate_process_timing.json`). S2 training record retains `test_accessed: false`; the separate `final_test_evaluation` record marks test accessed. RAM guardian shutdown succeeded after states were released.
