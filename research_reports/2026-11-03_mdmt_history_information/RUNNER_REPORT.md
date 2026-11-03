# EXPERIMENT EXECUTION REPORT

## 1. Experiment ID and repository state

- Experiment: `MDMT_HISTORY_INFORMATION_20261103`
- Execution package: `experiments/mdmt_history_information_20261103/`
- Attempt: `artifacts/mdmt_history_information_20261103/attempt1/`
- Review: PASS WITH FIXES; blocking issues empty. Reviewed source hash: `7f16a451f5f0e41d9082d18bf548ad29d1d31f6c350175c5fb9f06ddc8813685`.
- The workspace has no local Git metadata. The task’s requested publication branch is `26-11-03_history_information`; repository publication is not part of this runner record.

## 2. Configuration and seeds

The frozen D0 queue used `research_plans/26-11-03/experiment_queue.yaml`. All new formal trajectories used seed 2022, original natural 80/10/10 split, batch size 40960, 50 epochs, FP32, and no early stopping. Dataset-specific model, Adam, coupled weight decay, fixed EMA, and RNG settings are recorded in `RUN_MANIFEST.json` and each raw run `result.json`.

MovieLens D0 arms: A0f, A1, M, T0, TM, H_REAL, H_MIX, C_MIX, SCALE. KuaiRand D0 arms: A0f, A1, M, T0, H_REAL, H_MIX, C_MIX, SCALE. The KuaiRand default TM identity was marked `COST_BLOCKED` after S0 and was not dispatched. Existing A0/A1 history was imported by identity; the run manifest contains 21 identities, including historical numerical references.

## 3. Run status

- S0: PASS on `cuda:0`; two natural datasets, 40960-row batch, A0f/M/H_MIX/TM, two warm-up and three measured steps per cell; 20 measured/warm-up training steps per dataset. No formal trajectory or test access occurred during S0.
- D0 dispatch: complete, 17 scientific identities accounted for, dispatch failure/block list empty, and 13 new 50-epoch trajectories completed. Four D0 A0/A1 identities were reused from history. No failed runs or infrastructure replays were recorded. Separately, one S0 cost block was recorded for the default KuaiRand TM configuration; that identity was not dispatched.
- The D0 decision file records `NO_GAIN` for both datasets, no selected family, and `next_stage: null` for both. D1, REPLICATE, D2, and CONFIRM were therefore `NOT_TRIGGERED` by the frozen decision procedure. No additional configuration or seed was run.
- Configuration and selection were frozen before test evaluation. Test evaluation completed all 126 endpoints in the frozen endpoint list. Historical endpoints were reused by recorded identity; new endpoints were evaluated after freeze.
- Report export completed: 21 run-manifest rows, 252 combined validation/test endpoint rows, 1,512 six-cell rows, and 14,256 paired rows.

## 4. Raw metrics and logs

Raw results are retained under `artifacts/mdmt_history_information_20261103/attempt1/runs/`; epoch commits and worker launches are recorded in `epoch_commit_ledger.jsonl` and `launch_ledger.jsonl`. Test endpoint files are in `test_endpoints/`. Worker logs are in `launch_logs/`.

The report directory contains `RUN_MANIFEST.json`, `endpoints.csv`, `six_cell.csv`, `validation_epochs.csv`, `paired_seed.csv`, `paired_summary.csv`, `transport_interactions.csv`, and `costs.csv`. They contain AUC and LogLoss at cell, domain/task, macro, traffic, pooled, and worst-cell levels where defined, for online and fixed EMA selected/E10/E50 endpoints. This execution report does not interpret those values.

## 5. Runtime and resource record

- S0 worker time: 215.416 seconds; peak reserved CUDA memory: 1,474,297,856 bytes. Per-dataset/per-arm medians and counts are in `STAGE0_REPORT.json`.
- S0 median measured step seconds: MovieLens A0f 0.585455, M 0.929591, H_MIX 0.822010, TM 1.239829; KuaiRand A0f 0.377959, M 0.740132, H_MIX 0.584381, TM 1.742561.
- The KuaiRand TM median was 4.61 times its KuaiRand A0f median. Only the measured default KuaiRand TM configuration was marked `COST_BLOCKED`; other authorized arms remained eligible.
- Formal queue dispatch was resumed with the same attempt and D0 stage after the original dispatcher was stopped by PID. All seven already launched workers were registered with the guardian before that dispatcher was stopped; those worker processes continued and were not relaunched. The resumed dispatcher used devices `0,1`; the launch ledger records assigned devices and job identities. The CLI default was already `0,1`; the initial all-`cuda:0` allocations reflected the scheduler’s then-current per-device checks, not a missing devices argument.
- The resumed dispatcher reported 3,149.336 seconds of queue wall time. Per-run worker time, training-step GPU time, validation time, and peak memory are retained in result and cost records. External GPU jobs were left running.

## 6. Runtime errors and warnings

- Failed runs: 0.
- Infrastructure replays: 0.
- Missing or incomplete dispatched identities: 0.
- Runtime emitted PyTorch TF32 API deprecation notices and a diagnostic tensor-to-scalar warning. S0, formal training, and evaluation completed.

## 7. Missing runs and output locations

No dispatched D0 identity is missing. Later phases are recorded as `NOT_TRIGGERED` in the D0 decision because no dataset passed the frozen development gate. The test endpoint count matches the frozen list: 126/126. After final evaluation, all 13 nonhistorical guardian states had been released. A post-evaluation status check found 13 registered jobs, all `COMPLETE`, with zero RAM states retained; the idle guardian was then shut down.
