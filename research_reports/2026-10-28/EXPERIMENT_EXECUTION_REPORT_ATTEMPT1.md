# EXPERIMENT EXECUTION REPORT — P1 attempt 1

## Experiment ID and state

`BASP-A-P1-attempt1` dispatched the four development workers for MovieLens and KuaiRand with seeds 2022 and 2023. The run stopped after a shared implementation error in the first worker. No worker completed the six-arm P1 matrix, and no worker remains running.

The repository directory has no Git metadata. Execution used the reviewed local implementation in `experiments/mdmt_basp_v1/`; the reviewed base and implementation reference are recorded in each worker configuration.

## Configuration

- Dataset and seed pairs: MovieLens/2022, MovieLens/2023, KuaiRand/2022, KuaiRand/2023.
- Device assignment: MovieLens/2022 and MovieLens/2023 on `cuda:1`; KuaiRand/2022 and KuaiRand/2023 on `cuda:0` and `cuda:1`, respectively.
- P1 arms, in order: TARGET_MATCH, STATIC_LOCAL, CLUSTER_ONLY, FIXED_READ, RELATION_ONLY, CLUSTER_RELATION.
- Each complete worker uses the reviewed 24-epoch native prefix, 12 source-pilot passes, and six 20-epoch arms with matched updates. No batch or epoch settings were reduced.
- Effective configurations: `artifacts/mdmt_basp_v1/workers/<dataset>/seed<seed>/EXECUTION_CONFIG.json`.

## Worker status and raw completion counts

|Dataset|Seed|PID|Device|Terminal state|Native epochs|Source passes|Completed adaptation epochs|Elapsed seconds|
|---|---:|---:|---|---|---:|---:|---|---:|
|MovieLens|2022|327800|cuda:1|FAILED — TypeError|24/24|12/12|TARGET_MATCH 20/20; STATIC_LOCAL 3/20|66.96|
|MovieLens|2023|328528|cuda:1|Stopped after shared error|23/24|0/12|None|58.63|
|KuaiRand|2022|328529|cuda:0|Stopped after shared error|3/24|0/12|None|58.60|
|KuaiRand|2023|328808|cuda:1|Stopped after shared error|2/24|0/12|None|41.58|

The first worker failed in `trainer.py` structure-window diagnostics when `epoch` was supplied twice to `dict()`. The remaining workers were sent SIGTERM after that shared-code failure. Their resident RAM states ended with their processes. No checkpoint was written. The full exception and partial logs are preserved.

## Partial raw metrics

MovieLens/2022 completed NATIVE_RAW and TARGET_MATCH selected/final exports before the failure. Exact per-domain/per-task AUC and LogLoss, supports, task-pooled AUC, logits and labels are in the corresponding `*_CELL_METRICS.csv`, `*_metrics.json`, and `*_predictions.npz` files. The run also wrote the corresponding V and Q NATIVE_RAW exports.

|Dataset/seed|Method/state|Role|Macro AUC|Macro LogLoss|Natural LogLoss|Pooled task AUC|
|---|---|---|---:|---:|---:|---|
|MovieLens/2022|NATIVE_RAW|V|0.746417460|0.534050737|0.534960749|0.746450226, 0.744428785|
|MovieLens/2022|NATIVE_RAW|Q|0.807398034|0.476800376|0.475711645|0.800579464, 0.813753848|
|MovieLens/2022|TARGET_MATCH selected, epoch 15|V|0.748027448|0.530448137|0.531453077|0.748739842, 0.745883858|
|MovieLens/2022|TARGET_MATCH final, epoch 20|V|0.747787207|0.531145499|0.532180385|0.747532383, 0.745325370|
|MovieLens/2022|TARGET_MATCH selected, epoch 15|Q|0.808705414|0.474409536|0.473566097|0.802451480, 0.815145201|
|MovieLens/2022|TARGET_MATCH final, epoch 20|Q|0.808554477|0.475012457|0.474177238|0.802608338, 0.814278685|

These are partial-run measurements and do not constitute a completed P1 comparison. The source training losses, per-epoch validation values, and completed adapter epochs are recorded in each worker log. Other workers stopped before producing evaluation exports.

## Logs and receipts

- Launch commands, PIDs, and per-worker logs: `artifacts/mdmt_basp_v1/run_logs/`.
- Machine-readable status and raw completion counts: `artifacts/mdmt_basp_v1/run_logs/P1_attempt1_RUN_STATUS.json` and each worker's `RUN_STATUS.json`.
- Failed worker traceback: `artifacts/mdmt_basp_v1/workers/movielens/seed2022/FAILED.json`.
- Partial model outputs and epoch records: `artifacts/mdmt_basp_v1/runs/movielens/TARGET_MATCH/seed2022/`.

No P1 worker completed; all remaining P1 runs are missing from this attempt. Retry requires the diagnosed implementation correction and the required review before a new attempt root is launched.
