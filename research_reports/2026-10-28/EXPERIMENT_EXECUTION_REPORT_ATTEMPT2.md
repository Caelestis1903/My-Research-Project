# EXPERIMENT EXECUTION REPORT — P1 attempt 2

## Experiment ID and execution state

`BASP-A-P1-attempt2` completed the approved P1 worker matrix and released E evaluation for MovieLens and KuaiRand with seeds 2022 and 2023. All four canonical workers reached `COMPLETE` after the primary-selected global freeze and E evaluation release. The worker processes exited after writing their final metrics.

The workspace directory has no Git metadata. Effective reviewed configuration files are stored per worker at `artifacts/mdmt_basp_v1/attempt2/workers/<dataset>/seed<seed>/EXECUTION_CONFIG.json`.

## Configuration

- Dataset/seed pairs: MovieLens/2022, MovieLens/2023, KuaiRand/2022, KuaiRand/2023.
- Native prefix: 24 epochs; source pilot: 12 passes.
- Adaptation arms: TARGET_MATCH, STATIC_LOCAL, CLUSTER_ONLY, FIXED_READ, RELATION_ONLY, CLUSTER_RELATION; each 20 epochs plus the configured 24 paired extra update steps. No batch or epoch reduction was made.
- Approved read-intervention diagnostic receipt: `artifacts/mdmt_basp_v1/attempt2/INTERVENTION_REVIEW_RECEIPT.json` (PASS WITH FIXES; blocking count 0).
- Initial device assignment: MovieLens/2022, MovieLens/2023, KuaiRand/2023 on `cuda:1`; initial KuaiRand/2022 worker was on `cuda:0` and was operationally rescheduled to `cuda:1` as detailed below.

## Worker status and completion counts

|Dataset|Seed|PID|Device|Status|Native|Source pilot|Adaptation arms|READY time (local)|
|---|---:|---:|---|---|---:|---:|---|---|
|movielens|2022|332312|cuda:1|COMPLETE|24/24|12/12|6 × 20/20|2026-10-28T00:19:05+08:00|
|movielens|2023|333413|cuda:1|COMPLETE|24/24|12/12|6 × 20/20|2026-10-28T00:21:16+08:00|
|kuairand|2022|345976|cuda:1|COMPLETE|24/24|12/12|6 × 20/20|2026-10-28T00:41:35+08:00|
|kuairand|2023|333417|cuda:1|COMPLETE|24/24|12/12|6 × 20/20|2026-10-28T00:28:26+08:00|

All four canonical workers have six methods in their READY manifests. The per-method selected/final epoch and final main-step values are recorded in `artifacts/mdmt_basp_v1/attempt2/ATTEMPT2_RAW_METRICS.json` under `workers`.

## Resource reschedule record

The first KuaiRand/2022 process (PID 333415, `cuda:0`) was stopped by SIGTERM at the primary agent's direction after native 24/24, source 12/12, TARGET_MATCH 20/20, and STATIC_LOCAL 7/20. This is recorded as `RESOURCE_RESCHEDULE`, not as a scientific failure. The partial worker directory and completed partial runs were preserved under `artifacts/mdmt_basp_v1/attempt2/resource_reschedule_attempt1/`; the original log remains `artifacts/mdmt_basp_v1/attempt2/run_logs/kuairand_seed2022_cuda0.log`.

The replacement KuaiRand/2022 worker (PID 345976) used the same seed, root, reviewed configuration, batch, and epoch counts on `cuda:1`, and completed the canonical worker. Before proceeding with its adaptation matrix, direct `numpy.array_equal` checks compared the archived and replacement F `h`, `z`, `phi` caches and all arrays in `Q_NATIVE_RAW_predictions.npz`. Every array matched exactly; each maximum absolute difference was 0.0. The full record is `artifacts/mdmt_basp_v1/attempt2/resource_reschedule_attempt1/PREFIX_PARITY.json`.

## Raw metrics

The machine-readable raw metric extraction is `artifacts/mdmt_basp_v1/attempt2/ATTEMPT2_RAW_METRICS.json` (156 records: 104 V/Q records and 52 E records). It preserves, for native V/Q/E and each method's selected/final V/Q/E exports, the original metric JSON values: per-cell AUC and LogLoss with supports, macro AUC, macro LogLoss, natural LogLoss, and pooled AUC by task. The associated original prediction arrays and CSV exports remain under the worker and run directories.

Read-intervention diagnostics were written for the three reviewed foreign-method branches per worker. Files present:

- `movielens/seed2022/FIXED_READ/FUNCTION_INTERVENTIONS.csv` (862028 bytes)
- `movielens/seed2022/RELATION_ONLY/FUNCTION_INTERVENTIONS.csv` (862375 bytes)
- `movielens/seed2022/CLUSTER_RELATION/FUNCTION_INTERVENTIONS.csv` (864008 bytes)
- `movielens/seed2023/FIXED_READ/FUNCTION_INTERVENTIONS.csv` (862635 bytes)
- `movielens/seed2023/RELATION_ONLY/FUNCTION_INTERVENTIONS.csv` (863704 bytes)
- `movielens/seed2023/CLUSTER_RELATION/FUNCTION_INTERVENTIONS.csv` (864484 bytes)
- `kuairand/seed2022/FIXED_READ/FUNCTION_INTERVENTIONS.csv` (860160 bytes)
- `kuairand/seed2022/RELATION_ONLY/FUNCTION_INTERVENTIONS.csv` (855307 bytes)
- `kuairand/seed2022/CLUSTER_RELATION/FUNCTION_INTERVENTIONS.csv` (853487 bytes)
- `kuairand/seed2023/FIXED_READ/FUNCTION_INTERVENTIONS.csv` (860800 bytes)
- `kuairand/seed2023/RELATION_ONLY/FUNCTION_INTERVENTIONS.csv` (858136 bytes)
- `kuairand/seed2023/CLUSTER_RELATION/FUNCTION_INTERVENTIONS.csv` (857121 bytes)

## Global freeze and E evaluation

`artifacts/mdmt_basp_v1/attempt2/GLOBAL_FREEZE.json` records `FROZEN` for all four canonical workers using `research_reports/2026-10-28/STAGE_DECISION.json`. The runner issued evaluate release commands for all four worker keys after the freeze succeeded. All four workers reached `COMPLETE`; 52 E metrics files were written (4 native plus 48 selected/final adaptation exports). The workers exited after completion.

## Logs and runtime errors

- Per-worker stdout/stderr logs and launch receipt: `artifacts/mdmt_basp_v1/attempt2/run_logs/` and `launch_receipts.jsonl`.
- Resource reschedule metadata and parity evidence: `artifacts/mdmt_basp_v1/attempt2/resource_reschedule_attempt1/`.
- Attempt 2 runtime errors: none recorded; all canonical workers reached COMPLETE.
- Missing P1 or E worker runs: none.
