# EXPERIMENT EXECUTION REPORT

## 1. Experiment ID

`GAUSSIAN_DOMAIN_COVERAGE_20261104` — attempt `attempt1`.

## 2. Git and worktree state

- Branch: `research/26-11-04-gaussian-domain-coverage`
- Initial base HEAD: `9e682f3e3f608c7d63c2e9306b8657b16d822541`.
- Source publication commit: `de7947e01c49aaecbd9d438c5ef662de46cca8f9` (same frozen implementation; source hash below did not change during execution).
- Frozen implementation source hash: `c8c76da2cb644559ea1cdf69d56e75fef00ae1fb70e214529708a023884b096d`
- Worktree contained the experiment artifacts and generated report exports during execution.

## 3. Config

- Dataset: KuaiRand; protocol `FULLTRAIN_NATURAL80_10_10`.
- Arms: `G-D0` (target domain 0) and `G-D02` (target domains 0 and 2).
- Seeds: 2022, 2023, 2024, 2025, 2026.
- 50 epochs per job; batch size 40,960; float32; Adam settings and Gaussian parameters are frozen in `artifacts/gaussian_domain_coverage_20261104/attempt1/CONFIGURATION_FROZEN.json`.
- Checkpoint and optimizer state storage: RAM only. Early stopping: disabled.
- GPU smoke: both arms passed one original-batch training step on CUDA device 0, with one optimizer step and one EMA update per arm.

## 4. Seeds

Both arms completed all five seeds (10 trajectories total). Each trajectory committed 50 epochs and 2,450 optimizer steps.

## 5. Run status

- Final status: `FINAL_COMPLETE`.
- Completed jobs: 10/10.
- Committed epochs: 500/500.
- Logical optimizer updates: 24,500; EMA updates: 24,500; backward calls: 24,500.
- Epoch commit ledger: 500 records; every job has 50 commits.
- Queue final state: all 10 jobs `DONE`; aggregate `failure_count` is 0.
- Coordinator wall time from first dispatch to completion: 3,761.7 seconds.
- Coordinator returned exit code 0. Guardian shutdown record is `SHUTDOWN`; guardian PID 1598527 was absent after shutdown.

## 6. Raw metrics and endpoint exports

- Endpoint rows: 270 total; 150 reused historical endpoints and 120 endpoints from the new trajectories.
- New endpoints: 60 validation and 60 test rows, with 30 rows per arm in each split.
- Total split rows: 135 validation and 135 test.
- Raw AUC, log-loss, per-cell/domain/task values, counts, model/checkpoint, epoch, and provenance fields are in `ENDPOINTS.csv`.
- Raw validation-curve records are in `VALIDATION_CURVES.csv`; cost records are in `COSTS.csv`.
- Summary and paired CSVs are retained as mechanical exports; no scientific interpretation is included here.

## 7. Runtime costs

Totals across the 10 new trajectories, from `COSTS.csv`:

- Training time: 12,935.993 seconds.
- Validation time: 1,025.700 seconds.
- Worker time: 13,961.696 seconds.
- Physical worker time across 500 epoch launches: 16,529.491 seconds.
- Optimizer steps / EMA updates / backward calls: 24,500 each.
- Forward calls: 61,250.
- Validation rows processed: 250,767,000.
- Original rows processed: 1,003,066,000.
- Label exposures: 2,006,132,000.
- Extra views: 108,754,000 (D0: 86,391,000; D2: 22,363,000).
- GPU dispatch records: device 0: 306; device 1: 207.

## 8. Logs and artifacts

- Attempt artifacts, full epoch ledger, queue state, events, smoke output, guardian records, and per-job logs: `artifacts/gaussian_domain_coverage_20261104/attempt1/`.
- Per-job full logs: `artifacts/gaussian_domain_coverage_20261104/attempt1/runs/<job_id>/worker.log`.
- Endpoint and cost exports: `research_reports/2026-11-04_gaussian_domain_coverage/ENDPOINTS.csv` and `COSTS.csv`.

## 9. Runtime errors and recovery

- No unrecovered job failures; final queue `failure_count` total is 0.
- Logs contain 12 worker admission retries reporting `WAIT_RESOURCE: original configuration does not fit` and one retry reporting `WAIT_RESOURCE: external owner changed`. These were retried by the coordinator; the logs do not distinguish which sub-condition of the generic resource gate caused each resource rejection.
- There were 513 dispatch records for 500 committed epoch launches, consistent with 13 retry launches.
- Logs also preserve PyTorch TF32 deprecation and read-only NumPy array warnings.

## 10. Missing runs

None. All 10 configured trajectories and all 270 endpoint rows are present.
