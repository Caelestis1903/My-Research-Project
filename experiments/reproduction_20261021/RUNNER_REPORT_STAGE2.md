# Stage2 Experiment Execution Report

1. Experiment ID: `M3OE-supplied-reproduction-20261021-stage2`.

2. Git/worktree: execution cwd `/home/xzf/Code/phase0-3`; reviewed Stage2 training/control source receipts are recorded in `stage2_source_manifest_after_scheduler_amendment.json`. The scheduler-only memory reservation change is separately recorded. Training artifacts remain under `/home_new/xzf_m3oe_reproduction_20261021`.

3. Config: controls `Gtau0` and `Dconst01`; full batch size 40960; epoch cap 50; patience 10; strict validation macro-AUC best checkpoint and restoration. The scheduler used 3 GiB MovieLens and 6 GiB KuaiRand reservation floors; no training configuration was changed.

4. Seeds: 2022, 2023, 2024, and 2025 for both controls and datasets.

5. Run status: 16/16 unique valid formal Stage2 keys; missing=0, extra=0, duplicate records=0, failed completion attempts=0. Stage1 plus Stage2 valid formal keys=48 (expected 48).

6. Raw metrics: each `RUNS_STAGE2.csv` row contains the unmodified completion `test_metrics` object in `test_metrics_json`; native `completion.json` and `test_metrics.csv` remain in the attempt. Per-epoch training, validation, and mechanism records remain in each run directory. This report does not compare methods.

7. Logs/artifacts: each attempt retains stdout/stderr, epoch/mechanism/test CSVs, config, data manifest, split-row IDs, checkpoint and completion receipt. Scheduler queue and launch logs are under OUT. The run CSV records exact argv, selected/stopped epoch, steps, wall time, peak allocated memory, checkpoint hash, parent references, and raw metrics.

8. Runtime/operational errors: one KuaiRand Dconst01 seed2022 attempt was interrupted with SIGTERM under external GPU memory pressure (PID 927107, last recorded epoch 9 / step 441). It remains preserved as invalid `resource_interrupted` with reason `external_memory_pressure`; that key was rerun into a fresh attempt and is valid. Queue exit records are absent for three valid keys: KuaiRand Gtau0 seeds 2022 and 2023, and MovieLens Gtau0 seed 2024. All three have valid completion receipts, matching checkpoint SHA256, selected-epoch/checkpoint agreement, and complete model/optimizer state. After the nonzero interruption, the initial scheduler entered its wait branch for already-running jobs without writing per-job exit events; the MovieLens item may have completed before the scheduler polled it. Original queue logs are preserved unchanged. This is a queue-log completeness limitation, not a failed training run. The restarted queue completed the remaining keys. Invalid resource-interruption receipts: 1; failed algorithm/runtime completion receipts: 0.

9. Missing runs: none.

Raw integrity: Stage2 checkpoint SHA256 receipts were recomputed; model/optimizer/RNG/loader/epoch/step/validation payloads, selected-best epochs and stopping rules were checked. Parent config/data SHA receipts and Stage1 initialization/data matches were checked, along with within-dataset raw/array/split/feature identity and paired backbone hashes. Gtau0 scales and Dconst01 fixed-weight/auxiliary-gradient-state receipts were checked mechanically. The causal training/data/model hash map is identical across the 16 runs; the scheduler source changed only for the documented operational memory reservation. Other whole-source receipt variations are listed in `INTEGRITY_STAGE2.json` and concern noncausal helper files.

## Files

- [RUNS_STAGE2.csv](RUNS_STAGE2.csv) — per-key commands, status, timing/memory, hashes, and raw test metric objects.
- [INTEGRITY_STAGE2.json](INTEGRITY_STAGE2.json) — completion, file, checkpoint, parent, data, initialization, mechanism, queue, and source receipts.
