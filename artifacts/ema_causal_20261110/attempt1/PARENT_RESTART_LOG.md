# Scheduler restart log

- 2026-11-10 14:31:09 +08:00: parent PID 918801, elapsed 1:44:12, CPU time 2:51:50, 90 threads; guardian PID 918875.
- Guardian status before stop: payload 1,407,769,320 bytes; ML-C seeds 2022–2026 all COMPLETE, epoch 50, launches 1; blocks empty; no block output.
- Latest epoch commit ledger mtime: 2026-11-10 12:55. Parent-attribution elapsed after last epoch commit is approximately 96 minutes; discarded block computation only, no training state.
- Action authorized by orchestrator after reviewer confirmation: TERM parent PID 918801 only; preserve guardian and all completed worker packets. Resume same attempt with OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1; no training rerun expected.
