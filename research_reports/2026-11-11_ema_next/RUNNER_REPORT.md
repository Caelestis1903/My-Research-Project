# Experiment execution report

Runner: `/root/runner` (`luna-runner`). This records execution facts; scientific interpretation is in FINAL_REPORT.md.

- Experiment: EMA_NEXT_20261111, attempt1.
- Worktree: `/home/xzf/Code/phase0-3_ema_next_20261111`.
- Scientific implementation SHA-256: `20ccc69b2421efd690bb363db405f198b991cd6efe03ccb912298c1ee4b784b6`.
- Datasets: MovieLens and KuaiRand; seeds 2022–2026.
- Task status: 95 COMPLETE, 0 FAILED; no missing triggered runs.
- Test: 160 frozen endpoints, 147 unique forward objects.
- RAM state: RELEASED; released tensor bytes 669716384.
- Runtime attempt directory: `artifacts/ema_next_20261111/attempt1/`.
- Raw epoch/job records, cost tables, curves, resource samples and logs are preserved in the published report and `evidence/` directory.
- Final execution completed without a new runtime error. Earlier controller exceptions and admission corrections are recorded in PROVENANCE.md and scheduler_revisions. A later monitoring alert read a stale traceback; it did not indicate worker/controller failure or restart training.

The runtime index contained 3142 files, including local execution bookkeeping. The published ARTIFACT_INDEX.json separately enumerates only the files delivered with this report.
