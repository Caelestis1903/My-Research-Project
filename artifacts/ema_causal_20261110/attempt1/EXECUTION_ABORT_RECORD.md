# Attempt 1 execution abort record

Status: `ABORTED_ENGINEERING_PERFORMANCE`. This is an execution-status record, not a scientific result.

- Study/config: `EMA_CAUSAL_20261110`, source `ffa3de067fcda5a1293af83a465d741e069493f1`, implementation `575fb1c9bc7f87e900bf56007e3fb21728c92e4e95756c68965313a5818d8b36`, seeds 2022–2026, 20 jobs, 50 epochs/job.
- At 2026-11-10 14:55 +08:00: only ML-C seeds 2022–2026 had completed (50 epochs each, 1 launch each). Committed formal Adam updates: 5,000. KR-C, ML-G, KR-G: not run. No scientific block tables were produced; `blocks=[]`.
- Guardian before shutdown: PID 918875, RSS 2,480,054,272 bytes, active payload 1,407,769,320 bytes. Completed ML-C states were retained in RAM until process termination. No checkpoint/model/optimizer files were written.
- Parent total wall time from launch (2026-11-10 12:46:56) to abort (14:55:56): 2h09m00s. The last ML-C E50 commit was at 12:55:06, so completed training took 8m10s; only subsequent attribution is counted as discarded work.
- First parent attribution wall time: 12:55:06 to TERM at 14:31:09 = 1h36m03s. The same attempt resumed with OMP/MKL/OpenBLAS thread limits set to 1. Second attribution wall time: 14:31:40 to abort at 14:55:56 = 24m16s. Total discarded attribution wall time: 2h00m19s. The five completed training states remained in the guardian during the first parent restart; they were not rerun.
- Wasted formal training cost for this aborted attempt: 5,000 Adam steps (five ML-C seeds, each 1,000 steps). Do not count this partial attempt as satisfying the planned 34,500-step matrix.
- The implementation review identified an engineering performance bottleneck: repeated fixed-pair Floyd sampling dominated attribution runtime. No code or scientific configuration was changed in this attempt. A bounded-memory pair reuse optimization requires implementation and review before a new attempt.
- At the orchestrator's direction, parent PID 979808 and guardian PID 918875 were terminated. There is no standalone guardian shutdown command; `release` was not called because it would write a success `RELEASE_RECEIPT`.
- All logs, manifests, epoch records and attempt files are retained. No completion, release, or scientific-result receipt is asserted.
