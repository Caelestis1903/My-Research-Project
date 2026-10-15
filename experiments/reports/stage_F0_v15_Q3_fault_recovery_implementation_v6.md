# v1.5 Q3 fault-recovery qualification harness — implementation v6

Status: **IMPLEMENTATION CHECKPOINT ONLY; independent review required.** This test-only correction addresses the eager future-artifact read in review v5. It does not authorize qualification RUN, production action, release, or training. Historical B6 remains **OPEN/FAIL** and production/RUN remains **NO-GO**.

## Scope and correction

Only `experiments/tests/test_stage_F0_successor_v15.py` changed. Atomic-temp payload expectations now use an `if`/`elif` by the selected temp file class. The pending, pointer, and hold cases no longer evaluate or read the final archive; only `final_gate_temp` reads `final_archive`. The existing exact temp pathname/UUID, parent directory, regular-fd, captured payload, and destination pre-rename byte assertions remain intact, along with all earlier recovery/seed and terminal-oracle fixes.

## Verification

Using `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`:

- `python -m py_compile experiments/tests/test_stage_F0_successor_v15.py` — **PASS**, exit 0.
- `experiments/stage_F0_outputs/runs/run_f0_python_task.sh q3-v6-collect /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest --collect-only -q -p no:cacheprovider --basetemp=/tmp/f0-q3-v6-collect-20261014 experiments/tests/test_stage_F0_successor_v15.py -k 'preimage_archive_file_and_parent_fsync_faults_resume_by_identity or each_release_syscall_boundary_has_independent_restart_oracle'` — **PASS**, `2/62 tests collected (60 deselected) in 0.14s`.

The fault matrix was not executed. The syscall and temp-payload assertions remain unexecuted. No training, GPU, claim, release, or production migration was started.

## File hashes

- `experiments/tests/test_stage_F0_successor_v15.py`: SHA-256 `80a6362977f80735bc569793ec1324004f3596cf6bcb43216b33f16c691642f9`
- `experiments/stage_F0_successor_v15.py` (unchanged): SHA-256 `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`
- `experiments/reports/stage_F0_v15_Q3_fault_recovery_implementation_review_v5.md` (review input): SHA-256 `83d74c280e2fde904cbbbd5ca77c121215b7e0a9f2c052339b3fc4ef5115c398`

No production gate, active hold, queue, status, run, checkpoint, ledger, or immutable incident/remediation artifact was written. The B6 incident disposition and absent pre-incident evidence remain unresolved; this harness work neither repairs nor reverses them.
