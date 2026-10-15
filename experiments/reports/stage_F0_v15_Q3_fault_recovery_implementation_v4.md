# v1.5 Q3 fault-recovery qualification harness — implementation v4

Status: **IMPLEMENTATION CHECKPOINT ONLY; independent review required.** This test-only correction addresses the three static selector defects in review v3. It does not authorize qualification RUN, production action, release, or training. Historical B6 remains **OPEN/FAIL** and production/RUN remains **NO-GO**.

## Scope and corrections

Only `experiments/tests/test_stage_F0_successor_v15.py` changed; the v1.5 runtime and protected production files were not edited.

- Directory-fsync selection now uses explicit `(file class, phase, boundary ID)` rows with unique `(file class, phase)` and unique boundary assertions. Forward archive creation and existing-archive recovery have separate identities, so the recovery rows cannot overwrite forward archive labels.
- Recovery file IDs are matched as `recovery:<file-class>:file`; normal forward IDs remain `<file-class>:file`. The stage is explicit: each recovery test first runs its named seed hook in `seed` mode with recovery selectors disabled, asserts that exact seed hook raised and the intended partial artifact state is visible, then switches to `recovery` mode and enables only the selected retry fsync.
- Every syscall-matrix case records its matching boundary, phase, fd kind, file class, and fd path. The case requires exactly one injected match and checks that its regular-file or parent-directory path is the expected transaction path. The eight preimage/PREPARED cases similarly check their exact file-fsync and parent-directory-fsync descriptor sequence and paths.
- The declared matrix is asserted to contain 42 unique PREPARED-or-later callsites (30 forward plus 12 recovery points). The independent preimage list is asserted to contain eight unique cases. These declarations encode 50 distinct intended fault specifications; the syscall matrix itself was not executed in this checkpoint.
- The v3 fixes remain: v1.5 terminal classification requires COMMITTED state and current ledger digest, and both PREPARED ledger fsync interruptions reject a foreign transaction before original-identity recovery.

## Verification

Using `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`:

- `python -m py_compile experiments/tests/test_stage_F0_successor_v15.py` — **PASS**, exit 0.
- `experiments/stage_F0_outputs/runs/run_f0_python_task.sh q3-v4-collect-final /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest --collect-only -q -p no:cacheprovider --basetemp=/tmp/f0-q3-v4-final-collect-20261014 experiments/tests/test_stage_F0_successor_v15.py -k 'preimage_archive_file_and_parent_fsync_faults_resume_by_identity or each_release_syscall_boundary_has_independent_restart_oracle'` — **PASS**, `2/62 tests collected (60 deselected) in 0.12s`.

No fault matrix or E2E was run. The assertions that each intended syscall matches once are present but remain unexecuted. No training, GPU, claim, release, or production migration was started.

## File hashes

- `experiments/tests/test_stage_F0_successor_v15.py`: SHA-256 `a78b67b3d6d0e0eef74ce6c0f7bfdd4a6b7708ed4c813fff9754b57d3d274019`
- `experiments/stage_F0_successor_v15.py` (unchanged): SHA-256 `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`
- `experiments/reports/stage_F0_v15_Q3_fault_recovery_implementation_review_v3.md` (review input): SHA-256 `17fd59e23b524c56964f1ea7575878930019a87ccac9ddbe92eef9d0d7422676`

No production gate, active hold, queue, status, run, checkpoint, ledger, or immutable incident/remediation artifact was written. The B6 incident disposition and absent pre-incident evidence remain unresolved; this harness work neither repairs nor reverses them.
