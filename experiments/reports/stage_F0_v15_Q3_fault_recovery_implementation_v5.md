# v1.5 Q3 fault-recovery qualification harness — implementation v5

Status: **IMPLEMENTATION CHECKPOINT ONLY; independent review required.** This test-only correction addresses the atomic-replacement temporary-file path finding in review v4. It does not authorize qualification RUN, production action, release, or training. Historical B6 remains **OPEN/FAIL** and production/RUN remains **NO-GO**.

## Scope and correction

Only `experiments/tests/test_stage_F0_successor_v15.py` changed. For the four atomic-replacement file-fsync cases (`pending_temp:file`, `pointer_temp:file`, `hold_temp:file`, and `final_gate_temp:file`), the test now treats the fsynced file descriptor as the temporary file that exists before `os.replace`:

- Its parent must equal the destination parent, and its name must be exactly the destination filename followed by `.tmp.` and a 32-character lowercase hexadecimal UUID.
- The open fd must be a regular file. The test captures the temporary bytes and the destination bytes before the injected exception, then verifies the destination remains byte-identical at its prior transaction phase because rename has not happened yet.
- Payload checks bind pending-gate bytes to the pending graph, pointer bytes to the release payload, released-hold bytes to the PREPARED `released_hold_sha256` with `active == false`, and final-gate bytes to the immutable final archive and COMMITTED state.

All v4 phase-aware recovery selectors, seed/recovery separation, unique boundary IDs and exact fd checks remain in place. Runtime and production files were not edited.

## Verification

Using `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`:

- `python -m py_compile experiments/tests/test_stage_F0_successor_v15.py` — **PASS**, exit 0.
- `experiments/stage_F0_outputs/runs/run_f0_python_task.sh q3-v5-collect /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest --collect-only -q -p no:cacheprovider --basetemp=/tmp/f0-q3-v5-collect-20261014 experiments/tests/test_stage_F0_successor_v15.py -k 'preimage_archive_file_and_parent_fsync_faults_resume_by_identity or each_release_syscall_boundary_has_independent_restart_oracle'` — **PASS**, `2/62 tests collected (60 deselected) in 0.12s`.

The fault matrix was not executed. The temp-path and recovery syscall assertions are authored but unexecuted. No training, GPU, claim, release, or production migration was started.

## File hashes

- `experiments/tests/test_stage_F0_successor_v15.py`: SHA-256 `1e7fc1382de18575510c059849015b7cec804b6fa02fc447a895cdd0cdc06df3`
- `experiments/stage_F0_successor_v15.py` (unchanged): SHA-256 `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`
- `experiments/reports/stage_F0_v15_Q3_fault_recovery_implementation_review_v4.md` (review input): SHA-256 `ef3f3fef076e7257d4b99ad9d8845a1fe59364b937349e6d11e055a642e21afb`

No production gate, active hold, queue, status, run, checkpoint, ledger, or immutable incident/remediation artifact was written. The B6 incident disposition and absent pre-incident evidence remain unresolved; this harness work neither repairs nor reverses them.
