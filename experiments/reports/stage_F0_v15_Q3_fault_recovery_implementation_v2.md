# F0 v1.5 §10/Q3 fault and recovery test implementation v2

This is a test-harness-only response to Q3 review v1, which returned FAIL with two MAJOR findings. The sole modified source is `experiments/tests/test_stage_F0_successor_v15.py`; runtime, algorithm/scientific files, production pointers, hold, release, queue, status, run, checkpoint, and production ledgers were not changed. B6 remains OPEN/FAIL and production/RUN remains NO-GO.

## Corrections

The new `test_each_release_syscall_boundary_has_independent_restart_oracle` creates one valid isolated PREPARED baseline and a fresh clone per boundary. It identifies the actual fsync descriptor using `/proc/self/fd`, file type and exact path; ledger append boundaries are distinguished by the last complete event, observation appends by their stage, atomic replacement temporary-file fsyncs by destination and payload identity, and terminal durability-fence file fsync separately from rename-parent fsync. The fault hook first calls the real `os.fsync`, then raises at the selected point. Parent-directory cases are separate clones and inject after the real parent fsync.

The 30 PREPARED-or-later cases independently cover:

- Both file fsync and parent-directory fsync for release, pending-gate and final-gate immutable archives.
- File and parent-directory fsync for each of the five post-PREPARED release-ledger events: `RELEASE_ARTIFACT_DURABLE`, `PENDING_GATE_DURABLE`, `RELEASE_POINTER_DURABLE`, `HOLD_RELEASED`, and `COMMITTED`.
- File and parent-directory fsync for both fresh runtime-observation decision stages: `LOCKED_RELEASE_DECISION` and `PRE_HOLD_RELEASE_DECISION`.
- Temporary-file fsync and rename-parent fsync for canonical pending gate, release pointer, released hold, and final canonical gate.
- Canonical terminal-gate durability-fence file fsync and parent-directory fsync.

The existing named `fault_after` matrix remains responsible for the separate post-rename/pre-parent-fsync interruption points and tests both final-rename-survived and simulated rename-not-survived restart branches. The preimage test independently covers gate/hold/release-pointer preimage archive file and parent fsync, plus PREPARED release-ledger append file and parent fsync.

Every PREPARED-or-later syscall case asserts its exact ordered visible release-ledger prefix, byte-exact PREPARED row, source/prelaunch bytes, preimage archive digests, canonical gate phase, hold state, and release/pending/final archive or pointer bytes as applicable. Incomplete gates must fail admission; a complete final gate may pass. A valid alternate nonce must reject before telemetry and preserve the transaction and observation bytes. A later active hold blocks replay while preserving the hold and transaction graph. Restoring only the clone hold allows same-identity recovery, with exactly one instance of each release event and unchanged PREPARED/source/prelaunch identities. A second injected active hold blocks terminal acknowledgement; after restoring the exact released projection, the read-only ACK must preserve all gate/hold/release/pending/pointer/archive/ledger/observation bytes and issue no telemetry query.

For pre-PREPARED preimage archive interruptions there is not yet a durable transaction nonce to compare against. Those cases instead require the exact original gate, hold, release pointer and source/prelaunch bytes, the expected ordered set of immutable preimage files, no optimistic release artifacts, and admission rejection. A later active hold must reject the original request without changing those artifacts; same-identity recovery then creates exactly one PREPARED row and one COMMITTED row. Wrong-nonce rejection is asserted after PREPARED exists, including terminal state.

## Verification and limits

Interpreter: `/home/xzf/miniconda3/envs/phase0-3-env/bin/python` (project Conda environment).

- `python -m py_compile experiments/tests/test_stage_F0_successor_v15.py` — exit 0.
- Required wrapper collection command:
  `bash experiments/stage_F0_outputs/runs/run_f0_python_task.sh v15-q3-v2-collect-20261014-c /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest --collect-only -q experiments/tests/test_stage_F0_successor_v15.py::test_each_release_fault_boundary_uses_independent_prepared_clone experiments/tests/test_stage_F0_successor_v15.py::test_preimage_archive_file_and_parent_fsync_faults_resume_by_identity experiments/tests/test_stage_F0_successor_v15.py::test_each_release_syscall_boundary_has_independent_restart_oracle --basetemp=/tmp/phase0-3-v15-q3-v2-collect-c-20261014`
  Result: `3 tests collected in 0.02s`, exit 0.

No fault/recovery matrix was executed. The new assertions are implemented and collected, but this implementation report does **not** claim they pass; execution of these long isolated historical-source validations is deferred to a separately authorized qualification run and independent review. No worker, claim, training, GPU, release, or production transaction was started.

## Exact hashes

- `experiments/tests/test_stage_F0_successor_v15.py` — SHA-256 `de825ab0e3305775984652d00a9005c11755a7e0722eb5e0189bc37fa9a54a21`.
- Unmodified `experiments/stage_F0_successor_v15.py` — SHA-256 `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`.
- Unmodified `experiments/stage_F0_B6_evidence.py` — SHA-256 `f76433a63a9fafa9332752f837a058c470473405b3596bfbba6e9e599da5b404`.
- Q3 v1 FAIL report remains immutable: `experiments/reports/stage_F0_v15_Q3_fault_recovery_implementation_review_v1.md`, SHA-256 `04570a5ffba1fee67c064598c574175ed5f30e925d766ad11842e382f4c77667`.

This implementation remains at IMPLEMENT. It does not close Q3 or B6 and conveys no authority to RUN or release production state.
