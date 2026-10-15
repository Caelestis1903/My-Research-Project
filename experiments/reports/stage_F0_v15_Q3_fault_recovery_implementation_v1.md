# F0 v1.5 §10/Q3 test-harness implementation v1

This change adds qualification coverage only in `experiments/tests/test_stage_F0_successor_v15.py`. It does not change the v1.5 runtime, scientific code, production state, or B6 evidence tool. The tests use isolated temporary clone roots when executed. No production release, gate, hold, queue, status, run, checkpoint, ledger, or log was written by this implementation task.

## Coverage added

The independent fault-boundary matrix now checks, at every existing release interruption point:

- A distinct, correctly nonce-derived transaction identity is rejected without changing the transaction ledger, canonical gate, hold, release artifact, pending gate, release pointer, or final archive.
- A later active hold blocks recovery and terminal acknowledgement without changing the active hold or transaction graph.
- After restoring only the isolated clone's matching released hold, same-identity terminal acknowledgement reports `COMMITTED_IDEMPOTENT`, preserves exact bytes for the gate, hold, release artifact, pending gate, release pointer, final archive, release ledger, and observation ledger, and does not query telemetry.

A separate isolated-clone test distinguishes failures after actual file fsync and parent-directory fsync for each of the three immutable source preimage archives (canonical gate, incident hold, and historical release pointer), plus the PREPARED ledger append. Each case verifies the visible archive/ledger state at interruption and resumes the exact same transaction identity to one COMMITTED event. Existing fault cases retain both final-gate rename-visible recovery and the alternate rename-not-survived restart path.

## Verification

Interpreter: `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`.

- `python -m py_compile experiments/tests/test_stage_F0_successor_v15.py` — passed.
- Required wrapper collection command:
  `bash experiments/stage_F0_outputs/runs/run_f0_python_task.sh v15-q3-collect-20261014 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest --collect-only -q experiments/tests/test_stage_F0_successor_v15.py::test_each_release_fault_boundary_uses_independent_prepared_clone experiments/tests/test_stage_F0_successor_v15.py::test_preimage_archive_file_and_parent_fsync_faults_resume_by_identity --basetemp=/tmp/phase0-3-v15-q3-collect-20261014`
  Result: `2 tests collected in 0.25s`, exit 0.
- A direct pytest collection attempt without the required wrapper exited 75 because the repository conftest requires a prelaunch process-slot identity. It did not execute tests; the successful wrapper collection above is authoritative for collection.

The fault matrices themselves were **not executed** in this task. Their graph construction and restart paths perform substantial isolated historical-source validation, and the authorized scope was limited to syntax/basic checks without a long qualification run. Therefore the newly authored syscall/recovery assertions are implemented and collected, but remain unverified by execution. This report does not claim those cases pass.

No training, GPU test, worker/claim process, release transaction, or production mutation was started.

## Exact files and hashes

- Modified test file `experiments/tests/test_stage_F0_successor_v15.py` — SHA-256 `95625db76bdf69ad3b351f100fd952447a6ed2834444e6de9b4499aa62a126f2`.
- Unmodified v1.5 runtime `experiments/stage_F0_successor_v15.py` — SHA-256 `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`.
- Unmodified B6 evidence tool `experiments/stage_F0_B6_evidence.py` — SHA-256 `f76433a63a9fafa9332752f837a058c470473405b3596bfbba6e9e599da5b404`.

This is a test implementation handoff only. It does not close qualification review, authorize RUN, or change the active production hold / NO-GO state.
