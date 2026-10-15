# v1.5 Q3 fault-recovery qualification harness — implementation v3

Status: **IMPLEMENTATION CHECKPOINT ONLY; independent review required.** This change addresses the three findings in review v2 within the approved synthetic test harness. It does not authorize qualification RUN, production changes, release, or training. Historical B6 remains **OPEN/FAIL** and production/RUN remains **NO-GO**.

## Scope and changes

Only `experiments/tests/test_stage_F0_successor_v15.py` was changed. Runtime and protected production assets were not edited.

The syscall recovery matrix now has 12 additional independently named re-entry cases: existing release, pending archive, and final archive each exercise file and parent-directory fsync; existing pending canonical gate, current release pointer, and released hold each exercise file and parent-directory fsync. Each case first creates the intended partial state with its exact interruption hook, then performs a second call with the fsync fault still active. The test classifies the opened regular file and parent directory by transaction path, confirms the fault boundary was reached, and applies the existing per-case phase, admission, foreign nonce, later-hold, same-identity recovery, immutable PREPARED/source/prelaunch, exactly-once ledger, and terminal acknowledgement byte/no-telemetry assertions.

The terminal oracle now requires the v1.5 schema, `transaction_state == "COMMITTED"`, `launch_authorized is True`, and `committed_ledger_sha256` equal to the current ledger bytes. It asserts the observed terminal classification matches the specific final-gate publication boundary. An older v1.4 authorized source gate is therefore treated as source state, not as a v1.5 terminal graph.

For both PREPARED ledger file-fsync and parent-directory-fsync interruptions, the preimage test now attempts a distinct valid foreign nonce immediately after the injected failure and before original-identity recovery. It snapshots canonical gate, hold, pointer, release/pending/final artifacts, release and observation ledgers, and telemetry-query count, and requires rejection with all bytes and query count unchanged.

## Verification

Using the repository project interpreter `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`:

- `python -m py_compile experiments/tests/test_stage_F0_successor_v15.py` — **PASS**, exit 0.
- `experiments/stage_F0_outputs/runs/run_f0_python_task.sh q3-v3-collect /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest --collect-only -q -p no:cacheprovider --basetemp=/tmp/f0-q3-v3-collect-20261014 experiments/tests/test_stage_F0_successor_v15.py -k 'preimage_archive_file_and_parent_fsync_faults_resume_by_identity or each_release_syscall_boundary_has_independent_restart_oracle'` — **PASS**, `2/62 tests collected (60 deselected) in 0.13s`.

The fault matrix itself was **not executed**. Collection and compilation do not establish runtime behavior or passing recovery assertions. No training, GPU, claim, release, or production migration was started.

## File hashes

- `experiments/tests/test_stage_F0_successor_v15.py`: SHA-256 `163aa615984eb729d7428d53aa88b6973c5327252acec7b615a294abeea796ae`
- `experiments/stage_F0_successor_v15.py` (unchanged runtime): SHA-256 `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`
- `experiments/reports/stage_F0_v15_Q3_fault_recovery_implementation_review_v2.md` (review input): SHA-256 `2ca0e4b05a684b6b7fb23749a235c7cb5e87a74b615b490c59ce808e93cb95aa`

No production gate, active hold, queue, status, run, checkpoint, ledger, or immutable incident/remediation artifact was written by this implementation step. The prior B6 incident disposition and missing pre-incident evidence remain unresolved; this harness work does not repair or reverse them.
