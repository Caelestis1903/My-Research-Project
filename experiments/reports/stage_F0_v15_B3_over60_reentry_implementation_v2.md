# v1.5 B3 Controlled >60-Second PREPARED Re-entry — Implementation Report v2

**Disposition: IMPLEMENT correction for independent review.** This report records the controlled-aging PREPARED re-entry fixture and the subsequent correction to its PRE_HOLD NO-GO assertions. No production/runtime source was changed.

The isolated real-validator fixture advances its process-local monotonic clock by 61.25 seconds after the fault-injected PREPARED state, then submits the aged same-target observation for re-entry. It asserts fresh telemetry is queried only for the frozen target index/UUID pairs. The immutable prelaunch audit and source snapshot remain byte-identical, and the PREPARED row remains the exact first ledger row through deferral and commit.

At `PRE_HOLD_RELEASE_DECISION`, the release API is allowed to have durably published the release artifact, pending gate, and release pointer. The fixture now asserts precisely those three ledger stages after PREPARED, compares those artifact bytes to their canonical requested payloads, verifies the canonical gate remains pending and unauthorized, checks the active hold bytes remain unchanged, and asserts no `HOLD_RELEASED` or `COMMITTED` event exists. It does not incorrectly require the ledger to remain PREPARED-only.

After the eligible same-target re-entry commits, the fixture snapshots exact bytes for the release artifact, pending gate archive, release pointer, final gate archive, released hold, canonical gate, committed ledger, and observation ledger. Exact-identity terminal acknowledgement must preserve those bytes and must not issue another telemetry query. A separate short test verifies controlled monotonic aging crosses the 60-second freshness limit.

## Verification

Project interpreter: `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`.

Focused clock test previously passed:

```text
F0_PROCESS_SCHEDULER_ROOT=/tmp/f0v15-b3-focused-scheduler-20261014 experiments/stage_F0_outputs/runs/run_f0_python_task.sh b3-controlled-clock /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q -p no:cacheprovider --basetemp=/tmp/f0v15-b3-focused-basetemp-20261014 experiments/tests/test_stage_F0_successor_v15.py::test_controlled_clock_expires_live_observation_after_sixty_seconds
.
[100%]
1 passed in 0.09s
```

After the PRE_HOLD assertion correction, project-environment syntax compilation passed for the B6 tool, B6 tests, and v15 tests:

```text
syntax compile: OK (3 files)
```

The long real-validator v15 release/re-entry E2E was not run, per instruction. Thus the corrected stage/event/artifact assertions are implemented but not claimed as executed or passed. No training, dataset access, GPU test, claim, release, or production-state write was performed.

Test file `experiments/tests/test_stage_F0_successor_v15.py` SHA-256: `f619d97bc34d0a2754267d1869ef248c1a5a6af6fb87c18fd328e3657344ee94`. This report's own SHA-256 is supplied alongside its path.
