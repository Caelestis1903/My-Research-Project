# v1.5 B3 Controlled >60-Second PREPARED Re-entry — Implementation Report

Status: IMPLEMENTATION CHECKPOINT. This is a synthetic qualification-test change only. It does not authorize RUN, release, or production migration.

## Scope and change

Only `experiments/tests/test_stage_F0_successor_v15.py` was changed. The existing isolated real-validator release/re-entry fixture now records the immutable prelaunch audit and source-snapshot bytes before release, takes a live observation for the exact prepared targets, and injects a deterministic 61.25-second monotonic-clock advance after the fault-injected PREPARED append. This models an interruption past the 60-second freshness limit without sleeping or running a long qualification E2E.

The fixture asserts that the aged observation is stale, the same prepared target index/UUID pairs are retained, and fresh target telemetry is queried on both the resource-defer attempt and the later eligible re-entry. The NO-GO attempt must leave the release ledger byte-identical to its PREPARED-only bytes and preserve the prelaunch audit and source snapshot byte-for-byte. After commit, the PREPARED JSONL row remains byte-identical. The terminal idempotent acknowledgement must preserve the entire committed ledger and observation log and must not increase the telemetry-query count.

A short focused clock test separately verifies that the deterministic clock advances an observation from fresh to stale after 61.25 seconds. No production/runtime module was edited.

## Verification

Project interpreter: `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`.

Syntax check:

```text
/home/xzf/miniconda3/envs/phase0-3-env/bin/python -c "from pathlib import Path; compile(Path('experiments/tests/test_stage_F0_successor_v15.py').read_bytes(), 'experiments/tests/test_stage_F0_successor_v15.py', 'exec'); print('syntax compile: OK')"
syntax compile: OK
```

Focused test command:

```text
F0_PROCESS_SCHEDULER_ROOT=/tmp/f0v15-b3-focused-scheduler-20261014 experiments/stage_F0_outputs/runs/run_f0_python_task.sh b3-controlled-clock /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q -p no:cacheprovider --basetemp=/tmp/f0v15-b3-focused-basetemp-20261014 experiments/tests/test_stage_F0_successor_v15.py::test_controlled_clock_expires_live_observation_after_sixty_seconds
```

Raw result:

```text
.
[100%]
1 passed in 0.09s
```

The modified full real-validator PREPARED/re-entry/terminal-acknowledgement E2E was deliberately not run, as requested. Therefore its newly added end-to-end assertions are implemented but not claimed as executed or passed in this checkpoint. No training, dataset access, GPU test, claim, release, or production-state write was performed.

## Files and hashes

SHA-256 values:

- `experiments/tests/test_stage_F0_successor_v15.py`: `58df59fc80b6500bc40552655e0e18cb75118bf7995db0b836b1223e731084f1`.
- The report's own SHA-256 is supplied alongside the report path in the implementation handoff.

Production gate, hold, release ledger, queue, status, run, checkpoint, and worker-log state were not intentionally accessed for mutation by this task.
