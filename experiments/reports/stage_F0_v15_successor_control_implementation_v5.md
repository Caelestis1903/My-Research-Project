# Stage F0 v1.5 successor control — implementation v5 report

**Disposition: IMPLEMENT checkpoint for independent review.** This report records a test-harness coordination correction and archive corrections. It does not authorize RUN, release, or training. B6 remains OPEN/FAIL, and the production hold remains active.

## Changes

In `experiments/tests/test_stage_F0_successor_v15.py`, the real peer qualification fixture now coordinates at the actual supervisor boundary. Worker 0's wrapper first calls and returns from `capture_f140_state`, writes the exact returned snapshot and its digest, then waits in the supervisor before child spawn. The peer runner can therefore make its real canonical/status claim after the captured snapshot and before worker 0's deferral child exists. The child no longer waits in a patched runner-main function or in the resource-deferral callback: it records the selected deferral decision and raises the real `ResourceDeferred` signal directly. The fixture asserts the order `snapshot < peer canonical-claim phase < peer claim < supervisor resume < child deferral decision`, and that the snapshot bytes remain those captured before the peer claim.

The wait in the supervisor wrapper occurs after `capture_f140_state` returns. At the production call site in `TrajectorySupervisor._spawn_one`, snapshot capture is not enclosed by `execution_state_lock` or a canonical/member lock. The capture function only inventories and hashes frozen F140 members. This fixture does not infer lock contention from a wait or from a lock-owner snapshot.

Added a CPU fixture-integrity test that extracts and compiles the embedded child and supervisor scripts and checks that the peer synchronization is in the supervisor after snapshot publication and before `return state`, while the child has no peer wait or runner-main patch.

The synthetic live-peer positive fixture previously used the pytest process as a fake supervisor. The live validator correctly rejected it because real worker identity requires the supervisor to own its PGID (`supervisor_pid == supervisor_pgid`). The test now starts an isolated supervisor subprocess in a new session, asserts its PGID equals its PID, and has it hold the worker lock, write matching worker owner/control records, start a child that holds the canonical and GPU flocks, and append its own `CHILD_STARTED` identity row. The test then checks the real validator accepts the matching live claim and rejects host/identity alterations. This remains a focused schema fixture; it is not represented as the `TrajectorySupervisor.run()` end-to-end qualification.

No runtime module or runner production source was edited for v5.

## Verification

All Python execution used `/home/xzf/miniconda3/envs/phase0-3-env/bin/python` (project Python 3.13). Python process count never exceeded two: one pytest process and, in the live-peer schema test, its isolated supervisor/child process chain. Tests used unique scheduler roots and unique absolute pytest basetemp paths. No CUDA visibility or training was requested; fixtures use temporary roots and sentinel data.

Compile command:

```text
/home/xzf/miniconda3/envs/phase0-3-env/bin/python -m py_compile experiments/tests/test_stage_F0_successor_v15.py experiments/stage_F0_successor_v15.py experiments/stage_F0_outputs/runs/run_search_claim_worker.py
```

Result: exit 0.

The first focused run before correcting the stale supervisor fixture produced `1 failed, 4 passed in 0.14s`; the failing assertion was the synthetic live-peer positive. Diagnosis showed its pytest process did not satisfy the runtime's real supervisor-PGID identity contract. A pass-only subset then produced `4 passed in 0.05s`; it is recorded as an intermediate subset, not a substitute for the failing positive. Two subsequent fixture attempts failed before predicate evaluation: one lacked the subprocess import path (`ModuleNotFoundError`), and one redundantly called `setpgid` after `start_new_session` (`EPERM`). Both outputs are retained.

One initial direct pytest invocation, made before I checked the repository's mandatory slot-launcher hook, exited 75 before collection because the reserved-slot environment was absent. It ran no test bodies. Its output was observed in the tool result but was not separately archived as a raw stdout file; all later pytest runs used `run_f0_python_task.sh` and retained raw stdout.

After the fixture correction, the complete focused group ran with the slot scheduler and a fresh basetemp:

```text
F0_PROCESS_SCHEDULER_ROOT=/tmp/f0-v15-v5-focused-20261014-d/scheduler
/home/xzf/Code/phase0-3/experiments/stage_F0_outputs/runs/run_f0_python_task.sh v15-v5-focused-final
/home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q
  --basetemp=/tmp/f0-v15-v5-focused-20261014-d/basetemp
  experiments/tests/test_stage_F0_successor_v15.py::test_changed_peer_fixture_compiles_and_orders_snapshot_before_peer_and_deferral
  experiments/tests/test_stage_F0_successor_v15.py::test_non75_diagnostic_persists_output_with_missing_markers
  experiments/tests/test_stage_F0_successor_v15.py::test_authenticated_live_peer_claim_is_accepted_during_deferral
  experiments/tests/test_stage_F0_successor_v15.py::test_deferred_race_rejects_forged_peer_completion_without_real_receipt
  experiments/tests/test_stage_F0_successor_v15.py::test_deferred_race_accepts_completed_peer_only_with_real_validated_receipt
```

Raw result: `..... [100%]` and `5 passed in 0.15s` (exit 0). The full raw stdout is archived at `/tmp/f0-v15-v5-implementation-evidence-20261014/focused-final.stdout.txt`. The four earlier focused stdout files are archived beside it. `SHA256SUMS.txt` covers all five stdout files; its SHA-256 is `a8097d1305ad57c66913ef9d1e14b71585bea5e03050559c020a20c8a3d9625f`.

These checks do not close the design's full real `TrajectorySupervisor.run()` peer-continuity qualification. In particular, this v5 turn did not run the two-boundary changed-peer E2E, controlled >60-second PREPARED reentry, or complete fault/restart matrix. Those remain for review and a separately authorized qualification run. No incomplete result is labeled passed.

## v4 report correction

`experiments/reports/stage_F0_v15_successor_control_implementation_v4.md` is corrected in place. The archived process snapshot shows worker-0 child `wchan=hrtimer_nanosleep`, worker-0 supervisor `wchan=0`; the `/proc/locks` row identifies a lock owner at one instant but does not prove a deadlock. The peer did not reach canonical claim, the auth predicate was not invoked, and the reason for the wrapper wait remains undetermined. The aggregate archive manifest contains 172 rows including itself: 171 payload entries match and its self-entry does not. The pre-kill manifest has 38 rows, all matching. The previous v4 report's stronger lock-causality wording was incorrect and has been removed.

## Production protection check

The following current production hashes match the frozen values reported in v4. The v5 tests operated on pytest temporary roots and a unique `/tmp` scheduler root.

| Protected asset | Current SHA-256 |
|---|---|
| Active global hold | `e0a32f323494f30a9cfcfd089e8d99f5b52580633c24dc7fa50ad3100a933d51` |
| Canonical execution gate | `f6e95ea7cbf9ca8d540a955ac17b17b14e6f274608977f1d7e2b5025a242be64` |
| v1.4 release pointer | `4dc29d63a1fdea9bb909a551a8574ad3a0d0c9ce2aa85c394f162a90e62dcd63` |
| Search queue | `43e77fd4931629595485844090eeaffcdf92d008dc6633e12cc20237a13a9b23` |
| Search gate | `958ec6e9649300221b233dc58f3898f48ec6b9b0cfe30f87831344cae372485e` |
| `audited_worker_0.jsonl` | `b27ac612b0545f8d962cc02072a778aabee57eb9988559912d13cd4f4e3eb7c2` |
| `audited_worker_1.jsonl` | `1725b85a04446bd3dbf650412c1ec896815b7a1a3924fe3c31e34c653022f960` |
| v1.5 approval record | `0bdc330cd594b5c8ac519888fc93182047f1302eeee6a43ce2e8cfc8fc165138` |
| v1.5 no-go candidate | `7126c372c2382516d81d20c91c8de71aeec9175397e2fd164b7635b116f64cef` |

No production hold, gate, release pointer/ledger, queue, status/run/checkpoint, worker audit log, or immutable remediation artifact was intentionally written. No production migration or release was executed. No training phase was started. The active hold stays in place; B6 remains OPEN/FAIL because trusted incident-preimage and full pre-edit inventory evidence are absent.

## File hashes

- `experiments/stage_F0_successor_v15.py`: `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2` (not changed in v5).
- `experiments/stage_F0_outputs/runs/run_search_claim_worker.py`: `e44114d1d9f6a12635e4c0d0d42ed55d32b272bf775c6e42d966af839c40ac90` (not changed in v5).
- `experiments/tests/test_stage_F0_successor_v15.py`: `461f30d4dac24bc80d7eb73e0cac4b3abfde5f5044a45be6c8ac83948158edda`.
- Corrected v4 report: `experiments/reports/stage_F0_v15_successor_control_implementation_v4.md`, SHA-256 `3c89908be8dcb70dbb8abe52f81c01cfc45ff105100fccb22b71fb4d0be02d37`.
- This v5 report: SHA to be recorded after report write.
