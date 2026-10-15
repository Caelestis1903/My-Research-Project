# Stage F0 v1.5 marker continuity fixture — implementation report

This report records a test-harness-only correction to the peer race evidence. It does not authorize a qualification E2E, RUN, release, or training. B6 remains OPEN/FAIL.

## Change

The generated worker-0 supervisor wrapper now records its initial F140 snapshot, snapshot digest, and post-peer supervisor-resume marker only during the first `capture_f140_state` call. Later captures return their current state without overwriting those original markers. The child deferral decision remains in its separate `V15_DEFER_READY` marker. The integration assertions now verify that the supervisor resume digest equals the original snapshot digest; the rendezvous helper verifies strict monotonic order through the child deferral marker. Therefore a later validation capture cannot rewrite the original resume timestamp or substitute a post-race snapshot into the initial evidence.

A focused regression extracts the actual embedded `capture_then_signal` function and executes it twice. The first call captures the initial state, waits for a peer marker, and records resume; the test then publishes the child defer marker and calls capture again with a changed state. It proves that the initial snapshot bytes, snapshot metadata, and resume metadata remain byte-identical, that resume and snapshot hashes match, and that the recorded order is strictly `snapshot < peer-before-claim < peer-claim < resume < child-defer`. The embedded-wrapper integrity check also asserts the first-capture guard precedes marker writes and the guard is set only after the resume record.

No scientific runtime/model code or production state was changed.

## Verification

All Python checks used `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`. `py_compile` passed for `experiments/tests/test_stage_F0_successor_v15.py`. Focused tests used the required process-slot wrapper, a unique scheduler root, and unique basetemp:

```text
/home/xzf/Code/phase0-3/experiments/stage_F0_outputs/runs/run_f0_python_task.sh v15-marker-immutability-focused-final /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q --basetemp=/tmp/f0-v15-marker-immutability-20261014-b/basetemp experiments/tests/test_stage_F0_successor_v15.py::test_initial_peer_snapshot_and_resume_are_immutable_across_later_capture experiments/tests/test_stage_F0_successor_v15.py::test_changed_peer_fixture_compiles_and_orders_snapshot_before_peer_and_deferral
```

Raw result: `2 passed in 0.13s`, exit 0. This includes the focused test that executes the extracted function twice; it does not run the qualification E2E. An earlier first run had one failure in the wrapper-integrity test because its order assertion selected the guard's early-return `return state`; that assertion was corrected to inspect the final return after resume. Both raw outputs are preserved.

Final focused stdout: `/tmp/f0-v15-marker-immutability-evidence-20261014/marker-immutability-focused.stdout.txt`, SHA-256 `05969f68db42c0d80b5f203b4c913fe621ff8643af5b714630e061a4653df4b3`.

Evidence manifest: `/tmp/f0-v15-marker-immutability-evidence-20261014/SHA256SUMS.txt`, SHA-256 `5d8da28cf7a62ea0a92d9269bd5655ddf04100c974cf3a0762bed75dc0a1317d`.

No qualification E2E, release, migration, CUDA test, or training process was started.

## Production protection

The outer historical logs remain byte-identical to their frozen hashes:

- `audited_worker_0.jsonl`: `b27ac612b0545f8d962cc02072a778aabee57eb9988559912d13cd4f4e3eb7c2`.
- `audited_worker_1.jsonl`: `1725b85a04446bd3dbf650412c1ec896815b7a1a3924fe3c31e34c653022f960`.

The change only affects the isolated test clone when the peer-continuity fixture runs. It does not clean, rewrite, or authorize the malformed production history; B6 stays OPEN/FAIL/NO-GO.

## File hashes

- `experiments/tests/test_stage_F0_successor_v15.py`: `02d02e34ec811830411278fba3ffc4b09c0d4ec84763ad23afe26f79a7dc4fcb`.
- `experiments/stage_F0_successor_v15.py`: unchanged by this task.
- This report: SHA to be recorded after report write.
