# Stage F0 v1.5 marker continuity fixture — implementation report v2

**Disposition: IMPLEMENT checkpoint for independent review.** This follow-up is limited to the focused test harness. It does not authorize qualification RUN, release, or training. B6 remains OPEN/FAIL; the production hold and historical evidence are unchanged.

## Correction

The focused immutability regression now reproduces the reachable `Path.write_text` open/truncate-before-write window. It patches only the test process's `Path.open` method for the target snapshot marker, opens the actual destination, signals that the open has occurred, and pauses the writer before returning the handle. The test confirms that the marker exists with size zero, then releases the writer.

The reader uses bounded polling and accepts the row only after it parses as a complete JSON object with the expected phase, a finite monotonic timestamp, and a snapshot SHA-256 matching the snapshot bytes. It retries ordinary read/parse/incomplete-content states until the two-second deadline. This keeps the test compatible with the current producer's ordinary `Path.write_text` publication.

All work after the capture thread starts is protected by `finally`: it releases the paused writer, publishes the peer-ready marker if needed to release the embedded capture's rendezvous wait, joins the capture thread, and fails if the thread remains alive. Thus an assertion or malformed intermediate row cannot leave a background thread waiting for the wrapper's 600-second timeout.

No runtime/validator, production artifact, historical log, gate, hold, queue, claim, or release state was changed. This fixture test does not remediate any malformed production history; B6 remains OPEN/FAIL/NO-GO.

## Verification

All Python checks used `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`.

Syntax check:

```text
/home/xzf/miniconda3/envs/phase0-3-env/bin/python -m py_compile experiments/tests/test_stage_F0_successor_v15.py
```

Exit status: 0.

Focused regression command (the repository process-slot wrapper, unique scheduler slot name, and unique absolute pytest basetemp):

```text
experiments/stage_F0_outputs/runs/run_f0_python_task.sh v15-partial-marker-focused-20261014 /home/xzf/miniconda3/envs/phase0-3-env/bin/python -m pytest -q --basetemp=/tmp/f0-v15-partial-marker-fix-20261014/basetemp experiments/tests/test_stage_F0_successor_v15.py::test_initial_peer_snapshot_and_resume_are_immutable_across_later_capture experiments/tests/test_stage_F0_successor_v15.py::test_changed_peer_fixture_compiles_and_orders_snapshot_before_peer_and_deferral
```

Raw pytest output:

```text
..                                                                       [100%]
2 passed in 0.13s
```

Exit status: 0. The raw stdout is `/tmp/f0-v15-partial-marker-fix-20261014/archive/focused.stdout.txt`, SHA-256 `05969f68db42c0d80b5f203b4c913fe621ff8643af5b714630e061a4653df4b3`.

No long qualification E2E, CUDA test, release, migration, or training process was started. The test used one pytest Python process and one in-process helper thread.

## File hashes

- `experiments/tests/test_stage_F0_successor_v15.py`: SHA-256 `49af0922b90493264a899feeb2a6256ba213b2a1e8a69e0c6ebb6e46ce5275c5`.
- `experiments/stage_F0_successor_v15.py`: unchanged, SHA-256 `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`.
- This report's SHA-256 is recorded in the external evidence manifest alongside the raw output.
