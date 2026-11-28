# Coder implementation report — shared runtime fixes

Implemented both blocking findings from the runtime review under §7 of the approved B execution design. [Runtime review](B_RUNTIME_IMPLEMENTATION_REVIEW.md#3-bugs-and-deviations) · [Execution design](B_EXECUTION_DESIGN.md#7-implementation-requirements)

## Changes

- A1 and B1 now read and atomically update the same study-root `CAPACITY.json` under the existing `SUPERVISION.lock`. The receipt still uses GPU UUIDs and workload keys. A new workload on a device with an OOM latch enters isolated measurement if it lacks a completed residency receipt. [Scheduler](../../experiments/three_directions_20261128/scheduler.py#L152)
- When a worker exits with `CUDA_OOM`, the supervisor persists the device latch, raises the workload’s required epoch to the maximum recorded failure epoch, and invalidates its prior residency measurement before checking for a halt. An active stop suppresses requeue. Ordinary OOM exits still requeue for diagnosis when supervision continues, and isolated diagnostic OOM exits still trigger a technical pause. [Exit handling](../../experiments/three_directions_20261128/scheduler.py#L200)

## Checks

- Both new regression cases failed on the prior implementation. B1 launched an ordinary worker before its isolated measurement, and a scoped stop left the study capacity receipt missing. With the fixes, B1 measures in isolation first, and a stopped A1 invocation retains its failed epoch for the next supervisor invocation. [Regression tests](../../experiments/three_directions_20261128/tests/test_b_runtime.py#L128)
- `conda run --no-capture-output -n phase0-3-env python -m pytest -p no:cacheprovider -q experiments/three_directions_20261128/tests/test_b_runtime.py experiments/three_directions_20261128/tests/test_b_ram_state.py` passed **8 tests** in 3.69 seconds. PyTorch emitted one warning about its deprecated TF32 settings in the existing RAM test. Both changed Python files compiled in the project environment.

## Review handoff

Both original workspaces and the current worktree had clear registered stop controls at the pre-edit check. The checked study roots contained zero capacity receipts to migrate, and the current worktree's study root contained zero frozen stage rosters. The normal roster freeze will capture the updated scheduler hash.

The regression tests simulate worker outcomes and CUDA telemetry. Live CUDA fit remains for supervised RUN if an actual incident occurs. Formal trajectories started in this implementation pass total zero. Test stayed sealed.
