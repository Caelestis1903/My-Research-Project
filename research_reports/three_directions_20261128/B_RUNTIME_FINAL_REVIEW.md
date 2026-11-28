# Final implementation review for the shared A1/B1 runtime

## 1. Verdict — PASS WITH FIXES

**A1 and B1 pass the implementation gate for supervised RUN.** Both major scheduler defects from the first review are corrected and independently verified. Refresh the registered stop controls before dispatch. [B execution design](B_EXECUTION_DESIGN.md#7-implementation-requirements) · [B coder report](B_CODER_IMPLEMENTATION_REPORT.md) · [First runtime review](B_RUNTIME_IMPLEMENTATION_REVIEW.md#3-bugs-and-deviations) · [Runtime fix report](B_RUNTIME_FIX_REPORT.md)

## 2. Requirement verification

| Requirement | Independent finding |
| --- | --- |
| Study-wide post-OOM capacity | The scheduler uses the study-root `CAPACITY.json` under the same `SUPERVISION.lock` as dispatch. The OOM latch is keyed by GPU UUID. Required epochs and completed residency measurements are keyed by workload within each device. The cross-stage fixture verifies that after the A1 OOM, B1 measures its own workload in isolation on the affected device before ordinary dispatch. |
| OOM receipt and scoped stop | On `CUDA_OOM`, the scheduler records the device latch, the maximum required epoch and the invalidation of the workload’s prior measurement before the halt branch. With `USER_STOPPED` active, the fixture returns the scoped stop and launches only the original worker. When the fixture clears its marker, the next A1 invocation diagnoses through the saved failure epoch. |
| Current telemetry and physical fit | The focused checks verify independent per-card decisions, strict GPU memory and power thresholds below 85%, a strict host RAM threshold below 90%, RAM-monitor fallback, and fresh samples after allocation acknowledgements. After a recorded OOM, ordinary admission also requires current occupancy plus completed workload residency to fit within measured CUDA-usable capacity. |
| RAM state and selection | The focused cross-process checks passed for RAM readback, interrupted-write recovery, verified epoch fallback, exact continuation and selected-checkpoint retention. The prior review also established resume equivalence and selected readback through a reduced A1 P0 worker. |
| Algorithms, gradients and baselines | The two fixes concern scheduler state and its regression fixture. The independent A and B numerical findings apply, including A1 matched exposure and native Adam parity, B P00/A21 parity, and B2 shared-gradient routing with the stock private-gradient path. |

[Scheduler](../../experiments/three_directions_20261128/scheduler.py#L121) · [Regression fixtures](../../experiments/three_directions_20261128/tests/test_b_runtime.py#L128) · [A implementation review](A_IMPLEMENTATION_REVIEW.md) · [B numerical review](B_RUNTIME_IMPLEMENTATION_REVIEW.md#2-requirement-verification)

In `phase0-3-env`, pytest passed all **8** cases from `test_b_runtime.py` and `test_b_ram_state.py` in **3.68 s**. The only warning was the inherited PyTorch TF32 API deprecation. The ten registered `USER_STOPPED` and `PAUSE_TRAINING` marker paths across the two original workspaces and the study worktree were clear. The study artifact root had no stage-local capacity receipt or frozen roster.

## 3. Bugs and deviations

- **MAJOR, resolved.** Stage changes could discard the device OOM latch. A1 and B1 now load the same capacity receipt.
- **MAJOR, resolved.** A scoped stop could suppress OOM recording. The supervisor now persists the incident before handling the halt and suppresses retry while stopped.

The corrected paths passed review with no further blocking issues.

## 4. Corrections made

Review report only. The coder’s scheduler fixes passed independent review as submitted.

## 5. Remaining risks

The OOM regressions use simulated worker outcomes and CUDA telemetry. Full-schedule timing and physical-fit measurements following an actual current-study OOM remain observations for the supervised runner.

## 6. Run safety

**A1 and B1 are cleared for supervised RUN** with their frozen rosters, current resource gates and verified RAM state handling. Review started zero formal trajectories. Test remains sealed.
