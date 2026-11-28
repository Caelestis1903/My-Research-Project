# Implementation review for B and the shared A1/B1 runtime

## 1. Verdict — FAIL

The B algorithm passes. Two shared scheduler defects can omit the required CUDA physical-fit check after a recorded OOM. Both A1 and B1 dispatch are blocked until those defects are corrected and independently verified. A1 already passes its independent algorithm review. [B execution design](B_EXECUTION_DESIGN.md) · [B coder report](B_CODER_IMPLEMENTATION_REPORT.md) · [A execution design](A_EXECUTION_DESIGN.md) · [A implementation review](A_IMPLEMENTATION_REVIEW.md)

## 2. Requirement verification

| Requirement | Independent result |
| --- | --- |
| Frozen source, controller arms and natural order | The verified source includes the audited v3 and A21 dependencies and the pinned GA31 callback amendment. The rosters contain 40 A1 and 30 B1 jobs for seeds 2022–2026, with 50 epochs and batches of 40,960. B0 passes source `P00`, B1 and B2 pass `A21`, and B3 observes B1. The source loader seed and context reproduce the seed-2022 first and natural-tail batch order on both datasets. The CPU fixtures matched stock P00 and A21 parameters, gradients, optimizer/diagnostic state, EMA, predictions, sampler records and RNG. |
| B probe, gradients and pair clock | The old-state functional call binds the preceding pre-update S_DENSE values, current private values, disposable copies of pre-forward buffers and restored pre-forward RNG to the same batch and pair draw. The first event with an equal snapshot uses the same raw gradient. B2 adds `.05 × (w/q) × (ĝ−g)` to the shared mixed gradients and routes `ĝ` through the stock A21 base, squared mixed moment and Q update. Private gradients reach source Adam untouched. The tests covered finite differences, probe state conservation, exactly equal observer/base trajectories, the valid zero event, no-event clock/Q conservation and invalid-gradient rejection. |
| Train/Validation and sealed Test | The loader validates both canonical caches against audited Train/Validation array, feature, row, scope and data identities. Its data inputs are the seven read-only cache files. The worker calls the v3 evaluator for each full Validation readout and exposes no Test evaluator path. The entry point froze both rosters from `m3oe/examples` and honored temporary `USER_STOPPED` markers before dispatch with zero completed jobs and zero Test accesses. |
| Current admission and controls | The dispatcher independently evaluates each device using measured memory and power below 85% and host RAM below 90%. It resamples after launches and allocation acknowledgements and records `/proc/meminfo`, `psutil` or `free -b` as the RAM source. Live host telemetry used `/proc/meminfo`. Stop precedence and scope matched both original workspaces and the new worktree. All six registered scopes were clear of current `USER_STOPPED` and `PAUSE_TRAINING` markers. |
| Post-OOM measurement and admission | The per-device fit test uses measured CUDA-usable bytes and full NVML process residency. It supports a supervised isolated replay where a workload has no completed same-device measurement. The OOM latch fails to persist across stages or when a stop is active, as detailed below. |
| RAM, resume and selection | Named RAM packets survived writer exit. Fresh reads validated transport and decoded-state hashes, generation, writer attempt, GPU and all trajectory bindings. Pending-write interruption and corrupt newest packet tests recovered the preceding verified epoch. A reduced production A1.P0 worker stopped after E05, resumed through E11 and exactly matched an uninterrupted worker. It verified both selected readouts and fixed-EMA E05/E10 via independent readback and generated the 12 selected six-cell rows. The selection fixture preserved earliest exact ties and selected-checkpoint LogLoss. |

[Protocol and data binding](../../experiments/three_directions_20261128/protocol.py) · [B adapter](../../experiments/three_directions_20261128/freshness_optimizer.py) · [Source A21](../../experiments/conditional_history_shared_update_20261122/history_controls.py) · [Source optimizer](../../experiments/conditional_information_momentum_20261102_v3/optimizers.py) · [Scheduler](../../experiments/three_directions_20261128/scheduler.py) · [Worker](../../experiments/three_directions_20261128/runner.py) · [RAM journal](../../experiments/three_directions_20261128/ram_state.py)

## 3. Bugs and deviations

1. **MAJOR — A stage transition loses the device OOM latch.** `CAPACITY.json` is resolved under the stage directory. A supervised scheduler fixture recorded a CUDA OOM in A1, then observed B1 launch an ordinary worker on the same device without a measured B1 residency. Persist the latch in the common study artifact root. Continue keying measurements by device and workload. [Scheduler](../../experiments/three_directions_20261128/scheduler.py#L152)
2. **MAJOR — A stop can suppress the CUDA OOM latch for the same stage.** The supervisor records the child exit, then skips CUDA OOM processing when `halt` is set. A fixture put up a scoped stop marker immediately before reaping an OOM child and observed `USER_STOPPED` with no capacity receipt. Persist the actual incident before the halt branch and honor the stop by suppressing requeue. [Scheduler](../../experiments/three_directions_20261128/scheduler.py#L210)

## 4. Corrections made

Review report only. The scheduler corrections belong to the implementation stage.

## 5. Remaining risks and verification scope

All 14 focused CPU tests passed in 70.28 seconds. The reduced production-worker fixture used the actual A1 P0 adapter, M3oE factory, Validation evaluator and RAM journal with a small canonical MovieLens subset and mocked CUDA telemetry. The actual CLI verified both rosters and tested stop-before-dispatch. The failure reproductions used simulated worker CUDA OOM outcome receipts. Live physical CUDA fit after an actual current-study incident and full-schedule timing remain runtime observations.

## 6. Run safety

Scientific trajectories started during review total zero. Test stayed sealed. Refresh the scoped stop checks before dispatch. Once both shared defects pass independent verification, the completed A algorithm review and this B numerical review support A1 and B1 dispatch under foreground supervision.
