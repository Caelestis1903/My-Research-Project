# CODER IMPLEMENTATION REPORT — B and shared A1/B1 execution

**Implemented and ready for independent review.** The worktree is `/home/xzf/Code/phase0-3_three_directions_20261128` at audited main `641b5b8c4242c1ba71f7e4d348a41ec7c630a7fc`. B1 freezes 30 trajectories and A1 freezes 40. Scientific trajectories started during this implementation were zero. Test remains sealed. [B design](B_EXECUTION_DESIGN.md) · [A design](A_EXECUTION_DESIGN.md)

## Changes

| Source | Implementation |
| --- | --- |
| `experiments/three_directions_20261128/__init__.py` and `protocol.py` | Read-only Train and Validation cache loader, roster and per-job algorithm/config identities. The scoped data digest matched both S0 identities. Source verification uses the audited main files, the original workspace's SHA-pinned v3 implementation manifest, and the A coder's pinned GA31 callback amendment. The parameter manifest hash uses the original S0 sorted JSON with default separators. |
| `experiments/three_directions_20261128/freshness_optimizer.py` | B0 runs stock P00 mixed Adam with read-only trace capture. B1 runs stock A21 and measures B3 on its trajectory. B2 applies the FP32 correction to the stock mixed shared gradient and passes the corrected shared gradient to A21. The strict functional probe reuses the current private values, the previous pre-update shared snapshot, the current draw, and copies of pre-forward buffers and RNG. The adapter promotes its snapshot after the successful step and records separate observer costs. |
| `experiments/three_directions_20261128/runner.py` | Foreground worker entry, exact batch order with natural tails, full Validation through the v3 evaluator, independent earliest-tie online and fixed-EMA selection, and raw epoch/endpoint/six-cell/cost/readback outputs. A1 uses the A coder's native Adam and step adapter through the same factory. Completion for A1.B0/P0 requires verified compact `fixed_ema-E05`, `fixed_ema-E10`, and `fixed_ema-selected` states with full identities. |
| `experiments/three_directions_20261128/scheduler.py` | Scoped stop precedence, per-card current GPU utilization and host RAM admission, foreground subprocess supervision, allocation acknowledgements, and device/workload CUDA capacity after an actual OOM. The worker captures NVML residency around setup, the first batch, the first distinct-state probe and Validation, and the supervisor samples fresh telemetry after each acknowledgement. An isolated diagnostic replays the original schedule through the failed epoch when the card lacks a completed residency measurement. |
| `experiments/three_directions_20261128/ram_state.py` | Python 3.13 named POSIX RAM packet transport, capacity check and reservation, per-job writer/journal locks, pending packet SHA/digest/identity verification, atomic promotion, previous-epoch fallback, and compact readback. Both selected readouts are verified before epoch packets are released. A later consumer can open a compact key with the expected readout, epoch and model hash through `Journal.read_model`. |
| `experiments/three_directions_20261128/tests/test_b_freshness.py`, `test_b_runtime.py`, and `test_b_ram_state.py` | Direction B numeric, source parity, scheduling, and cross-process RAM checks. |

The B adapter adds logging hooks around the stock forward to obtain the actual pair margin. B0 has no source raw-PAIR VJP and reports its zero-gradient field as null on observed PAIR events. A2, the B confirmation dose, C0 execution, and Test access remain subject to their separate stage gates. The runner emits raw observations for the orchestrator's paired statistical calculations.

## Checks

- **14 passed** in 69.27 seconds. P00 and A21 matched their wrapped trajectories exactly, including RNG, gradients, state and output on the real seed-2022 first and natural tail batches for both datasets. The real fixture uses the canonical tail sampler step with the first-batch pre-update snapshot as a probe input. Synthetic checks covered independent old/current gradients, a centered directional finite difference, state conservation and comparator formulas, zero/no-event clocks, missing/nonfinite old gradients, and exact B2 continuation across an epoch boundary.
- Runtime checks covered the 40- and 30-job rosters, exact S0 source/data/parameter manifest scope, independent GPU thresholds, host RAM fallback, re-sampling on allocation acknowledgement, post-OOM physical-fit fixtures and stop scopes.
- RAM checks used an exiting child writer and a new reader process. They verified pending write interruption, corruption fallback to the preceding epoch, exact reduced-model resume, earliest exact ties with same-checkpoint LogLoss, selected readback, and cleanup of the fixture's named segments.
- The actual module entry point from `m3oe/examples` froze both rosters in a temporary artifact directory. A second B1 invocation verified the frozen roster. No scoped stop marker was present and the host monitor used `/proc/meminfo`.
- PyTorch emitted one inherited deprecation warning for the frozen TF32 configuration API.

```bash
cd /home/xzf/Code/phase0-3_three_directions_20261128
project_python=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
"$project_python" -m compileall -q experiments/three_directions_20261128
"$project_python" -m pytest -q \
  experiments/three_directions_20261128/tests/test_b_freshness.py \
  experiments/three_directions_20261128/tests/test_b_runtime.py \
  experiments/three_directions_20261128/tests/test_b_ram_state.py
git diff --check
```

## Reviewer handoff

The remaining runtime verification is live CUDA admission, full trajectory wall time and actual-incident post-OOM calibration. The implementation checks used canonical CPU B batches, mocked dispatch telemetry and a reduced cross-process RAM fixture. Independent review gates both stages. The A coder's separate report covers A adapter numerical checks.

After review, use the same stage argument for roster freeze and supervised RUN. `A1` and `B1` write stage-specific rosters and artifacts while sharing the worktree's supervision lock.

```bash
cd /home/xzf/Code/phase0-3_three_directions_20261128/m3oe/examples
export PYTHONPATH=/home/xzf/Code/phase0-3_three_directions_20261128
export PYTHONDONTWRITEBYTECODE=1
project_python=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
study_artifacts=/home/xzf/Code/phase0-3_three_directions_20261128/artifacts/three_directions_20261128
"$project_python" -m experiments.three_directions_20261128.runner --stage A1 --artifacts "$study_artifacts" --freeze-roster
"$project_python" -m experiments.three_directions_20261128.runner --stage A1 --artifacts "$study_artifacts"
```
