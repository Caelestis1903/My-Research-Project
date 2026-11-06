# CODER IMPLEMENTATION REPORT — core

Implemented the approved DESIGN SPECIFICATION in the new package only. Frozen source files were not modified. Scientific experiment execution and interpretation remain with the runner and orchestrator.

## Owned files

- `__init__.py`, `shadows.py`: physical EMA bank, F aliases, exact after-Adam clocks and finite pulse state.
- `readouts.py`: one original probability forward with tower-logit capture, full-function gate option, original probability AUC/log-floor LL, full-Validation panel extraction and group hybrids.
- `observers.py`: immediate group innovation reductions, M0/M1, both full/source-detach M2 scalar reductions, exact M3 and panel replicas. Complete gradients remain transient.
- `worker.py`: source clean/G-D0 loss, coupled Adam, original loader order/noise identity, all regular readouts, independent source selection score, pulse anchors, epoch scalar record and CPU RAM export/restore.
- `state.py`: dedicated authenticated local RAM guardian, ownership, 4 GiB model-state cap, scalar/prediction atomic epoch commits with rollback, prediction whitelist and 2 GiB disk cap, deterministic replay comparison and one-restart budget.
- `run.py`: all eleven public CLI commands and internal guardian/worker entrypoints, full-batch engineering smoke, two-worker scheduler and per-GPU resource admission.
- `tests/test_core.py`, `README.md`.

`protocol.py` was initially drafted here and then transferred to the protocol coder before integration. That coder owns its final implementation and `panels.py`. `analysis.py`/`evaluation.py` belong to the analysis coder; shared interfaces are those in DESIGN section 7.

## Necessary checks completed

Project interpreter: `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`.

`python -m pytest experiments/ema_mechanism_20261106/tests/test_core.py -q`: **8 passed**. The tests cover:

1. EMA alias and pulse/group-coordinate equivalence, inactive clock and invalid update order.
2. Probability ties, saturated source LL floor and missing-class AUC.
3. Prediction-only NPZ field/dtype/shape whitelist, size cap and idempotence.
4. RAM commit idempotence, conflicting replay refusal and rollback.
5. Clean/G-D0 source loss, gradients, one Adam update and F equality, followed by RAM restore and equal continuation.
6. Full functional derivative versus finite differences, identical full/source forward, hook cleanup and online gradient purity.
7. A complete small real-input observer pass: 72 group/cell feature rows, exactly 24 functional supports, 27×6×5 pulse metric rows, all manifest feature-block fields, no online feedback.

`compileall` passed for the new package at the initial integration point. All eleven public subcommands' `--help` passed in one interpreter. PyTorch emitted its existing TF32-setting deprecation warning from the unchanged source setup function.

## Runtime handoff

The review must be copied to `out/IMPLEMENTATION_REVIEW.md` before `train`. `preflight` freezes the current implementation hash. `validate` and both dataset smoke receipts bind that same hash; code changes require new matching receipts. The scheduler keeps the guardian alive through S1–S4 and the independent Test process. `release` requires final report and endpoint artifacts and frees only this task's RAM.

No scientific trajectory or GPU smoke was started by this coder. Full-batch resource measurements, real IPC lifecycle and complete workflow acceptance are to be executed after the independent reviewer accepts the code. The same output directory and manifest must be used for a legitimate resume; a lost guardian can only trigger one deterministic replay of each affected logical trajectory.

Final integration check: core, protocol and panel tests together **12 passed**. CLI/smoke source was syntax-checked after adding explicit RNG/loader/Adam purity assertions and per-epoch IPC accounting (`IPC_COSTS.jsonl`). `RESOURCE_ESTIMATE.json` scales measured smoke training, Validation, pulse and gradient times using the frozen counts; its formula is included in that artifact.

## Review revision R1 — deterministic observation reduction

The reviewer identified FP32 CUDA `scatter_add_` atomic accumulation in observation reductions. It could alter M2 numeric precision and the exact scientific-record identity required for replay. `observers.reduce_groups` now iterates the frozen twelve groups, preserves their coordinate order and sums in float64 without scatter atomics. Innovation/geometry differences and products, M2 displacement differences and gradient products/energies are formed in float64 before summation. Source online loss, gradients, Adam and EMA updates remain unchanged.

Added one small cancellation-scale/exact-repeat check. Re-ran the existing finite-difference, source-equality, RAM-continuation and observer-schema checks: `tests/test_core.py` **9 passed**. No GPU or scientific trajectory was launched for this revision.

## Review revision R2 — measured guardian RSS

Added actual guardian `VmRSS` and `resource.ru_maxrss` byte measurements. Guardian status exposes current RSS and peak; every successful epoch commit records `peak_guardian_ram_bytes` in cost and the RAM summary, so the existing `peak_` maximum aggregation carries it into COSTS. Smoke receipts include the worker process RSS peak. Model payload accounting remains a separate `tensor_bytes` measure; scientific records/metrics are unchanged. The existing commit test now checks both record and summary RSS fields. Core checks: **9 passed**.

## Preflight revision R3 — unequal Validation domain populations

Real KuaiRand Validation has domain populations 2692/246232/1843, so the prescribed natural mechanism sample counts are 2692/4096/1843. The designer approved a fixed 4096-row storage container with `valid_mask` derived from nonnegative source indices. Core readouts now forward only those real rows and scatter outputs into a zero-initialized container; full Validation extraction also starts from zero and gathers only valid indices. Every V_mech/V_A/V_B pulse metric intersects its row mask with `valid_mask`. No sentinel enters a forward or metric. Source full-Validation training/selection population and the prediction byte cap remain unchanged.

One targeted unequal-domain test verifies the actual forwarded row count, zero unused slots, per-domain/per-row-side metric counts, exact full-Validation metric preservation at the same batch partition and correct panel gathering. Core checks: **10 passed**. Preflight had not written PREPARED; no smoke or scientific Adam trajectory had started when this issue was corrected.
