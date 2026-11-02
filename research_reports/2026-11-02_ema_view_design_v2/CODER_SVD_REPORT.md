# CODER IMPLEMENTATION REPORT — SVD placement

Implemented the approved `DESIGN_ADDENDUM_SVD_PLACEMENT.md` (SHA256 `bdd471732067d48ec42843680ca25c9f54f251ea1ee678639755414c91af0bfb`) before scientific dispatch.

Only `views_ema_field.null_basis` changed: the normalized FP64 `[n,2,F]` matrix is passed to the same `torch.linalg.svd(..., full_matrices=False)` on CPU, and the retained FP64 basis and rank counts return to the original device. Normalization, singular thresholds, projector, fallback, scaling/cap, final dtype and RNG are unchanged. This placement is fixed for both datasets and every F1/FR/F_PR call. Vendored source remains unchanged.

## Bounded engineering measurement

The actual KuaiRand seed2022 first shuffled original training batch contained40960 original rows and1843 target rows. Its actual teacher Jacobian had shape `[1843,2,33]`. All model/EMA/Jacobian/vector states stayed in process RAM. The scalar receipt is `SVD_MICROCHECK.json`.

| Operation | CUDA SVD path | CPU SVD path including transfers |
|---|---:|---:|
| Null basis | 2.693276913s | .005797760s |
| Same-state F1 view | 2.857549159s | .031553833s |
| Same-state FR view | 2.724092493s | .036790044s |
| Same-state F_PR view | 2.724384216s | .032562698s |

The microcheck used cuda:0. Its recorded precheck load was20864/24576MiB and199.13/350W; cuda:1 had18189MiB and271.83W. External jobs remained running. These are engineering timings under the recorded concurrent load.

Ranks agreed for every actual row. Fixed-vector FP64 projected maximum absolute error was `2.220446049250313e-15`; maximum normalized joint residual was `3.711471586074082e-15`. Both meet the addendum tolerances. F1/FR/F_PR final FP32 deltas were identical in this same initial actual model/EMA state (maximum absolute error0); fallback/rank/cap decisions and scalar work counters agreed. Model, EMA and RNG hashes were unchanged by the paired checks.

Total microcheck physical time including source/data loading, actual probes, transfers, both SVD paths and comparisons was53.324206686s. Work:7 teacher probes,4 student probes,18 input-gradient calls,8 SVD calls (4 CUDA/4 CPU),0 optimizer steps,0 parameter backwards,0 test accesses. This is calibration work and adds no scientific repetition.

## Focused acceptance and identity

The existing focused rank/projection/degeneracy/FR-energy test passed under the changed source:

`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/ema_view_design_20261102_v2/tests/test_core.py::test_joint_projection_cap_degeneracy_and_fr_energy -q`

Result: **1 passed in1.58s**. No broader test suite or research training was added.

- Source manifest before: `e524699526ae6e8eaf38950a0255eb359c7ca7705a64dab4f60bf9aa1ea93307`.
- Source manifest after: `f2658c2163902e6e69d0ea562b0aded9ba0c635477c19edb140d6353bcb15e82`.
- Changed file SHA256: `fed4dbdb06da69a318bd1f26aabc223fc564dd5f02b2d94f8caee490f8e353da`.

Independent review and the prescribed repeated real-data F1 Stage0 remain the next required stages before S1. This coder did not run scientific trajectories or interpret method performance.
