# CODER IMPLEMENTATION REPORT — core

Implemented the approved DESIGN SPECIFICATION (SHA256 `9e97e74cc00311cc9b9035a4dedb370b94c1a5324442d700dc78dd0049124617`) in `experiments/ema_view_design_20261102_v2/`.

## Implemented responsibilities

- `protocol.py`: original YAML plus pinned dataset identities, registered configurations, standard single Adam, fixed post-step EMA, source RAM/record helpers.
- `rng.py`: frozen stateless uint64 SplitMix64 with Box–Muller normals; study/dataset/split/seed/epoch/step/row/role keys. J donor keys are shared across J arms.
- `input_probe.py`: copied STAR/skip/expert/gate/tower numerical forward, true raw logits, detached functional parameter map. Only the probe removes gate input detach. The original source training forward remains unchanged.
- `views_joint_label.py`: exact two-donor joint-label pools, self exclusion, shared capped radius, random/max-min/mean selection, no positive-score gate.
- `views_ema_field.py`: FP64 rank-truncated joint projection, expected equal-model fallback, uniform gain cap, F0/F1/FR/F_PR, current-state FR counterfactual energy, diagnostic teacher responses.
- `views.py`: conditional TG/RG/AG directions and task-specific view dispatch.
- `objectives.py`: exact source A0/A1 callable, original `2B` budget, detached perturbations, task-specific TG/RG losses, one supervised backward/Adam step/EMA update. A1 and constructed views record actual perturbation energy. Disabled and empty branches preserve source A0.

The core follows Section14 public APIs. `train_step` owns gradient zeroing, optimizer and EMA updates; the executor must not repeat them. Source `make_model` is available under `vendor.gradient_aug_20261031_v2.views`.

## Sources and adaptations

`vendor/SOURCE_MANIFEST.json` lists 21 exact source files with original paths, GATS release commit `e59efa2fd954c4d2231b679a742406cfbe2a1077`, Git blob, SHA256 and byte count. Source relative imports are unchanged and resolve under the new vendor namespace. The historical top-level protocol reader and persistent source-data cache loader are not called.

The short EMA/strict earliest-selected helpers follow `experiments/gradient_aug_momentum_20261101/protocol.py`, blob `d1ef89c67dc0ddb7bc5c529c2e95f1b1dd253615` in that GATS release. `state_digest` follows the same recursive scheme and additionally hashes NumPy arrays by bytes for RAM/RNG identities. These helpers are in the new protocol module; no historical file was edited.

## Checks

Command: `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/ema_view_design_20261102_v2/tests/test_core.py -q`.

Final result: **5 passed in 3.21s**. The five grouped semantic tests cover:

1. Probe/source probability agreement, full FP64 numerical input derivative, constant parameters and preserved probe RNG/state.
2. Row-order independent keyed draws, joint-label/self-excluded distinct donors, singleton handling and J0/J1/J_SUM common radii.
3. Projection ranks0/1/2, idempotence, joint residual, uniform cap, equal-model degeneracy and FR row-energy matching.
4. Exact A0/A1 scalar and parameter gradients against the pinned source with the same explicit Gaussian; original tail-batch denominator, empty/disabled branches and TG/RG task budget.
5. Whole-delta identity derivative, Adam None/explicit-zero coupled-decay behavior, post-Adam EMA order and one finite supervised update for every registered arm.

The initial test collection required the package's empty `tests/__init__.py`; it was added. No research algorithm changed in response to test outcomes.

No formal or CUDA experiment was run by this coder. Runtime/data/guardian/gates are owned by the separate execution coder; their checks and implementation report complete the composed implementation. No in-scope core blocker remains for independent review.
