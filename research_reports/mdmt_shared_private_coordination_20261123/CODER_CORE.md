# Coder implementation report · source and numerical core

The approved D2 design is bound by SHA-256 `7c1df7c873468de90b17ff53e32deab0721001bdd53dab2600de3b924467a74b`. This report covers the source adapter, native update primitives, and S0 CPU engineering checks.

## Implementation

- `source.py` reuses the v3 A0 configuration and fixed1 constructor, EMA02 native Adam and EMA, and the donor seed, RNG, tower-logit and state digest helpers. The Train/Validation-only data adapter verifies the raw byte hash, limits decoding to the frozen raw cutoff with pandas `nrows`, and uses the source vocabulary, scaler/schema and fingerprint definitions. On KuaiRand the first pass stops at the raw train cutoff. The validated frozen population admits only tabs 0/1/4, which the adapter maps after establishing the original positional boundaries. The returned bundle's fields are `features`, `train_x`, `train_y`, `train_row_ids`, `val_x`, `val_y`, and `val_row_ids`.
- `source.load_data(dataset)` returns `(bundle, identity)`. The identity binds raw boundaries, Train/Validation array, row and ID fingerprints, schema and preprocessing, label counts and a nested `sealed_test_provenance` from the frozen receipt. `data_hash` is the SHA-256 of the sorted, compact JSON identity prior to adding that field.
- `blocks.groups(model)` returns frozensets of canonical trainable names for `U`, `P_CELL`, `P_AXIS` and their registered private union `P_FULL`. `blocks.step(model, opt, ema, batches, cfg, *, names=None, detailed=True)` consumes one or two device-resident `(x, y)` pairs. For a single batch it preserves the canonical two-stage probability-BCE reduction and runs one native Adam step followed by one EMA tick. For two batches it computes independent gradients at the same online state, weights them by actual row counts, and performs one native Adam step. The two-batch EMA ticks occur at the first pre-step and second post-step online states. Both sequential orders call the one-batch primitive twice, recomputing the second forward after the first update.
- Included explicit-zero gradients reach native Adam with its existing moments and coupled weight decay. Excluded gradients are cleared to None and their live parameter and state identity is preserved. Named-group snapshots used for the detailed return are transient device tensors. `detailed=False` skips the snapshots and metric reductions for normal training.
- `s0.audit(output_dir)` writes source and data manifests and returns the verified dataset bundles. `s0.cpu_checks(output_dir, datasets=None)` uses the manifests, audits parameter membership and structural and empirical six-cell support, and checks source/adapter joint fidelity with each dataset's real first batch and natural tail. `s0.joint_fidelity(dataset, bundle, device, seed=2022)` is the equivalent one-device entry point for the reviewed CUDA smoke. The `s0` CLI exposes `prepare`, `cpu` and `gpu-fidelity` for these functions.

## Blocks return fields

| Field | Meaning per call |
| --- | --- |
| `loss`, `rows` | Natural batch BCE weighted by actual row count, and total original rows consumed |
| `training_forwards`, `parameter_backward_calls` | Actual optimization forward/backward counts, either one or two |
| `optimizer_steps`, `ema_updates` | One native Adam invocation and one EMA tick per consumed batch |
| `seconds` | Call wall time, including diagnostics when enabled and a CUDA stream synchronization on CUDA |
| `groups` | Detailed stats for U/P_CELL/P_AXIS/P_FULL, or an empty mapping in lightweight mode |
| `parameters`, `included_parameters`, `updated_parameters`, `absent_included_gradients` | Counts of named tensor parameters, configured inclusion, actual Adam participation, and included tensors with None gradients |
| `explicit_zero_gradients`, `nonzero_gradients` | Counts of participating parameter tensors with zero and nonzero effective supervised gradients before native decay |
| `gradient_norm`, `displacement_norm`, `coupled_decay_norm` | L2 norms of the group’s supervised gradient, observed Adam displacement, and coupled Adam gradient component `weight_decay × pre_step_parameter`, restricted to participating tensors |
| `exp_avg_norm_range`, `exp_avg_sq_norm_range` | Min/max L2 norms of post-step per-parameter moments across the whole group |
| `exp_avg_delta_norm_range`, `exp_avg_sq_delta_norm_range` | Min/max L2 norms of per-parameter moment displacement across the whole group |
| `step_before_range`, `step_after_range`, `step_delta_range` | Min/max native Adam per-parameter clocks and their changes across the whole group |

Absent native state contributes zero to reported clocks and moments. Empty per-parameter ranges are null. P_FULL overlaps its two component rows and must be counted only once when totaling private work.

## Checks

- Source and data manifests were written before numerical checks. The relevant v3 source donor tree and published model blob match both pinned branches. Runtime Train/Validation identities matched both the resolved protocol and 11-22 receipts.
- The six-cell support audit inspects the canonical native optimizer's membership and order, both report and source route orders, the four numeric fusion constants, gate input detachment, empirical per-parameter forward support under a deterministic perturbation, and six natural single-cell gradients. The audit reports exact zero and None gradients distinctly along with observed embedding row support. Captured tower logits reconstruct routed probabilities within `rtol=1e-6` and `atol=1e-7`.
- `python -m pytest -q -p no:cacheprovider experiments/mdmt_shared_private_coordination_20261123/tests/test_core.py` in `phase0-3-env` passed all six cases. They cover canonical joint fixture identity, disjoint/exhaustive parameter groups, concatenated-gradient equivalence for equal and unequal CPU batches (`rtol=2e-5`, `atol=2e-7`), both sequential recomputed poststates, EMA timing, and Adam masking and explicit zeros with nonzero prior moments and coupled decay.
- The real-batch CPU fidelity receipt gives the original 40,960-row first batch from the seed-2022 loader and the natural tail of 21,927 rows for MovieLens and 40,052 for KuaiRand, with an engineering update on the first batch followed by an engineering update on the tail. Interior batch training is outside this two-step fixture. The source comparison checks probabilities, natural loss, online and EMA states, and native Adam state with zero tolerance. The CUDA helper uses `rtol=2e-5`, `atol=2e-6` for source/adapter state comparison.

## Review handoff

The CPU receipts are in [S0 artifacts](../../artifacts/mdmt_shared_private_coordination_20261123/S0/). Independent review and the separate genuine CUDA/RAM smoke are pending. This implementation scope performed CPU engineering checks only. Test remains sealed. PyTorch 2.9 emitted its deprecation warning for the donor's TF32 setter and accepted the frozen setting.
