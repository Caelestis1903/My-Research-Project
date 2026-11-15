# Experiment Matrix and Conditional Stages

## Stage 0 — read-only audit

No training. Verify repository commit, source hashes, data identity, existing outputs, and the 2026-11-15 run reuse decision. Produce a machine-readable audit and a per-arm reuse table.

## Stage 1 — zero-perturbation attribution

Primary new arms:
- Z-PAIR: two datasets × five seeds = 10 full trajectories.
- Z-EXAMPLE: two datasets × five seeds = 10 full trajectories.

Required internal comparators:
- CLEAN fixed EMA;
- B/TGA-Pair;
- C/TGA-Example.

Reuse these only if the identity checks in 02_SOURCE_CODE_AUDIT.md pass. Otherwise rerun only the mismatched arms under the same five seeds and frozen protocol. Stage 1 maximum planned new training is 20 trajectories / 1000 epochs, excluding mandatory reruns caused by identity failure.

Every arm: seeds 2022–2026, 50 epochs, batch 40960, natural loader order and inherited Adam/EMA. No alpha search. No Test.

### Stage 1 metrics and diagnostics

- Validation selected online and fixed-EMA six-cell AUC and LogLoss.
- Six cell AUC/LogLoss, Macro, Traffic and Pooled metrics.
- Per-seed paired deltas for Z-PAIR vs CLEAN, Z-EXAMPLE vs CLEAN, B vs Z-PAIR, C vs Z-EXAMPLE, and Z-PAIR vs Z-EXAMPLE.
- Selected epoch and common epochs 10, 30 and 50.
- Per-cell/per-label effective auxiliary weight, accepted/proposed counts, skip rates, unique row IDs and duplicate appearances.
- Clean/auxiliary/total loss values and gradient norm/cosine diagnostics.
- Training wall time, throughput, peak allocated/reserved GPU memory.
- Confidence intervals are descriptive and do not become extra hard gates beyond the predeclared decision rules.

## Stage 1 interpretation gates

The key gate is not merely whether Z differs from CLEAN. Consider the decomposition:

B - CLEAN = (Z-PAIR - CLEAN) + (B - Z-PAIR)

C - CLEAN = (Z-EXAMPLE - CLEAN) + (C - Z-EXAMPLE)

This is a descriptive decomposition across separately trained trajectories, not an exact additive causal identity because the training trajectories interact nonlinearly.

Advance to Stage 2 only if at least one of:
1. A zero arm is neutral/positive and the corresponding perturbation increment is positive; or
2. A zero arm is harmful but a clear positive perturbation increment shows the perturbation can compensate, motivating a natural-weight-preserving objective; or
3. A specific label/task-weight issue is identified in the effective-weight ledger and can be corrected with one lower-assumption objective.

Do not advance if both zero arms are harmful and both matched perturbation increments are non-positive with no clear cell-specific signal.

## Stage 2 — natural-distribution-preserving augmentation

Initial matrix, only after Stage 1 gate:
- NDP-A (full natural-batch task-specific view mixture);
- CLEAN fixed EMA reference;
- optional NDP-B only if measured full-batch cost makes NDP-A impractical.

Two datasets × five seeds per arm. Initially fix lambda=.25 as a nominal mixture coefficient only if the mixture formula is implemented exactly; otherwise choose a single predeclared value after deriving the scale. No multi-cell coefficients.

No cross-task projection in the first version. Keep task-specific views and target-task losses. Test remains sealed.

## Stage 3 — bounded hyperparameter search

Only if Stage 2 shows a positive validation signal:
- Search one global lambda in {0.1, 0.25, 0.5}, using all five seeds on both datasets for each value.
- If perturbation radius alpha is still new/unfrozen, search alpha in {0.01, 0.03, 0.05}; only search alpha if Stage 1/2 diagnostics show radius is a decision-relevant uncertainty.
- Do not simultaneously expand lambda, alpha and subset probability. One dimension per search round.
- Select by the predeclared mean validation fixed-EMA Macro AUC across datasets; tie-break toward lower lambda and smaller alpha.
- Report per-dataset and per-cell differences; a pooled mean must not conceal a major dataset-specific reversal.

If candidate results are inconsistent, first localize the condition (dataset, task, stage, label support) using existing outputs. Add a new experiment only if that distinction changes the next algorithm choice.

## Stage 4 — frozen evaluation

Freeze code commit, configuration, data hashes, sampler, objective, optimizer, EMA, seed list and checkpoint rule before opening Test.

Evaluate all five seeds on Test for:
- CLEAN reference;
- selected candidate;
- its decisive zero-delta or matched control;
- strongest applicable published models, if available under the same population/split and selection protocol.

No Test-driven hyperparameter changes. No independent-seed claim: the same 2022–2026 seed pool is reused by instruction. Report this explicitly as same-seed paired confirmation, not untouched-seed generalization.

## Stage 5 — conditional ablation

Only run a targeted ablation if the main result depends on a component whose effect is not otherwise identified. Candidate questions:
- task-specific view vs one shared perturbation view;
- full-batch vs inclusion-probability-corrected subset;
- no protection vs one soft non-target trust-region constraint;
- convex mixture vs extra auxiliary loss, if the latter is still considered.

Do not launch a full factorial design.
