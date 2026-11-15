# Causal Questions, Hypotheses and Interpretation

## 1. Target estimands

The Stage 1 design separates three quantities for each sampler family:

1. Auxiliary-objective effect: Z-PAIR minus CLEAN, and Z-EXAMPLE minus CLEAN.
2. Perturbation increment under a fixed auxiliary objective: B minus Z-PAIR, and C minus Z-EXAMPLE.
3. Auxiliary-construction difference: Z-PAIR minus Z-EXAMPLE.

All comparisons are paired by dataset and seed. They are validation-selected fixed-EMA comparisons; online results are reported separately. These are internal mechanism contrasts, not formal published-baseline comparisons.

## 2. Primary hypotheses

- H1 — Auxiliary objective explains most of the loss: zero-delta auxiliary arms reproduce a substantial part of the AUC degradation seen in B/C. This would indicate that loss composition, label prior, cell coverage or auxiliary weighting is the primary problem, rather than the direction of input perturbation.
- H2 — Perturbation has a negative incremental effect: B-Z-PAIR and/or C-Z-EXAMPLE are negative even when the auxiliary target and sampling rule are held constant. Then the current adversarial view is not helping ranking under that objective.
- H3 — Perturbation has an independent positive increment masked by the current objective: B-Z-PAIR or C-Z-EXAMPLE is positive, but the complete B/C arm remains below CLEAN. This supports preserving the perturbation idea while redesigning the objective.
- H4 — Pair construction is unnecessary: Z-PAIR and Z-EXAMPLE have similar outcomes, or C is better while requiring less machinery. Prefer the simpler example-level route unless pair construction has a clear independent effect.
- H5 — Dataset differences arise from different label priors or objective weights: inspect the actual per-cell/per-label contribution ledger. Do not use domain size as a difficulty proxy.

## 3. What the zero-delta controls identify

Z-PAIR estimates the effect of adding the pair-sampled, target-only auxiliary BCE with the same nominal loss scaling and pair sampler, but with the input embedding unchanged.

Z-EXAMPLE estimates the effect of adding the example-sampled, target-only auxiliary BCE with the same nominal loss scaling and row sampler, but with the input embedding unchanged.

B-Z-PAIR and C-Z-EXAMPLE estimate the additional effect of the perturbation operator conditional on the corresponding auxiliary-objective family and adaptive acceptance policy.

Z-PAIR vs Z-EXAMPLE compares two complete auxiliary constructions. It does not isolate only label balance: it also changes selection distribution, task/domain coverage and potentially effective cell weights.

## 4. What they do not identify alone

- Z-PAIR vs CLEAN does not isolate label balance from the additional BCE scale, six-cell auxiliary coverage or target-only task weighting.
- Z-EXAMPLE vs CLEAN does not isolate natural-label sampling from extra auxiliary mass or the target-only objective.
- B-Z-PAIR and C-Z-EXAMPLE are not exact per-step fixed-support interventions if acceptance masks depend on the evolving model. They identify the effect of using the perturbation algorithm under the same keyed candidate sampler and acceptance rule. The report must state this adaptive-support interpretation.
- C vs RANDOM/GAUSSIAN from the 2026-11-15 pilot cannot identify a task-gradient effect because support and realized radii were not fully matched.
- A small non-target logit drift does not imply preservation of the other task's AUC; ranking is a population-level pairwise property.
- LogLoss improvement does not establish ranking improvement or calibration improvement without a separate calibration analysis.

## 5. Support matching protocol

For every dataset/seed/epoch/global_step, use keyed independent RNG streams for row candidates, perturbation directions and diagnostics. Z-PAIR uses the same pair sampler and validity tests as B, then replaces every accepted delta with zero. Z-EXAMPLE uses the same example sampler and gradient/RMS validity tests as C, then replaces every accepted delta with zero.

Both treatment and zero arm log:
- proposed row IDs and task IDs;
- accepted row IDs and task IDs;
- per-cell counts and positive/negative counts;
- repeated row IDs across task cells;
- skip reason and mask;
- perturbation norm before/after cap and drift backtracking;
- auxiliary BCE numerator, denominator and per-cell contribution.

Because each arm's parameters evolve differently, the same keyed candidate rows do not guarantee identical accepted rows at every later step. Do not claim exact row-level matching unless the implementation actually replays a shared frozen acceptance trace. If acceptance rates or support composition differ materially, report it and trigger at most one targeted intention-to-augment support-control only when it would change the Stage 2 choice.

## 6. Decision interpretation

- If Z-PAIR and Z-EXAMPLE both degrade AUC materially, current auxiliary-objective design is the first problem. Do not tune perturbation alpha.
- If Z-PAIR degrades and Z-EXAMPLE is neutral or better, drop pair-balanced auxiliary BCE and prioritize natural-label example sampling.
- If zero arms are neutral/positive but perturbation increments are negative, stop current adversarial directions and investigate simpler task-specific view construction only if another mechanism has a concrete rationale.
- If C-Z-EXAMPLE is positive but C remains below CLEAN, preserve the direction as a secondary candidate while redesigning the objective to maintain natural per-row/task weights.
- If no zero arm or perturbation increment supplies a positive validation signal, stop this particular task-sensitive adversarial augmentation family rather than expanding the search space.
