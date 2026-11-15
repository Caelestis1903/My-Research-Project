# Evaluation and Decision Tree

## 1. Primary endpoint

Validation-selected Test Macro AUC under AUC_SELECTED_V1. Report all six cell AUC and LogLoss, Macro AUC/LL, traffic and pooled metrics.

## 2. Decision sequence

1. Correctness gate fails: stop implementation; classify ENGINEERING_FAILED.
2. Projected gradient is numerically zero for most pairs: DT-PGNA is NOT_INFORMATIVE under this model; evaluate TGA only if its gradient is valid.
3. Non-target logit drift exceeds tolerance after backtracking: stop DT-PGNA; retain TGA only as an alternative if it improves matched controls.
4. TGA and DT-PGNA both fail to beat matched random/Gaussian controls on validation in five seeds: stop the family; do not expand hyperparameter search.
5. TGA improves but DT-PGNA does not improve further: retain TGA as the simpler candidate; do not claim cross-task preservation adds value.
6. DT-PGNA improves over TGA at matched norm and cost: advance DT-PGNA to frozen five-seed confirmation.
7. Candidate beats internal fixed EMA but not published baseline: report component-level promise, not formal algorithm promotion.
8. Candidate passes +0.003 Macro AUC and six-cell −0.001 protection: eligible for formal promotion, subject to independent review and literature positioning.
9. AUC passes, LL worsens: preserve AUC decision and report LL cost separately.
10. LL improves, AUC does not: do not change primary objective after observing results.
11. Stage not triggered: record NOT_TRIGGERED, never FAIL.
12. One dataset positive and another negative: retain dataset-specific boundary; do not average away contradiction or reset positive evidence.

## 3. Required causal contrasts

- DT-PGNA vs TGA at matched perturbation norm: value of non-target task constraint.
- TGA vs random direction: value of target-task gradient direction.
- TGA vs isotropic Gaussian: value of directional augmentation over generic noise.
- Candidate EMA vs candidate online: EMA contribution, separate from algorithm.
- Candidate vs clean fixed EMA: incremental algorithm contribution.

No experiment is justified merely to prove that domain size is not difficulty. Domain size is not used as a difficulty proxy anywhere in this plan.


## Binding current-task decision rules (2026-11-15)

Use the disjoint seed pools in manifest v3: development/HPO 2027–2031 and formal confirmation 2022–2026. Development results are validation-only; no Test access before formal freeze.

The route set is:
- A: DT-PGNA, primary.
- B: TGA-Pair, projection ablation.
- C: TGA-Example, pairless task-BCE adversarial alternative.
- Controls: matched random direction and matched isotropic Gaussian.

Decision sequence:
1. Correctness/forward-graph failure → ENGINEERING_BLOCKED.
2. Near-zero projected q or excessive nonlinear non-target drift after the frozen backtracking rule → DT-PGNA version is not numerically viable; do not relabel as an AUC failure.
3. If A does not beat B at matched perturbation norm, projection has no demonstrated incremental value; prefer B only if it beats C and matched generic controls.
4. If B does not beat C, pair construction and margin capping do not earn their extra complexity; prefer C only if it beats matched generic controls.
5. If C does not beat matched random/Gaussian controls, stop this specific adversarial augmentation family; do not revive PAIR, PCGrad/ECCM or adaptive EMA as automatic next steps.
6. HPO and route selection use all five development seeds and validation only. Formal confirmation uses the separate five formal seeds.
7. Test is read only after the formal configuration is frozen. Report each candidate's total gain vs published models and incremental gain vs clean fixed EMA separately.
8. If a route or stage is not executed, label it NOT_TRIGGERED. If the process fails technically, label ENGINEERING_BLOCKED. Neither is a scientific negative.
9. Formal promotion requires paired mean Macro AUC gain >= +0.003 vs the strongest applicable published model and each cell mean AUC delta >= -0.001. All-seed positivity is not required. LogLoss remains independent.
