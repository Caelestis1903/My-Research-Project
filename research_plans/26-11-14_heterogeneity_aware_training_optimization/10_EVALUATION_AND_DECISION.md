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
