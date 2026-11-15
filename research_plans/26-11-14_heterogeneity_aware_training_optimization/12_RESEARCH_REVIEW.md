# Independent Research Design Review

## 1. Assumption audit

| Assumption | Status | Control |
|---|---|---|
| Per-example embedding gradients are separable | Must verify in source graph | Test batch independence and use per-example Jacobians if needed |
| Nullspace contains useful target sensitivity | Not guaranteed | Record projected norm and skip near-zero cases |
| Local preservation implies nonlinear preservation | False generally | Measure actual logit drift and bounded backtracking |
| Harder views improve generalization | Empirical hypothesis | Five-seed comparison to matched Gaussian/random controls |
| Pair labels define other-task utility | Not assumed | Other-task labels are not used in pair sampling |
| Six independent policies are needed | No evidence | Shared hyperparameters only |
| More projection is always better | Not assumed | TGA is matched simpler route |

## 2. Alternatives

- Isotropic Gaussian is cheaper; included as control.
- Random direction matched norm tests whether any perturbation is sufficient.
- TGA tests whether target gradient direction alone is sufficient.
- PAIR pairwise ranking loss is historical; excluded as a new route.
- Parameter-gradient projection is historical; excluded.
- A learned utility predictor is unnecessary and conflicts with prior prediction/action evidence.

## 3. Key risks that can change the decision

1. SAF (AAAI 2026) may be closer than abstract-level search suggests. Read full paper before novelty claims.
2. DTME-MTL (ICCV 2025) is a close token-space neighbor.
3. The actual M3oE graph may not expose clean embedding injection or pre-sigmoid logits without a small adapter.
4. Sensitivity extraction may be too expensive at current batch size.
5. The nullspace may remove nearly all target sensitivity in some domain-task cells.
6. Adversarial perturbation may harm calibration or amplify label noise.

## 4. What is not required before proceeding

- Proving a universal mechanism across all recommender architectures.
- Testing every epsilon or pair count.
- Repeating fixed EMA efficacy experiments.
- Repeating PAIR/P01 configurations.
- Adding a new architecture.
- Explaining all historical domain/task differences.

## 5. Decision

The plan is suitable for a bounded implementation pilot, not yet a confirmed paper method. The critical go/no-go is whether DT-PGNA provides measurable incremental AUC over TGA and matched generic perturbations at acceptable cost. This can be decided with the planned five-seed stages.


## Current-task review supplement (2026-11-15)

- Current audit scope is main-only, documented in 01_REPOSITORY_AUDIT.md and 16_CURRENT_MAIN_AUDIT_AND_EXECUTION_REFINEMENT.md. The previous branch inventory in the historical refinement file is not part of this task's audit.
- The current main model wrapper/reference and PAIR implementation were inspected directly. The proposed intervention object is the input embedding, not parameter gradients or pairwise loss.
- The initial same-other-task-label PAIR proposal is explicitly excluded as duplicate.
- The two-task exact projection replaces the earlier damped projection; lambda_J is removed.
- TGA-Example is included as the simpler adversarial augmentation comparator because target-logit gradient perturbation and per-example BCE adversarial perturbation are directionally equivalent up to the BCE scalar/sign, after normalization.
- HPO and formal confirmation now use disjoint five-seed pools.
- The remaining material risks are: measured wall-time/peak-memory cost, actual nonlinear non-target-logit drift, and whether DT-PGNA adds AUC beyond TGA-Example and matched generic perturbations. These are decision-relevant and belong in the planned pilot/confirmation; no additional speculative experiment is required at planning time.
