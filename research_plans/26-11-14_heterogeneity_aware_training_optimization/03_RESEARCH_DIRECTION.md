# Research Direction

## 1. Current conclusion

The most defensible new question is not “how to add another conditional pairwise loss?” It is:

> Can domain–task-specific sensitivity generate harder training views for one task, while explicitly preserving the other task's predictions on the same samples?

This is an input/representation-level training intervention. It does not alter expert, gate, shared-module, or knowledge-fusion architecture. It does not depend on domain size.

## 2. Evidence-based motivation

1. Full configurations combining conditional information and fixed EMA can improve total AUC, but component increments are not always positive. This motivates isolating a new component against fixed-EMA control.
2. Gaussian D0 has dataset-dependent results; isotropic noise does not encode which task output is sensitive to which input direction.
3. PAIR has already conditioned sample-pair selection on task/domain and joint labels. Repeating that condition alone is not a research gap.
4. Historical parameter-gradient operations, including PCGrad-like work and GATS B2, do not establish that an input-space perturbation guided by a task's ranking sensitivity is ineffective.
5. EMA mechanism research shows that better signal prediction does not automatically produce useful actions. The proposed method directly creates a perturbed view and trains on it; it does not predict future gains.

## 3. Proposed distinction

For a target domain-task cell (d,t), use the target task's own logit sensitivity to reduce the positive-negative pair margin. For each member of the pair, project that target sensitivity into the nullspace of the other task's logit Jacobian. The perturbation therefore makes the target task's pair harder while preserving other-task logits to first order.

The augmented view is trained with the original target-task BCE labels. No auxiliary pairwise ranking loss is added. The pair is a device for defining the perturbation direction, not a new pairwise objective.

## 4. Explicit assumptions

- The fixed M3oE forward graph is differentiable from the embedding tensor to each task logit.
- Per-example gradients can be computed without cross-example coupling; verify against the actual forward path. If batch normalization or other cross-example operations exist, compute per-example Jacobians correctly or stop the route.
- A local nullspace of non-target logits exists with non-negligible target-task sensitivity. If projected target gradient is nearly zero, skip that sample/pair rather than amplify numerical noise.
- A small perturbation can create a useful hard view without changing target-label semantics. This is an empirical augmentation assumption, not a theorem.
- Local first-order preservation is only approximate in the nonlinear network; actual post-perturbation non-target-logit drift must be measured and bounded.

## 5. Non-assumptions

- Domain ID is not a difficulty score.
- Task label equality is not assumed to imply gradient alignment.
- Gradient cosine is not treated as a proxy for AUC gain.
- A larger perturbation is not assumed better.
- The projected method is not assumed better than its unprojected alternative.
- The full method is not assumed to improve both AUC and LogLoss.

## 6. Research decision

Proceed to a bounded five-seed pilot only after code-level Jacobian and projection checks pass. The principal question is whether projection adds useful value over simple task-gradient augmentation at matched perturbation norm and compute. If it does not, retain TGA only if it beats matched Gaussian/random controls. Otherwise stop this route without relabeling the result as a failure of all heterogeneity-aware optimization.
