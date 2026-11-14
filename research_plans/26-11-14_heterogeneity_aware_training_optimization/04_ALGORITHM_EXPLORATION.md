# Algorithm Exploration and Route Comparison

## Route A — DT-PGNA (primary)
Domain–Task Pairwise Gradient-Nullspace Augmentation. Uses target-task pair sensitivity and projects each per-example perturbation away from non-target task-logit sensitivities. Adds target-only BCE on perturbed examples.

Strength: explicit MDMT interaction constraint at the input representation level; directly acts on training data; no new model parameters or inference cost.  
Risk: second-order/nonlinear drift, Jacobian cost, and close conceptual neighbors in token-space gradient manipulation and adversarial augmentation.

## Route B — TGA (fallback)
Task-Gradient Augmentation. Same target-task adversarial perturbation, but no nullspace projection. It is simpler and cheaper.

Strength: lower compute, simpler implementation, isolates whether task-specific sensitivity is useful at all.  
Risk: can damage other-task predictions and may be no better than ordinary Gaussian noise.

## Route C — PAIR-style conditional ranking loss (not retained)
A target positive/negative pair constrained by the other-task label, followed by a pairwise ranking objective.

Decision: do not promote as a new route. Historical PAIR already selected within domain/task and joint-label boundaries; P01/P10 separated components; the five-seed P01 configuration has mixed component-level cell costs. A different sampler or notation is not enough. Reopen only if a genuinely different mechanism is specified.

## Route D — parameter-gradient balancing/projection (not retained)
Decision: no new generic gradient scaling, PCGrad, or ECCM variation. Historical S7/S8, GATS B2, CP, and EMA-related experiments cover much of this space. A new optimizer route would need a distinct signal and an explicit actual-Adam-update mechanism, not merely a new norm statistic.

## Comparison

| Dimension | DT-PGNA | TGA |
|---|---|---|
| Evidence gap | Non-target-output-preserving task-conditioned perturbation not directly tested in reviewed reports | Task-gradient input perturbation not directly isolated from Gaussian |
| Duplicate risk | Medium: DTME-MTL, adversarial augmentation, CrossGrad are adjacent | High: adversarial/task-guided augmentation is established |
| MDMT specificity | Domain-task target plus explicit other-task output constraint | Domain-task target, but no explicit interaction protection |
| Core assumption | Local nullspace direction retains target sensitivity and remains label-valid | Target gradient direction yields useful hard examples |
| Complexity | Higher; small Gram solve per sample | Lower |
| Inference cost | Zero | Zero |
| Paper potential | Medium, conditional on incremental AUC and mechanism | Low-to-medium as standalone; strong control/fallback |
| Decision | First route to test | Fallback / nested control |

## Why not use six independent learned coefficients?

No evidence supports six independently learned policy parameters. They increase degrees of freedom and invite overfitting. The primary route uses a shared epsilon rule and the domain-task cell only to select which task logit supplies the sensitivity. Per-cell adaptation is not assumed necessary.

## Why not use other-task label matching?

The repository's PAIR experiments already used joint-label-conditioned pair selection. The current design instead uses the other task's output Jacobian as a local constraint. It does not infer output invariance from label equality.
