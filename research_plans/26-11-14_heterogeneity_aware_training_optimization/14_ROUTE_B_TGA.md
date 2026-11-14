# Route B — TGA

## Objective

Determine whether task-gradient-guided hard views are useful without cross-task projection.

## Algorithm

Use the same target positive/negative pairs and same pair-level margin cap as DT-PGNA, but set q_t(x)=g_t(x)=grad_e z_t(x).

Positive samples are perturbed to reduce their target logit; negative samples are perturbed to increase it. Train only the target task's BCE on the augmented view, in addition to the clean multi-task objective.

## Why retain it

TGA is the simplest alternative to DT-PGNA. If TGA works and DT-PGNA does not improve at matched norm, the cross-task constraint is unnecessary. If TGA fails against Gaussian/random controls, there is no reason to spend compute on a more complex projection.

## Limits

TGA has no guarantee of preserving non-target outputs. Its actual cross-task logit drift must be reported. It is a fallback/control, not automatically a publishable method.
