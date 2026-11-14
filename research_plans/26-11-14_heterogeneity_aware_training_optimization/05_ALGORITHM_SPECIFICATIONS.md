# Algorithm Specifications

## 1. Notation

For domain d and target task t, let a training pair contain a target-positive sample x+ and target-negative sample x− from the same domain. Let e(x) be the concatenated field-embedding tensor at the verified insertion point. Let z_s(x) be the pre-sigmoid logit for task s in domain d.

The clean target margin is m_t=z_t(x+)-z_t(x−).

The pair is sampled using target task labels only. No constraint is imposed on other-task labels.

## 2. Per-sample task sensitivity

For one sample x, define g_t(x)=grad_e z_t(x). Stack the gradients of all non-target task logits:

J_-t(x) = [grad_e z_s(x)^T] for all s != t.

For the current 3-domain x 2-task platform, J has one row per sample. For a future T-task extension it has T−1 rows.

The damped nullspace projection is:

P_-t(x) = I - J_-t^T (J_-t J_-t^T + lambda_J I)^(-1) J_-t.

q_t(x) = P_-t(x) g_t(x).

This is a stable damped projection, not an exact projector when lambda_J is nonzero. The implementation must report residual ||Jq|| / (||J|| ||q|| + eps).

## 3. Pair-directed hard perturbation

For the positive member, decrease the target logit; for the negative member, increase it:

delta+(x+) = -epsilon_x q_t(x+) / (||q_t(x+)||_2 + epsilon0)

delta-(x-) = +epsilon_x q_t(x-) / (||q_t(x-)||_2 + epsilon0).

Thus the target pair margin is locally reduced, while non-target logits are locally preserved to first order.

Use a scale-relative radius r(x)=RMS(e(x)), epsilon_x=alpha*r(x). A pair-level cap is applied so the predicted first-order margin reduction does not exceed kappa*max(|m_t|,m_floor). If either projected norm is below a fixed numerical threshold, skip the pair and record it. Do not normalize near-zero vectors into large perturbations.

The exact cap formula and default m_floor are frozen in the implementation config before HPO; they are not tuned independently per domain-task cell.

## 4. Training objective

Clean base objective remains the exact existing M3oE fixed1 multi-task objective:

L_clean = sum over domain-task cells of existing BCE group losses.

For each sampled target pair, forward the perturbed positive and negative examples and compute only target-task BCE:

L_aug = mean over sampled pairs of [BCE(z_t(x+ + delta+),1) + BCE(z_t(x- + delta-),0)].

L = L_clean + lambda_aug * L_aug.

No pairwise softplus ranking loss is added. The pair selects the perturbation direction; the original BCE labels supervise the augmented view. Other-task losses continue to be computed on clean examples.

## 5. Route B (TGA)

Replace q_t(x)=P_-t(x)g_t(x) with q_t(x)=g_t(x). All other operations, pair selection, scale, cap and objective remain identical. This is the matched simpler alternative.

## 6. Inference and model structure

- No new trainable parameters.
- No expert/gate/tower/shared-module changes.
- No inference-time perturbation.
- No added inference latency.
- Model architecture and fixed1 fusion remain unchanged.

## 7. Complexity

Let B be clean batch size, K sampled pairs per domain-task cell, F embedding dimension, T tasks, and C_f forward/backward cost.

The clean model cost remains unchanged. Augmented forward cost is approximately O(2KDT) additional sample evaluations, where duplicate samples may be deduplicated. Sensitivity extraction requires task-logit gradients with respect to embedding outputs; the Gram solve is O((T−1)^3) per sample after forming J J^T, negligible for T=2 compared with the backward pass. Actual cost must be profiled, not inferred from FLOPs alone.

## 8. Failure modes

- q_t approximately zero: skip; report skip rate.
- Non-target logit drift exceeds tolerance after actual nonlinear forward: reduce radius once using fixed backtracking; if still over tolerance, skip.
- Target margin flips too often: lower radius or reduce kappa; do not tune per cell.
- AUC does not improve over matched Gaussian/random controls: stop route.
- AUC improves but non-target cells violate the six-cell protection rule: not eligible for promotion.
- LogLoss worsens: report separately; do not reinterpret it as AUC failure.
