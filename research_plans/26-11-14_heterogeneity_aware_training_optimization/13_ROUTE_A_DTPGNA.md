# Route A — DT-PGNA

## Objective

Generate a task-specific hard training view for each domain-task pair while preserving the other task's output locally.

## Algorithm

For every domain d and target task t:
1. Sample K target-positive and K target-negative examples from the current natural minibatch, paired without replacement where possible.
2. Extract embedding tensor e(x) and logits z_s(x).
3. Compute g_t(x)=grad_e z_t(x), and J_-t(x) from all other task logits.
4. Compute q_t(x)=g_t(x)-J^T solve(J J^T + lambda_J I, J g_t(x)).
5. For target-positive examples perturb opposite to g_t; for target-negative examples perturb along g_t. Normalize by q norm and scale by alpha times embedding RMS.
6. Apply the frozen margin cap. Skip near-zero q.
7. Run actual perturbed forward. If non-target logit drift exceeds tolerance, apply at most two deterministic radius halvings; otherwise skip.
8. Compute target-only BCE on augmented examples.
9. Add lambda_aug times augmented BCE to original clean objective.

## Pseudocode

    for each minibatch:
        clean_logits, embedding = model.forward_with_embedding(x)
        L_clean = original_multitask_BCE(clean_logits, y)
        L_aug = 0
        for d in domains:
            for t in tasks:
                pairs = sample_target_positive_negative(batch[d], y[:, t], K)
                for x in unique_members(pairs):
                    g = grad(logit[d,t](x), embedding[x])
                    J = stack(grad(logit[d,s](x), embedding[x]) for s != t)
                    q = g - J.T @ solve(J @ J.T + lambda_J * I, J @ g)
                    if norm(q) < threshold: skip
                    delta = signed_normalized(q) * alpha * RMS(embedding[x])
                apply_pair_margin_cap_and_nonlinear_drift_check()
                L_aug += target_BCE(model(embedding + delta), target_labels)
        loss = L_clean + lambda_aug * normalize(L_aug)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)

The code must ensure one clean model update and one EMA update per batch. Do not retain per-example graphs after constructing detached perturbations.

## Expected signature

- non-target logits have low actual drift;
- target pair margins become harder but preserve label sign for capped pairs;
- target-task AUC improves over matched controls;
- other cells do not violate six-cell protection.

## Rejection

If projected gradients are nearly zero or projection does not improve over TGA at matched norm, do not keep projection merely because it looks more MDMT-specific.
