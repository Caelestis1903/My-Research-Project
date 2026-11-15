# Algorithm and Control Specifications

## 1. Notation

A natural minibatch contains B rows, with domain d_i and two binary task labels y_it. The clean model produces probabilities p_it and per-entry probability BCE ell_it. The current clean objective is:

L_clean = (1/(2B)) sum_i sum_t ell_it.

For a selected auxiliary row-task entry j, denote its target task by t_j, label by y_j and perturbed embedding by e_j + delta_j.

## 2. Z-PAIR

- Reuse B/TGA-Pair's domain-task positive/negative pair sampler.
- Per domain-task cell, select k=min(128,n_positive,n_negative) positives and negatives without replacement.
- Preserve pair ordering, target-task IDs, labels and pair membership.
- Run the same direction-validity and pair-margin checks as B so that the zero arm follows the same adaptive acceptance policy.
- Replace accepted delta with an exact all-zero tensor; no input representation changes.
- Apply the same target-task-only auxiliary probability BCE and exact same reduction as B.
- No auxiliary loss on the other task.

## 3. Z-EXAMPLE

- Reuse C/TGA-Example's domain-task row sampler: select 2k rows uniformly without replacement from each domain, where k is determined by the same available positive/negative count cap used by C.
- Do not force 50/50 labels; record the realized labels and compare with the natural batch prior.
- Run the same near-zero BCE-gradient and embedding-RMS validity tests as C.
- Replace accepted delta with exact zero.
- Apply the same target-task-only auxiliary BCE and reduction as C.

## 4. Loss normalization and effective weights

Let M be accepted row-task entries in a minibatch. The current implementation uses:

L_aug = 2 * (1/M) * sum_j ell_aug(j)
L_total = L_clean + 0.25 * L_aug

Thus the scalar coefficient of the auxiliary mean is 0.5, not 0.25. The coefficient assigned to each accepted auxiliary entry is 0.5/M. A clean row-task entry has coefficient 1/(2B), so the ratio is B/M.

Codex must compute, not infer:
- M overall and per domain-task cell;
- positive/negative counts per cell;
- total auxiliary loss mass by domain, task and label;
- each auxiliary entry's coefficient relative to the corresponding clean entry;
- duplicate row appearances across task-specific selections;
- per-batch and epoch-aggregated effective objective weights.

Do not treat lambda=.25 as a 25% per-sample weight. Do not silently replace sum/mean conventions.

## 5. Auxiliary objective equivalence tests

- With delta=0, Z-PAIR/Z-EXAMPLE's augmented forward probabilities must match the corresponding clean forward values within strict deterministic tolerance.
- For the same model state, selected support and labels, zero-delta auxiliary BCE must equal the BCE computed on the clean embedding subset.
- The resulting total loss must equal clean loss plus the declared auxiliary term; if M=0 it must equal clean loss exactly.
- Verify the gradient decomposition numerically: clean gradient plus 0.25 times auxiliary gradient equals total gradient within tolerance.
- Confirm one backward, one Adam update and one EMA update per minibatch, exactly as source.
- No optimizer or EMA state update may occur during sensitivity extraction.
- Report gradient norm and cosine of clean vs auxiliary gradients for diagnosis only; do not use them as performance proxies or stage gates.

## 6. Stage 2 candidate designs

Stage 2 is conditional on Stage 1. Compare at most two designs initially.

### NDP-A — Natural-batch task-specific view mixture (preferred starting design)

For each original natural batch row i and task t, construct a task-specific perturbation delta_it from that task's own per-example BCE sensitivity. Keep all original rows and labels. Compute the target-task loss on its own view only:

L_NDP-A = (1/(2B)) sum_i sum_t [(1-lambda) * ell(z_t(e_i), y_it) + lambda * ell(z_t(e_i + delta_it), y_it)].

This preserves the original empirical row/task weights exactly: every original row contributes once to each task's mixture, and the total coefficient per row-task entry remains 1/(2B). The objective is a convex mixture of clean and augmented views, not an extra auxiliary mean appended to the original loss. It directly avoids the selected-subset reweighting problem.

Do not apply task t's perturbed view to the other task loss. That would add cross-task perturbation effects not needed for the initial question.

### NDP-B — Subset augmentation with inverse-probability correction

Only if full-batch per-row task perturbation is too expensive, sample a subset S_it with known inclusion probability pi_it and use Horvitz-Thompson weighting:

L_NDP-B = (1/(2B)) sum_i,t [ell_clean(i,t) + lambda * I(i in S_it)/pi_it * (ell_aug(i,t)-ell_clean(i,t))].

This estimates the same clean/augmented mixture objective without changing the expected natural empirical weights. Inclusion probabilities must be strictly positive for all eligible row-task entries and logged. Do not use self-normalized weights without stating the changed estimator. Cap extreme weights only if a predeclared cap is justified and its bias is quantified.

NDP-B is a compute-saving alternative, not the preferred method if full-batch NDP-A is feasible.

### NDP-C — Explicit convex loss mixture

The normalized mixture is:

L_NDP-C = (1-lambda) L_clean + lambda L_aug_natural,

where L_aug_natural is computed over the same natural row/task population with each row-task entry counted once. If the augmented view is task-specific, it uses the corresponding task view. This is mathematically the same target objective as NDP-A if implemented with identical per-entry weighting. It is not a distinct algorithm; use it as a clarity/implementation equivalence check, not as a third experimental arm.

Do not assume L_clean + lambda*L_aug is equivalent to a convex mixture. They are equivalent only under explicit normalization and population identity conditions. Adam is scale-sensitive through epsilon, weight decay and moment histories; preserving the intended objective and gradient scale is preferable to relying on informal scalar similarity.

## 7. Cross-task protection

Do not carry strict orthogonal projection forward by default. A had tiny non-target-logit drift, but its projected direction norm fell to about 36% (ML) and 16% (KR) of the original; A showed no AUC advantage over B. Also, logit stability is not AUC stability.

Stage 2 starts without projection. Measure per-task six-cell validation AUC/LogLoss and actual non-target logit drift. Add a protection mechanism only if:
1. unprotected NDP-A produces a meaningful non-target task loss; and
2. the protection mechanism has a plausible route to recover that loss without materially suppressing target-task perturbation.

If triggered, compare one soft trust-region penalty on the non-target task output change with no-protection at matched perturbation radius. Do not rerun a large projection/penalty grid. Strict projection is a secondary reference only if a clear measured need remains.

## 8. Algorithm assumption audit

- Task BCE sensitivity exists mathematically, but it does not imply the resulting perturbation improves AUC.
- Adversarial ascent increases local loss to create a robustness-training view; it does not directly improve ranking margin.
- Preserving another task's logit locally does not guarantee its global AUC is preserved.
- The current assumption that a separately averaged auxiliary BCE can be added at lambda=.25 without changing natural task/domain weights is false in general; actual weights depend on M and selected-cell counts.
- Uniform lambda across all cells is a design choice, not an established optimum. Avoid six free lambdas; start with one global mixture coefficient.
- Pair sampling is not inherently more aligned with AUC than per-example BCE; compare it, then prefer the simpler natural-row objective if performance is comparable.
