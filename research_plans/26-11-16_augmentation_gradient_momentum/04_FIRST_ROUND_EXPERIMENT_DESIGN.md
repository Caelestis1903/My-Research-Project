# First-Round Design — Task-View Gradient Residual Momentum (TV-GRM)

## Candidate update

Let L_c be source clean sample-mean probability BCE over natural 2B row-task entries. Let L_v be the mean of the two task-specific target BCE view losses, each normalized over all B rows. Use audited NDP-A task-view construction; no pair sampling, reweighting or extra loss mass.

Compute both gradients at identical theta_t:

```text
g_c = grad_theta L_c
g_v = grad_theta L_v
r   = g_v - g_c
```

Preserve source coupled weight decay exactly once in the clean Adam gradient:

```text
g_c_wd = g_c + weight_decay * theta
m_c[t] = beta1 * m_c[t-1] + (1-beta1) * g_c_wd
v_c[t] = beta2 * v_c[t-1] + (1-beta2) * square(g_c_wd)
q[t]   = beta1 * q[t-1] + (1-beta1) * r
```

There is no weight decay in q. Apply bias correction using one consistent step counter:

```text
u_c = mhat_c / (sqrt(vhat_c) + eps)
u_r = qhat   / (sqrt(vhat_c) + eps)
s   = min(1, gamma * norm2(u_c) / (norm2(u_r) + eps))
delta_theta = -lr * (u_c + s * u_r)
```

The cap is global across trainable parameters, not domain-specific. Gamma values are .05/.10/.20. SHUFFLE uses a deterministic parameter-space Rademacher vector matched to norm2(r), then the same q/preconditioner/cap at gamma=.10.

## Training order

Same initialization/RNG as CLEAN. Compute clean and view gradients at the same parameters; update clean m/v and residual q; apply exactly one parameter update; then one after-Adam EMA update. No second optimizer or microbatch update.

## Critical implementation conditions

Preserve source grad=None semantics: if clean Adam state is undefined for a parameter, do not create a residual-only update. Weight decay is never applied to residual. Gamma=0 must reproduce source clean Adam model/optimizer/EMA/RNG/loader states across multiple steps and resume.

## Distinction from NDP-A

Adam on a weighted clean/view loss updates both moments from the combined gradient. TV-GRM keeps v_c tied to the clean objective and stores view-minus-clean separately. This changes the optimizer recurrence, not just the loss weight. Primary contrast is selected fixed-EMA validation Macro AUC vs CLEAN fixed EMA. Also report online and total vs CLEAN-online.
