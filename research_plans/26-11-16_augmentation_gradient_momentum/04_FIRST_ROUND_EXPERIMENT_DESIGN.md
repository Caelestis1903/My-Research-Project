# First-Round Design — Task-View Gradient Residual Momentum (TV-GRM)

Let (L_c) be source clean sample-mean probability BCE over natural (2B) row-task entries. Let (L_v) be the mean of two task-specific target BCE view losses, each normalized over all B rows. Use audited NDP-A task-view construction; no pair sampling, reweighting or extra loss mass.

Compute at identical (	heta_t):
[
g_c=\nabla_\theta L_c,\quad g_v=\nabla_\theta L_v,\quad r=g_v-g_c.
]
Preserve coupled weight decay once in clean Adam:
[
g_c^{wd}=g_c+\lambda_{wd}\theta,quad
m_t^c=\beta_1m_{t-1}^c+(1-\beta_1)g_{c,t}^{wd},
]
[
v_t^c=\beta_2v_{t-1}^c+(1-\beta_2)(g_{c,t}^{wd})^2,quad
q_t=\beta_1q_{t-1}+(1-\beta_1)r_t.
]
No WD in q. Apply bias correction with one step count:
[
u_c=\hat m_t^c/(\sqrt{\hat v_t^c}+\epsilon),quad
u_r=\hat q_t/(\sqrt{\hat v_t^c}+\epsilon).
]
Global cap and update:
[
s_t=\min(1,\gamma\|u_c\|_2/(\|u_r\|_2+\epsilon)),quad
\Delta\theta_t=-\eta(u_c+s_tu_r).
]
Cap is global over trainable parameters, not domain-specific. Gamma values .05/.10/.20. SHUFFLE uses deterministic parameter-space Rademacher vector matched to (|r_t|_2), same q/preconditioner/cap at gamma=.10.

Training order: same initialization/RNG as CLEAN; compute clean and view gradients at same parameters; update clean m/v and residual q; one parameter update; one after-Adam EMA update. No second optimizer or microbatch update.

Preserve source grad=None semantics: if clean Adam state is undefined for a parameter, do not create residual-only update. Weight decay is never applied to residual. Gamma=0 must reproduce source clean Adam model/optimizer/EMA/RNG/loader states across multiple steps and resume.

This is not Adam on a weighted loss: ordinary mixture updates both moments from combined gradient; TV-GRM keeps v tied to clean objective and stores view-minus-clean separately. Primary contrast is selected fixed-EMA validation Macro AUC vs CLEAN fixed EMA. Also report online and total vs CLEAN-online.
