# Research Questions

Q1: Does the task-view gradient contain useful innovation beyond the clean gradient?

```text
g_c = grad_theta L_c
g_v = grad_theta L_v
r   = g_v - g_c
```

Q2: Is any effect directional, or merely extra update magnitude? Compare with a matched-norm isotropic residual control.

Q3: Does keeping Adam's second moment clean matter? Clean m/v use clean gradient plus coupled weight decay; a separate residual first moment uses r.

Q4: Can a bounded residual channel avoid A_MATCH feedback damage? Test trust caps gamma in {0.05, 0.10, 0.20}.

Q5: Does the effect differ for online and fixed EMA readouts? Report both independently.

Hypotheses, not assumptions:
- H1: residuals contain useful information not represented in clean gradient state.
- H2: clean v_t is a better preconditioner than a second moment built from combined gradients.
- H3: bounded injection retains local value without harmful persistent feedback.
- H4: any gain is explained by magnitude, random direction, or clean-gradient rescaling.

All may be false. Proxies are not AUC evidence.
