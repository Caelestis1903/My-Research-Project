# Research Questions

Q1: Does task-view gradient contain useful innovation beyond clean gradient? (g_c=\nabla L_c, g_v=\nabla L_v, r=g_v-g_c).

Q2: Is any effect directional or just extra update magnitude? Compare matched-norm isotropic residual control.

Q3: Does keeping Adam's second moment clean matter? Clean (m,v) use clean gradient plus coupled WD; separate residual first moment uses (r).

Q4: Can a bounded residual channel avoid A_MATCH feedback damage? Search trust cap (gamma\in\{.05,.10,.20\}).

Q5: Does the effect differ for online and fixed EMA readouts? Report independently.

Hypotheses, not assumptions: H1 residual carries useful information; H2 clean (v_t) is a better preconditioner than combined-gradient (v_t); H3 bounded injection retains value without harmful feedback; H4 any gain is explained by magnitude/random direction/rescaling. All may be false; proxies are not AUC evidence.
