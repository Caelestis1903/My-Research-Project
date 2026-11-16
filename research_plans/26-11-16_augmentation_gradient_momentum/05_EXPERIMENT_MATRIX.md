# Experiment Matrix

Completed DESIGN and preflight PASS -> coder/basic CPU checks -> different reviewer static/CPU review -> runner real full-batch smoke -> any needed reviewer smoke signoff -> frozen formal matrix -> main analysis.

| Arm | gamma | ML new runs | KR new runs |
|---|---:|---:|---:|
| TV-GRM-005 | .05 | 5 | 5 |
| TV-GRM-010 | .10 | 5 | 5 |
| TV-GRM-020 | .20 | 5 | 5 |
| TV-GRM-SHUFFLE-010 | .10 | 5 | 5 |
| TV-GRM-INSTANT-010 | .10 | 5 | 5 |
| CLEAN / NDP-A | historical identity reuse | 0 | 0 |

Exactly seeds2022–2026, 50epochs each. 50 new trajectories, 2500epochs, 86250 complete parameter/EMA updates. CLEAN and NDP-A each ten historical trajectories, not new independent confirmation. Fix a concrete identity mismatch; do not silently add reference retraining.

Both controls are mandatory, not conditional on a positive preliminary result. All three gamma configurations complete five seeds; no1–4 seed scientific screen. Gamma=.10 is the predeclared control anchor; winning another gamma does not inherit .10 mechanism evidence. No broad alpha/lambda/beta/rho search or extra crossed control grid.

All gammas nonpositive on both datasets => this version NO-GO -> report/context -> fresh Astra. Positive mean preserves a candidate; CI crossing zero or non-unanimous seeds is not an extra veto. Test opens only after USER_STOPPED, final freeze and eligible complete selected RAM. It never opens at candidate PASS or a within-cycle freeze.
