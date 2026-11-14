# Hyperparameters and Statistics

## 1. Frozen inherited rules

- Five seeds: 2022–2026.
- Validation six-cell Macro AUC selects the earliest strict maximum.
- Test is opened only after config and checkpoint selection are frozen.
- Main algorithm gate: paired mean Macro AUC gain at least +0.003 against strongest applicable published model; each of six cell mean AUC differences must be at least −0.001.
- Report LogLoss independently. It is not a substitute for AUC.
- Do not require all seeds to be positive. Report positive seed count.
- Report mean, sample SD, paired 95% CI, and positive seed count.
- Five seeds, not epochs, cells, examples, or pairs, are the independent statistical units.

## 2. Hyperparameter search

New values:
- alpha (embedding RMS-relative perturbation radius): {0.01, 0.03, 0.05}.
- K: fixed 128 for first search; only expand to {64, 256} if profiling or skip-rate evidence changes the design decision.
- lambda_aug: fixed 0.25 in initial search; only test {0.1, 0.25, 0.5} if augmentation is active but too weak/strong by predeclared diagnostics.
- lambda_J: numerical damping, set from stable linear algebra tolerance, not treated as a performance knob.
- kappa: cap predicted margin change; initialize 0.5 and do not search unless margin-flip rate indicates the cap is the direct cause.

Selection is validation-only, across all five seeds. Use paired seed differences against clean fixed-EMA and compare alpha choices using mean Macro AUC plus six-cell protection. Do not choose by a single seed or test.

## 3. Statistical summaries

For each seed and condition record six cell AUC/LL, Macro AUC/LL, traffic and pooled metrics. For paired differences x_s:

mean = (1/5) sum_s x_s; SD = sample standard deviation; CI95 = mean ± t(0.975,4)*SD/sqrt(5), with t=2.776445. Report all five seed differences.

Use paired comparisons because seeds share data and seed identity. Do not bootstrap examples as if they were independent training replications. Example-level bootstrap may be additional within-test uncertainty analysis, never a replacement for seed-level variation.

## 4. Comparisons

Separate:
1. total candidate vs published baseline;
2. candidate component increment vs clean fixed-EMA internal control;
3. projection increment DT-PGNA vs TGA;
4. augmentation increment vs Gaussian/random matched controls.

Do not attribute fixed EMA's existing gain to DT-PGNA. Report online and EMA shadow independently. Macro AUC gate and per-cell protection must both pass.
