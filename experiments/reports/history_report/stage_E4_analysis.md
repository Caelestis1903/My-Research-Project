# Stage E4 Main Scientific Analysis

## Integrity

All 12 joint runs and 12 matched split-first/filter-second oracle runs are FORMAL_VALID. Oracle coverage is 72/72 cells; all four three-arm blocks are VALID. Seed (n=4) is the replication unit.

## Aggregate curves

| weighting   |   overall_auc |   overall_logloss |   worst_domain_auc |   pooled_auc |   pooled_logloss |   negative_transfer_rate |
|:------------|--------------:|------------------:|-------------------:|-------------:|-----------------:|-------------------------:|
| equal       |      0.650736 |          0.451315 |           0.637057 |     0.69489  |         0.438569 |                 0.333333 |
| sample      |      0.670967 |          0.44105  |           0.645462 |     0.727038 |         0.431637 |                 0.333333 |
| sqrt        |      0.669982 |          0.440119 |           0.647821 |     0.717077 |         0.429644 |                 0.333333 |

## Domain curves

| weighting   |   domain |   seed |   transfer |      auc |   logloss |   negative_transfer_rate |   magnitude_share |   signed_contribution |   mismatch |   raw_gradient |   weighted_gradient |
|:------------|---------:|-------:|-----------:|---------:|----------:|-------------------------:|------------------:|----------------------:|-----------:|---------------:|--------------------:|
| equal       |        0 | 2023.5 |   0.084507 | 0.637057 |  0.237663 |                        0 |          0.206641 |              0.049896 |   0.710588 |       0.02278  |            0.02278  |
| equal       |        1 | 2023.5 |  -0.043677 | 0.669865 |  0.438887 |                        1 |          0.166417 |              0.042231 |   0.710588 |       0.017395 |            0.017395 |
| equal       |        2 | 2023.5 |   0.029135 | 0.645287 |  0.677395 |                        0 |          0.626942 |              0.407873 |   0.710588 |       0.069791 |            0.069791 |
| sample      |        0 | 2023.5 |   0.092912 | 0.645462 |  0.238096 |                        0 |          0.094458 |              0.007821 |   0.083883 |       0.037918 |            0.009797 |
| sample      |        1 | 2023.5 |  -0.005584 | 0.707957 |  0.43201  |                        1 |          0.773299 |              0.475058 |   0.083883 |       0.055786 |            0.145483 |
| sample      |        2 | 2023.5 |   0.04333  | 0.659482 |  0.653045 |                        0 |          0.132244 |              0.017121 |   0.083883 |       0.101801 |            0.013618 |
| sqrt        |        0 | 2023.5 |   0.097074 | 0.649624 |  0.237275 |                        0 |          0.155378 |              0.025007 |   0.30778  |       0.032207 |            0.019733 |
| sqrt        |        1 | 2023.5 |  -0.017666 | 0.695876 |  0.429989 |                        1 |          0.553078 |              0.34172  |   0.30778  |       0.046399 |            0.090315 |
| sqrt        |        2 | 2023.5 |   0.048295 | 0.664447 |  0.653093 |                        0 |          0.291544 |              0.133273 |   0.30778  |       0.099183 |            0.043724 |

## Fixed paired contrasts

| contrast     | metric                 |   improvement |    ci_low |   ci_high |   positive_seeds |   over_margin_seeds | evidence   |
|:-------------|:-----------------------|--------------:|----------:|----------:|-----------------:|--------------------:|:-----------|
| sample-equal | overall_auc            |      0.020231 |  0.017081 |  0.022865 |                4 |                   4 | STRONG     |
| sample-equal | worst_domain_auc       |      0.008405 |  0.002171 |  0.016349 |                4 |                   3 | STRONG     |
| sample-equal | pooled_auc             |      0.032148 |  0.031076 |  0.033428 |                4 |                   4 | STRONG     |
| sample-equal | overall_logloss        |      0.010265 |  0.005557 |  0.016673 |                4 |                   4 | STRONG     |
| sample-equal | negative_transfer_rate |      0        |  0        |  0        |                0 |                   0 | WEAK       |
| sample-equal | d0_transfer            |      0.008405 |  0.002171 |  0.016349 |                4 |                   3 | STRONG     |
| sample-equal | d1_transfer            |      0.038093 |  0.036481 |  0.039768 |                4 |                   4 | STRONG     |
| sample-equal | d2_transfer            |      0.014195 |  0.008125 |  0.024139 |                4 |                   4 | STRONG     |
| sqrt-equal   | overall_auc            |      0.019246 |  0.01581  |  0.023389 |                4 |                   4 | STRONG     |
| sqrt-equal   | worst_domain_auc       |      0.010765 |  0.005411 |  0.016975 |                4 |                   4 | STRONG     |
| sqrt-equal   | pooled_auc             |      0.022187 |  0.019323 |  0.024714 |                4 |                   4 | STRONG     |
| sqrt-equal   | overall_logloss        |      0.011196 |  0.003741 |  0.018651 |                4 |                   3 | STRONG     |
| sqrt-equal   | negative_transfer_rate |      0        |  0        |  0        |                0 |                   0 | WEAK       |
| sqrt-equal   | d0_transfer            |      0.012567 |  0.005411 |  0.022383 |                4 |                   4 | STRONG     |
| sqrt-equal   | d1_transfer            |      0.026011 |  0.022177 |  0.02983  |                4 |                   4 | STRONG     |
| sqrt-equal   | d2_transfer            |      0.01916  |  0.009451 |  0.032203 |                4 |                   4 | STRONG     |
| sqrt-sample  | overall_auc            |     -0.000985 | -0.003953 |  0.001984 |                2 |                   2 | WEAK       |
| sqrt-sample  | worst_domain_auc       |      0.00236  | -0.000755 |  0.005475 |                3 |                   2 | MEDIUM     |
| sqrt-sample  | pooled_auc             |     -0.009961 | -0.012818 | -0.007543 |                0 |                   0 | WEAK       |
| sqrt-sample  | overall_logloss        |      0.000931 | -0.003848 |  0.006998 |                2 |                   2 | WEAK       |
| sqrt-sample  | negative_transfer_rate |      0        |  0        |  0        |                0 |                   0 | WEAK       |
| sqrt-sample  | d0_transfer            |      0.004163 | -0.000377 |  0.00739  |                3 |                   3 | MEDIUM     |
| sqrt-sample  | d1_transfer            |     -0.012082 | -0.015399 | -0.008714 |                0 |                   0 | WEAK       |
| sqrt-sample  | d2_transfer            |      0.004965 | -0.000359 |  0.01029  |                2 |                   2 | WEAK       |

## Equal dominant-domain degradation

|   mean_transfer |    ci_low |   ci_high |   negative_seeds |
|----------------:|----------:|----------:|-----------------:|
|       -0.043677 | -0.046146 | -0.041207 |                4 |

## Domain/task contrasts

|   domain |   task | contrast           |      mean |    ci_low |   ci_high |   positive_seeds |
|---------:|-------:|:-------------------|----------:|----------:|----------:|-----------------:|
|        0 |      0 | sample_minus_equal | -0.002879 | -0.008301 |  0.002808 |                1 |
|        0 |      0 | sqrt_minus_equal   |  0.001162 | -0.002637 |  0.004961 |                2 |
|        0 |      0 | sqrt_minus_sample  |  0.004041 | -0.004304 |  0.008661 |                3 |
|        0 |      1 | sample_minus_equal |  0.019688 |  0.002214 |  0.036389 |                3 |
|        0 |      1 | sqrt_minus_equal   |  0.023973 |  0.010824 |  0.040872 |                4 |
|        0 |      1 | sqrt_minus_sample  |  0.004284 | -0.007315 |  0.015033 |                3 |
|        1 |      0 | sample_minus_equal |  0.038583 |  0.037183 |  0.040279 |                4 |
|        1 |      0 | sqrt_minus_equal   |  0.026873 |  0.023063 |  0.030449 |                4 |
|        1 |      0 | sqrt_minus_sample  | -0.01171  | -0.014825 | -0.008384 |                0 |
|        1 |      1 | sample_minus_equal |  0.037602 |  0.03578  |  0.039453 |                4 |
|        1 |      1 | sqrt_minus_equal   |  0.025148 |  0.021087 |  0.02921  |                4 |
|        1 |      1 | sqrt_minus_sample  | -0.012454 | -0.01595  | -0.009067 |                0 |
|        2 |      0 | sample_minus_equal |  0.012939 |  0.005437 |  0.02303  |                4 |
|        2 |      0 | sqrt_minus_equal   |  0.018901 |  0.008446 |  0.035828 |                4 |
|        2 |      0 | sqrt_minus_sample  |  0.005963 |  0.000164 |  0.011762 |                3 |
|        2 |      1 | sample_minus_equal |  0.015451 |  0.010351 |  0.025077 |                4 |
|        2 |      1 | sqrt_minus_equal   |  0.019419 |  0.009813 |  0.029025 |                4 |
|        2 |      1 | sqrt_minus_sample  |  0.003968 | -0.000882 |  0.009458 |                2 |

## Common-support checks

| contrast     | metric             |   improvement |   ci_low |   ci_high |   positive_seeds |   over_margin_seeds | evidence   |
|:-------------|:-------------------|--------------:|---------:|----------:|-----------------:|--------------------:|:-----------|
| sample-equal | mean_macro_auc     |      0.013173 | 0.010396 |  0.01783  |                4 |                   4 | STRONG     |
| sqrt-equal   | mean_macro_auc     |      0.01231  | 0.008987 |  0.015632 |                4 |                   4 | STRONG     |
| sample-equal | last_macro_auc     |      0.032488 | 0.026738 |  0.039301 |                4 |                   4 | STRONG     |
| sqrt-equal   | last_macro_auc     |      0.021062 | 0.018121 |  0.02316  |                4 |                   4 | STRONG     |
| sample-equal | mean_macro_logloss |      0.013457 | 0.010168 |  0.019554 |                4 |                   4 | STRONG     |
| sqrt-equal   | mean_macro_logloss |      0.009326 | 0.006211 |  0.013979 |                4 |                   4 | STRONG     |
| sample-equal | mean_pooled_auc    |      0.021084 | 0.020037 |  0.021996 |                4 |                   4 | STRONG     |
| sqrt-equal   | mean_pooled_auc    |      0.017867 | 0.017036 |  0.018699 |                4 |                   4 | STRONG     |

## Shared contribution

| weighting   |   domain |   magnitude_share |   signed_contribution |   mismatch |   raw_gradient |   weighted_gradient |
|:------------|---------:|------------------:|----------------------:|-----------:|---------------:|--------------------:|
| equal       |        0 |          0.206641 |              0.049896 |   0.710588 |       0.02278  |            0.02278  |
| equal       |        1 |          0.166417 |              0.042231 |   0.710588 |       0.017395 |            0.017395 |
| equal       |        2 |          0.626942 |              0.407873 |   0.710588 |       0.069791 |            0.069791 |
| sample      |        0 |          0.094458 |              0.007821 |   0.083883 |       0.037918 |            0.009797 |
| sample      |        1 |          0.773299 |              0.475058 |   0.083883 |       0.055786 |            0.145483 |
| sample      |        2 |          0.132244 |              0.017121 |   0.083883 |       0.101801 |            0.013618 |
| sqrt        |        0 |          0.155378 |              0.025007 |   0.30778  |       0.032207 |            0.019733 |
| sqrt        |        1 |          0.553078 |              0.34172  |   0.30778  |       0.046399 |            0.090315 |
| sqrt        |        2 |          0.291544 |              0.133273 |   0.30778  |       0.099183 |            0.043724 |

## Dataset shift

|   domain |   task |   domain_share |   train_sample_count |   validation_sample_count |   test_sample_count |   train_positive_rate |   validation_positive_rate |   test_positive_rate |
|---------:|-------:|---------------:|---------------------:|--------------------------:|--------------------:|----------------------:|---------------------------:|---------------------:|
|        0 |      0 |       0.086127 |      172782          |                      2692 |                2613 |              0.090889 |                   0.09101  |             0.087256 |
|        1 |      0 |       0.869284 |           1.7439e+06 |                    246232 |              246284 |              0.377832 |                   0.209737 |             0.210517 |
|        2 |      0 |       0.044589 |       89452          |                      1843 |                1870 |              0.616867 |                   0.576777 |             0.588235 |
|        0 |      1 |       0.086127 |      172782          |                      2692 |                2613 |              0.041214 |                   0.037519 |             0.045924 |
|        1 |      1 |       0.869284 |           1.7439e+06 |                    246232 |              246284 |              0.256083 |                   0.111521 |             0.111315 |
|        2 |      1 |       0.044589 |       89452          |                      1843 |                1870 |              0.484807 |                   0.44981  |             0.442246 |

## Scientific assessment

1. **STRONG — equal weighting produces stable dominant-domain degradation.** d1 matched transfer is negative for both tasks and all four seeds; the domain mean is about -0.0437, far beyond the 0.001 margin.
2. **STRONG — Sample repairs most d1 degradation and improves global metrics.** Sample-minus-Equal d1 transfer is about +0.0381, overall macro AUC +0.0202, pooled AUC +0.0321, worst-domain AUC +0.0084, and macro LogLoss improves by about 0.0103. The AUC directions repeat across all four seeds. Negative-transfer rate does not fall because both d1 tasks remain slightly below their single-domain oracle in every seed.
3. **STRONG — the contribution mechanism transfers directionally.** d1 shared magnitude share rises from about 0.166 (Equal) through 0.553 (Sqrt) to 0.773 (Sample), while global mismatch falls from 0.711 to 0.308 to 0.084. This matches the MovieLens coefficient/contribution response despite different physical-domain composition.
4. **No stable Large/minority sacrifice at domain aggregate level.** Sample improves d1 and d2 transfer and slightly improves d0 domain-average transfer. However, d0 task 0 declines by about 0.0029 in 3/4 seeds while d0 task 1 improves strongly; this is a real task-specific contradiction with MEDIUM evidence, not a domain-wide reversal.
5. **Sqrt is not a fixed winner.** It improves Equal strongly, gives the best mean worst-domain AUC and macro LogLoss, but is practically tied with Sample on overall AUC and is worse on d1 transfer and pooled AUC. It does not satisfy superiority over both endpoints.
6. **Checkpoint-only explanation is weakened.** Sample/Sqrt improvements persist in mean and last common-support epochs. Macro and pooled AUC move in the same direction, although pooled gains are larger because test is about 98.2% d1.
7. **Temporal/data shift remains a strong competing explanation for residual error.** Even Sample leaves d1 transfer near -0.0056 and negative-transfer rate unchanged at 1/3. Train→test domain-mixture TV is 0.112839, and d1 label priors fall sharply for both tasks. E4 cannot causally attribute that residual to time shift.

## Cross-dataset verdict

**Supported — STRONG evidence.** The preregistered Supported gate is met: Sample consistently improves d1 transfer, macro/pooled and worst-domain AUC, does not increase negative-transfer rate, changes shared contribution in the MovieLens direction, and causes no stable domain-aggregate minority loss. The d0 task-0 exception and persistent d1 oracle gap limit universality but do not form a stable dataset-level reversal; they motivate task heterogeneity and temporal-shift studies.

The strongest falsification-oriented next step is a separate temporal/data-shift causal design: static Sample versus a prespecified schedule, a reverse schedule, and matched integrated exposure, evaluated on both chronological and shift-controlled splits. E4 itself does not prove a temporal method or adaptive controller effective.

## Reproducibility

Detailed seed-level tables are under `experiments/stage_E4_analysis_tables/`; plots are under `experiments/stage_E4_analysis_figures/`.
