# Stage E3 Main Scientific Analysis

## Integrity

All 60 joint runs and 36 frozen oracle references are formal-valid. Oracle coverage is 360/360 cells; all 12 five-alpha blocks are VALID. Seed is the uncertainty cluster; tasks, cells, epochs, and rotations are not treated as independent seed replicates.

## Core response curves

|   alpha |   L_transfer |   M_transfer |   S_transfer |   overall_auc |   worst_domain_auc |   negative_transfer_rate |   macro_logloss |
|--------:|-------------:|-------------:|-------------:|--------------:|-------------------:|-------------------------:|----------------:|
|    0    |    -0.015545 |     0.018645 |     0.059787 |      0.70976  |           0.672158 |                 0.347222 |        0.639895 |
|    0.25 |    -0.009951 |     0.024264 |     0.082749 |      0.721152 |           0.694055 |                 0.347222 |        0.573493 |
|    0.5  |    -0.005297 |     0.028624 |     0.093856 |      0.727859 |           0.705434 |                 0.291667 |        0.549035 |
|    0.75 |    -0.000597 |     0.028693 |     0.095421 |      0.72997  |           0.705063 |                 0.138889 |        0.556826 |
|    1    |    -0.00121  |     0.030075 |     0.097276 |      0.730845 |           0.706218 |                 0.222222 |        0.552124 |

## Endpoint evidence (alpha=1 improvement over alpha=0)

| metric                 |   mean_improvement |   ci_low |   ci_high |   positive_blocks |   positive_seeds | evidence   |
|:-----------------------|-------------------:|---------:|----------:|------------------:|-----------------:|:-----------|
| overall_auc            |           0.021085 | 0.014999 |  0.026691 |                12 |                4 | STRONG     |
| worst_domain_auc       |           0.03406  | 0.024469 |  0.046212 |                11 |                4 | STRONG     |
| negative_transfer_rate |           0.125    | 0.069444 |  0.194444 |                 5 |                4 | MEDIUM     |
| macro_logloss          |           0.087772 | 0.07284  |  0.098617 |                12 |                4 | STRONG     |
| L_transfer             |           0.014335 | 0.008391 |  0.020279 |                12 |                4 | STRONG     |
| L_auc                  |           0.014335 | 0.008391 |  0.020279 |                12 |                4 | STRONG     |
| M_transfer             |           0.01143  | 0.006642 |  0.015115 |                12 |                4 | STRONG     |
| M_auc                  |           0.01143  | 0.006642 |  0.015115 |                12 |                4 | STRONG     |
| S_transfer             |           0.037488 | 0.027944 |  0.050117 |                11 |                4 | STRONG     |
| S_auc                  |           0.037488 | 0.027944 |  0.050117 |                11 |                4 | STRONG     |

## Approximate monotonicity

| metric                 |   endpoint_improvement |   reverse_steps_over_margin |   positive_rho_blocks |   median_rho | approximately_monotonic   |
|:-----------------------|-----------------------:|----------------------------:|----------------------:|-------------:|:--------------------------|
| overall_auc            |               0.021085 |                           0 |                    12 |     0.9      | True                      |
| worst_domain_auc       |               0.03406  |                           0 |                    11 |     0.9      | True                      |
| negative_transfer_rate |               0.125    |                           1 |                    10 |     0.568089 | False                     |
| macro_logloss          |               0.087772 |                           1 |                    12 |     0.7      | False                     |
| L_transfer             |               0.014335 |                           0 |                    12 |     0.9      | True                      |
| L_auc                  |               0.014335 |                           0 |                    12 |     0.9      | True                      |
| M_transfer             |               0.01143  |                           0 |                    12 |     0.7      | True                      |
| M_auc                  |               0.01143  |                           0 |                    12 |     0.7      | True                      |
| S_transfer             |               0.037488 |                           0 |                    11 |     0.9      | True                      |
| S_auc                  |               0.037488 |                           0 |                    11 |     0.9      | True                      |

## Fixed interior comparisons

|   alpha |   overall_vs_0 |   ci_low_vs_0 |   ci_high_vs_0 |   positive_blocks_vs_0 |   overall_vs_1 |   ci_low_vs_1 |   ci_high_vs_1 |   positive_blocks_vs_1 |   worst_vs_best_endpoint |   holm_p_vs_0 |   holm_p_vs_1 |
|--------:|---------------:|--------------:|---------------:|-----------------------:|---------------:|--------------:|---------------:|-----------------------:|-------------------------:|--------------:|--------------:|
|    0.25 |       0.011391 |      0.007194 |       0.013975 |                     11 |      -0.009693 |     -0.012732 |      -0.006939 |                      0 |                -0.012462 |      0.062814 |      0.062814 |
|    0.5  |       0.018099 |      0.012546 |       0.024002 |                     12 |      -0.002986 |     -0.004117 |      -0.002047 |                      1 |                -0.001082 |      0.062814 |      0.062814 |
|    0.75 |       0.02021  |      0.013625 |       0.025755 |                     12 |      -0.000875 |     -0.001436 |      -0.0001   |                      5 |                -0.001453 |      0.062814 |      0.125933 |

## Large/Small trade-off checks

| metric   |   alpha | role   |     mean |   ci_low |   ci_high |   positive_blocks |
|:---------|--------:|:-------|---------:|---------:|----------:|------------------:|
| transfer |    0.25 | L      | 0.005593 | 0.001888 |  0.009018 |                10 |
| transfer |    0.25 | S      | 0.022962 | 0.017271 |  0.028308 |                11 |
| transfer |    0.5  | L      | 0.010248 | 0.005662 |  0.014834 |                 9 |
| transfer |    0.5  | S      | 0.034069 | 0.022039 |  0.046778 |                11 |
| transfer |    0.75 | L      | 0.014948 | 0.008697 |  0.021199 |                11 |
| transfer |    0.75 | S      | 0.035634 | 0.024318 |  0.047658 |                10 |
| transfer |    1    | L      | 0.014335 | 0.008391 |  0.020279 |                12 |
| transfer |    1    | S      | 0.037488 | 0.027944 |  0.050117 |                11 |
| auc      |    0.25 | L      | 0.005593 | 0.001888 |  0.009018 |                10 |
| auc      |    0.25 | S      | 0.022962 | 0.017271 |  0.028308 |                11 |
| auc      |    0.5  | L      | 0.010248 | 0.005662 |  0.014834 |                 9 |
| auc      |    0.5  | S      | 0.034069 | 0.022039 |  0.046778 |                11 |
| auc      |    0.75 | L      | 0.014948 | 0.008697 |  0.021199 |                11 |
| auc      |    0.75 | S      | 0.035634 | 0.024318 |  0.047658 |                10 |
| auc      |    1    | L      | 0.014335 | 0.008391 |  0.020279 |                12 |
| auc      |    1    | S      | 0.037488 | 0.027944 |  0.050117 |                11 |

## Role/task validation-selected alpha (leave-one-seed-out)

| size_role   |   task |   chosen_alpha |   count |   mean_gain_vs_alpha1 |
|:------------|-------:|---------------:|--------:|----------------------:|
| L           |      0 |           1    |       4 |              0        |
| L           |      1 |           0.75 |       3 |              0.000841 |
| L           |      1 |           1    |       1 |              0        |
| M           |      0 |           1    |       4 |              0        |
| M           |      1 |           0.75 |       1 |             -0.007381 |
| M           |      1 |           1    |       3 |              0        |
| S           |      0 |           1    |       4 |              0        |
| S           |      1 |           1    |       4 |              0        |

## Pareto analysis

|   alpha |   L_transfer |   M_transfer |   S_transfer |   overall_auc |   worst_domain_auc |   negative_transfer_rate |   macro_logloss | on_mean_frontier   |   nondominated_probability |
|--------:|-------------:|-------------:|-------------:|--------------:|-------------------:|-------------------------:|----------------:|:-------------------|---------------------------:|
|    0    |    -0.015545 |     0.018645 |     0.059787 |      0.70976  |           0.672158 |                 0.347222 |        0.639895 | False              |                     0      |
|    0.25 |    -0.009951 |     0.024264 |     0.082749 |      0.721152 |           0.694055 |                 0.347222 |        0.573493 | False              |                     0      |
|    0.5  |    -0.005297 |     0.028624 |     0.093856 |      0.727859 |           0.705434 |                 0.291667 |        0.549035 | True               |                     0.6836 |
|    0.75 |    -0.000597 |     0.028693 |     0.095421 |      0.72997  |           0.705063 |                 0.138889 |        0.556826 | True               |                     0.9356 |
|    1    |    -0.00121  |     0.030075 |     0.097276 |      0.730845 |           0.706218 |                 0.222222 |        0.552124 | True               |                     1      |

## Temporal validation-selected alpha

| phase   |   chosen_alpha |   block_count |   mean_gain_vs_alpha1 |
|:--------|---------------:|--------------:|----------------------:|
| early   |           0    |             2 |              0.029293 |
| early   |           0.25 |             3 |              0.017317 |
| early   |           0.5  |             1 |              0.039259 |
| early   |           0.75 |             6 |              0.007536 |
| late    |           0.75 |             1 |             -0.0012   |
| late    |           1    |            11 |              0        |
| middle  |           0.75 |             5 |             -0.000416 |
| middle  |           1    |             7 |              0        |

Temporal consistency summary:

|   early_mean_gain_vs_alpha1 |   ci_low |   ci_high |   early_positive_blocks |   early_over_margin_blocks |   early_subalpha1_blocks |   late_alpha1_blocks |   positive_seed_averages |
|----------------------------:|---------:|----------:|------------------------:|---------------------------:|-------------------------:|---------------------:|-------------------------:|
|                    0.016251 | 0.006636 |  0.027937 |                      11 |                         10 |                       12 |                   11 |                        4 |

## Physical-domain/task curves

|   alpha |   physical_domain |   task |   transfer |      auc |   logloss |
|--------:|------------------:|-------:|-----------:|---------:|----------:|
|    0    |                 0 |      0 |   0.029624 | 0.712151 |  0.769357 |
|    0    |                 0 |      1 |   0.022443 | 0.720512 |  0.524709 |
|    0    |                 1 |      0 |   0.022012 | 0.709921 |  0.783359 |
|    0    |                 1 |      1 |   0.024833 | 0.733264 |  0.526177 |
|    0    |                 2 |      0 |   0.009365 | 0.678545 |  0.709413 |
|    0    |                 2 |      1 |   0.017499 | 0.704168 |  0.526356 |
|    0.25 |                 0 |      0 |   0.040368 | 0.722896 |  0.689434 |
|    0.25 |                 0 |      1 |   0.036292 | 0.734361 |  0.458778 |
|    0.25 |                 1 |      0 |   0.036544 | 0.724453 |  0.672702 |
|    0.25 |                 1 |      1 |   0.039765 | 0.748196 |  0.431078 |
|    0.25 |                 2 |      0 |   0.015671 | 0.684851 |  0.684106 |
|    0.25 |                 2 |      1 |   0.025483 | 0.712153 |  0.504863 |
|    0.5  |                 0 |      0 |   0.045953 | 0.728481 |  0.65699  |
|    0.5  |                 0 |      1 |   0.041683 | 0.739752 |  0.44627  |
|    0.5  |                 1 |      0 |   0.043414 | 0.731323 |  0.634802 |
|    0.5  |                 1 |      1 |   0.044496 | 0.752927 |  0.425514 |
|    0.5  |                 2 |      0 |   0.0285   | 0.697679 |  0.650446 |
|    0.5  |                 2 |      1 |   0.030321 | 0.71699  |  0.480189 |
|    0.75 |                 0 |      0 |   0.050144 | 0.732671 |  0.652985 |
|    0.75 |                 0 |      1 |   0.044645 | 0.742714 |  0.44703  |
|    0.75 |                 1 |      0 |   0.047773 | 0.735682 |  0.634188 |
|    0.75 |                 1 |      1 |   0.047762 | 0.756193 |  0.428837 |
|    0.75 |                 2 |      0 |   0.028054 | 0.697234 |  0.667557 |
|    0.75 |                 2 |      1 |   0.028658 | 0.715327 |  0.510357 |
|    1    |                 0 |      0 |   0.051358 | 0.733886 |  0.635524 |
|    1    |                 0 |      1 |   0.045317 | 0.743387 |  0.447609 |
|    1    |                 1 |      0 |   0.046616 | 0.734525 |  0.654494 |
|    1    |                 1 |      1 |   0.046741 | 0.755172 |  0.434348 |
|    1    |                 2 |      0 |   0.030496 | 0.699676 |  0.648918 |
|    1    |                 2 |      1 |   0.031754 | 0.718423 |  0.49185  |

## Scientific assessment

### Stable repeated phenomena

1. **STRONG — increasing alpha improves endpoint performance.** Alpha=1 versus alpha=0 improves overall AUC by 0.021085 (seed-cluster 95% CI 0.014999 to 0.026691; 12/12 blocks), worst-domain AUC by 0.034060 (CI 0.024469 to 0.046212; 11/12), macro LogLoss by 0.087772, and Large/Medium/Small transfer by 0.014335/0.011430/0.037488. All four seed rotation-averages agree.
2. **STRONG — the AUC/transfer response is approximately monotonic.** Overall, worst-domain, and all role AUC/transfer curves have no adjacent reversal larger than 0.001; blockwise alpha Spearman direction is positive in 11–12 of 12 blocks.
3. **STRONG — shared optimization allocation follows alpha continuously.** L shared magnitude share rises 0.164→0.712, S falls 0.642→0.062, and mismatch falls 0.653→0.111. Raw gradient norms do not follow the same mechanical curve, so coefficient-adjusted contribution—not raw norm alone—tracks the intervention.
4. **STRONG — no stable Large-gain/Small-loss trade-off.** Relative to alpha=0, both Large and Small AUC/transfer improve at every nonzero alpha in aggregate; the endpoint improvements have CIs excluding zero and 11–12/12 block consistency.

### Contradictions and competing explanations

- Negative-transfer rate and macro LogLoss are non-monotonic: alpha=0.75 has the lowest negative-transfer rate (0.139 versus 0.222 at alpha=1), while alpha=0.5 has the lowest macro LogLoss (0.549 versus 0.552 at alpha=1). Thus alpha=1 is not literally best on every core metric.
- No fixed interior alpha passes the preregistered superiority gate. Alpha=0.75 is practically equivalent to alpha=1 on overall AUC (-0.000875), but not superior; alpha=0.5 and 0.25 are materially worse than alpha=1. Holm-adjusted tests likewise do not establish an interior winner.
- The epsilon-Pareto mean frontier contains alpha=0.5, 0.75, and 1.0. Only alpha=1 reaches nondominated probability 1.0; alpha=0.75 reaches 0.936, below the 0.95 clear-dominance threshold. The frontier is therefore real but does not justify a single fixed interior optimum.
- Leave-one-seed-out role/task selection mostly chooses alpha=1. The only repeated alternative is Large task 1 choosing alpha=0.75 in 3/4 held-out seeds, but its mean gain over alpha=1 is only 0.000841, below the practical margin. **Static role/task heterogeneity evidence is WEAK.**
- Physical-domain curves differ in level, but the positive alpha response rotates with L/M/S roles and repeats across physical domains. **MEDIUM evidence** favors size role over a fixed-domain-only explanation; oracle coverage is complete, although size and intrinsic difficulty remain correlated observational factors.
- Checkpoint epochs vary, but all five arms are compared within formal blocks and common support. The monotonic endpoint pattern across rotations/seeds is too stable for a checkpoint-only explanation; **MEDIUM evidence against checkpoint selection as the main cause.**

### Temporal diagnostic and final Go/Stop verdict

Early common-support validation selects alpha<1 in 12/12 blocks; late training selects alpha=1 in 11/12. Validation-selected early choices improve held-block test AUC over alpha=1 by 0.016251 on average (seed-cluster CI 0.006636 to 0.027937), with 11/12 positive blocks, 10/12 above 0.001, and all 4 seed averages positive. The precise early alpha varies from 0 to 0.75, so this supports a changing region rather than one universal early value.

**Final verdict: Temporal weighting justified — STRONG diagnostic evidence.** Static alpha=1 is the best or practically equivalent choice for final selected-checkpoint AUC and has the only 1.0 Pareto nondominance probability, while no fixed interior alpha or role/task-specific rule clears its gate. However, the stable early-to-late shift clears the design's motivation criterion for a separate temporal causal experiment. This does not prove Curriculum Learning works; it authorizes a falsification-oriented schedule experiment comparing a prespecified low/intermediate→1 schedule against static alpha=1 and matched exposure controls.

The next experiment should predefine the schedule from validation-only aggregate evidence, include a reverse schedule and static alpha=1/0.75 controls, match integrated coefficient exposure and optimizer steps, and test whether any gain survives common-support and all four seeds. A failure would falsify the temporal interpretation despite the diagnostic shift.

## Reproducibility

Tables are in `experiments/stage_E3_analysis_tables/`; required figures are in `experiments/stage_E3_analysis_figures/`. Practical equivalence margins are 0.001 for AUC/transfer and LogLoss. Holm-adjusted p-values are reported for the six fixed interior-vs-endpoint overall-AUC contrasts.
