# Stage E1 Main Scientific Analysis

## Scope and data integrity

All 24 formal runs completed and all equal/sample pairs were VALID. Analysis uses seed-level paired comparisons; positive AUC delta means sample weighting is better, while positive LogLoss improvement means sample weighting has lower loss. Oracle fields were not available, so transfer and independent operational-difficulty claims are not made.

## Selected-checkpoint performance

| size_role   |   n |     mean |      std |   median |       min |      max |   improved_n |   declined_n |
|:------------|----:|---------:|---------:|---------:|----------:|---------:|-------------:|-------------:|
| L           |  24 | 0.014335 | 0.010844 | 0.014769 | -9e-05    | 0.040359 |           23 |            1 |
| M           |  24 | 0.01143  | 0.007608 | 0.010896 | -0.00249  | 0.024096 |           23 |            1 |
| S           |  24 | 0.037488 | 0.022782 | 0.035636 | -0.012819 | 0.082516 |           23 |            1 |

LogLoss improvement (equal minus sample):

| size_role   |   n |     mean |      std |   median |       min |      max |   improved_n |   declined_n |
|:------------|----:|---------:|---------:|---------:|----------:|---------:|-------------:|-------------:|
| L           |  24 | 0.011722 | 0.012862 | 0.010332 | -0.010865 | 0.03747  |           20 |            4 |
| M           |  24 | 0.018143 | 0.013796 | 0.017853 | -0.011585 | 0.053437 |           21 |            3 |
| S           |  24 | 0.233449 | 0.183488 | 0.19486  | -0.044134 | 0.58297  |           22 |            2 |

Per role and task AUC deltas:

| size_role   |   task |   n |     mean |      std |   median |       min |      max |   improved_n |   declined_n |
|:------------|-------:|----:|---------:|---------:|---------:|----------:|---------:|-------------:|-------------:|
| L           |      0 |  12 | 0.01485  | 0.01037  | 0.015555 | -9e-05    | 0.036585 |           11 |            1 |
| L           |      1 |  12 | 0.01382  | 0.011737 | 0.013813 |  0.000608 | 0.040359 |           12 |            0 |
| M           |      0 |  12 | 0.010843 | 0.007699 | 0.011303 | -0.00249  | 0.023605 |           11 |            1 |
| M           |      1 |  12 | 0.012018 | 0.00781  | 0.010623 |  0.000533 | 0.024096 |           12 |            0 |
| S           |      0 |  12 | 0.041777 | 0.023051 | 0.039676 |  0.005651 | 0.082516 |           12 |            0 |
| S           |      1 |  12 | 0.0332   | 0.022667 | 0.032476 | -0.012819 | 0.075643 |           11 |            1 |

Overall six-cell paired AUC delta (12 setting×seed pairs):

|   n |     mean |      std |   median |      min |      max |   improved_n |   declined_n |
|----:|---------:|---------:|---------:|---------:|---------:|-------------:|-------------:|
|  12 | 0.021085 | 0.009768 | 0.020021 | 0.001382 | 0.035721 |           12 |            0 |

Seed-level role aggregates (12 setting×seed pairs per role):

| size_role   |   n |     mean |      std |   median |       min |      max |   improved_n |   declined_n |
|:------------|----:|---------:|---------:|---------:|----------:|---------:|-------------:|-------------:|
| L           |  12 | 0.014335 | 0.010798 | 0.015385 |  0.000538 | 0.038472 |           12 |            0 |
| M           |  12 | 0.01143  | 0.007201 | 0.01127  |  0.000738 | 0.022959 |           12 |            0 |
| S           |  12 | 0.037488 | 0.021234 | 0.036515 | -0.003584 | 0.064629 |           11 |            1 |

## Shared optimization diagnostics

Contribution decomposition (sample minus equal; trajectory columns compare model states under a common reference weighting):

| size_role   |   actual_equal |   actual_sample |   total_actual_delta |   direct_delta_equal_state |   direct_delta_sample_state |   equal_ref_trajectory_delta |   sample_ref_trajectory_delta |   update_alignment_equal |   update_alignment_sample |   update_alignment_delta |
|:------------|---------------:|----------------:|---------------------:|---------------------------:|----------------------------:|-----------------------------:|------------------------------:|-------------------------:|--------------------------:|-------------------------:|
| L           |       0.060482 |        0.411968 |             0.351486 |                   0.346387 |                    0.331607 |                     0.019879 |                      0.005099 |                 0.05949  |                  0.102549 |                 0.043059 |
| M           |       0.088885 |        0.079536 |            -0.00935  |                  -0.003467 |                   -0.010915 |                     0.001566 |                     -0.005882 |                 0.08287  |                  0.085762 |                 0.002893 |
| S           |       0.350633 |        0.008497 |            -0.342136 |                  -0.342919 |                   -0.320692 |                    -0.021445 |                      0.000783 |                 0.071998 |                  0.051946 |                -0.020052 |

Magnitude allocation by size role:

| size_role   |   equal_actual_share |   sample_actual_share |   equal_state_sample_cf_share |   sample_state_equal_cf_share |   equal_actual_overrep |   sample_actual_overrep |
|:------------|---------------------:|----------------------:|------------------------------:|------------------------------:|-----------------------:|------------------------:|
| L           |             0.17676  |              0.696491 |                      0.685922 |                      0.194896 |                0.22095 |                0.870614 |
| M           |             0.236627 |              0.241459 |                      0.247435 |                      0.239588 |                1.28601 |                1.31228  |
| S           |             0.586614 |              0.06205  |                      0.066643 |                      0.565516 |               36.6634  |                3.87812  |

Mismatch decomposition across matched probes:

|   equal_state_equal_cf |   equal_state_sample_cf |   sample_state_equal_cf |   sample_state_sample_cf |   direct_equal_state |   direct_sample_state |   trajectory_equal_ref |   trajectory_sample_ref |   total_actual |
|-----------------------:|------------------------:|------------------------:|-------------------------:|---------------------:|----------------------:|-----------------------:|------------------------:|---------------:|
|               0.639906 |                0.140731 |                0.614508 |                 0.132993 |            -0.499176 |             -0.481515 |              -0.025398 |               -0.007737 |      -0.506913 |

Shared gradient norm, variance, and SNR:

| weighting   | size_role   |   raw_gradient_norm |   weighted_gradient_norm |   gradient_variance |   gradient_SNR |
|:------------|:------------|--------------------:|-------------------------:|--------------------:|---------------:|
| equal       | L           |            0.05     |                 0.05     |            0.040019 |       0.909985 |
| sample      | L           |            0.042552 |                 0.102124 |            0.03864  |       0.775033 |
| equal       | M           |            0.069608 |                 0.069608 |            0.042914 |       0.820183 |
| sample      | M           |            0.057799 |                 0.031905 |            0.040357 |       0.826109 |
| equal       | S           |            0.179335 |                 0.179335 |            0.085372 |       0.479859 |
| sample      | S           |            0.149236 |                 0.007163 |            0.068665 |       0.480107 |

## Competing diagnostics

Routing:

| size_role   |   entropy_equal |   entropy_sample |   entropy_delta |   experts_delta |   top1_delta |
|:------------|----------------:|-----------------:|----------------:|----------------:|-------------:|
| L           |        0.322161 |         0.306091 |       -0.01607  |       -0.024525 |     0.018382 |
| M           |        0.348894 |         0.296336 |       -0.052558 |       -0.10513  |     0.024824 |
| S           |        0.304832 |         0.360099 |        0.055267 |        0.095092 |    -0.026513 |

Raw shared-gradient pairwise cosine on common epochs:

| domain_pair   |    equal |   sample |    delta |
|:--------------|---------:|---------:|---------:|
| 0-1           | 0.056658 | 0.100585 | 0.043928 |
| 0-2           | 0.059636 | 0.093933 | 0.034296 |
| 1-2           | 0.069972 | 0.109468 | 0.039496 |

Parameter-location summary (cross-group L2 norms are descriptive only; RMS is required for strict cross-group scale comparison):

| weighting   | parameter_group   |   gradient_norm | gradient_SNR   |   contribution |
|:------------|:------------------|----------------:|:---------------|---------------:|
| equal       | architecture      |        0.002002 |                |       0.166667 |
| equal       | cell_specific     |        0.154157 |                |       0.166667 |
| equal       | domain_specific   |        0.020822 |                |       0.166667 |
| equal       | shared            |        0.099648 | 0.736676       |       0.166667 |
| equal       | task_specific     |        0.018845 |                |       0.166667 |
| sample      | architecture      |        0.001944 |                |       0.166667 |
| sample      | cell_specific     |        0.145127 |                |       0.166667 |
| sample      | domain_specific   |        0.020125 |                |       0.166667 |
| sample      | shared            |        0.083196 | 0.693749       |       0.166667 |
| sample      | task_specific     |        0.017508 |                |       0.166667 |

Actual contribution mismatch by parameter location:

| parameter_group   |    equal |   sample |     delta |
|:------------------|---------:|---------:|----------:|
| architecture      | 0.524935 | 0.110083 | -0.414852 |
| cell_specific     | 0.48796  | 0.091563 | -0.396397 |
| domain_specific   | 0.541957 | 0.107383 | -0.434574 |
| shared            | 0.639906 | 0.132993 | -0.506913 |
| task_specific     | 0.549475 | 0.097189 | -0.452287 |

Checkpoint/convergence diagnostic (12 pairs per role):

| size_role   |   common_mean |   common_positive |   last_common_mean |   last_common_positive |   equal_checkpoint_mean |   equal_checkpoint_positive |
|:------------|--------------:|------------------:|-------------------:|-----------------------:|------------------------:|----------------------------:|
| L           |      0.021268 |                12 |           0.004431 |                     11 |                0.014098 |                          12 |
| M           |      0.00932  |                10 |           0.007861 |                     10 |                0.010078 |                          11 |
| S           |      0.027217 |                10 |           0.067036 |                     12 |                0.035506 |                          11 |

## Timing and association

Across common-support epoch observations, Spearman correlation between sample-minus-equal Large test AUC and actual shared mismatch delta was -0.1825; corresponding correlations with Large routing entropy delta and top-1-frequency delta were -0.2381 and 0.2895. Post-hoc onset tables use two-consecutive-epoch thresholds |AUC delta|>=0.001 and |mismatch delta|>=0.01; these thresholds were not preregistered and are descriptive only.

## Scientific assessment

### Stable repeated phenomena

1. **STRONG — sample weighting improves performance under this protocol.** The six-cell test-AUC mean improved in 12/12 setting×seed pairs (mean +0.021085). Seed-level role means improved for Large in 12/12 (+0.014335), Medium in 12/12 (+0.011430), and Small in 11/12 (+0.037488). Both tasks show the same aggregate direction. Mean LogLoss also improved for every role, although individual cells contain exceptions.
2. **STRONG — equal weighting produces severe size-share mismatch, and sample weighting reduces it.** Shared mismatch fell from 0.639906 to 0.132993 (delta -0.506913) in all 12 paired runs. Equal weighting allocated mean shared magnitude shares L/M/S=0.1768/0.2366/0.5866 against data shares 0.800/0.184/0.016; sample weighting changed them to 0.6965/0.2415/0.0621. Small remained overrepresented (3.88x), but far less than under equal weighting (36.66x).
3. **STRONG — actual shared update alignment moves toward Large and away from Small.** Large alignment increased in 12/12 seed-setting pairs (mean +0.0427); Small decreased in 12/12 (mean -0.0199). Pairwise raw-gradient cosine also became more positive for all three physical-domain pairs.
4. **STRONG — the result is not a small-domain-improvement/large-domain-decline trade-off.** Large, Medium, and Small all improve on average; Small improves the most. This directly contradicts a claimed trade-off for the tested sample-weighting intervention.

### Contradictions and limits

- Cell-level AUC is not unanimous: each role has 23/24 improving cells, with one decline in each. At the seed-role level, Large and Medium are unanimous, while Small has one decline (d0_large, seed 2024).
- LogLoss has more exceptions than AUC: 4/24 Large, 3/24 Medium, and 2/24 Small cells worsen despite positive mean improvements.
- The mismatch reduction is dominated by the deterministic coefficient change: at the equal-trained state, the direct counterfactual change is -0.499176 versus total actual -0.506913. Common-reference trajectory effects are much smaller and inconsistent (equal reference negative in 8/12 pairs; sample reference negative in 6/12).
- The epoch-level association between Large AUC improvement and mismatch reduction is directionally compatible but weak (Spearman -0.1825). Routing associations are of comparable magnitude, so routing cannot be dismissed as a competing mediator.
- Oracle/baseline rows are entirely missing. Transfer, independent operational difficulty, and a clean size-versus-difficulty decomposition cannot be claimed.

### Competing explanations

- **H2 raw gradient/noise — MEDIUM evidence against as the main explanation.** Small raw gradients and variance fall, but Small SNR is essentially unchanged; Large SNR worsens while Large AUC improves. Noise/SNR alone does not match the performance pattern.
- **H3 private parameters — MEDIUM evidence for a shared component, not exclusivity.** Mismatch falls in every parameter group, with the largest absolute reduction in shared (-0.5069), but private and architecture groups also change substantially (-0.3964 to -0.4523).
- **H4 routing — unresolved (WEAK evidence either way).** Routing changes are modest but appear from the same first observed epoch, and their epoch-level correlations with Large AUC are comparable to the mismatch correlation.
- **H5 checkpoint/convergence — STRONG evidence against a selection-only explanation.** Large improvement persists across common epochs (12/12 positive mean), at the equal arm's checkpoint (12/12), and at the last common epoch (11/12). Best-epoch shift averages only +0.25 epoch and varies from -3 to +5.
- **H6 fixed physical domain/difficulty — MEDIUM evidence against a fixed-domain-only explanation.** Large and Medium role improvements hold for all four seeds in all three rotations; physical-domain magnitudes differ, but no oracle difficulty data are available.
- **H7 instrumentation artifact — STRONG evidence against.** Review-stage ON/OFF checks had identical batches, losses, metrics, and final parameters, and all formal pairs passed identity/fingerprint checks.

### Overall hypothesis judgment

**H1 evidence strength: MEDIUM.** The predicted contribution distortion, role rotation, actual-update alignment, and performance improvement are highly repeatable, and shared mismatch changes more than any private group. However, E1 is observational, the dominant mismatch change is the mathematically forced coefficient effect, common-reference trajectory effects are inconsistent, the performance–mismatch association is weak, and routing remains a viable mediator. Therefore E1 supports shared contribution distortion as a mechanism candidate but does not establish it as the primary causal explanation.

**Domain size as the main explanatory variable: MEDIUM for optimization allocation, WEAK for performance causality.** Role rotation makes size a plausible driver of contribution allocation, but absent oracle difficulty data and simultaneous routing changes prevent a main-cause claim for AUC.

### Best next falsification experiment

Proceed to the preregistered E2 objective × sampling factorial. Cross target objective (equal-cell versus sample-proportional) with batch sampling (natural versus domain-balanced), using importance correction so each target objective is unbiased under either sampler. Keep expected loss scale, examples processed, initialization, optimizer, checkpoint rule, and seeds fixed. If AUC follows the target objective across samplers, the coefficient/objective explanation gains causal support; if it follows sampling/exposure or gradient variance instead, H1 is weakened. This is more discriminating than adding another weighting heuristic.

## Availability

Non-missing oracle rows: 0. Detailed seed-level tables are in `stage_E1_analysis_tables/`.
