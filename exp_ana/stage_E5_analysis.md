# Stage E5 Main Scientific Analysis

## Integrity

All 20 fixed-budget runs and four five-arm blocks are valid. Each run has 18 epochs, identical per-seed batch order and optimizer-step counts, and complete E3 frozen-oracle coverage. Seed (n=4) is the replication unit.

## Arm results

| view        | arm_key         |   seed |   epoch |   overall_auc |   overall_logloss |   worst_domain_auc |   worst_cell_auc |   peak_epoch |   fixed_final_epoch |   negative_transfer_rate |
|:------------|:----------------|-------:|--------:|--------------:|------------------:|-------------------:|-----------------:|-------------:|--------------------:|-------------------------:|
| fixed-final | forward         | 2023.5 |   17    |      0.715299 |          0.676285 |           0.659812 |         0.647575 |         7.75 |                  17 |                 0.166667 |
| fixed-final | reverse         | 2023.5 |   17    |      0.713915 |          0.700553 |           0.654788 |         0.646778 |         6.75 |                  17 |                 0.166667 |
| fixed-final | static_interior | 2023.5 |   17    |      0.711597 |          0.717638 |           0.65078  |         0.64082  |         7.75 |                  17 |                 0.208333 |
| fixed-final | static_matched  | 2023.5 |   17    |      0.714193 |          0.695993 |           0.654852 |         0.643748 |         7    |                  17 |                 0.166667 |
| fixed-final | static_sample   | 2023.5 |   17    |      0.717079 |          0.655851 |           0.662206 |         0.651412 |         6.75 |                  17 |                 0.125    |
| selected    | forward         | 2023.5 |    7.75 |      0.726469 |          0.568671 |           0.679788 |         0.669526 |         7.75 |                  17 |                 0.125    |
| selected    | reverse         | 2023.5 |    6.75 |      0.728243 |          0.546869 |           0.684281 |         0.674817 |         6.75 |                  17 |                 0.166667 |
| selected    | static_interior | 2023.5 |    7.75 |      0.726563 |          0.569842 |           0.6786   |         0.669963 |         7.75 |                  17 |                 0.083333 |
| selected    | static_matched  | 2023.5 |    7    |      0.72561  |          0.559779 |           0.677299 |         0.665318 |         7    |                  17 |                 0.25     |
| selected    | static_sample   | 2023.5 |    6.75 |      0.728388 |          0.549482 |           0.684903 |         0.67542  |         6.75 |                  17 |                 0.208333 |

## Forward paired contrasts

| view        | contrast                | metric           |   improvement |      std |    median |       min |       max |    ci_low |   ci_high |   positive_seeds |   over_margin_seeds | evidence   |
|:------------|:------------------------|:-----------------|--------------:|---------:|----------:|----------:|----------:|----------:|----------:|-----------------:|--------------------:|:-----------|
| fixed-final | forward-static_sample   | overall_auc      |     -0.00178  | 0.002552 | -0.002625 | -0.003818 |  0.001946 | -0.003472 |  0.000756 |                1 |                   1 | WEAK       |
| fixed-final | forward-static_sample   | overall_logloss  |     -0.020435 | 0.024009 | -0.016819 | -0.052805 |  0.004704 | -0.042956 | -0.00153  |                1 |                   1 | WEAK       |
| fixed-final | forward-static_sample   | worst_domain_auc |     -0.002394 | 0.004124 | -0.003685 | -0.005558 |  0.003354 | -0.005383 |  0.001214 |                1 |                   1 | WEAK       |
| fixed-final | forward-static_matched  | overall_auc      |      0.001106 | 0.007492 | -0.000361 | -0.006349 |  0.011495 | -0.00472  |  0.008399 |                2 |                   1 | WEAK       |
| fixed-final | forward-static_matched  | overall_logloss  |      0.019707 | 0.049229 |  0.028263 | -0.045527 |  0.06783  | -0.022924 |  0.056357 |                3 |                   3 | MEDIUM     |
| fixed-final | forward-static_matched  | worst_domain_auc |      0.004961 | 0.016224 | -0.001022 | -0.006873 |  0.02876  | -0.005152 |  0.020712 |                2 |                   2 | WEAK       |
| fixed-final | forward-reverse         | overall_auc      |      0.001384 | 0.004648 |  0.000159 | -0.002782 |  0.008    | -0.0019   |  0.005893 |                2 |                   1 | WEAK       |
| fixed-final | forward-reverse         | overall_logloss  |      0.024267 | 0.027322 |  0.023915 | -0.007561 |  0.056799 |  0.002599 |  0.046288 |                3 |                   3 | MEDIUM     |
| fixed-final | forward-reverse         | worst_domain_auc |      0.005024 | 0.011033 |  0.001606 | -0.004097 |  0.02098  | -0.002315 |  0.015781 |                3 |                   2 | MEDIUM     |
| fixed-final | forward-static_interior | overall_auc      |      0.003701 | 0.003456 |  0.004178 | -0.000937 |  0.007386 |  0.000438 |  0.006488 |                3 |                   3 | MEDIUM     |
| fixed-final | forward-static_interior | overall_logloss  |      0.041353 | 0.020387 |  0.042958 |  0.016755 |  0.06274  |  0.025123 |  0.057582 |                4 |                   4 | STRONG     |
| fixed-final | forward-static_interior | worst_domain_auc |      0.009032 | 0.008123 |  0.010087 | -0.0017   |  0.017655 |  0.001687 |  0.015323 |                3 |                   3 | MEDIUM     |
| selected    | forward-static_sample   | overall_auc      |     -0.001918 | 0.00135  | -0.001635 | -0.003759 | -0.000643 | -0.003133 | -0.000948 |                0 |                   0 | WEAK       |
| selected    | forward-static_sample   | overall_logloss  |     -0.019189 | 0.019461 | -0.019784 | -0.037703 |  0.000513 | -0.035855 | -0.002523 |                1 |                   0 | WEAK       |
| selected    | forward-static_sample   | worst_domain_auc |     -0.005115 | 0.003336 | -0.004485 | -0.009637 | -0.001851 | -0.008134 | -0.002725 |                0 |                   0 | WEAK       |
| selected    | forward-static_matched  | overall_auc      |      0.00086  | 0.003346 | -0.000413 | -0.001526 |  0.00579  | -0.001176 |  0.004136 |                2 |                   1 | WEAK       |
| selected    | forward-static_matched  | overall_logloss  |     -0.008893 | 0.012702 | -0.006443 | -0.024276 |  0.001592 | -0.01932  |  0.001534 |                2 |                   2 | WEAK       |
| selected    | forward-static_matched  | worst_domain_auc |      0.002489 | 0.010433 |  0.000351 | -0.007783 |  0.017036 | -0.005659 |  0.012773 |                2 |                   1 | WEAK       |
| selected    | forward-reverse         | overall_auc      |     -0.001774 | 0.001412 | -0.00113  | -0.003887 | -0.000947 | -0.003189 | -0.001001 |                0 |                   0 | WEAK       |
| selected    | forward-reverse         | overall_logloss  |     -0.021802 | 0.022612 | -0.020779 | -0.045432 | -0.000219 | -0.041071 | -0.002534 |                0 |                   0 | WEAK       |
| selected    | forward-reverse         | worst_domain_auc |     -0.004493 | 0.003355 | -0.003067 | -0.009491 | -0.002347 | -0.007827 | -0.002586 |                0 |                   0 | WEAK       |
| selected    | forward-static_interior | overall_auc      |     -9.4e-05  | 0.00089  |  0.000122 | -0.001355 |  0.000735 | -0.000973 |  0.000569 |                3 |                   0 | WEAK       |
| selected    | forward-static_interior | overall_logloss  |      0.001171 | 0.005687 |  0.002078 | -0.006359 |  0.006886 | -0.003817 |  0.005349 |                3 |                   2 | MEDIUM     |
| selected    | forward-static_interior | worst_domain_auc |      0.001188 | 0.002122 |  0.000864 | -0.001005 |  0.004029 | -0.000422 |  0.003122 |                3 |                   2 | MEDIUM     |

## Primary per-seed deltas

|   seed |   forward_minus_static_sample |   forward_minus_static_matched |   forward_minus_reverse |   forward_minus_static_interior |
|-------:|------------------------------:|-------------------------------:|------------------------:|--------------------------------:|
|   2022 |                     -0.003818 |                       0.00017  |                0.000746 |                       -0.000937 |
|   2023 |                     -0.002813 |                      -0.006349 |               -0.002782 |                        0.004562 |
|   2024 |                      0.001946 |                       0.011495 |                0.008    |                        0.007386 |
|   2025 |                     -0.002437 |                      -0.000891 |               -0.000427 |                        0.003795 |

## Role safety relative to static alpha=1

| view        | role   | metric   |   improvement |    ci_low |   ci_high |   positive_seeds |
|:------------|:-------|:---------|--------------:|----------:|----------:|-----------------:|
| fixed-final | L      | auc      |     -0.000645 | -0.002279 |  0.000989 |                2 |
| fixed-final | L      | transfer |     -0.000645 | -0.002279 |  0.000989 |                2 |
| fixed-final | L      | logloss  |     -0.002905 | -0.008441 |  0.001237 |                2 |
| fixed-final | M      | auc      |     -0.002302 | -0.003819 |  0.000155 |                1 |
| fixed-final | M      | transfer |     -0.002302 | -0.003819 |  0.000155 |                1 |
| fixed-final | M      | logloss  |     -0.015304 | -0.031916 | -0.005811 |                0 |
| fixed-final | S      | auc      |     -0.002394 | -0.005383 |  0.001214 |                1 |
| fixed-final | S      | transfer |     -0.002394 | -0.005383 |  0.001214 |                1 |
| fixed-final | S      | logloss  |     -0.043095 | -0.098416 |  0.006721 |                1 |
| selected    | L      | auc      |      0.00216  | -0.001396 |  0.007657 |                2 |
| selected    | L      | transfer |      0.00216  | -0.001396 |  0.007657 |                2 |
| selected    | L      | logloss  |     -0.000157 | -0.008201 |  0.007888 |                2 |
| selected    | M      | auc      |     -0.0028   | -0.004989 | -0.000612 |                0 |
| selected    | M      | transfer |     -0.0028   | -0.004989 | -0.000612 |                0 |
| selected    | M      | logloss  |     -0.012348 | -0.020322 | -0.004374 |                0 |
| selected    | S      | auc      |     -0.005115 | -0.008134 | -0.002725 |                0 |
| selected    | S      | transfer |     -0.005115 | -0.008134 | -0.002725 |                0 |
| selected    | S      | logloss  |     -0.045062 | -0.084554 | -0.00557  |                0 |

## Per-role/task safety

| view        | role   |   task | metric   |   forward_vs_static_sample |    ci_low |   ci_high |   positive_seeds |
|:------------|:-------|-------:|:---------|---------------------------:|----------:|----------:|-----------------:|
| fixed-final | L      |      0 | auc      |                  -0.00168  | -0.005116 |  0.001756 |                2 |
| fixed-final | L      |      0 | transfer |                  -0.00168  | -0.005116 |  0.001756 |                2 |
| fixed-final | L      |      0 | logloss  |                  -0.004722 | -0.012249 |  0.000561 |                1 |
| fixed-final | L      |      1 | auc      |                   0.000389 | -0.001015 |  0.001962 |                3 |
| fixed-final | L      |      1 | transfer |                   0.000389 | -0.001015 |  0.001962 |                3 |
| fixed-final | L      |      1 | logloss  |                  -0.001088 | -0.005893 |  0.004064 |                2 |
| fixed-final | M      |      0 | auc      |                  -0.004272 | -0.005876 | -0.001884 |                0 |
| fixed-final | M      |      0 | transfer |                  -0.004272 | -0.005876 | -0.001884 |                0 |
| fixed-final | M      |      0 | logloss  |                  -0.021381 | -0.040879 | -0.01016  |                0 |
| fixed-final | M      |      1 | auc      |                  -0.000332 | -0.001762 |  0.002193 |                1 |
| fixed-final | M      |      1 | transfer |                  -0.000332 | -0.001762 |  0.002193 |                1 |
| fixed-final | M      |      1 | logloss  |                  -0.009227 | -0.023205 | -0.001433 |                0 |
| fixed-final | S      |      0 | auc      |                  -0.003837 | -0.011108 |  0.003036 |                2 |
| fixed-final | S      |      0 | transfer |                  -0.003837 | -0.011108 |  0.003036 |                2 |
| fixed-final | S      |      0 | logloss  |                  -0.01816  | -0.076977 |  0.029807 |                2 |
| fixed-final | S      |      1 | auc      |                  -0.00095  | -0.008964 |  0.004129 |                3 |
| fixed-final | S      |      1 | transfer |                  -0.00095  | -0.008964 |  0.004129 |                3 |
| fixed-final | S      |      1 | logloss  |                  -0.068031 | -0.114252 | -0.021968 |                0 |
| selected    | L      |      0 | auc      |                   0.000367 | -0.004032 |  0.004653 |                3 |
| selected    | L      |      0 | transfer |                   0.000367 | -0.004032 |  0.004653 |                3 |
| selected    | L      |      0 | logloss  |                  -0.004255 | -0.013958 |  0.004464 |                2 |
| selected    | L      |      1 | auc      |                   0.003953 | -4.3e-05  |  0.01076  |                3 |
| selected    | L      |      1 | transfer |                   0.003953 | -4.3e-05  |  0.01076  |                3 |
| selected    | L      |      1 | logloss  |                   0.003942 | -0.003428 |  0.011311 |                2 |
| selected    | M      |      0 | auc      |                  -0.000383 | -0.003551 |  0.001799 |                3 |
| selected    | M      |      0 | transfer |                  -0.000383 | -0.003551 |  0.001799 |                3 |
| selected    | M      |      0 | logloss  |                  -0.012966 | -0.023228 | -0.002704 |                0 |
| selected    | M      |      1 | auc      |                  -0.005218 | -0.008911 | -0.001837 |                0 |
| selected    | M      |      1 | transfer |                  -0.005218 | -0.008911 | -0.001837 |                0 |
| selected    | M      |      1 | logloss  |                  -0.011731 | -0.026335 |  0.002873 |                2 |
| selected    | S      |      0 | auc      |                  -0.005894 | -0.008976 | -0.002808 |                0 |
| selected    | S      |      0 | transfer |                  -0.005894 | -0.008976 | -0.002808 |                0 |
| selected    | S      |      0 | logloss  |                  -0.05504  | -0.10006  | -0.015463 |                0 |
| selected    | S      |      1 | auc      |                  -0.004335 | -0.011391 |  0.000824 |                1 |
| selected    | S      |      1 | transfer |                  -0.004335 | -0.011391 |  0.000824 |                1 |
| selected    | S      |      1 | logloss  |                  -0.035084 | -0.074491 |  0.004323 |                1 |

## Exposure audit

| arm_key         | size_role   |   seed |   alpha_exposure_total |   coefficient_integral_cell |   step_weighted_coefficient_integral_cell |
|:----------------|:------------|-------:|-----------------------:|----------------------------:|------------------------------------------:|
| forward         | L           | 2023.5 |                  15.75 |                    41.1924  |                                 205.962   |
| forward         | M           | 2023.5 |                  15.75 |                    11.4165  |                                  57.0825  |
| forward         | S           | 2023.5 |                  15.75 |                     1.39115 |                                   6.95576 |
| reverse         | L           | 2023.5 |                  15.75 |                    41.1924  |                                 205.962   |
| reverse         | M           | 2023.5 |                  15.75 |                    11.4165  |                                  57.0825  |
| reverse         | S           | 2023.5 |                  15.75 |                     1.39115 |                                   6.95576 |
| static_interior | L           | 2023.5 |                  13.5  |                    38.9806  |                                 194.903   |
| static_interior | M           | 2023.5 |                  13.5  |                    12.9463  |                                  64.7314  |
| static_interior | S           | 2023.5 |                  13.5  |                     2.0731  |                                  10.3655  |
| static_matched  | L           | 2023.5 |                  15.75 |                    41.253   |                                 206.265   |
| static_matched  | M           | 2023.5 |                  15.75 |                    11.4016  |                                  57.0081  |
| static_matched  | S           | 2023.5 |                  15.75 |                     1.34542 |                                   6.72708 |
| static_sample   | L           | 2023.5 |                  18    |                    43.2     |                                 216       |
| static_sample   | M           | 2023.5 |                  18    |                     9.936   |                                  49.68    |
| static_sample   | S           | 2023.5 |                  18    |                     0.864   |                                   4.32    |

| role   |   forward_minus_static_matched |   reverse_minus_static_matched |   forward_minus_reverse |
|:-------|-------------------------------:|-------------------------------:|------------------------:|
| L      |                      -0.302992 |                      -0.302992 |                      -0 |
| M      |                       0.074308 |                       0.074308 |                       0 |
| S      |                       0.228683 |                       0.228683 |                       0 |

## Early/middle/late performance

| arm_key         | phase   |   seed |   test_auc |   test_logloss |   validation_auc |
|:----------------|:--------|-------:|-----------:|---------------:|-----------------:|
| forward         | early   | 2023.5 |   0.670044 |       0.566454 |         0.666288 |
| forward         | late    | 2023.5 |   0.717772 |       0.654585 |         0.715542 |
| forward         | middle  | 2023.5 |   0.722623 |       0.578189 |         0.718513 |
| reverse         | early   | 2023.5 |   0.663999 |       0.567169 |         0.658911 |
| reverse         | late    | 2023.5 |   0.717961 |       0.657075 |         0.716312 |
| reverse         | middle  | 2023.5 |   0.723883 |       0.580725 |         0.719312 |
| static_interior | early   | 2023.5 |   0.669965 |       0.567101 |         0.666037 |
| static_interior | late    | 2023.5 |   0.715146 |       0.680925 |         0.713162 |
| static_interior | middle  | 2023.5 |   0.721572 |       0.583489 |         0.717648 |
| static_matched  | early   | 2023.5 |   0.669052 |       0.567464 |         0.665081 |
| static_matched  | late    | 2023.5 |   0.717003 |       0.663641 |         0.71389  |
| static_matched  | middle  | 2023.5 |   0.721876 |       0.583791 |         0.717858 |
| static_sample   | early   | 2023.5 |   0.663712 |       0.5669   |         0.658714 |
| static_sample   | late    | 2023.5 |   0.719817 |       0.635395 |         0.718257 |
| static_sample   | middle  | 2023.5 |   0.72459  |       0.573845 |         0.719789 |

## Early/middle/late shared contribution

| arm_key         | phase   | size_role   |   magnitude_share |   signed_contribution |   raw_gradient |   weighted_gradient |   variance |      snr |   mismatch |
|:----------------|:--------|:------------|------------------:|----------------------:|---------------:|--------------------:|-----------:|---------:|-----------:|
| forward         | early   | L           |          0.774732 |              0.448331 |       0.063378 |            0.138314 |   0.060809 | 1.78648  |   0.086912 |
| forward         | early   | M           |          0.190027 |              0.049698 |       0.079578 |            0.056614 |   0.15776  | 2.04095  |   0.086912 |
| forward         | early   | S           |          0.035241 |              0.00197  |       0.086776 |            0.009565 |   0.148716 | 0.738878 |   0.086912 |
| forward         | late    | L           |          0.587711 |              0.361087 |       0.027777 |            0.065895 |   0.001035 | 0.066288 |   0.212549 |
| forward         | late    | M           |          0.298348 |              0.119938 |       0.057469 |            0.032965 |   0.004312 | 0.121234 |   0.212549 |
| forward         | late    | S           |          0.113941 |              0.018975 |       0.226243 |            0.012336 |   0.058271 | 0.196863 |   0.212549 |
| forward         | middle  | L           |          0.603701 |              0.364683 |       0.029462 |            0.067458 |   0.001142 | 0.108393 |   0.198223 |
| forward         | middle  | M           |          0.295762 |              0.116132 |       0.050708 |            0.032131 |   0.003165 | 0.239187 |   0.198223 |
| forward         | middle  | S           |          0.100537 |              0.019184 |       0.146483 |            0.010755 |   0.02742  | 0.245976 |   0.198223 |
| reverse         | early   | L           |          0.823199 |              0.46493  |       0.062766 |            0.149826 |   0.060834 | 1.71239  |   0.07943  |
| reverse         | early   | M           |          0.154449 |              0.033465 |       0.07828  |            0.043739 |   0.157676 | 2.08995  |   0.07943  |
| reverse         | early   | S           |          0.022352 |              0.001605 |       0.089132 |            0.004538 |   0.149087 | 0.688354 |   0.07943  |
| reverse         | late    | L           |          0.4632   |              0.275198 |       0.025691 |            0.056562 |   0.000886 | 0.083431 |   0.336919 |
| reverse         | late    | M           |          0.341715 |              0.168977 |       0.06297  |            0.04395  |   0.006371 | 0.104753 |   0.336919 |
| reverse         | late    | S           |          0.195085 |              0.055826 |       0.234043 |            0.024057 |   0.071228 | 0.163834 |   0.336919 |
| reverse         | middle  | L           |          0.585674 |              0.358489 |       0.028377 |            0.065125 |   0.001119 | 0.126926 |   0.215761 |
| reverse         | middle  | M           |          0.297243 |              0.117758 |       0.050397 |            0.031898 |   0.003082 | 0.253055 |   0.215761 |
| reverse         | middle  | S           |          0.117082 |              0.023753 |       0.162185 |            0.012373 |   0.035302 | 0.209033 |   0.215761 |
| static_interior | early   | L           |          0.759122 |              0.441355 |       0.063076 |            0.136597 |   0.060805 | 1.80708  |   0.094195 |
| static_interior | early   | M           |          0.199643 |              0.055516 |       0.079794 |            0.057391 |   0.15781  | 1.98596  |   0.094195 |
| static_interior | early   | S           |          0.041235 |              0.003129 |       0.08795  |            0.010129 |   0.148917 | 0.698601 |   0.094195 |
| static_interior | late    | L           |          0.442316 |              0.262333 |       0.026917 |            0.058292 |   0.000955 | 0.072558 |   0.357758 |
| static_interior | late    | M           |          0.332295 |              0.16659  |       0.063281 |            0.045514 |   0.005914 | 0.098432 |   0.357758 |
| static_interior | late    | S           |          0.225389 |              0.071077 |       0.264016 |            0.030407 |   0.088634 | 0.154401 |   0.357758 |
| static_interior | middle  | L           |          0.538772 |              0.329492 |       0.029268 |            0.063383 |   0.001116 | 0.119769 |   0.261788 |
| static_interior | middle  | M           |          0.305029 |              0.128372 |       0.049238 |            0.035414 |   0.002971 | 0.205175 |   0.261788 |
| static_interior | middle  | S           |          0.156199 |              0.042136 |       0.154511 |            0.017795 |   0.029446 | 0.232807 |   0.261788 |
| static_matched  | early   | L           |          0.795645 |              0.455589 |       0.061883 |            0.141826 |   0.060784 | 1.98089  |   0.076884 |
| static_matched  | early   | M           |          0.175417 |              0.043195 |       0.078565 |            0.049765 |   0.157746 | 1.90137  |   0.076884 |
| static_matched  | early   | S           |          0.028937 |              0.001216 |       0.087015 |            0.006504 |   0.14849  | 0.808703 |   0.076884 |
| static_matched  | late    | L           |          0.534312 |              0.328165 |       0.028132 |            0.064475 |   0.00104  | 0.075886 |   0.26575  |
| static_matched  | late    | M           |          0.303771 |              0.132907 |       0.058156 |            0.036838 |   0.004551 | 0.130459 |   0.26575  |
| static_matched  | late    | S           |          0.161917 |              0.038928 |       0.260711 |            0.019487 |   0.089273 | 0.181355 |   0.26575  |
| static_matched  | middle  | L           |          0.596757 |              0.371286 |       0.028476 |            0.065262 |   0.001062 | 0.11848  |   0.2036   |
| static_matched  | middle  | M           |          0.281789 |              0.102822 |       0.047777 |            0.030263 |   0.002737 | 0.192615 |   0.2036   |
| static_matched  | middle  | S           |          0.121454 |              0.025893 |       0.169238 |            0.01265  |   0.036086 | 0.219483 |   0.2036   |
| static_sample   | early   | L           |          0.834093 |              0.469493 |       0.063    |            0.1512   |   0.060872 | 1.70694  |   0.079627 |
| static_sample   | early   | M           |          0.146969 |              0.029494 |       0.077984 |            0.043047 |   0.157636 | 2.093    |   0.079627 |
| static_sample   | early   | S           |          0.018938 |              0.001013 |       0.088146 |            0.004231 |   0.148967 | 0.734185 |   0.079627 |
| static_sample   | late    | L           |          0.598776 |              0.360544 |       0.025967 |            0.06232  |   0.000897 | 0.061075 |   0.20196  |
| static_sample   | late    | M           |          0.304313 |              0.122997 |       0.058034 |            0.032035 |   0.004869 | 0.123983 |   0.20196  |
| static_sample   | late    | S           |          0.096911 |              0.016459 |       0.207105 |            0.009941 |   0.053736 | 0.165409 |   0.20196  |
| static_sample   | middle  | L           |          0.661153 |              0.401669 |       0.028898 |            0.069356 |   0.001173 | 0.115425 |   0.14257  |
| static_sample   | middle  | M           |          0.264934 |              0.087822 |       0.048378 |            0.026705 |   0.002801 | 0.288988 |   0.14257  |
| static_sample   | middle  | S           |          0.073913 |              0.01051  |       0.150673 |            0.007232 |   0.027129 | 0.266191 |   0.14257  |

## Oracle operational difficulty

|   physical_domain | size_role   |   domain_size |   domain_share |   task |   oracle_auc |   oracle_auc_std |   oracle_logloss |
|------------------:|:------------|--------------:|---------------:|-------:|-------------:|-----------------:|-----------------:|
|                 0 | L           |        150000 |          0.8   |      0 |     0.750268 |         0.001807 |         0.616017 |
|                 0 | L           |        150000 |          0.8   |      1 |     0.762286 |         0.0036   |         0.426187 |
|                 1 | M           |         34500 |          0.184 |      0 |     0.703408 |         0.004011 |         0.679265 |
|                 1 | M           |         34500 |          0.184 |      1 |     0.725246 |         0.007731 |         0.459158 |
|                 2 | S           |          3000 |          0.016 |      0 |     0.608291 |         0.004204 |         0.676286 |
|                 2 | S           |          3000 |          0.016 |      1 |     0.617968 |         0.003916 |         0.505344 |

## Strict GO audit

| criterion                                | pass   |     value |
|:-----------------------------------------|:-------|----------:|
| final AUC gain >.001 vs static_sample    | False  | -0.00178  |
| final AUC gain >.001 vs static_matched   | True   |  0.001106 |
| final AUC gain >.001 vs reverse          | True   |  0.001384 |
| >=3/4 positive for all three             | False  |  1        |
| selected directions agree for all three  | False  | -0.001918 |
| LogLoss not worse >.001 vs static_sample | False  | -0.020435 |
| Large transfer no >.001 degradation      | True   | -0.000645 |
| Small transfer no >.001 degradation      | False  | -0.002394 |
| worst-domain no >.001 degradation        | False  | -0.002394 |

## Scientific assessment

1. **WEAK evidence for temporal order.** At fixed epoch 17, Forward beats Reverse by about 0.00138 and Static .875 by about 0.00111, but each contrast is positive in only 2/4 seeds. Reverse beats Forward in seeds 2023 and 2025; Static .875 beats it in the same two seeds. Direction inconsistency prevents a temporal-order claim.
2. **STRONG evidence against Forward as an improvement over the simple baseline.** Forward is worse than static alpha=1 by about 0.00178 at fixed final, positive in only 1/4 seeds. At validation-selected checkpoints it is also worse by about 0.00192; Reverse and static alpha=1 are likewise better. This is not a checkpoint-only artifact.
3. **Exposure matching does not rescue the hypothesis.** Forward and Reverse have identical alpha integrals and step counts; nonlinear coefficient integrals differ slightly from Static .875 and are reported explicitly. Yet Forward-vs-Reverse and Forward-vs-matched signs are unstable, so neither integrated exposure nor time order yields a repeatable advantage.
4. **No beneficial Large/Small trade-off.** Relative to static alpha=1, Forward slightly degrades Large and Medium transfer and degrades Small transfer by about 0.00239; worst-domain AUC is lower by about 0.00239. The schedule does not buy a compensating small-domain improvement.
5. **Contribution timing changes as designed but is not sufficient.** Forward shifts shared magnitude toward Large over training and Reverse shifts it away, while Static controls remain fixed. The absence of stable performance gains despite this manipulation falsifies the claim that the E3 descriptive early/late optimum directly implies a useful linear schedule.
6. **C4 and C8 are ruled out by construction/evidence.** Fixed epochs, steps, examples, batches, and architecture objective are identical. C1/C2 remain descriptive alternatives for individual seeds, but neither explains a stable cross-seed advantage. C7 gradient noise is plausible because the two positive and two negative seed results coincide with varying raw-gradient variance/SNR. C6 cannot be separated from size because E5 uses only canonical d0_large.
7. Oracle difficulty does not equal size: the Small physical domain is not uniformly the hardest across both tasks. With only one role assignment, domain size cannot be claimed as the main causal variable.

## GO / STOP

**STOP — DROP Curriculum.** The mandatory GO criteria fail: Forward does not beat static alpha=1, matched static, and Reverse by >0.001 with >=3/4 seed consistency; selected-checkpoint directions also fail; safety metrics do not compensate. Therefore: `Contribution Curriculum currently lacks empirical support and is dropped.` No schedule/warmup/ratio/shape expansion is warranted.

The best next discriminating step is not another schedule grid. Proceed only to the next independently motivated group3 experiment, with static alpha=1 as the main baseline and without Curriculum. Any future temporal claim would require a new causal design addressing seed-dependent gradient-noise interactions rather than tuning this failed schedule.

## Reproducibility

Tables: `experiments/stage_E5_analysis_tables/`; figures: `experiments/stage_E5_analysis_figures/`.
