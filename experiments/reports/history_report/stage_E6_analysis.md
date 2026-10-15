# Stage E6 Main Scientific Analysis

## Integrity

All 24 runs and four six-arm blocks are valid, with complete E3 frozen-oracle coverage. Sample-static and sample-EMA-control have identical aggregate performance, confirming that EMA bookkeeping alone does not alter training. Seed (n=4) is the replication unit.

## Arm results

| arm_key              |   seed |   overall_auc |   overall_logloss |   worst_domain_auc |   worst_cell_auc |   selected_epoch |   stopped_epoch |   last_common_epoch |   negative_transfer_rate |   last_common_auc |   last_common_logloss |   last_common_transfer |
|:---------------------|-------:|--------------:|------------------:|-------------------:|-----------------:|-----------------:|----------------:|--------------------:|-------------------------:|------------------:|----------------------:|-----------------------:|
| adaptive_ema         | 2023.5 |      0.729972 |          0.536694 |           0.688992 |         0.680007 |             5.75 |           15.75 |               15.25 |                 0.208333 |          0.722271 |              0.612369 |               0.027693 |
| adaptive_ema_uniform | 2023.5 |      0.715031 |          0.620112 |           0.655902 |         0.638337 |             6.75 |           16.75 |               15.25 |                 0.375    |          0.699696 |              0.848705 |               0.005119 |
| adaptive_noema       | 2023.5 |      0.729182 |          0.539305 |           0.688192 |         0.678719 |             5.75 |           15.75 |               15.25 |                 0.25     |          0.722325 |              0.606697 |               0.027747 |
| equal_static         | 2023.5 |      0.715449 |          0.597375 |           0.662057 |         0.642524 |             5.75 |           15.75 |               15.25 |                 0.291667 |          0.696819 |              0.857306 |               0.002241 |
| sample_ema_control   | 2023.5 |      0.728388 |          0.549482 |           0.684903 |         0.67542  |             6.75 |           16.75 |               15.25 |                 0.208333 |          0.719112 |              0.639972 |               0.024534 |
| sample_static        | 2023.5 |      0.728388 |          0.549482 |           0.684903 |         0.67542  |             6.75 |           16.75 |               15.25 |                 0.208333 |          0.719112 |              0.639972 |               0.024534 |

## Controller diagnostics

| arm_key              |   calibration_error |   coefficient_variation | ema_tracking_error   |   mismatch |   gradient_variance |   gradient_snr |   update_alignment |
|:---------------------|--------------------:|------------------------:|:---------------------|-----------:|--------------------:|---------------:|-------------------:|
| adaptive_ema         |            0.079997 |                0.016161 | 0.486913             |   0.079997 |            0.00048  |        11.8992 |           0.091094 |
| adaptive_ema_uniform |            0.11915  |                0.027323 | 0.471643             |   0.51003  |            0.000875 |        21.6702 |           0.086187 |
| adaptive_noema       |            0        |                0.08866  |                      |   0        |            0.00038  |        13.8245 |           0.084907 |
| equal_static         |            0.621817 |                0        |                      |   0.621817 |            0.001896 |        19.4514 |           0.094455 |
| sample_ema_control   |            0.096592 |                0        | 0.561041             |   0.096592 |            0.000406 |        19.762  |           0.09116  |
| sample_static        |            0.096592 |                0        |                      |   0.096592 |            0.000406 |        19.762  |           0.09116  |

## Paired comparisons

| candidate            | baseline           | metric                 |   improvement |      std |    median |       min |       max |    ci_low |   ci_high |   positive_seeds | evidence   |
|:---------------------|:-------------------|:-----------------------|--------------:|---------:|----------:|----------:|----------:|----------:|----------:|-----------------:|:-----------|
| adaptive_noema       | sample_static      | overall_auc            |      0.000795 | 0.002458 |  0.001767 | -0.002853 |  0.002498 | -0.001665 |  0.002283 |                3 | WEAK       |
| adaptive_noema       | sample_static      | overall_logloss        |      0.010177 | 0.028085 |  0.001072 | -0.012428 |  0.050992 | -0.008193 |  0.037652 |                2 | WEAK       |
| adaptive_noema       | sample_static      | worst_domain_auc       |      0.00329  | 0.003153 |  0.003842 | -0.000724 |  0.006199 |  0.000797 |  0.005782 |                3 | MEDIUM     |
| adaptive_noema       | sample_static      | negative_transfer_rate |     -0.041667 | 0.083333 | -0        | -0.166667 | -0        | -0.125    |  0        |                0 | WEAK       |
| adaptive_noema       | sample_static      | last_common_auc        |      0.003213 | 0.003032 |  0.002917 | -0.000163 |  0.007182 |  0.000697 |  0.006025 |                3 | MEDIUM     |
| adaptive_noema       | sample_static      | last_common_logloss    |      0.033275 | 0.016331 |  0.029266 |  0.018973 |  0.055594 |  0.021263 |  0.047584 |                4 | STRONG     |
| adaptive_ema         | sample_static      | overall_auc            |      0.001584 | 0.002406 |  0.001926 | -0.001664 |  0.004149 | -0.000752 |  0.003578 |                3 | MEDIUM     |
| adaptive_ema         | sample_static      | overall_logloss        |      0.012788 | 0.028742 |  0.004802 | -0.012549 |  0.054099 | -0.007956 |  0.041519 |                3 | MEDIUM     |
| adaptive_ema         | sample_static      | worst_domain_auc       |      0.00409  | 0.00377  |  0.003503 |  0.000849 |  0.008503 |  0.000953 |  0.007226 |                4 | STRONG     |
| adaptive_ema         | sample_static      | negative_transfer_rate |      0        | 0        | -0        | -0        | -0        |  0        |  0        |                0 | WEAK       |
| adaptive_ema         | sample_static      | last_common_auc        |      0.003159 | 0.000856 |  0.003355 |  0.002035 |  0.003892 |  0.002465 |  0.003824 |                4 | STRONG     |
| adaptive_ema         | sample_static      | last_common_logloss    |      0.027603 | 0.005227 |  0.027318 |  0.021668 |  0.034105 |  0.023447 |  0.032042 |                4 | STRONG     |
| adaptive_noema       | sample_ema_control | overall_auc            |      0.000795 | 0.002458 |  0.001767 | -0.002853 |  0.002498 | -0.001665 |  0.002283 |                3 | WEAK       |
| adaptive_noema       | sample_ema_control | overall_logloss        |      0.010177 | 0.028085 |  0.001072 | -0.012428 |  0.050992 | -0.008193 |  0.037652 |                2 | WEAK       |
| adaptive_noema       | sample_ema_control | worst_domain_auc       |      0.00329  | 0.003153 |  0.003842 | -0.000724 |  0.006199 |  0.000797 |  0.005782 |                3 | MEDIUM     |
| adaptive_noema       | sample_ema_control | negative_transfer_rate |     -0.041667 | 0.083333 | -0        | -0.166667 | -0        | -0.125    |  0        |                0 | WEAK       |
| adaptive_noema       | sample_ema_control | last_common_auc        |      0.003213 | 0.003032 |  0.002917 | -0.000163 |  0.007182 |  0.000697 |  0.006025 |                3 | MEDIUM     |
| adaptive_noema       | sample_ema_control | last_common_logloss    |      0.033275 | 0.016331 |  0.029266 |  0.018973 |  0.055594 |  0.021263 |  0.047584 |                4 | STRONG     |
| adaptive_ema         | sample_ema_control | overall_auc            |      0.001584 | 0.002406 |  0.001926 | -0.001664 |  0.004149 | -0.000752 |  0.003578 |                3 | MEDIUM     |
| adaptive_ema         | sample_ema_control | overall_logloss        |      0.012788 | 0.028742 |  0.004802 | -0.012549 |  0.054099 | -0.007956 |  0.041519 |                3 | MEDIUM     |
| adaptive_ema         | sample_ema_control | worst_domain_auc       |      0.00409  | 0.00377  |  0.003503 |  0.000849 |  0.008503 |  0.000953 |  0.007226 |                4 | STRONG     |
| adaptive_ema         | sample_ema_control | negative_transfer_rate |      0        | 0        | -0        | -0        | -0        |  0        |  0        |                0 | WEAK       |
| adaptive_ema         | sample_ema_control | last_common_auc        |      0.003159 | 0.000856 |  0.003355 |  0.002035 |  0.003892 |  0.002465 |  0.003824 |                4 | STRONG     |
| adaptive_ema         | sample_ema_control | last_common_logloss    |      0.027603 | 0.005227 |  0.027318 |  0.021668 |  0.034105 |  0.023447 |  0.032042 |                4 | STRONG     |
| adaptive_ema         | adaptive_noema     | overall_auc            |      0.000789 | 0.000754 |  0.00071  |  8.7e-05  |  0.001651 |  0.000159 |  0.00142  |                4 | WEAK       |
| adaptive_ema         | adaptive_noema     | overall_logloss        |      0.002611 | 0.004086 |  0.001493 | -0.000734 |  0.008194 | -0.000428 |  0.006115 |                2 | WEAK       |
| adaptive_ema         | adaptive_noema     | worst_domain_auc       |      0.0008   | 0.002056 |  0.000766 | -0.00147  |  0.003138 | -0.00086  |  0.00246  |                2 | WEAK       |
| adaptive_ema         | adaptive_noema     | negative_transfer_rate |      0.041667 | 0.083333 | -0        | -0        |  0.166667 |  0        |  0.125    |                1 | WEAK       |
| adaptive_ema         | adaptive_noema     | last_common_auc        |     -5.4e-05  | 0.002859 |  0.000908 | -0.004229 |  0.002198 | -0.002872 |  0.001802 |                3 | WEAK       |
| adaptive_ema         | adaptive_noema     | last_common_logloss    |     -0.005672 | 0.014626 | -0.001379 | -0.02681  |  0.00688  | -0.020326 |  0.004689 |                1 | WEAK       |
| adaptive_ema_uniform | adaptive_ema       | overall_auc            |     -0.01494  | 0.007968 | -0.012826 | -0.025485 | -0.008624 | -0.021345 | -0.008774 |                0 | WEAK       |
| adaptive_ema_uniform | adaptive_ema       | overall_logloss        |     -0.083418 | 0.074283 | -0.078965 | -0.178201 |  0.00246  | -0.15119  | -0.020099 |                1 | WEAK       |
| adaptive_ema_uniform | adaptive_ema       | worst_domain_auc       |     -0.03309  | 0.023033 | -0.024839 | -0.067004 | -0.015676 | -0.056215 | -0.018215 |                0 | WEAK       |
| adaptive_ema_uniform | adaptive_ema       | negative_transfer_rate |     -0.166667 | 0.235702 | -0.083333 | -0.5      | -0        | -0.375    |  0        |                0 | WEAK       |
| adaptive_ema_uniform | adaptive_ema       | last_common_auc        |     -0.022575 | 0.005846 | -0.025242 | -0.025975 | -0.01384  | -0.025841 | -0.016807 |                0 | WEAK       |
| adaptive_ema_uniform | adaptive_ema       | last_common_logloss    |     -0.236335 | 0.027085 | -0.240112 | -0.264535 | -0.200582 | -0.256673 | -0.212221 |                0 | WEAK       |
| sample_ema_control   | sample_static      | overall_auc            |      0        | 0        |  0        |  0        |  0        |  0        |  0        |                0 | WEAK       |
| sample_ema_control   | sample_static      | overall_logloss        |      0        | 0        | -0        | -0        | -0        |  0        |  0        |                0 | WEAK       |
| sample_ema_control   | sample_static      | worst_domain_auc       |      0        | 0        |  0        |  0        |  0        |  0        |  0        |                0 | WEAK       |
| sample_ema_control   | sample_static      | negative_transfer_rate |      0        | 0        | -0        | -0        | -0        |  0        |  0        |                0 | WEAK       |
| sample_ema_control   | sample_static      | last_common_auc        |      0        | 0        |  0        |  0        |  0        |  0        |  0        |                0 | WEAK       |
| sample_ema_control   | sample_static      | last_common_logloss    |      0        | 0        | -0        | -0        | -0        |  0        |  0        |                0 | WEAK       |
| adaptive_ema         | equal_static       | overall_auc            |      0.014523 | 0.00706  |  0.015797 |  0.005531 |  0.020965 |  0.008935 |  0.02011  |                4 | STRONG     |
| adaptive_ema         | equal_static       | overall_logloss        |      0.060681 | 0.048145 |  0.062835 |  0.009951 |  0.107104 |  0.01975  |  0.101612 |                4 | STRONG     |
| adaptive_ema         | equal_static       | worst_domain_auc       |      0.026936 | 0.024245 |  0.021899 |  0.004918 |  0.059025 |  0.008431 |  0.047255 |                4 | STRONG     |
| adaptive_ema         | equal_static       | negative_transfer_rate |      0.083333 | 0.096225 |  0.083333 | -0        |  0.166667 |  0        |  0.166667 |                2 | WEAK       |
| adaptive_ema         | equal_static       | last_common_auc        |      0.025452 | 0.00605  |  0.025429 |  0.019954 |  0.030997 |  0.02022  |  0.030684 |                4 | STRONG     |
| adaptive_ema         | equal_static       | last_common_logloss    |      0.244937 | 0.024222 |  0.243595 |  0.219937 |  0.272619 |  0.22501  |  0.264864 |                4 | STRONG     |

## Primary seed deltas

|   seed |   adaptive_noema_minus_sample |   adaptive_ema_minus_sample |   ema_minus_noema |
|-------:|------------------------------:|----------------------------:|------------------:|
|   2022 |                     -0.002853 |                   -0.001664 |          0.001188 |
|   2023 |                      0.001897 |                    0.001984 |          8.7e-05  |
|   2024 |                      0.002498 |                    0.004149 |          0.001651 |
|   2025 |                      0.001637 |                    0.001867 |          0.000231 |

## Role/task safety

| candidate            | baseline      | metric            |   improvement |      std |    median |       min |       max |    ci_low |   ci_high |   positive_seeds | evidence   | role   | scope   | task     |
|:---------------------|:--------------|:------------------|--------------:|---------:|----------:|----------:|----------:|----------:|----------:|-----------------:|:-----------|:-------|:--------|:---------|
| adaptive_noema       | sample_static | auc               |      0.001133 | 0.004163 |  0.001082 | -0.003712 |  0.00608  | -0.002158 |  0.004476 |                2 | WEAK       | L      | role    |          |
| adaptive_noema       | sample_static | transfer          |      0.001133 | 0.004163 |  0.001082 | -0.003712 |  0.00608  | -0.002158 |  0.004476 |                2 | WEAK       | L      | role    |          |
| adaptive_noema       | sample_static | logloss           |     -0.000515 | 0.011467 |  0.003367 | -0.017202 |  0.008409 | -0.011618 |  0.00677  |                3 | WEAK       | L      | role    |          |
| adaptive_noema       | sample_static | negative_transfer |     -0.125    | 0.25     | -0        | -0.5      | -0        | -0.375    |  0        |                0 | WEAK       | L      | role    |          |
| adaptive_noema       | sample_static | auc               |     -0.002039 | 0.0042   | -0.00154  | -0.007496 |  0.002423 | -0.005715 |  0.00114  |                1 | WEAK       | M      | role    |          |
| adaptive_noema       | sample_static | transfer          |     -0.002039 | 0.0042   | -0.00154  | -0.007496 |  0.002423 | -0.005715 |  0.00114  |                1 | WEAK       | M      | role    |          |
| adaptive_noema       | sample_static | logloss           |     -0.003104 | 0.008082 | -0.000217 | -0.014793 |  0.00281  | -0.010658 |  0.002277 |                2 | WEAK       | M      | role    |          |
| adaptive_noema       | sample_static | negative_transfer |      0        | 0        | -0        | -0        | -0        |  0        |  0        |                0 | WEAK       | M      | role    |          |
| adaptive_noema       | sample_static | auc               |      0.00329  | 0.003153 |  0.003842 | -0.000724 |  0.006199 |  0.000797 |  0.005782 |                3 | MEDIUM     | S      | role    |          |
| adaptive_noema       | sample_static | transfer          |      0.00329  | 0.003153 |  0.003842 | -0.000724 |  0.006199 |  0.000797 |  0.005782 |                3 | MEDIUM     | S      | role    |          |
| adaptive_noema       | sample_static | logloss           |      0.03415  | 0.072907 |  0.001918 | -0.01006  |  0.142824 | -0.007675 |  0.105796 |                2 | WEAK       | S      | role    |          |
| adaptive_noema       | sample_static | negative_transfer |      0        | 0        | -0        | -0        | -0        |  0        |  0        |                0 | WEAK       | S      | role    |          |
| adaptive_noema       | sample_static | auc               |      0.001261 | 0.002239 |  0.001458 | -0.001208 |  0.003338 | -0.000629 |  0.003152 |                2 | WEAK       | nan    | task    | 0.000000 |
| adaptive_noema       | sample_static | transfer          |      0.001261 | 0.002239 |  0.001458 | -0.001208 |  0.003338 | -0.000629 |  0.003152 |                2 | WEAK       | nan    | task    | 0.000000 |
| adaptive_noema       | sample_static | logloss           |      0.007701 | 0.02208  |  0.000645 | -0.010306 |  0.03982  | -0.007096 |  0.029554 |                2 | WEAK       | nan    | task    | 0.000000 |
| adaptive_noema       | sample_static | auc               |      0.000328 | 0.00338  |  0.001243 | -0.004498 |  0.003324 | -0.002959 |  0.0027   |                3 | WEAK       | nan    | task    | 1.000000 |
| adaptive_noema       | sample_static | transfer          |      0.000328 | 0.00338  |  0.001243 | -0.004498 |  0.003324 | -0.002959 |  0.0027   |                3 | WEAK       | nan    | task    | 1.000000 |
| adaptive_noema       | sample_static | logloss           |      0.012653 | 0.034108 |  0.001499 | -0.01455  |  0.062165 | -0.009289 |  0.04575  |                2 | WEAK       | nan    | task    | 1.000000 |
| adaptive_ema         | sample_static | auc               |      0.002026 | 0.004267 |  0.001832 | -0.002616 |  0.007055 | -0.001373 |  0.005424 |                2 | WEAK       | L      | role    |          |
| adaptive_ema         | sample_static | transfer          |      0.002026 | 0.004267 |  0.001832 | -0.002616 |  0.007055 | -0.001373 |  0.005424 |                2 | WEAK       | L      | role    |          |
| adaptive_ema         | sample_static | logloss           |      0.003205 | 0.013106 |  0.009398 | -0.016433 |  0.010459 | -0.009876 |  0.010127 |                3 | MEDIUM     | L      | role    |          |
| adaptive_ema         | sample_static | negative_transfer |      0        | 0        | -0        | -0        | -0        |  0        |  0        |                0 | WEAK       | L      | role    |          |
| adaptive_ema         | sample_static | auc               |     -0.001363 | 0.003493 | -0.000901 | -0.005921 |  0.002269 | -0.004403 |  0.001214 |                2 | WEAK       | M      | role    |          |
| adaptive_ema         | sample_static | transfer          |     -0.001363 | 0.003493 | -0.000901 | -0.005921 |  0.002269 | -0.004403 |  0.001214 |                2 | WEAK       | M      | role    |          |
| adaptive_ema         | sample_static | logloss           |      0.003136 | 0.005205 |  0.003061 | -0.002991 |  0.009412 | -0.00111  |  0.007456 |                3 | MEDIUM     | M      | role    |          |
| adaptive_ema         | sample_static | negative_transfer |      0        | 0        | -0        | -0        | -0        |  0        |  0        |                0 | WEAK       | M      | role    |          |
| adaptive_ema         | sample_static | auc               |      0.00409  | 0.00377  |  0.003503 |  0.000849 |  0.008503 |  0.000953 |  0.007226 |                4 | STRONG     | S      | role    |          |
| adaptive_ema         | sample_static | transfer          |      0.00409  | 0.00377  |  0.003503 |  0.000849 |  0.008503 |  0.000953 |  0.007226 |                4 | STRONG     | S      | role    |          |
| adaptive_ema         | sample_static | logloss           |      0.032024 | 0.074291 |  0.001947 | -0.018224 |  0.142427 | -0.012145 |  0.106271 |                2 | WEAK       | S      | role    |          |
| adaptive_ema         | sample_static | negative_transfer |      0        | 0        | -0        | -0        | -0        |  0        |  0        |                0 | WEAK       | S      | role    |          |
| adaptive_ema         | sample_static | auc               |      0.001434 | 0.001774 |  0.001597 | -0.000873 |  0.003416 | -0.00019  |  0.002895 |                3 | MEDIUM     | nan    | task    | 0.000000 |
| adaptive_ema         | sample_static | transfer          |      0.001434 | 0.001774 |  0.001597 | -0.000873 |  0.003416 | -0.00019  |  0.002895 |                3 | MEDIUM     | nan    | task    | 0.000000 |
| adaptive_ema         | sample_static | logloss           |      0.009494 | 0.019926 |  0.004233 | -0.008481 |  0.037991 | -0.005163 |  0.029412 |                3 | MEDIUM     | nan    | task    | 0.000000 |
| adaptive_ema         | sample_static | auc               |      0.001734 | 0.003058 |  0.002255 | -0.002456 |  0.004882 | -0.001242 |  0.004189 |                3 | MEDIUM     | nan    | task    | 1.000000 |
| adaptive_ema         | sample_static | transfer          |      0.001734 | 0.003058 |  0.002255 | -0.002456 |  0.004882 | -0.001242 |  0.004189 |                3 | MEDIUM     | nan    | task    | 1.000000 |
| adaptive_ema         | sample_static | logloss           |      0.016083 | 0.037603 |  0.00537  | -0.016617 |  0.070207 | -0.010469 |  0.053347 |                3 | MEDIUM     | nan    | task    | 1.000000 |
| adaptive_ema_uniform | sample_static | auc               |     -0.005016 | 0.002856 | -0.003837 | -0.009266 | -0.003122 | -0.007868 | -0.003342 |                0 | WEAK       | L      | role    |          |
| adaptive_ema_uniform | sample_static | transfer          |     -0.005016 | 0.002856 | -0.003837 | -0.009266 | -0.003122 | -0.007868 | -0.003342 |                0 | WEAK       | L      | role    |          |
| adaptive_ema_uniform | sample_static | logloss           |     -0.00691  | 0.007228 | -0.005692 | -0.016792 |  0.000536 | -0.01387  | -0.001168 |                1 | WEAK       | L      | role    |          |
| adaptive_ema_uniform | sample_static | negative_transfer |     -0.375    | 0.478714 | -0.25     | -1        | -0        | -0.75     |  0        |                0 | WEAK       | L      | role    |          |
| adaptive_ema_uniform | sample_static | auc               |     -0.006053 | 0.006689 | -0.00368  | -0.015577 | -0.001276 | -0.01206  | -0.001392 |                0 | WEAK       | M      | role    |          |
| adaptive_ema_uniform | sample_static | transfer          |     -0.006053 | 0.006689 | -0.00368  | -0.015577 | -0.001276 | -0.01206  | -0.001392 |                0 | WEAK       | M      | role    |          |
| adaptive_ema_uniform | sample_static | logloss           |     -0.012168 | 0.008967 | -0.011093 | -0.024071 | -0.002415 | -0.020569 | -0.004842 |                0 | WEAK       | M      | role    |          |
| adaptive_ema_uniform | sample_static | negative_transfer |      0        | 0        | -0        | -0        | -0        |  0        |  0        |                0 | WEAK       | M      | role    |          |
| adaptive_ema_uniform | sample_static | auc               |     -0.029    | 0.020788 | -0.023886 | -0.058502 | -0.009727 | -0.049574 | -0.013541 |                0 | WEAK       | S      | role    |          |
| adaptive_ema_uniform | sample_static | transfer          |     -0.029    | 0.020788 | -0.023886 | -0.058502 | -0.009727 | -0.049574 | -0.013541 |                0 | WEAK       | S      | role    |          |
| adaptive_ema_uniform | sample_static | logloss           |     -0.19281  | 0.226481 | -0.134551 | -0.498726 | -0.003412 | -0.384206 | -0.022029 |                0 | WEAK       | S      | role    |          |
| adaptive_ema_uniform | sample_static | negative_transfer |     -0.125    | 0.25     | -0        | -0.5      | -0        | -0.375    |  0        |                0 | WEAK       | S      | role    |          |
| adaptive_ema_uniform | sample_static | auc               |     -0.016134 | 0.005386 | -0.016618 | -0.022212 | -0.00909  | -0.0208   | -0.010986 |                0 | WEAK       | nan    | task    | 0.000000 |
| adaptive_ema_uniform | sample_static | transfer          |     -0.016134 | 0.005386 | -0.016618 | -0.022212 | -0.00909  | -0.0208   | -0.010986 |                0 | WEAK       | nan    | task    | 0.000000 |
| adaptive_ema_uniform | sample_static | logloss           |     -0.08904  | 0.09577  | -0.062524 | -0.217067 | -0.014048 | -0.167119 | -0.015662 |                0 | WEAK       | nan    | task    | 0.000000 |
| adaptive_ema_uniform | sample_static | auc               |     -0.010578 | 0.007663 | -0.008715 | -0.02046  | -0.004422 | -0.016639 | -0.004518 |                0 | WEAK       | nan    | task    | 1.000000 |
| adaptive_ema_uniform | sample_static | transfer          |     -0.010578 | 0.007663 | -0.008715 | -0.02046  | -0.004422 | -0.016639 | -0.004518 |                0 | WEAK       | nan    | task    | 1.000000 |
| adaptive_ema_uniform | sample_static | logloss           |     -0.052218 | 0.055851 | -0.039143 | -0.127685 | -0.002902 | -0.100279 | -0.010482 |                0 | WEAK       | nan    | task    | 1.000000 |

## Method gates

| candidate      |   overall_gain |   overall_positive_seeds |   logloss_improvement |   worst_domain_gain |   negative_transfer_reduction |   last_common_gain |   nonnegative_roles |   worst_role_transfer_delta |   both_tasks_positive | GO    | CONDITIONAL   |
|:---------------|---------------:|-------------------------:|----------------------:|--------------------:|------------------------------:|-------------------:|--------------------:|----------------------------:|----------------------:|:------|:--------------|
| adaptive_noema |       0.000795 |                        3 |              0.010177 |             0.00329 |                     -0.041667 |           0.003213 |                   2 |                   -0.002039 |                     2 | False | False         |
| adaptive_ema   |       0.001584 |                        3 |              0.012788 |             0.00409 |                      0        |           0.003159 |                   2 |                   -0.001363 |                     2 | False | False         |

## Controller phases

| arm_key              | phase   |   domain |   coefficient |   actual_share |   target_share |   calibration_error |   coefficient_variation | ema_tracking_error   |
|:---------------------|:--------|---------:|--------------:|---------------:|---------------:|--------------------:|------------------------:|:---------------------|
| adaptive_ema         | early   |        0 |      2.50336  |       0.864468 |       0.8      |            0.091826 |                0.004319 | 41.772350            |
| adaptive_ema         | early   |        1 |      0.461375 |       0.123593 |       0.184    |            0.091826 |                0.004319 | 41.772350            |
| adaptive_ema         | early   |        2 |      0.035265 |       0.011939 |       0.016    |            0.091826 |                0.004319 | 41.772350            |
| adaptive_ema         | middle  |        0 |      2.57531  |       0.772521 |       0.8      |            0.052323 |                0.010044 | 0.254289             |
| adaptive_ema         | middle  |        1 |      0.406669 |       0.204699 |       0.184    |            0.052323 |                0.010044 | 0.254289             |
| adaptive_ema         | middle  |        2 |      0.018026 |       0.02278  |       0.016    |            0.052323 |                0.010044 | 0.254289             |
| adaptive_ema         | late    |        0 |      2.614    |       0.791991 |       0.8      |            0.040745 |                0.008774 | 0.179254             |
| adaptive_ema         | late    |        1 |      0.376906 |       0.191246 |       0.184    |            0.040745 |                0.008774 | 0.179254             |
| adaptive_ema         | late    |        2 |      0.009091 |       0.016763 |       0.016    |            0.040745 |                0.008774 | 0.179254             |
| adaptive_ema_uniform | early   |        0 |      1.25926  |       0.442023 |       0.333333 |            0.183256 |                0.008796 | 43.678719            |
| adaptive_ema_uniform | early   |        1 |      0.934566 |       0.249836 |       0.333333 |            0.183256 |                0.008796 | 43.678719            |
| adaptive_ema_uniform | early   |        2 |      0.806177 |       0.308141 |       0.333333 |            0.183256 |                0.008796 | 43.678719            |
| adaptive_ema_uniform | middle  |        0 |      1.5253   |       0.295192 |       0.333333 |            0.098299 |                0.027629 | 0.261928             |
| adaptive_ema_uniform | middle  |        1 |      1.09482  |       0.322204 |       0.333333 |            0.098299 |                0.027629 | 0.261928             |
| adaptive_ema_uniform | middle  |        2 |      0.379878 |       0.382604 |       0.333333 |            0.098299 |                0.027629 | 0.261928             |
| adaptive_ema_uniform | late    |        0 |      1.83689  |       0.324784 |       0.333333 |            0.078205 |                0.021021 | 0.206241             |
| adaptive_ema_uniform | late    |        1 |      0.872774 |       0.353268 |       0.333333 |            0.078205 |                0.021021 | 0.206241             |
| adaptive_ema_uniform | late    |        2 |      0.290335 |       0.321948 |       0.333333 |            0.078205 |                0.021021 | 0.206241             |
| adaptive_noema       | early   |        0 |      2.2619   |       0.8      |       0.8      |            0        |                0.117486 |                      |
| adaptive_noema       | early   |        1 |      0.687556 |       0.184    |       0.184    |            0        |                0.117486 |                      |
| adaptive_noema       | early   |        2 |      0.050544 |       0.016    |       0.016    |            0        |                0.117486 |                      |
| adaptive_noema       | middle  |        0 |      2.59053  |       0.8      |       0.8      |            0        |                0.091266 |                      |
| adaptive_noema       | middle  |        1 |      0.397145 |       0.184    |       0.184    |            0        |                0.091266 |                      |
| adaptive_noema       | middle  |        2 |      0.012322 |       0.016    |       0.016    |            0        |                0.091266 |                      |
| adaptive_noema       | late    |        0 |      2.6261   |       0.8      |       0.8      |            0        |                0.070589 |                      |
| adaptive_noema       | late    |        1 |      0.365503 |       0.184    |       0.184    |            0        |                0.070589 |                      |
| adaptive_noema       | late    |        2 |      0.008396 |       0.016    |       0.016    |            0        |                0.070589 |                      |
| equal_static         | early   |        0 |      1        |       0.384735 |       0.8      |            0.426261 |                0        |                      |
| equal_static         | early   |        1 |      1        |       0.242546 |       0.184    |            0.426261 |                0        |                      |
| equal_static         | early   |        2 |      1        |       0.37272  |       0.016    |            0.426261 |                0        |                      |
| equal_static         | middle  |        0 |      1        |       0.14443  |       0.8      |            0.676925 |                0        |                      |
| equal_static         | middle  |        1 |      1        |       0.189583 |       0.184    |            0.676925 |                0        |                      |
| equal_static         | middle  |        2 |      1        |       0.665987 |       0.016    |            0.676925 |                0        |                      |
| equal_static         | late    |        0 |      1        |       0.122471 |       0.8      |            0.684803 |                0        |                      |
| equal_static         | late    |        1 |      1        |       0.225695 |       0.184    |            0.684803 |                0        |                      |
| equal_static         | late    |        2 |      1        |       0.651834 |       0.016    |            0.684803 |                0        |                      |
| sample_ema_control   | early   |        0 |      2.4      |       0.834093 |       0.8      |            0.079627 |                0        | 39.530542            |
| sample_ema_control   | early   |        1 |      0.552    |       0.146969 |       0.184    |            0.079627 |                0        | 39.530542            |
| sample_ema_control   | early   |        2 |      0.048    |       0.018938 |       0.016    |            0.079627 |                0        | 39.530542            |
| sample_ema_control   | middle  |        0 |      2.4      |       0.661153 |       0.8      |            0.14257  |                0        | 0.243926             |
| sample_ema_control   | middle  |        1 |      0.552    |       0.264934 |       0.184    |            0.14257  |                0        | 0.243926             |
| sample_ema_control   | middle  |        2 |      0.048    |       0.073913 |       0.016    |            0.14257  |                0        | 0.243926             |
| sample_ema_control   | late    |        0 |      2.4      |       0.612274 |       0.8      |            0.188708 |                0        | 0.195002             |
| sample_ema_control   | late    |        1 |      0.552    |       0.286148 |       0.184    |            0.188708 |                0        | 0.195002             |
| sample_ema_control   | late    |        2 |      0.048    |       0.101577 |       0.016    |            0.188708 |                0        | 0.195002             |
| sample_static        | early   |        0 |      2.4      |       0.834093 |       0.8      |            0.079627 |                0        |                      |
| sample_static        | early   |        1 |      0.552    |       0.146969 |       0.184    |            0.079627 |                0        |                      |
| sample_static        | early   |        2 |      0.048    |       0.018938 |       0.016    |            0.079627 |                0        |                      |
| sample_static        | middle  |        0 |      2.4      |       0.661153 |       0.8      |            0.14257  |                0        |                      |
| sample_static        | middle  |        1 |      0.552    |       0.264934 |       0.184    |            0.14257  |                0        |                      |
| sample_static        | middle  |        2 |      0.048    |       0.073913 |       0.016    |            0.14257  |                0        |                      |
| sample_static        | late    |        0 |      2.4      |       0.612274 |       0.8      |            0.188708 |                0        |                      |
| sample_static        | late    |        1 |      0.552    |       0.286148 |       0.184    |            0.188708 |                0        |                      |
| sample_static        | late    |        2 |      0.048    |       0.101577 |       0.016    |            0.188708 |                0        |                      |

## Oracle difficulty and size

|   domain |   task | size_role   |   domain_size |   domain_share |   oracle_auc |   oracle_auc_std |   oracle_logloss |
|---------:|-------:|:------------|--------------:|---------------:|-------------:|-----------------:|-----------------:|
|        0 |      0 | L           |        150000 |          0.8   |     0.750268 |         0.001807 |         0.616017 |
|        0 |      1 | L           |        150000 |          0.8   |     0.762286 |         0.0036   |         0.426187 |
|        1 |      0 | M           |         34500 |          0.184 |     0.703408 |         0.004011 |         0.679265 |
|        1 |      1 | M           |         34500 |          0.184 |     0.725246 |         0.007731 |         0.459158 |
|        2 |      0 | S           |          3000 |          0.016 |     0.608291 |         0.004204 |         0.676286 |
|        2 |      1 | S           |          3000 |          0.016 |     0.617968 |         0.003916 |         0.505344 |

## Scientific assessment

1. **MEDIUM — Adaptive-EMA gives a small selected-checkpoint overall gain over Sample.** Mean gain is about +0.00158, but it is not a clean universal improvement: the Medium-role transfer declines by about 0.00136, exceeding the safety margin. Adaptive-NoEMA gains only about +0.00079 overall and is practically equivalent to Sample.
2. **STRONG — exact calibration is not sufficient.** NoEMA drives target calibration error and shared mismatch essentially to zero, yet fails the >0.001 overall-AUC gate. Sample already has modest mismatch (~0.097), so further exact matching yields little performance benefit and shifts performance between roles.
3. **STRONG — EMA smooths coefficients but is not a justified method component.** Adaptive-EMA reduces coefficient variation from about 0.0887 to 0.0162, but its AUC advantage over NoEMA is only about 0.00079, below the practical margin. Because neither adaptive method passes the method gate, the correct conclusion is `EMA not justified because the adaptive method failed`.
4. **STRONG — the size prior matters for this controller, but does not validate the method.** Uniform-target Adaptive-EMA is much worse than data-share Adaptive-EMA on overall and worst-domain AUC and has high mismatch. This falsifies the without-size-prior alternative within the tested controller, while leaving size versus physical-domain difficulty causally unresolved in canonical d0_large.
5. **No Small-improvement/Large-decline trade-off for Adaptive-EMA, but a Medium cost appears.** Relative to Sample, Adaptive-EMA improves Large and Small transfer while decreasing Medium transfer beyond 0.001. The benefit is redistribution rather than uniform improvement.
6. **Checkpoint and bookkeeping explanations are not sufficient.** Sample-static equals Sample-EMA-control, and last-common comparisons do not reveal a hidden robust advantage large enough to rescue the failed safety gate.
7. **Competing explanations:** C1 direct matching is weakened because perfect calibration fails; C2 noise remains plausible because NoEMA variation is high; C3 EMA lag explains smoother but imperfect matching; C4 size prior is supported within-family; C5 private/routing state remains unmeasured as a mediator; C6 checkpoint-only is weakened; C7 one-role redistribution is supported by the Medium decline; C8 physical identity/difficulty is unresolved by one rotation; C9 optimizer/history dependence remains plausible.
8. Domain size remains a plausible allocation prior, not a proven main performance cause. Frozen oracle AUC/LogLoss show operational difficulty differs by task and physical domain; Small is not treated as synonymous with difficult.

## GO / CONDITIONAL / STOP

**STOP — Method too complex for gain.** Neither Adaptive-NoEMA nor Adaptive-EMA passes GO or CONDITIONAL. Adaptive-NoEMA is overall-equivalent without a qualifying robustness gain; Adaptive-EMA clears the mean overall threshold but violates the remaining-domain safety requirement through Medium transfer degradation. The method only beats Equal reliably, not the strong Sample baseline without trade-offs.

**EMA conclusion:** `EMA not justified because the adaptive method failed`. **Curriculum conclusion:** `Curriculum unnecessary / dropped by E5`.

Do not add controller inputs, tune EMA decay, clip coefficients, blend targets, or revive schedules. If E7 has an independent generalization purpose, it must use static alpha=1 as the primary method and may treat these adaptive results only as a negative ablation; otherwise E7's adaptive-method gate is not met.

## Reproducibility

Tables: `experiments/stage_E6_analysis_tables/`; figures: `experiments/stage_E6_analysis_figures/`.
