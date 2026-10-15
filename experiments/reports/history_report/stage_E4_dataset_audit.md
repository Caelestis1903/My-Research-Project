# Stage E4 Dataset Audit — KuaiRand-Pure

## Scope

本审计是 Stage E4 DESIGN 的只读数据核对，不是实验运行。数据来自现有文件：

- processed dataset：`D:/CODE/data/kuairand/processed/kuai_rand.csv`
- preprocessing reference：`experiments/history_exp/first/p3_prepare.py`
- loader reference：`m3oe/examples/run_kuaishou.py`

没有修改数据、切分、特征、标签或任何源码。

## Dataset and domain identity

- KuaiRand-Pure filtered rows：2,507,666
- domain mapping：
  - local d0 = `tab=0`
  - local d1 = `tab=1`（dominant domain）
  - local d2 = `tab=4`
- tasks：
  - task 0 = `is_click`
  - task 1 = `long_view`
- preprocessing sorts all retained events globally by `(date,time_ms)` and then writes the processed rows in that order.
- existing split is by row order: first 80% train, next 10% validation, final 10% test. It is therefore a chronological split and must not be shuffled or reconstructed before partitioning.

Full-dataset domain counts remain consistent with the earlier pipeline audit:

| domain | rows | full proportion |
|---|---:|---:|
| d0 | 178,087 | 7.10% |
| d1 | 2,236,414 | 89.18% |
| d2 | 93,165 | 3.72% |

## Train/validation/test domain structure

Every row has both task labels, so the sample count below applies independently to task 0 and task 1.

| split | total rows | d0 count / proportion | d1 count / proportion | d2 count / proportion |
|---|---:|---:|---:|---:|
| train | 2,006,132 | 172,782 / 8.612693% | 1,743,898 / 86.928378% | 89,452 / 4.458929% |
| validation | 250,767 | 2,692 / 1.073506% | 246,232 / 98.191548% | 1,843 / 0.734945% |
| test | 250,767 | 2,613 / 1.042003% | 246,284 / 98.212285% | 1,870 / 0.745712% |

The formal E4 weighting coefficients must be computed from the train proportions, not the full-dataset or validation/test proportions.

## Per-domain/task label counts and rates

### Train

| domain | task | samples | positives | positive rate |
|---|---|---:|---:|---:|
| d0 | is_click | 172,782 | 15,704 | 0.090889 |
| d0 | long_view | 172,782 | 7,121 | 0.041214 |
| d1 | is_click | 1,743,898 | 658,901 | 0.377832 |
| d1 | long_view | 1,743,898 | 446,583 | 0.256083 |
| d2 | is_click | 89,452 | 55,180 | 0.616867 |
| d2 | long_view | 89,452 | 43,367 | 0.484807 |

### Validation

| domain | task | samples | positives | positive rate |
|---|---|---:|---:|---:|
| d0 | is_click | 2,692 | 245 | 0.091010 |
| d0 | long_view | 2,692 | 101 | 0.037519 |
| d1 | is_click | 246,232 | 51,644 | 0.209737 |
| d1 | long_view | 246,232 | 27,460 | 0.111521 |
| d2 | is_click | 1,843 | 1,063 | 0.576777 |
| d2 | long_view | 1,843 | 829 | 0.449810 |

### Test

| domain | task | samples | positives | positive rate |
|---|---|---:|---:|---:|
| d0 | is_click | 2,613 | 228 | 0.087256 |
| d0 | long_view | 2,613 | 120 | 0.045924 |
| d1 | is_click | 246,284 | 51,847 | 0.210517 |
| d1 | long_view | 246,284 | 27,415 | 0.111315 |
| d2 | is_click | 1,870 | 1,100 | 0.588235 |
| d2 | long_view | 1,870 | 827 | 0.442246 |

## Temporal distribution-shift assessment

Temporal shift is present and material under the existing chronological split:

1. Domain-mixture total-variation distance is 0.112632 between train and validation and 0.112839 between train and test. Validation and test are nearly identical to each other (TV=0.000315).
2. Dominant d1 rises from 86.93% of train to about 98.2% of validation/test; d0 and d2 jointly fall from 13.07% to about 1.8%.
3. Within d1, `is_click` positive rate falls from 0.377832 in train to 0.209737/0.210517 in validation/test. `long_view` falls from 0.256083 to 0.111521/0.111315.
4. Within d2, both task positive rates also fall; d0 rates are comparatively stable.
5. Overall label rates fall from 0.363777 to about 0.211–0.212 for click and from 0.247776 to about 0.113 for long-view.

These facts establish domain-mixture shift and label-prior shift. Feature-conditional covariate shift and concept shift are not established by this audit and remain unresolved. Because all E4 weighting arms use the same fixed split, temporal shift is controlled across arms but remains a strong competing explanation for why a MovieLens weighting law may fail to transfer.

## Design consequences

- Preserve the exact existing chronological row split.
- Do not rebalance, resample, downsample, rotate roles, or create random splits.
- Treat d1 as dominant by train size; do not call d0/d2 “difficult” merely because they are small.
- Report macro six-cell metrics separately from pooled/micro metrics because validation/test composition is much more d1-dominant than train.
- Any single-domain oracle must first make the global chronological split and only then filter each split by original physical domain. Filtering before the 80/10/10 split would create a different time boundary and is not a matched oracle.
- Weighting coefficients use train shares `(0.0861269348, 0.8692837759, 0.0445892892)`.
