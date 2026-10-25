# CAL可达区间与GEOM固定几何诊断

CAL的feature_prediction_equivalence仅指同一CAL快照自身未clip概率头与实际clip部署的数学等价，不指与另一条L0_FEATURE轨迹的逐样本预测相同。DELETION context来自完整Q_explore缓存，各selected/24各一条，给出总体比例。其他EVAL混有validation及Q、MAIN/META等记录，缺少每call n时不得把它们平均后称总体比例。全部为3101开发诊断，不增加seed。

## 完整Q_explore快照比例

| dataset | seed | epoch | interval_inside_fraction | lower_saturation_fraction | upper_saturation_fraction | zero_alpha_row_task_fraction | zero_alpha_coefficient_fraction | solve_max_abs_error | deployed_max_abs_error | inside_max_abs_correction_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| movielens | 3101 | 12 | 0.2924415171 | 0.1315861791 | 0.5759723186 | 0.08007148653 | 0.5459954739 | 1.110223025e-16 | 5.541369319e-08 | 2.980232239e-08 |
| movielens | 3101 | 24 | 0.04572835565 | 0.4443736076 | 0.5098980069 | 0.02068336308 | 0.5205802321 | 2.775557562e-17 | 1.65309757e-08 | 7.450580597e-09 |
| kuairand | 3101 | 18 | 0.06437237561 | 0.7941519618 | 0.1414756775 | 1.495409197e-05 | 0.4966827035 | 2.775557562e-17 | 1.516309567e-08 | 7.450580597e-09 |
| kuairand | 3101 | 24 | 0.0939116925 | 0.3837718368 | 0.5223165154 | 0.001271097804 | 0.4969020188 | 1.387778781e-17 | 1.11758709e-08 | 7.450580597e-09 |

## 几何fit收据（去除原始行号及中心数组，保留hash）

### movielens

```json
{
  "sklearn_version": "1.6.1",
  "n_init": 5,
  "max_iter": 300,
  "tolerance": 0.0001,
  "fit_only": true,
  "labels_used": false,
  "centers_and_tau_frozen": true,
  "subset_plan": "PCG64(seed+91000+1009*source+400000000), first min(20000,Nfit_source), no labels",
  "sources": [
    {
      "source": 0,
      "n": 20000,
      "rows_hash": "33b1ca32f7191ddc66f7867ad6b35215bd4dc7060242fc2bb6c4d4154e77bfea",
      "features_hash": "a9d6839fe0808085c82f44e8d0b4477a533404bee7a8fee006f2f37e4c63b522",
      "centers_hash": "94f8c1fc4e43f6ccf8e33a5616d6e202a48980cb29b2dd01a015cf7d09f02140",
      "temperature": 1.308123757441847,
      "counts": [
        4953,
        5182,
        4676,
        5189
      ],
      "inertia": 35837.75,
      "iterations": 24,
      "kmeans_seed": 94101,
      "warnings": [],
      "target_entropy": 0.6931471805599453,
      "endpoint_temperatures": [
        0.0005491665989950847,
        54916.65989950846
      ],
      "endpoint_entropies": [
        0.3488224560134001,
        1.386294357775
      ],
      "attained_entropy": 0.6931471805599453,
      "median_positive_squared_distance": 5.491665989950846,
      "unattainable": false
    },
    {
      "source": 1,
      "n": 20000,
      "rows_hash": "efbd99e828fa323254568636f476d4c239bd4ab1c6715bbcbd365971b5daee3e",
      "features_hash": "47ee0634aa24225500e5c6fa44c778fc019639376b9313976f34871554dbdd64",
      "centers_hash": "2f01c523e6064d8a392d94e32be957649c6da98bc27a6e46265cb233ac0ec497",
      "temperature": 1.142340516932074,
      "counts": [
        3793,
        5931,
        4395,
        5881
      ],
      "inertia": 33998.90625,
      "iterations": 36,
      "kmeans_seed": 95110,
      "warnings": [],
      "target_entropy": 0.6931471805599453,
      "endpoint_temperatures": [
        0.0004962713034860464,
        49627.13034860464
      ],
      "endpoint_entropies": [
        0.34891482778691496,
        1.386294356599375
      ],
      "attained_entropy": 0.6931471805599453,
      "median_positive_squared_distance": 4.962713034860464,
      "unattainable": false
    },
    {
      "source": 2,
      "n": 20000,
      "rows_hash": "814aacd45145dace24096ac1b2484e5fd7d9d1673ca923cef374ac00168a658a",
      "features_hash": "a74b30b7f7902281c2f1cd7807b7e1dd02fd6df72f0449d7b8a72f661dd24d05",
      "centers_hash": "aaed5597c59cc70a4784bf4ecf5b45c3482d9e5450424d022092f062885f01e7",
      "temperature": 0.8605596104460183,
      "counts": [
        6653,
        4327,
        5653,
        3367
      ],
      "inertia": 36306.4140625,
      "iterations": 57,
      "kmeans_seed": 96119,
      "warnings": [],
      "target_entropy": 0.6931471805599453,
      "endpoint_temperatures": [
        0.00043900598306181654,
        43900.59830618165
      ],
      "endpoint_entropies": [
        0.3489475020966658,
        1.3862943584034289
      ],
      "attained_entropy": 0.6931471805599454,
      "median_positive_squared_distance": 4.390059830618165,
      "unattainable": false
    }
  ]
}
```

### kuairand

```json
{
  "sklearn_version": "1.6.1",
  "n_init": 5,
  "max_iter": 300,
  "tolerance": 0.0001,
  "fit_only": true,
  "labels_used": false,
  "centers_and_tau_frozen": true,
  "subset_plan": "PCG64(seed+91000+1009*source+400000000), first min(20000,Nfit_source), no labels",
  "sources": [
    {
      "source": 0,
      "n": 20000,
      "rows_hash": "83d62b12e1b93b0d46193d38e2c93697eb142e295a001a15d700ebd7891210f1",
      "features_hash": "cd978e6b83a3a88efba22e735ba3da7e4126a97664ea2dbf074c8ab3cd2828c2",
      "centers_hash": "7cc58161e28ed822b97258444d6fc5536e52e75543a5c4d7e531455af0710b40",
      "temperature": 1.3248800135804788,
      "counts": [
        3994,
        5565,
        6329,
        4112
      ],
      "inertia": 127496.96875,
      "iterations": 41,
      "kmeans_seed": 94101,
      "warnings": [],
      "target_entropy": 0.6931471805599453,
      "endpoint_temperatures": [
        0.001039582402802834,
        103958.2402802834
      ],
      "endpoint_entropies": [
        0.3489383470117556,
        1.3862943604211837
      ],
      "attained_entropy": 0.6931471805599452,
      "median_positive_squared_distance": 10.39582402802834,
      "unattainable": false
    },
    {
      "source": 1,
      "n": 20000,
      "rows_hash": "867a5da9cfc68e05f841466b9622ff7bf1afaec1a80a69d0bf231ddf947c87f3",
      "features_hash": "2c205afed0d8fc03a5ea114ea40672d4bdaecda6392eda2d53a5f930d369b10a",
      "centers_hash": "a4419c90e265c3496c5c094ef0d40c45a55a2f15a7ef6405ba4c15a65367a191",
      "temperature": 1.2608580332255581,
      "counts": [
        4590,
        5749,
        3711,
        5950
      ],
      "inertia": 116854.875,
      "iterations": 73,
      "kmeans_seed": 95110,
      "warnings": [],
      "target_entropy": 0.6931471805599453,
      "endpoint_temperatures": [
        0.0009666977113918112,
        96669.77113918112
      ],
      "endpoint_entropies": [
        0.3488593591784858,
        1.3862943602789572
      ],
      "attained_entropy": 0.6931471805599453,
      "median_positive_squared_distance": 9.666977113918112,
      "unattainable": false
    },
    {
      "source": 2,
      "n": 20000,
      "rows_hash": "3b6e82985db4cc33df6e446e9d62991c6645b6a9705cf88101bffa735d026f2a",
      "features_hash": "b491152dedbcc969153aa3881688012d637ea2444c5d8d92c981f5164697297a",
      "centers_hash": "9817bd69306975fc006978fcaf7ff4456e1b56b439ca9b81a2aed92b3e3ae33b",
      "temperature": 0.9704226302736788,
      "counts": [
        4065,
        4686,
        5742,
        5507
      ],
      "inertia": 109220.734375,
      "iterations": 54,
      "kmeans_seed": 96119,
      "warnings": [],
      "target_entropy": 0.6931471805599453,
      "endpoint_temperatures": [
        0.0008450773928397926,
        84507.73928397926
      ],
      "endpoint_entropies": [
        0.34914581423435387,
        1.3862943604801428
      ],
      "attained_entropy": 0.6931471805599452,
      "median_positive_squared_distance": 8.450773928397926,
      "unattainable": false
    }
  ]
}
```

## 来源

| path | sha256 |
| --- | --- |
| artifacts/mdmt_next_v3/v3_0/analysis_snapshots/P2_MECHANISM_DIAGNOSTICS/movielens_cal_projection_contexts_development_complete.jsonl | 7a74e0f74725eab52c3a72e5fc2e07f98126aa01a1796ea9e9f914912e977e17 |
| artifacts/mdmt_next_v3/v3_0/runs/P2_movielens_LPS_GEOM_3101/geometry_fit_receipt.json | ef170e0e50025d51965122aa1b894d15f76ddea352fffd5211ffd6a6ebdb07a9 |
| artifacts/mdmt_next_v3/v3_0/analysis_snapshots/P2_MECHANISM_DIAGNOSTICS/kuairand_cal_projection_contexts_development_complete.jsonl | fef340ba59b34682860ea2c049f3cd6ba9c96031941966f47b3fbadd71f27e6c |
| artifacts/mdmt_next_v3/v3_0/runs/P2_kuairand_LPS_GEOM_3101/geometry_fit_receipt.json | 271bd78231604a63af5f5951e02fd55e21d07291c311d9658822d17fafb0528d |

## 全部逐call投影日志分页

- [APPENDIX_CAL_PROJECTION_CALLS_001.md](APPENDIX_CAL_PROJECTION_CALLS_001.md)：800条。
- [APPENDIX_CAL_PROJECTION_CALLS_002.md](APPENDIX_CAL_PROJECTION_CALLS_002.md)：800条。
- [APPENDIX_CAL_PROJECTION_CALLS_003.md](APPENDIX_CAL_PROJECTION_CALLS_003.md)：800条。
- [APPENDIX_CAL_PROJECTION_CALLS_004.md](APPENDIX_CAL_PROJECTION_CALLS_004.md)：675条。
