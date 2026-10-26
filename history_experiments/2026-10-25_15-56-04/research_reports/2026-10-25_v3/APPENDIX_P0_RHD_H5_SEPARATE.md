# 分开标识H5记录

只含事件/六格聚合，非逐行数据，不当作独立seed。STATIC在旧C8编码与新C8编码须按各自source解释，不能仅比较整数ID。

## rhd_H5_separate.json

服务器来源 `artifacts/mdmt_next_v3/v3_0/p0/rhd_H5_separate.json`；SHA256 `706e822d8448e7b951307b40803f7d1280f7607dd7f39a67eac5941abb4ff5cd`；8条聚合记录。

| dataset | seed | anchor | horizon | policy | predicted_from | action | improvement_over_STATIC_cells | improvement_over_STATIC_macro | source | sha256 | FREEZE_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| movielens | 3101 | 11 | 5 | CUM | strict H1 prefix anchors0..10; not H5-trained history | 5 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | 0.0000000000 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/diagnostics/S1_movielens_native_KNone_3101/utility_H5_011.npz | 21cf9d9bb45456cd032f93a7cf6f4fed5d53c5fd32f36e75a637873f0ce6b09a | NOT_SCORED_BEFORE12_SELECTION_EVENTS |
| movielens | 3101 | 11 | 5 | EMA | strict H1 prefix anchors0..10; not H5-trained history | 5 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | 0.0000000000 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/diagnostics/S1_movielens_native_KNone_3101/utility_H5_011.npz | 21cf9d9bb45456cd032f93a7cf6f4fed5d53c5fd32f36e75a637873f0ce6b09a | NOT_SCORED_BEFORE12_SELECTION_EVENTS |
| movielens | 3101 | 11 | 5 | EMA_WINDOW32 | strict H1 prefix anchors0..10; not H5-trained history | 5 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | 0.0000000000 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/diagnostics/S1_movielens_native_KNone_3101/utility_H5_011.npz | 21cf9d9bb45456cd032f93a7cf6f4fed5d53c5fd32f36e75a637873f0ce6b09a | NOT_SCORED_BEFORE12_SELECTION_EVENTS |
| movielens | 3101 | 11 | 5 | STATIC | strict H1 prefix anchors0..10; not H5-trained history | 5 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | 0.0000000000 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/diagnostics/S1_movielens_native_KNone_3101/utility_H5_011.npz | 21cf9d9bb45456cd032f93a7cf6f4fed5d53c5fd32f36e75a637873f0ce6b09a | NOT_SCORED_BEFORE12_SELECTION_EVENTS |
| kuairand | 3101 | 11 | 5 | CUM | strict H1 prefix anchors0..10; not H5-trained history | 4 | [1.670769415795803e-05, 4.283097223378718e-05, -0.00020738504827022552, -0.0002735486486926675, 7.652759086340666e-05, 0.0001357142609776929] | -0.0000348589 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/diagnostics/S1_kuairand_native_KNone_3101/utility_H5_011.npz | 8dfdb77ddb2891d266711b0da9bf70409f80f909314397d80548af3d395ba883 | NOT_SCORED_BEFORE12_SELECTION_EVENTS |
| kuairand | 3101 | 11 | 5 | EMA | strict H1 prefix anchors0..10; not H5-trained history | 1 | [0.0017332823481410742, 0.0011750914563890547, 0.004103436411242001, -0.0027557803550735116, -0.0005631645326502621, -0.0060533920041052625] | -0.0003934211 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/diagnostics/S1_kuairand_native_KNone_3101/utility_H5_011.npz | 8dfdb77ddb2891d266711b0da9bf70409f80f909314397d80548af3d395ba883 | NOT_SCORED_BEFORE12_SELECTION_EVENTS |
| kuairand | 3101 | 11 | 5 | EMA_WINDOW32 | strict H1 prefix anchors0..10; not H5-trained history | 1 | [0.0017332823481410742, 0.0011750914563890547, 0.004103436411242001, -0.0027557803550735116, -0.0005631645326502621, -0.0060533920041052625] | -0.0003934211 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/diagnostics/S1_kuairand_native_KNone_3101/utility_H5_011.npz | 8dfdb77ddb2891d266711b0da9bf70409f80f909314397d80548af3d395ba883 | NOT_SCORED_BEFORE12_SELECTION_EVENTS |
| kuairand | 3101 | 11 | 5 | STATIC | strict H1 prefix anchors0..10; not H5-trained history | 0 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | 0.0000000000 | /home/xzf/Code/phase0-3/artifacts/mdmt_next_v2/v2_0/diagnostics/S1_kuairand_native_KNone_3101/utility_H5_011.npz | 8dfdb77ddb2891d266711b0da9bf70409f80f909314397d80548af3d395ba883 | NOT_SCORED_BEFORE12_SELECTION_EVENTS |

