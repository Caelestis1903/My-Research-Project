# 开发主比较中两个seed均出现的六格下降

仅按聚合工具固定提示阈值筛选：AUC候选−参照<−.001，或LL>+.002。它不是统计显著性或额外停机规则；所有六格原值与未标记项仍在完整附录。seed是独立单位，selected和24不混合。

来源 `artifacts/mdmt_next_v3/v3_0/analysis_snapshots/P1_AUX_complete/cell_contrasts.json`，SHA256 `3cbe52a4348f2f272590756373214c1849a3e59dc5fc2da954a4a4c163745ef3`。

|数据|比较|view|domain|task|metric|seed3101差|seed3102差|
|---|---|---|---:|---:|---|---:|---:|
| kuairand | L4_BLOCK_META-L0_FEATURE | endpoint | 0 | 1 | logloss | +0.0021155358 | +0.0020639289 |
| kuairand | L4_BLOCK_META-L0_FEATURE | endpoint | 2 | 0 | logloss | +0.0135179870 | +0.0058710397 |
| kuairand | L4_BLOCK_META-L2_BLOCK_UNIFORM | selected | 2 | 1 | logloss | +0.0022579605 | +0.0030509175 |
| kuairand | L4_BLOCK_META-L3_SCALAR_META | endpoint | 0 | 0 | auc | -0.0022276559 | -0.0019736099 |
| movielens | L4_BLOCK_META-L0_FEATURE | endpoint | 0 | 0 | logloss | +0.0156405909 | +0.0141176737 |
| movielens | L4_BLOCK_META-L0_FEATURE | endpoint | 0 | 1 | logloss | +0.0117901589 | +0.0060117389 |
| movielens | L4_BLOCK_META-L0_FEATURE | endpoint | 1 | 0 | logloss | +0.0112051836 | +0.0150756047 |
| movielens | L4_BLOCK_META-L0_FEATURE | endpoint | 1 | 1 | logloss | +0.0093794422 | +0.0075060085 |
| movielens | L4_BLOCK_META-L0_FEATURE | endpoint | 2 | 1 | logloss | +0.0104388584 | +0.0050385783 |
| movielens | L4_BLOCK_META-L0_FEATURE | selected | 2 | 1 | auc | -0.0022827181 | -0.0016710993 |
| movielens | L4_BLOCK_META-L0_FEATURE | selected | 2 | 1 | logloss | +0.0024797365 | +0.0044841993 |
| movielens | R6_PB_HB-R0_BASE | endpoint | 0 | 1 | auc | -0.0023819272 | -0.0021540608 |
