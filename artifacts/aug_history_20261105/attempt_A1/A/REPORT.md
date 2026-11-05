# 11/05 增强响应与历史估计：实验数据报告

A：两数据集、2022–2026 五 seed、每条 50 epoch，15 个非反馈影子；共 10 条在线轨迹。
各候选完整验证，超参数与端点按 validation 冻结。A/C 的状态见 DECISION.json 和后续执行记录。

| 数据集 | 家族 | 验证赢家 |
|---|---|---|
| movielens | A_GROUP | A_GROUP_020_K20 |
| movielens | A_GLOBAL | A_GLOBAL_020_K20 |
| movielens | A_FIXED | A_FIXED_T_FAST |
| movielens | U | F |
| kuairand | A_GROUP | A_GROUP_005_K40 |
| kuairand | A_GLOBAL | A_GLOBAL_020_K20 |
| kuairand | A_FIXED | A_FIXED_S_FAST |
| kuairand | U | U_FAST |

逐 seed、六格、Macro/Traffic/Pooled、selected/E10/E50、配对均差与 95% 区间分别见 ENDPOINTS.csv、PAIRED_SEED.csv、PAIRED_SUMMARY.csv；成本见 COSTS.csv。
HyperGate 沿用来源标记 UNVERIFIED_PROPOSAL。科学解释由主代理结合完整数表撰写。
