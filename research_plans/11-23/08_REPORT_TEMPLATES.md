# 08｜Report Templates & Deliverables

## 1. 必须交付的机器可读文件

S0：
- SOURCE_MANIFEST.json
- DATA_MANIFEST.json
- MODEL_FIDELITY.json
- PARAMETER_MANIFEST.json
- PARAMETER_SUPPORT.csv
- IMPLEMENTATION_MAP.md
- CPU_TESTS.json
- GPU_SMOKE.json
- RESOURCE_AUDIT.json

S1：
- S1_RUN_MANIFEST.json
- BASELINE_ENDPOINTS.csv
- ANCHOR_INDEX.json
- FORK_ENDPOINTS.csv
- PAIRED_SEEDS.csv
- PAIR_TRANSITIONS.csv
- CELL_RESPONSE.csv
- SCOPE_DECISION.json
- COSTS.csv
- S1_REPORT.md
- S1_REVIEW.md

S2：
- S2_RUN_MANIFEST.json
- EPOCH_VALIDATION.csv
- ENDPOINTS.csv
- PAIRED_SEEDS.csv
- PAIRED_SUMMARY.csv
- SIX_CELL.csv
- COSTS.csv
- CONFIGURATION_FREEZE.json
- S2_DECISION.json
- S2_REPORT.md
- S2_REVIEW.md

S3：
- FINAL_FREEZE.json
- TEST_INTENTS.json
- TEST_EVALUATED.json
- FINAL_REPORT.md
- FINAL_REVIEW.md
- FINAL_DECISION.json
- ARTIFACT_INDEX.json

所有 CSV 必须包含 dataset、arm、seed、readout、epoch/anchor、source hash、data hash、config hash。数表只保存标量/row IDs/hash，不保存模型参数或 optimizer tensors。

## 2. S1 报告结构

1. 当前结论（H1/H2 分别判定）。
2. M3oE 实际参数组及梯度支持。
3. E10/E25/E40 的共享更新响应。
4. P_CELL/P_AXIS/P_FULL 适配结果。
5. U→P 与 P→U、JOINT-ACC、JOINT-2 对照。
6. 六格 AUC/LogLoss 与 pair corrected/damaged。
7. V_A scope 选择与 V_B 确认。
8. seed-level 数据、CI、成本。
9. S1 GO/NO-GO 和适用边界。
10. 不能从本轮推出的结论。

## 3. S2 报告结构

1. 最终 scope 与固定算法公式/伪代码。
2. 四个训练 arm 的实际训练日程。
3. 每 epoch validation 与 selected checkpoint。
4. online/EMA 分开。
5. Macro/六格/Traffic/Pooled/LogLoss。
6. 与 J_ORIG/J_ACC/P→U 的配对比较。
7. 更新次数、前后向次数、wall time、显存和额外成本。
8. S2 GO/NO-GO。
9. 机制是否转化为持续收益。

## 4. 最终报告结构

1. Executive decision。
2. 核心问题与算法定义。
3. M3oE 结构/参数组审计。
4. 训练协议和来源身份。
5. S1 机制识别结果。
6. S2 完整训练结果。
7. 正式发表基线比较。
8. 内部控制与独立增量。
9. 六格代价、LogLoss、traffic、pooled。
10. seed/CI/成本。
11. 机制解释边界。
12. 与 STGU/NTKMTL 等文献的差异。
13. 适用范围、失败条件与下一步唯一建议。

## 5. 最终决策措辞

可用：
- RESPONSE_HETEROGENEITY_SUPPORTED
- POST_SHARED_ADAPTATION_SUPPORTED
- ORDER_SPECIFIC_EFFECT_SUPPORTED
- NO_ACTIONABLE_ADAPTATION_SIGNAL
- LOCAL_EFFECT_NOT_TRANSFERRED_TO_TRAINING
- ALGORITHM_INCREMENT_SUPPORTED
- ALGORITHM_INCREMENT_NOT_SUPPORTED
- DATASET_CONDITIONAL_EFFECT
- TECHNICAL_BLOCKED

不得把“局部一步效应”写成“长期训练收益”；不得将 S1 诊断 fork 数量当作独立训练 seed 数。
