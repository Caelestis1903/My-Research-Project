# 09 — 全机制报告与机器工件模板

## 1. 全局 manifest

RUN_MANIFEST.json：study_id、plan commit、source branch/commit、code hash、data hash、protocol hash、datasets、seeds、anchors、families、variants、arms、panels、expected fork slots、stage statuses。

VARIANT_REGISTER.csv：每行一个 family × variant × dataset applicability，包含假设、action、controls、hyperparameter grid、diagnostic adapter、status。

SEED_MANIFEST.csv：dataset × seed baseline trajectory、source/config/data identity。

SNAPSHOT_MANIFEST.csv：anchor parent、epoch、model/Adam/EMA/RNG/loader hashes、next batch IDs、live readback。

FORK_ENDPOINTS.csv：每个 arm 的 parent state、batch IDs、action receipt、endpoint hash、V_A/V_B metrics、costs、status。

FAMILY_RESULTS.csv：每个 family/variant 的所有 seed 结果、配对差、机制指标和决策。

CROSS_FAMILY_INTERACTIONS.csv：X1–X5 component/combined/control effects。

COSTS.csv：GPU/CPU/wall seconds、peak VRAM、host RAM、steps、forward/backward、EMA ticks、retry。

DECISION_LEDGER.json：每个 family/variant/dataset 的阶段状态与依据。

## 2. 机制专属工件

- M1_GRADIENT_SHAPING.csv
- M2_ROW_TASK_INFLUENCE.csv
- M3_STAGE_GROUP_INTERACTION.csv
- M4_FEATURE_DEPENDENCE.csv
- M5_AUGMENTATION_VALIDITY_AND_EFFECT.csv
- M6_OUTPUT_SENSITIVITY.csv
- M7_DISTRIBUTIONAL_ATTRIBUTION.csv
- M8_TASK_CONFLICT_CONTROL.csv

每个文件必须包含 dataset、seed、anchor、variant、arm、source/config/data hashes。

## 3. 人类报告

- S0_ALL_FAMILY_AUDIT.md
- S1_ALL_MECHANISM_SCREENING.md
- S2_ALL_VARIANT_CONFIRMATION.md
- S3_ALL_SHORT_TRAJECTORIES.md
- S4_ALL_QUALIFIED_FULL_RUNS.md
- CROSS_FAMILY_INTERACTION_REPORT.md
- FINAL_ALL_MECHANISMS_REPORT.md

报告必须按 M1–M8 家族完整列出所有候选，包括负向、无差异、技术阻塞、NOT_APPLICABLE、失败重试和未触发项。不得只写表现最好的方法。

## 4. 最终总结

最终报告不按主观研究优先级排序。使用统一表格比较：
- 机制是否执行；
- 信号是否存在；
- 干预是否改变目标对象；
- V_A/V_B 是否复现；
- 短程轨迹是否持续；
- 六格 AUC/LogLoss；
- 成本；
- 适用边界；
- 后续是否具备完整算法晋级条件。
