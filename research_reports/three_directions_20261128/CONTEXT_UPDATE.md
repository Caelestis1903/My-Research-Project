# EXP083 · 研究上下文更新

2026-11-28 的三方向研究完成 Validation。A 的 clean/Gaussian 配对损失、B 的 γ=.5 PAIR 梯度新鲜度校正和 C 的历史正确配对记忆均未晋级。C0 的 Train 诊断通过并触发 C1。A2、B2 的 γ=.25 确认、C4 和 S4 按预设规则记为 `NOT_TRIGGERED`。研究状态为 `VALIDATION_COMPLETE_TEST_PENDING_AUTHORIZATION`，新 Test 访问为 0。 [结项报告](FINAL_REPORT.md) · [运行清单](../../artifacts/three_directions_20261128/RUN_MANIFEST.json) · [Validation 冻结](../../artifacts/three_directions_20261128/VALIDATION_FREEZE.json)

| 证据 | 后续设计应保留的约束 |
| --- | --- |
| **STRONG · A1** 两数据的固定 EMA 在 E10 均为 5/5 seed 配对 AUC 改善，P0−B0、GP0−G0 四项 selected EMA AUC 均值均为负。KR 的 GP0−G0 两读出均为 5/5 seed AUC 下降与 LL 上升。 | 记录共同轮次和 各臂独立选点结果。Gaussian BCE 和配对端点加噪分开控制，配对 ID、剂量和 seed 匹配。 |
| **STRONG · B1** γ=.5 的 B2−B1 四组 selected 均值低于 +.001，三组跌破六格保护线。**MEDIUM** · KR A21−普通 PAIR Adam 在 E10、E50 和 selected 的方向或幅度有别。 | 同 PAIR 普通 Adam 与 A21 分别用于判断历史路径和校正的增量。固定参数 EMA 作为读出保留，与训练中的辅助梯度历史分别归因。 |
| **STRONG · C0** ML/KR 的 B0 E10→selected 历史正确配对翻转率为 2.947%/11.246%，P0 为 2.788%/8.334%。**MEDIUM** · 早期 margin 越小，翻转越多。 | 观察时距、同一配对集合与各臂自身正确子集分开记录。精确配对重合低支持可区分性，端点及标签监督的可替代性由训练对照判断。 |
| **STRONG · C1** C3−C2 四组 selected 均值都低于 +.001，KR 两读出的逐格保护均失败。C2/C3 相对持续当前配对监督在 E50 改善 AUC 和 LL，selected 优势缩小。ML online 的随机软记忆 C2−C1 为 +.001076，5/5 seed 为正，六格通过。 | 继续配对监督和相同容量的随机软记忆构成两个直接参照。归因软目标需要同一配对身份、剂量及抽样流的硬目标对照，C4 本轮未触发。 |
| **WEAK · 域规模主解释** KR 的 A1 固定 EMA 存在 d0 两格升、最大域 d1 两格降、最小域 d2 一升一降，B 和 C 的损益随任务、读出和参照变化。 | 以 Train 行数记录规模，以同人口基线 AUC、LL、标签支持记录表现难度，并保留逐格和时点分析。 |

[A1 复核](A1_RESEARCH_REVIEW.md) · [B1 复核](B1_RESEARCH_REVIEW.md) · [C0 复核](C0_RESEARCH_REVIEW.md) · [C1 分析](C1_ANALYSIS.md) · [C1 配对数表](../../artifacts/three_directions_20261128/C1/paired_summary.csv) · [数据审计](../../artifacts/three_directions_20261128/DATA_AUDIT.json)

EXP083 使用 seeds 2022–2026，新增 90 条训练轨迹，共 4,500 个 epoch 和 155,250 次更新。C0 的 20 条离线诊断和 C1 链接的 20 条对照均来自 A1。分析覆盖 M3oE fixed1 的两个自然数据集。归档 EXP080 的已暴露 Test 和用户补充 EXP082 的历史证据沿用原来源身份，当前匹配结果按 Validation 记录。归档执行源码提交为 `70455ed0e`，最终交付提交待记录。 [源码归档](../../artifacts/three_directions_20261128/A_B_SOURCE_SNAPSHOT.json) · [历史来源说明](../../research_plans/26-11-28_three_directions/00_MASTER_PLAN.md#1-任务背景和现有证据)

[研究索引](../../mdmt_research_context/01_AI_EXPERIMENT_INDEX.md) · [实验详情](../../mdmt_research_context/02_AI_EXPERIMENT_DETAILS.md#exp083) · [指标与原表](../../mdmt_research_context/03_EXPERIMENT_RESULTS.md#exp083--三方向-validation)
