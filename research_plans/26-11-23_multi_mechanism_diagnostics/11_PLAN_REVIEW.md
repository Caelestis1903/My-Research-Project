# 11 — 全候选方案独立审查

## 1. 审查结论

**PASS WITH CONDITIONS — 全候选版。**

本版修正此前方案与夜间任务目标不一致的问题：不再对 M1–M8 做主次划分，不再排除 M8，不再将 M4/M5 限制为只读审计，不再让 M3 依赖 M2 阳性结果，也不再只对 Top-K 候选执行确认或短程训练。

## 2. 审查项目

| 审查点 | 本版要求 |
|---|---|
| 是否覆盖第一阶段全部候选 | M1–M8 全部纳入，并拆为多个变体 |
| 是否有主观优先级 | 无；round-robin 公平调度 |
| 是否提前排除候选 | 无；技术阻塞逐变体记录，其他任务继续 |
| 是否有机制间污染 | 每个 arm 独立 state/optimizer/EMA/RNG |
| M3 是否依赖 M2 | 不依赖；同时执行 score-defined 和 random matched groups |
| M4/M5 是否真实测试 | 有实际干预和训练变体；语义无效项标记 NOT_APPLICABLE |
| M8 是否覆盖 | PCGrad reference、adaptive weights、common-descent projection、cell gating |
| 是否仅选 Top-K | 否；S2 全候选，S3 全可训练变体，S4 所有通过者 |
| 是否存在多重比较 | Holm family-level、BH variant-level、交互族单独校正 |
| 是否把局部代理当 AUC | 明确分开 |
| 是否满足正式规则 | 2022–2026、两数据集、50 epoch、Validation 选点、Test freeze |
| 是否可夜间无人值守 | 全 manifest、round-robin、原子 checkpoint、自动恢复、无单个 NO_GO 全局停止 |

## 3. 需要 S0 解决的技术问题

1. D2 RAM states 当前是否仍可 live readback。
2. M1 二阶梯度是否能在实际硬件执行。
3. M2 row-task virtual Adam 计算成本。
4. M4 conditional feature interventions 的字段支持。
5. M5 各变换在真实 schema 下的合法性和标签一致性。
6. M6 Jacobian/trust-region solver 的数值可行性。
7. M7 五 seed 对齐与留一 seed 外推的实现正确性。
8. M8 projection/weight solver 的稳定性。
9. 五组 cross-family combination 的预算与状态隔离。

这些问题按 family/variant 分开审计，不允许一个 family 的问题停止整个任务。

## 4. 修正记录

- 将旧版 M1 primary / M2 secondary 改为 M1–M8 全候选。
- 将旧版 M3 conditional trigger 改为无条件执行的阶段/组交互矩阵。
- 将 M4/M5 feasibility-only 改为实际干预与训练变体；单个不合法变换仍可 NOT_APPLICABLE。
- 将 M8 exclude 改为四种冲突控制变体。
- 将 S2 Top-K confirmation 改为所有候选 V_B readout。
- 将 S3 gated short trajectory 改为所有可训练变体 5-epoch trajectory。
- 将 S4 top candidate 改为所有通过门槛的候选逐一完整训练。
- 增加五组跨机制交互和 2,130 个局部 fork slots。
- 增加 family-round-robin，取消主观调度优先级。

## 5. 审查限制

PASS WITH CONDITIONS 不表示所有变体一定适用于两个数据集，也不表示所有算法预期有效。它表示设计已给出完整执行定义；具体技术适用性由 S0 receipts 决定。NOT_APPLICABLE/TECHNICAL_BLOCKED 必须逐变体留档，不得被隐藏。
