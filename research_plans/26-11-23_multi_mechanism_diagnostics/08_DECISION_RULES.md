# 08 — 全候选决策规则

## 1. 核心原则

M1–M8 全部执行。一个机制的 NO_GO 不得终止其他机制。S1/S2 不做 Top-K，不因筛选结果删减其他候选的 S3。所有可训练变体都执行短程轨迹；S4 对所有通过机制与短程门的变体执行，不设候选数量上限。

## 2. 状态

- PASS：预注册目标达到，执行与身份审计通过。
- NO_GO：实验正常完成，但该具体变体/动作在当前协议下未达到目标。
- TECHNICAL_BLOCKED：代码、数据、资源、状态或算子不支持，无法作科学判断。
- INCONCLUSIVE：运行完成但样本支持、独立 panel 或统计信息不足。
- NOT_APPLICABLE：该变体所需字段/标签在该数据集真实不存在或不适用，必须提供证据。
- NOT_TRIGGERED：仅用于预注册条件性附加读数；不得用于 M3 或 M1–M8 主家族跳过。

## 3. M1–M8 family-specific gates

每个变体分别定义：
1. action execution gate；
2. target signal gate；
3. matched-control contrast；
4. V_A response；
5. V_B independent response；
6. S3 trajectory persistence；
7. cell guard；
8. computational cost。

具体指标按 02_MECHANISM_SELECTION_AND_HYPOTHESES.md 与 06_METRICS_AND_STATISTICS.md。不得把代理指标改善直接写成 AUC 算法收益。

## 4. S1/S2/S3 全量执行

- S1：全部 2,130 slots。
- S2：全部注册变体在 V_B 读取，不只确认 S1 阳性。
- S3：所有可训练变体执行 5 epochs，不以 S1/S2 结果删减。
- 某个变体 TECHNICAL_BLOCKED 时，只隔离该变体；其余变体继续。
- 某个变体 NO_GO 时，仍保留结果并继续该 family 其他变体。

## 5. S4 完整算法晋级

所有满足以下条件的候选逐一晋级，不设 Top-K：
- S2 上有独立、可解释的机制或预测响应；
- S3 有持续、累积或明确阶段条件响应；
- 相对匹配预算控制有可重复差异；
- cell mean guard ≥ −0.001；
- 机制动作可部署，复杂度和训练成本可接受；
- 独立实现审查 PASS。

正式训练 2 datasets × 5 seeds × 50 epochs。正式 AUC 门槛 +0.003。LogLoss 独立评价。Test 只在 freeze 后评价。

## 6. 特殊规则

- M3 无论 M2 是否阳性都执行；其 group pair 包括 score-defined pair 和随机匹配 pair。
- M4 不得将 dependence/HSIC 下降写成因果证明。
- M5 不得将未经验证的字段替换写成语义合法；无效变换标记 NOT_APPLICABLE，但继续其余变换。
- M7 五 seed 分布只描述本项目五条训练轨迹，不外推总体分布。
- M8 的 PCGrad 是历史参考；其他冲突控制变体分别报告，不以 PCGrad 历史负结果排除整个 M8。
- 小域不是困难域。
- 固定 EMA 不重新证明。
