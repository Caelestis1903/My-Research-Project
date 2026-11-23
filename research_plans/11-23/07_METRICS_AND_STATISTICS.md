# 07｜Metrics, Statistics & Decision Gates

## 1. 统计单位

唯一独立训练重复单位为 seed，n=5，seeds 2022–2026。Dataset、epoch、anchor、cell、batch、pair 均不是独立训练重复。所有主要差异使用同 seed 配对。

报告：
- mean paired delta；
- sample SD；
- 95% paired t interval（df=4，t critical=2.776445）；
- 每 seed 原始差；
- 正/负/零 seed 数；
- 六格逐格差；
- online/EMA、selected/E10/E25/E40 分开。

CI 跨零不自动否决；但不得只报告均值而隐藏 seed 分歧。

## 2. AUC/LogLoss

主指标：六格等权 Macro AUC，六格为 d0t0、d0t1、d1t0、d1t1、d2t0、d2t1。

主要性能规则：
- 五 seed 配对平均 Macro AUC 增益 ≥ +0.003，才达到总体 AUC 增益标准。
- 六格每一格的五 seed 平均 ΔAUC ≥ -0.001。
- Macro 达标但某 cell 低于 -0.001：标记 MACRO_GAIN_WITH_CELL_TRADEOFF，不得写整体合格。
- LogLoss 独立报告，负 ΔLogLoss 表示改善；不得用 LogLoss 改善替代 AUC 排序成功。
- Traffic 与 pooled 指标分别报告，不替代六格 Macro。

最终正式基线比较与内部组件增量分开：
1. U→P vs 正式发表模型 M3oE、HyperGate、AutoIFS（只在同数据人口、可比协议下）。
2. U→P vs J_ORIG（完整方案对标准联合训练的总增量）。
3. U→P vs J_ACC（排除梯度累计/减少更新次数解释）。
4. U→P vs P→U（排除一般 block schedule/顺序无关解释）。
5. 内部 Clean+fixed EMA 仅作为内部参照，不列正式发表 baseline。

若算法只超过 M3oE 但低于更强可比发表模型，不宣称超过 SOTA。

## 3. Validation 选点

每 epoch 结束完整评价六格 Validation AUC/LogLoss。
- online、EMA 各自独立选点；
- 主 selected 为 fixed EMA；
- strict greater 更新，平局保留最早 epoch；
- 不 early stop，完整 50 epochs；
- E10/E25/E40 仅作预注册机制分析，不代替 selected。
Test 只能在 freeze 后运行。

## 4. S1 机制指标

每个 anchor/fork：
- 六格 AUC/LogLoss；
- Macro、Traffic、Pooled；
- logit shift、pair margin shift；
- corrected/damaged/unchanged pair；
- 每 cell support；
- U/P 更新范数和实际 Adam 位移；
- 每参数 step/m/v 变化范围；
- optimizer steps、forward/backward counts；
- online/EMA readout；
- V_A/V_B 独立结果。

S1 的重点对照：
- U→P − U_ONLY；
- U→P − P→U；
- U→P − J_ACC；
- U→P − J2；
- P_CELL/P_AXIS/P_FULL 的作用范围差异。

Scope 选择只使用 V_A；V_B 对冻结首选 scope 做确认。

## 5. S1 GO/NO-GO

GO 需满足：
1. 至少一个 scope 在 V_A 的 Macro AUC 均值上同时超过 U_ONLY、反向 P→U 和 J_ACC；
2. 冻结 scope 在 V_B 上 U→P−P→U 与 U→P−J_ACC 的 Macro AUC 均值均为正；
3. 六格平均保护线满足，或只有单一可解释 cell 代价且仍不低于 -0.001；
4. 至少两个 anchor 阶段方向一致，不由单个 epoch 驱动；
5. AUC 有增量，不只是 LogLoss 改善；
6. batch AB/BA 两种映射平均后仍保留顺序优势。

若没有 scope 满足，S1 NO-GO。若只有一个数据集满足，另一个数据集不显著负向，可进入 S2 作为条件性验证，但报告必须限制泛化结论；若两个数据集均为负向，不进入 S2。

## 6. S2 GO/NO-GO

S2 candidate U→P 必须在 Validation：
- Macro AUC 平均高于 J_ORIG、J_ACC、P→U；
- 六格均值不低于 -0.001；
- 至少一个 dataset 的正向增量不依赖单个 seed；
- 两数据集合并等权平均为正；
- AUC 改善不是由 LogLoss/pooled 单指标替代。

如果 U→P 与 J_ACC 相同，说明 block update 未提供额外价值；若 U→P 与 P→U 相同，说明顺序特异性不成立；若只在一个数据集有效，保留条件性方向但不能宣称通用 MDMT 算法。S2 未达到这些条件不开放 Test。

## 7. S3 最终算法资格

Test 结果中：
- 相对最强可比正式发表基线 Macro AUC ≥ +0.003；
- 任一 cell 平均 ΔAUC 不低于 -0.001；
- 相对 J_ORIG 与 J_ACC 的独立增量为正；
- 报告 LogLoss、traffic、pooled、阶段和 seed-level 结果。
若总 AUC 达标但 cell guard 失败，保留结果并如实标记，不通过更改 Macro 定义救援。

## 8. 缺失/异常

- 某 cell AUC 无正负两类支持：标记 NOT_EVALUABLE，不补造标签或重采样。
- 缺少某 seed 完整训练：该 arm 不具备五 seed 资格，修复/重跑同 seed；不换 seed。
- 置信区间跨零：报告不确定性，但不自动否定均值已达到的用户门槛。
- Test 访问前 freeze 未通过：拒绝执行。
