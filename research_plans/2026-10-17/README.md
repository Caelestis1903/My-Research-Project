# My-Research-Project：最新实验分析与下一轮方案

日期：2026-10-17。

本次基于实验分支 **codex/execute-2026-10-16-v3** 的固定提交 [2d04c6de6c92](https://github.com/Caelestis1903/My-Research-Project/tree/2d04c6de6c925eca8563f81b66af3f0e95f99183)。main分析时仍为v3计划版本。交付为分析与新预注册建议，不代表已提交仓库或启动训练。

## 先看这三个判断

1. 现象已经成立于具体配置，但其方向取决于阶段、优化器/LR和评价对象，不能概括成均衡采样普遍有害。
2. calAdam E50的同一test预测在macro AUC下差值为−.009074，在test-count加权单元AUC下为+.003108；短pilot的Adam LR赢家到E50也发生反转。
3. 下一轮优先N0评价分解、N1对称调参、N2域计数控制与N3局部C几何；N4/N5按预先写明的条件触发。历史Equal结论与v3 NO_GO保持不变。

## 文件

|文件|用途|
|---|---|
|EXPERIMENT_ANALYSIS.md|详细现象、数值复核、因果边界与文献判断|
|NEXT_EXPERIMENT_PLAN.md|可交给执行端的N0–N6矩阵、公式、种子、预算、触发与交付要求|
|QUESTION_DISPOSITION.md|原46项与新6项的人类可读去向表|
|QUESTION_DISPOSITION.json|完整保留原46项历史字段，并逐项增加新建议；Q47–Q52状态为OPEN_PLANNED|
|NUMERIC_FINDINGS.json|独立复算的344条baseline汇总、498条交互汇总与评价权重分解|

所有数字来源和论文链接见分析报告。NUMERIC_FINDINGS来自逐seed、逐cell聚合表，不能当作全部原始逐样本预测审计。

## 启动顺序和预算

先完成N0、N6原文差异表与N1 manifest。N1为32条H100对称pilot，接着32–64条H100独立确认；N2通常只有触发后新增16条quota训练。N3主要复用模型做局部计算。N4若触发，使用新的8seed，核心48条共同eta拓扑轨迹，另给明确的同预算调参。详细预算与复用规则以方案为准，不预估未经实际吞吐测量的GPU小时。

原问题状态：14 RESOLVED、1 FALSIFIED、22 BOUNDARY、9 NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN。新问题继续增加不设上限；未触发、被反证、结构不可能与缺乏精度分别记录。

现有test已有历史暴露，不能声称全新holdout。下一轮设计及选择规则先冻结，训练分支由预设diagnostic决定，再统一读取新模型的test结果。