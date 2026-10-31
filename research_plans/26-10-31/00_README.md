# 2026-10-31｜梯度调节、数据增强与动量优化联合研究

本目录是基于 2026-10-31 任务提示词形成的研究规划。本轮只完成历史证据审查、候选算法探索、相关工作对照和首轮实验规划；**没有启动训练，也没有修改既有历史实验报告或代码**。

## 阅读顺序

1. [01｜历史证据与研究问题](01_历史证据与研究问题.md)：Sample objective、10-21 增强结果、S0/S5/S6/S8 动量证据、旧路线假设审查。
2. [02｜候选算法与相关工作](02_候选算法与相关工作.md)：8 类候选设计、核心公式、风险、替代方案、论文近邻与创新边界。
3. [03｜首轮实验与 Codex 实施规范](03_首轮实验与Codex实施规范.md)：推荐实验矩阵、预算、评价指标、决策树、实现开关和停止条件。

## 最终研究决策

优先研究：**Sample 目标不变时，增强样本梯度是否提供真实样本梯度之外的泛化增量，以及它应该如何进入当前更新与 Adam 历史状态。**

主方案为 SAGU（真实/增强梯度增量价值控制）；SAOS（增强感知优化器状态）和 SGVS（Sample 保持的梯度方差采样）为备选。第一轮先以 P0–P3 的 Sample × loss-scale × augmentation 2×2 对照分离增强与统一缩放，再决定是否进入梯度分流和优化器状态研究。

## 主要历史依据

- experiments/Experimental_Stable_Findings.md
- experiments/Experimental_Global_Summary.md
- experiments/Experimental_Branch_Synthesis_20261026.md
- experiments/reports/stage_S0_analysis.md
- experiments/reports/stage_S5_analysis.md
- experiments/reports/stage_S6_analysis.md
- experiments/reports/stage_S8_analysis.md
- history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/

**注意：** 10-21 D 套件采用 legacy_group_sum，不是 Sample；其收益是增强组合的经验依据，不是 Sample 下增强独立有效的证明。