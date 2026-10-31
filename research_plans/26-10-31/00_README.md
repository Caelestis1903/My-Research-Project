# 2026-10-31｜梯度调节、数据增强与动量优化联合研究

**修订 v2 已冻结首轮执行协议，可交付 Codex 实现、验收并运行。** 本次只修订规划，没有启动训练、修改历史报告或声称新增实验收益。

## 交付入口与阅读顺序

直接把 [04_CODEX执行提示词.md](04_CODEX执行提示词.md) 交给 Codex；其执行合同是 [03_首轮实验与Codex实施规范.md](03_首轮实验与Codex实施规范.md)，冻结参数和数据身份见 [stage0_protocol.json](stage0_protocol.json)。

背景依次阅读：

1. [01｜历史证据与研究问题](01_历史证据与研究问题.md)：Sample、增强、S0/S5/S6/S8 历史证据。
2. [02｜候选算法与相关工作](02_候选算法与相关工作.md)：八类研究候选和数学约束，不是自动续跑指令。
3. [05｜修订说明与核验](05_修订说明与核验.md)：修订原因、来源选择和本次核验范围。

## 当前研究决策

优先回答：**在保持 Sample 原样本和领域任务权重预算时，固定增强是否有超过统一损失缩放的 selected AUC 增量？** 增强改变输入风险，不能声称完整 Sample 经验目标不变。

第一轮使用自然 MovieLens/KuaiRand、fixed1 M3oE、单 Adam、相同原样本 shuffle，以“增强开/关×loss_scale 1/0.1”四臂、四配对种子完成 32 次正式训练，附离线只读探针。增强为最小训练域的一份非路由稀疏 Embedding 相对高斯视图，λ=0.5、κ=0.1，原样本两视图权重和为 1。

首轮结束后停止。SAGU 是梯度控制主候选；SAOS、SGVS 保留为备选。SAOS 是否进入下一轮可由增强和历史诊断直接决定，不要求 SAGU 先成功。同 β、固定系数的一阶 EMA 分拆再混合完全等价，不能当作独立优化机制。

## 历史依据和边界

主要依据：experiments/Experimental_Stable_Findings.md、Experimental_Global_Summary.md、Experimental_Branch_Synthesis_20261026.md，experiments/reports/stage_S0/S5/S6/S8_analysis.md，以及 history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/。

10-21 D 使用 legacy_group_sum，支持旧增强组合的经验价值，不能直接作为新 Sample 增强的独立因果证据。新协议是明确界定的一种增强策略，阴性不推翻已观察到的旧组合收益。执行准备度与性能有效性是两件事；本次完成前者，后者由正式实验判断。

