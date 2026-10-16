# 2026-10-16 v3：Objective-Equivalent Sampling 的终局闭环实验方案

## 1. 目的

本目录不是 v2 的简单续写，而是把当前项目中已经确认、尚未确认、曾经下过但证据不完整、以及 v2 新暴露出的所有关键科学问题重新组织成一套“终局闭环”实验计划。

核心目标不是证明某个预设机制，而是：

1. 先确认 Objective-Equivalent Sampling 现象是否真实、是否实现正确；
2. 严格区分训练轨迹、checkpoint selection、过拟合、优化预算和真正的泛化差异；
3. 将 domain exposure、sample repetition、coverage、batch order、gradient estimator geometry 分开；
4. 继续把 Adam 从“默认根因”降级为 optimizer interaction，只有出现必要证据才重新进入 state-level 分解；
5. 判断 shared representation / routing 等 MDMT 参数块是否只是相关，还是必要 mediator；
6. 检查 domain size、difficulty、task/domain imbalance 是否真的解释现象；
7. 在第二环境验证边界和可迁移性；
8. 正面检查 SAF / D3 / AdaTask 及相关 sampling/order/optimizer-memory 文献是否已经覆盖机制；
9. 最后才决定是否值得方法化；
10. 无论成功、失败、边界还是矛盾，都形成终局结论。

## 2. 与 v2 的关键关系

v2 已经给出了一个重要转向：

- S5–S8 已将“Equal objective 下 Adam history 是旧问题的根因，因此设计 Adam correction”这条路线基本收束；
- v2 P1 在 Sample objective 下发现 SB-SN 在 SGD 上仍为 4/4 负向，而 Adam 为 3/4 负向，故 sampling effect 不需要 Adam 才能出现；
- v2 的循环采样、第二环境、AdaTask、P9 反事实和 P10 方法化尚未完整闭环；
- v2 同时暴露了新的关键替代解释：早期学习速度差异、晚期过拟合、checkpoint selection、优化器校准边界、repetition/coverage/order 混杂、梯度协方差几何以及 shared/router 参数块。

因此 v3 **不重新假设 Adam 是根因**，也不把“后期退化减慢”自动视为有价值的方法效果。

## 3. 最终研究问题

核心问题：

> 在 MDMT/M3OE 中，当目标函数的期望保持不变，仅改变 domain sampling distribution 时，为什么训练轨迹和性能仍可能改变？这种差异是有限样本 stochastic estimator 的结果、数据 exposure/repetition/order 的结果、参数共享结构的结果、optimizer interaction 的结果，还是仅仅是 checkpoint selection / overfitting / optimization-budget artifact？

最终必须给出以下类型之一的结论：

- 已被实验确认；
- 已被实验否证；
- 只在特定边界成立；
- 只能得到关联而不能得到因果识别，并明确指出识别限制；
- 当前实验无法解决，但已经完成最小必要实验并把新问题完整登记。

## 4. 无限开放问题原则

本计划**不限制实验结束后暴露的新问题数量**。

原始问题必须逐项闭环。实验过程中出现 Q15、Q16、Q17……或更多问题时，必须原样登记，不得为了“看起来完成”而压缩、合并或删除。

若新问题会改变核心结论，则必须继续设计最小区分实验，直到：

- 得到结论；
- 得到明确边界；
- 或证明当前实验系统无法识别，并把“不可识别性”本身作为结果。

若新问题不阻碍核心结论，也必须进入最终报告的“新暴露问题”章节，不能静默丢弃。

## 5. 推荐阅读顺序

1. `EXPERIMENT_PLAN.md`：完整阶段和决策树；
2. `PROTOCOL.md`：统计、selection、overfitting、objective-equivalence 和实验公平性规范；
3. `RUN_MANIFEST.md`：运行矩阵；
4. `QUESTION_REGISTRY.md`：原始问题和动态问题登记；
5. `LITERATURE_BOUNDARY.md`：已有工作覆盖检查；
6. `FINAL_REPORT_SCHEMA.md`：最终必须输出什么。

## 6. 禁止的研究叙事

在证据出现以前，禁止预设：

- “Adam 是根因”；
- “momentum 是根因”；
- “小域就是困难域”；
- “Balanced 一定更差”；
- “variance 更大所以性能更差”；
- “shared representation 一定是 mediator”；
- “selection 导致了全部现象”；
- “晚期曲线回升/下降就是方法有效”；
- “某论文没有做过所以一定有创新”。

只有实验闭环后才能使用因果、机制、创新等强表述。
