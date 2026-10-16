# V10 已有工作覆盖边界：原始来源审计与已完成对照

本轮启动日文献截止2026-10-16，原始来源9项及具体公式/章节已记录于 `literature_source_receipt.json`。本报告合并该实际来源审计、V1目标审计、V10四种子完整诊断结果和V11已完成的局部影子性能统计；不宣称文献检索穷尽，也不将未完成的V5/V7/完整V11说成已完成。新test仍冻结未读。

| 工作 | 同目标？ | 同干预？ | 已有覆盖及当前边界 |
|---|---|---|---|
| [D3，2024](https://ojs.aaai.org/index.php/AAAI/article/download/28699/29354) | 完整DBO的学习复杂度/特异度权重一般改变目标 | 同时改变权重与架构，不是仅改变q并做p/q校正 | Eq15–19权重机制不能作为相同经验目标的机制等价解释；常数退化关闭其权重机制，不是完整复现。未构造虚假D3替身。 |
| [SAF，2026](https://ojs.aaai.org/index.php/AAAI/article/download/38491/42453) | Eq11–14的学习权重乘积一般改变目标 | 权重、门控与任务优化共同变化 | 不能因同叫sample weighting便声称等价；完整方法可以是另一个目标下的性能方法，不能直接关闭本轮同目标q问题。 |
| [AdaTask，2023](https://arxiv.org/abs/2211.15055) | 接入组件保留任务半和原始梯度，L2各半 | 同目标改变task-wise优化历史，属于相关状态对照 | 已实际做四种子native/AdaTask匹配预算50/100/50/50。早期1–14仍有AUC gap−.007839，三负一正；最终预算gap四负，但两臂绝对AUC优于native。只说明该组件、该固定LR未使q不敏感，不能外推最优完整论文方法普遍无效。 |
| [Importance Sampling for Minibatches，2018](https://www.jmlr.org/papers/v19/16-241.html) | 重要性校正同目标 | 改变采样/mini-batch | 同目标不同q可改变方差和收敛已被覆盖；不能把该一般事实当作新机制。 |
| [Not All Samples Are Created Equal，2018](https://proceedings.mlr.press/v80/katharopoulos18a.html) | 校正每样本梯度 | 梯度范数采样与方差降低 | scalar variance与有限预算速度作用是已有内容；当前CPRE相关谱差异不自动提供新增必要性证据。 |
| [Optimizer Memory Makes Shuffle Order a First-Order Source of Fine-Tuning Noise，2026](https://arxiv.org/html/2606.29554v1) | 相同multiset的局部比较 | 顺序与扩展优化器状态 | 已覆盖一般order-memory路径依赖。局部窗口/暖启动分析不能自动确定从头MDMT的长期唯一mediator；本轮局部m/v影响也不能因场景不同即称创新。 |
| [AdamCB，2025](https://arxiv.org/html/2512.06795v1) | 含无偏估计构造，需保持其约定 | 自适应组合batch选择与Adam | 采样和Adam收敛交互已存在相关先例；不能把二者联合出现当作新颖性。 |
| [Random Reshuffling Dominates Stochastic Gradient Descent，2026](https://arxiv.org/abs/2606.32005) | 同一有限和 | reshuffling与随机采样 | 指定光滑凸设置的结果不是本轮非凸、stopped-routing图梯度的自动推论；重复/顺序主题已有覆盖。 |
| [DIDS，2025](https://arxiv.org/html/2504.13227v1) | 已审查章节未建立本轮逐样本p/q身份 | 自适应域配比/域内采样 | 不能假定改变data mixing就是相同目标。只能记录相关覆盖，不能在未核对目标时声称同构。 |

D3/SAF的“不满足同目标”是明确的公式边界，证据STRONG；这不意味着它们不能改善另一个评价目标，也不构成新颖性证明。AdaTask组件下的残余q敏感性是实际四种子诊断证据MEDIUM，存在明显阶段、指标和种子反例；最终test只作冻结后的补充评估，不再改该条件定义。

本轮已经确认的通用内容——同目标采样的协方差变化、有限更新路径变化、batch顺序与历史状态的局部影响——均有相关先例。潜在剩余问题只能收缩为：在本轮明确配置和边界内，是否有CPRE共享结构或其他可干预通路对下游差距具有独立必要性证据。现有几何相关、局部m/v绝对性能变化及AdaTask残差均不足以满足这一要求。V5、V7和完整V11仍须按规划做完；不能因一般主题已被覆盖而跳过已触发实验，也不能因有残差就宣布未覆盖的新机制。

因此Q12与Q25在此关闭为BOUNDARY，整体证据MEDIUM：明确区分“完整方法目标不等价”“相同目标组件在预定配置仍有残差”“一般机制已有部分覆盖”“唯一MDMT特异机制尚未识别”。Q13的最终机制缺口和Q14的方法化决策继续等待反事实总合成。没有以现有文献未发现某项等价实验为理由声称不存在此类工作，也没有预设创新必须成立。
