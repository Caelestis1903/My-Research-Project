# WP3 / Stage 2A：novelty matrix 与理论边界

2026-10-16，主代理。15篇原始论文/作者来源，完整14项协议字段见[机器矩阵](novelty_matrix.json)、[CSV](novelty_matrix.csv)、[简表](novelty_matrix.md)。检索覆盖规划必需方法及importance sampling/selection同构机制；不是全球穷尽查新。NOT_VERIFIED只表示当前原文或附件覆盖不足，不能写成论文未做。

## Q1：objective weighting 与 sampling 是否已有明确区分？

**有，STRONG（概念先例）**。重要性采样保持经验风险、改变抽样并以逆概率校正，是已有理论；不能把alpha/q本身当作新方法。[Zhao & Zhang](https://proceedings.mlr.press/v37/zhaoa15.html)、[Katharopoulos & Fleuret](https://proceedings.mlr.press/v80/katharopoulos18a.html)。

多域DoReMi也明确区分proxy的域采样与动态域目标权重；其最终改变训练mixture，不能归为固定alpha的estimator-only实验。[DoReMi §2](https://arxiv.org/html/2305.10429v2)。所审MDMT文献没有提供本项目拟议四臂的完整因果对照；这只是本次来源范围内的观察。

## Q2：same objective / different estimator 对 Adam 的影响是否已有研究？

**有直接先例，STRONG（主题重合）**。Lahire明确研究重要性采样与SGD/RMSProp/Adam的相互作用；其展示的adaptive推导使用scalar norm二阶量，不能逐式当成coordinate-wise PyTorch Adam证明。本文自己的数值oracle按真实逐坐标Adam复核。[原文§3.2及附录E](https://arxiv.org/html/2303.16529v1)。

因此“改变q，均值梯度不变，Adam状态/轨迹改变”不能单独宣称创新。可检验的剩余问题是：在实际MDMT、正式batch、训练阶段和冻结选点下，是否存在稳定的实用性能影响及独立干预证据。当前不能给出方法创新PASS。

## Q3：MDMT的sampling→variance→adaptive optimizer是否已被明确研究？

已审来源中，未核实有完整same-alpha/q/state/selection因子分离；**全球缺口判断WEAK**。但SAF已把样本权重、稳定表示/融合与AdaTask式adaptive rates组合，不能将类似组合当新贡献。其作者链接本次返回404，完整训练选点协议仍有未核实项。[SAF §3–4](https://ojs.aaai.org/index.php/AAAI/article/view/38491/42453)。

AdaTask已经直接研究task-wise累积梯度/自适应更新；task换名为domain，或再加动量，不构成区别。[AdaTask](https://arxiv.org/html/2211.15055v2)。PEPNet的参数/embedding个性化属于结构融合先例，不等于固定目标估计器对照。[PEPNet](https://arxiv.org/html/2302.01115v3)。

## Q4：checkpoint selection 是否已有训练策略组成部分的研究？

**有，STRONG（原则先例）**。有限验证准则的反复优化可能产生选择偏差；DomainBed明确把model selection纳入算法评价。不能把“选点重要”当全新原理，也不能由这些论文推断本项目反转的具体原因。[Cawley & Talbot](https://jmlr.org/papers/v11/cawley10a.html)、[DomainBed](https://openreview.net/pdf?id=lQdXeXDoWtI)。

## 四层taxonomy

| 层 | 固定/变化量 | 可以回答 | 不足以回答 |
|---|---|---|---|
| Objective | alpha_d，任务聚合及loss绝对scale | 想优化哪个总体风险 | 采用何种估计器最有效 |
| Estimator | q_d、域内采样、alpha/q、固定分母 | 偏差、协方差、曝光、ESS | selected AUC一定改善 |
| State | m、v、step、bias correction、decay、C/P/A归属 | 相同theta和输入流下的局部更新变化 | 沿异轨迹拼状态后的因果分量 |
| Selection | validation准则、预算、tie、common/selected/final | 哪个轨迹点被报告 | 已使用test上的独立泛化确认 |

PCGrad改变梯度方向，CAGrad改变聚合方向，FAMO改变动态任务权重；都不是固定alpha下纯q干预。[PCGrad](https://arxiv.org/html/2001.06782v3)、[CAGrad](https://arxiv.org/html/2110.14048v2)、[FAMO](https://arxiv.org/html/2306.03792v3)。M3oE的fusion和bi-level优化也是已有结构机制；本轮未新增或恢复learned weights。[M3oE](https://arxiv.org/html/2404.18465v1)。

## 固定theta下的可证伪预言（本报告推导）

令每域row梯度均值mu_d，单样本二阶矩H_d=E[g_i g_i^T|d]，任务先均值。域内IID均匀抽样，q_d>0，w_d=alpha_d/q_d：

mu = Σ_d alpha_d mu_d；Cov(g_batch) = [Σ_d alpha_d² H_d/q_d − mu mu^T]/B。

因此相同alpha不保证不同q有相同covariance。对空域记0的batch-cell Equal，期望loss是Σ_d[1−(1−q_d)^B] L_d/3；不等于任意小batch下的总体Equal。fixed-p和cell归一化差异不能当算法收益。

固定theta、IID时间流、fresh state、coupled decay将梯度改为h=g+lambda*theta：E[mhat_t]=E[h]，E[vhat_t]=E[h²]。不同q下E[mhat]相同、E[vhat]可不同；m样本路径分离不是其均值改变。SGD累计delta期望相同，但**SGD样本路径也会分离**。Adam的非线性除法使E[delta]通常不能由E[h]单独决定。这是检验假设，不预设某个q有益。

AdamW的decay不进入m/v；必须区分coupled L2和decoupled shrinkage。[Adam](https://arxiv.org/pdf/1412.6980)、[AdamW](https://arxiv.org/html/1711.05101v3)。本轮64步replay每次固定theta，累计delta是局部固定场积分，不是沿新theta求梯度的正常训练，也没有证明S5历史因果。

## 查新结论与门

Stage2A的指定方法/问题已完成有边界的来源审查；没有证明“尚无同构方法”。通用alpha/q、variance-aware采样、task-wise自适应状态、dynamic weighting、adaptive fusion、selection-aware评价均已有先例。Track A可以作为待证伪经验问题保留，**E1方法开发保持NOT_ALLOWED**。本轮没有论文方法的公平训练对比，不能声称胜过AdaTask/PCGrad/CAGrad/FAMO/SAF。
