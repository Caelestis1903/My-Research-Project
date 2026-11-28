# 01 — 研究背景、证据链与相关文献

## 1. 当前研究定位

本项目的稳定经验事实是：固定参数 EMA 能改善两数据集的 selected AUC；Gaussian D0 与 EMA 组合在 MovieLens 有相对 M3oE 的完整方法收益；A21 在 KuaiRand 的历史 Test 口径下相对同 PAIR Adam 有五 seed 正增量。三个现象分别涉及参数历史读出、增强视图监督和辅助梯度历史。它们共同指向“训练过程中产生的历史信息可能包含当前模型尚未充分利用的有效排序信息”，但三者的信息对象和作用路径不同。

下一阶段不再把“历史是否有用”当作宽泛问题，而是分成三个可干预的对象：
- 数据视图中的 pairwise 排序监督；
- 参数坐标中的历史 PAIR 梯度；
- 函数输出中的历史正负排序关系。

## 2. 项目证据

### 2.1 固定 EMA

EXP082 的 80 条冻结轨迹中 77 条 EMA Test AUC 高于 online。MovieLens / KuaiRand 平均 EMA−online 分别为 +0.004533 / +0.004816。该证据支持固定 EMA 作为强内部读出。EMA 是模型参数读出，不参与 online 轨迹反馈。不能把 EMA 的收益归到 A21 或新的 pairwise 组件。

### 2.2 Gaussian D0 + EMA

Test Macro AUC：
| Dataset | G-D0 + EMA | M3oE | Δ |
|---|---:|---:|---:|
| MovieLens | 0.779357 | 0.775190 | +0.004167 |
| KuaiRand | 0.679180 | 0.671906 | +0.007274 |

KuaiRand 的 HyperGate 为 0.682113，因此 Gaussian 组合未成为该数据集最强模型。MovieLens LogLoss 相对 M3oE 增加 +0.000931。MovieLens G-D0 相对同协议 clean+EMA 的组件增量约 +0.000318；该数值表明整体收益主要不能简单归因于 Gaussian 单独增量。A 方向必须直接比较 Gaussian Pairwise 对 G-D0+EMA，而不是把 G-D0 已有收益重新计入。

### 2.3 A21 与 C1/C2

A21 相对同 PAIR Adam + EMA 的 Test ΔAUC：ML −0.000083，KR +0.004062；KR 五 seed 全正、配对 95% 区间约 [0.000469,0.007656]。但 KR Validation ΔAUC −0.001989，故这是一条历史 Test 信号，不是正常 Validation 选点下的算法晋级。C1、C2 的结果已经是 NO_ACTIONABLE_INCREMENT。后续 B 应研究同 batch、同 pair、共享参数坐标的梯度校正，而非重复 C1/C2 的来源二阶矩或历史预条件方向累计。

## 3. 2025–2026 相关工作

### 3.1 历史梯度与动量

**MARS: Unleashing the Power of Variance Reduction for Training Large Models** — Huizhuo Yuan et al., ICML 2025, PMLR 267, 73553–73587.  
MARS 将预条件梯度更新与 scaled stochastic recursive momentum 结合，使用同一随机样本上的当前/旧参数梯度差构造校正项。它说明历史梯度复用与当前参数点校正可以统一在递归动量框架中。迁移启发：在本项目 PAIR 辅助来源上，用相同 pair minibatch 比较当前共享参数和上一步共享参数的梯度。区别：本项目的候选只针对 PAIR 来源及 S_DENSE 参数子空间，不是完整 MARS-AdamW，也不继承其语言模型实验结论。正式论文：https://proceedings.mlr.press/v267/yuan25f.html

**Understanding MARS: When Scaling Momentum Correction Provably Helps** — Egor Shulgin et al., ICML 2026, PMLR 306, 112936–112965.  
该工作分析 correction scale γ 与 gradient-difference similarity 的关系，说明 γ<1 在某些条件下可能改善复杂度界。迁移启发：校正系数是控制校正强度的机制参数，应有有限、预先冻结的候选值，而非无限扫描。它的理论条件和大模型训练结果不等于本项目的 AUC 收益。正式论文：https://proceedings.mlr.press/v306/shulgin26b.html

**Reducing the Variance in Online Optimization by Transporting Past Gradients** — Arnold et al., NeurIPS 2019.  
提出 implicit gradient transport，将旧参数处的梯度隐式运输到当前参数点，减少在线优化中的陈旧偏差。迁移启发：需要明确“旧梯度与当前参数状态的差异”并使用当前模型函数信息校正。其方法不等同于简单历史梯度 EMA，也不应将本项目有限差分写成原论文 IGT。论文：https://papers.neurips.cc/paper/8779-reducing-the-variance-in-online-optimization-by-transporting-past-gradients.pdf

**The AdEMAMix Optimizer: Better, Faster, Older** — Matteo Pagliardini, Pierre Ablin, David Grangier, ICLR 2025.  
指出单一 EMA 不能同时保留近端梯度和远期梯度，提出双 EMA 混合。作者实验显示历史梯度在部分语言/视觉任务中可持续提供信息。对本项目的价值：若 B 的 freshness correction 不成立，后续可比较长短历史梯度的并行保留；但本项目已经测试过多种时间尺度/自适应 EMA，因此不能直接再做“增加第二个 EMA”的简单重复。只有明确作用于 PAIR 来源、与 A21 和普通 Adam 的直接对照不同，才考虑后续独立研究。正式论文：https://proceedings.iclr.cc/paper_files/paper/2025/hash/a2cf225ba392627529efef14dc857e22-Abstract-Conference.html

**How Memory in Optimization Algorithms Implicitly Modifies the Loss** — Matias D. Cattaneo, Boris Shigida, NeurIPS 2025.  
提出将有历史的优化算法近似改写为当前点的 memoryless 更新加上由历史引起的 correction term，并解释其隐式正则化作用。迁移启发：历史信息不只是“更平滑”，也可能改变优化器隐含偏好的损失几何。可用于 B 的机制解释：校正项是否在训练中形成了有利于 pairwise ranking 的隐式偏置。不能据此直接声称 A21 的机制已经被解释。正式论文：https://proceedings.neurips.cc/paper_files/paper/2025/file/e4cc8ab4a64e99f962f36d07a7723d94-Paper-Conference.pdf

**Backpropagated Output Momentum: Relocating Optimizer History from Parameters to Task Space** — 2026 arXiv preprint.  
该预印本将历史信息保存为输出残差的 EMA，再通过当前网络反向传播到参数空间，减少参数坐标中的历史固定性。它为 B/C 提供另一种历史信息表征：保留历史输出误差/排序残差，而非固定参数梯度。状态为预印本，非已确认顶会正式成果；不进入首轮矩阵。链接：https://arxiv.org/abs/2609.36738

### 3.2 排序增强与推荐

**Triplet Contrastive Learning with Learnable Sequence Augmentation for Sequential Recommendation (TCLARec)** — Wei Wang et al., SIGIR 2025, pp.1519–1529.  
工作指出启发式随机增强可能产生与用户真实意图不一致的 positive view，并结合可学习序列增强和 ranking-based triplet contrastive loss。对 A 的启发：增强的质量必须与目标排序关系一致，不能将“加入噪声”自动视为有效增强；增强 view 应通过 pairwise ranking 目标接受直接评价。差异：TCLARec 面向序列推荐和自监督 triplet，不是 M3oE 的多域多任务 CTR，也没有验证 Gaussian embedding noise。正式会议页面：https://sigir2025.dei.unipd.it/detailed-program/paper?paper=17d63b1625c816c22647a73e1482372b

**Improving Sequential Recommenders through Counterfactual Augmentation of System Exposure** — SIGIR 2025.  
从系统曝光反事实角度构造增强，说明推荐数据增强可以围绕曝光机制和目标行为而非简单随机变换设计。对本项目的启发是：若 Gaussian Pairwise 不提供独立增量，下一类增强应从监督语义/样本关系建模切入，而不是继续扩大噪声种类。正式论文：https://doi.org/10.1145/3726302.3730005

**Harnessing Pairwise Ranking Prompting Through Sample-Efficient Ranking Distillation** — Junru Wu et al., arXiv 2025.  
研究用 pairwise teacher labels 蒸馏 pointwise ranker，并指出只需一小部分 pair 也可能达到完整 pair 集合的性能。它不是推荐系统顶会成果，作为 pair sampling 和 pair 预算设计的补充先例。对 A/C 的启发：pair exposure 预算需要显式匹配，避免 pair 数量增加带来伪增益。链接：https://arxiv.org/abs/2507.04820

### 3.3 多任务优化

**Towards Consistent Multi-Task Learning: Unlocking the Potential of Task-Specific Parameters** — Xiaohan Qin et al., CVPR 2025.  
提出 ConsMTL，指出 task-specific parameters 也会影响传回 shared parameters 的梯度冲突，并用 bi-level 机制协调两类参数。对本项目的意义：共享参数梯度的行为取决于完整模型当前状态，B 的共享参数校正应在相同私有参数状态下控制，明确其作用子空间。差异：本项目不引入选择性共享或任务专属结构，也不直接迁移 ConsMTL。正式论文：https://openaccess.thecvf.com/content/CVPR2025/html/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Task-Specific_Parameters_CVPR_2025_paper.html

## 4. 研究空间判断

- A 的价值在于把已有效的 Gaussian view 与最终排序指标对齐，直接测试 pairwise objective 和 Gaussian view 的交互。Pairwise loss 本身可能已经足够，故必须有 clean pairwise 与 Gaussian BCE 控制。
- B 的价值在于从 A21 的历史辅助梯度出发，显式估计同一 pair batch 在两个共享参数状态下的梯度变化。该设计直接针对参数坐标的新鲜度，不重复 EMA 时间尺度、C1 二阶来源耦合和 C2 预条件累积。
- C 的价值在于历史关系是函数空间信息，不是历史梯度。若它只是重放当前 pairwise 监督，则没有独立机制贡献；必须证明记忆中的关系目标包含当前 pair sampler 没有提供的时间信息。
- 更高风险的替代方案是输出空间历史残差动量（BOM 思路）。若共享参数快照校正成本过高，或历史关系记忆无法低成本区分当前 pairwise loss，可考虑以历史 logit residual 为存储对象，再经当前网络反向传播。先不进入本轮训练矩阵。

## 5. 参考文献登记

1. Yuan et al. 2025. MARS. ICML. https://proceedings.mlr.press/v267/yuan25f.html
2. Shulgin et al. 2026. Understanding MARS. ICML. https://proceedings.mlr.press/v306/shulgin26b.html
3. Pagliardini et al. 2025. AdEMAMix. ICLR. https://proceedings.iclr.cc/paper_files/paper/2025/hash/a2cf225ba392627529efef14dc857e22-Abstract-Conference.html
4. Arnold et al. 2019. Implicit Gradient Transport. NeurIPS. https://papers.neurips.cc/paper/8779-reducing-the-variance-in-online-optimization-by-transporting-past-gradients.pdf
5. Cattaneo & Shigida. 2025. How Memory in Optimization Algorithms Implicitly Modifies the Loss. NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2025/file/e4cc8ab4a64e99f962f36d07a7723d94-Paper-Conference.pdf
6. Wang et al. 2025. TCLARec. SIGIR. https://sigir2025.dei.unipd.it/detailed-program/paper?paper=17d63b1625c816c22647a73e1482372b
7. Qin et al. 2025. ConsMTL. CVPR. https://openaccess.thecvf.com/content/CVPR2025/html/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Task-Specific_Parameters_CVPR_2025_paper.html
8. Wu et al. 2025. Pairwise Ranking Prompting Distillation. arXiv. https://arxiv.org/abs/2507.04820
9. Li et al. 2026. Backpropagated Output Momentum. arXiv preprint. https://arxiv.org/abs/2609.36738
10. SIGIR 2025. Improving Sequential Recommenders through Counterfactual Augmentation of System Exposure. https://doi.org/10.1145/3726302.3730005

会议年份以正式 proceedings 为准；预印本明确标记，不将其包装为正式顶会论文。
