# 08｜相关研究与可迁移边界

检索与核验日期：2026-11-06。以下只纳入与本轮机制判断直接相关的九篇一手文献，不据此宣称已穷尽 MDMTRec 文献。公式定义和本轮干预见 03；本文件不引入新的待训练算法。

## 1. 相关研究给本轮提供什么

参数历史平均已有充分的方法基础。当前研究的具体价值是：在已观察到稳定收益的 MDMT 模型、固定数据协议和 Adam 轨迹上，明确收益发生在哪里、哪些历史改变实际有利、不同 cell 是否存在可重复偏好，以及训练数据上可计算的量能否预测这种偏好。无需把“参数平均”本身包装成首次提出，也不因已有平均方法而否定这组具体问题。

本轮优先借用三个研究工具：偏差与方差的分解、相同在线轨迹上的历史读出、预测函数与参数空间的分离。文献中的算法性能和理论条件不直接替代本仓库的机制证据。

## 2. 核心文献

| 编号 | 准确题名、作者与发表身份 | 本轮可使用的内容 | 本轮不能直接沿用的结论 |
|---|---|---|---|
| R1 | B. T. Polyak, A. B. Juditsky. **Acceleration of Stochastic Approximation by Averaging**. SIAM Journal on Control and Optimization, 30(4):838–855, 1992. [出版方原文](https://epubs.siam.org/doi/10.1137/0330046) | 为随机逼近中的轨迹平均提供经典收敛和统计效率基础，支持区分最后迭代与历史平均估计。 | 不能将其假设下的渐近结论直接当作非凸 M3oE、Adam、固定窗口 EMA 的 AUC 保证。 |
| R2 | Pavel Izmailov et al. **Averaging Weights Leads to Wider Optima and Better Generalization**. UAI 2018. [会议原文](https://www.auai.org/uai2018/proceedings/papers/313.pdf)；[PMLR 页面](https://proceedings.mlr.press/r16/izmailov18a.html) | SWA 展示 SGD 轨迹上的权重平均可改善泛化，并研究其几何及近似集成性质，提示本轮分开考察参数平均与预测变化。 | SWA 的均匀平均、周期/恒定学习率与本轮固定 Adam 轨迹不同；不能因“更宽的解”一语认定本轮收益已由平坦性解释。PMLR 2026 年重刊不改变 UAI 2018 的发表年份。 |
| R3 | Antti Tarvainen, Harri Valpola. **Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results**. NeurIPS 2017. [正式论文页](https://proceedings.neurips.cc/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html) | 明确的参数 EMA 教师先例，说明历史模型可作为监督目标。 | 教师一致性损失会反馈到训练；本轮 F 是非反馈读出。两种因果结构不同，不能用教师效果证明纯读出效果。 |
| R4 | Daniel Morales-Brotons, Thijs Vogels, Hadrien Hendrikx. **Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits**. TMLR, April 2024. [论文 PDF](https://arxiv.org/pdf/2411.18704)；[期刊评审页](https://openreview.net/forum?id=2M9CUnYnBA) | 系统研究独立于训练反馈的参数 EMA、训练阶段、衰减率及泛化/校准等性质，是本轮多影子共用轨迹和同阶段比较的直接方法参考。 | 其主要实验并非本仓库 MDMT/Adam 设置；不能假设相同超参数、阶段和各项指标关系自动迁移。原文已讨论 BN；本仓库没有 BN，不因此新增 BN 实验。 |
| R5 | Xuheng Li, Quanquan Gu. **Understanding SGD with Exponential Moving Average: A Case Study in Linear Regression**. arXiv:2502.14123, 2025. [作者预印本](https://arxiv.org/abs/2502.14123)；[完整理论](https://arxiv.org/pdf/2502.14123) | 在线线性回归中分解 EMA 的偏差、方差及协方差特征子空间风险，为“不同方向可能有不同历史收益结构”提供具体理论参照。 | 需满足文中二/四阶矩、噪声、步长等条件；不是 Adam、门控网络或 AUC 的保证。本次已打开的一手版本未确认正式接收身份，按 2025 预印本引用，不将匿名 ICLR 投稿页当作接收证明。 |
| R6 | Xianliang Li et al. **On the Performance Analysis of Momentum Method: A Frequency Domain Perspective**. ICLR 2025. [正式论文页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/76c64372d25513fed60a9e4176a62481-Abstract-Conference.html)；[全文](https://proceedings.iclr.cc/paper_files/paper/2025/file/76c64372d25513fed60a9e4176a62481-Paper-Conference.pdf) | 把梯度动量解释为随阶段变化的滤波器，启发按训练阶段考察时间尺度，而不是只比较最终一点。 | 它滤波的是梯度并改变在线更新；本轮滤波参数并只改变读出。其低频放大、高频抑制的实验结论不能直接认定 MDMT 参数变化的有害/有益方向。 |
| R7 | Jonathan Patsenker, Henry Li, Yuval Kluger. **Exponential weight averaging as damped harmonic motion**. ICML Workshop on New Frontiers in Learning, Control, and Dynamical Systems, 2023. [作者预印本及 workshop 身份](https://arxiv.org/abs/2310.13854) | 用动力学解释关联 EMA 与动量，并提出 BELAY；可帮助清楚区分读出平均与向在线模型反馈的变体。 | workshop 论文不标成 ICML 主会。物理类比不直接给出本轮排序收益；BELAY 改变在线行为，本轮不默认新增其训练对照。 |
| R8 | Jun Yuan, Guohao Cai, Zhenhua Dong. **A Parameter Update Balancing Algorithm for Multi-task Ranking Models in Recommendation Systems**. ICDM 2024. [作者原文](https://arxiv.org/abs/2410.05806)；[会议 program](https://www.icdm2024.org/program/) | PUB 关注多任务实际参数更新与原始梯度不一致，支持本轮把实际 Adam 位移、训练梯度、功能方向导数分别记录。 | PUB 是参数更新平衡，不是跨训练时刻参数 EMA。不能将它写成“2023 年已经研究固定参数 EMA”的证据，也不能把其梯度/更新相似度直接用作本轮收益指标。 |
| R9 | Jiaxi Tang et al. **Improving Training Stability for Multitask Ranking Models in Recommender Systems**. KDD 2023. [研究机构论文页](https://research.google/pubs/improving-training-stability-for-multitask-ranking-models-in-recommender-systems/)；[作者算法讲义](https://jiaxit.github.io/resources/kdd23_slides.pdf) | Clippy 将训练稳定性问题落实到实际优化器参数更新，说明推荐系统中的更新机制已有明确研究。 | 防止损失发散与提高非反馈 EMA 的 AUC 是不同目标；不据此默认本轮存在发散，也不增加无必要的裁剪/优化器重跑。 |

以上每一项只用于相应有限结论。R4 的 PDF 首行明确为 TMLR 04/2024；arXiv 的 2024 年 11 月上传时间不是期刊首次发表时间。R8 的作者页面明确标注 Accepted by ICDM'24。

## 3. 梯度动量与参数 EMA 的联系

两者都可以采用指数加权历史，但作用对象和反馈路径不同。

| 对象 | 递推 | 直接改变什么 |
|---|---|---|
| 梯度动量 | \(m_k=u m_{k-1}+v g_k\) | 用 m 生成下一次在线参数更新 |
| 非反馈参数 EMA | \(F_k=\rho F_{k-1}+(1-\rho)\theta_k\) | 从已生成的在线参数历史形成评价/推理模型 |

对固定系数，二者形式相似的传递函数分别为

\[
H_m(z)=\frac{v}{1-uz^{-1}},\qquad
H_F(z)=\frac{1-\rho}{1-\rho z^{-1}}.
\]

因此 \(H_m(1)=v/(1-u)\) 可以大于 1，而 \(H_F(1)=1\)。这个差异来自公式，不能用“都涉及 EMA”消除。参数 F 的低通性质已经明确；“哪些被减弱的参数变化有害”仍需由本轮同轨迹、六格和组脉冲收益回答。

Adam 本身已有梯度历史，并不意味着第二个参数历史估计必然无效；两者估计不同对象。也不意味着加两层历史就一定更好。本轮固定 Adam，直接测量参数读出的边际作用，避免重复已完成且不能改变当前决策的优化器消融。

## 4. 参数平均、预测集成与混合参数

R2 提供参数平均与集成相关的经验和近似视角。对一般非线性模型，

\[
f\!\left(x;\sum_jw_j\theta_j\right)
\ne \sum_jw_jf(x;\theta_j).
\]

二者在线性或局部一阶条件下可以接近，非线性表示、门控及参数组配合会产生差异。03 的功能 RMS、排序翻转、M2 一阶量及非加性残差正是用于观察这种差异。

本轮不默认追加预测集成训练。已有 epoch 预测的离线平均使用的采样点和历史核，与每步参数 EMA 不同；即使后续用作说明，也必须标明其估计对象，不能称为严格等核对照。

## 5. 对“PUB 2023”说法的处理

本次能够核实的 PUB 是 R8：ICDM 2024、参数更新平衡。其核心机制不是将历史模型参数指数平均。因此，本轮文献与动机文件不沿用“2023 PUB 已完成参数 EMA”这一归因。

参数历史平均早已有 R1–R4 等明确先例，所以修正文献身份不需要改变研究动机，也不需要主张本轮方法绝对原创。如果后续找到此前讨论所指的另一篇具体论文，再补充准确题名与机制定位即可；不因此暂停本轮机制研究。

## 6. 由文献直接约束本轮研究决策

1. 借用经典平均理论建立偏差/波动竞争解释；用实际收益决定哪种解释在本协议有用。
2. 借用 EMA 实证研究的同轨迹、多时间尺度方法；保持 Adam、Gaussian 和自然训练人口不变。
3. 将参数低通、函数变化、排序收益和预测价值分层记录；任何一层的有效结论都不因其他层尚未完全解释而归零。
4. 保留参数组及六格研究的特异性，但不预设复杂自适应 EMA 应当超过固定 F。局部偏好如果只存在于选点噪声或不能预测未来，同样是清晰的研究结论。
5. 不将 PUB、Clippy、Mean Teacher、BELAY 等不同反馈机制顺手加入主实验。只有现存竞争解释会实质改变研究决策时，才启用 04/06 已定义的有限控制。
