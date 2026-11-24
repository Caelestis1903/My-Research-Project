# 02 — 文献综合与历史信息的数学关系

检索与核读日期：2026-11-24。内部证据固定于 [main@358d3a4](https://github.com/Caelestis1903/My-Research-Project/tree/358d3a465747eda6af1d55acc937759fa651a10b)。本章给出理论依据与可区分的机制，不改变[执行合约](00_EXECUTION_RUNBOOK.md)、[完整矩阵](06_EXPERIMENT_MATRIX.md)和[阶段决策](08_DECISION_RULES.md)。

## 1. 当前结论：研究历史信息进入更新的方式

**固定参数 EMA 与 A21 已经提供充分的研究起点。本轮应保留这些积极事实，集中检验两个尚未被旧实验覆盖的运算：共同二阶矩的来源耦合，以及预条件与辅助历史累积的先后次序。** 前者形成 C1「来源二阶耦合调节」，后者形成 C2「预条件辅助历史」。二者保持自然 BCE、条件 PAIR、原作用域及训练预算，直接接受完整训练评价。

EMA 整合已经执行的参数更新；A21 保存自然 BCE 与 PAIR 的原始梯度历史，再用当前共同预条件器生成更新。两者共享“利用过去信息”的原则，信息对象和使用位置具有明确差别。已有 EMA 参数组、窗口和函数干预支持其表示与排序价值；A21 的 KuaiRand 历史 Test 增量支持来源历史的实际潜力。两类证据都不依赖先建立唯一统一机制。[内部证据与端点重分析](01_EVIDENCE_AND_RESEARCH_DECISION.md)

| 信息对象 | 典型组织 | 进入模型的位置 | 本轮需要回答的增量问题 |
| --- | --- | --- | --- |
| 历史参数状态 | 固定参数 EMA | 不反馈训练的模型读出 | 新在线机制的收益能否保留到 EMA 读出 |
| 原始来源梯度 | A21 的 BCE、PAIR 不同时间核 | 当前共同二阶预条件之前 | 已保存方向是否受到来源二阶耦合影响 |
| 当时预条件后的来源方向 | C2 的 PAIR 更新历史 | 每次预条件后累积，再用于在线更新 | 保存当时几何是否优于用当前几何重释旧梯度 |
| 历史函数或 pair margin | 预测组合、关系记忆 | 读出组合或训练约束 | 纠正与破坏的净排序价值；本轮只分析，C3 留作未来机会 |

## 2. 文献如何改变本项目的研究选择

### 2.1 参数平均：提供强读出参照，保留轨迹内互补

Polyak–Juditsky 的经典工作给出随机逼近中轨迹平均的收敛与效率依据。SWA 把同一 SGD 轨迹的多点平均用于深度模型泛化；EMA 系统研究则区分在线轨迹与平均读出，并讨论噪声、学习率和预测一致性。2025 年 AlgoPerf 研究表明平均与学习率退火可以互补。这些成果共同支持把平均作为独立可评价的训练组成，并分别衡量效率、最终预测与选点效果。[R01](https://epubs.siam.org/doi/10.1137/0330046)；[R03](https://arxiv.org/abs/1803.05407)；[R04](https://arxiv.org/html/2411.18704v1)；[R07](https://proceedings.mlr.press/v267/ajroldi25a.html)

Model Soups 将参数平均与 logit 组合的接近程度联系到局部几何和预测置信度；SWAD 将稠密权重平均用于未见域泛化。这两条线索分别提供函数空间比较和多源领域应用的视角。它们的训练目标及评价人口与本项目不同，MDMT 的六格排序收益以项目内的匹配实验裁定。[R05](https://proceedings.mlr.press/v162/wortsman22a.html)；[R06](https://proceedings.neurips.cc/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html)

**研究后果：** 固定 EMA 继续作为每条轨迹的被动读出。已有参数组和窗口结论直接接受；新增工作用于计算新候选与 EMA 的交互、纠正和破坏关系，不重跑一轮“平均是否有效”。两态输出和普通 KD 已有完整结果，不能借上述文献重新包装为未探索的主候选。

### 2.2 梯度历史：持续信息、估计噪声与陈旧偏差分别处理

Adam 已维护一、二阶矩，额外历史必须说明超出原状态的具体运算。AdEMAMix 同时保留快慢一阶核，展示单个时间核在近期响应和较早信息之间的限制；MoCo 通过历史梯度估计处理随机多目标组合中的偏差；MARS 将递归梯度差分校正与预条件结合，其后续理论分析刻画校正强度与梯度差分结构的关系。这些工作分别改变历史核、非线性组合中的估计质量及参数漂移校正，不能统一概括成“更长动量”。[R02](https://arxiv.org/pdf/1412.6980)；[R08](https://arxiv.org/html/2409.03137v1)；[R10](https://arxiv.org/abs/2210.12624)；[R11](https://proceedings.mlr.press/v267/yuan25f.html)；[R12](https://proceedings.mlr.press/v306/shulgin26b.html)

Schedule-Free 将迭代平均嵌入优化，说明平均可以直接参与在线算法。该先例扩展了设计空间，但本项目已测的 EMA 反馈和动态动作失败仍约束具体实现；引用新的优化器名称不构成重跑旧反馈的理由。[R18](https://proceedings.neurips.cc/paper_files/paper/2024/hash/136b9a13861308c8948cd308ccd02658-Abstract-Conference.html)

**研究后果：** A22 的普通长历史控制已经完成。本轮只对进入后续阶段的候选使用合约内短辅助核控制 S，回答新增运算是否依赖长 PAIR 历史。梯度运输与递归差分校正保留为后备机会，不增加本轮主优化器矩阵。

### 2.3 来源状态：共享程度本身是算法变量

AdaTask 为任务分别维护累积梯度和自适应尺度；Adam 版本的一阶与二阶状态均按任务分开。DualOptim 将不同目标的动量解耦，DualOptim+ 进一步用共享 base 与目标 residual 连接共享和分离状态。后者的多任务实验中完全分离与部分共享有不同表现，表明来源结构应依据实际目标关系选择。[R09](https://arxiv.org/html/2211.15055v2)；[R13](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7b76eea0c3683e440c3d362620f578cd-Abstract-Conference.html)；[R14](https://arxiv.org/html/2605.21539v1)

Bloop 把主目标梯度的 EMA 用作辅助梯度的投影参考，随后把构造方向交给优化器。它展示了历史可以改善主辅目标关系的估计；其主目标优先的双层目标、投影及 EMA 更新时序与本项目的固定 BCE＋PAIR 优化不同。ForkMerge 通过分支训练和参数合并处理辅助任务泛化，也提醒研究者区分瞬时梯度关系与完整训练收益。[R20](https://arxiv.org/html/2402.02998v2)；[R15](https://proceedings.neurips.cc/paper_files/paper/2023/hash/60f9118a849e8e9a0c67e2a36ad80ebf-Abstract-Conference.html)

**研究后果：** 本轮“来源”指自然 BCE 与条件排序 PAIR，两者都跨域任务作用于共享表示；它不等于把六格机械拆成六套历史。C1 保留一个共同分母，仅调节平方前混合产生的交叉项。最接近的 AdaTask 来源适配作为条件控制 P，回答共同几何与分源分母哪一个足够。Bloop 的投影会增加另一项干预，本轮不据此恢复旧 C/H 投影家族。

### 2.4 C2 的直接先例：LaProp 已提出预条件后累积动量

LaProp 的 Algorithm 1 先更新二阶矩，把当前梯度除以当时的偏差修正分母，再把所得方向写入一阶历史。原文指出，普通 Adam 用当前预条件器重新缩放过去的动量；LaProp 保留参数更新空间中的累积方向。这是 C2 必须明确引用的直接先例，登记为 2020 年首发、2021 年 v3 的 arXiv 预印本。[R19 原文 §2.2–2.3、Algorithm 1](https://arxiv.org/html/2002.04839v3)

本轮据此提出的具体问题是：**在同一个 mixed 二阶几何下，保留 BCE 的快速原始梯度历史，只让 PAIR 来源保存当时预条件后的较长历史，能否改善共享训练的净排序结果？** C2 的来源选择、两种时间核、PAIR 活动时钟和固定作用域共同定义待测算法。研究贡献由相对 H0、同 PAIR Adam 和 LaProp 适配的增量决定；“预条件与动量换序”本身归于已有工作。

合约内 C2 的 P 控制是在 S_DENSE 对完整 mixed 梯度采用 LaProp 式先预条件、后 .9 动量，其余协议保持一致。它是适配当前目标和骨干的内部优化器参照，不等同于复现原论文任务或新增一个已发表推荐模型基线。

### 2.5 函数记忆：把“历史有用”落到具体排序关系

Example Forgetting 在无显式任务切换的普通监督训练中测量“曾正确、后错误”的事件；Dark Experience Replay 通过训练轨迹中的历史 logits 约束当前函数。它们提供历史关系可记录、可使用的先例。将分类正确性转换为同格正负 pair 的排序正确性，是本项目面向 AUC 的推论。[R16](https://arxiv.org/abs/1812.05159)；[R17](https://proceedings.neurips.cc/paper_files/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)

本轮用纠正、破坏和净排序变化解释 C1/C2 与 EMA；训练标签核验的历史正确 pair margin 保留列为 C3 未来机会。它若启动，需要同 pair 曝光和固定 margin 等实质匹配，具体方案另行冻结，不占用本轮实现和训练矩阵。

## 3. 数学关系：哪些等价，哪些操作有实质差异

以下恒等式由本方案定义直接推导。它们用于保证可识别的操作与正确解释，性能结论由完整训练决定。

### 3.1 参数 EMA 是对实际更新的确定性重新加权

令完整有效更新为 $\theta_k=\theta_{k-1}+\Delta\theta_k$，并令 $\bar\theta_k=\rho\bar\theta_{k-1}+(1-\rho)\theta_k$，$\bar\theta_0=\theta_0$。展开并交换求和可得：

$$
\bar\theta_T=\rho^T\theta_0+(1-\rho)\sum_{j=1}^T\rho^{T-j}\theta_j
=\theta_0+\sum_{k=1}^T(1-\rho^{T-k+1})\Delta\theta_k,
$$
$$
\bar\theta_T-\theta_T=-\sum_{k=1}^T\rho^{T-k+1}\Delta\theta_k.
$$

这里的 $\Delta\theta_k$ 是已执行的完整参数增量，包含学习率、来源组合、Adam 状态、正则和当时参数。EMA 在最终读出中减弱近期增量，同时保留此前累积；它不是在当前参数上重新计算旧梯度。

因此，“EMA 与梯度历史都包含过去更新”有严格数学依据；“EMA 等于 A21 或 C2”则不成立。C2 只记录一个监督来源的当时预条件方向，EMA 汇总完整在线轨迹。两者关系应由 online/EMA 增量和交互测量，而不是根据名称推断。

### 3.2 同核一阶来源历史的线性等价与 A21 的非等价环节

在同一更新时钟、相同 $\beta$、相同零初始化（或联合态初始化为源态的加权和）及固定 $\lambda$ 下：

$$
\operatorname{EMA}_{\beta}(g^B+\lambda g^P)
=\operatorname{EMA}_{\beta}(g^B)+\lambda\operatorname{EMA}_{\beta}(g^P).
$$

这说明“多存一份一阶变量”自身不产生新运算；不同时间核、活动时钟或后续非线性处理才能改变实际更新。早期 S0 的统一梯度 EMA 已形成相应强控制，A20/A22 也已覆盖条件数量和长普通历史。[已有审计](01_EVIDENCE_AND_RESEARCH_DECISION.md)

本轮以原生参数步 $t$ 和 PAIR 活动事件计数 $n_t$ 区分两个时钟。记 $I_t$ 为活动标志，$b_t=g^{BCE}_t+d\theta_{t-1}$，$h_t=(w_s/q_s)r_{PAIR,t}$；非活动步 $h_t=0$。H0 的来源状态为：

$$
m^B_t=.9m^B_{t-1}+.1b_t,\qquad
Q_{n_t}=.99Q_{n_t-1}+.01h_t\quad(I_t=1),
$$
$$
V_t=.999V_{t-1}+.001(b_t+\lambda h_t)^{\odot2},\qquad
D_t=\operatorname{diag}\!\left((\sqrt{\hat V_t}+\epsilon)^{-1}\right).
$$

H0 更新方向为 $D_t\hat m^B_t+\lambda I_tD_t\hat Q_{n_t}$。PAIR 不活动时 Q 不衰减、不递增事件计数、不注入辅助方向；自然一阶与二阶仍按原生参数步处理。不同核和不同活动时钟使 H0 与普通 mixed Adam 存在明确差别。这里的 $t$ 不是用任意 batch 总数替代原生参数计数。

两个来源每步看到的信息也不同：自然 BCE 对当前大量自然行的两个任务各计算一项损失；PAIR 在当前 batch 的活动条件中抽取一个 $(d,t,a,b)$ 标签边界，再有放回抽取 2048 个正负对。故长辅助核可以跨事件积累不同边界的信息，这是 MDMT 共享语境下的具体候选解释。[目标原实现](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/conditional_information_momentum_20261102_v3/objectives.py)；[边界采样实现](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/conditional_information_momentum_20261102_v3/boundary_sampler.py)

给定参数、当前自然 batch 和活动支持，设条件 $s$ 的采样概率为 $q_s>0$、目标质量为 $w_s$、抽样 pair 梯度为 $r_s$。importance 校正满足：

$$
\mathbb E[h]=\sum_s w_s\mathbb E[r_s],\qquad
\mathbb E[h^{\odot2}]=\sum_s\frac{w_s^2}{q_s}\mathbb E[r_s^{\odot2}].
$$

其中内层期望包含条件内 pair 抽样；$r_s$ 固定时可去掉内层期望。这说明一阶目标均值与二阶组成是不同对象：一阶校正保持给定目标，二阶矩仍取决于来源抽样和条件内变异。C1/C2 固定 q、支持与质量，研究这些信息进入历史与预条件的方式；只记录条件覆盖和核内质量，不重复 q 或条件表实验。

### 3.3 C1：共同二阶矩中的交叉项有符号

逐坐标展开：

$$
(b_t+\lambda h_t)^{\odot2}
=b_t^{\odot2}+\lambda^2h_t^{\odot2}+2\lambda b_t\odot h_t.
$$

在同一已给定梯度路径、相同 $\beta_2$、初始化和偏差修正下，混合平方与源平方和之间满足：

$$
V^{mix}_t-V^{sum}_t
=\operatorname{EMA}_{.999}[2\lambda b_t\odot h_t].
$$

C1 使用 $V^\zeta=(1-\zeta)V^{mix}+\zeta V^{sum}$，等价于直接递推输入 $(1-\zeta)(b+\lambda h)^{\odot2}+\zeta(b^{\odot2}+\lambda^2h^{\odot2})$。当 $0\leq\zeta\leq1$ 时二阶输入非负。$\zeta=0$ 应执行原 mixed 张量路径，保证退化点的浮点算术身份。

**在同一路径上，某坐标的累积交叉项为正时，提高 $\zeta$ 使其分母变小；累积交叉项为负时，分母变大。** C1 同时改变自然与辅助分子的共同预条件，不能预称为“消除有害冲突”或仅改变 PAIR 的剂量。交叉项是未中心化二阶量；其期望含均值乘积和协方差，不能直接当作噪声协方差。

$b_t$ 包含源码规定的一次 coupled WD，所以交叉项也含 $2\lambda d\theta\odot h$。可在现有日志中分解 BCE×PAIR 与 WD×PAIR，实际算法保持原定义；不额外加入一组“去 WD”训练。

共同分母 $\frac{\hat m^B+\lambda\hat Q}{\sqrt{\hat V^\zeta}+\epsilon}$ 与 AdaTask 型分源分母 $\frac{\hat m^B}{\sqrt{\hat V^B}+\epsilon}+\lambda\frac{\hat Q}{\sqrt{\hat V^P}+\epsilon}$ 通常不等价。前者保留共同坐标几何，后者让两个来源使用不同尺度；条件控制 P 正好区分这两种处理。

### 3.4 C2：预条件与时间平均不交换

将活动时刻写作 $\tau_1,\ldots,\tau_n$。PAIR 偏差修正权重为 $\kappa_{n,j}=(1-\beta_P)\beta_P^{n-j}/(1-\beta_P^n)$，其中 $\beta_P=.99$。在活动步 t、同一路径和共同分母定义下，H0 的辅助方向为：

$$
u^P_{H0,t}=D_t\sum_{j=1}^{n_t}\kappa_{n_t,j}h_{\tau_j}.
$$

C2 在当步 V 更新和偏差修正之后，记录 $U_n=.99U_{n-1}+.01D_t h_t$。其纯预条件历史方向和差值为：

$$
u^P_{U,t}=\sum_{j=1}^{n_t}\kappa_{n_t,j}D_{\tau_j}h_{\tau_j},\qquad
u^P_{U,t}-u^P_{H0,t}
=\sum_{j=1}^{n_t}\kappa_{n_t,j}(D_{\tau_j}-D_t)h_{\tau_j}.
$$

这就是时间预条件与历史平均的交换差。$D$ 恒定时严格为零；$D$ 随训练改变时，差值同时依赖几何变化、梯度内容和二者的时间对齐，而不是只依赖历史长度。C2 保持 $\zeta=0$，用 $\alpha$ 在两个辅助方向之间插值；整个方向为：

$$
u_t=D_t\hat m^B_t+\lambda I_t\big[(1-\alpha)D_t\hat Q_{n_t}+\alpha\hat U_{n_t}\big].
$$

U 读取时不再乘当前 D。它记录的是“当前 PAIR 梯度在当时几何下的方向”，并非 H0 当时已经执行的完整更新；后者还包括基础历史及辅助历史的组合。固定学习率和 $\lambda$ 使 $D_{\tau_j}h_{\tau_j}$ 可对应一个当时的辅助瞬时更新分量，但不改变上述区分。

交换差是同一路径上的精确关系。完整 C2 与 H0 轨迹分离后，应在各自参数与状态上计算影子方向；不能把跨轨迹状态移植后所得差值解释为单一换序效应。C2 保留旧梯度的几何时间戳，并未重算 $h_{\tau_j}$ 在当前参数上的值，因此也不等同于梯度运输或陈旧梯度校正。

既有 SAD、H_REAL/H_MIX、TV-GRM 均把原始来源梯度历史置于当前预条件之前；H 系列的当前度量投影之后也返回 raw 空间存储。C2 的 PAIR 内容与运算顺序均有明确区别。[历史实现核对](01_EVIDENCE_AND_RESEARCH_DECISION.md)

### 3.5 均值、方差与偏差：降噪只解决其中一项

在固定参数或给定均值过程的分析模型中，写辅助事件梯度为 $h_j=\mu_j+\varepsilon_j$，$\mathbb E\varepsilon_j=0$。则：

$$
\mathbb E\hat Q_n=\sum_j\kappa_{n,j}\mu_j,\quad
\operatorname{Bias}_n=\sum_j\kappa_{n,j}(\mu_j-\mu_n),\quad
\operatorname{Cov}(\hat Q_n)=\sum_{i,j}\kappa_{n,i}\kappa_{n,j}\operatorname{Cov}(\varepsilon_i,\varepsilon_j).
$$

若噪声独立同分布且均值固定，稳态方差缩小为 $(1-\beta_P)/(1+\beta_P)$ 倍。均值漂移、时间相关和来源切换会改变这个结论；初始化偏差修正消除零初始化衰减，不消除相对当前梯度的陈旧偏差。

长核约有 $\beta/(1-\beta)$ 个事件的平均年龄，而事件间距由活动时钟决定。因而 .99 辅助核与 .9 自然核不能仅按 epoch 长度比较。需要关注的是哪些持续方向跨事件保留下来，以及这些方向通过当前或当时 D 后怎样影响排序。

对独立、平稳、等方差的标量事件，偏差修正核的有效样本数为 $N_{\mathrm{eff},n}=1/\sum_j\kappa_{n,j}^2$，稳态趋于 $(1+\beta)/(1-\beta)$：.9 对应约 19，.99 对应约 199。这是核的理想化统计容量，不是独立训练 seed 数，也不把 2048 个 pair 当独立条件。较长 PAIR 历史因而有机会在同 q 下整合更多跨条件信息；真实收益仍取决于相关性、漂移、importance 质量和净排序，不能解释为小域天然更难。

本项目已有反向噪声配对“方差下降而 AUC 未改善”、局部 utility 改善却未改善完整训练等结果，足以关闭“降噪必然提高排序”的推断。该事实保留噪声过滤的局部价值，同时把新算法的判据落到净排序和完整训练。[内部证据](01_EVIDENCE_AND_RESEARCH_DECISION.md)

### 3.6 总损失尺度、coupled WD 与 epsilon 的完整等价

令 $c>0$，L 包含同一固定相对权重的 BCE＋PAIR。coupled Adam 满足：

$$
c\nabla L+d\theta=c\left(\nabla L+\frac dc\theta\right),\qquad
\frac{cm}{\sqrt{c^2v}+\epsilon}=\frac{m}{\sqrt v+\epsilon/c}.
$$

因此同学习率下，$\mathrm{Adam}(cL,d,\epsilon)$ 与 $\mathrm{Adam}(L,d/c,\epsilon/c)$ 在对应状态下等价。冷启动用相同零状态；热启动将一阶态、二阶态分别按 $c$、$c^2$ 映射并保留计数。有限 epsilon、原生缺失梯度语义和全部附加运算都属于等价条件。

该映射也适用于本方案固定 $\zeta,\alpha$：来源二阶输入是二次齐次量，$D h$ 在对应映射下保持不变。比较未缩放与放大c倍的对应热状态时，raw一阶态满足 $m_B^{scaled}=c m_B$、$Q^{scaled}=cQ$，二阶态满足 $V^{scaled}=c^2V$，但预条件历史必须保持 $U^{scaled}=U$，不能把U当raw一阶态再乘c；反向映射分别除以c/c²并保持U。常数损失缩放可改变有效正则和 epsilon，不能直接记为新的历史内容收益。EXP056 的普通 L2 工作替代与严格等价有不同精度；EXP081 SCALE 已完成对应冷热状态论证，直接引用即可。[EXP081 及尺度证据](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/mdmt_research_context/02_AI_EXPERIMENT_DETAILS.md)

C1/C2 的逐坐标或历时变化一般无法被一个全局常数吸收。合约内 M 控制在自身轨迹上匹配候选的组级实际步幅、保留 H0 方向，检验足以影响研究选择的简单尺度解释；不重新扩大 SCALE、学习率或 WD 搜索。

### 3.7 参数平均与函数组合的 Taylor 关系

对权重和为 1 的历史状态，令 $\bar\theta=\sum_jw_j\theta_j$，$\delta_j=\theta_j-\bar\theta$。逐输出在可微局部邻域展开：

$$
\sum_jw_jf(x;\theta_j)
=f(x;\bar\theta)+\frac12\sum_jw_j\delta_j^\top H_f(x;\bar\theta)\delta_j+\sum_jw_jR_j.
$$

一阶项因 $\sum_jw_j\delta_j=0$ 抵消；差异由函数曲率和历史状态分散决定。ReLU 网络的局部展开需位于可微区域，跨激活边界时直接测量函数差异。把 logits 经 sigmoid 后再平均还增加输出非线性差异，参数平均、logit 平均和 probability 平均应分别命名。

因此函数组合覆盖部分 EMA 收益是有信息的结果；其余差异可以来自参数空间与函数空间的非线性关系。已有两态与全历史读出结果足以支持互补分析，当前无需以训练重建强求三者完全等价。[R05](https://proceedings.mlr.press/v162/wortsman22a.html)；[既有函数读出证据](01_EVIDENCE_AND_RESEARCH_DECISION.md)

### 3.8 排序变化的净值决定 AUC

对格 c 的全部正负样本对集合 $P_c$，用正式评价的预测概率定义 $s_\theta(i,j)=\mathbf1[p_i>p_j]+\frac12\mathbf1[p_i=p_j]$，则：

$$
\Delta AUC_c=\frac1{|P_c|}\sum_{(i,j)\in P_c}\big[s_{\theta'}(i,j)-s_\theta(i,j)\big].
$$

该式在完整对集合上精确，在冻结抽样面板上为估计。没有并列时等于“错误被纠正数减正确被破坏数”除以总对数；有并列时按半权计数。正式概率保留浮点 sigmoid 饱和形成的 ties，不能用 logit 排序替换这一精确闭合；logit 用于下述 margin 诊断。减少所有排序变化可能同时减少纠正和破坏，所以更稳定、更低方差或更低 LogLoss 均不能替代 AUC。

对单步 $\Delta\theta=-\eta u$，pair margin 的局部变化为 $\Delta(z_i-z_j)\approx-\eta\nabla_\theta(z_i-z_j)^\top u$，更高阶项与参数漂移随后进入。该量适于定位新操作如何作用于六格关系，其相关性或单步符号不承担完整算法的否决权。

## 4. MDMT 问题导向与合理贡献范围

正常 MDMT 的共享参数同时接收多域、多任务与不同监督形式的信号。自然 BCE 按原样本支持学习概率，条件 PAIR 对标签边界中的排序关系施加另一种监督；相同参数坐标上的二阶混合和时间尺度因此具有具体来源。C1/C2 研究这些来源如何在共同几何中被使用，无需预设小域更困难、冲突普遍有害或所有来源都必须独立。

| 需要建立的联系 | 可接受的证据 | 导向的研究结论 |
| --- | --- | --- |
| 新运算与实际更新组成 | 同状态代数验收、影子方向及来源日志 | 操作作用于预定对象，具有局部识别能力 |
| 更新与跨格排序 | 纠正/破坏、共同阶段、完整 selected 六格 | 说明哪些关系获益、代价发生在哪里 |
| 来源处理与方法增量 | H0、同 PAIR Adam、M/S/P 的匹配完整比较 | 决定保留来源专门处理还是采用更简单通用优化 |
| 共享结构与适用范围 | 条件作用域控制、各自父参照的增量 | 支持共享位置的适用性；绝对分数不单独证明共享必要性 |
| 骨干可迁移性 | 触发后的 HyperGate 同组件比较 | 扩展方法范围；无需证明单任务绝无收益 |

MDMT 问题导向来自真实的共享学习问题、可观察的来源作用和匹配收益。一个操作在单任务也有收益，与它解决 MDMT 中的具体问题相容。若收益主要由通用预条件或尺度解释，则保留有效的简化算法并相应收窄贡献；若来源选择提供进一步增量，则发展针对共享监督组织的方法。

## 5. 文献登记与核读依据

下表将正式会场年份与预印本身份分开。正文使用原论文、官方会议页和作者发布材料；所有“本项目后果”均是针对当前证据的研究推论。LaProp 的记录采用预印本身份；Bloop 的 2025 arXiv 修订不改变 ICML 2024 的发表年份。

| ID | 文献及正式身份 | 已核读依据与本项目用途 |
| --- | --- | --- |
| R01 | Polyak & Juditsky, Acceleration of Stochastic Approximation by Averaging；SIAM J. Control Optim. 30(4), **1992** | [期刊原始页](https://epubs.siam.org/doi/10.1137/0330046)：随机逼近平均的经典理论，提供噪声分析起点 |
| R02 | Kingma & Ba, Adam: A Method for Stochastic Optimization；**ICLR 2015**，arXiv 首发 2014 | [原文 Algorithm 1](https://arxiv.org/pdf/1412.6980)：矩状态、偏差修正、有限 epsilon |
| R03 | Izmailov et al., Averaging Weights Leads to Wider Optima and Better Generalization；**UAI 2018** | [原论文及会场声明](https://arxiv.org/abs/1803.05407)：同轨迹权重平均 |
| R04 | Morales-Brotons et al., Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits；**TMLR 04/2024**，arXiv 上传 2024-11 | [原文](https://arxiv.org/html/2411.18704v1)、[期刊 PDF](https://openreview.net/pdf?id=2M9CUnYnBA)：被动 EMA 动态与训练收益 |
| R05 | Wortsman et al., Model soups；**ICML 2022**, PMLR 162 | [正式论文页](https://proceedings.mlr.press/v162/wortsman22a.html)：参数平均与 logit 组合的局部关系 |
| R06 | Cha et al., SWAD: Domain Generalization by Seeking Flat Minima；**NeurIPS 2021** | [正式论文页](https://proceedings.neurips.cc/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html)：多源域泛化中的密集平均 |
| R07 | Ajroldi, Orvieto & Geiping, When, Where and Why to Average Weights?；**ICML 2025**, PMLR 267 | [正式论文页](https://proceedings.mlr.press/v267/ajroldi25a.html)：平均、效率和退火的系统比较 |
| R08 | Pagliardini, Ablin & Grangier, The AdEMAMix Optimizer: Better, Faster, Older；**ICLR 2025**，arXiv 首发 2024 | [原文 Algorithm 1](https://arxiv.org/html/2409.03137v1)、[正式会场页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a2cf225ba392627529efef14dc857e22-Abstract-Conference.html)：快慢历史与遗忘 |
| R09 | Yang et al., AdaTask: A Task-Aware Adaptive Learning Rate Approach to Multi-Task Learning；**AAAI 2023** | [原文 §Our Proposed Method、Algorithm 2](https://arxiv.org/html/2211.15055v2)、[AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/download/26275/26047)：分源分母的最近参照 |
| R10 | Fernando et al., Mitigating Gradient Bias in Multi-Objective Learning: A Provably Convergent Approach；**ICLR 2023**，2024 arXiv 修订标题加入 Stochastic | [修订原论文](https://arxiv.org/abs/2210.12624)、[ICLR 2023 官方登记](https://iclr.cc/Downloads/2023)：随机多目标方向偏差与历史估计 |
| R11 | Yuan et al., MARS: Unleashing the Power of Variance Reduction for Training Large Models；**ICML 2025**, PMLR 267 | [正式论文页](https://proceedings.mlr.press/v267/yuan25f.html)：预条件与递归梯度校正 |
| R12 | Shulgin et al., Understanding MARS: When Scaling Momentum Correction Provably Helps；**ICML 2026**, PMLR 306 | [正式论文页](https://proceedings.mlr.press/v306/shulgin26b.html)：校正尺度与梯度差分结构 |
| R13 | Zhong, Luo & Liu, DualOptim: Enhancing Efficacy and Stability in Machine Unlearning with Dual Optimizers；**NeurIPS 2025** | [正式论文页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7b76eea0c3683e440c3d362620f578cd-Abstract-Conference.html)：不同目标的解耦动量 |
| R14 | Zhong et al., DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models；**ICML 2026**, PMLR 306 | [正式论文页](https://proceedings.mlr.press/v306/zhong26j.html)、[原文 §3/§4.5](https://arxiv.org/html/2605.21539v1)：共享 base 与来源 residual |
| R15 | Jiang et al., ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning；**NeurIPS 2023** | [正式论文页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/60f9118a849e8e9a0c67e2a36ad80ebf-Abstract-Conference.html)：辅助优化与泛化收益的区别 |
| R16 | Toneva et al., An Empirical Study of Example Forgetting during Deep Neural Network Learning；**ICLR 2019**，arXiv 首发 2018 | [原论文](https://arxiv.org/abs/1812.05159)：静态训练中正确关系被破坏 |
| R17 | Buzzega et al., Dark Experience for General Continual Learning: a Strong, Simple Baseline；**NeurIPS 2020** | [正式论文页](https://proceedings.neurips.cc/paper_files/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)：历史 logits 记忆与回放 |
| R18 | Defazio et al., The Road Less Scheduled；**NeurIPS 2024** | [正式论文页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/136b9a13861308c8948cd308ccd02658-Abstract-Conference.html)：迭代平均进入在线优化 |
| R19 | Liu Ziyin, Wang & Ueda, LaProp: Separating Momentum and Adaptivity in Adam；**arXiv 预印本，2020 首发、2021 v3** | [版本登记](https://arxiv.org/abs/2002.04839)、[原文 §2、Algorithm 1](https://arxiv.org/html/2002.04839v3)：C2 的直接运算先例 |
| R20 | Hsieh et al., Careful with that Scalpel: Improving Gradient Surgery with an EMA；**ICML 2024**, PMLR 235；arXiv v2 为 2025 | [正式论文页](https://proceedings.mlr.press/v235/hsieh24a.html)、[原文 §2、Algorithm 1](https://arxiv.org/html/2402.02998v2)：主辅目标关系估计中的历史价值 |

## 6. 从理论进入实验的决定

C1/C2 都具有明确的数学差异、正向项目线索和接近的文献参照，完整训练资格已经成立。它们优先检验“如何使用已保存信息”；普通平滑、步幅和已知通用优化器通过条件控制区分。局部诊断的作用是验收运算、解释作用位置和指导后继选择，具体安排见[机制诊断](04_MECHANISM_DIAGNOSTICS.md)。

八个竞争假设及接受、排除、继续条件见[03](03_HYPOTHESES_AND_PRIORITIES.md)。主训练矩阵、最终性能门及统计单位分别由[06](06_EXPERIMENT_MATRIX.md)、[08](08_DECISION_RULES.md)、[09](09_METRICS_AND_STATISTICS.md)统一维护；本文不增加额外调参、重复失败方案或新的 Test 调用。
