# 2026-10-16 v3：完整终局实验方案

## 0. 总原则：v3 的任务是“闭环”，不是“继续堆实验”

v2 已经把研究从 Equal-objective / Adam-history 问题推进到 Objective-Equivalent Sampling 问题。

当前定义：

[
J(	heta)=sum_d p_d,mathbb E_{xsim P_d}[ell(x;	heta)]
]

采用 domain sampling distribution (q) 时，importance-corrected estimator 为：

[
hat g_q(	heta)=
rac1Bsum_{i=1}^{B}
rac{p_{d_i}}{q_{d_i}}g_i.
]

因此：

- Natural：(q=p)，权重为 1；
- Balanced：(q_d=1/D)，当前三域情况下权重为 (3p_d)。

理论上：

[
mathbb E_q[hat g_q]=
abla J(	heta)
]

但有限 batch 下：

[
hat g_q
eq 
abla J(	heta)
]

而且 (q) 会改变 estimator 的 covariance、sample repetition、domain exposure、batch composition 和时间序列。

v3 必须回答的不是“Balanced 为什么不好”，而是：

[
q
ightarrow
{	ext{exposure,repetition,order,geometry}}
ightarrow
	ext{trajectory}
ightarrow
	ext{generalization}
]

中哪些箭头真实存在、哪些只是测量或选择伪象。

---

# Phase V0：历史结论封存与实现审计

## V0.1 已经关闭的旧路线

把 S5–S8 的结论作为 frozen prior：

1. Equal objective 下的 target-only/历史 Adam 救援是特定 objective 条件下的下游机制；
2. 转入 Sample objective 后，该救援在 selected checkpoint 上基本消失；
3. 后期干预如果只能减缓已经发生的 overfitting，而不能超过原始最佳 checkpoint，不视为方法价值；
4. 不得重新把上述旧路线当成 v3 的主问题。

## V0.2 v2 已完成事实

冻结以下事实，不重复作为主要实验：

- Sample objective 下 Adam 的 SB-SN：4 seed 中 3 负；
- Sample objective 下 SGD 的 SB-SN：4 seed 中 4 负；
- SGD 效应小于 Adam，但稳定性更高；
- Adam/SGD interaction 区间跨 0，不能声称 optimizer-specific；
- v2 P1 因 SGD 仍保留负效应而进入 Case C；
- v2 P2-B、P7-SGD、P8、P9、P10 等尚未完整闭环；
- P6 已观察到 covariance trace / effective-rank 差异，但尚不足以完成机制识别。

## V0.3 只修会影响结论的工程问题

必须审计：

- objective 权重公式；
- domain sampling 概率；
- batch size；
- 总样本数；
- optimizer step 数；
- seed；
- data split；
- checkpoint selection；
- test 是否被提前读取；
- metrics 是否与训练协议一致。

不进行与科学结论无关的“防御性测试堆砌”。

**V0 通过条件：** 所有会改变 scientific conclusion 的实现问题均已关闭；否则停止后续阶段。

---

# Phase V1：Objective Equivalence — 先证明“同 objective”真的成立

这是 v3 新增的硬门槛。

## V1.1 精确 empirical objective

在固定 (	heta) 下，分别计算每个 domain 的完整 empirical gradient：

[
G_d(	heta)=
rac1{N_d}sum_{xin d}
ablaell(x;	heta)
]

再构造：

[
G^*(	heta)=sum_d p_dG_d(	heta).
]

至少在 epoch 1、5、10、20、40 的若干固定 checkpoint 上完成。

## V1.2 Monte Carlo estimator audit

对 SN/SB 分别抽取至少 512 个独立诊断 batch，计算：

- (hat g)；
- (|hat g-G^*|)；
- cosine((hat g,G^*))；
- estimator mean；
- covariance；
- block-wise covariance。

必须验证：

[
|mathbb E[hat g_q]-G^*|
]

处于 Monte Carlo 误差范围内。

若不成立：

> 不得进入机制分析；优先判定实现/objective mismatch。

## V1.3 单 batch 数值身份

同一个 batch 同时计算：

- unweighted natural estimator；
- balanced importance-weighted estimator；
- domain decomposition。

检查权重和样本计数是否精确。

## V1.4 deterministic diagnostic

在一个可控的小型固定子集上，用 exact/full-batch gradient 做：

- exact objective optimization；
- SN/SB stochastic optimization。

目标不是证明真实模型的最终 full-batch 性能，而是判断：

> 当 stochastic estimator 被移除后，SN/SB 差异是否显著收缩。

若收缩：

**支持“有限样本 estimator / trajectory effect”。**

若仍存在：

必须回查 objective implementation 或模型数据管线。

---

# Phase V2：Overfitting、Learning Speed 与真正泛化差异

这是 v3 的第一核心阶段。

## V2.1 不能再用“最终 selected AUC”单独回答问题

对每个 seed、每个 arm，完整保存 epoch 1..50：

- validation macro AUC；
- validation macro LogLoss；
- test AUC/LogLoss，但 test 只在 selection 冻结后读取；
- train objective；
- train loss；
- 每域 AUC；
- 每 task AUC。

## V2.2 建立独立 diagnostic validation

若现有验证集允许，将 validation 固定拆成：

- selection validation：唯一允许用于 checkpoint selection；
- diagnostic validation：从未参与 LR/方法/selection 决策。

拆分必须在所有 v3 训练前冻结。

diagnostic validation 只用于事后判断：

> 某个 sampling effect 是否在 overfitting 前就存在。

## V2.3 定义 overfitting onset

对每个 arm/seed，在 diagnostic validation 上：

1. 找到历史最高 AUC；
2. 若之后 AUC 连续三个 epoch 低于峰值，并且下降超过该 epoch-level metric 的 bootstrap uncertainty，则标记 overfitting onset；
3. 同时检查 LogLoss 是否持续恶化。

这个规则只用于分类，不用于修改模型。

## V2.4 四种性能 readout

必须同时报告：

### A. Common-epoch effect

[
Delta(t)=M_B(t)-M_N(t)
]

在：

[
tin{5,10,15,20,30,40,50}
]

报告。

### B. Pre-overfit effect

只在两个 arm 都尚未进入 overfitting 的 epoch 比较。

### C. Learning-curve area

计算固定预算区间的 AULC，尤其是：

- epoch 1–10；
- epoch 1–20；
- 各 seed 的共同 pre-overfit window。

### D. Selected effect

[
Delta_{selected}
=
M_B(t_B^*)-M_N(t_N^*)
]

其中 (t^*) 只能由 selection validation 决定。

## V2.5 三种结论

### 若只有 late epoch 出现改善

且没有超过双方原始最佳 checkpoint：

> **Late-overfitting rescue only。**

不进入方法化。

### 若早期已存在稳定 effect，但延长训练后消失

> **Optimization-speed / finite-budget effect。**

继续做 V4 的 optimization-budget analysis。

### 若 pre-overfit 和充分训练后仍存在稳定 generalization gap

> 才允许进入真正 mechanism analysis。

---

# Phase V3：Checkpoint Selection 完整拆解

## V3.1 必须区分三种对象

1. trajectory：
   [
   M_B(t)-M_N(t)
   ]

2. independently selected：
   [
   M_B(t_B^*)-M_N(t_N^*)
   ]

3. common-selected epoch：
   在同一个冻结 epoch 比较。

## V3.2 Selection uncertainty

对 selection validation 做 row-level bootstrap：

- 每个 seed 至少 1000 次 bootstrap；
- 每次重新执行 selection；
- 得到 (t^*) 分布；
- 得到 selected AUC 分布；
- 得到 selected SB-SN effect 分布。

这不能替代 seed uncertainty，必须分开报告。

## V3.3 Selection robustness

测试：

- macro AUC selection；
- macro LogLoss selection；
- early-window selection；
- fixed-budget selection。

Test 集只能在所有规则冻结后读取。

## V3.4 最终判断

若 trajectory gap 在 common epoch 已存在：

> selection 只是 modifier。

若只有 selected gap：

> 重点调查 selection/validation noise。

若 selection 能发生方向反转：

> 把“selection instability”列为独立结果。

---

# Phase V4：Optimization Budget 与“是不是只是训练速度不同”

这是 v3 必须新增的第二核心阶段。

## V4.1 当前 SGD LR 边界问题

v2 SGD 的 (mathrm{LR}=0.1) 为搜索边界 winner，因此不能直接把 Adam 与 SGD 的结果当作公平 optimizer equivalence。

重新做 validation-only LR calibration：

- SGD：([0.03,0.1,0.3,1.0])；
- Adam：([0.003,0.01,0.03,0.1])；
- 2 seeds pilot；
- 冻结规则后用 4 seeds 复核 finalist；
- calibration 只允许看 selection validation。

## V4.2 等样本预算

保持：

- total examples；
- total optimizer updates；
- epoch 数；
- model initialization；
- data split。

分别比较 SN/SB。

## V4.3 延长预算

若 epoch 50 尚未达到训练 objective plateau，则延长至 100 或更长，直到：

- train objective 基本平台；
- 或 validation 已明显进入 overfit。

重点不是用更长训练“挑最好结果”，而是判断：

> SB 是否最终追上 SN，还是差异在充分优化后仍存在。

## V4.4 Batch-size sensitivity

以当前生产 batch (B_0) 为基准，至少加入：

- (B_0)；
- (2B_0) 或等价 gradient accumulation。

总样本预算保持一致。

如果硬件限制导致只能做 accumulation，必须明确记录：

> optimizer-step geometry 是否发生变化。

## V4.5 结论

- 差异随 batch 增大快速消失：支持 stochastic-estimator explanation；
- 差异只在固定 step 下存在、按 examples 后消失：优化速度解释；
- 差异在大 batch / 长预算后仍稳定：进入 V5/V6 的更深机制。

---

# Phase V5：Repetition / Coverage / Exposure 因果拆解

v2 的 cyclic control 同时改变 repetition、coverage、temporal dependence，因此不能把它直接当作单因素识别。

## V5.1 设计原则

所有条件必须保持：

- Sample objective；
- total examples；
- model；
- optimizer；
- seed；
- domain exposure target。

只改变一个候选因素。

## V5.2 Repetition control

每个 (q) 都构造两种 domain-internal sampler：

### WR

with-replacement。

### PERM

对每个 domain 使用固定随机 permutation，耗尽后再开始下一轮 permutation。

这样：

- domain quota 相同；
- marginal sampling distribution 相同；
- replacement/repetition pattern 不同。

比较：

[
N_{WR}leftrightarrow N_{PERM}
]

[
B_{WR}leftrightarrow B_{PERM}
]

如果 B 的 effect 只在 WR 出现：

> repetition 是强候选。

## V5.3 Order control：同一 multiset，不同 batch order

这是 v3 的关键新实验。

对每个 seed、每个 q：

1. 先生成完整 epoch 的 batch list；
2. 保存每个 batch 的 exact row indices；
3. 构造两个 replay：

- ORDER-A：原始 batch 顺序；
- ORDER-B：完全相同的 batch multiset，仅改变 batch 顺序。

因此每个 optimizer step 的 batch 内容不变：

[
{G_1,G_2,ldots,G_T}
]

完全相同，只改变：

[
(G_1,G_2,ldots,G_T)
ightarrow
(G_{pi(1)},G_{pi(2)},ldots,G_{pi(T)}).
]

这才是干净的 order intervention。

## V5.4 Domain exposure

在 PERM 条件下比较：

[
N_{PERM}leftrightarrow B_{PERM}
]

此时 repetition 被显著控制，若 gap 仍存在：

> exposure/domain composition 仍是候选机制。

## V5.5 Coverage

记录：

- unique rows / epoch；
- cumulative unique rows；
- per-domain unique coverage；
- repeat count distribution；
- time-to-first-seen；
- time-to-90% coverage。

如果性能差异与 coverage 高度同步，但在 coverage-matched control 下消失：

> coverage 是候选 mediator。

## V5.6 Batch composition

进一步记录每个生产 batch：

- domain counts；
- task counts；
- effective sample size；
- weight variance；
- maximum importance weight。

判断是否为：

> 同样 domain exposure，但 batch-level composition 不同。

---

# Phase V6：Gradient Estimator Geometry

只有 V1/V5 表明 sampling effect 不是简单 implementation error 后，才把 geometry 上升到主要机制。

## V6.1 固定 theta 的 gradient probe

在 epoch：

[
1,5,10,20,40
]

至少 512 个独立 batch。

每个 batch 记录：

- full gradient；
- C/P/R/E block gradient；
- norm；
- cosine to exact (G^*)；
- covariance。

## V6.2 必须报告的几何量

### 一阶

[
E[g],quad bias=E[g]-G^*
]

### 二阶

[
Tr(Cov(g))
]

[
lambda_{max}
]

[
effective rank
]

### 各向异性

[
lambda_{max}/Tr(Cov)
]

以及 top-k explained variance。

### 时间

[
Corr(g_t,g_{t-k}),quad k=1,2,5
]

### 方向

- cosine(g, exact gradient)；
- consecutive-gradient cosine；
- cross-domain gradient cosine；
- cross-task gradient cosine。

## V6.3 参数块

分别分析：

- shared C；
- P；
- routing R；
- E / 其他已有定义 block。

不得因为 C/R 的距离较大就直接称其为 mediator。

## V6.4 Mediator 判定

只有当：

1. sampling 改变 estimator；
2. estimator 改变 block update；
3. block intervention 能显著削弱 downstream effect；

才能称该 block 为必要 mediator 候选。

否则只写 association。

---

# Phase V7：Shared / Private Architecture 必要性

## V7.1 第一层：局部反事实

在相同 (	heta) 上构造 one-step counterfactual：

- SN update；
- SB update；
- SB-with-C-difference-blocked；
- SB-with-R-difference-blocked；
- 必要时 P/E block。

观察下一步：

- validation change；
- objective change；
- block displacement。

## V7.2 第二层：full-training intervention

只有局部结果显示 C/R 可能是必要 mediator 才做：

- shared-C privateized；
- shared-R/private-router control；
- 参数量尽量匹配；
- 初始化和训练预算匹配。

比较 SN/SB gap 是否消失。

## V7.3 结论

- gap 消失：shared block necessity 有证据；
- gap 缩小但不消失：multiple pathways；
- gap 不变：停止“shared parameter 是核心机制”的叙事。

---

# Phase V8：Domain Size / Difficulty / Task Imbalance

## V8.1 禁止“小域=困难域”

所有解释必须同时报告：

- domain size；
- positive rate；
- baseline difficulty；
- per-domain/task AUC；
- gradient norm；
- exposure；
- repeat rate。

## V8.2 Size manipulation

若工程可行，对小 domain 做 dataset duplication：

- 2x；
- 4x。

保持 row distribution / labels 不变，使 empirical domain distribution基本不变，只改变 finite-pool size。

观察：

[
SB-SN
]

是否随 pool size 改变。

## V8.3 Difficulty manipulation

选择一个可控输入 corruption，例如 feature masking/noise：

- clean；
- mild；
- strong。

保持：

- domain size；
- q；
- sample counts；
- labels。

只改变可控的 domain difficulty proxy。

若 effect 随 difficulty 单调改变，记录为 sensitivity；不能把它等同于自然 domain difficulty 的充分因果证明。

## V8.4 Task/domain imbalance

在每个 domain-task cell 统计：

- train count；
- validation count；
- positive count；
- exposure；
- loss contribution。

只有 3 physical domains 时，不进行虚假的高自由度 domain-level regression。

若样本数不足以识别 task/domain causal effect：

> 直接把“当前数据规模无法进行 domain-level causal identification”作为边界结果，而不是强行拟合回归。

---

# Phase V9：Cross-Environment Boundary

至少完成：

## E1 KuaiRand + M3OE

主环境。

## E2 MovieLens / 项目中已有第二 domain configuration

沿用当前工程，不随意更换模型。

最小矩阵：

- Sample objective；
- SN vs SB；
- 关键 optimizer；
- 4 seeds。

若 E2 不复现：

> 不视为失败；形成明确 boundary。

必须进一步判断 boundary 来自：

- imbalance；
- dataset；
- architecture；
- objective；
- optimizer；
- overfitting regime。

若 E2 复现，再复制最关键的一个 mediator experiment。

---

# Phase V10：SAF / D3 / AdaTask 与文献边界

## V10.1 Literature audit

执行日期固定为实验启动日。

搜索至少覆盖：

- objective-equivalent importance sampling；
- multi-domain sampling；
- domain balancing；
- sample reweighting；
- training order；
- data mixing；
- optimizer memory；
- Adam state；
- task-wise optimizer；
- MDMT/MoE recommendation。

尤其检查 2024–2026 新工作。

## V10.2 D3

先做 objective audit：

[
J_{D3}(	heta)stackrel{?}{=}J_{Sample}(	heta).
]

如果 D3 改变 objective：

> D3 可以作为“现有方法能否缓解现象”的 baseline，但不能声称它已经解释 objective-equivalent sampling mechanism。

若存在 objective-preserving configuration，再进行同 objective 对照。

## V10.3 SAF

审计 sample weights：

- 是否改变 expected objective；
- 是否只是 reweighting；
- 是否能保持 estimator expectation 不变。

只有 objective-equivalent SAF configuration 才能作为 mechanism-equivalent comparison。

## V10.4 AdaTask

AdaTask 是最重要的 optimizer-state 对照之一。

必须测试：

> 在不改变 Sample objective 的条件下，task-wise optimizer state 能否消除 SN/SB gap？

若不能：

> AdaTask 与当前 estimator-induced sampling problem 不是同一机制。

若能：

> 才需要进一步拆解到底是 task-wise state 还是更一般的 adaptive-state mediation。

## V10.5 相关工作边界

必须额外检查近期关于 domain order、optimizer memory、minibatch perturbation dynamics 的工作。

即使它们研究的是单任务或 fine-tuning，也必须判断：

[
	ext{domain sampling in MDMT}
]

是否只是它们的特殊实例。

不得仅因为“场景不同”就声称创新。

---

# Phase V11：Optimizer Interaction，不再默认 Adam 根因

## V11.1 基础结论

若：

[
SB-SN<0
]

在 Adam 和 SGD 都成立：

> sampling effect is optimizer-independent at the necessity level.

## V11.2 interaction

估计：

[
I=
(Delta_{Adam})-(Delta_{SGD}).
]

只有 interaction 稳定且有实际量级，才继续：

- Adam beta1=0；
- Adam beta2=0；
- m-only / v-only；
- full-state shadow replay。

否则直接关闭 P2-A。

## V11.3 为什么这样设计

这是为了避免重新重复 S5–S8。

S5–S8 已经把“Adam history 是旧 Equal 问题的根因”这条路线收束。

v3 只有在新的 sampling effect 显示 optimizer-specific interaction 后，才允许重新进入 state decomposition。

---

# Phase V12：反事实总闭环

构造最终 causal graph：

[
X=q
]

候选 mediator：

[
Zin
{
exposure,
repetition,
coverage,
order,
gradient geometry,
shared block,
optimizer state
}.
]

结果：

[
Yin
{
pre	ext{-}overfit AULC,
common	ext{-}epoch AUC,
selected AUC,
test AUC
}.
]

至少完成：

### CF1

同 objective：

[
q_Nightarrow q_B
]

### CF2

同 sampling：

[
Adamightarrow SGD
]

### CF3

同 q，阻断 repetition。

### CF4

同 q、同 multiset，改变 batch order。

### CF5

同 q，阻断候选 parameter block。

### CF6

大 batch / deterministic gradient。

如果某 mediator 的 intervention 让 effect 消失，而其他主要解释不能做到：

> 才允许写“necessary mediator candidate”。

否则只能写：

> associated mechanism.

---

# Phase V13：最终是否值得方法化

## GO-METHOD 必须同时满足

1. pre-overfit 阶段存在稳定 effect；
2. effect 不是 selection artifact；
3. effect 不是单纯训练速度差异；
4. effect 至少在两个环境成立，或者明确具有可解释 boundary；
5. 至少一个 mediator 有反事实支持；
6. 主要 alternative explanations 被排除；
7. SAF/D3/AdaTask 不能直接覆盖；
8. 方法能改善真正 primary endpoint，而不是只减轻 late overfitting；
9. 方法在 4 seeds 上稳定；
10. 不退化原始 objective 或其他 task/domain。

## NO-GO

任何以下情况都停止方法开发：

- 只在 late overfitting 出现；
- 延长训练即可消失；
- 只在 checkpoint selection 中出现；
- 只在一个 optimizer；
- 只在一个数据环境且无机制边界；
- exposure/repetition/order 已完整解释且已有方法覆盖；
- 新方法只是 balancing / weighting / Adam modification 的简单组合。

---

# Phase V14：终局报告与动态问题闭环

终局报告必须回答：

1. 同 objective 不同 q 是否真的改变 training trajectory？
2. 是否存在 objective implementation mismatch？
3. 是否只是 checkpoint selection？
4. 是否只是 overfitting rescue？
5. 是否只是 optimization speed？
6. 是否需要 Adam？
7. Adam interaction 是否真实？
8. repetition 是否必要？
9. exposure 是否必要？
10. coverage 是否必要？
11. order 是否必要？
12. gradient geometry 是否必要？
13. shared C/P/R/E 哪些只是相关，哪些可能是必要 mediator？
14. domain size 是否解释？
15. difficulty 是否解释？
16. task/domain imbalance 是否解释？
17. 是否跨环境？
18. SAF/D3/AdaTask 是否覆盖？
19. 近期 domain-order / optimizer-memory 文献是否覆盖？
20. 是否存在未覆盖的 mechanism gap？
21. 是否值得方法化？

除此之外，所有新暴露问题全部进入动态问题表。

---

# Phase V15：实验终止与升级规则

## Stop-1：Objective 不等价

停止机制研究，回到实现。

## Stop-2：Sampling effect 不复现

停止故事，转数据/随机实现审计。

## Stop-3：Effect 仅 late overfitting

关闭方法化，输出 negative/boundary result。

## Stop-4：Effect 仅 selection

关闭 trajectory mechanism，转 selection uncertainty。

## Stop-5：Effect 随训练预算消失

归类为 optimization-speed / finite-budget effect。

## Stop-6：Repetition 完全解释

关闭更复杂的 Adam/geometry 叙事。

## Stop-7：Order 完全解释

把结果重新定位为 sequence/order effect，并检查文献覆盖。

## Stop-8：No optimizer interaction

关闭 Adam-specific state decomposition。

## Stop-9：Shared block 不必要

不声称 MDMT shared-state mechanism。

## Stop-10：第二环境消失

形成 boundary，不强行泛化。

## Stop-11：已有工作覆盖

形成 negative novelty finding。

## Stop-12：出现阻塞性新问题

自动扩展 Question Registry，并设计最小区分实验。

---

# Phase V16：最终成功标准

v3 不以“得到正结果”为成功。

成功定义为：

> 对当前原始问题给出可证据化的终局结论，并将所有会改变该结论的新问题继续追踪到结论、边界或不可识别性；不允许为了形成漂亮故事而隐藏矛盾结果。

