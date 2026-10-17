# My Research Project v3 实验现象分析与研究判断

日期：2026-10-17  
分析基准：Caelestis1903/My-Research-Project，codex/execute-2026-10-16-v3，commit 2d04c6de6c925eca8563f81b66af3f0e95f99183。  
最新研究成果位于实验分支；分析时 main 仍为 d18f194f2760a0f2914c69a345c7145b00c25786。本文分析已上传的代码、配置、报告及聚合数据，没有重新执行训练。数值复核到逐 seed、逐域任务聚合表，不冒充从未上传的全部逐样本预测重新计算 AUC。

## 1. 最值得注意的六个发现

1. **保持 Sample 目标相同，改变采样分布仍会改变有限预算训练结果。** 这是已建立的配置内事实，不再需要重做目标等价性证明；它不等于均衡采样普遍有害。
2. **同一批 test 预测，宏平均评价可以变差，按 test 域占比评价却变好。** calAdam .003 的 E50 是直接例子。训练目标、checkpoint 选择目标和最终评价目标必须分别写清。
3. **学习率与训练阶段会改变结论。** SGD .3 的早期平均 test AUC 中 SB 更好，独立选轮后却更差；Adam .003 在前十轮优于 .01，到 E50 两臂却均更差。
4. **C 共享表示相关大块确有局部 AUC 差异归因信号，但“差距缩小”并不自动是性能改善。** E10 的独立 diagnostic 改善不稳定，E40 的替换实际降低 AUC。
5. **Adam 不是所有采样效应的必要条件；AdaTask 也没有令该配置对采样不敏感。** 这不支持复活旧 Equal 下的 Adam-history 方法路线，也不等于 AdaTask 论文整体无效。
6. **下一轮最有价值的是评价取舍、对称调参、域计数噪声和 C 更新路径的分离。** 泛化的 sampling、variance、optimizer memory 主题已有文献覆盖，应先问是否剩下可重复且有 MDMT 结构含义的边界。

以上判断分别由 [S1]、[S2]、[S3]、[S4]、[S5] 支持。

## 2. 执行完毕与问题解决的区别

v3 规定且触发的执行流程已完成；原问题表 Q1–Q46 的状态是：

|原状态|数量|含义|
|---|---:|---|
|RESOLVED|14|相应问题已有可支撑的答案|
|FALSIFIED|1|相应假设被反证|
|BOUNDARY|22|获得限定条件下的结论，不能外推为普遍结论|
|NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN|9|当前设计不能区分所问机制|

因此 BOUNDARY-CONFIRMED 和 NO_GO 应读作“本轮得到了明确边界，未达到原方案的方法化准入要求”。它们不意味着所有科学机制已经解决，也不意味着整个方向被证明没有价值。原报告关于完成规模的统计属于仓库执行回执；本文独立做的是相关表的聚合复核。[S1] [S5]

历史 S5–S8 关于 Equal 特定条件下的 Adam 历史及 late rescue 结论保持冻结：改为 Sample 后，旧 selected rescue 路线已停止。现在比较的是 Sample 目标内部的 SN 与 SB 两种采样估计器，是另一个干预变量；不把旧问题重新包装成新方法成功。[S1]

## 3. SN 与 SB 到底改变了什么

令 p_d=N_d/N，域内两任务平均损失为 L_d：

\[
J(\theta)=\sum_d p_d L_d(\theta).
\]

SN 按训练域比例 q=p 抽样，每样本权重为 1；SB 按均匀域 q_d=1/3 抽样，并乘 p_d/q_d。故两者在同一 theta、同一既有计算图下：

\[
\mathbb E[\hat g_{\rm SN}]=\mathbb E[\hat g_{\rm SB}]
=\sum_d p_d\widetilde G_d(\theta).
\]

这里的波浪号保留现有 routing 输入 emb.detach() 的导数语义，不能换成不存在的普通全 routing 导数。[S1] [S6]

v3 的 16 个固定点审计覆盖完整 1,614,254 个坐标，每 q 512 批；完整经验图梯度分解残差最大约 3.35e−7，SB/SN covariance trace 比为 2.3155–2.7522。故“同均值、不同估计器分布”有很强支持；“方差更大必然导致宏 AUC 下降”没有同等证据。[S1]

理解 SB 的重要性权重也很直观：KuaiRand 训练 p 约为 (.0861,.8693,.0446)，所以 SB 的三域权重约为 (.2584,2.6079,.1338)。大域的抽样份额从约 86.9% 降到约 33.3%，但抽到的大域样本会乘更大的权重。同目标保证的是平均贡献，不是每步的组成、噪声和优化器更新相同。ESS/n≈.4357 是权重集中程度摘要，不能单独代表全部梯度方向、参数块或泛化机制。[S1] [S6]

## 4. 最清楚的性能证据：早期与选轮不是同一个问题

下表全部为冻结 test 上的宏 AUC 差 SB−SN。正值表示 SB 更好。“早期”列是 AULC，即 1..T 每轮 AUC 的平均差，不是第 T 轮单点；各配置 T 不同，不能把它们直接用于同窗口交互推断。

|配置|共同早期窗口|早期 AULC 差|早期 seed 正/负|validation 各自选轮后的 test 差|selected 正/负|
|---|---|---:|---|---:|---|
|固定 Adam .01|1–27|−.005387|0/4|−.007069|2/2|
|固定 SGD .1|1–50|−.001430|0/4|−.002518|0/4|
|校准 Adam .003|1–22|−.009821|0/4|−.008698|0/4|
|校准 SGD .3|1–37|+.001614|4/0|−.000972|0/4|
|2B SGD .3|1–50|−.001236|0/4|−.000253|1/3|
|MovieLens SGD .3|1–16|+.003015|4/0|+.000539|1/3|
|AdaTask .003|1–14|−.005742|0/4|−.018917|0/4|

表由 [S2] 的 8,112 行逐单元数据聚合复算；主要选轮值与终报吻合。

### 4.1 两个已经可以肯定的结论

固定无动量 SGD .1 也有共同早期负差，所以 Adam 不是这种现象的普遍必要条件。固定和共同选轮读数仍有负差，所以 selection 也不是所有差异的唯一来源。[S1] [S2]

### 4.2 两个不能跳过的反例

校准 SGD 的早期 test AULC 为正、selected 为负；MovieLens 的早期为正、E50 为 −.000986 四负、selected 均值为正但仅一正三负。因此不能写成跨配置、跨阶段始终存在同方向损害。[S1] [S2]

此外固定 SGD 早期 LL 差 −.000411，与 AUC 的负差方向不同。AUC 排序和概率损失对训练路径的响应可以不同，不能称全面退化。[S1]

### 4.3 四个 seed 能说明多少

四个独立训练 seed 下，同号结果有重复性价值，但当前精确双侧符号随机化检验最小 p=.125。20,000 次 bootstrap 不会把四个训练 seed 变成 20,000 次独立训练。某些 percentile 区间不跨零可以照报，但不能由此把四同号写成同一检验下 p<.05。样本重采样、选轮重采样和训练 seed 不确定性分开。[S7]

## 5. 一个比“均衡采样有害”更有意思的现象：评价权重改变了输赢

### 5.1 同一 E50、同一模型预测的方向反转

calAdam .003：

|域|域内两任务平均 AUC 差 SB−SN|宏平均中的贡献|test 样本比例加权中的贡献|
|---|---:|---:|---:|
|d0|−.014657326|−.004885775|−.000152730|
|d1|+.003441643|+.001147214|+.003380117|
|d2|−.016007180|−.005335727|−.000119367|
|合计||**−.009074288**|**+.003108019**|

按训练 p 加权的单元 AUC 差另为 +.00101563，两正两负。test-count 加权差为四正，宏差为一正三负。[S2] [S8]

这里所有加权 AUC 都是“域任务内 AUC 的加权和”，没有跨域正负样本配对项，不能称 pooled AUC。

### 5.2 为什么会反转

KuaiRand 训练三域数量为 172782 / 1743898 / 89452；test 为 2613 / 246284 / 1870。大域 d1 占比从训练 86.928% 上升到 test 98.212%。宏平均仍给各域 1/3，于是两个较小域的下降主导宏指标；按 test 流量加权则几乎由 d1 的改善决定。[S8]

用固定预测进行权重插值：

\[
w_d(\alpha)=(1-\alpha)/3+\alpha r_{{test},d}
\]

得到平均差：

\[
\Delta(\alpha)=-.0090742877+.0121823069\alpha.
\]

均值在 alpha≈.7449 时过零。这是评价权重敏感性，不改变训练 q，也不证明人口漂移的因果机制。它直接说明“总体性能”必须说明服务于何种域权重。

AdaTask E50 也有类似 LL 反转：macro LL 差 +.039043 四正，而 train-p 加权 LL 差 −.007328 四负。[S2]

### 5.3 对研究的意义

Sample BCE 训练目标、宏 AUC 选轮、test 流量加权评价是三个不同对象。同目标的训练路径差异可能在这些评价对象上表现为不同取舍。下一轮应保留原主指标，同时把六单元贡献、排序能力和概率校准分开解释，不能事后换成有利的权重。

diagnostic 的 d0t1 只有 50 个正例、1345 行；test 该单元有 120 个正例、2613 行，但仍占 macro AUC 的 1/6。评价人口和小单元估计精度也能放大 selection/diagnostic/test 之间的差别。已有预测足以先做这部分复算，不需要为已知分布差异重新训练大矩阵。[S8] [S9]

## 6. 学习率“校准”本身存在尚未闭合的条件

现有 16 个 LR pilot 都用 SN，SN 的赢家再给 SN/SB 共用；Adam .003 位于原网格下边界，pilot 的两个种子又包含在四个 finalist 种子内。因此目前回答的是“共同固定 LR 下改变 q 的效果”，尚未回答“双方获得同等调参预算后的最好性能差”。[S5]

更重要的是 .003 相对 .01 的 test 数值：

|Adam .003−.01|SN|SB|种子模式|
|---|---:|---:|---|
|AULC 1–10|+.023848|+.019048|两臂各四正|
|E50|−.028489|−.029002|两臂各四负|
|各自 validation 宏 AUC 选轮后的 test|−.004472|−.006101|SN 一正三负；SB 四负|

这不证明“小 LR 导致过拟合”，却足以说明短期调参赢家不能直接代表长期或 selected endpoint 赢家。下一轮调参的预算、选择规则和最终用途必须一致。[S2]

2B 也不能当纯方差操作：KuaiRand 每轮由 49 次更新变 25 次更新，同名义 LR 下参数漂移和 coupled weight decay 的累积次数均改变。大 batch 后 gap 较小是有意义的边界，但没有独立证明“只要降低方差就解决了”。[S6] [S10]

## 7. C 块值得研究，但要准确理解其阳性

V7 在相同 theta/状态下，保持 SB 的其他块更新，只把 C 的更新坐标换成 SN。其衰减定义：

\[
A=1-\frac{|M_{\rm hybrid}-M_{\rm SN}|}{|M_{\rm SB}-M_{\rm SN}|}.
\]

它衡量靠近 SN 的程度，不是相对 SB 的改善率。

C 在 E10/E40 的 selection AUC 衰减均满足四 seed 至少 50%，是真实的局部归因阳性；完整训练未触发的原因是 E40 的 loss/J 联合门槛不满足。不能写成 C 无效或所有局部结果均阴性。[S3]

实际 C替换−SB：

|阶段|selection AUC|diagnostic AUC|selection LL|train J|
|---|---:|---:|---:|---:|
|E10|+.00083573|+.00015250（两正两负）|+.00017185|+.00039394|
|E40|−.00038412|−.00062990|+.00001778|+.00010295|

E10 loss/J “衰减为正”仍可以意味着相对 SB 的 loss/J 变差，因为 SN 本来就可能在这些指标上更差。E40 原本有利的 SB AUC 被拉向较差的 SN，也能产生很高衰减。[S3]

C 是 embedding、共享 STAR 权重与偏置、skip、star_mlp、shared experts 的大集合，并非一个单独机制。它占约 41.0% 参数，P 约 53.5%，所以也不能简单说 C 因为“参数最多”而阳性。当前每 theta/q 只有一个大 batch，theta 又来自 SN 轨迹；下一步应先用双路径锚点、多 draw、C 子块和逐块范数/方向控制确认，再决定是否长期改变共享拓扑。[S3] [S11] [S12]

复制 C 可以匹配初始函数，却增加有效自由度；缩窄 C 可以匹配参数量，却通常无法匹配原初始函数。下一轮必须分别设计 tied/untied 与真实/随机分组控制，不承诺不可能同时精确控制的所有因素。

## 8. Adam、AdaTask、ORDER 与困难域实验分别说明了什么

### Adam 状态

calibrated Adam/SGD 存在配置内交互，值得保留；但 m-only 与 beta2=0 的更新幅度并不与 Adam 匹配。beta2=0 的终局两臂 AUC=.5 是破坏性 floor，不能靠差距归零证明状态必要。全局更新等范数也没有固定 C/P/R/E 各块幅度。[S1] [S4] [S6]

也要保留相反证据：v-only 在共同 E50 的两臂 AUC 都有改善，并非所有状态干预都破坏学习；它在原共同早期窗口对 SN 的影响又不同。因此当前证据支持阶段依赖的状态作用，没有证明稳定的采样差距修复。[S4]

更干净的后续是同 theta、同 m/v、同梯度库下比较实际 Adam 的平均更新与“共同滞后一拍分母”的线性映射。梯度均值相同不保证当前梯度参与分母后的 Adam 更新均值相同。该诊断只在 Adam 残差仍重要时触发，不需要重复破坏性完整训练。

### AdaTask

固定 .003 的 AdaTask 仍有采样敏感性；selected test gap 为 −.018917，四 seed 负。它说明当前 task-wise state 组件没有使该设置对 q 不敏感；不说明完整论文所有调参配置都失效。gap 交互也不能替代两臂绝对性能评价。[S1] [S2] [S4]

### ORDER

同 batch multiset 不同顺序确实产生条件路径差异，但当前只是独立于数据难度的随机置换，不是课程策略。最终 964/1000 个 ORDER arm-epoch 移动了短尾 batch。长期与 selected/固定端点还能反转。纯内容顺序若以后成为核心，固定尾部位置的新对照有用；它的优先级低于对称 LR 与 C 局部归因。[S1] [S10]

### 困难域、样本量与跨环境

2x/4x 行别名复制没有增加独立信息；在 WR 下保持经验分布及耦合更新相同，不能识别真实新增样本的效果。mask 改变的是特定输入信息条件，不等于纯自然困难度。在各 seed 实际最终预算处，强 mask 的 d2 test AUC gap 从 −.004085 缩至 −.000262，但 SN/SB 的 d2 绝对 AUC 分别下降约 .030257/.026434；这不是宏 AUC 同时改善的结论。[S1] [S13]

MovieLens 的 seed 同时改变训练子集与优化随机性，每轮 5 步，KuaiRand 每轮 49 步。同 epoch 不是同优化时钟；现有比较只能给配置束边界，不能把反转唯一归因于数据集、域规模或困难度。[S1] [S5]

## 9. 文献边界必须进一步收紧

D3 与 SAF 的完整学习权重通常改变目标，不能直接作为本轮严格同目标采样机制的等价解释；AdaTask 是更接近的状态机制对照。但一般同目标采样导致不同方差/收敛，以及顺序和 optimizer memory，已有直接研究。[L1]–[L6]

本次额外检索确认两项应加入下一轮边界审查：

- **Exploring Variance Reduction in Importance Sampling for Efficient DNN Training**：明确用梯度 covariance trace 定义有效 batch size，并据此调整 SGD/Adam 学习率。故“根据采样噪声调学习率”已有覆盖。[L7]
- **Multiple Importance Sampling for Stochastic Gradient Estimation**：明确讨论单一 proposal 对高维梯度不同参数的折中，并组合针对参数子集的多种 proposal。故“不同参数块用不同采样分布”本身也不是空白。[L8]

这些论文没有自动解释本仓库所有域任务的 AUC/LL/阶段反转；但下一轮必须证明具体剩余边界，不能只凭“放到 MDMT”宣称创新。本文没有声称文献检索穷尽。

## 10. 下一步研究定位

建议问题改写为：

> 在固定 Sample 训练目标下，采样分布如何改变有限预算的域任务排序与概率质量取舍？这种效应在对称调参之后是否仍稳定存在，能否由域计数噪声或共享表示的局部更新几何解释？

三个门槛分开：

1. 现象门槛：配置内可重复、有实际量级；有限预算或边界结果本身也可成立。
2. 机制门槛：至少一个针对明确机制的受控对照，排除对应替代解释；不要求唯一中介。
3. 方法门槛：独立确认集上主指标改善，关键域任务在预先冻结的容忍界内，同时报告 loss、成本与边界。

这不改写 v3 的冻结判定。下一轮为新预注册设计，先完成 N0 评价与人口复算、N1 对称 q×LR，再用 N2 域计数和 N3 C 局部反事实决定是否触发长期架构与跨环境分支。完整矩阵、公式、预算和 Q1–Q52 的去向见配套 NEXT_EXPERIMENT_PLAN.md 和 QUESTION_DISPOSITION.json。

## 仓库证据

- [S1：v3 最终实验报告][S1]
- [S2：七类 baseline 的逐 seed、逐域任务 test 读数][S2]
- [S3：V7 局部 C 反事实结果][S3]
- [S4：V11 完整状态诊断分析][S4]
- [S5：原始 Q1–Q46 问题注册表][S5]
- [S6：v3 训练实现][S6]
- [S7：bootstrap 与不确定性范围][S7]
- [S8：KuaiRand 固定 split 与人口数量][S8]
- [S9：域任务正负例上下文][S9]
- [S10：共同预算与学习速度分析][S10]
- [S11：采样及参数分块实现][S11]
- [S12：模型前向与共享支持定义][S12]
- [S13：mask 条件逐 seed、逐单元 test 读数][S13]

[S1]: https://github.com/Caelestis1903/My-Research-Project/blob/2d04c6de6c925eca8563f81b66af3f0e95f99183/research_plans/2026-10-16_v3/execution_20261016/FINAL_REPORT.md
[S2]: https://github.com/Caelestis1903/My-Research-Project/blob/2d04c6de6c925eca8563f81b66af3f0e95f99183/research_plans/2026-10-16_v3/execution_20261016/analysis/test_baseline7_summary/seed_cell_test_readouts.csv
[S3]: https://github.com/Caelestis1903/My-Research-Project/blob/2d04c6de6c925eca8563f81b66af3f0e95f99183/research_plans/2026-10-16_v3/execution_20261016/V7_local_counterfactual_analysis.md
[S4]: https://github.com/Caelestis1903/My-Research-Project/blob/2d04c6de6c925eca8563f81b66af3f0e95f99183/research_plans/2026-10-16_v3/execution_20261016/V11_fullstate_final_diagnostic_analysis.md
[S5]: https://github.com/Caelestis1903/My-Research-Project/blob/2d04c6de6c925eca8563f81b66af3f0e95f99183/research_plans/2026-10-16_v3/execution_20261016/question_registry.json
[S6]: https://github.com/Caelestis1903/My-Research-Project/blob/2d04c6de6c925eca8563f81b66af3f0e95f99183/experiments/v3_train_20261016.py
[S7]: https://github.com/Caelestis1903/My-Research-Project/blob/2d04c6de6c925eca8563f81b66af3f0e95f99183/research_plans/2026-10-16_v3/execution_20261016/bootstrap_uncertainty_scope.md
[S8]: https://github.com/Caelestis1903/My-Research-Project/blob/2d04c6de6c925eca8563f81b66af3f0e95f99183/research_plans/2026-10-16_v3/execution_20261016/splits/kuairand_seed2022.json
[S9]: https://github.com/Caelestis1903/My-Research-Project/blob/2d04c6de6c925eca8563f81b66af3f0e95f99183/research_plans/2026-10-16_v3/execution_20261016/analysis/V8_final_domain_task_context/domain_task_context.csv
[S10]: https://github.com/Caelestis1903/My-Research-Project/blob/2d04c6de6c925eca8563f81b66af3f0e95f99183/research_plans/2026-10-16_v3/execution_20261016/V4_final_budget_and_learning_speed_analysis.md
[S11]: https://github.com/Caelestis1903/My-Research-Project/blob/2d04c6de6c925eca8563f81b66af3f0e95f99183/experiments/v2_20261016.py
[S12]: https://github.com/Caelestis1903/My-Research-Project/blob/2d04c6de6c925eca8563f81b66af3f0e95f99183/experiments/stage_S1_models.py
[S13]: https://github.com/Caelestis1903/My-Research-Project/blob/2d04c6de6c925eca8563f81b66af3f0e95f99183/research_plans/2026-10-16_v3/execution_20261016/analysis/test_V8_final_summary/seed_cell_test_readouts.csv

## 原始论文

- [L1：D3，AAAI 2024][L1]
- [L2：Stable and Adaptive Fusion for Multi-domain Multi-task Recommendation，AAAI 2026][L2]
- [L3：AdaTask][L3]
- [L4：Importance Sampling for Minibatches][L4]
- [L5：Not All Samples Are Created Equal][L5]
- [L6：Optimizer Memory Makes Shuffle Order a First-Order Source of Fine-Tuning Noise][L6]
- [L7：Exploring Variance Reduction in Importance Sampling for Efficient DNN Training][L7]
- [L8：Multiple Importance Sampling for Stochastic Gradient Estimation][L8]

[L1]: https://ojs.aaai.org/index.php/AAAI/article/view/28699
[L2]: https://ojs.aaai.org/index.php/AAAI/article/view/38491
[L3]: https://arxiv.org/abs/2211.15055
[L4]: https://www.jmlr.org/papers/v19/16-241.html
[L5]: https://proceedings.mlr.press/v80/katharopoulos18a.html
[L6]: https://arxiv.org/abs/2606.29554
[L7]: https://epubs.siam.org/doi/10.1137/25M1726339
[L8]: https://arxiv.org/abs/2407.15525

