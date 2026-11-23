# 02 — 机制选择、数学定义与假设

## 1. 最终选择

### M1：共享梯度一致性塑形（主候选）

**核心假设 H1**：在固定共享表示 U 时，改变专有参数 P 可以改变各 cell 对 U 的梯度；存在一个可由 consistency objective 捕获的方向，使六个 cell 的共享梯度更接近自然联合梯度。该变化不必然提高 AUC，必须独立测量。

对 cell c：
g_c(U,P) = ∇_U L_c(U,P)
g_J(U,P) = ∇_U L_joint(U,P)

方向一致性损失：
L_cons(P) = Σ_c w_c [1 − (g_c·g_J)/(||g_c|| ||g_J|| + ε)]

主诊断固定 w_c=1/6；natural traffic 权重仅作为描述性敏感度，不替代主定义。

M1 action：固定 U，使用训练 batch 计算 L_cons，通过二阶自动微分更新 P；随后在同样的训练预算下执行一次 joint update。比较 P-consistency、P-BCE、coordinate-shuffled consistency 与 joint baseline。

**区分性**：D2 只比较 P 范围、更新顺序和联合更新。M1 优化的是 P 对共享梯度几何的塑形目标，新增了 ConsMTL 类型的二阶机制，不是重做静态 U/P scope sweep。

**预测**：
- L_cons 应下降；
- 六个 cell 的 g_c 与 g_J cosine dispersion 应下降；
- 如机制有预测价值，独立 Validation panel 上排序变化应与 consistency 变化相关；
- 若只有几何变化而没有预测收益，接受机制现象但不晋级算法。

控制：
- joint baseline，两次 joint step；
- P-BCE step + joint step（相同步数/额外计算对照）；
- within-tensor coordinate-shuffled consistency + joint step；
- consistency-only（只测直接 P output response，不参与预算主比较）。

### M2：样本×任务条件影响（主诊断候选）

**核心假设 H2**：相同训练样本在不同任务上产生的共享参数更新，对不同 cell 的排序 margin 有不同的预测影响；task-specific score 优于 task-agnostic 或 shuffled score。

对样本 i、任务 t、目标 cell c：
s(i,t,c) = ∇_θ m_c^cal · Δθ(i,t)

m_c^cal 是从 Train-only calibration pairs 构造的正负样本 logit margin；Δθ(i,t)=θ_after−θ_before 是 anchor Adam state 下样本 i 的任务 t 梯度所对应的带符号虚拟 Adam 单步位移。正分数表示预测 margin 增大。此量仅为局部一阶预测代理，不称为因果 influence。

Action：在每个 domain-task 内归一权重、总 loss mass 不变地调整样本权重。权重 clipping 与缩放范围必须 S0/S1 前冻结，并轻量搜索。

对照：
- natural uniform weights；
- task-specific score weights；
- task-agnostic collapsed score；
- within-domain shuffled score。

必须匹配 row exposure、batch size、loss mass、optimizer steps、Adam state 和 weight decay。

### M3：阶段性与组合式影响（条件诊断）

**触发**：M2 的 top/bottom group 在至少两个 seed、至少两个 anchors 方向一致，且样本支持充分。

固定总 batch size n：
I_AB = ΔU(A/B mixture) − 0.5ΔU(A-only) − 0.5ΔU(B-only)

A-only、B-only、A/B mixture 的 batch size 相同；混合组由 A、B 各 n/2 行组成。另设 A/A 与 B/B 混合 control 以区分单组内部重复。不同 epoch anchors 分别报告，不把阶段差异当作课程收益。

### M4：域任务条件有害特征依赖（仅审计）

仅在可定位具体字段、可定义干预并有标签语义依据时继续。统计相关/HSIC 变化本身不构成有害依赖或因果证据。

### M5：语义合法多标签增强（仅审计）

必须为每种字段变换提供标签保持依据。MovieLens/KuaiRand 的 user_id、movie_id、domain indicator、类别ID、稀疏离散属性不得任意置换或插值。没有业务/数据生成语义支持则 STOP。

## 2. 辅助测量

- M6 输出敏感度：记录 logits、pair margins、六格 AUC 对参数更新的响应，不作为独立方法。
- M7 seed 分布：报告五个配对 seed 的每 seed 方向、均值、SD、区间和符号数；不把 seed 内多次 fork 扩充 n。
- M8 通用冲突控制器：排除，避免与 PCGrad、GradCraft、DTJS、ConsMTL 的通用机制重叠。

## 3. 可证伪条件

M1：
- 若 consistency objective 无法稳定降低梯度 dispersion，判当前实现 NO_GO；
- 若能降低 dispersion，但独立 V_B 的 cell AUC/pair correction 无响应，接受“梯度几何可塑但不转化为排序收益”，停止当前方法化；
- 若改善只来自多一次 P 更新，P-BCE 匹配对照也应改善，不能归因于 consistency。

M2：
- 若 task-specific score 与 task-agnostic/shuffled 无区别，说明当前数据/模型下未识别出可用的任务条件信息；
- 若 score 只预测 loss 变化、不预测 pairwise ranking，停止作为 AUC 机制；
- 若结果由标签支持极稀疏或某 seed 主导，标记 INCONCLUSIVE。

M3：
- 若 interaction residual 在 seed/anchor 间符号不稳定，停止 group attribution 扩张；
- 若 interaction 存在但无可执行动作收益，只保留诊断结论。

## 4. 超参数

只搜索：
- M1 consistency step size multiplier：0.1 / 0.3 / 1.0 × canonical Adam lr；
- M1 inner steps：1 / 2，默认 1；
- M2 score temperature/clip quantile：预先固定两档轻量值，不做大网格；
- M2 weight amplitude：0.05 / 0.10 / 0.20，作用为 w=normalize(1+α·clip(score))，每个 domain-task 内 mean=1。

S1 用五 seed 综合 Validation AUC 与机制读数选择一个配置；S2 冻结，不得按单 seed 选择。若新超参与 D2 的现有值重叠，仍须记录来源和是否为新参数。


## 5. 条件晋级算法：CGPA

仅当 M1 在 S2 和 S3 通过时，才执行 CGPA 完整训练。

对每个 batch：
- 计算六格共享梯度 g_U,c 与自然联合梯度 g_U,J；
- 构造 L_cons；
- 固定 U，对 P 做一次 consistency-only Adam step；
- 重新计算同一 batch 的 canonical joint BCE；
- 执行一次 joint Adam step；
- 执行一次固定 EMA tick。

consistency step 的学习率乘数从 0.1、0.3、1.0 轻量搜索，inner steps 固定为 1；若 S3 显示单步不足，inner steps=2 仅可在 S3 预注册前作为第二档，不允许运行中追加。最终超参由五 seed 综合 Validation 结果选择并在 S4 freeze 前固定。

对照 P-BCE-prestep + joint 使用相同两次 optimizer steps、同 batch、相同 EMA ticks，区分 consistency objective 与额外私有更新收益。CGPA 不叠加 M2 weighting，不叠加 M3 group sampler，不改变固定 EMA。
