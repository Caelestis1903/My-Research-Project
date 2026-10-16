# 2026-10-16 v2：MDMT 中 Objective-Equivalent Sampling 的机制、边界与终局实验方案

> **状态：PROPOSED_NOT_EXECUTED**
>
> 本方案是 Stage2D 完成后的完整机制闭环计划。目标不是预设“Balanced 有害”“Adam 是根因”或“某个新方法有效”，而是把当前所有尚未解决的科学问题逐一转化为可证伪实验；每个分支都定义明确的终止条件。若实验不能支持原假设，则必须记录反例并转向新的问题，不能用结果选择性保留故事。

---

## 0. 总研究问题

当前已经确认：

1. Equal 与 Sample 的历史差异主要来自 objective/loss aggregation，而不是可以简单归因于“小域困难”或“Adam history”。
2. Sample 已成为当前强基线。
3. Stage2D 进一步发现：在**固定 Sample objective** 时，仅改变 sampling distribution（Natural (q=p) vs Balanced (q=1/3)），真实训练结果仍发生稳定变化：
   [
   SB-SN\approx -0.0104\;AUC,
   ]
   且 4/4 seeds 同方向。
4. 共同 epoch 比较显示该差异存在于训练轨迹中，而不是纯 checkpoint-selection artifact。
5. sampling 改变了 Adam (m,v) 与参数轨迹，但尚未证明 Adam state 是性能差异的因果中介。
6. 仅用 gradient covariance、ESS 或“方差更大/更小”无法解释全部现象。
7. Equal 下 Balanced 的 selected 与 common-epoch 行为明显不同，提示 selection 与 trajectory 存在交互。
8. 小域并不等于困难域；不同小域可以出现相反效应。

因此 v2 的最终问题不是“Balanced 好不好”，而是：

> **当两个训练程序具有相同 population objective、相同期望梯度时，为什么在 MDMT 中仍可能产生系统性的不同训练轨迹和泛化结果？这种差异究竟来自 sampling/exposure、optimizer state、shared-parameter geometry、selection，还是它们的交互？其边界在哪里？**

---

## 0.1 未解决问题数量不设上限，但已有问题必须逐项闭环

本方案对**实验结束后剩余问题、新暴露问题以及需要进一步理清的研究方向的数量不设任何上限**。发现越多，只要这些问题确实由实验和证据暴露出来，就应当全部记录，而不是为了满足“终局必须简化”为一个问题而人为合并或删除。

同时，必须严格区分两类问题：

1. **v2 开始前已经存在的问题**：不得因为实验分支、结果不理想或成本原因而被静默跳过。P1–P9 应覆盖这些问题；每个问题至少要有明确的实验验证、反事实尝试、边界测试或“因前置条件不成立而关闭/不适用”的正式结论。
2. **v2 实验过程中新暴露的问题**：数量不限。每一个新问题都应单独登记其来源、证据、当前结论、未解决部分以及建议的后续方向；不要求本轮实验把所有新问题全部解决，也不要求把它们压缩成一个问题。

因此，本计划真正要求的是：**实验开始前已有的问题必须全部进入实验闭环并被尝试解决；实验结束后新产生的问题可以是一个、多个、很多个，数量本身不是终止条件。**

---

# 1. 最终闭环结构

整个研究按下面因果链逐层切断：

[
\boxed{
Objective
\rightarrow
Sampling/Estimator
\rightarrow
Exposure
\rightarrow
Gradient Sequence
\rightarrow
Optimizer State
\rightarrow
Parameter Trajectory
\rightarrow
Selection
\rightarrow
Generalization
}
]

每一层必须通过**保持其他层不变**的实验确定是否为必要条件。

最终允许的结论只有三类：

### C1：已证实机制
实验支持、跨 seed 稳定、反事实对照成立。

### C2：已证伪机制
关键必要条件被移除后现象消失，或出现稳定反例。

### C3：未识别但边界明确
多个机制无法仅凭当前实验区分；必须把**所有由实验暴露出的 unresolved/open problems 逐项明确记录**，并为每个问题说明当前证据、尚未识别的原因、已尝试的区分实验以及最小后续实验。**问题数量不设上限，也不要求压缩成一个或有限个问题。**不能为了形成单一故事而选择性保留其中一个、忽略其他已经被实验暴露的问题。

---

# 2. 全局实验底座与不可修改项

除非某阶段明确要求改变 optimizer/model，否则保持：

- M3OE 当前 fixed-fusion 实验底座；
- 单一 optimizer；
- 不恢复可学习 objective 权重；
- 不恢复 AutoML/DARTS；
- 数据、train/validation/test split 固定；
- seeds = 2022, 2023, 2024, 2025；
- 主指标：macro AUC、macro LogLoss；
- 必须同时报告 domain × task 指标；
- 所有 paired contrasts 使用相同 seed、相同数据划分；
- 所有最终 checkpoint 必须保存 prediction、state、optimizer state、config、hash；
- 训练轨迹必须至少保存 epoch 0/1/5/10/20/30/40/50，关键阶段增加 step-level readout；
- selection 必须同时报告：
  - independent validation-selected；
  - common-epoch；
  - fixed-endpoint；
  - test readout；
- 禁止通过查看 test 结果选择 checkpoint；
- 所有机制结论必须优先使用 paired seed differences，而非独立均值；
- 任何新增方法必须先完成机制识别实验，不得提前调参追求 AUC。

---

# 3. Phase P0：实验系统与历史结论封存

## P0.1 目的

防止新实验重新污染已经关闭的问题。

## 必须冻结

### 已关闭问题
- “小域天然困难”；
- “Equal/Sample 的差异主要是 Adam history”；
- “generic momentum 必然有效”；
- “Balanced 天然优于 Natural”；
- “gradient conflict 可以由一个简单 variance scalar 解释”。

### 已知强事实
- Sample objective 是当前强基线；
- Stage2D 的 SB-SN 负效应；
- Equal 的 objective-weighting effect；
- selection 会改变 observed effect；
- sampling 会改变 Adam state。

## P0.2 通过条件

只有所有已有 Stage2D artifacts 可复现、hash 一致、seed pairing 一致，才进入 P1。

---

# 4. Phase P1：Optimizer Necessity —— Adam 是否是必要条件？

这是整个 v2 的第一分叉实验。

## P1.1 核心问题

Stage2D 观察到：

[
SN > SB
]

同时：

[
AdamState_{SN}\neq AdamState_{SB}.
]

现在问：

> 如果移除 Adam 的 adaptive/history 特性，SB-SN 的性能差异是否仍存在？

## P1.2 实验

固定：

[
\alpha=Sample
]

比较：

| ID | Objective | q | Optimizer |
|---|---|---|---|
| P1-SN-A | Sample | Natural | Adam |
| P1-SB-A | Sample | Balanced | Adam |
| P1-SN-S | Sample | Natural | SGD |
| P1-SB-S | Sample | Balanced | SGD |

SGD 必须经过小规模、预注册的 learning-rate calibration；calibration 只允许依据 validation，不得利用 SB-SN test effect。

4 seeds。

## P1.3 关键 readout

- same-epoch AUC；
- validation-selected AUC；
- trajectory；
- gradient norm；
- update norm；
- domain exposure；
- sample repetition；
- Adam state（Adam arms）；
- parameter distance；
- loss/AUC trajectory。

## P1.4 决策

### Case A
Adam 下 SB-SN 明显负，SGD 下消失/显著缩小：

> Adam/adaptive optimization 是必要条件候选。

进入 P2-A。

### Case B
Adam、SGD 都稳定存在 SB-SN：

> Adam 不是必要条件。

进入 P2-B。

### Case C
两者都不稳定：

> Stage2D effect 可能依赖 M3OE/特定 optimizer hyperparameter，而不是普遍 sampling effect。

进入 P3 boundary tests，不得宣称普遍机制。

---

# 5. Phase P2-A：Adam-specific Causal Decomposition

仅在 P1 支持 Adam necessity 时执行。

## P2-A.1 问题

到底是：

[
m_t
]

还是：

[
v_t
]

还是：

[
m_t/\sqrt{v_t}
]

的历史积累造成 trajectory difference？

## P2-A.2 实验

在 matched parameter checkpoints 上执行 counterfactual one-step readout：

### A：当前 gradient only
比较同一 (	heta_t) 下不同 sampling 产生的 (g_t)。

### B：共享 m
固定相同 (m_t)，只让 (v_t) 根据各自 gradient 更新。

### C：共享 v
固定相同 (v_t)，只让 (m_t) 更新。

### D：共享完整 state
在同一 (	heta_t) 上使用相同 Adam state，只替换当前 gradient。

### E：完整历史
正常 SN/SB。

禁止跨不同 parameter trajectory 直接拼接 state；所有 counterfactual 必须在同一 (	heta_t) 上执行。

## P2-A.3 判断

如果：

- 共享 m 后 effect 消失 → m 是必要中介候选；
- 共享 v 后 effect 消失 → v 是必要中介候选；
- 共享完整 state 后仍存在 → optimizer history 不是充分解释；
- 只有完整历史存在 → persistent adaptive state interaction 最可信。

如果多个条件同时成立，只能写“multiple interacting mediators”。

---

# 6. Phase P2-B：Non-Adam Mechanism —— Exposure / Repetition / Estimator Geometry

仅在 P1 表明 SGD 仍保留 SB-SN 时执行；若 Adam effect 已排除，也必须执行最小版本以确认非 Adam 机制。

## P2-B.1 问题

Balanced 与 Natural 的核心差异可能来自：

1. domain exposure；
2. sample repetition；
3. unique coverage；
4. temporal correlation；
5. gradient geometry。

不能把这些合并成一个“sampling effect”。

## P2-B.2 实验矩阵

构造尽可能保持 objective 与 total update budget 相同的 sampling controls：

| Arm | Domain exposure | Within-domain repetition | q |
|---|---|---|---|
| N | natural | natural | p |
| B | balanced | balanced | 1/3 |
| BX | balanced | exposure-matched | 1/3 |
| RX | natural | repetition-matched | p |
| UX | matched unique coverage | controlled | controlled |

具体实现以可实现且不改变 population objective 的 estimator 为准；若某控制无法严格实现，必须标记为 partial control，而不是声称因果识别。

## P2-B.3 关键指标

- cumulative unique rows；
- unique rows/epoch；
- repeat count distribution；
- per-domain exposure；
- per-task exposure；
- gradient temporal autocorrelation；
- cosine similarity (g_t,g_{t-1})；
- cross-domain gradient cosine；
- covariance spectrum，而非只报告 trace；
- AUC/Loss trajectory。

## P2-B.4 决策

目标是确定：

[
q\rightarrow Exposure
]

是否足以解释性能。

---

# 7. Phase P3：Shared vs Private Parameter Mechanism

这是 MDMT-specific 性的重要验证。

## P3.1 问题

如果 sampling effect 只是普通 single-task stochastic optimization phenomenon，那么 shared/private parameter 对 effect 不应有特别结构。

如果 effect 来自 MDMT shared representation，则应该在：

[
C
]

共享参数和：

[
P
]

私有参数之间表现不同。

## P3.2 实验

使用 Stage2D 的同一 SN/SB 配置：

- C/shared parameters；
- P/private parameters；
- A_arch/router parameters。

同时记录：

[
\Delta\theta_C,quad
\Delta\theta_P,quad
\Delta\theta_{arch}.
]

并记录 domain/task gradient reachability。

## P3.3 关键判断

### 若 C 的 divergence 明显高于 P：

支持 shared-parameter mediation。

### 若 C/P/A_arch 都同样变化：

不能称为 MDMT-specific shared mechanism。

### 若只有某个 task/domain branch 变化：

进入 P4 的 domain-task geometry。

---

# 8. Phase P4：Domain × Task × Exposure 的真正拆解

## P4.1 不允许使用“大小=困难”假设

为每个 domain/task 建立独立 profile：

- sample count；
- positive rate；
- label entropy；
- baseline AUC；
- baseline loss；
- gradient norm；
- gradient covariance；
- exposure；
- repeat count；
- unique coverage。

## P4.2 核心问题

问：

[
\Delta AUC_d
]

到底由：

- domain size；
- baseline difficulty；
- task imbalance；
- exposure；
- gradient geometry；

哪个解释？

## P4.3 统计分析

预注册：

[
\Delta AUC_d
\sim
Size_d+
Difficulty_d+
Exposure_d+
GradientGeometry_d
]

只作为探索性关联分析，不能声称因果。

特别检查：

[
d0\neq d2
]

尽管二者都是小域。

如果 size 与 effect 不一致，正式关闭“小域困难”叙事。

---

# 9. Phase P5：Selection × Trajectory 完整闭环

## P5.1 问题

为什么 Equal-Balanced 的 selected result 与 late common-epoch result 差异这么大？

## P5.2 必须同时报告

对所有四臂：

[
epoch=1,5,10,20,30,40,50
]

报告：

- validation AUC；
- validation loss；
- test AUC；
- test loss；
- domain/task AUC；
- selection probability；
- selected epoch distribution。

## P5.3 关键分析

定义：

[
Effect_{fixed}(t)=M_B(t)-M_N(t)
]

和：

[
Effect_{selected}=M_B(t_B^*)-M_N(t_N^*).
]

分解：

[
Effect_{selected}
=
Effect_{trajectory}
+
Effect_{selection}.
]

## P5.4 终局

如果 selection 只是缓冲 trajectory effect：

> selection 是 modifier，不是 root cause。

如果 selection 本身造成 reversal：

> 必须把 selection uncertainty 作为独立研究问题保留。

---

# 10. Phase P6：Estimator Geometry——彻底排除“variance scalar”解释

不能只继续报告：

[
Trace(Cov(g)).
]

必须增加：

### 一阶
[
E[g]
]

### 二阶
[
Cov(g)
]

### 几何
[
\cos(g_i,g_j)
]

### 时间
[
Corr(g_t,g_{t-k})
]

### 坐标谱
covariance eigenvalue / effective rank。

### 参数块
分别对：

- C；
- P；
- A_arch

进行分析。

## 目标

判断 SB-SN 是否能够被单一：

[
Var(g)
]

解释。

若不能：

[
oxed{
	ext{sampling effect 不是 scalar variance effect}
}
]

这是重要的 negative result。

---

# 11. Phase P7：跨数据/模型边界

如果前面得到稳定机制，必须问：

> 这是不是只在当前 KuaiRand + M3OE 上成立？

至少选择一个已有项目中、成本可控的第二 dataset/domain configuration；如果没有严格可比的数据，则选择第二种 domain imbalance regime，而不是随意换 dataset。

最小矩阵：

- Sample objective；
- Natural vs Balanced；
- 关键 optimizer condition；
- 4 seeds。

不需要完整复刻所有机制实验。

## 判断

### 如果第二环境重复：
支持 generality。

### 如果消失：
得到重要 boundary：

> effect depends on domain imbalance / architecture / objective / optimizer regime。

boundary 本身就是结果。

---

# 12. Phase P8：与 SAF / D3 / AdaTask 的正面区分实验

这一阶段不是为了证明“我们的算法更好”，而是为了回答：

> 当前科学问题是否已经被已有方法覆盖？

必须建立 literature-to-mechanism matrix。

## D3

检查：

- domain complexity；
- domain specificity；
- loss weighting；
- domain balancing。

实验上比较：

[
	ext{objective change}
]

与：

[
	ext{objective-fixed sampling change}.
]

如果 D3 类方法能消除 SB-SN，而不改变 estimator objective，说明已有方法可能覆盖机制干预；否则仍有 gap。

## SAF

重点比较：

- sample weighting；
- adaptive weighting；
- sample-level objective modification。

不能仅比较最终 AUC。

必须检查：

[
J(\theta)
]

是否改变。

如果方法改变 objective，则它不能直接解释 Stage2D 的 objective-equivalent phenomenon。

## AdaTask

重点比较：

- task-wise optimizer state；
- task-specific adaptive learning rate；
- persistent Adam state。

实验必须回答：

> AdaTask 能否在**不改变 objective**的前提下消除 SN/SB trajectory gap？

如果不能，则说明“task-wise Adam”与当前 estimator-induced state problem 并非同一问题。

---

# 13. Phase P9：反事实机制闭环

这是最终决定是否可以形成论文核心结论的阶段。

至少需要构造三个反事实：

## CF1：保持 objective，改变 sampling

已由 Stage2D 完成：

[
SN\rightarrow SB.
]

## CF2：保持 sampling，改变 optimizer

P1 完成。

## CF3：保持 objective + sampling，但阻断候选 mediator

P2 完成。

最终形成：

[
\boxed{
X\rightarrow Z\rightarrow Y
}
]

的必要条件证据。

只有当：

1. X 改变确实改变 Y；
2. 阻断 Z 后 Y effect 显著消失；
3. 其他候选机制不能同样解释；
4. 4 seeds 稳定；
5. 至少一个 boundary/control 成立；

才允许称 Z 为“causal mechanism candidate”。

否则只能称为 association。

---

# 14. Phase P10：是否需要新方法？

**最后才允许进入方法化。**

触发条件：

### GO-METHOD

至少满足：

- 核心现象跨 4 seeds；
- 已排除主要 alternative explanations；
- mechanism 至少有一个必要 mediator；
- SAF/D3/AdaTask 无法直接覆盖；
- 至少一个 external boundary 已测；
- 新方法不是简单复制已有 weighting / balancing / Adam-state separation。

### NO-GO

如果：

- effect 只在当前 M3OE 上；
- 或 SGD/Adam 均不稳定；
- 或 exposure/optimizer/existing method 已完整解释；
- 或新方法只是 SAF/D3/AdaTask 的简单组合；

则停止方法开发，输出机制/边界论文方向或继续寻找更基础问题。

---

# 15. 方法化阶段的硬性创新约束

禁止直接提出：

- “更好的 Balanced Sampling”；
- “更好的 sample reweighting”；
- “domain-aware Adam”；
- “task-wise Adam for domains”；
- “gradient variance reduction”。

除非前面实验已经证明其对应机制是**必要且未被已有工作覆盖**。

如果最终方法产生，必须满足：

[
\text{new method}
\neq
\text{existing weighting}
+
\text{existing balancing}
+
\text{existing Adam modification}.
]

---

# 16. 每个实验的统一统计协议

每个 primary contrast：

[
\Delta_s=M_{B,s}-M_{N,s}
]

必须报告：

- 4 seed values；
- mean；
- SD；
- median；
- sign consistency；
- paired bootstrap；
- common-epoch effect；
- selected effect；
- test readout。

默认 primary criterion：

### Strong
4/4 seeds 同方向 + effect CI 不跨 0。

### Moderate
3/4 同方向且 effect magnitude 稳定。

### Null
方向随机或 CI 明显跨 0。

禁止：

> “均值更高所以有效。”

---

# 17. 所有阶段的提前终止条件

## Stop A：Sampling effect 不复现

若 SN/SB 在重新校验后不稳定：

> 回到数据/实现审计，不继续机制故事。

## Stop B：Adam 被证伪

若 SGD 与 Adam 完全相同：

> 停止 Adam-specific 路线。

## Stop C：Exposure 完全解释

若 matched-exposure control 消除 effect：

> 停止复杂 Adam-state 解释，转向 exposure/generalization。

## Stop D：Shared parameter 被证伪

若 shared/private 完全一致：

> 不再声称 MDMT-specific shared-state mechanism。

## Stop E：已有工作完全覆盖

若 D3/SAF/AdaTask 的现有机制在 objective-equivalent setting 下已经解释全部现象：

> 停止该创新线，明确记录“negative novelty finding”。

## Stop F：现象只属于单一配置

若第二环境不复现：

> 不强行泛化，形成 boundary result。

---

# 18. 最终必须产生的科学输出

无论结果如何，整个 v2 完成时必须得到一份终局报告，明确回答：

### Q1
Objective-equivalent sampling 是否真的改变 MDMT training？

### Q2
是否为 checkpoint-selection artifact？

### Q3
是否依赖 Adam？

### Q4
如果依赖 Adam，是 m、v、adaptive ratio 还是完整 history？

### Q5
如果不依赖 Adam，是 exposure、repetition、coverage 还是 gradient geometry？

### Q6
shared parameters 是否是必要条件？

### Q7
domain size 是否解释 effect？

### Q8
domain difficulty 是否解释 effect？

### Q9
task/domain imbalance 是否解释 effect？

### Q10
validation selection 是否改变结论？

### Q11
这个 phenomenon 是否跨环境成立？

### Q12
SAF/D3/AdaTask 是否已经覆盖？

### Q13
最终是否存在未被现有工作覆盖的 mechanism gap？

### Q14
如果存在，是否值得方法化？

---

# 19. 最终状态机

[
\boxed{
Stage2D
}
]

↓

**P1 Adam necessity**

↓

### Adam-specific
→ P2-A state decomposition

### Adam-independent
→ P2-B exposure/repetition/geometry

↓

P3 shared/private

↓

P4 domain/task heterogeneity

↓

P5 selection

↓

P6 estimator geometry

↓

P7 boundary/generalization

↓

P8 SAF/D3/AdaTask novelty audit

↓

P9 counterfactual causal closure

↓

### 若机制闭环 + novelty gap
→ P10 Method

### 若机制已被已有工作覆盖
→ **STOP：negative novelty conclusion**

### 若机制仍无法识别
→ **STOP：暴露所有新的明确 open problems（数量不限），逐项记录证据、边界和后续最小实验，而不是猜测或强行压缩成单一问题**

---

# 20. 最终研究目标

本计划不要求最终一定得到一个“新算法”。

真正的成功标准是：

[
\boxed{
	extbf{把 Stage2D 的“同目标不同采样却产生不同结果”现象解释到足够深。}
}
]

最终至少必须得到下面三种结果之一：

### Outcome A — 完整机制
[
Sampling
\rightarrow
Mediator
\rightarrow
Trajectory
\rightarrow
Generalization
]

被实验闭环验证。

### Outcome B — 清晰边界
证明现象只在某些：

[
MDMT + objective + optimizer + exposure
]

条件下成立，并明确这些边界。

### Outcome C — 新问题集合
如果多个机制仍不可区分，必须把**所有 unresolved questions 逐项明确化**：每个问题都应有清晰定义、对应证据、当前已排除/未排除的解释、实验已经做到的边界，以及最小可检验的后续实验。**问题数量不设上限，不要求压缩成一个或有限个问题。**

允许最终得到：
- 1 个新的 open problem；
- 多个彼此独立的 open problems；
- 多个相互关联但需要分别验证的 open problems；
- 一组新的研究方向/边界问题。

只要它们都是真实由实验暴露出来的问题，就应全部保留。禁止为了让研究故事“更集中”而人为删除、合并或选择其中一个作为唯一后续问题。

---

## 21. 推荐执行顺序与成本优先级

### 第一层（必须）
1. P1 Adam vs SGD
2. P5 trajectory/selection
3. P3 shared/private

### 第二层（由 P1 分叉）
4. P2-A Adam state counterfactual
或
5. P2-B exposure/repetition controls

### 第三层
6. P4 domain/task
7. P6 estimator geometry

### 第四层
8. P7 boundary
9. P8 SAF/D3/AdaTask

### 最终
10. P9 causal closure
11. P10 method GO/NO-GO

---

## 22. 与原研究方向的关系

本方案**不改变原研究主线**，而是将其推进一层：

[
\text{原问题：}
Equal\ vs\ Sample
]

↓

[
\text{已解决：}
objective/loss\ aggregation
]

↓

[
\text{新发现：}
same\ objective\neq same\ training
]

↓

[
\boxed{
\text{当前主问题：objective-equivalent estimator 的 MDMT optimization dynamics}
}
]

因此 v2 不应再回到“Equal 是否优于 Sample”“小域是否困难”“Adam history 是否能救 Equal”等已经有明确结论或已关闭的问题。

它的唯一任务是：

> **把 sampling estimator → exposure/gradient sequence → optimizer state → shared trajectory → selection/generalization 这条链拆开，并一直做到能够形成最终结论；如果拆不开，就把拆不开的原因和最小剩余问题明确暴露出来。**
