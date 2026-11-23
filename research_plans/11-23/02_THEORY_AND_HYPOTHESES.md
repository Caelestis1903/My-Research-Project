# 02｜Theory, Estimands & Hypotheses

## 1. 变量定义

令模型参数分解为：
[
\theta=(\theta_U,\theta_{P_c},\theta_{P_a}),
]
其中 U 为共享/上游表示与共享专家；(P_c) 为 cell-specific Gate/Tower；(P_a) 为 domain/task axis experts。完整专有路径 (P=P_c\cup P_a)。

对 cell (c=(d,t))，预测 logit 为：
[
z_c(x;\theta_U,\theta_P).
]

对一个固定自然训练 batch (B)，普通联合梯度：
[
g_U=\nabla_{\theta_U} L_B(\theta),\qquad
g_P=\nabla_{\theta_P} L_B(\theta).
]

本轮不改变自然目标：
[
L_B=\frac{1}{2|B|}\sum_{i\in B}\sum_{t=0}^{1}
\mathrm{BCE}(p_{i,t},y_{i,t}).
]
域任务 cell 指标仍按真实 domain mask 统计，不将六格训练目标改成等域重加权。

## 2. 共享更新后的响应

在相同 pre-state 上，对 U 执行一次 Adam 更新，得到：
[
\theta_U^+=\mathrm{AdamStep}_U(\theta_U,g_U,m_U,v_U).
]

固定专有参数，定义 cell 的共享变化响应：
[
\Delta z_c^U(x)=z_c(x;\theta_U^+,\theta_P)-z_c(x;\theta_U,\theta_P).
]

对固定正负 pair (q=(x^+,x^-))，margin 响应：
[
\Delta m_{c,q}^U=
[z_c(x^+;\theta_U^+,\theta_P)-z_c(x^-;\theta_U^+,\theta_P)]
-
[z_c(x^+;\theta_U,\theta_P)-z_c(x^-;\theta_U,\theta_P)].
]

响应异质性描述为六格的 (Delta AUC_c)、pair flip、margin 变化和 logit shift 分布；不得只用梯度余弦、参数范数或 loss 变化代替排序效用。

## 3. 专有路径适配效应

共享更新后，固定 U，只更新选定 P scope：
[
\theta_P^{++}=\mathrm{Adapt}_P(
\theta_U^+,\theta_P;B_{next},K).
]

定义适配恢复：
[
R_c^{post}=AUC_c(\theta_U^+,\theta_P^{++})
-AUC_c(\theta_U^+,\theta_P).
]

但 (R_c^{post}>0) 本身不能证明“共享更新后的适配”具有特殊价值，因为额外专有更新可能在任何时刻都有用。必须与相同更新预算的反向顺序比较：
[
R_c^{order}=AUC_c(U\rightarrow P)-AUC_c(P\rightarrow U),
]
以及梯度累积联合控制：
[
R_c^{joint}=AUC_c(U\rightarrow P)-AUC_c(JOINT\_ACC).
]

核心 estimand 是在同一初始化、同一自然训练数据、同一参数组更新次数和相同反向/前向预算下，U→P 相对于 P→U 与 JOINT-ACC 的配对排序增量。

## 4. 假设与可证伪条件

### H1：共享更新响应存在域任务异质性

预测：相同 U 更新对六格的 AUC/pair margin 影响不一致；局部改善和局部损害可能并存。

否定/弱化条件：六格响应差异接近数值噪声，且跨 seed/阶段没有稳定结构。即使 H1 不成立，也不自动否定 M3oE 其他共享机制。

### H2：专有路径在共享更新后存在可利用的适配收益

预测：U→P 的适配后验证排序优于 U-only，并且优于相同 scope 的 P→U 和 JOINT-ACC；该方向在独立 V_B 上保留。

竞争解释：
- 普通额外更新带来的收益；
- 数据 batch A/B 难度或顺序差异；
- 参数组更新次数不同；
- 适配对同一训练 batch 过拟合；
- 只是 LogLoss 下降，没有排序改善。

通过对照：AB/BA 双顺序、JOINT-ACC、JOINT-2、V_A/V_B、固定 pair transitions。

### H3：协调更新可形成独立训练算法增量

预测：完整 50 epoch U→P 在 Validation selected Macro AUC 上超过 J_ORIG、J_ACC 和 P→U；六格代价满足保护线；最终 Test 增益达到正式门槛。

若 H2 成立而 H3 不成立，结论是局部适配效应存在，但当前固定 block schedule 不能转化为可部署算法。不得把 H2 写成算法成功。

## 5. 机制层级

必须分开报告：
1. **经验现象**：cell 输出、AUC、LogLoss、pair flip 变化。
2. **受控因果效应**：同状态 fork 中只改变更新组/更新顺序的差异。
3. **机制解释**：哪一组参数/预测路径产生变化，是否存在共享—专有交互。
4. **泛化**：两个数据集、五 seed、独立 V_B 与最终 Test 的适用范围。

参数变化、梯度方向或单步 loss 不得单独作为机制结论。
