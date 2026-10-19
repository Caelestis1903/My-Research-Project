# P6 固定目标域 proposal 与经典 IS 边界

令 p_d 为训练人口份额，一条记录的实际训练梯度 g_i 是两任务 BCE 的均值。
先取 d~q，再从域内均匀抽记录，估计量为 (p_d/q_d)g_i。只要 q_d>0，

    E[g_hat] = sum_d p_d E[g|d] = mu
    tr Cov(g_hat) = sum_d (p_d²/q_d) E[||g||²|d] − ||mu||²

对固定的六个完整预测风险单位方向 h_c，同理

    Var(h_c·g_hat) = sum_d (p_d²/q_d) E[(h_c·g)²|d] − (h_c·mu)²

其非负加权和的最优自由 proposal 为 q_d ∝ p_d sqrt(E[gᵀMg|d])。
M=I 给范数 IS，M=mean_c h_c h_cᵀ 给本轮方向 IS。这是普通广义 IS，
不是新公式。2% 自然 proposal 混合用于保留覆盖；所有臂一致约束。

每种子设计/留出各 48 行/域，共 288 个单记录梯度。均值、二阶矩和投影
均在明确的有限留出人口上计算，方差不是总体 covariance 的无误差估计。
各域均匀样本数并没有把部署 p 改成 1/3；目标 p 始终来自完整训练人口。
B=40960 的除数只展示 IID 平均的方差缩放，不证明同训练预算泛化等价。

比较：自然 p、范数 IS、EMAIS 风格的 logit-residual 代理、方向 IS、以及
自然/范数/两个任务 residual proposal 的 trace 最优凸混合。后两种代理/
混合不等于忠实完整在线 EMAIS/OMIS。MIS 混合权重只使用设计样本，
留出一次评估；不按留出结果调 sampler。

若每条记录同时有两标签且执行成本相同、无域特有标签缺失和路径预算，
这一问题就是已有向量 IS。只有真实标签/共享计算约束改变可实现 proposal
集合，才有新的 MDMT 理论对象；本轮数据没有提供这种约束。
