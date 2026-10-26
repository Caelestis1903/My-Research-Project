# V6：已完成的几何与时间方向读出（干预结论仍待后续）

当前结果支持 q 改变有限 batch 的梯度分布；尚未证明几何量、共享参数或 Adam 状态是泛化差异的必要中介。全部数值可追溯到 `analysis/V1_full_coordinate_geometry.csv`、`V1_paired_geometry.csv`、`analysis/V1_task_verified/` 与 `analysis/V6_geometry_synthesis_initial50/`。后者的 320 条 run×stage×block 记录经过生产者 SHA 校验，主 agent 独立复算四个原始案例并核对全部观测数量；`independent_review=false`。

固定参数的 16 个 checkpoint 均使用每个 q 的 512 个独立 batch、完整 1,614,254 个坐标。已完成的 objective 数值身份与均值审计见 V1，不能把 Monte Carlo 不拒绝当作数学等价的证明；代数权重身份另行成立。SB/SN covariance trace 比在全部 16 个点为 2.3155–2.7522。对 epoch10 的四种子，Adam 为 2.4155–2.5569，SGD 为 2.3155–2.4665。此配置中的噪声尺度变化是 **STRONG** 的数值观察；它是否导致泛化差异仍为 **WEAK** 因果证据。

谱形状有重要反例。SGD epoch10 的 effective-rank 比四种子均低于 1（0.7681–0.8987），anisotropy 比均高于 1（1.0290–1.1376），重复性证据 **MEDIUM**。Adam 的 rank 比为 0.8878–1.0061，seed2022 略高于 1，不能声称所有 optimizer、seed、阶段都发生秩下降。经验协方差的秩最多为 511；这是 512 个完整向量产生的有限样本谱，不是模型全部潜在噪声方向的精确总体谱。

固定 θ 的中心化 IID 噪声方向与移动 θ 下的原始梯度方向必须分开。epoch10、ALL 块、两 q 与四种子合计，固定 θ 的 lag1/2/5 平均 cosine 分别为 Adam −0.000484/+0.003080/−0.002916，SGD −0.005441/+0.002653/+0.000987。这些是按既定 IID 抽样顺序计算的中心化向量 cosine，不是标量 Pearson 自相关，也不是生产训练序列。

校准后的真实训练在 epoch10 的 C 块前六个 batch 中，轮内相邻梯度 cosine 四种子均负。SGD SN 范围 −0.98490..−0.96099，SB 为 −0.97514..−0.93591；Adam SN 为 −0.61023..−0.17939，SB 为 −0.37913..−0.04633。即使无 Adam 状态的 SGD 也可出现很强的相邻方向交替。因此负相邻 cosine 本身不能指认 Adam memory，也不能由固定 θ 的 IID 噪声相关性直接解释。学习阶段、步长对应的轨迹变化与 batch 采样均是竞争解释；当前读出没有单独识别它们。注意 V1 固定 θ 来自旧的固定参考 LR（Adam .01、SGD .1），移动轨迹来自校准 LR（.003、.3），不能将两者视为同一参数点的配对反事实。

时间遥测边界 Q35 已关闭为 **RESOLVED / STRONG**：训练器每轮仅记录前六个梯度，未记录其余 step，却跨轮保留观测历史。在 KuaiRand 的 49 step/epoch 下，轮首原始 lag1 记录实际跨越 44 个 optimizer step。新读出只使用 batch_index≥lag 的轮内配对，lag1/2/5 分别每个 stage/block 仅有 5/4/1 对。原始值完整保留，不改动训练；不能把这些短片段推广为整轮时间相关性。四个独立原始案例的 step 差核验均为 44，过滤后的 lag 均值复算误差为 0。

矩状态范数采用 Q34 已验证的 `corrected_m_norm` 和 `v_norm`，保留原始混合范数列供追溯。共有 80,000 条已归档遥测被恢复，最大相对一阶矩修正为 22.51%，不能称为可忽略。真实优化器状态的八个块 oracle 最大误差为 4.34e−19；优化器更新和性能并未改变。m/v 范数、update/gradient 比与 block 距离都只是描述量，不证明状态或该 block 的必要性。

跨任务完整梯度读出已在两 optimizer×四种子的 epoch10 θ 验证两任务半和身份。按 train-p 加权的两任务梯度 cosine 全部为正：SGD 0.1111–0.2880、Adam 0.2738–0.4262；个别域或参数块仍有负值。因此“所有任务普遍冲突”不被数据支持，也不能把正总体方向当作不存在局部竞争的证明。P 内既有 domain-private 也有 task-shared 路径；R gates/towers 已为 domain-task-private，不能仅凭 R 位移较大提出虚假的 shared-R privatization。

这批几何数据不能识别 domain size 为主要解释变量。三个物理域同时具有不同标签率、分布、目标权重、覆盖和基线 AUC；种子重复不增加独立域数量。当前校准 Adam 的 pre-window 各域均下降，小域与大域之间没有稳定正负交换；MovieLens 最小域在早期四种子改善却不延续到固定末轮，进一步表明必须保留模型、环境与阶段。完整尺寸/难度敏感性结论仍需 V8 既定实验，几何观察不替代它。

最有区分力的后续是已冻结的 V5 同 batch multiset 顺序干预及 V7 同 θ/同 optimizer state 的局部坐标替换，再依据 selection 触发规则决定是否需要参数容量匹配的全训练干预。只有 V11 实际四种子 primary interaction 触发，才允许进行新的 Adam 状态分解。V6 的观测部分完成；V5/V7/V11 及最终因果闭环仍在执行，不能据此结束全任务。


补充可识别箭头的代数边界：在固定θ、逐行可加且WR的B个独立抽样下，设单样本图梯度为g，目标均值G*=Σp_d E_d[g]，则

$$Cov(\hat g_q)=B^{-1}\left[\sum_d\frac{p_d^2}{q_d}E_d[gg^T]-G^*G^{*T}\right].$$

这里使用的是已建立计算图的导数，包含Q28的routing detach边界。该式说明同一均值不要求相同二阶矩；不是新提出的importance-sampling理论。当前无动量SGD映射为Δθ=−η(ĝ_q+λθ)，固定θ时λθ为常量，故每个坐标块的更新协方差恰为η²Cov(ĝ_q)。因此q→估计器分布以及固定θ下估计器→单步SGD更新有代数依据；沿移动θ的多步轨迹和泛化性能不由这个等式独立识别。不得把固定θ的恒等式当作geometry→AUC的必要性证明，也不能由trace增大单独排除谱形状、方向、顺序或步长/阶段解释。

训练使用两臂相同的weight_decay=1e−5；报告的empirical train objective是数据目标J，优化器还包含相同的λθ项。固定θ下该确定项不改变更新协方差，但实用数据loss平台不等同于证明正则化目标的全局最优。
