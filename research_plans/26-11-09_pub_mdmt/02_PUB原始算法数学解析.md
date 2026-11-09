# PUB原始算法数学解析

论文：Jun Yuan, Guohao Cai, Zhenhua Dong, A Parameter Update Balancing Algorithm for Multi-task Ranking Models in Recommendation Systems, ICDM 2024, DOI: 10.1109/ICDM59182.2024.00117；arXiv:2410.05806。

## 算法流程
每步分别求任务梯度g_i=∇θL_i；使用同一份更新前优化器状态计算候选更新u_i=f(g_i;m_t,v_t)；构造D=[u_1,...,u_n]并求alpha；以sum alpha_i u_i更新共享参数；再以相同alpha组合原始梯度更新Adam一阶、二阶状态。

论文Adam候选式：
u_i=[β1 m+(1−β1)g_i]/[sqrt(β2 v+(1−β2)g_i²)+eps]。
论文伪代码未显式写bias correction；迁移必须逐项对齐论文公式与PyTorch Adam实际bias correction，不能重复校正。

## 效用与权重
任务效用为u_iᵀΔθ。PUB求解：
max_{Δθ∈Bε} Σ_i log(Δθᵀu_i)。
令Δθ=D alpha，正权重满足G alpha=1/alpha，其中G=DᵀD，alpha_i>0。倒数为逐元素倒数。原始PUB没有softmax，也没有Σalpha=1约束；alpha不是概率或归一化任务占比。

论文使用CCP序列凸近似：φ_i(alpha)=log(alpha_i)+log(u_iᵀD alpha)，迭代线性化凹项。论文报告最多200次迭代，实践中每10或100 step更新alpha；每100 step相对每10 step出现性能下降。

## 理论前提与局限
论文推导依赖：单任务梯度得到的候选更新对该任务合理；非Pareto-stationary时任务更新线性独立；候选计算时优化器moment作为常量；更新内积可作为每步任务效用。
局限：Adam预条件与历史状态使候选更新不等于独立训练轨迹最优更新；正内积不等于真实BCE/AUC改善；同一alpha更新m/v是作者降低复杂度的经验策略，论文承认缺少理论分析；log-utility不等价于六格Macro AUC最大化。

## Gram含义
G_ij=u_iᵀu_j。对角线为候选更新平方范数，非对角线为任务候选更新对齐程度。它不是原始梯度Gram，也不是任务损失相关矩阵。可分块累积小型Gram，无需长期保存完整D。

## 作者代码审计状态
公开仓库：https://github.com/yjdy/Multitask-Optimization-Recommendation-Library，MIT。当前规划读取到论文正文和仓库入口，但未固定某一commit的PUB源码blob，故不声称已完成代码级差异审计。Codex必须先记录commit SHA并核查：状态初始化、step/bias correction、候选式、alpha solver约束与回退、专有参数处理、weight decay、梯度裁剪、更新顺序、alpha频率、moment是否使用同alpha，以及任务顺序是否污染状态。论文可核验V-A至V-C、Algorithm 1、VI-C。论文AliExpress收益不能当成本项目预期收益。
