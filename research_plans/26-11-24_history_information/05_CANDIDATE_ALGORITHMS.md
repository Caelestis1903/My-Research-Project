# 05｜候选算法的数学定义与实现规范

日期：2026-11-24。状态：**待实现的冻结算法规格；本次交付不启动训练。**

本轮直接检验两个尚未在仓库中完整测量的操作：**C1 调节 BCE 与 PAIR 在共同二阶矩中的交叉项；C2 保存 PAIR 梯度在产生时的预条件方向。** 两者继承 A21 的监督、源一阶历史、S_DENSE 作用域及自然训练协议，分别改变一个运算环节。H0 是原 A21 的精确入口，构成两个候选的共同父方法。既有实现的依据为 [S1–S5]；已完成机制及失败路线的边界见本文件第 12 节。

诊断和完整训练分别见 [04_MECHANISM_DIAGNOSTICS.md](04_MECHANISM_DIAGNOSTICS.md)、[06_EXPERIMENT_MATRIX.md](06_EXPERIMENT_MATRIX.md)；状态事务与工程验收见 [07_IMPLEMENTATION_SPEC.md](07_IMPLEMENTATION_SPEC.md)；晋级及组合触发见 [08_DECISION_RULES.md](08_DECISION_RULES.md)。以下数学定义与这些文件共同冻结，执行时不得自行增加归一化、梯度裁剪、warmup、学习率调度或辅助损失重加权。

## 1. 臂、参数与作用范围

| 臂 | 完整定义 | 与直接父方法的改变量 |
|---|---|---|
| B0 | 原 clean M3oE fixed1 + 原生 Adam，旧 A00 | 自然 BCE 基线 |
| B1 | 相同静态条件 PAIR + 原生 Adam，旧 A10/P00 | 相对 B0 增加既有 PAIR 监督 |
| H0 | 原 A21；静态 PAIR、自然通道 .9、辅助事件历史 .99、mixed 二阶矩 | 相对 B1 改变一阶历史组织 |
| C1 | H0 分子，$\zeta\in\{0.5,1\},\alpha=0$ 的共同二阶矩 | 跨来源二阶耦合程度 |
| C2 | H0 的 mixed 二阶矩，$\zeta=0,\alpha\in\{0.5,1\}$ | 辅助历史与预条件的运算顺序 |
| 条件组合 | 使用已冻结的 $\zeta_\star,\alpha_\star$ 代入第 5 节统一式 | 两个已晋级操作的组合 |
| M | 在自身轨迹上，H0 方向匹配候选的局部整组步长 | 简单组尺度解释 |
| S | 候选其余操作不变，辅助事件核改为 $\beta_P=.9$ | 长辅助历史的增量 |
| P | C1 对应 AdaTask-inspired 分源分母；C2 对应 LaProp-inspired mixed 预条件历史 | 最接近已有运算的适配参照 |

M/S/P 均绑定清单中明确的父候选及其已冻结参数；表中括号式引用只是父子关系，不生成额外臂或网格。主矩阵的零点 $(\zeta,\alpha)=(0,0)$ 直接复用 H0；默认新增科学训练只有 C1 两配置与 C2 两配置。

**旁路按完整背景解析。** `no_PAIR_parent(backbone, base_objective, scope)`只移除PAIR和新增历史，不改基础目标：主矩阵返回B0，Gaussian返回G_B0（保留GA31-R3），HyperGate返回HG_B0（保留完整native objective）。`.99`候选的α/ζ共同零点返回同骨干、同基础目标、同scope的历史父入口：主矩阵H0、N范围H0_N、Gaussian G_H0、HyperGate HG_H0。零点规则只适用于这些候选及其明确的父历史身份，不把B1或P按两个数值零强制变成H0；S的βP=.9也不能被旁路改回.99。G_B0的λPAIR=0不表示λaug=0。

固定 $\eta=.01,\lambda=.05,\beta_B=.9,\beta_P=.99,\beta_2=.999,\epsilon=10^{-8}$。$\lambda$ 专指 PAIR 权重，$\omega$ 专指 coupled weight decay：MovieLens 为 $10^{-4}$，KuaiRand 为 $10^{-5}$；域下标仍用 d。S 只将 $\beta_P$ 改为 .9，不添加通用 beta 网格。[S3–S5]

基础设置为两数据集自然分布、50 epochs、batch 40960、保留自然尾批、FP32、AMP/TF32 关闭；正式 seed 固定为 2022–2026。原生 Adam 的 `foreach=False, fused=False, amsgrad=False` 与原参数组身份一并继承。[S3,S5]

**主作用域 $J=\mathrm{S\_DENSE}$** 精确匹配 `shared_name`：`shared_weight`、`shared_bias`、`skip_conn.*`、`star_mlp.*`、`expert.*`。EXPERT 内即使名称含 `domain_specific`，仍按原注册属于共享专家；EMBED 不在 J。fixed1 融合常数不新增可训练 FUSION 参数。[S1,S5]

组范数控制固定为两个互斥组：STAR_SKIP=`shared_weight/shared_bias/skip_conn.*/star_mlp.*`；EXPERT=`expert.*`。J 外每参数仍由原生 Adam 处理**完整 BCE+PAIR 目标**，每个参数每个有效 batch 只提交一次更新。

## 2. 时钟、梯度与支持语义

记 $t$ 为有效 batch/优化事务的全局编号，$\theta_{p,t-1}$ 为参数 p 的更新前状态。$e_t=1$ 表示当前 batch 存在可采样 PAIR，$e_t=0$ 表示无活动边界。全局 PAIR 计数 $k_t=k_{t-1}+e_t$ 用于采样与来源登记。

每参数原生时钟为 $\tau_{p,t}=\tau_{p,t-1}+\mathbf1[\mathrm{native\_grad}_{p,t}\ne\mathrm{None}]$。原 S_DENSE 每步必须有非 None 梯度，缺失即报实现错误，因此其中 $\tau_p=t$、辅助事件计数等于全局 k；仍保留私有时钟，不用 batch 下标代替状态中的 step。[S1,S2]

为条件 N 作用域定义每参数辅助计数：

$$
k_{p,t}=k_{p,t-1}+e_t\mathbf1[\mathrm{native\_grad}_{p,t}\ne\mathrm{None}].
$$

这只扩展局部支持语义；S_DENSE 内始终 $k_p=k$，H0 继续使用原实现的全局 `pair_clock`。B1 源 adapter 的 `pair_clock` 字段不承担事件计数，保留其原值；B1 的 k 从只读事件记录或 selected 次数求和获得，不改写旧 optimizer 状态。后文对一般 p 写 $\tau_p,k_p$，在主实验中可直接按此等式对应源码。

| 当前支持 | 参数、WD、$m_B,V,\tau_p$ | $Q,U,k_p$ | 是否注入辅助历史 |
|---|---|---|---|
| native grad=None | 全部跳过，保持原生 None | 全部保持 | 否 |
| native grad 为零张量，e=0 | 正常原生步；WD 加一次 | 保持 | 否 |
| native grad 非 None，e=1，raw r 为零 | 正常更新 | 正常衰减、加零，$k_p+1$ | 是，按更新后的历史 |
| native grad 非 None，e=1，raw r=None | 仅辅助通道按同形零处理 | 同上一行 | 是 |
| native grad 非 None，e=0 | 正常更新 | 不衰减、不增计数 | 否，即使过去历史非零 |

原 S_DENSE 的 raw r 由 `autograd.grad(..., allow_unused=False)` 获得，None 视为错误。`r=None→辅助零` 只适用于显式登记的通用/N 支持适配；它不能将 native None 变成零。辅助事件由是否采到合法 PAIR 决定，不由梯度是否非零、范数大小或 loss 大小决定。

对 native grad 非 None 的 p，令 $r_{p,t}=\nabla_p R_{s_t}$，$a_t=w_{s_t}/q_{s_t}$，定义

$$
h_{p,t}=e_ta_tr_{p,t},\qquad b_{p,t}=g^{\rm BCE}_{p,t}+\omega\theta_{p,t-1},\qquad g^{\rm mix}_{p,t}=b_{p,t}+\lambda h_{p,t}.
$$

所有平方、乘法、除法均逐坐标。$r,h,Q,U$ 不含额外 WD；coupled WD 只在 b 中加入一次，并按 b 所参与的矩递推影响更新。这里 b 已含 WD，所以 C1 所调节的 $2\lambda b h$ 包含 BCE–PAIR 与 WD–PAIR 两部分交叉项。

**保留 A21 数值路径。** 源码先得到 `mixed = p.grad.add(p, alpha=wd)`，再在 PAIR 步执行 `real = mixed - (lambda * ratio) * r[p]`；数学上的 real=b。实现沿用这个减法顺序，不为 b 另跑一次 forward/backward，也不把 `mixed` 重建为另一次 `b+lambda*h`。H0 的 Q 沿用 `Q.mul_(.99).add_(r, alpha=.01*ratio)`，零点直接调用原入口。[S1]

## 3. 完整 PAIR 计算图、采样和 RNG

自然目标严格为 $L_B=(2B)^{-1}\sum_{i=1}^B\sum_{u=0}^1\mathrm{BCE}(p_{iu},y_{iu})$，即原 `binary_cross_entropy(..., reduction='mean')`；不改成六格各自求均值再平均。$y$ 与概率/联合 logit 的 shape 都是 `[B,2]`，domain 是原 x 的最后一行 `x[-1]`，输入布局沿用源 loader。[S2,S5]

一次原模型 forward 同时获得概率和 tower hooks 的原始 logit。tower 索引为 $3u+d$，日志 cell 索引为 $2d+u$，两者不能混用。保留原 gate 输入的 `embedding.detach()`；PAIR loss 沿原完整模型图反传到所有可达参数。S_DENSE 仅限定历史更新位置，不截断 PAIR 对 Embedding、tower 或其他参数的正常梯度。[S2,S5]

联合标签 $j=2y_0+y_1$；条件 $s=(d,u,a,b)$ 满足同域、任务 u 在标签 a 中为正、在 b 中为负。原 `CONDITIONS` 的固定枚举顺序共 24 条，当前 batch 两端都非空才活动。对活动集 $A_t$：

$$
w_s=\frac{n_d}{2B}\frac{n_{da}}{n^+_{du}}\frac{n_{db}}{n^-_{du}},\quad Z_t=\sum_{s\in A_t}w_s\le1,\quad q_s=\frac{.25}{|A_t|}+.75\frac{w_s}{Z_t}.
$$

这是 `dynamic=False` 时原采样式的代数形式。代码继续调用原 `boundaries/sample`，保留其三项计算和浮点顺序；w 保持原目标质量，不额外归一化为和 1。静态采样应满足 $\sum q=1,q>0,w/q\le4$。原 PCOLL 默认 dynamic=True，H0 必须由 `history_config('A21',...)` 显式覆盖成 False。[S2–S4]

每个 PAIR 事件只抽一个条件，从原 batch 两端分别有放回抽 2048 行，构成 2048 对：

$$
R_s=\frac1{2048}\sum_j\operatorname{softplus}[-(z^+_{j,u}-z^-_{j,u})],\qquad L=L_B+\lambda a_tR_s.
$$

先在同一图上求 J 内 raw r，使用 `retain_graph=True`，保存 detached 张量；再由完整 L 的一次 `.backward()` 生成真正 native grads。B1 不需要额外 r sweep。新增优化器数学在 `no_grad` 中完成；不让 Q/U、shadow 或 D 建立可微历史图。[S2]

PAIR 随机流严格由 `dataset|seed|epoch|global_step|stream` 的 SHA256 前 8 bytes、小端整数构造 NumPy `default_rng`；stream 为 `condition` 与 `pair_indices`。键不含 arm、$\zeta$、$\alpha$ 或 $\lambda$。同数据/seed/批序的静态臂具有相同条件和 pair 索引，即使参数轨迹已分叉；记录 `pair_index_hash`、selected、w/q 与原 row IDs。[S2]

无活动条件时直接只用 $L_B$，继续一次正常优化事务。$\lambda=0$ 在最外层跳转当前背景的canonical no-PAIR parent（主矩阵为B0）：不建 PAIR hooks、不调用 sampler、raw r、额外历史状态或其 RNG；保留Gaussian/HG背景本来需要的视图或原生目标。loader、Python/NumPy/CPU/CUDA RNG 与原模型随机路径均保持原身份。

## 4. B0、B1 与 H0 的精确定义

### 4.1 原生 Adam 基线

B0 对完整模型使用原 clean 入口，原生梯度为 $g^{\rm BCE}$。B1 使用原 `v3.config('P00', dataset, .05)`、`make_optimizer` 与原完整 `train_step`，静态 PAIR 的完整 L 对全部参数使用原生 Adam；每参数非 None 时执行

$$
m_t=.9m_{t-1}+.1g^{\rm mix}_t,\quad V_t=.999V_{t-1}+.001(g^{\rm mix}_t)^2,
\quad\Delta\theta_t=-\eta\frac{m_t/(1-.9^{\tau_p})}{\sqrt{V_t/(1-.999^{\tau_p})}+\epsilon}.
$$

B0 中把 $g^{\rm mix}$ 换为 b。实际仍调用原生 optimizer，以保留原参数组、None/zero、私有 step 和浮点路径；公式用于审查，不另写一个“近似 Adam”替代它。[S1,S3]

### 4.2 H0=A21

H0 精确调用 [S4] 的 `history_config('A21', dataset, .05)` 与 `make_history_optimizer`，使用其原完整目标与 EMA 入口；科学标签 H0 写在外层任务元数据，不改写原 optimizer config 后再宣称源状态相等。S_DENSE 内冷启动 $m_B=V_0=Q=0$，所有时钟为 0：

$$
m_{B,t}=.9m_{B,t-1}+.1b_t,\qquad V_{0,t}=.999V_{0,t-1}+.001(g^{\rm mix}_t)^2.
$$

当 $e_t=1$ 且 native grad 非 None 时，$Q\leftarrow.99Q+.01h_t$，并增加 $k_p$；定义 $\widehat m_B=m_B/(1-.9^{\tau_p})$、$\widehat V_0=V_0/(1-.999^{\tau_p})$、$\widehat Q=Q/(1-.99^{k_p})$。$k_p=0$ 时不计算 Q 的除法。

$$
D_{0,t}=(\sqrt{\widehat V_{0,t}}+\epsilon)^{-1},\qquad
u^{H0}_t=D_{0,t}[\widehat m_{B,t}+e_t\lambda\widehat Q_t],\qquad \Delta\theta_t=-\eta u^{H0}_t.
$$

源码在 Q 更新后更新 m/V，使用当前 bias-corrected V；分子先相加，再调用一次 `addcdiv_`。保持该顺序及 `eps` 在平方根外的位置。J 外原 Adam 在同一步处理完整混合目标。[S1,S4]

原 `HistoryOptimizer` 还维护六格 M9 和六格 Q 的被动 shadow，来源 cell=`2*d+task`；它们不反馈参数。H0 源等价验收保留这些 shadow、`selected`、`pair_clock`、参数清单和完整序列化字段；对源 `Q=sum_c Q_c` 的重构沿用原容差。删除这些状态后只能称活跃更新等价，不能称原完整 StatePacket 相同。[S4]

## 5. C1/C2 的统一递推

所有新态从零开始。$m_B,Q$ 与 H0 相同；只有 S 将辅助 beta 改为 .9。对每个 native 非 None 参数，以同一 $\tau_p$ 更新一个共同二阶态：

$$
V_{\zeta,t}=\beta_2V_{\zeta,t-1}+(1-\beta_2)
\left[(1-\zeta)(g^{\rm mix}_t)^2+\zeta(b_t^2+\lambda^2h_t^2)\right],
\quad D_{\zeta,t}=\left[\sqrt{\frac{V_{\zeta,t}}{1-\beta_2^{\tau_p}}}+\epsilon\right]^{-1}.
$$

$\zeta=0$ 必须直接执行原 `mixed,mixed` 的 `addcmul_` 路径；$\zeta>0$ 用上述非负凸组合计算，不用“mixed² 减交叉项”的消减实现。不改变 $\beta_2$、时钟、bias correction 或 epsilon。

在完成本步 V 和 D 后，活动 PAIR 参数更新

$$
U\leftarrow\beta_PU+(1-\beta_P)D_{\zeta,t}h_t,\qquad
\widehat U=U/(1-\beta_P^{k_p}).
$$

U 与 Q 共用该参数的同一个辅助事件时钟；本步 $k_p$ 只加一次。U 中保存的是预条件后的辅助方向，不包含 $\lambda$ 或 lr；读取后不能再乘 D。$e_t=0$ 时 U/Q 均保持、均不注入。最终方向为

$$
\boxed{u^{\zeta,\alpha}_t=D_{\zeta,t}\widehat m_{B,t}
+e_t\lambda\left[(1-\alpha)D_{\zeta,t}\widehat Q_t+\alpha\widehat U_t\right]},\qquad
\Delta\theta_t=-\eta u^{\zeta,\alpha}_t.
$$

仅在需要 U 的分支计算 U；$\alpha=0$ 的活跃参数更新按 H0 的“先合并分子、再除共同分母”执行。本轮所有 C2 配置包括 $\alpha=1$ 都保留 Q，供同轨迹重构与统一序列化使用。

实现固定使用源系数字面量 `.9/.1`、`.99/.01`、`.999/.001`；数学式中的 $1-\beta$ 表示对应已冻结系数，不另外用浮点相减重算后改变继承路径。Q 保留 raw r 与 `alpha=coefficient*ratio` 的原加法顺序；新运算中的 h 只物化一次，U 使用当步已完成的 D 与该 h。

### 5.1 C1：来源二阶耦合调节

C1 固定 $\alpha=0$，逐配置取 $\zeta=.5,1$。它继续使用一个共同分母，目标是识别跨来源二阶项对历史更新价值的影响。

在同一被观测参数轨迹、同一支持和从零初始化的 shadow 上，令 $V_{\rm sep}$ 以 $b^2+\lambda^2h^2$ 递推，则

$$
V_0-V_{\rm sep}=\operatorname{EMA}_{.999}[2\lambda b\odot h],\qquad
V_\zeta=(1-\zeta)V_0+\zeta V_{\rm sep}.
$$

该式在 FP32 中按数值容差验收。交叉项可正可负：减去正项降低分母，减去负项增大分母；因此记录有符号分布与实际方向，不把 C1 预先解释为“清除有害冲突”。跨独立训练轨迹的 V 差不满足这个固定轨迹分解。

### 5.2 C2：预条件辅助历史

C2 固定 $\zeta=0$，取 $\alpha=.5,1$，自然通道继续采用当前 $D_t\widehat m_B$。设某参数已发生 k 个有效辅助事件，事件步为 $t_1,\ldots,t_k$，$\kappa_{k,j}=(1-\beta_P)\beta_P^{k-j}/(1-\beta_P^k)$。在相同轨迹、相同 D/m/Q shadows 且当前为 PAIR 事件时，

$$
u^{C2}_t-u^{H0}_t=\alpha\lambda\sum_{j=1}^k\kappa_{k,j}(D_{t_j}-D_t)h_{t_j}.
$$

恒定 D 时两者相等；首个辅助事件也相等。该恒等式指出 C2 改变的是旧梯度在当前坐标尺度下的重新解释。它为保留产生时的有效更新提供直接动机；完整训练是否提高排序由 C2−H0 的 full50 结果决定。

LaProp 已给出“先当前预条件、再积累动量”的一般操作 [L1,L2]。C2 的研究问题是：在 MDMT 的 BCE/PAIR 两种监督中，对 PAIR 使用事件历史，同时保留 BCE 当前预条件历史，是否提供独立价值。LaProp 提供近邻参照；本轮以来源特定设计及其完整训练增量确定研究贡献。

### 5.3 条件组合

仅在主决策文件触发时，用已冻结 $\zeta_\star,\alpha_\star$ 运行同一个框内公式；U 必须由该组合自己的 $D_{\zeta_\star,t}$ 构造。不能混用另一个 C2 轨迹的 U、H0 的 V 或先训练 C1 再热移植 C2。组合从共同初始化开始完整训练，与较强单法比较；不追加 $\zeta\times\alpha$ 网格。

## 6. M：在自身轨迹上匹配候选局部整组步长

S控制先完成后，依06/08冻结该家族的一个C或S代表，再派发M；M的目标完整ID及βP进入config hash。M 每步只在自己的 $\theta^M$ 上 forward/backward 一次，并从初始化维护两个概念上完整的 optimizer shadows：H0 与该代表 C⋆。它们都接收 M 当前的 b、h、None mask、PAIR 事件；历史只来自 M 走过的路径。相同的 $m_B$ 可共用一份；仅当两者βP都为.99且时钟相同时才共用Q。目标是S代表时分别维护H0的Q99和候选Q9/U9，偏差修正各用自己的核，不能共用Q。V/U按各自递推维护。

先计算两套本地拟提交更新 $d^H=-\eta u^{H0|M}$、$d^C=-\eta u^{C\star|M}$，包含完整自然/辅助方向和一次 coupled WD 的影响。对固定组 $G\in\{\mathrm{STAR\_SKIP},\mathrm{EXPERT}\}$，以原固定参数顺序串联该组当前有效坐标，FP64 累加平方范数：

$$
d^M_G=\begin{cases}
d^H_G\,\|d^C_G\|_2/\|d^H_G\|_2,&\|d^H_G\|_2>0,\\
0,&\|d^H_G\|_2=0.
\end{cases}
$$

H0 范数为 0、目标范数也为 0 时记录 `MATCHED_ZERO`；H0 为 0 而目标非 0 时记录 `UNMATCHED_ZERO` 并提交零，不生成随机方向。目标为 0、H0 非 0 时按比例提交零。先分支再除法完成零保护，不把 $\|d^H\|+\epsilon$ 放入所有非零比例，不裁剪比例，不搜索额外 guard 超参。

组半径由完整拟提交更新定义；参数 FP32 加法后的实际位移另记数值残差。该构造检验组尺度是否足以解释候选收益；它保留 H0 的组内方向，并不匹配逐坐标尺度或假定所有正则化解释已排除。

完成两个组的决定后只提交一次 M 更新；J 外是 M 自己的原生 Adam，EMA 读取最终 M 后态一次。禁止从真实 C⋆ 或真实 H0 轨迹拷贝状态/范数 trace；禁止让 shadow 执行隐含参数提交或多更新一次时钟。shadow 计算单列运算成本，不计为新的训练轨迹。

## 7. S 与 P：有限匹配控制

### 7.1 S：同操作的短辅助核

对已冻结父候选，将全部辅助历史中的 $\beta_P=.99$ 改为 .9，并将 Q/U 的偏差修正同步改为 $1-.9^{k_p}$。$\beta_B=.9,\beta_2=.999,\zeta,\alpha$、PAIR 图、V 的生成、作用域和无 PAIR 语义均不变；完整训练从零状态启动。

C1 的 S 改 Q；C2 的 S 同时改 Q 与 U，避免插值的两端使用不同核。它回答新增操作是否仍依赖较长辅助历史；不会重复 A22 对整个 mixed 梯度统一使用 .99 的设计，也不把当前 h 替代 Q/U 加成另一个默认臂。

### 7.2 P 对 C1：AdaTask-inspired 两来源分母

在相同静态 PAIR、相同参数支持、相同 $m_B,Q$ 与事件注入规则下，维护

$$
V_{B,t}=.999V_{B,t-1}+.001b_t^2,\qquad V_{P,t}=.999V_{P,t-1}+.001h_t^2,
\quad D_{B/P,t}=\left[\sqrt{V_{B/P,t}/(1-.999^{\tau_p})}+\epsilon\right]^{-1},
$$
$$
u^{P(C1)}_t=D_{B,t}\widehat m_{B,t}+e_t\lambda D_{P,t}\widehat Q_t.
$$

**两个二阶态都在 native 参数步上更新、使用同一 $\tau_p$。** 无 PAIR 时 h=0，因此 V_P 正常衰减；Q 保持且不注入。V_P 累积未乘 $\lambda$ 的 h²，$\lambda$ 仅在方向相加时外乘；b 独占 WD。不能把 V_P 改为事件时钟，不能先乘 $\lambda$ 再按独立 Adam 抵消其权重。

该适配继承 AdaTask 分离来源累积尺度的核心操作 [L3,L4]，保留 H0 的 .9/.99 一阶核以直接比较共同几何与分源几何。该臂登记为两监督来源的内部 optimizer 适配参照；正式发表推荐模型基线按主矩阵另列。

### 7.3 P 对 C2：LaProp-inspired mixed 预条件历史

S_DENSE 内对同一完整 mixed 梯度与原 V0，先更新 V0 并计算 D0，再在每个 native 参数步更新

$$
T_t=.9T_{t-1}+.1D_{0,t}g^{\rm mix}_t,\qquad
u^{P(C2)}_t=T_t/(1-.9^{\tau_p}).
$$

无 PAIR 的 native 步仍正常更新 T；没有单独事件 Q/U，也不二次预条件。J 外原生 Adam 不变。这个参照在完整 mixed 梯度上执行预条件后动量，与 C2 的来源分离和辅助事件核形成实际近邻比较。

LaProp 原文 Algorithm 1 与作者代码均为先 V/当前分母、后动量 [L1,L2]。作者代码默认 eps=1e−15，且在梯度更新后施加解耦 WD；这里明确适配为本协议 eps=1e−8、固定 lr=.01、b 内一次 coupled WD、无 centered/AMSGrad。固定 lr 时其 lr-weighted 动量可等价写成上式，不引入 lr schedule。

C2−H0 识别所提辅助运算的增量；C2−P(C2) 比较来源组织的完整方案。P 同时改变自然通道历史与辅助时钟，其差值应按完整 optimizer 参照解释。

## 8. 每步事务与伪码

下面是待实现的执行顺序，不是已存在的训练入口。关键中间量都来自更新前同一 $\theta$；所有拟提交方向生成后，参数才提交。

```text
step(batch, arm, state):
  if lambda_PAIR == 0:
      return canonical_no_PAIR_parent(context).step_and_one_ema_update(batch)
  if arm is original H0 on main S_DENSE, or a registered beta_P=.99 candidate zero point:
      return canonical_history_parent(context).step_and_one_ema_update(batch)
  if arm is B1:
      return canonical_static_PAIR_Adam_train_step_and_one_ema_update(batch)
  zero_grad(set_to_none=True)
  probability, logits = original_forward_with_tower_hooks(batch)
  L_B = original_mean_BCE(probability, y)
  pair = original_static_sample(original_domains, y, row_ids, keyed_context)
  raw_r = None
  if pair exists:
      R = mean_softplus_of_2048_original_logit_margins(pair)
      if source_split_needed:  # C1/C2/M/S/P(C1)/H0_N；B1和P(C2)不需要
          raw_r = detach_each_present_tensor(grad(R, custom_scope, retain_graph=True))
      L = L_B + lambda_PAIR * ratio * R
  else:
      L = L_B
  L.backward()                          # 真正native梯度，覆盖完整图
  assert loss and all present gradients are finite
  freeze native_None_mask; snapshot only required prestate
  increment global pair count once if pair exists
  for p in custom_scope, in frozen parameter order:  # B1的custom_scope为空
      if native_grad[p] is None: continue  # 主S_DENSE此前应断言不存在
      tau[p] += 1
      mixed = native_grad[p] + wd * theta_pre[p]
      if arm is P(C2):
          update V0 from mixed, then D0, then T; record T/bias(tau[p]); continue
      r_p = auxiliary_tensor_or_registered_zero(raw_r[p]) if pair exists else None
      b = mixed if no_pair else mixed - (lambda_PAIR * ratio) * r_p
      h = 0 if no_pair else ratio * r_p
      mB = .9*mB + .1*b
      update requested V(s) on tau[p]; construct current bias-corrected D(s)
      if pair exists:
          k_p[p] += 1
          Q.mul_(beta_P).add_(r_p, alpha=frozen_complement_beta_P*ratio)
          if U needed: U = beta_P*U + (1-beta_P)*D_current*h
      build requested bias-corrected direction; omit Q/U injection if no_pair
  if arm is M: use M-local H0/Cstar shadows and fixed-group norm rule
  prepare original native Adam update for scope complement, complete mixed loss
  commit each parameter once; update whole-model EMA once
  log clocks, source/support identity, finite checks and update diagnostics
```

该伪码表达依赖顺序；H0 及 $(0,0)$ 数值验收使用原 A21 方法，不用伪码中的重排覆盖其浮点路径。新候选中 Q 可先于 V 计算，U 必须后于当前 V/D；为减小实现歧义，正式新增路径固定上面的顺序并登记源码 hash。

若发生非有限 loss/梯度/矩/方向或状态提交失败，不登记成功 step，也不更新 EMA；该 attempt 按工程事务恢复，不由 optimizer 静默跳过一次。已有只读诊断不修改 live 状态、原 RNG 或 loader。

## 9. EMA、条件作用域与形状约束

EMA 初始复制 online，并在完整参数事务后调用原 `update_ema` 一次：浮点参数满足 $F_t=\rho F_{t-1}+(1-\rho)\theta_t$。$\rho_{ML}=.9828205985452511$，$\rho_{KR}=.995$；buffer/mode 的处理继承原 utility。EMA 不参与训练 forward、PAIR 选择、V/Q/U 更新或控制比例计算。[S3,S5]

同一轨迹保留 online 与 fixed EMA 两个读出，各自按完整 Validation 六格 Macro AUC 选最早严格最大 epoch；新方法主读出为 online，固定 EMA 是增量及交互评价。两个读出不增加训练轨迹或独立 seed 数。

**条件 N 对照。** `H0_N` 与 `Cstar_N` 以相同 N=D/T/DT 替换 J，保持 N 与共享 EMBED/S_DENSE 互斥，域字段 embedding 的三域坐标归 D。两者都采用第 2 节 $\tau_p,k_p$ 与 raw-r None 语义；其余坐标对完整 mixed 目标执行普通 Adam。用 $Cstar_N-H0_N$ 与 $Cstar-H0$ 比较作用域增量，不从绝对 AUC 推断共享结构的必要性。

冻结 `parameter_scope_manifest`：参数名、shape、dtype、device 类别、scope、固定序号及必要的坐标 mask。原 S_DENSE 为整参数集合；若 N 需要部分张量，native None 与 $\tau_p$ 仍按原参数，scope 更新和普通更新按互斥 mask 合并为一次提交，禁止同一整张量先原生 Adam 再叠加一个完整自定义更新。该适配必须通过 scope 外原生更新等价验收。

N 的 `autograd.grad` 可返回未用 PAIR 路径的 None；只在 native 非 None 且被注册的自定义坐标上将它解释为辅助零。若 raw r 非零而完整 native grad=None，判为图/梯度采集错误。N 的 $k_p$ 必须序列化，不用全局 PAIR 次数修正跳过过 native 步的局部历史。

每份 m/V/Q/U 与其参数或显式 scope view 同 shape、同 FP32 dtype；不以 flatten 后错误广播替代逐参数配对。raw r 列表、state 和 `named_parameters()` 顺序使用同一个冻结清单；shape/name 不同立即拒绝恢复。全局范数仅用于只读记录和 M 的固定组比例。

HyperGate 迁移只在主决策触发后使用其已审查的共享参数注册与匹配父对照；Gaussian 分支只在确认原 GA31-R3 身份后接入，令 $b=\nabla L_{\rm GA31\text{-}R3}+\omega\theta$，其中 L 是原已定义的 Gaussian 混合目标，PAIR 继续来自原自然 forward。两者不改变 C1/C2 公式，也不自动把模型所有 dense 参数命名为 S_DENSE。

## 10. 必要退化、等价与状态验收

这些是未来实现的工程验收，不是追加科学训练条件；本轮规划不执行实际模型训练。

| 验收 | 输入/操作 | 明确通过标准与用途 |
|---|---|---|
| no-PAIR旁路 | 所有新臂设 $\lambda_{PAIR}=0$ | 与当前骨干/基础目标的canonical no-PAIR parent精确相同；主矩阵B0、Gaussian G_B0、HyperGate HG_B0；增强/native目标保留 |
| H0 零点 | 注册的.99候选 $\zeta=\alpha=0$，原输入序列 | 与同背景/骨干/scope历史父入口相同；主矩阵与原A21完整config、参数、optimizer/shadow、EMA、时钟同设备精确相同 |
| 全程无 PAIR | 从零状态起所有 e=0 | H0/C1/C2/组合及M/S数学上为原clean Adam；按FP32容差验收；P(C2)按自己的clean预条件后动量定义执行 |
| 单次无 PAIR | 先产生历史，再插入 e=0 | Q/U/k_p 完全保持、无辅助注入；m/V/$\tau_p$ 正常更新，不要求历史回到 B0 |
| 零辅助梯度 | e=1、r=0，已有历史 | Q/U 正常衰减、事件时钟增加；区别于 e=0 |
| None/zero | N 合法 unused r、native None、native zero | 严格按支持表；None 不新增 WD/step，zero 仍正常步 |
| C1 二阶闭合 | 固定同路径 b/h，零态，同步 native 时钟 | 凸组合等于三 shadow 分解；有符号交叉闭合；所有 V 非负 |
| C2 运算交换 | 固定 D，和一次 PAIR 事件的序列 | Uhat=D·Qhat；C2−H0 为零至算术容差；变化 D 时差满足第 5.2 节式 |
| 同 beta 特例 | $\zeta=\alpha=0,\beta_P=.9$，每 native 步有 PAIR | 两源分子数学上等于 B1 的 .9 mixed m；存在无PAIR步时不要求此等价 |
| M 半径 | 非零 H0 组方向 | 拟提交步范数与 M-local C⋆ 目标一致至浮点容差，记录所有 unmatched zero |
| WD 与提交 | 非零 θ、零数据梯度及普通梯度样例 | 每有效参数一次 coupled WD、一次参数提交、一次 whole-model EMA |
| 保存恢复 | 同一批序中断/恢复 | 含 native 与辅助时钟、Q/U/shadows、RNG、loader 的 StatePacket read-back 后续精确重现 |

解析式张量检查优先 FP64（`rtol=1e-10, atol=1e-12`）；FP32 新式与独立参考计算用 `rtol=1e-5, atol=1e-7` 并报告最大误差。原 H0 六格 shadow 重构沿用源 `rtol=1e-4, atol=1e-6`。canonical 路径与序列化读回采用精确比较，不能用上述容差替代。若误差超限先定位算术顺序/状态/支持问题，不扩大容差使其通过。

## 11. 状态、内存与日志

记 $N$ 为模型参数元素数、$N_J$ 为当前自定义 scope 元素数、$N_O=N-N_J$。下表按全部参数已进入原生更新计算算法活跃态，不含额外诊断 shadow；每个 FP32 元素 4 bytes，模型、梯度、激活、临时 r 与 EMA 另计。支持按整参数分配时，部分 mask 以实际分配元素计费。

| 方法 | J 内活跃状态 | J 外状态 | 相对全模型原生 Adam 的最低额外状态 |
|---|---|---|---|
| B0/B1 | m,V：$2N_J$ | $2N_O$ | 0 |
| H0/C1 | mB,V,Q：$3N_J$ | $2N_O$ | $4N_J$ bytes |
| C2/组合 | mB,V,Q,U：$4N_J$ | $2N_O$ | $8N_J$ bytes |
| P(C1) | mB,Q,VB,VP：$4N_J$ | $2N_O$ | $8N_J$ bytes |
| P(C2) | V0,T：$2N_J$ | $2N_O$ | 0 |
| M(C1) | 共用mB/Q，V0与Vζ：$4N_J$ | $2N_O$ | $8N_J$ bytes |
| M(C2) | 共用mB/Q/V0，另U：$4N_J$ | $2N_O$ | $8N_J$ bytes |

表内M(C1/C2)是目标βP=.99时的最小态。M目标为S_C1时维护mB、Q99、Q9、V0、Vζ，目标为S_C2时维护mB、Q99、Q9、V0、U9，均为 $5N_J$；相对Adam多 $12N_J$ bytes。M若绑定组合需要mB/Q/V0/Vζ/U，共 $5N_J$；本轮C12不另开M训练。未做共享存储优化时两套shadow的实际占用照实登记。固定 EMA 增加约 $4N$ bytes 参数态及其原 buffer。原 H0 包装器另有 $12N_J$ 的被动六格 shadows，即额外 $48N_J$ bytes；算法最低态与源包装器实际驻留分别报告，不能漏计。

raw r 与 h/D/方向等临时量按峰值记录；只读分解需要 V0/Vsep 或完整方向时按需申请并登记。C1/C2 无额外模型 forward，PAIR 活动步继承一次 J 上的 raw-r autograd sweep，再做完整 loss backward；P(C2)只需 mixed 梯度。M 用同一 r，不另对 shadow 做 forward/backward。实测壁钟、峰值显存/RAM优先于状态字节估算。

StatePacket 至少保存 online 参数/buffers/mode、原生 optimizer 参数组与私有 step、mB/V/Q/U/各 P 和 M states、全局 k 与适用的局部 k_p、PAIR provenance、被动 shadows、EMA、RNG、loader permutation/cursor、selected 端点及 source/data/config/group hash。模型训练状态沿用 RAM-only 协议；该文件仅规定字段，不创建运行状态。

每步或冻结诊断点记录 e、活动条件、条件/索引 hash、ratio、$\tau_p/k_p$ 摘要及 native None mask hash；记录每组 b/h、历史与真实拟提交更新的范数、M 比例/零例外、V 非负性与有限性。C1 记录有符号 cross 与共同分母变化；C2 记录 Uhat−D·Qhat 及两个辅助方向的夹角/范数；这些指标不作为取消 full50 的局部门。

## 12. 与既有路线的差别及算法研究决策

| 已有对象 | 已有运算/结论边界 | 本轮保留的新操作 |
|---|---|---|
| A20/A21/A22 | 条件历史数量、两源一阶核、整个 mixed 的较长普通历史已比较 [S4,S5] | C1 改共同二阶交叉项；C2 改辅助预条件与历史的顺序 |
| SAD/AIG | 原始增强差分先平均，再除当前 mixed 分母 [S6] | 条件 PAIR 来源及产生时预条件方向的保存 |
| H_REAL/H_MIX | 当前度量下投影旧 raw 历史，仍以当前 D 作用 [S7] | U 直接记忆 $D_s h_s$，不增加投影 |
| TV-GRM | raw view-clean 残差历史，再用当前 clean D 和 cap [S8] | 本轮固定原 PAIR 与 mixed 来源，不引入 cap；C1/C2 的操作各自独立 |
| EMA 参数/窗口归因 | 共享尤其 EMBED 和跨期组合价值已有证据 [S9] | 用现有证据提出 optimizer 候选，复用同轨迹 online/EMA 估计新增交互 |
| 通用分源/预条件动量 | AdaTask、LaProp 已提供相关原理 [L1–L4] | 以 P 近邻参照检验 MDMT 中来源组织的实测独立贡献 |

**研究决定：** 两类机制都有明确、可退化、可对照的完整算法，默认各两配置进入 full50。新结果分别回答“操作确实改变了更新”“完整排序是否提高”“与 H0/普通 PAIR/固定 EMA 是否有独立增量”。某个数据集失败会收窄适用范围；局部方向负相关不会取消另一数据集的可靠收益，也不会跳过已冻结完整候选。

最终候选由主文件的 Validation 选择和六格保护规则冻结；M/S/P 仅在相应候选进入条件验证时运行。C1/C2 未形成完整训练收益时，结束对应版本，不扩充随机历史、A20/A22、B selector、NQR、EMA 时间窗或任意 beta 网格来延长论证。

## 来源

- [S1：原 PairOptimizer、shared_name、A21 活跃递推](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/experiments/conditional_information_momentum_20261102_v3/optimizers.py)。
- [S2：完整目标与原采样器](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/experiments/conditional_information_momentum_20261102_v3/objectives.py)；[boundary_sampler.py](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/experiments/conditional_information_momentum_20261102_v3/boundary_sampler.py)。
- [S3：原 v3 协议常量与 EMA utility 入口](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/experiments/conditional_information_momentum_20261102_v3/protocol.py)。
- [S4：A21 静态配置、HistoryOptimizer 与六格 shadow](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/experiments/conditional_history_shared_update_20261122/history_controls.py)。
- [S5：EXP080 设计与实现身份](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/research_reports/conditional_history_shared_update_20261122/DESIGN_SPECIFICATION.md)。本轮主读出采用 online，旧研究 fixed-EMA 主读出不覆盖本轮选择规则。
- [S6：SAD 原报告](https://github.com/Caelestis1903/My-Research-Project/blob/ee4cf0512a7723624c9ee6a1ef192f0f7f17686b/research_reports/2026-11-01_momentum/REPORT.md)；[AIG 原报告](https://github.com/Caelestis1903/My-Research-Project/blob/6f5742c2085c1e49d1767b0a7079bb9f28160eda/research_reports/2026-11-01_aig_m/REPORT.md)。
- [S7：H/T 优化器实际实现](https://github.com/Caelestis1903/My-Research-Project/blob/f9c2a8ef7b29b899e63b1beba823df1b6ea22c72/experiments/mdmt_history_information_20261103/optimizers.py)。
- [S8：TV-GRM 实际 core](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/experiments/task_view_gradient_residual_momentum_20261116/core.py)。
- [S9：EMA 共享参数与窗口归因](https://github.com/Caelestis1903/My-Research-Project/blob/9df48737dd6693f892b85db2344cdee76fb05d6f/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)。
- [L1：LaProp 原文，arXiv:2002.04839v3，Algorithm 1](https://arxiv.org/html/2002.04839v3)；[论文记录](https://arxiv.org/abs/2002.04839)。
- [L2：LaProp 作者代码 laprop.py](https://github.com/Z-T-WANG/LaProp-Optimizer/blob/master/laprop.py)，本规划于 2026-11-24 读取并对照默认 eps、WD 与递推顺序。
- [L3：AdaTask 原文，AAAI 2023，Algorithm 2](https://arxiv.org/html/2211.15055v2)。
- [L4：AdaTask 作者代码](https://github.com/EnnengYang/AdaTask/blob/master/adatask.py)。
