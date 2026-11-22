# 方向 A：条件样本对与梯度历史更新

本文件是后续实现与执行的设计规格；本次交付不实现算法、不启动训练。研究快照、数据身份和复用凭据见 [研究上下文](01_RESEARCH_CONTEXT.md) 与 [来源登记](12_SOURCE_AND_REUSE_REGISTER.md)。总调度、跨方向预算与最终资格分别以 [执行入口](00_EXECUTION_RUNBOOK.md)、[实验矩阵](05_EXPERIMENT_MATRIX.md) 和 [决策规则](09_DECISION_RULES.md) 为准。

## 1. 当前结论、关键缺口与研究决定

**当前接受的结论：** P01 完整配置在 KuaiRand 相对 clean+固定 EMA 有五种子历史 selected Test Macro AUC 增量 +0.004085725，但 d1t0/d1t1 分别下降约 0.001334/0.001047，当前版本未通过逐格保护。MovieLens 对应增量为 −0.000623736。两个数据集的差异界定该完整方法的适用范围，不抹除 KuaiRand 已观察到的宏收益。[五种子报告](../../research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md)

**关键缺口：** 原 P01−P00 只有 seeds 2022/2024；KuaiRand 历史 Test 均值为 +0.012098830，而 Validation 为 −0.000100784。现有证据值得补足五种子，但不能将这两种子的 Test 结果直接升级为已确认的历史组件收益。[原实验报告](../../research_reports/2026-11-02_mdmt_information_momentum_v3/REPORT.md)

**研究决定：** 先补 12 条缺失完整训练，区分辅助监督、历史更新包和动态抽样；有历史正均值后，再用静态合并历史和单份长 Adam 历史判断复杂结构是否值得保留。只有相对未被削弱的 clean+EMA 仍有实际增量，才进入剂量与作用位置研究。方向 A 停止不取消方向 B 的核心机制判别。

这里有四个不同层次的结论：

| 层次 | 本轮允许得出的结论 | 不由该层次自动推出的结论 |
|---|---|---|
| 完整方法经验结果 | 某数据、既定协议下完整训练的宏与六格变化 | 某一个新增组件的贡献 |
| 匹配完整训练对照 | 在同辅助监督下改变历史处理的训练政策增量 | 24 个条件身份分别建表是唯一原因 |
| 同轨迹冻结干预 | 指定组或历史来源在当前状态的一步排序效应 | 连续 50 epoch 应用一定有益 |
| 正式候选完整训练 | 所选实现相对强内部对照的独立增量及保护状态 | 对所有模型或数据人口普遍成立 |

**小域不是困难域。** 本方向不设计该结论的证明、否定或假设性实验，不把样本数量当作困难度、历史效用或损害的代理。

## 2. 冻结实验协议与参数覆盖

### 2.1 数据、训练和读出

沿用原 v3 自然人口协议：

- MovieLens、KuaiRand，原行序 80/10/10；KuaiRand 先切分再保留 tab 0/1/4；编码与 scaler 只拟合训练集，OOV=0。
- MovieLens 标签为 rating>3/rating>4；KuaiRand 为 is_click/long_view。不得人工补造不存在的联合标签。
- M3oE fixed1：embedding 维度 16，4 个共享专家，FCN [512,256,256,64]，三个域、两个任务。模型结构及梯度图不改。
- 50 epoch，无早停；batch size 40960，保留自然尾批。MovieLens 每 epoch 20 步、全程 1000 步；KuaiRand 每 epoch 49 步、全程 2450 步。
- FP32；AMP/TF32 关闭；Adam lr=.01，β2=.999，eps=1e−8，coupled weight decay 为 ML 1e−4 / KR 1e−5；foreach/fused/amsgrad 均为 False。
- 所有正式性能比较、所有超参数和作用位置选择均使用 2022、2023、2024、2025、2026，且新正式配置都在两数据完整评价。一个数据的正向信号可以触发阶段，不要求两数据同时正。
- 每条训练派生 online 与 fixed_ema 两个读出；主读出为 fixed_ema。ρ 为 ML .9828205985452511、KR .995；不搜索 ρ。
- 每个读出独立选择 E1–E50 中 Validation 六格 Macro AUC 最早严格最大值；排除 E0。E10/E50 是预注册阶段补充，不替代主选点。

上述设置来自原 [protocol.py](../../experiments/conditional_information_momentum_20261102_v3/protocol.py)、[训练 worker](../../experiments/conditional_information_momentum_20261102_v3/run.py) 及其冻结 vendor 依赖。数据 raw SHA256 与完整 source manifest 由来源登记核验；执行不得凭文件名相同即判定可复用。

### 2.2 真实参数映射

| 本轮组名 | 真实参数名称或前缀 | A20/P01 的处理 |
|---|---|---|
| EMBED | embedding.embed_dict.* | 完整混合梯度的普通 Adam，β1=.9 |
| STAR_SKIP | shared_weight、shared_bias、skip_conn.*、star_mlp.* | ECCM 一阶历史 |
| EXPERT | expert.* | ECCM 一阶历史 |
| 其他参数 | slot_weight.*、slot_bias.*、domain_expert.*、task_expert.*、gate.*、tower.* | 完整混合梯度的普通 Adam，β1=.9 |
| FUSION | fixed1 的 exp_d、exp_t、bal_d、bal_t | 值为 1 的普通常量，不是 Parameter；可训练 FUSION 组为空 |

定义 S_dense = STAR_SKIP ∪ EXPERT。它不是全部共享参数，尤其不包含 EMBED。expert.* 内部名称即使含 domain_specific，也按外层共享专家归属 EXPERT；不能按子串误分为非共享组。make_model 原有 _weight_ / dbo_ 参数排除断言保留。[分组实现](../../experiments/conditional_information_momentum_20261102_v3/optimizers.py)

PAIR 辅助梯度仍沿真实训练图作用于所有可达参数；“仅在 S_dense 维护历史”不等于“PAIR 只更新 S_dense”。gate 对 embedding 的 detach 和其他原图行为保持不变。原 hook 只读取真实 tower 的 sigmoid 前 logit，不新增独立头或改变前向路径。[logit 获取实现](../../experiments/conditional_information_momentum_20261102_v3/boundary_sampler.py)

## 3. 原条件样本对：精确采样与目标

### 3.1 24 个条件与有效支持

六格用 c=(d,t) 表示，d∈{0,1,2}，t∈{0,1}。联合标签 a=2y₀+y₁，编码 0/1/2/3 对应 00/01/10/11。边界条件 i=(d,t,a⁺,a⁻) 必须同域，且目标任务 t 在 a⁺ 为正、在 a⁻ 为负。

源码按 d、t、a⁺、a⁻ 的升序枚举，完整索引如下；“有效”仍须当前真实 batch 同时存在两端样本。

| 每域局部索引 k | t | a⁺→a⁻ | 全局条件 ID | MovieLens 理论标签支持 |
|---:|---:|---|---|---|
| 0 | 0 | 10→00，即 2→0 | 8d+0 | 可有效 |
| 1 | 0 | 10→01，即 2→1 | 8d+1 | 不存在 01，不有效 |
| 2 | 0 | 11→00，即 3→0 | 8d+2 | 可有效 |
| 3 | 0 | 11→01，即 3→1 | 8d+3 | 不存在 01，不有效 |
| 4 | 1 | 01→00，即 1→0 | 8d+4 | 不存在 01，不有效 |
| 5 | 1 | 01→10，即 1→2 | 8d+5 | 不存在 01，不有效 |
| 6 | 1 | 11→00，即 3→0 | 8d+6 | 可有效 |
| 7 | 1 | 11→10，即 3→2 | 8d+7 | 可有效 |

因而 MovieLens 最多 12 个有效条件；KuaiRand 保留真实存在的 01，最多 24 个。当前缺席的条件不补样本、不借其他域填补，也不把“标签支持较少”解释为困难度。[条件枚举与抽样](../../experiments/conditional_information_momentum_20261102_v3/boundary_sampler.py)

### 3.2 权重、静态 q 与动态 q

设当前 batch 大小为 B，n_d 为域 d 的行数，n_{d,a} 为联合标签 a 的行数，n_{d,t,+}/n_{d,t,−} 为目标任务的正/负行数。活动条件集为 A，K=|A|。对 i=(d,t,a⁺,a⁻)：

$$
w_i=\frac{n_d}{2B}
 \frac{n_{d,a^+}}{n_{d,t,+}}
 \frac{n_{d,a^-}}{n_{d,t,-}},\qquad
Z=\sum_{i\in A}w_i\le 1.
$$

活动条件非空时，静态 P00/P01 及全部新静态历史对照使用：

$$
q_i=\frac{.25}{K}+.75\frac{w_i}{Z}.
$$

动态 P10 使用：

$$
q_i=\frac{.25}{K}+.25\frac{w_i}{Z}
 +.5\frac{w_i(.05+D_i)}{\sum_{j\in A}w_j(.05+D_j)}.
$$

D 初始为 .5；采样读取更新前 D。两式都保留原目标质量 w；改变 q 只改变抽样分配和随机估计路径，不将动态需求直接作为新的目标权重。当前 Z<1 时也不将 w 重新归一为和 1。

每步从 A 抽一个条件 s，然后分别从其正端与负端行集合有放回均匀抽取 2048 个索引，组成 2048 对。不是枚举全体正负笛卡尔积，也不是改变真实训练 loader。源码保证 w_i/q_i≤4、.01/q_i≤.96；沿用该性质，不另加裁剪或温度参数。

### 3.3 辅助目标与原始梯度

以模型真实 tower logit z 为目标：

$$
R_s=\frac1{2048}\sum_{\ell=1}^{2048}
 \operatorname{softplus}\bigl[-(z_{p_\ell,t_s}-z_{n_\ell,t_s})\bigr],
$$
$$
L_0=\frac1{2B}\sum_{r=1}^{B}\sum_{t=0}^1
 \operatorname{BCE}(p_{rt},y_{rt}),\qquad
L=L_0+\lambda\frac{w_s}{q_s}R_s .
$$

核心 λ=.05。r_s=∇R_s 是未乘 λ、未乘 w/q、未含 weight decay 的原始辅助梯度；g₀=∇L₀。所有控制臂保持相同目标、归一化和参数可达范围。

**没有 RMS 归一化。** w/q 是重要性权重，不是梯度范数匹配；没有 r/∥r∥、按参数 tensor 重标定或相对真实梯度的尺度平衡。Gaussian 路径使用过的每字段 embedding RMS 不属于 PAIR，不能搬入核心比较。未来若引入训练期 RMS，会同时改变辅助目标和历史输入，须另设匹配普通 Adam 对照，本轮不展开这项训练网格。

活动集为空时，辅助 loss 为 0；真实训练继续，不补辅助质量。λ=0 完全旁路 PAIR、历史与其额外 RNG，必须回到 A00 语义。

### 3.4 配对随机性与动态需求更新

PAIR 随机数由 dataset/seed/epoch/global_step/stream 的 SHA256 派生；stream 仅为 condition、pair_indices，不含 arm 或 λ。loader 使用独立 generator，seed+11000。静态臂在同 dataset/seed 下具有逐步相同的真实 batch、条件和 pair 行索引；梯度在各自已分叉的 θ 上计算，不要求训练后梯度值相同。

P10 在本步 optimizer 与 EMA 完成后才更新选中的 D：

$$
D_s\leftarrow .9D_s+.1\,\frac1{2048}
\sum_\ell \sigma\bigl[-(z_{p_\ell,t_s}-z_{n_\ell,t_s})\bigr].
$$

该 error 来自本步更新前 forward。静态臂不更新 D。P10 与 P00 的相同 RNG 流不保证抽中条件相同，因为 q 已不同；不能把它报告为逐步匹配 pair。原实现与调用顺序见 [objectives.py](../../experiments/conditional_information_momentum_20261102_v3/objectives.py)。

## 4. 四种历史形式与状态时钟

以下公式均针对某个非 None 梯度参数 p；τ_p 是它自身的有效 optimizer step。记 η=.01，ω 为数据对应 coupled WD，β₂=.999。当前混合梯度（含一次 WD）为：

$$
a_p=g_{0,p}+\lambda(w_s/q_s)r_{s,p}+\omega\theta_p.
$$

无 pair 时省略辅助项。当前参数值是更新前 θ，不对历史或独立辅助梯度再次加 WD。

### 4.1 A10/P00：普通 Adam

$$
m_p\leftarrow .9m_p+.1a_p,\quad
v_p\leftarrow .999v_p+.001a_p^2,\quad
\Delta\theta_p=-\eta
\frac{m_p/(1-.9^{\tau_p})}
{\sqrt{v_p/(1-.999^{\tau_p})}+10^{-8}}.
$$

A30/P10 使用同一 Adam，只更换动态 q。A00 没有 PAIR，其他 Adam/EMA 规则相同。

### 4.2 A20/P01：ECCM 条件历史

仅 S_dense 用此更新。每个当前活动条件 i 的时钟 n_i 加一；仅被抽中的 s 刷新 H_s：

$$
H_s\leftarrow H_s+\frac{.01}{q_s}(r_s-H_s).
$$

其余 H 保持不变。H 从 0 初始化；本步读取刷新后的 H_s。缺席条件的 n_i 冻结，本步也不注入该条件历史；n_i 是活动次数，不是被抽中次数。selected_count 只作日志，不参与偏差校正。

$$
m^R_p\leftarrow .9m^R_p+.1(g_{0,p}+\omega\theta_p),\qquad
v_p\leftarrow .999v_p+.001a_p^2,
$$
$$
N_p^{ECCM}=
\frac{m^R_p}{1-.9^{\tau_p}}+
\lambda\sum_{i\in A}\frac{w_iH_{i,p}}{1-.99^{n_i}},
\qquad
\Delta\theta_p=-\eta\frac{N_p^{ECCM}}
{\sqrt{v_p/(1-.999^{\tau_p})}+10^{-8}}.
$$

实现先对 S_dense 求 r_s，再对完整 mixed loss backward，并通过 mixed−λ(w_s/q_s)r_s 恢复真实数据梯度；WD 在真实通道和总二阶矩中各依公式出现一次。不能把 H 更新改成每次抽中便执行 .99H+.01r，不能按抽中次数校正，也不能删除 q 校正。

空 pair 步不更新 H/n、不注入辅助历史，但 mR、v 和 τ 正常推进。普通参数 None 梯度意味着不推进 step、WD、m、v；显式 0 梯度正常推进。S_dense 按原实现必须有梯度，意外 None 是接线错误，应技术阻塞，不能默默跳过。[精确 optimizer 源码](../../experiments/conditional_information_momentum_20261102_v3/optimizers.py)

在固定当前梯度与活动条件下，抽样刷新在条件期望中对应 .99H+.01r；这解释了原频率校正目的。完整训练里 θ、q、活动支持和 bias correction 都随轨迹变化，报告不得把这一局部恒等式扩写成整个方法的精确无偏性证明。

### 4.3 A21：PCOLL_STATIC_099，单份辅助历史

PAIR、静态 q、λ、S_dense、mR、v、WD 和 EMA 与 A20 相同，只保留一份辅助历史 Q：

$$
Q\leftarrow .99Q+.01(w_s/q_s)r_s,\qquad n_\Sigma\leftarrow n_\Sigma+1,
$$
$$
N_p^{COLL}=
\frac{m^R_p}{1-.9^{\tau_p}}
+\lambda\frac{Q_p}{1-.99^{n_\Sigma}}.
$$

nΣ 只在有效 pair 步推进；无 pair 时 Q/nΣ 冻结且不注入，真实通道继续。所有其他参数仍使用 β1=.9 的普通 Adam。

**原 config("PCOLL",…) 的 dynamic=True，不能直接作为 A21。** 本轮必须新增静态配置，明确 mode=collapsed、dynamic=False、lambda_pair=.05、history_scope=S_dense；数值内核复用原 collapsed 分支。A21 不是声称已有正式性能结果：旧 PCOLL 仅有工程 smoke。

A20−A21 的解释包括“条件分开存储、条件活动时钟与当前条件权重重组”相对于“统一保存过去已加权梯度”的差别。P01 用当前 w 乘各条件历史；PCOLL 在 Q 中记住过去 w/q，因此两者并非只把相同数字换一个存储容器。只有在本轮所测替代中确有更好的完整训练结果，才保留条件表作为有用结构，不写成所有可能合并历史都已被排除。

### 4.4 A22：ADAM_S99，单份完整混合历史

S_dense 对完整混合梯度 a 使用单份标准 Adam m，β1=.99：

$$
m_p^{99}\leftarrow .99m_p^{99}+.01a_p,\qquad
N_p^{99}=\frac{m_p^{99}}{1-.99^{\tau_p}}.
$$

v、WD、lr、eps 不变；S_dense 以外 β1=.9。没有独立辅助 Q、没有条件 H、没有 pair clock 参与 Adam 偏差校正。非 None 参数在空 pair 步仍以真实梯度加 WD 正常更新 .99 历史。

λ=0 是统一的 A00 旁路：新配置工厂直接构造全参数 β1=.9 的 A00 optimizer，不能仅关闭 PAIR 却保留 S99 的 .99 参数组。空 pair 的 λ>0 正常步骤与全程 λ=0 的基线旁路是两种不同语义。

A22 检查较长单历史是否已足够；它同时改变真实数据与 WD 的一阶时间尺度。A21−A22 是两通道/两时间尺度方案相对单通道长历史的比较，不将其差值全部归因于条件身份或辅助梯度记忆。

### 4.5 每步执行顺序

1. 从原 loader 取当前自然 batch，完成一次真实 forward 与只读 logit hook。
2. 用旧 D 和专用 RNG 抽条件与 2048 对；无 pair/λ=0 执行相应旁路。
3. 需要显式历史时对 S_dense 求原始 r_s，retain_graph；随后 mixed loss backward。
4. 捕获诊断所需同轨迹梯度并更新不反馈 shadow；诊断不能改 live grad、模型、RNG 或 optimizer。
5. ECCM/COLL 在选定参数组更新历史与时钟，再更新 mR/v/θ；普通 Adam 或 S99 按互斥参数组更新，每个参数只 step 一次。
6. 所有在线参数更新完成后统一 EMA 一次：φ←ρφ+(1−ρ)θ，初始 φ₀=θ₀，buffer 原样复制。
7. 动态臂更新 D；写标量与身份日志；进入下一真实 batch。

P00/P01 的二阶矩**公式一致**，不是训练分叉后 v 数值仍相同。冻结更新比较只有在明确共享同一当前状态和梯度时，才使用同一个真实 v 作共同分母。

## 5. A_CORE：最小五种子补全

### 5.1 正式矩阵与复用

| ID | 原名/处理 | 两数据已完成 seeds | 两数据各新增 seeds | 新完整训练 |
|---|---|---|---|---:|
| A00 | clean、普通 Adam，online/F 两读出 | 2022–2026 | 无 | 0 |
| A10 | P00，静态 PAIR+普通 Adam，λ=.05 | 2022/2024 | 2023/2025/2026 | 6 |
| A20 | P01，静态 PAIR+ECCM，λ=.05 | 2022–2026 | 无 | 0 |
| A30 | P10，动态 PAIR+普通 Adam，λ=.05 | 2022/2024 | 2023/2025/2026 | 6 |
| 合计 | 40 条逻辑训练轨迹 | 28 条数值复用 | 12 条新训练 | 12 |

不重复已有 A00/P01 全程训练以再次确认已知收益；不把 online/F 两个读出计为两条训练。原 supplement 与五种子表的 A00/P01 已核实 96 条同身份端点，模型 hash/AUC 全同，LL 仅浮点舍入差。正式执行只重核文件身份、复用映射和来源冻结，不再重复整轮科学训练。[复用来源](12_SOURCE_AND_REUSE_REGISTER.md)

每数据输出五种子配对差：

- A10−A00：固定条件辅助监督的增量。
- A20−A10：相同静态辅助监督下整个 ECCM 历史处理包的增量。
- A20−A00：完整历史方案相对强内部对照的总增量。
- A30−A10：动态条件抽样的增量。
- A30−A00：动态辅助监督路线相对强内部对照的总增量。

统计单位是 seed，报告逐 seed、均值、SD、配对 95% t 区间、正负方向数、六格与 Macro AUC/LL；区间和方向数用于描述，不额外要求五 seed 全正、4/5 正或 CI 不跨零。详细计算遵循 [统计规范](07_METRICS_AND_STATISTICS.md)。

### 5.2 核心分支

对数据 d 定义 Validation selected/F 的配对均值：

$$
H_d=\overline{A20-A10},\quad
C_d=\overline{A10-A00},\quad
J_d=\overline{A20-A00}.
$$

- 两数据均 H≤0：本版 ECCM 无五种子 Validation 继续信号，不进入历史复杂度阶段；若 C>0，保留 PAIR_ONLY。A30 若有相对 A00 的增量，单独保留动态监督的条件性结果，不冒充历史收益。
- 任一数据 H>0：接受该数据历史包的正均值信号，触发 A21/A22 在两数据五种子完整运行。该触发不要求 J>0 或最终 +.002/+.003。
- H>0 但 J≤0：允许完成简单替代判断，但在没有新历史形式超过 A00 之前，不进入六格补救。相对被 PAIR 拉低的 Adam 的改善不能单独取得新算法资格。
- 只有一个数据有收益：保留数据条件性结论；新增正式比较仍在两数据完整完成，不删除另一个数据。

不足五 seed、训练失败或状态不可用都记未完成/技术阻塞，不写科学阴性。早期单 seed 结果不能取消已触发正式矩阵。

## 6. A_HISTORY_CONTROLS：是否需要复杂历史

若核心触发：

| ID | 配置 | 数据与 seeds | 新完整训练 |
|---|---|---|---:|
| A21 | PCOLL_STATIC_099，λ=.05，S_dense | ML/KR × 2022–2026 | 10 |
| A22 | ADAM_S99，λ=.05，S_dense | ML/KR × 2022–2026 | 10 |

不搜索 β1；.99 直接匹配原辅助历史名义时间尺度。此阶段只增加 20 条。A10 提供普通 .9 Adam 参照；不再新跑一套同配置 Adam。

H* 专指非普通历史 A20/A21/A22 中选出的更新形式，不能取 A10；普通 A10 只用于判断历史是否值得保留。每数据先按相对 A00 的六格保护状态分层，保护通过者优先；同层取 Validation Macro 最大者，距该层最优≤ε_simple=.0001 时按 A22 单份完整 m → A21 mR+Q → A20 mR+条件表 采用更简单者。若都未通过保护，可保留宏最好的历史供剂量研究，但不能称为合格算法。另与 A10 比较：同保护状态且宏差在 ε_simple 内优先普通 Adam，或 A10 已复制/超过历史效果时不推进历史复杂度。ε_simple 是复杂度选择分辨率，不是统计等效性、置信区间门或最终收益门。

各数据最佳形式 H*_d 可以不同，报告明确保留这种差异。后续跨数据主比较使用一个冻结更新形式 H*：从本轮具有相对 A00 与 A10 独立增量的目标数据确定，并在两数据完整评价。若仅一数据满足，取该数据；若两数据都满足且最佳形式不同，选 H*_d−A10 Validation 均值较大的数据作为目标，精确平局按数据 ID 字典序。该目标和形式在 A_GATE 中冻结，另一个数据用于如实展示迁移范围。不能把 ML 的 A22 与 KR 的 A20 拼成一行“通用方法”；也不能让非晋级数据的 A40 退化为 A10，造成 A40/A41 同身份重复训练。不同数据允许选择各自 λ，但主表的更新形式必须一致。

具体解释与动作：

- A10 足以达到相近或更好结果：采用普通 Adam 的条件监督，撤下历史必要性主张。
- A22 相近或更好：采用单份长历史，撤下真实/辅助拆分和 24 条件表的必要性主张。
- A21 胜 A22，且超过 A10：支持本设置的两通道时间尺度处理；若 A20 未超出复杂度分辨率，保留 A21。
- A20 超过所测更简单方案且相对 A00 有收益：保留 ECCM，明确支持的是当前条件表及重组规则的可用性。
- 胜者仍不超过 A00：保留历史或监督的组件现象，结束当前算法改进；不靠继续加模块来修复主动退化。

所有结论按数据报告，不合并 ML/KR 成十个“同分布 seed”。A21/A22 的对照已经承担简单替代判别，不默认追加 PCV、全 β 网格或 P11。

## 7. A_DOSE：已获收益方法的有限剂量选择

### 7.1 触发与历史 Test 的角色

剂量阶段须先满足：所选非普通历史 H* 在至少一个数据相对 A00 有五 seed Validation 宏正增量，且相对同 λ 普通 A10 仍有历史增量。然后满足以下任一预注册研究动机：

1. 当前完整 Validation 结果存在任一格均值下降超过 .001；
2. 本轮开始前已登记的 P01 历史 Test 六格代价尚是该路线的实际未解决问题。

第二条只允许事先固定的二点剂量研究进入队列，**不把历史 Test 作为选择输入**。本轮不重读新 Test，不按旧 Test 的 d1 标签指定保护对象；六格对称评价，λ 与后续作用组只由本轮 Validation/训练诊断决定。尤其 KuaiRand 原 P01 五 seed Validation 六格保护通过，不能把这一事实改写成 Validation 已存在同样的 d1 损害。

若新的 Validation 不支持减弱 λ 或限制作用域，就按 Validation 保留原配置；报告明确“当前流程没有解决已登记的历史 Test 代价”，不返回 Test 调参，也不预告新配置必定修复旧 Test。

### 7.2 配对剂量矩阵

主更新形式 H* 在本阶段开始前冻结，两数据使用同一形式，只比较 λ∈{.025,.05}：

| ID | 配置 | 数据与 seeds | 新完整训练 |
|---|---|---|---:|
| H* 的原 ID | 已选历史、λ=.05 | 两数据五 seed，复用 A20/A21/A22 | 0 |
| A10 | 普通 Adam、λ=.05 | 两数据五 seed，复用核心 | 0 |
| A40_HISTORY_L025 | 同 H*、λ=.025 | ML/KR × 2022–2026 | 10 |
| A41_ADAM_L025 | 同静态 PAIR、普通 Adam、λ=.025 | ML/KR × 2022–2026 | 10 |

新增共 20 条。A41 是必要匹配控制：降低 λ 同时改变辅助监督，只有 H*(.025)−Adam(.025) 才能判断该剂量的历史独立作用。不能用 H*(.025)−A10(.05) 代替，也不能只因 H*(.025) 比 H*(.05) 好便认定历史改进。

在各数据用五 seed Validation 选择 λ；.05 与 .025 使用同样静态 pair、原 loader/RNG、模型、预算、EMA 和选点。优先满足六格保护；同保护状态宏接近时遵循 ε_simple，并优先无需新历史的匹配 Adam。若普通 .025 Adam 已复制或超过历史结果，则选简单方案并撤下最终配置的历史贡献；保留已经识别的 .05 历史现象，不混为同一个估计对象。

只允许上述两个 λ，不扩展更细网格，不根据 Test 回选 .2。λ 选择完成后冻结各数据 λ*，主更新形式 H* 保持一致；下一阶段不再联合搜索历史形式、λ 和作用组。

## 8. A_SCOPE：真实轨迹定位后才限制历史位置

### 8.1 必须存在的操作依据

仅当剂量阶段后仍有本轮 Validation/实际更新诊断支持的格代价，且在所选 H* 的真实训练轨迹上定位到 STAR_SKIP 或 EXPERT 的历史项造成排序损害，才进入作用组试验。旧 P01 Test 的损害不能直接继承成 A40 或 A22 的已测损害。

合格定位依据为：在五 seed、预注册锚点的汇总中，将某组历史更新替换为同轨迹普通 Adam，能改善本轮观察到的受损格且带来正 Macro 排序效用；同组等更新尺度控制后仍有方向作用的证据。梯度负余弦或高方差不够。若只减小全组步长就解释效果，记录尺度解释，不增加条件组限制来包装机制。

定位针对实际存在的训练问题；无真实损害、无可操作位置或固定 EMA 已充分消除该动作的可改进部分时，停止范围改进。该停止不抹除 H* 已有宏收益。

### 8.2 只运行两个预注册作用域

| ID | 历史作用位置 | 其余参数 | 数据与 seeds | 新完整训练 |
|---|---|---|---|---:|
| A50_STAR_SKIP_ONLY | 仅 STAR_SKIP 使用已冻结 H* 历史形式 | 完整 mixed loss 的普通 β1=.9 Adam | ML/KR × 2022–2026 | 10 |
| A51_EXPERT_ONLY | 仅 EXPERT 使用已冻结 H* 历史形式 | 完整 mixed loss 的普通 β1=.9 Adam | ML/KR × 2022–2026 | 10 |

两个作用域都从共同初始化开始完整训练，λ=λ*，PAIR 仍更新原可达参数；不是删掉其余参数上的辅助监督，也不是改变模型共享结构。不按单 seed 或单锚点只保留一个臂。H*=A22 时，作用域指 β1=.99 的长历史位置，不能称作“24 条件表范围”；H*=A21 时只在作用域保留 mR+Q。

两数据可报告不同作用域的条件性结果；统一最终主方法仍冻结一个作用域并用它在两数据已有的五 seed 结果完整报告，不能将各自最优作用域拼成同一无条件跨数据算法主张。

主比较为各候选对 A00、同 λ 普通 Adam、同 λ 全 S_dense 的 H*。更小状态量可以决定近似表现时的选择，不能代替相对正常强对照的收益。限制作用域只改善一个退化配置而未超过 A00 时，结束本版；历史独立增量消失时采用简单对照，不继续加门控、惩罚项、任务权重或更细组搜索。

## 9. A_ACTUAL_TRAJECTORY：不反馈的历史分解与因果探针

### 9.1 状态重建、锚点与评价面板

原 P01 完整训练状态已释放，端点表或单独模型权重不能恢复 m/v/H、活动时钟与 RNG。本阶段优先使用新候选训练期间直接流式捕获的状态；只有问题必须在已释放的 H* 轨迹上回答时，才做同源 STATE_REPLAY。

最多 ML/KR 各五 seed，即 10 条状态重放，不新增科学 seed。锚点统一为 E5/E20/E35/E50 的最后一个真实 batch 更新前，与方向 B 共用 40 个锚点标识和固定评价面板。无需另开 E10/30/50 面板。每个锚点包含：

- 模型 θ、EMA φ、完整普通 Adam 状态及原生 None mask；
- H/Q 或 S99 m、真实 mR、v、每参数 step、活动/抽中/pair clocks、动态 D（如适用）；
- 当前 batch/row ID、固定 pair 抽样结果、loader generator、全局 RNG、global_step、选点状态；
- 从初始化起维护的只读 shadow，以及相应参数和模型身份 hash。

不把 P00、clean 或其他 seed 的 m/v 移植到 P01 来伪装其真实状态。B 的 clean 状态可以提供共同位置与评价流程，不能充当 P01 自身轨迹的历史归因。

shadow 必须从初始化开始累计，不能到锚点才置零。ECCM/COLL 原步已经计算一次独立辅助 autograd；它们的 shadow 主要增加张量运算。S99 为获得原始 r_s 的六格历史分解，需要每个有效 pair 步额外一次共享参数 autograd，两数据五 seed 单个配置最多 17250 次，按实际 autograd_sweeps 单列。新历史候选可在训练开始时预挂观察器以免重放；没有预挂而后续确需完整历史时，使用本阶段既定重放额度，不伪造状态。

模型/optimizer/训练张量只存 RAM；磁盘写来源身份、标量和评价记录。候选训练时预先接入所需只读诊断，能免重放则免重放；不为了诊断形式完整无条件重跑十条。不同候选不会各分配十条 A 状态重放，A 专用重放总上限是 10。

### 9.2 ECCM/COLL 相对同轨迹普通 Adam 的精确分解

对六格 c，在候选自己的真实轨迹维护不反馈的普通 Adam 辅助一阶 shadow：

$$
M_{c,p}\leftarrow
.9M_{c,p}+.1\lambda(w_s/q_s)r_{s,p}\mathbf1[c(s)=c].
$$

每次参数实际推进时所有六个 M 都衰减；无 pair 步也衰减，只是输入为 0。按该轨迹相同 g₀、WD、参数时钟维护的普通 Adam 一阶状态恰为：

$$
m_p^{shadow}=m_p^R+\sum_c M_{c,p},\qquad
N_p^{shadow}=\frac{m_p^R+\sum_cM_{c,p}}{1-.9^{\tau_p}}.
$$

ECCM 的六格历史分子为：

$$
E_{c,p}=\lambda\sum_{\substack{i\in A\\c(i)=c}}
\frac{w_iH_{i,p}}{1-.99^{n_i}},\qquad
\Delta N_{c,p}=E_{c,p}-\frac{M_{c,p}}{1-.9^{\tau_p}}.
$$

COLL 额外维护六个只读 Q_c：有效 pair 步全部按 .99 衰减，仅选中来源加 .01(w_s/q_s)r_s；空 pair 步均冻结。验证 Σ_cQ_c=Q，使用 E_c=λQ_c/(1−.99^{nΣ})。空 pair 步 ECCM/COLL 的 E_c 都设为 0，仍保留已经衰减的普通 Adam M_c，正确表达“不注入历史”与普通 Adam 继续携带旧辅助动量的差别。

令 d_p=√[v_p/(1−.999^{τ_p})]+eps，取候选本步真实更新后的同一个 v：

$$
\Delta\theta_p^{H}-\Delta\theta_p^{shadow}
=-\eta\,\frac{\sum_c\Delta N_{c,p}}{d_p}.
$$

该等式是同候选轨迹、同当前分母下的精确分解；不是两条已分叉完整训练的 P01−P00 更新分解。它隔离一阶历史结构，保留当步混合二阶矩和 WD，不捏造另一条轨迹的 optimizer 历史。

### 9.3 S99 必须保留真实/WD 残差

若 H*=A22，不能使用“实际 mR 与普通 Adam 相同”的假设。对 β∈{.9,.99}，在同一 S99 轨迹维护：

$$
R_p^\beta\leftarrow \beta R_p^\beta+(1-\beta)(g_{0,p}+\omega\theta_p),
$$
$$
M_{c,p}^\beta\leftarrow \beta M_{c,p}^\beta+
(1-\beta)\lambda(w_s/q_s)r_{s,p}\mathbf1[c(s)=c].
$$

核验实际 m99=R99+ΣM99。再定义：

$$
\Delta N_{real,p}=
\frac{R_p^{.99}}{1-.99^{\tau_p}}
-\frac{R_p^{.9}}{1-.9^{\tau_p}},
\qquad
\Delta N_{c,p}=
\frac{M_{c,p}^{.99}}{1-.99^{\tau_p}}
-\frac{M_{c,p}^{.9}}{1-.9^{\tau_p}}.
$$

于是 N99−Nshadow9=ΔNreal+Σ_cΔNc。real 项包含真实数据及 coupled WD 时间尺度差，必须单独报告。移除整组长历史时同时替换 real 和全部 pair 残差；仅替换某格辅助残差时明确其只是局部机制诊断。不能把所有 S99−Adam 差异都记到 PAIR 条件历史。

### 9.4 固定动作、尺度控制和可解释范围

在每个锚点克隆同一前态，使用相同真实 batch/pair，比较：

1. 原 H* 更新。
2. STAR_SKIP 或 EXPERT 的全部历史残差替换为同轨迹 β1=.9 shadow。
3. 在上述组内分别替换一个六格来源的辅助历史残差；S99 的 real 残差单独保留/替换并标注。
4. 对 2/3 的同一参数组构造更新尺度匹配控制。

未干预参数取原更新；lr、WD、共同 v、EMA 起点和评价样本相同。由这些一次性替换得到的参数与 EMA 都只用于评价，不反馈 live 训练。

对组 G：

$$
\operatorname{RMS}_G(u)=
\sqrt{\frac{\sum_{p\in G}\|u_p\|_2^2}
{\sum_{p\in G}\operatorname{numel}(p)}},\quad
u^{match}_{alt,G}=u_{alt,G}
\frac{\operatorname{RMS}_G(u_{ref,G})}
{\operatorname{RMS}_G(u_{alt,G})}.
$$

这是整组实际参数更新的等 L2/RMS 尺度控制，不逐 tensor 缩放，不对训练梯度做 RMS。两者均 0 时无需控制；alt 为 0 而 ref 非 0 时记不可匹配，禁止补随机方向。ref 为 0 而 alt 非 0 时匹配后为 0，原始未匹配结果仍保留。

评价固定面板的六格真实 AUC、固定正负对排序翻转、Macro、LL；更新范数和余弦作为解释量。主方向与强弱判断用同一未改变人口的 Validation 面板，训练面板用于可部署信息及桥接，协议遵循方向 B 与统计文件。一次 EMA 分支差仅为 (1−ρ) 倍在线参数差，不能单独称为“选择性过滤有害任务”。

无需为 A 另开无限短段反馈搜索：定位足以决定两个预注册作用域是否值得完整训练；A50/A51 才检验长期政策。冻结同分母干预本身不证明后续所有 v、梯度和预测都相同。

## 10. 新接口合约与旧代码边界

### 10.1 真实现有入口

原模块 experiments/conditional_information_momentum_20261102_v3/ 中已有：

- boundary_sampler.py：CONDITIONS、boundaries、sample、forward_logits。
- objectives.py：train_step(model, optimizer, ema, x, y, row_ids, config, context, diagnostic=False)。
- optimizers.py：shared_name、PairOptimizer、state_dict/load_state_dict。
- protocol.py：config、make_optimizer、load_dataset 及 EMA/身份工具。
- run.py：job(dataset, arm, seed, stage, split_id="original", lambda_pair=0.) 及旧 worker/schedule/freeze/evaluate。

原 config 只允许 λ∈{0,.05,.2}，原 PCOLL 默认动态，原 scheduler 有旧种子与生命周期约束。**不修改旧脚本或旧结果来容纳新研究。** 本轮新增包为 experiments/conditional_history_shared_update_20261122/，正式函数和类型由 [工程规范](06_IMPLEMENTATION_SPEC.md) 实现；以下全标为 NEWAPI，不暗示现有仓库已经具备。

### 10.2 NEWAPI 的 A 侧语义

| NEWAPI 职责 | 必须接受的输入 | 必须输出或保证的结果 |
|---|---|---|
| A 配置工厂 | experiment_id、dataset、seed、history_kind、history_scope、λ、source/data hash | 不可变配置；A21 静态、A22 分组 β、A40/A41 .025 合法；原字段完整继承 |
| 参数 dispatch | named_parameters、冻结 scope、history_kind | 每参数唯一 optimizer 路径；覆盖清单、形状、参数数量；EMBED 不进入历史 |
| 只读梯度观察器 | live mixed grad、原始 pair grad、WD、τ、active/q/w、来源 c | 从初始化起更新 shadow；不修改 live tensor、顺序或 RNG |
| 状态探针 | 原生完整 RAM snapshot、当前 batch/pair、固定动作、固定面板 | 精确分解误差、原/反事实排序结果、范数匹配状态和身份 |
| A 阶段分析 | 两数据五 seed Validation endpoints、复用清单 | A_GATE、H*、λ*、scope 冻结及必要新任务；不读取 Test |
| 最终冻结 | 全部任务状态、selected epoch/hash、匹配对照、评价脚本版本 | 全局 FINAL_FREEZE 后才允许一次最终 Test 评价 |

原 PairOptimizer 的外层接口同时暴露 demand、active、selected、pair_clock、mode 和 step(pair,r)。新 A22 或 scope adapter 若复用原 objective，必须完整实现这些调用契约，不能只返回一个裸 torch.optim.Adam 后静默丢失日志或 pair 参数。诊断模式需要 r 时可增加只读 autograd，但不得更改训练用 gradient accumulation 或随机轨迹。

可规划的命令入口如下；它们属于 NEWAPI，须先实现及独立审查：

~~~bash
PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
PLAN=research_plans/26-11-22_conditional_history_shared_update
$PY -m experiments.conditional_history_shared_update_20261122.cli run --plan "$PLAN" --stage A_CORE
$PY -m experiments.conditional_history_shared_update_20261122.cli analyze --plan "$PLAN" --stage A_CORE
$PY -m experiments.conditional_history_shared_update_20261122.cli advance --plan "$PLAN"
~~~

advance 只展开本目录已定义阶段和有限矩阵，不自行新增 β、λ、条件门控或动态 ECCM。运行继续遵循当前会话监督与 RAM_ONLY；资源准入和故障恢复见 [资源规范](08_RESOURCES_AND_SCHEDULER.md)。

## 11. 有限预算与不运行的项目

| 阶段 | 新科学训练 | 条件 | 解决的关键决定 |
|---|---:|---|---|
| A_CORE | 12 | 必做 | 监督、历史包与动态抽样的独立增量 |
| A21+A22 | 20 | 任一数据 H>0 | 普通/单份/两通道/条件表择优 |
| A40+A41 | 20 | 有强对照增量且存在已登记保护研究问题 | 降 λ 的增量是否仍来自历史 |
| A50+A51 | 20 | 剂量后仍有当前损害并获真实轨迹定位 | 限制历史位置是否长期有效 |
| A 实际轨迹重建 | ≤10 | 状态确已丢失且当前决定需要 | 恢复机制所需 m/v/history/RNG，不新增 n |

方向 A 新科学训练上界 **72 条**，共 3600 epoch、124200 次 optimizer 更新；A 专用重放最多再增加 10 条、500 epoch、17250 次更新。A 最大物理量为 **82 条**；B 的十条 clean 重建单独登记，不重复计入 A。默认只启动 A_CORE 的 12 条；其余 NOT_TRIGGERED 是执行状态，不是负结果。

下列项目没有当前决策价值，保持不运行：

- 重复 A00/P01 的五 seed 总收益确认，或把数值复用与状态重建计成新重复。
- 默认补 P11；已有动态抽样交互线索不支持强制组合，本轮 A30 足以判断动态监督自身。
- PCV 正式训练、全 β 网格、更多 λ、额外种子池、更多模型或数据集。
- RMS 训练配平、把 EMBED 直接加入 ECCM、创建可学习 FUSION、按旧 Test 坏格加任意门控。
- 没有真实损害便先做保护结构，或先削弱 A00 再以修复量主张算法成功。

## 12. 最小正确性审查与阶段交付

继承原源码已覆盖的采样、None/0、WD 和恢复测试；不为未改的科学内核重做整套历史试验。新增实现只补会改变结论的检查：

1. 静态臂的 batch/condition/pair_index_hash 匹配；λ=0 与原 A00 数值和 RNG 等价。
2. A21 在显式相同静态配置下复现原 collapsed 内核；A22 在 S_dense 上与原生 Adam β1=.99 一致，其他参数 β1=.9。
3. .025 配置只改变 λ；A40/A41 的采样、WD、参数覆盖、EMA、预算和选点匹配。
4. 条件活动/抽中/pair/参数时钟、空 pair、显式零和 None 语义正确；WD 仅一次。
5. shadow 求和精确重构同轨迹 mixed Adam；ECCM/COLL 六格残差、S99 real+六格残差重构实际更新差。
6. A50/A51 参数分组互斥完整，非作用组恢复普通 Adam；每参数一次 step，没有旧历史误注入。
7. RAM roundtrip 恢复模型、EMA、全部 optimizer/history/shadow、时钟、D、loader/RNG 与选点；下一步数值与 EMA 一致。禁止用磁盘检查点替代 RAM。
8. 阶段函数只读完整五 seed Validation；未齐返回未完成，阈值用未舍入数值。历史 Test 仅出现在预登记动机/背景元数据，不能进入选参或组选择函数。

每阶段交付：运行/复用身份、逐 seed 宏与六格、匹配差值、简化选择、接受结论、适用边界、进入或停止的明确理由。A_ACTUAL_TRAJECTORY 另报 exact reconstruction 误差、None/时钟合约、尺度控制及同轨迹估计对象。

最终先冻结方法、λ、scope、主/组件比较与 selected 模型，再统一 Test。旧已暴露结果始终标历史探索；新训练在同一历史 Test 人口上的冻结评价也不声称是新的数据人口确认。最终 +.002/+.003 与单格 −.001 资格按统一规范判断，机制发现本身不必达到这些完整算法阈值。

A 的结案必须明确落入可执行决定：保留条件历史；采用更简单历史；只保留条件监督；或停止当前方案。无论哪一种都完成报告，不强制设计新算法，不自动扩张搜索。
