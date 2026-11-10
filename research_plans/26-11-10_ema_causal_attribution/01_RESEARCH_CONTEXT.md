# 01｜已有结论、剩余问题与本次研究边界

状态：**DESIGN_ONLY**。本文件建立本次规划的证据起点，不报告尚未运行的新实验结果。总执行入口见 [00_EXECUTION_RUNBOOK.md](00_EXECUTION_RUNBOOK.md)，不可变来源与资产审计见 [10_SOURCE_EVIDENCE.md](10_SOURCE_EVIDENCE.md)。

## 1. 当前结论与研究目标

**固定 EMA 的 AUC 收益及其早期学习滞后已经成立；本次要识别的是完整 EMA 修正通过哪些参数、哪些历史更新以及哪些模型计算环节，纠正了哪些真实正负排序。** 过去的局部时间尺度脉冲主要回答“稍微改变 EMA 快慢是否有用”，没有完成“固定 EMA 相对在线模型的全部修正为什么有用”的因果归因。因此，本次可以直接以固定 EMA 为强内部对照，进入参数修正与函数变化的机制研究。[来源：原机制总报告](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)

最终解释须连接五层证据：EMA 的数学作用、可重复的排序现象、真实参数干预效应、模型函数的变化路径、原在线训练产生这些更新的来源。不同层次分别判断；不因训练来源尚不能唯一拆开，就撤销已经识别的参数干预和功能作用。

本轮采用 MovieLens、KuaiRand × Clean、Gaussian D0 × seeds 2022–2026，共 20 条 50 epoch 轨迹。Clean 是主要机制发现条件，Gaussian D0 是预先确定的跨训练目标边界条件。所有归因优先共享这些轨迹；读出、参数替换、历史分解、功能探针均不反馈到在线训练。本次文档交付不实现新代码、不运行训练。

## 2. 已审计仓库状态与来源优先级

规划审计时 main 为 a646c5a29b74d4f3d731a8ae2c13038b06f5250f，已阅读：

- [EXP_AI_CONTEXT.md](https://github.com/Caelestis1903/My-Research-Project/blob/a646c5a29b74d4f3d731a8ae2c13038b06f5250f/EXP_AI_CONTEXT.md)。
- [experiments/Experimental_Global_Summary.md](https://github.com/Caelestis1903/My-Research-Project/blob/a646c5a29b74d4f3d731a8ae2c13038b06f5250f/experiments/Experimental_Global_Summary.md)。
- [MDMT_ALGORITHM_METRICS_LEDGER.md](https://github.com/Caelestis1903/My-Research-Project/blob/a646c5a29b74d4f3d731a8ae2c13038b06f5250f/MDMT_ALGORITHM_METRICS_LEDGER.md)。

main 已归档核心研究结论，不能据此假定来源分支的训练实现、原数表和 RAM 状态也已并入 main。实施时应依据 [00](00_EXECUTION_RUNBOOK.md) 从固定来源建立隔离 worktree，复用已审计实现；不得为取得源码把整个实验分支合并进 main。原报告和原产物保持只读。main 的旧阶段说明、旧计划提出的可选后续实验，不会自动扩大本轮任务。

| 研究 | 不可变来源 commit | 本轮用途 |
|---|---|---|
| AUG_HISTORY，分支 26-11-06 | 40705dfe0bb4fe614bdca835c0bae567348c87d0 | 方向响应、路径分组时间尺度与历史漂移增强的已测设计 |
| EMA_MECHANISM，分支 26-11-07 | ffa3de067fcda5a1293af83a465d741e069493f1 | **主要实施来源**；训练、EMA、支持审计、预测读出、RAM 生命周期 |
| EMA_ADAPTIVE，分支 26-11-09 | 1391affa5f08556c79b4cb2bfad15228a8b9129f | 训练收益持续性、外推失败与动作价值的历史证据 |
| PUB-MDMT，分支 26-11-09_v2 | a5f461de714232a416a2e95ca6fdc57fa7ce7ae8 | 代理效用、实际更新与排序收益必须区分的参考 |

## 3. 已有结论与剩余问题清单

### 3.1 直接接受，不重新设实验确认的结论

| 已有结论 | 核心证据 | 本轮处理 |
|---|---|---|
| 固定 EMA 在两数据、两目标均有可靠排序收益 | 五 seed 独立选点后的 F−online Test Macro AUC 均值均为正，四条件见下表 | 接受；O/F 同步读出仅用于定位和归因，不是重新立项证明有效 |
| 收益超出单纯 checkpoint 延迟 | 晚期 F 相对理论年龄匹配 online 的 Validation AUC 四条件均有正增量 | 不重做年龄匹配网格；研究历史组合如何改变函数 |
| 早期有学习滞后，晚期排序更稳定 | E1–10 的 F−online 约 ML −0.003、KR −0.0203/−0.0206；早期同样可以更稳定 | 稳定性只作描述或预测特征，不直接命名为收益原因 |
| 已试复杂自适应没有可靠额外组件收益 | AUG_HISTORY B/A/C、11/06 M0–M3 和 11/09 A0 结果相互约束 | 旧设计不换名重跑；先建立收益修正的机制链条 |
| 收益预测与实际动作价值不同 | 数值 MAE 改善、方向持续性或当前风险预测，可以伴随动作收益近零或为负 | H1/H2/H3 同时检查真实动作、简单固定参照与信号迁移 |
| AUC 与 LogLoss 不必同向 | ML F 的 selected Test AUC 提升同时 LL 增大；KR 依目标变化 | 分开评价，不互相覆盖 |
| Gaussian × EMA 交互有条件性 | 阶段、Validation/Test 和选点方式可改变交互方向 | Gaussian 是已冻结的边界条件，不预设统一协同 |

前五项来源为 [11/06 原机制报告](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)、[AUG_HISTORY 报告](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/REPORT.md) 和 [11/09 自适应报告](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md)。

下表从原始 ENDPOINTS.csv 的 split=test、checkpoint=selected 按 seed 配对重算。online 和 F **分别按各自完整 Validation 曲线选点**，因此这些是最终性能结果，不是同一时刻的机制干预。MovieLens 的 F 与 H2 共用一个 physical readout，须通过 aliases 识别，不能重复计数。

| 条件 | 平均 ΔMacro AUC | 配对差样本 SD | AUC 正向 seed | 平均 ΔMacro LL |
|---|---:|---:|---:|---:|
| MovieLens Clean | +0.003848845 | 0.000374396 | 5/5 | +0.001248644 |
| MovieLens Gaussian D0 | +0.003856802 | 0.000645238 | 5/5 | +0.005334282 |
| KuaiRand Clean | +0.004616399 | 0.003048773 | 4/5 | −0.001225342 |
| KuaiRand Gaussian D0 | +0.007766927 | 0.004257500 | 5/5 | +0.000327687 |

四条件的各域任务格五 seed 平均 AUC 增量均为正。[原始逐 seed 端点](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/ENDPOINTS.csv)

### 3.2 已削弱或排除的解释

| 解释 | 当前裁定 | 适用边界 |
|---|---|---|
| EMA 只是拿到了较旧 checkpoint | 已被年龄匹配结果排除为充分解释 | 不等于已经说明参数平均的功能机制 |
| 只要减少排序波动就会提高 AUC | 已被早期“更稳但更差”削弱 | 低波动仍可作描述量 |
| 增强响应方向、分组快慢或漂移统计一旦非零，就应有额外收益 | 已被对应真实启用而无增量的结果否定 | 未否定所有不同方向或不同动作 |
| 复杂指标预测更准，即足以支持复杂控制器 | 已被误差与动作分离的结果否定 | 需要检验实际选择是否改变和是否胜过简单策略 |
| 当前收益完全瞬时、没有持续性 | 11/09 训练方向持续性提供反证 | 训练持续性向验证迁移明显减弱，不能据此直接部署 |
| PUB 的 C1/C2 比 C0 好，说明发现可用的 MDMT 优化机制 | 不成立；三者仍严重低于普通 Adam 和 F | 保留内部比较事实，但不据此扩大本次任务 |

这些裁定只覆盖已测配置与对应命题。没有结论支持把所有自适应 EMA、所有动量修改或所有参数组差异统一判为无效。[机制报告](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/FINAL_REPORT.md)；[自适应报告](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md)；[PUB 报告](https://github.com/Caelestis1903/My-Research-Project/blob/a5f461de714232a416a2e95ca6fdc57fa7ce7ae8/research_reports/2026-11-09_pub_mdmt/FINAL_REPORT.md)

### 3.3 本次必须收敛的开放问题

1. 完整 $R_t=\bar\theta_t-\theta_t$ 纠正与破坏了哪些真实正负排序；净变化能否精确还原六格 AUC 增量？
2. S 与非 S 的单独作用和交互是什么；主要作用是否需要继续细分至特定参数子类？
3. $R_t$ 的近期与更早更新成分各自怎样改变排序；二者在不同参数组和阶段是否存在功能互补？
4. 同时刻、同权重历史预测组合能否再现参数 EMA 的排序修正；二者的差异出现在什么模型计算环节？
5. 重要修正是抵消了实际观察到的近期不利更新、保留了有用历史，还是依赖表示与下游共同变化形成的函数状态？
6. 旧指标是否错过主要收益源，旧 $\rho$ 动作是否表达不了有益修正，或当前可操作空间本来就小？哪些已有证据足以决定 A/B/C/D 研究方向？

“所有参数唯一微观贡献”“所有可能优化器机制”“跨所有模型的普适性”不属于本轮完成条件。只在竞争解释会改变 A/B/C/D 决策时追加预先限定的来源探针。

## 4. 历史自适应设计究竟改变了什么

### 4.1 公式与观察对象登记

以下 $\phi$ 为某个候选自己的历史状态，$\theta$ 为共同在线模型，$U(g)$ 为同一个旧 Adam 状态下的真实参数存储位移。

| 历史研究 | 实际指标或构造 | 实际递推 / 干预 | 已测结果与本轮用途 |
|---|---|---|---|
| AUG_HISTORY B，方向响应 | $d_t=U(g)-U(g-a_t)$，对两个任务的增强响应构造正交基 $Q$；合并响应和固定符号打散作控制 | $\phi'=\rho_F\phi+(1-\rho_F)\theta+(\rho_F-\rho_A)QQ^\top(\theta-\phi)$，$\rho_A=\rho_F^{1/h}$，$h=.5/2$，刷新 K10/20 steps | D−F 的 Test AUC 仅 ML +0.000000104、KR −0.000004946。方向启用已审计，但旧日志未记录 innovation 投影能量；本次可核对实际作用覆盖，不能直接判成指标完全无信息 |
| AUG_HISTORY A，路径时间尺度 | 12 组或全局 q；以训练正负对 softplus 排序损失更新 q | $\rho_b(q)=\rho_F^{2^{-q_b}}$；虚拟 $\psi_b=\rho_b^K\phi_b+(1-\rho_b^K)\theta_b$；$q'\!=\mathrm{clip}(q-\eta g/\max(\lVert g\rVert_\infty,10^{-12}),-1,1)$；再执行一次分组 EMA | GROUP/GLOBAL 的 selected Test 均未稳定超过 F；早期优势不必转成 selected 优势。虚拟 K 步假设与实际本步动作不同，不应误写为真实未来 K 步的直接收益优化 |
| AUG_HISTORY C，漂移条件增强 | 输入扰动导数 $u=\partial_\delta[z_\theta-z_\phi]$；归一化后 $A=\mathrm{mean}(\hat u\hat u^\top)$，$C=16e^{\gamma A}/\mathrm{tr}(e^{\gamma A})$ | 用 $C^{1/2}$ 塑形 Gaussian 噪声；online 训练改变，teacher 的 $\rho_F$ 固定 | 仅 ML 运行，任务、合并、旋转三控制均值未胜 F。不是固定 online 上的纯 EMA 干预，不能用于替代完整 R 归因 |
| 11/06 EMA_MECHANISM，单组 pulse | 几何、函数幅度、完整/截断功能梯度、真实训练面板 AUC/rank risk | 从窗口起点 F 出发，持续一 epoch 使用 Fast=$\rho_F^2$ 或 Slow=$\sqrt{\rho_F}$，终点只把指定组换入 F 背景 | 回答局部快慢动作价值；没有对完整 F−online 的参数组与历史成分作因果分解 |
| 11/09 EMA_ADAPTIVE，A0 | C_persist 在身份/阶段上增加当前 Fast−F、Slow−F；C_carry 直接沿用当前收益 | 复用每隔五 epoch 观测的一 epoch S pulse；预测下一观测窗口，不形成递归控制训练 | 新训练/Test 均为零；指定拟合器无稳定增量，task 静态策略退化为统一 Fast；不能写成已运行完整自适应模型失败 |

公式来源分别为 [B 实际代码](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/aug_history_20261105/directed_ema.py)、[A 实际代码](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/aug_history_20261105/grouped_ema.py)、[C 设计与接口](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/DESIGN_SPECIFICATION_C.md)、[11/06 解析协议](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/RESOLVED_PROTOCOL.json) 和 [11/09 实际分析代码](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/experiments/ema_adaptive_20261109/analysis.py)。

### 4.2 必须保留的具体失败条件

**方向动作作用很小，与方向信息无用不是同一命题。** B 的 D/M/S 与 F 几乎重合，且响应构造真实启用。其历史日志没有 innovation 投影能量，旧 Q 状态也已释放。本次依据已识别的参数组、历史时段和功能作用，判断旧方向设计是否对应主要收益来源；不虚构无法从归档恢复的数值投影，也不为补齐这一个历史解释重跑 B 网格。旧响应的精确功能覆盖仍可保留为有边界的开放项。[AUG_HISTORY 报告](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/REPORT.md)

**完整函数导数与训练反传导数不同。** 原 M3oE 的 gate 输入使用 h.detach()；A 的 q 探针保留这一语义，而真实参数替换会改变 h 以及 gate 输入值。此实现事实不能单独解释 A 的失败，但要求本次功能积分使用 scratch 上前向数值相同的 full_functional_graph=True，再以实际端点预测检验闭合；不能把训练截断梯度当作完整参数到预测的导数。[A 设计](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/DESIGN_SPECIFICATION_A.md)；[已存在的完整功能读出](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/experiments/ema_mechanism_20261106/readouts.py)

**11/09 的失败有具体外推与动作退化证据。** KR Gaussian 晚期训练侧 Fast−Slow 方向一致率 0.96，验证侧 0.60；指定 C_persist 对未来 Fast/Slow 的预测均值为 −0.006196/+0.004019，而训练审计侧真实未来均值为 +0.000498912/−0.000421092，发生方向和尺度外推偏差。C_carry 的验证动作收益为 +0.000101319，却比统一 Fast 低 0.000000478；所有主条件静态 task 策略均选 Fast。因此不能把失败重新概括成“收益没有持续性”，也不能把高方向率直接写成“动态控制可用”。[11/09 总报告](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md)；[静态动作原表](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/artifacts/ema_adaptive_20261109/attempt1/A0_STATIC_POLICIES.csv)

## 5. 本次比 11 月 6 日新增什么识别能力

| 识别问题 | 11/06 已完成 | 本次新增 |
|---|---|---|
| 固定 EMA 的总排序收益 | O/F 选点、阶段、年龄匹配、排序稳定性 | 不重证；精确建立 O→F 正负排序 3×3 转移账本，净转移还原全量 AUC |
| 哪些参数产生主要收益 | 在 F 背景中替换 Fast/Slow 的某组 pulse | 对完整 $R=F-O$ 做 O、O+$R_S$、O+$R_N$、F 的必做二因素干预；直接估计两组组合交互 |
| 哪些历史产生收益 | 比较不同 $\rho$ 的局部机会和常驻读出 | 按 optimizer step 精确分解近期与更早 $R$，与主要参数组交叉；不把近期直接称为噪声 |
| 参数平均与预测组合 | 没有相同全部 step、相同权重的直接对照 | 固定面板从 step0 起在线维护概率与 logit 预测 EMA；参数 EMA 与历史预测平均同钟比较 |
| 参数如何经网络影响排序 | 梯度代理、函数幅度与训练面板直接效用 | 对完整表示切口做真实 activation 替换，拆分表示、下游和交互；必要时积分核对真实 margin 变化 |
| 原训练发生了什么 | 阶段观察与局部脉冲收益 | 沿同轨迹记录真实排序变化与目标变化，将近期不利更新造成的翻转和 R 修正逐对连接；仅有决策价值时执行有限同状态 Adam 来源探针 |
| 为什么自适应无额外收益 | 局部特征预测、动作价值、跨人口迁移 | 将已识别收益源与旧指标、有限 $\rho$ 动作菜单和机制修正菜单比较，区分 H1/H2/H3 |
| 深度非线性 | 局部脉冲交互及未触发的第二背景诊断 | 完整 S/非S 交互始终量化；重要函数切口的真实替换与端点预测闭合，不要求唯一参数份额 |

旧 C_BG 第二背景控制未触发，只描述旧 pulse 的触发条件；不构成完整 R 无交互的证据。旧 RANK_FLIPS 也不替代本次完整 O→F 的收益账本。[旧设计与触发规则](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/DESIGN_SPECIFICATION.md)

本次主要因果对象是**固定历史轨迹条件下的模型读出状态**。例如用 $O+R_{\mathrm{recent}}$ 预测，可以识别该修正对当前函数和 AUC 的作用；它不是“删除过去更新后重新训练”的因果效应，因为那会改变后续梯度和 Adam 状态。对训练来源的结论按证据范围命名，不把读出替换偷换成闭环训练改进。

## 6. PUB 参考边界

PUB 迁移实际使用的更新组合是：

$$
d_{\mathrm{PUB}}
=\sum_i\alpha_iU(g_i)
+U\!\left(\sum_i\alpha_i g_i+\lambda\theta\right)
-U\!\left(\sum_i\alpha_i g_i\right),
$$

再以 $\theta\leftarrow\theta-\eta d_{\mathrm{PUB}}$ 更新，m/v 使用带一次 coupled WD 的合并梯度更新。此处 $U$ 是 PUB 的 Adam 归一化方向，与第4节 AUG_HISTORY 的“实际存储位移”按各自实现符号区分。它改变了优化更新，而固定参数 EMA 是在线轨迹外的历史参数读出；两者不能混称。[PUB 冻结报告](https://github.com/Caelestis1903/My-Research-Project/blob/a5f461de714232a416a2e95ca6fdc57fa7ce7ae8/research_reports/2026-11-09_pub_mdmt/FINAL_REPORT.md)

在原始 $\alpha$ 尺度与 coupled WD 设定下，ML C0/C1/C2 相对普通 Adam 的 AUC 为 −0.217376/−0.172438/−0.137374，KR 为 −0.099142/−0.048111/−0.052118，全部六格均值下降；求解器实际无失败和回退。C−G 或 C1/C2−C0 的优势保留为这些内部比较的事实，不解释成已经找到有用的新 MDMT 收益机制。[同一报告的主结果与诊断](https://github.com/Caelestis1903/My-Research-Project/blob/a5f461de714232a416a2e95ca6fdc57fa7ce7ae8/research_reports/2026-11-09_pub_mdmt/FINAL_REPORT.md)

本次只吸收一项方法要求：**代理效用、实际参数步、模型函数变化和最终排序要逐层核对。** 不优化 PUB 求解器、不修改其半径/WD、不追加 PUB+EMA，不把历史报告建议的修复实验放入本矩阵。

## 7. 固定协议与不可改写的研究边界

- 所有正式条件至少五 seed，2022–2026；同轨迹的读出、窗口、参数组和样本对都不增加独立 seed 数。
- 全量 Validation 六格等权 Macro AUC 的最早严格最大值选择 checkpoint；Test 在全局机制和配置冻结后评价。
- 主要算法晋级采用五 seed 平均 AUC 增量至少 +0.003；逐格平均下降不超过 0.001，违反时单独报告代价。参数消融和机制效应不套用 +0.003 晋级门。
- LogLoss 独立评价；Traffic、Pooled 与六格 Macro 分开命名。
- O/F 独立选点的最终性能比较，与同刻 O/F/混合参数的机制比较分表。F 选中时刻的成对状态标为 matched-at-F-selected，不能冒充各干预独立选点。
- 固定 EMA 是内部强对照；正式论文 baseline 仍限已发表模型。本轮不重训三套发表模型。
- **小域不是困难域。** 不提出检验或反驳该结论的假设，不以域样本量构造困难度。人口只用于数据身份、实际 step 数和资源预算。
- 不使用 Test 调整参数分组、窗口、指标、控制菜单或研究假设；不因某个 seed 或某个小面板暂时无效，停止其余 seed 的正式机制采集。
- 不预设近期变化是高频噪声，不预设较早历史有益，不把梯度冲突、过拟合或负迁移作为没有训练来源证据的结论。
- 旧模型状态已释放。本轮沿用模型/optimizer/EMA/参数派生向量只驻 RAM 的约束，按 [05](05_IMPLEMENTATION_SPEC.md)、[06](06_SCHEDULER_AND_RESOURCES.md) 进行边采集边归因及故障恢复，不将磁盘标量误作可恢复 checkpoint。

前六项与现行 [main 研究规则](https://github.com/Caelestis1903/My-Research-Project/blob/a646c5a29b74d4f3d731a8ae2c13038b06f5250f/EXP_AI_CONTEXT.md) 一致；状态释放和存储约束见 [释放收据](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/research_reports/2026-11-06_ema_mechanism/RELEASE_RECEIPT.json) 与 [执行环境规则](https://github.com/Caelestis1903/My-Research-Project/blob/ffa3de067fcda5a1293af83a465d741e069493f1/start.md)。

本轮完成时，允许得到“主要收益链条已识别，但若干微观训练来源未唯一分离”的结论；只要它能据真实干预和功能证据，明确支持下一阶段 A/B/C/D 的取舍，就应进入下一研究阶段，而不是无限追加验证。

