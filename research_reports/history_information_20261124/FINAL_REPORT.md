# 历史训练信息研究 — 来源二阶组织与历时预条件辅助历史

研究编号 `EXP082` · Study `history_information_20261124` · 状态 `COMPLETED` · 完成日期 2026-11-26（Asia/Shanghai）

证据基点 `358d3a465747eda6af1d55acc937759fa651a10b`。规划提交 `99a4dec1b3f2fa4eab80b03af4d6c682cc67a85b`。执行来源闭包 `4ae091233e81a9f79535b1113cab2f041de96bc4dec4c4aeadf4802d748b6346`。本轮训练 100 条轨迹，冻结后评价 160 个新 Test 端点并复用 20 个匹配的历史发表模型端点。Test 为 `historically_exposed_holdout`。 [执行审计](../../artifacts/history_information_20261124/AUDIT_RECEIPT.json) · [全局冻结](../../artifacts/history_information_20261124/FINAL_FREEZE.json) · [Test 收据](../../artifacts/history_information_20261124/TEST_RECEIPT.json)

## 1. 当前研究结论

固定参数 EMA 提供了本轮最稳定的排序收益。80 条冻结轨迹的同轨迹双读出在 Test 上有 77 条受益，MovieLens 和 KuaiRand 的平均 Macro AUC 增量分别为 +.004533 和 +.004816。C1 改变了来源耦合和单步排序关系，完整验证集结果未达到匹配控制的启动门。C2 在 MovieLens 的 fixed EMA 背景下超过长历史父方法和三项匹配控制的验证集均值门。到冻结 Test，C2 相对发表 M3oE online 达到 +.003177，相对同读出 PAIR Adam、原 A21 及尺度控制的 AUC 差分别为 −.000190、−.000107497、−.000124。KuaiRand 的验证集六格保护和匹配控制进一步限制了 C2 的应用范围。本轮保留固定 EMA 和 A21 的既有使用价值，两项新增定义以 `NO_ACTIONABLE_INCREMENT` 结案。 [冻结 Test 端点](../../artifacts/history_information_20261124/TEST_ENDPOINTS.csv) · [Test 配对汇总](../../artifacts/history_information_20261124/test/summary.csv) · [验证集决策](../../artifacts/history_information_20261124/S3d/DECISIONS.json)

| 研究问题 | 发现 | 证据强度 | 研究决策 |
| --- | --- | --- | --- |
| Q1 历史的有效信息与处理 | C1 改变共同二阶分母，C2 保留历史时刻的 PAIR 预条件方向。两项操作改变真实更新，完整训练的增量依赖数据、读出和参照 | 局部 **STRONG**，独立排名增量 **WEAK** | 关闭当前两种候选定义 |
| Q2 EMA 与在线历史 | 双读出稳定增益和训练阶段交互同时存在。KR 的 A21−PAIR 在早、晚阶段及两个评价人口上方向不同 | EMA **STRONG**，单一交互方向 **WEAK** | 保留被动 EMA，分别评价同阶段和部署选点 |
| Q3 多来源与共享学习 | 辅助来源的时间核、共享几何与域任务支持都影响排序。KR 的 d0、d1、d2 在切换操作后出现不同响应 | 实际支持与局部干预 **STRONG**，跨格解释 **MEDIUM** | 将占比与参照难度分别记录，后续以具体排序关系作为实验单位 |
| Q4 正式性能 | C2 fixed EMA 在 ML 通过发表比较的数值门，同读出增量门未通过。KR 固定发表参照 HyperGate 保持领先 | 门槛结案 **STRONG**，C2 的泛化增量 **WEAK** | 未产生 `Cstar`，保留冻结全部结果 |

## 2. 完成范围、配置与权限

| 阶段 | 实际轨迹 | 独立种子 | 工作及状态 |
| --- | ---: | ---: | --- |
| S1 | 30 | 每数据每臂 5 | B0、B1、H0 各 10 条父轨迹重建，完成 D1–D4。旧 A21 端点用于 D0 来源和验证集重分析 |
| S2 | 40 | 每数据每配置 5 | C1 和 C2 各两个非零配置，零点共用 S1 的 H0 |
| S3a | 10 | 每数据 5 | C2 的短历史 S_C2 |
| S3b | 20 | 每数据每臂 5 | C2 的组尺度 M_C2 和近邻 P_C2 |
| S3c / S3d | 0 | — | C12、N、HyperGate 迁移、Gaussian 组合均 `NOT_TRIGGERED` |
| 发表基线身份恢复 | 0 | 每数据每模型 5 端点 | M3oE 使用本轮 B0 online，HyperGate 与 AutoIFS 各复用 10 个锁定端点 |

MovieLens-1M 和 KuaiRand-Pure 沿用原始行序 80/10/10。统计行数见下表。词表和缩放参数在 Train 拟合，未知类别映射到 0。KuaiRand 按原行序切分后使用 tab 0/1/4 对应 d0/d1/d2。MovieLens 的年龄组分别为 `1/18`、`25`、`35/45/50/56`。任务分别是 `rating>3 / rating>4` 和 `click / long_view`。数据清单保存原 CSV、Train/Validation 输入数组、行 ID、编码器和缩放器哈希。Test 行数和新评价人口哈希来自一次冻结评估。 [数据清单](../../artifacts/history_information_20261124/DATA_MANIFEST.json) · [数据与实现合同](../../research_plans/26-11-24_history_information/07_IMPLEMENTATION_SPEC.md)

| 数据 | Train | Validation | Test | Train d0 / d1 / d2 |
| --- | ---: | ---: | ---: | --- |
| MovieLens | 800,167 | 100,021 | 100,021 | 155,949 / 318,599 / 325,619 |
| KuaiRand | 2,006,132 | 250,767 | 250,767 | 172,782 / 1,743,898 / 89,452 |

所有新增训练基于 M3oE fixed1，embedding 16，4 个共享专家，FCN `[512,256,256,64]`，保留门控输入 detach。种子为 2022–2026，50 epoch，batch 40,960，保留尾批，MovieLens 每轮 20 步，KuaiRand 每轮 49 步。FP32，AMP/TF32 关闭。HyperGate 的历史比较沿用原生目标、batch 1,024、50 epoch。AutoIFS 沿用 batch 4,096、10 epoch 搜索和 50 epoch reuse 阶段。 [M3oE 论文](https://doi.org/10.1145/3626772.3657686) · [固定发表模型清单](../../artifacts/history_information_20261124/PUBLISHED_ROSTER.json) · [来源登记](../../research_plans/26-11-24_history_information/11_SOURCE_AND_REUSE_REGISTER.md)

每条训练轨迹同时记录 online 与不反馈的 fixed EMA。EMA 每次成功更新后应用一次，MovieLens 的 $\rho=.9828205985452511$，KuaiRand 为 $.995$，buffer 直接复制。每个 seed 和读出各自在完整 Validation 上取最早达到最高六格 Macro AUC 的 epoch，LogLoss 绑定该选点。五个配对 seed 形成统计重复。Macro 为六格等权，Traffic 以自然流量加权，Pooled 先在每个任务内跨域合并。配对差报告样本 SD、df=4 的双侧 95% t 区间及方向数。 [统计口径](../../research_plans/26-11-24_history_information/09_METRICS_AND_STATISTICS.md)

用户授权冻结后一次 Test。2026-11-26 01:13（CST）完成全局冻结，160 个 selected 端点逐一读回并核对模型与 RAM packet 哈希。Test 从 01:15:10 到 01:18:25（CST），使用冻结配置和参照名单。历史暴露身份适用于对 Test 效应的解释，尤其是已登记的 A21 和发表基线比较。 [权限](../../artifacts/history_information_20261124/TEST_POLICY.json) · [访问凭证](../../artifacts/history_information_20261124/TEST_ACCESS.json) · [复用登记](../../artifacts/history_information_20261124/REUSE_REGISTER.csv)

## 3. 两个候选实际实现

B0 使用自然 BCE 和 Adam，B1 在相同 Adam 上增加静态 PAIR，H0 调用原 A21。自然目标平均批内两个任务的 BCE。每次活动事件按 $q_s=.25/|A|+.75w_s/\sum w$ 选一个注重自然质量的域任务联合标签边界，有放回采 2,048 对，使用 logit margin 的 softplus 排序损失，$\lambda=.05$。相同数据、种子与批号的静态臂共用采样随机流。历史修改限定在 `S_DENSE` 的 `STAR_SKIP` 与 `EXPERT`，其他参数用完整目标的原生 Adam。两数据集分别有 286,208 和 617,216 个 `S_DENSE` 参数坐标。 [算法规范](../../research_plans/26-11-24_history_information/05_CANDIDATE_ALGORITHMS.md) · [参数作用域清单](../../artifacts/history_information_20261124/PARAMETER_SCOPE_MANIFEST.json)

令 $b$ 为 BCE 梯度加一次 coupled WD，$h=(w/q)\nabla R$，$g=b+\lambda h$。$m_B$ 在参数步上以 .9 更新，$Q$ 在活动 PAIR 事件上以 $\beta_P$ 累积 $h$。lr=.01，$\beta_2=.999$，$\epsilon=10^{-8}$，WD 为 MovieLens $10^{-4}$、KuaiRand $10^{-5}$。$\tau$ 是原生参数步，$k$ 是辅助事件步。两个时钟分别做偏差修正。活动事件记为 $e$。

$$
V_{\zeta,t}=.999V_{\zeta,t-1}+.001[(1-\zeta)g_t^2+\zeta(b_t^2+\lambda^2h_t^2)],\qquad
D_{\zeta,t}=(\sqrt{V_{\zeta,t}/(1-.999^{\tau})}+\epsilon)^{-1}.
$$

$$
U_t\leftarrow\beta_P U_{t-1}+(1-\beta_P)D_{\zeta,t}h_t\quad(e=1),\qquad
\Delta\theta=-.01\{D_{\zeta,t}\widehat m_B+e\lambda[(1-\alpha)D_{\zeta,t}\widehat Q+\alpha\widehat U]\}.
$$

C1 固定 $\alpha=0$。$\zeta=.5/1$ 将来源平方和放进共同分母，降低 signed $2\lambda bh$ 的权重。交叉项含 BCE–PAIR 和 WD–PAIR。C2 固定 $\zeta=0$，$\alpha=.5/1$，将当步分母作用后的 $D_th_t$ 写入 $U$，读取 $\widehat U$ 后直接进入辅助方向。在 $e=0$ 时保持 Q/U 并省略辅助注入。$(0,0)$ 沿用原 H0 完整数值路径及被动六格 shadow。 [训练内核](../../experiments/history_information_20261124/objectives.py) · [优化器](../../experiments/history_information_20261124/optimizers.py)

| 实际臂 | $\zeta$ | $\alpha$ | 辅助 $\beta_P$ | 活跃状态或控制 |
| --- | ---: | ---: | ---: | --- |
| B0 / B1 | — | — | — | 原生 Adam 一阶、二阶矩 |
| H0 | 0 | 0 | .99 | $m_B,V_0,Q$ |
| C1_Z050 / C1_Z100 | .5 / 1 | 0 | .99 | $m_B,V_\zeta,Q$ |
| C2_A050 / C2_A100 | 0 | .5 / 1 | .99 | $m_B,V_0,Q,U$ |
| S_C2 | 0 | 1 | .9 | C2 的 Q/U 同步缩短事件核 |
| M_C2 | 0 | 1（影子目标） | .99 | 本轨迹 H0 方向与 C2 影子半径，逐组匹配并提交一步 |
| P_C2 | 0 | — | — | mixed 梯度先乘当步 $D_0$，再以 .9 累积按参数步更新的 $T$ |

LaProp 提供先预条件再累积动量的直接先例。C2 将操作限定在 PAIR 来源，P_C2 将同一操作用于完整 mixed 梯度，检验来源选择有没有额外价值。 [Adam 原文](https://arxiv.org/abs/1412.6980) · [LaProp 原文](https://arxiv.org/abs/2002.04839) · [本轮实现审查](IMPLEMENTATION_REVIEW.md) · [每轨迹配置哈希](trajectory_ledger.csv)

## 4. 默认完整训练与轻量搜索

四个非零配置均完成 50 epoch 和五种子。在线选型在 MovieLens 选择 C1_Z050、C2_A050，在 KuaiRand 选择 C1_Z050、C2_A100。C2_A100 的 MovieLens online 绝对均值略高，差值落入 $10^{-4}$ 的小参数优先带。KuaiRand 在线 C2_A100 对 B0/B1/H0 的均值均为正，d0t0 对三组参照均低于保护线。MovieLens fixed EMA C2_A100 对 B0/B1/H0 分别增益 +.000533、+.000451、+.000255，三组参照的六格保护均通过。该配置以 fixed EMA 触发两个数据集的 S/M/P。C1 用 online 网格代表完成冻结，其两个读出均保留。 [网格选型全表](../../artifacts/history_information_20261124/config_selection.csv) · [S2 决策](../../artifacts/history_information_20261124/S2/DECISIONS.json)

验证集所有配置的 Macro AUC/LogLoss、最差格差、每 seed E10/E25/E50 与 selected 数值见 [数值附录](NUMERICAL_APPENDIX.md#验证集网格和轨迹)。

## 5. 机制证据及竞争解释

D1 在 H0 的同一轨迹上分解了交叉项。两组的平均归一化 signed cross 为正，负坐标仍占约 17%–37%。BCE–PAIR RMS 约为 WD–PAIR RMS 的 28 至 8,404 倍，说明本设置中交叉项主要跟随 BCE/PAIR 的共同梯度。相对当前预条件方向的交换差在两个数据集均随训练减小。D2 的真实锚点显示，匹配组尺度会改变候选的局部排序纠正与破坏，两条候选在验证面板上的净变化接近零。D4 则呈现跨阶段的关系转移。完整结果的判定以五种子 full50 及 S/M/P 为依据。 [机制报告](MECHANISM_REPORT.md)

| 假设 | 判定 | 依据 |
| --- | --- | --- |
| H1 普通时间平滑 | **MEDIUM**，缩短 PAIR 核改变结果 | S_C2 的 fixed EMA 相对 C2 均值 ML −.000218、KR −.003538，未满足替代规则 |
| H2 辅助来源记忆 | **MEDIUM**，A21 在 KR 的 Test 排序收益依赖格与阶段 | H0−B1 Test EMA +.004062，5/5 正，验证集 −.001989，LogLoss +.000070 |
| H3 共同二阶耦合 | **STRONG** 支持局部作用，**WEAK** 支持新增排名收益 | 同轨迹矩恒等式闭合。C1 未越过验证集父参照门，KR Test EMA 相对 B1 的 +.003081 低于 H0 的 +.004062 |
| H4 历时预条件 | **STRONG** 支持方向变化，**WEAK** 支持普遍增益 | 交换差早高晚低。ML 后段有相对 H0 的收益，冻结 Test 上同读出均值返负 |
| H5 正确关系保持与修复 | **MEDIUM** 支持具体关系转移 | D4 中 KR H0−B1 的净纠正从 E10 +.001963 变为 E50 −.005090。固定关系样本与完整 AUC 分开量化 |
| H6 参数 EMA 交互 | 被动 EMA 收益 **STRONG**，稳定单向交互 **WEAK** | 冻结 Test 77/80 条 AUC 正，C2 与 B1 的交互符号随共同 epoch 变化 |
| H7 组尺度和近邻替代 | **MEDIUM**，均值分辨率下有覆盖 | ML Test EMA 的 C2−M 为 −.000124，C2−P 为 +.000051。KR 验证集两项均负 |
| H8 人口与选点 | **STRONG** 支持当前流程下的阶段和人口差异 | KR H0 的早晚轮次反转及 Validation/Test 反转在同一重建链复现。C2 的 ML EMA 发表比较与匹配比较也给出不同裁定 |

## 6. 完整匹配控制与方法收敛

S_C2 保留 C2 的历时预条件结构，改变 Q/U 的事件核。其两数据集等权验证集均差为 −.001878，KuaiRand 对 B1 的 d0t0 差为 −.010176，统一代表保留原 C2_A100。M_C2 使用本轨迹同步更新的 H0 与 C2 影子状态。在 17,250 个批次的 34,500 个组记录中，全部状态为 `MATCHED`，`UNMATCHED_ZERO` 为 0，实际半径与目标的最大绝对差为 $1.51\times10^{-7}$，FP32 位移舍入误差最大为 $9.32\times10^{-7}$。这个对照检验两个组的完整更新范数，组内坐标方向由 H0 保留。 [S3a 决策](../../artifacts/history_information_20261124/S3a/DECISIONS.json) · [M 实际位移校验](../../artifacts/history_information_20261124/S3b/M_ACTUAL_DISPLACEMENT_CHECK.json)

ML 的 fixed EMA 验证集上，C2 相对 S/M/P 的均差均超过 $10^{-4}$，五种子区间均跨零。KR 的 M/P 覆盖 C2，且 C2 对 B1 的 d0t0/d0t1 低于保护线。ML 冻结 Test 上，C2 相对 M 的 AUC −.000124、LogLoss −.002653（5/5 改善），相对 P 的 AUC +.000051、LogLoss −.000642。结果支持分别评价 C2 的排序和概率预测表现。区间与六格见 [控制配对表](NUMERICAL_APPENDIX.md#匹配控制与冻结对照) 和 [S3b 决策](../../artifacts/history_information_20261124/S3b/DECISIONS.json)。

## 7. EMA 交互与阶段关系

对固定候选与同一父参照，$I=\Delta_F-\Delta_O$，其中 $\Delta_O$ 和 $\Delta_F$ 分别为 online 与 fixed EMA 的候选差。各自最优 epoch 的交互描述部署选点，共同 E10/E25/E50 的交互描述同步训练阶段。以 B1 为参照，C2_A100 的 ML 部署选点交互为 −.000046，KR 为 −.001545，区间均跨零。KR 的同阶段均值在 E10/E25/E50 分别为 +.002749、−.002982、+.001331。ML 的 fixed EMA C2−H0 从 E10 −.000739 上升至 E50 +.000626（5/5 正），该末期结果与冻结 Test 的 −.000107497 对应不同端点。 [交互明细](NUMERICAL_APPENDIX.md#部署选点与同阶段交互) · [验证集配对数表](../../artifacts/history_information_20261124/S2/summary.csv)

## 8. MDMT 作用范围与条件迁移

| 模块 | 状态 | 决策依据 |
| --- | --- | --- |
| C12 | `NOT_TRIGGERED` | C1 在 S2 关闭，联合门要求两项原定义在同数据同读出均跨过完整控制 |
| N 非共享作用域 | `NOT_TRIGGERED` | 两数据六格保护与同一代表的控制增量共同决定 `Cstar`，本轮结果为空 |
| HyperGate 迁移 | `NOT_TRIGGERED` | 同一 clean `Cstar` 是迁移门的前提。原 HyperGate 继续作为发表比较 |
| Gaussian 组合 | `NOT_TRIGGERED` | 需要已合格 clean `Cstar` 与 fixed EMA 增量、两数据保护和 GA31-R3 身份。源身份已验证，科学门未触发 |

KR 的最大训练域 d1 占 86.93%，小域 d0 与 d2 分别占 8.61% 和 4.46%。以 B1 fixed EMA 为参照，H0 的 Test d0 两格分别 +.00695、+.01411，d1 为 −.00073、−.00111，d2 为 +.00363、+.00152。C2 在同一参照下使 d0 变为 −.00321、−.00626，d1 转正至 +.00163、+.00100，d2 提升 +.00552、+.00650。不同小域可以同时呈相反方向，域规模不足以解释这种分配。参照的分格 AUC 和正负样本支持描述基线难度，任务标签、训练阶段及被选的读出共同影响格结果。ML 在相同对照的 C2 Test EMA 下，d0 两格一正一负，d1t1 的下降最大，d2 两格也下降。小域上升与大域下降的取舍出现在 KR H0，适用范围为该数据、该算法和该比较。 [阶段触发决策](../../artifacts/history_information_20261124/S3d/DECISIONS.json) · [六格 Test 统计](../../artifacts/history_information_20261124/test/summary.csv)

以 B1 fixed EMA 的 Test 作为难度参照，KR 的 d0t0/d0t1 AUC 为 .638070/.657511，d2 为 .653821/.677358，d1 为 .702304/.731157。d0 的两任务正例数仅 228/120，d2 为 1,100/827，d1 为 51,847/27,415。最小域 d2 的参照 AUC 高于 d0，C2 对两者的响应方向相反。ML 最大训练域 d2 的参照 AUC 为 .756001/.770739，低于最小域 d0 的 .773518/.788648。规模和难度在两数据上的关系不同。 [B1 全部格的 AUC、LogLoss 与支持](../../artifacts/history_information_20261124/TEST_ENDPOINTS.csv)

## 9. 冻结后的正式性能

| 数据 | 方法与读出 | Macro AUC | Macro LogLoss | Traffic AUC | Pooled AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| MovieLens | 发表参照 M3oE online | .775190 | .494891 | .776435 | .776760 |
| MovieLens | B1 fixed EMA | .778557 | .495570 | .779809 | .780125 |
| MovieLens | H0 fixed EMA | .778474 | .494491 | .779727 | .780010 |
| MovieLens | C2_A100 fixed EMA | .778367 | .493989 | .779605 | .779912 |
| KuaiRand | 最强发表参照 HyperGate online | .682113 | .440590 | .712842 | .718065 |
| KuaiRand | B1 fixed EMA | .676703 | .435269 | .715631 | .720905 |
| KuaiRand | H0 fixed EMA | .680766 | .435339 | .714856 | .720053 |
| KuaiRand | C2_A100 fixed EMA | .677566 | .435298 | .716915 | .722199 |

C2_A100 fixed EMA 在 MovieLens 相对 M3oE online 的配对差为 +.003177，95% CI `[+.002013,+.004340]`，5/5 正，最差格 +.002504，通过发表比较数值门。同读出 B0/B1/H0 的增量均为负，C2 的 Test 独立排序贡献未通过内部对照。其 MovieLens LogLoss 低于 B1 和 H0。KR 的 C2 fixed EMA 相对 HyperGate 为 −.004546，95% CI `[−.012442,+.003350]`，相对 H0 的 Macro AUC 为 −.003199，Traffic 和 Pooled 则分别增加 +.002059、+.002146。C1 在 KR 同读出相对 B1 的 Test 增益为 +.003081，5/5 正。H0 的 Macro AUC 仍高 .000981，Validation 上 C1 未进入控制。C1 的 ML 发表比较为 +.002984541858，落在 +.003 门下。 [正式绝对表与选点](NUMERICAL_APPENDIX.md#冻结-test-绝对性能) · [发表比较门](../../artifacts/history_information_20261124/TEST_METHOD_GATES.json)

## 10. 计算与工程成本

| 阶段 | 轨迹 | epoch | 原生批事务及优化器调用 | 墙钟秒 | 累计 worker 秒 | 单进程峰值显存 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 30 | 1,500 | 51,750 | 35,718.270 | 33,047.631 | 1,888 |
| S2 | 40 | 2,000 | 69,000 | 8,297.188 | 48,652.354 | 1,850 |
| S3a | 10 | 500 | 17,250 | 1,878.145 | 10,323.317 | 1,848 |
| S3b | 20 | 1,000 | 34,500 | 2,230.020 | 24,653.062 | 1,856 |

累计 172,500 次训练前向与原生更新、120,750 次主训练额外 PAIR autograd sweep、50,000 次 Validation 前向。Train 自然行曝光合计 7,015,747,500，按每行两项 BCE 标签计 14,031,495,000。155,250 个带 PAIR 的合法批事件合计采样 317,952,000 对。D2 另计 40 参考步、400 虚拟步、40 个探针前向、240 个来源 autograd sweep、2,640 次面板前向。冻结 Test 新端点共 800 次前向，端点评价时间累计 84.234 秒。全程峰值主机使用约 43.19 GiB（含主机其他负载），单 worker 峰值 RSS 约 2.98 GiB。墙钟含资源等待，worker 秒为过程耗时累加，两个统计口径随并发和启动成本而变化。 [成本账本](costs.csv) · [运行收据](../../artifacts/history_information_20261124/S1/RUN_RECEIPT.json) · [训练轨迹账本](trajectory_ledger.csv)

两张 RTX 3090 按逐卡实测显存和功耗小于 85%、主机 RAM 小于 90% 派发。S1 受资源等待影响最明显，S3b 实测并发达到 16 个 worker。100 条轨迹全部首次完成，无 CUDA OOM、训练重放或状态丢失。所有状态保存和读回均在 RAM 进行，完成后前台正常退出并释放模型。独立审查的两轮修订及最终验收见 [实现审查](IMPLEMENTATION_REVIEW.md)。

## 11. 最终研究决策

| 对象 | 决策 | 保留结果 | 范围 |
| --- | --- | --- | --- |
| C1 | `NO_ACTIONABLE_INCREMENT` | 共同分母干预实际改变方向和排序关系，KR Test 相对 B1 存在 AUC 增量 | 验证集门未通过，KR 增量低于同读出 H0 |
| C2 | `NO_ACTIONABLE_INCREMENT` | ML 的 fixed EMA 验证集结果触发并完成 S/M/P，Test 的 LogLoss 和发表比较保留数值价值 | 同读出 Test AUC 未超过强内部参照，KR 六格及控制阻止晋级 |
| 最终方法 | `Cstar=null` | 固定 EMA 读出与 A21 的已复现 KR Test 事实保留 | 当前两套自然数据、M3oE fixed1 与已冻结选点协议 |

下一项最有区分力的实验是**在相同 pair 与相同曝光下保留经训练标签核验的历史正确排序关系**。固定 PAIR 目标、曝光、EMA 和参照模型，将对照绑定到同一批历史 pair。实验臂保留这些 pair 的历史正确 margin，对照使用等剂量的当前 margin 约束。以共同 epoch 的丢失率、净纠正和完整验证六格比较判断历史内容的增量，避免把额外曝光或普通平滑解释为关系记忆。 [预注册竞争假设](../../research_plans/26-11-24_history_information/03_HYPOTHESES_AND_PRIORITIES.md)
