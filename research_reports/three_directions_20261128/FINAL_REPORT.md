# EXP083 · MDMTRec 三方向 Validation 结项报告

在 MovieLens-1M 和 KuaiRand-Pure 的自然采样 M3oE fixed1 协议中，干净或高斯配对损失、PAIR 梯度新鲜度校正、历史正确配对筛选均未达到预设 Validation 晋级条件。配对损失在第 10 轮的排序收益没有延续到各臂选点。历史配对翻转在 Train 上可重复观测，历史正确配对筛选相对匹配随机记忆未达到排序增量门槛。两种记忆臂在第 50 轮相对持续当前配对监督均改善 AUC 和 LogLoss，选点结果的优势更小，并随数据和读出变化。 [运行清单](../../artifacts/three_directions_20261128/RUN_MANIFEST.json) · [Validation 冻结](../../artifacts/three_directions_20261128/VALIDATION_FREEZE.json)

**状态** `VALIDATION_COMPLETE_TEST_PENDING_AUTHORIZATION`。Test 保持封存，访问计数为 0。冻结清单记录选点身份、配置与判定，合格候选列表为空。 [Validation 冻结](../../artifacts/three_directions_20261128/VALIDATION_FREEZE.json)

## 设计、对照与选点

两个数据集均使用冻结的自然 Train/Validation 划分、FP32、batch 40,960 和 50 轮训练。每一比较以 seeds 2022–2026 为五个配对单位。各臂、每个 seed 的 online 与固定参数 EMA 分别取第 1–50 轮六格 Macro AUC 最高的轮次，严格并列取最早，LogLoss（LL）取同轮。E10/E50 是共同轮次诊断。AUC 与 LL 均以候选减对照计算，AUC 增加与 LL 下降分别表示改善。 [研究计划](../../research_plans/26-11-28_three_directions/README.md) · [数据审计](../../artifacts/three_directions_20261128/DATA_AUDIT.json) · [决策规则](../../research_plans/26-11-28_three_directions/06_METRICS_AND_DECISION_RULES.md)

| 阶段 | 比较及控制 | 预设 Validation 条件 |
| --- | --- | --- |
| A1 配对损失 | B0 为 clean BCE，P0 加 clean pair loss。G0 为归档 GA31-R3 D0 Gaussian BCE，GP0 加同剂量配对损失，D0 配对端点独立加噪。使用 macro6 条件，2,048 对/步，λ=.05，τ=1。 | P0−B0 ≥ +.002，GP0−G0 ≥ +.001，各自六格均 ≥ −.001。 |
| B1 新鲜度校正 | B0 为普通 PAIR Adam，B1 复现 A21 单份聚合的共享 PAIR 历史。B2 在 B1 上注入同批同对、相邻共享状态梯度差校正，γ=.5。B3 在 B1 轨迹上观察校正量。 | B2−B1 ≥ +.001，六格均 ≥ −.001。 |
| C0 离线审计 | 从 A1 B0/P0 读取 E5、E10 和 Validation-selected 固定 EMA。用同一候选池核对 Train 配对翻转、置信度和与当前抽样的精确重合。 | 可重复的选点前翻转、配对重合稀少、margin 可区分稳定性、合规 Train 快照可复核。 |
| C1 关系记忆 | C0/C1 两臂精确链接 A1 B0/P0。C2/C3 对齐 P0 前十轮，E11 起以 bank 配对替换当前配对损失，使用同一 E10 EMA 教师的软 margin。C2 从全候选随机选取，C3 从 E10 严格正确的候选中选取。两臂匹配条件、实际容量、抽样位置和剂量。 | 同一数据与读出的 C3−C2、C3−C1 均 ≥ +.001，且两项六格均 ≥ −.001。 |

C1 采用用户批准的每个活跃条件最多 2,048 个唯一配对，并在每步有放回抽 2,048 对。执行记录核对了 C2/C3 与原 P0 的 E10 边界、优化器与 loader/RNG 摘要、当前抽样身份以及后续配对曝光。 [A 执行设计](A_EXECUTION_DESIGN.md) · [B 执行设计](B_EXECUTION_DESIGN.md) · [C0 执行设计](C0_EXECUTION_DESIGN.md) · [C1 执行设计](C1_EXECUTION_DESIGN.md) · [C1 对照链接](../../artifacts/three_directions_20261128/C1/LINKED_CONTROLS.json) · [E10 边界](../../artifacts/three_directions_20261128/C1/E10_BOUNDARIES.json)

## Validation 主要结果

差值、标准差和区间单位均为 10⁻³。95% 区间为双侧配对 t 区间，正向数来自未舍入的五个 seed 差值。逐格保护线为 −1.000。 [A1 配对汇总](../../artifacts/three_directions_20261128/A1/paired_summary.csv) · [B1 配对汇总](../../artifacts/three_directions_20261128/B1/paired_summary.csv) · [C1 配对汇总](../../artifacts/three_directions_20261128/C1/paired_summary.csv)

| 阶段 | 数据 | 读出 | 比较 | ΔAUC | AUC SD | 95% CI | AUC 正向 | ΔLL | 最低格 ΔAUC | 六格 |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| A1 | ML | online | P0−B0 | −0.440 | 0.441 | [−0.988, +0.107] | 1/5 | +0.297 | d0t1 −0.905 | 通过 |
| A1 | ML | online | GP0−G0 | −0.173 | 0.353 | [−0.611, +0.266] | 1/5 | +1.460 | d2t0 −0.831 | 通过 |
| A1 | ML | EMA | P0−B0 | −0.189 | 0.498 | [−0.807, +0.429] | 2/5 | +1.015 | d1t1 −0.654 | 通过 |
| A1 | ML | EMA | GP0−G0 | −0.158 | 0.469 | [−0.741, +0.424] | 2/5 | +1.389 | d1t1 −0.525 | 通过 |
| A1 | KR | online | P0−B0 | −1.768 | 7.985 | [−11.683, +8.147] | 1/5 | +5.050 | d1t1 −6.244 | 未过 |
| A1 | KR | online | GP0−G0 | −8.414 | 4.588 | [−14.110, −2.717] | 0/5 | +10.783 | d0t1 −22.695 | 未过 |
| A1 | KR | EMA | P0−B0 | −0.912 | 7.474 | [−10.193, +8.368] | 1/5 | +4.454 | d1t1 −11.878 | 未过 |
| A1 | KR | EMA | GP0−G0 | −7.549 | 5.681 | [−14.602, −0.495] | 0/5 | +8.466 | d0t1 −15.686 | 未过 |
| B1 | ML | online | B2−B1 | +0.074 | 0.359 | [−0.372, +0.519] | 2/5 | +1.173 | d0t0 −1.046 | 未过 |
| B1 | ML | EMA | B2−B1 | +0.073 | 0.189 | [−0.162, +0.307] | 3/5 | +0.385 | d1t0 −0.217 | 通过 |
| B1 | KR | online | B2−B1 | +0.744 | 3.011 | [−2.995, +4.484] | 3/5 | −0.295 | d2t0 −2.799 | 未过 |
| B1 | KR | EMA | B2−B1 | +0.264 | 3.934 | [−4.621, +5.149] | 3/5 | +0.989 | d0t0 −5.005 | 未过 |
| C1 | ML | online | C3−C2 | −0.231 | 0.393 | [−0.719, +0.257] | 1/5 | −0.578 | d2t1 −0.501 | 通过 |
| C1 | ML | online | C3−C1 | +0.844 | 0.028 | [+0.810, +0.879] | 5/5 | −3.819 | d1t0 −0.082 | 通过 |
| C1 | ML | EMA | C3−C2 | −0.022 | 0.063 | [−0.100, +0.056] | 1/5 | −0.059 | d0t0 −0.249 | 通过 |
| C1 | ML | EMA | C3−C1 | +0.139 | 0.132 | [−0.025, +0.304] | 4/5 | −1.661 | d0t1 −0.173 | 通过 |
| C1 | KR | online | C3−C2 | +0.033 | 1.724 | [−2.107, +2.174] | 3/5 | −0.393 | d0t1 −4.120 | 未过 |
| C1 | KR | online | C3−C1 | +1.229 | 4.627 | [−4.516, +6.974] | 3/5 | −8.436 | d0t1 −11.835 | 未过 |
| C1 | KR | EMA | C3−C2 | −0.612 | 1.641 | [−2.650, +1.426] | 2/5 | −0.750 | d0t1 −8.176 | 未过 |
| C1 | KR | EMA | C3−C1 | +0.241 | 3.303 | [−3.861, +4.343] | 3/5 | −7.338 | d0t1 −22.051 | 未过 |

### A1 配对损失

**STRONG｜第 10 轮排序优势与选点结果。** E10 固定 EMA 的 P0−B0 和 GP0−G0 在 ML 分别为 +.001292、+.000910，在 KR 为 +.015504、+.014347。四项均为 5/5 seed 改善，AUC 区间均排除零。E50 两读出四组比较的均值均转负，选点时八项 AUC 均值仍低于父臂，LL 均值均上升。KR GP0 的两读出均为 0/5 seed 改善 AUC、5/5 seed LL 上升。KR P0 的 AUC 区间较宽，预设裁定由实测均值和逐格保护线给出。KR 固定 EMA 的 P0 在 E21–E29 选中，B0 在 E33–E50 选中，其中三个 seed 在 E50 达到窗口内最高值。 [分析](A1_ANALYSIS.md) · [独立研究复核](A1_RESEARCH_REVIEW.md) · [全端点](../../artifacts/three_directions_20261128/A1/endpoints.csv) · [逐种子选点](../../artifacts/three_directions_20261128/A1/per_seed_selected.csv)

**MEDIUM｜局部域和任务权衡。** KR 的 P0−B0 固定 EMA 在居中规模域 d0 的两格增加 .006638/.010105，在最大域 d1 的两格下降 .011217/.011878，在最小域 d2 一升一降。Traffic 和 Pooled AUC 分别下降 .011246 和 .011110。Online 的 d0t0 转负。ML 最小域 d0 的两任务在 P0−B0 的两读出上均下降。KR 的 GP0 在两读出上六格均下降，受损最大的是 G0 基线水平居中的 d0t1。A 的两项干预在各数据、各读出的 selected 两项任务平均 AUC 均下降。 [选点六格](../../artifacts/three_directions_20261128/A1/six_cell_selected.csv) · [按域和任务汇总](../../artifacts/three_directions_20261128/A1/paired_summary.csv)

### B1 PAIR 梯度新鲜度

**STRONG｜校正未过预设门槛。** B2−B1 四个选点均值均低于 +.001，三组跌破六格保护线。ML 两个区间上界均低于 +.001。KR 两个区间仍包含该值，逐格损害和实测均值共同决定阶段裁定。B2−B1 的四个 LL 区间均跨零。KR 固定 EMA 上，B2−B1 在最大域 d1 两格小幅上升，在最小域 d2 两格均下降，d0 两任务方向相反。 [分析](B1_ANALYSIS.md) · [独立研究复核](B1_RESEARCH_REVIEW.md) · [六格](../../artifacts/three_directions_20261128/B1/six_cell_selected.csv) · [逐种子选点](../../artifacts/three_directions_20261128/B1/per_seed_selected.csv)

**MEDIUM｜A21 在 KR 的阶段反转。** B1−B0 在 E10 的 online/EMA AUC 差为 +.004548/+.002022，在 E50 为 −.007119/−.005104，E50 两读出均为 0/5 seed 改善。独立选点后差距缩至 −.000749/−.001989。B2−B1 的 KR E50 均值转正，两读出区间仍跨零，B2−B0 在 E50 仍为负。KR 的训练 BCE 和混合损失随训练下降，E50 的 B1 两项训练损失均低于 B0。相邻状态梯度差及校正量刻画到的变化是否改善长期排序，证据为 **WEAK**。 [全端点](../../artifacts/three_directions_20261128/B1/endpoints.csv) · [梯度探针与判定](../../artifacts/three_directions_20261128/B1/decisions.json)

### C0 Train 配对诊断

**STRONG｜可重复翻转与有限窗口中的低精确重合。** E10 margin > 0、选点 margin < 0 的五 seed macro6 平均翻转率，ML 的 B0/P0 为 2.947%/2.788%，KR 为 11.246%/8.334%。以 .001 划定稳健正负 margin 时分别为 2.931%/2.775%、11.177%/8.284%。历史正确配对的翻转子集在当前配对抽样中，ML B0/P0 精确命中均为零，KR 为 626/217 次，均匀参照期望为 631.93/210.74 次，命中包含重复抽数。E5→selected 翻转率在 20 条轨迹上均高于 E10→selected，两个窗口的起点和时长同时变化。 [分析](C0_ANALYSIS.md) · [独立研究复核](C0_RESEARCH_REVIEW.md) · [翻转汇总](../../artifacts/three_directions_20261128/C0/SUMMARY.json) · [抽样重合](../../artifacts/three_directions_20261128/C0/OVERLAP.json)

**MEDIUM｜置信度与时距。** 最低正 margin 档的合并原始翻转率为 27.77%–30.76%，最高档为 .002%–.085%。KR 的 P0−B0 从 E10 到各自选点的翻转差为 −2.912 个百分点，配对区间 [−3.431, −2.393]，两个臂分别观察 11–19 轮和 23–40 轮。较高的选点 Validation AUC 与 Train 配对翻转共存，E10 到选点的 LL 在 ML 两臂五 seed 均上升，在 KR 两臂五 seed 均下降。翻转、覆盖和 margin 证据满足 C1 的诊断触发条件，保持历史关系对 Validation 的贡献由训练对照裁定。 [诊断判定](../../artifacts/three_directions_20261128/C0/decisions.json) · [Validation 对齐](../../artifacts/three_directions_20261128/C0/VALIDATION.json)

### C1 历史正确关系记忆

**STRONG｜历史正确性筛选未过预设门槛。** 四组 C3−C2 的选点 AUC 均值均低于 +.001，ML 两组区间上界也低于该值。KR online 的 C3−C1 达到 +.001229，C3−C2 只有 +.000033，且两比较均跌破六格线。KR 的 C3−C2 和 C3−C1 在两种读出均为 d0 两格下降、d1/d2 四格上升。固定 EMA 的 C3−C1 在 d0t1 为 −.022051，点击和长播任务平均分别为 +.001084/−.000602，Traffic 和 Pooled AUC 则为 +.011448/+.011502。ML online 的 C2−C1 为 +.001076，5/5 改善，六格通过。这是随机软记忆对持续当前配对监督的对照结果，历史正确性筛选的裁定取 C3−C2 与 C3−C1 的联合条件。 [C1 分析](C1_ANALYSIS.md) · [判定](../../artifacts/three_directions_20261128/C1/decisions.json) · [逐种子选点](../../artifacts/three_directions_20261128/C1/per_seed_selected.csv) · [六格](../../artifacts/three_directions_20261128/C1/six_cell_selected.csv)

**STRONG｜末轮收益，MEDIUM｜选点 LL 改善。** E50 时 C2/C3 对 C1 的四组读出 AUC 均为 5/5 seed 改善，ML C3 的 online/EMA 增量为 +.002953/+.002599，KR 为 +.009269/+.007915。两记忆臂的 E50 LL 在全部五个 seed 上均下降。C2/C3 的固定 EMA 选点为 ML E22–E28、KR E33–E50，持续当前配对的 C1 为 ML E19–E22、KR E21–E29。各自选点后 C3 的 AUC 增量缩小，LL 在四组均值下降，ML online 和 KR 两种读出的配对 LL 区间低于零。相对 clean C0，C3 的 selected AUC 只在 ML online 为正，四项均低于 +.002。C3−C2 的 E50 AUC 区间均跨零，随机 bank 与软目标共同构成了可解释晚期改善的替代方案。现有证据对独立软目标机制的归因为 **WEAK**。 [全端点](../../artifacts/three_directions_20261128/C1/endpoints.csv) · [完整配对汇总](../../artifacts/three_directions_20261128/C1/paired_summary.csv) · [运行报告](C1_RUNNER_REPORT.md)

## 适用范围与可判别的机制

域规模按 Train 行数计。ML d0/d1/d2 为 155,949/318,599/325,619 行，KR 为 172,782/1,743,898/89,452 行。同人口对照的 AUC、LL 与标签支持分别描述排序难度、概率拟合和样本支持。ML 最小域 d0 的基线 EMA AUC 最高，KR 最大域 d1 的最高。KR C0 审计中 d0 翻转最多、d1 最少、最小域 d2 居中。小域增益与大域损失出现在 KR A1 的特定读出；B1 与 C1 提供任务和对照变化后的不同方向。域规模作为主要解释的证据为 **WEAK**。 [数据审计](../../artifacts/three_directions_20261128/DATA_AUDIT.json) · [B1 域与任务分析](B1_ANALYSIS.md) · [C0 Validation 对齐](../../artifacts/three_directions_20261128/C0/VALIDATION.json)

本轮匹配对照均使用 M3oE fixed1。归档发表基线按原协议单列，ML/KR selected Test Macro AUC 为 M3oE fixed1 .775190/.671907、HyperGate .771412/.682113、AutoIFS .772425/.678597。ML 的最强发表基线为 M3oE，KR 为 HyperGate。归档 EXP080 的 A21−A10 在固定 EMA 上，ML/KR 的 selected Validation 增量 +.000195919/−.001988511 与本轮 B1−B0 的选点、模型哈希及逐 seed 差值一致。其已暴露 Test 增量为 −.000083/+.004062，KR 历史 Test 的 d1t1 为 −.001113。研究计划所列 EXP082 同数值保留为用户补充历史证据，其独立源码与轨迹身份待收据确认。 [历史结果索引](../../mdmt_research_context/03_EXPERIMENT_RESULTS.md) · [B1 来源复核](B1_RESEARCH_REVIEW.md) · [历史来源说明](../../research_plans/26-11-28_three_directions/00_MASTER_PLAN.md#1-任务背景和现有证据)

- **A，WEAK 归因。** 差中差同时涉及 Gaussian BCE 背景和配对端点噪声，四组区间均跨零。固定 Gaussian BCE、配对 ID、剂量与 seed 后，比较干净端点和 D0 独立加噪端点，可检验配对噪声贡献。 [A1 分析](A1_ANALYSIS.md)
- **B，WEAK 归因。** 早期状态差、辅助历史积累及共享二阶矩变化都可解释局部结果。共同阶段的 AUC、LL 与同批梯度探针可区分校正是否改变了训练过程，当前 matched 对照已经回答 γ=.5 下的长程增量问题。 [B1 分析](B1_ANALYSIS.md)
- **C，MEDIUM 诊断。** 对同一候选池和 E10 后相同轮距比较 Train 保持率，可区分观察时距与关系保持。C1 的 C3−C2 已隔离筛选规则的整体增量，受该规则影响的配对身份和教师 margin 分布同时变化。C2/C3 对 C1 的晚期收益同时涉及重复曝光和目标形式。固定配对身份、抽样与剂量的软/硬目标对照可再区分两者，原计划 C4 的正式执行受晋级条件约束。 [C0 复核](C0_RESEARCH_REVIEW.md) · [C1 分析](C1_ANALYSIS.md) · [C1 执行设计](C1_EXECUTION_DESIGN.md)

## 完成量、成本与阶段裁定

| 阶段 | 新轨迹 | 已提交轮次 | Train 更新 | 阶段墙钟秒 | Worker 秒合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 40 | 2,000 | 69,000 | 3,309.80 | 63,302.24 |
| B1 | 30 | 1,500 | 51,750 | 3,641.67 | 55,262.60 |
| C0 | 0 | 0 | 0 | 407.09 | — |
| C1 | 20 | 1,000 | 34,500 | 4,044.59 | 24,435.13 |
| 训练合计 | 90 | 4,500 | 155,250 | — | 142,999.97 |

C0 复用 A1 的 20 条轨迹与 60 份快照。C1 训练 20 条 C2/C3 轨迹，另精确链接 A1 的 20 条 B0/P0 轨迹。表中只计物理新训练，阶段墙钟来自各自收据。三个训练阶段均记录零失败尝试、零 CUDA OOM、零重试。A1 配对损失相对各父臂的平均 worker 时间比为 1.045–1.208×。B3 在 B1 上的观察额外用时合计 7,272.65 秒，B2 校正中的探针用时合计 6,355.31 秒。C1 的 E11–E50 共抽取 56,524,800 对 bank 配对，零 miss。ML/KR 每条轨迹的 bank 数组分别为 835,584 字节及 1,363,366–1,364,828 字节。两个数据集的 C2/C3 记录的平均 worker 时间均低于链接 C1，计时包含原并发调度、评价与读写。 [A1 成本](../../artifacts/three_directions_20261128/A1/costs.csv) · [B1 成本](../../artifacts/three_directions_20261128/B1/costs.csv) · [C0 成本](../../artifacts/three_directions_20261128/C0/COST.json) · [C1 成本](../../artifacts/three_directions_20261128/C1/costs.csv) · [C1 曝光](../../artifacts/three_directions_20261128/C1/EXPOSURE.json)

| 条件阶段 | 判定 | 理由 |
| --- | --- | --- |
| A2 共享/独立高斯配对噪声 | `NOT_TRIGGERED` | 两数据集 GP0−G0 均未达到 AUC 增量和六格联合条件。 |
| B2 γ=.25 确认 | `NOT_TRIGGERED` | γ=.5 的 B2−B1 在四组均未达到 +.001。 |
| C4 同一 bank 的硬目标 | `NOT_TRIGGERED` | C3 在同一数据和读出未同时通过两项改进与六格条件。 |
| S4 组合 | `NOT_TRIGGERED` | 独立晋级机制数为 0，预设要求至少 2 个。 |

预注册触发条件已给出本轮停点。剂量、噪声、γ 或 bank 容量扫描不会回答匹配对照后的同一个机制问题。源基准为 `641b5b8c4242c1ba71f7e4d348a41ec7c630a7fc`，A/B 运行及 C0 使用的原执行源码归档于 `70455ed0ebc898ae80275d17f14e7be80f8cfd57`。C1 的运行身份见其冻结 roster。交付提交待记录。 [阶段清单](../../artifacts/three_directions_20261128/RUN_MANIFEST.json) · [A/B 源码快照](../../artifacts/three_directions_20261128/A_B_SOURCE_SNAPSHOT.json) · [C1 roster](../../artifacts/three_directions_20261128/C1/ROSTER.json) · [上下文更新](CONTEXT_UPDATE.md)
