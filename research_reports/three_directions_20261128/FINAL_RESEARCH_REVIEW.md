# EXP083 · C1 与结项独立研究复核

**C1 复核 PASS。最终研究复核 PASS。** 历史正确性筛选的阶段裁定为 `NO_GO_VALIDATION`，硬目标对照 C4 为 `NOT_TRIGGERED`。EXP083 完成 90 条新训练轨迹、4,500 个 epoch 和 155,250 次更新。研究状态为 `VALIDATION_COMPLETE_TEST_PENDING_AUTHORIZATION`，合格候选列表为空，Test 封存且访问计数为 0。 [C1 裁定](../../artifacts/three_directions_20261128/C1/decisions.json) · [运行清单](../../artifacts/three_directions_20261128/RUN_MANIFEST.json) · [Validation 冻结](../../artifacts/three_directions_20261128/VALIDATION_FREEZE.json)

## C1 选点与晋级条件

从 1,000 份 C1 逐轮记录复算 40 个新轨迹读出选点，使用最早的最高六格 Macro AUC，LogLoss 取同轮。20 条链接对照的 120 个端点与原 A1 逐项一致。以 240 个端点重算 1,980 行配对统计，并用 480 个选点域任务格核对宏均值和保护线。每项比较按同数据、读出和 seed 配对，区间为五个差值的双侧 95% Student t 区间。 [执行设计](C1_EXECUTION_DESIGN.md#11-falsification-and-disposition) · [端点](../../artifacts/three_directions_20261128/C1/ENDPOINTS.json) · [选点](../../artifacts/three_directions_20261128/C1/SELECTED.json) · [六格](../../artifacts/three_directions_20261128/C1/SIX_CELL.json) · [配对汇总](../../artifacts/three_directions_20261128/C1/paired_summary.csv)

| 数据与读出 | 比较 | ΔMacro AUC [95% CI] | AUC 上升 | ΔMacro LogLoss | 最低格 ΔAUC | 六格保护 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| MovieLens online | C3−C2 | −0.231 [−0.719, +0.257] | 1/5 | −0.578 | d2t1 −0.501 | 通过 |
| MovieLens online | C3−C1 | +0.844 [+0.810, +0.879] | 5/5 | −3.819 | d1t0 −0.082 | 通过 |
| MovieLens 固定 EMA | C3−C2 | −0.022 [−0.100, +0.056] | 1/5 | −0.059 | d0t0 −0.249 | 通过 |
| MovieLens 固定 EMA | C3−C1 | +0.139 [−0.025, +0.304] | 4/5 | −1.661 | d0t1 −0.173 | 通过 |
| KuaiRand online | C3−C2 | +0.033 [−2.107, +2.174] | 3/5 | −0.393 | d0t1 −4.120 | 未通过 |
| KuaiRand online | C3−C1 | +1.229 [−4.516, +6.974] | 3/5 | −8.436 | d0t1 −11.835 | 未通过 |
| KuaiRand 固定 EMA | C3−C2 | −0.612 [−2.650, +1.426] | 2/5 | −0.750 | d0t1 −8.176 | 未通过 |
| KuaiRand 固定 EMA | C3−C1 | +0.241 [−3.861, +4.343] | 3/5 | −7.338 | d0t1 −22.051 | 未通过 |

**STRONG。** 表中单位为 10⁻³。同一数据与读出的两项比较都须达到 +1.000，且各自六格均 ≥ −1.000。MovieLens 的 C3−C2 两个区间上界均低于 +1.000，KuaiRand 两个区间包含 +1.000。八项中仅 KuaiRand online 的 C3−C1 达到均值门槛，同时 d0 两格跌破保护线。四组数据与读出均未满足联合条件，C4 的触发结果一致。 [判定](../../artifacts/three_directions_20261128/C1/decisions.json) · [逐种子配对](../../artifacts/three_directions_20261128/C1/paired_seed.csv)

## 训练边界、容量和阶段效应

**STRONG。** 核对两臂各 3,450 次 warmup 更新及 200 份逐轮记录，训练顺序、损失、Validation 和选点记录逐项与链接的 P0 对齐。10 组 C2/C3 的 E10 在线、EMA、优化器、loader 和 RNG 摘要相同，教师绑定与原 P0 一致。两臂各 13,800 次 E11–E50 更新的上下文、当前配对哈希、库内抽取位置哈希和位置去重数均匹配。每步有放回抽取 2,048 对，一次含 4,096 行的配对前向参与梯度。每条件唯一配对数在 MovieLens 为 2,048，在 KuaiRand 为 1,538–2,048。640 行曝光记录合计 14,131,200 次 warmup 抽取、56,524,800 次记忆抽取和零 miss。 [运行报告](C1_RUNNER_REPORT.md) · [容量裁定](../../artifacts/three_directions_20261128/C0/decisions.json) · [E10 边界](../../artifacts/three_directions_20261128/C1/E10_BOUNDARIES.json) · [曝光](../../artifacts/three_directions_20261128/C1/EXPOSURE.json)

**STRONG。** C2/C3 与持续当前配对的 C1 在 E10 的 Macro AUC 和 LogLoss 均相同。E50 两种记忆相对 C1 在两数据、两读出的五个 seed 上均提高 AUC、降低 LogLoss。E50 的 C3−C2 四个 AUC 区间跨零。KuaiRand 固定 EMA 的 E50 AUC 增量为 C2−C1 +.007845、C3−C1 +.007915，独立选点后缩至 +.000853、+.000241。MovieLens online 的 C2−C1 在选点达到 +.001076、五 seed 全正且通过六格保护，支持随机软记忆在这一读出上的结果。 [C1 分析](C1_ANALYSIS.md) · [配对汇总](../../artifacts/three_directions_20261128/C1/paired_summary.csv)

**MEDIUM。** KuaiRand 的 C3 对两父臂均表现为居中规模域 d0 的两任务下降、最大域 d1 和最小域 d2 的两任务上升，两读出方向一致。A1 的 KuaiRand 固定 EMA 在 d0/d1 呈相反方向，域行数对跨阶段损益的解释力为 **WEAK**。正确性筛选改变被回放的关系及其教师 margin 分布。重复曝光和软目标各自的贡献仍为 **WEAK**，可通过固定关系身份、抽样顺序和剂量的软硬目标对照区分。 [C1 分析](C1_ANALYSIS.md) · [A1 复核](A1_RESEARCH_REVIEW.md)

## 结项与来源

| 阶段 | 新轨迹 | 已提交 epoch | Train 更新 |
| --- | ---: | ---: | ---: |
| A1 | 40 | 2,000 | 69,000 |
| B1 | 30 | 1,500 | 51,750 |
| C1 | 20 | 1,000 | 34,500 |
| 合计 | 90 | 4,500 | 155,250 |

完成量由逐轨迹收据、已提交轮次文件和优化器步数复算。C0 的 20 条离线诊断和 C1 的 20 条链接对照均计入原 A1 轨迹。15 项冻结文件 SHA-256 与 180 项新训练选点身份均匹配。A1、B1、C0 沿用各自独立复核的 PASS，支持结项报告对自然采样 M3oE fixed1、两数据集和 Validation 终点的限定。A2、γ=.25、C4、S4 均为 `NOT_TRIGGERED`。复查两个原工作区与研究工作树的十条登记停止路径，均未触发。 [A1 复核](A1_RESEARCH_REVIEW.md) · [B1 复核](B1_RESEARCH_REVIEW.md) · [C0 复核](C0_RESEARCH_REVIEW.md) · [结项报告](FINAL_REPORT.md) · [停止路径](../../artifacts/three_directions_20261128/C1/STOP_CONTROLS.json)

三个 AI 上下文文件的 EXP083 引用可解析至结项报告、实验说明和 Validation 数表。EXP080 的已暴露历史 Test 沿用归档来源，EXP082 独立源码和轨迹身份待收据确认，其数值保留用户补充证据身份。当前结项记录本轮 Validation 结果与零 Test 访问。 [研究索引](../../mdmt_research_context/01_AI_EXPERIMENT_INDEX.md) · [实验详情](../../mdmt_research_context/02_AI_EXPERIMENT_DETAILS.md#exp083) · [指标与原表](../../mdmt_research_context/03_EXPERIMENT_RESULTS.md#exp083--三方向-validation) · [上下文更新](CONTEXT_UPDATE.md) · [历史来源](../../research_plans/26-11-28_three_directions/00_MASTER_PLAN.md#1-任务背景和现有证据)
