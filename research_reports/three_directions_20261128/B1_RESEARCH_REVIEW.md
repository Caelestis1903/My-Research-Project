# B1 独立研究复核

**PASS。** B1 的执行与分析支持 `NO_GO_VALIDATION`。γ=.5 的 B2 在四组主比较上均低于 +0.001 的选点 Macro AUC 门槛，三组跌破六格保护线。γ=.25 为 `NOT_TRIGGERED`。本轮端点均来自 Validation，Test 访问计数为 0。复核时两个原工作区及研究工作树登记的十个停止标记均未出现。 [分析](B1_ANALYSIS.md) · [设计](B_EXECUTION_DESIGN.md) · [判定](../../artifacts/three_directions_20261128/B1/decisions.json) · [停止路径](../../artifacts/three_directions_20261128/B1/STOP_CONTROLS.json)

## 选点与阶段

从 30 条 M3oE fixed1 轨迹的 1,500 条逐轮记录复算 180 个端点。每臂、每个种子、每种读出分别取最高六格 Macro AUC 的最早轮次，LogLoss 取同轮。逐种子配对与五种子的均值、样本标准差、双侧 t 区间和正向数均与发布表相符。 [运行报告](B1_RUNNER_REPORT.md) · [端点](../../artifacts/three_directions_20261128/B1/ENDPOINTS.json) · [逐种子配对](../../artifacts/three_directions_20261128/B1/paired_seed.csv) · [配对汇总](../../artifacts/three_directions_20261128/B1/paired_summary.csv)

**STRONG · 主比较门槛。** 下表为 selected B2−B1，差值单位为 10⁻³。六格保护下限为 −1.000。四组 LogLoss 配对区间均跨零。MovieLens 两组 AUC 区间上界均低于 +1.000。KuaiRand 两组区间均包含 +1.000，阶段判定依据实测均值和保护线。 [六格](../../artifacts/three_directions_20261128/B1/SIX_CELL.json) · [判定](../../artifacts/three_directions_20261128/B1/decisions.json)

| 数据与读出 | ΔAUC [95% CI] | 正向种子 | ΔLogLoss | 跌破保护线的格 |
| --- | ---: | ---: | ---: | --- |
| MovieLens online | +0.074 [−0.372, +0.519] | 2/5 | +1.173 | d0t0 −1.046 |
| MovieLens fixed EMA | +0.073 [−0.162, +0.307] | 3/5 | +0.385 | 0 格 |
| KuaiRand online | +0.744 [−2.995, +4.484] | 3/5 | −0.295 | d0t0 −2.729、d2t0 −2.799、d2t1 −2.194 |
| KuaiRand fixed EMA | +0.264 [−4.621, +5.149] | 3/5 | +0.989 | d0t0 −5.005、d2t0 −3.242、d2t1 −4.396 |

**MEDIUM · KuaiRand 的阶段反转。** B1−B0 在 E10 的 online/EMA AUC 差为 +4.548/+2.022，E50 为 −7.119/−5.104，E50 两种读出均为 0/5 正向。各臂独立选点后差值缩至 −0.749/−1.989。B2−B1 在 E50 转为 +4.684/+3.040，两个区间仍跨零，选点后两个均值均缩小。同一最小域、最大域和任务的表现随对照变化。以 B0 同人口的结果描述难度时，KuaiRand 最大域 d1 的 EMA AUC 最高，而 MovieLens 最小域 d0 的 EMA AUC 最高。域规模缺乏一致的解释力。 [端点](../../artifacts/three_directions_20261128/B1/ENDPOINTS.json) · [按域及任务配对](../../artifacts/three_directions_20261128/B1/paired_summary.csv)

## 观察成本与历史来源

B3 在 γ=0 的 B1 轨迹上计算校正量，梯度写入由训练 γ 控制。冻结源码哈希与实现一致，已复核的观察开启与关闭夹具给出完全相同的轨迹状态。复算 51,750 条训练记录确认三臂配对曝光及计时分解。MovieLens/KuaiRand 的 B3 额外观察时间合计 2,389.40/4,883.25 秒，B2 的训练探针时间合计 2,439.87/3,915.44 秒。B3 和 B2 各自在两数据集记录 4,995/12,245 次旧状态前向及等量 autograd。并发 worker 时间扣除观测成本后仍包含训练、Validation 和读写。校正幅度对排序结果的机制归因为 **WEAK**。 [冻结名单](../../artifacts/three_directions_20261128/B1/ROSTER.json) · [更新源码](../../experiments/three_directions_20261128/freshness_optimizer.py) · [实现复核](B_RUNTIME_IMPLEMENTATION_REVIEW.md) · [成本](../../artifacts/three_directions_20261128/B1/costs.csv)

EXP080 的 selected fixed EMA Validation 提供同协议复现证据。当前 B1−B0 与归档 A21−A10 的两数据集 20 条逐种子 AUC/LogLoss 差值、两臂绝对值、选点轮次及模型哈希逐项相同，Macro AUC 均值为 +0.000195919/−0.001988511。归档已暴露 Test 对同一 A21 对照记录 −0.000083/+0.004062，KuaiRand d1t1 为 −0.001113。后者只适用于历史固定人口与原选点。计划所列 `EXP082_HISTORICAL_TEST` 的同数值保留为用户补充记录，其独立源码和轨迹身份待收据确认。 [EXP080 原配对表](/home/xzf/Code/phase0-3/artifacts/conditional_history_shared_update_20261122/S5/paired_seed.csv) · [EXP080 汇总](/home/xzf/Code/phase0-3/artifacts/conditional_history_shared_update_20261122/S5/paired_summary.csv) · [S0 来源核验](DESIGN_SPECIFICATION.md#source-and-runtime-findings)
