# C1 历史正确排序关系记忆

**STRONG。** 按 E10 教师的排序正确性筛选历史配对，在两套数据、两种读出上均未达到相对随机记忆的选点 Macro AUC 门槛。MovieLens 的两个配对区间上界均低于 +0.001，KuaiRand 的两种读出均有域任务格跌破保护线。两种记忆在 E50 都优于持续当前配对监督，说明训练后段的收益由随机记忆对照共享。阶段判定为 `NO_GO_VALIDATION`，软目标对硬标签的 C4 对照为 `NOT_TRIGGERED`。本阶段使用 Validation，Test 访问数为 0。[阶段判定](../../artifacts/three_directions_20261128/C1/decisions.json) · [执行状态](../../artifacts/three_directions_20261128/C1/STAGE_STATUS.json)

## 训练条件与配对口径

M3oE fixed1 在 MovieLens-1M 和 KuaiRand-Pure 各运行至 E50，配对 seed 为 2022–2026。逻辑臂 C0 和 C1 分别链接 A1.B0 的 clean BCE 与 A1.P0 的 clean BCE 加当前硬标签 pairwise 监督，各 10 条原始轨迹。C2 和 C3 各新跑 10 条，E1–E10 执行同一 P0 训练，E11–E50 保留 clean BCE，以固定历史库的软目标替换当前配对监督。C2 随机保留候选关系，C3 从 E10 固定 EMA 教师的严格正 margin 关系中抽取。两组共用各 seed 的 Train 候选池和 E10 教师，软目标由截断到 [−5, 5] 的教师 margin 构造，温度为 1，配对损失系数为 .05。[执行设计](C1_EXECUTION_DESIGN.md) · [20 条链接对照](../../artifacts/three_directions_20261128/C1/LINKED_CONTROLS.json) · [运行报告](C1_RUNNER_REPORT.md)

每条轨迹在每种读出上以六格等权 Macro AUC 的最早峰值选点，LogLoss 取该选点的同一 epoch。E10 和 E50 的比较则固定到共同 epoch。所有差值按同数据集、同读出和同 seed 求候选减对照，统计量以五个配对差计算，报告样本 SD 和双侧 95% Student t 区间（df=4）。下表差值单位均为 0.001，AUC 正差和 LogLoss 负差分别表示改善。[逐种子选点](../../artifacts/three_directions_20261128/C1/per_seed_selected.csv) · [配对汇总](../../artifacts/three_directions_20261128/C1/paired_summary.csv)

## Validation 选点

| 数据集 | 读出 | 比较 | ΔMacro AUC 均值 ± SD [95% CI] | AUC 上升 | ΔMacro LogLoss 均值 ± SD [95% CI] | LogLoss 下降 |
|---|---|---|---:|---:|---:|---:|
| MovieLens | online | C3−C2 | −0.231 ± 0.393 [−0.719, +0.257] | 1/5 | −0.578 ± 3.595 [−5.042, +3.887] | 3/5 |
| MovieLens | online | C3−C1 | +0.844 ± 0.028 [+0.810, +0.879] | 5/5 | −3.819 ± 1.857 [−6.124, −1.514] | 5/5 |
| MovieLens | 固定 EMA | C3−C2 | −0.022 ± 0.063 [−0.100, +0.056] | 1/5 | −0.059 ± 1.415 [−1.817, +1.698] | 2/5 |
| MovieLens | 固定 EMA | C3−C1 | +0.139 ± 0.132 [−0.025, +0.304] | 4/5 | −1.661 ± 2.109 [−4.280, +0.958] | 4/5 |
| KuaiRand | online | C3−C2 | +0.033 ± 1.724 [−2.107, +2.174] | 3/5 | −0.393 ± 1.641 [−2.431, +1.644] | 3/5 |
| KuaiRand | online | C3−C1 | +1.229 ± 4.627 [−4.516, +6.974] | 3/5 | −8.436 ± 5.597 [−15.386, −1.487] | 5/5 |
| KuaiRand | 固定 EMA | C3−C2 | −0.612 ± 1.641 [−2.650, +1.426] | 2/5 | −0.750 ± 0.602 [−1.498, −0.002] | 4/5 |
| KuaiRand | 固定 EMA | C3−C1 | +0.241 ± 3.303 [−3.861, +4.343] | 3/5 | −7.338 ± 3.119 [−11.211, −3.466] | 5/5 |

五个 seed 的选点 Macro AUC 差如下，列顺序为 2022、2023、2024、2025、2026。同一检查点的逐 seed LogLoss 和其他指标随配对表保存。[逐种子配对差](../../artifacts/three_directions_20261128/C1/paired_seed.csv)

| 数据集 | 读出 | C3−C2 | C3−C1 |
|---|---|---|---|
| MovieLens | online | −0.261，−0.726，+0.339，−0.102，−0.408 | +0.875，+0.846，+0.832，+0.804，+0.864 |
| MovieLens | 固定 EMA | +0.081，−0.044，−0.011，−0.053，−0.083 | +0.317，+0.082，+0.077，+0.231，−0.011 |
| KuaiRand | online | +0.774，−2.793，−0.178，+0.605，+1.760 | +5.023，−5.257，+2.729，−1.794，+5.446 |
| KuaiRand | 固定 EMA | +1.187，−2.741，−0.792，−1.550，+0.835 | +3.737，−2.414，+2.480，−4.003，+1.406 |

**STRONG。** 注册门槛要求同一数据集与读出上的 C3−C2、C3−C1 都达到 +1.000，且各自六格差均 ≥ −1.000。MovieLens 的四组比较均通过六格保护，Macro AUC 均低于门槛。KuaiRand online 的 C3−C1 达到均值门槛，C3−C2 未达到，两组都跌破六格保护。KuaiRand 固定 EMA 的两组均值与六格均未过门。KuaiRand 的 C3−C2 区间仍包含 +1.000，样本精度允许更大的聚合效应，门槛判定依据实测均值和分格损失。[注册判据](C1_EXECUTION_DESIGN.md#11-falsification-and-disposition) · [逐项门槛](../../artifacts/three_directions_20261128/C1/decisions.json)

随机记忆 C2−C1 的选点 Macro AUC 在 MovieLens online 为 +1.076（SD 0.405，区间 [+0.573, +1.578]，5/5），固定 EMA 为 +0.161（[+0.044, +0.278]，5/5）。KuaiRand online 和固定 EMA 分别为 +1.196（[−2.958, +5.349]，3/5）和 +0.853（[−1.924, +3.631]，4/5）。该对照衡量随机历史库与软目标的联合变化，C3 的注册门槛以表中的两个父臂比较判定。KuaiRand 的 C2−C1 选点 LogLoss 在 online 和固定 EMA 平均下降 8.043 和 6.588，分别重现于 4/5 和 5/5 种子。C3−C1 在两个读出均为 5/5。C3−C2 的额外 LogLoss 降幅较小，固定 EMA 的区间上界接近零。[配对汇总](../../artifacts/three_directions_20261128/C1/paired_summary.csv) · [逐种子配对差](../../artifacts/three_directions_20261128/C1/paired_seed.csv)

## 六格结果、域规模与基线难度

表内为选点平均 ΔAUC / ΔLogLoss。加粗的 AUC 低于保护线。MovieLens 的 t0/t1 对应评分 >3 和 >4，KuaiRand 的 t0/t1 对应点击和长播。[六格选点原值](../../artifacts/three_directions_20261128/C1/six_cell_selected.csv) · [任务定义](B1_ANALYSIS.md) · [六格配对汇总](../../artifacts/three_directions_20261128/C1/paired_summary.csv)

| 数据集 | 读出 | 比较 | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| MovieLens | online | C3−C2 | −0.303 / −0.033 | +0.289 / −0.192 | −0.499 / −0.701 | −0.060 / −1.175 | −0.314 / −1.464 | −0.501 / +0.099 |
| MovieLens | online | C3−C1 | +0.730 / −7.494 | +0.659 / −5.371 | −0.082 / −1.790 | +0.939 / −2.476 | +1.636 / −5.147 | +1.183 / −0.634 |
| MovieLens | 固定 EMA | C3−C2 | −0.249 / +0.563 | −0.089 / −0.162 | −0.169 / +0.029 | +0.046 / −0.306 | +0.288 / −0.257 | +0.043 / −0.222 |
| MovieLens | 固定 EMA | C3−C1 | +0.210 / −2.715 | −0.173 / −3.640 | −0.062 / −0.413 | +0.373 / −0.921 | +0.387 / −1.425 | +0.100 / −0.850 |
| KuaiRand | online | C3−C2 | **−3.896** / +0.580 | **−4.120** / +0.903 | +0.905 / +0.929 | +0.637 / −0.269 | +1.523 / −1.504 | +5.153 / −2.999 |
| KuaiRand | online | C3−C1 | **−7.469** / −3.172 | **−11.835** / −4.609 | +7.911 / −6.034 | +7.766 / −5.133 | +5.028 / −16.906 | +5.975 / −14.765 |
| KuaiRand | 固定 EMA | C3−C2 | **−3.290** / +0.653 | **−8.176** / +0.582 | +0.836 / −1.526 | +0.278 / −1.079 | +1.827 / −0.801 | +4.853 / −2.330 |
| KuaiRand | 固定 EMA | C3−C1 | **−14.363** / +2.265 | **−22.051** / −0.271 | +11.830 / −11.567 | +11.779 / −9.195 | +5.785 / −11.279 | +8.467 / −13.982 |

表中的域规模使用两任务共用的 Train 行数。基线排序难度用固定 EMA 的 B0 Validation AUC 描述，并保留同阶段 P0 AUC 作为直接父臂参照。MovieLens 的 B0 选点从最小域 d0 到最大域 d2 逐域降低。KuaiRand 的 d0 在 E10 两任务均最低，至 B0 选点时 d0t0 仍略低于最小域的 d2t0，d0t1 则高于 d2t1。[分域 Train 与 E10 对齐结果](C0_ANALYSIS.md) · [选点原值](../../artifacts/three_directions_20261128/C1/per_seed_selected.csv)

| 数据集 | 域 | Train 行数 | B0 选点 AUC t0 / t1 | P0 选点 AUC t0 / t1 |
|---|---|---:|---:|---:|
| MovieLens | d0 | 155,949 | .76492 / .77106 | .76474 / .77099 |
| MovieLens | d1 | 318,599 | .75777 / .75547 | .75755 / .75481 |
| MovieLens | d2 | 325,619 | .75448 / .75338 | .75443 / .75341 |
| KuaiRand | d0 | 172,782 | .66960 / .68628 | .67624 / .69638 |
| KuaiRand | d1 | 1,743,898 | .70377 / .73108 | .69256 / .71920 |
| KuaiRand | d2 | 89,452 | .67072 / .66927 | .67044 / .67043 |

**MEDIUM。** KuaiRand 的 C3−C1 与 C3−C2 在两个读出上都表现为中等规模 d0 两格下降、最大域 d1 与最小域 d2 四格上升。固定 EMA 的 C3−C1 域内均值为 −18.207、+11.804、+7.126，按任务平均则为点击 +1.084、长播 −0.602。同时，样本加权和池化 ΔAUC 分别为 +11.448、+11.502，汇总指标覆盖了 d0t1 的 −22.051。A1 的固定 EMA C1−C0 在 KuaiRand 呈现相反的 d0/d1 方向，d0 分别 +6.638、+10.105，d1 分别 −11.217、−11.878。MovieLens 的 C3−C1 三个域的平均 AUC 都上升，最大域 d2 在两个读出上均增幅最大。C1 的结果不支持以域规模作为主要解释，也未复现跨数据集的小域改善、大域下降权衡。[域与任务汇总](../../artifacts/three_directions_20261128/C1/paired_summary.csv) · [A1 基线分析](A1_ANALYSIS.md)

## E10 与 E50

**STRONG。** 20 条新轨迹的 E1、E10 训练与评价记录和对应 A1.P0 一致。E10 的 online、固定 EMA 教师与已提交的 warmup batch 标识完全一致，C2/C3 的优化器、loader 和 RNG 摘要逐数据集和 seed 匹配。两种读出中，三组 C3−C2、C3−C1、C2−C1 在 E10 的逐 seed Macro AUC 与 Macro LogLoss 差均精确为零。[E10 边界](../../artifacts/three_directions_20261128/C1/E10_BOUNDARIES.json) · [运行验证](C1_RUNNER_REPORT.md) · [配对汇总](../../artifacts/three_directions_20261128/C1/paired_summary.csv)

两种记忆在 E50 相对 C1 均为五个 seed 的 AUC 同向上升、LogLoss 同向下降，两个数据集的 online 与固定 EMA 都复现。下表为相同 E50 的配对结果。相比之下，E50 的 C3−C2 平均 AUC 在 MovieLens online / 固定 EMA 分别为 +0.186 / +0.154，KuaiRand 为 −0.462 / +0.070，四个区间均跨零。[E50 配对汇总](../../artifacts/three_directions_20261128/C1/paired_summary.csv)

| 数据集 | 读出 | 比较 | ΔMacro AUC 均值 ± SD [95% CI] | ΔMacro LogLoss 均值 ± SD [95% CI] |
|---|---|---|---:|---:|
| MovieLens | online | C2−C1 | +2.767 ± 1.063 [+1.447, +4.087] | −10.894 ± 2.112 [−13.517, −8.272] |
| MovieLens | online | C3−C1 | +2.953 ± 1.257 [+1.392, +4.513] | −10.527 ± 1.797 [−12.759, −8.296] |
| MovieLens | 固定 EMA | C2−C1 | +2.445 ± 0.581 [+1.723, +3.167] | −12.144 ± 2.322 [−15.027, −9.262] |
| MovieLens | 固定 EMA | C3−C1 | +2.599 ± 0.906 [+1.474, +3.723] | −13.243 ± 2.531 [−16.386, −10.101] |
| KuaiRand | online | C2−C1 | +9.730 ± 3.062 [+5.929, +13.532] | −16.469 ± 6.226 [−24.200, −8.739] |
| KuaiRand | online | C3−C1 | +9.269 ± 3.945 [+4.370, +14.167] | −16.489 ± 5.917 [−23.835, −9.142] |
| KuaiRand | 固定 EMA | C2−C1 | +7.845 ± 2.447 [+4.806, +10.883] | −12.807 ± 3.084 [−16.636, −8.977] |
| KuaiRand | 固定 EMA | C3−C1 | +7.915 ± 4.903 [+1.827, +14.002] | −15.222 ± 2.498 [−18.324, −12.120] |

固定 EMA 的 C1 选点在 MovieLens 为 E19–22，在 KuaiRand 为 E21–29。C2/C3 的选点分别为 MovieLens E22–28 和 KuaiRand E33–50。固定 EMA 下，KuaiRand 的 E50 平均 AUC 优势约 +7.8 到 +7.9，允许各臂选点时降至 +0.853 和 +0.241。KuaiRand 的 C1−C0 在 E50 同为负值（online −9.929，固定 EMA −8.243），C3−C0 则变为 −0.661 和 −0.328。结果表明记忆能缓解持续当前配对在后段的下降，选点后相对当前配对的排名收益小得多。[逐种子选点](../../artifacts/three_directions_20261128/C1/per_seed_selected.csv) · [时点比较](../../artifacts/three_directions_20261128/C1/paired_summary.csv)

## 曝光与成本

用户批准每个活跃条件最多存储 2,048 个唯一有序关系，保留每步有放回抽取 2,048 对的剂量。每条件先取 4,096 个 Train 候选，实际配对库在 MovieLens 每条件均为 2,048 对，在 KuaiRand 为 1,538–2,048 对。两臂逐条件容量、事件上下文、当前 sampler 配对身份、库内抽取位置与位置去重数相同。E11–E50 无空库回退或缺失事件，每步完成一次 4,096 行的有梯度配对前向。下表以一个臂的五条轨迹为单位，唯一关系和交集按 seed、条件汇总。[容量裁定](../../artifacts/three_directions_20261128/C0/decisions.json) · [E10 库核验](../../artifacts/three_directions_20261128/C1/E10_BOUNDARIES.json) · [曝光明细](../../artifacts/three_directions_20261128/C1/EXPOSURE.json) · [运行报告](C1_RUNNER_REPORT.md)

| 数据集 | 每臂 warmup 抽数 | 每臂记忆事件 / 抽数 | 每臂记忆唯一对 | 与继续当前抽样的交集 C2 / C3 | 均匀参照每臂期望交集 |
|---|---:|---:|---:|---:|---:|
| MovieLens | 2,048,000 | 4,000 / 8,192,000 | 122,880 | 5 / 2 | 3.26 |
| KuaiRand | 5,017,600 | 9,800 / 20,070,400 | 200,653 | 2,867 / 2,810 | 2,801.77 |

记忆阶段共抽取 56,524,800 对，每个库成员的条件内期望暴露为 MovieLens 50–83 次、KuaiRand 31.4–226 次。C2/C3 的库成员仍有重合，MovieLens 为 61,383 / 122,880，KuaiRand 为 98,647 / 200,653。与当前 sampler 的原始配对交集接近均匀参照，符合 C0 对大配对空间的观察。两臂的曝光匹配允许检验正确性筛选，但该比较同时改变被回放的具体关系及其教师 margin 分布，效果对应这套筛选规则的整体增量。[E10 库核验](../../artifacts/three_directions_20261128/C1/E10_BOUNDARIES.json) · [C0 重放分析](C0_ANALYSIS.md) · [每条件曝光](../../artifacts/three_directions_20261128/C1/EXPOSURE.json)

20 条新轨迹共完成 1,000 个 epoch 和 34,500 次训练更新，实测墙钟 4,044.59 秒，worker 合计 24,435.13 秒。链接的 20 条对照成本来自原 A1 轨迹。表内为每条轨迹的平均 worker 秒，含运行调度下的训练、评价和读写，两套运行的时间比描述实测成本。各臂单独观察工具计时均为 0 秒。每条新轨迹的记忆库数组为 MovieLens 835,584 字节、KuaiRand 1,363,366–1,364,828 字节。含训练状态和库的 E10 RAM 包分别为约 27.02–27.03 MB 与 58.76–58.77 MB。[成本明细](../../artifacts/three_directions_20261128/C1/costs.csv) · [E10 RAM 记录](../../artifacts/three_directions_20261128/C1/E10_BOUNDARIES.json) · [执行报告](C1_RUNNER_REPORT.md)

| 数据集 | C0 | C1 | C2 | C3 | C2 / C1 | C3 / C1 | C3 / C2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MovieLens | 694.01 | 838.64 | 599.46 | 570.34 | .715 | .680 | .951 |
| KuaiRand | 1,956.12 | 2,333.65 | 1,925.99 | 1,791.24 | .825 | .768 | .930 |

## 机制判断与阶段去向

**WEAK。** C0 确认 E10 教师的一部分正确关系在后续发生翻转，C1 的匹配回放显示这类关系的正确性筛选对选点排名缺少额外收益证据。两组共用的重复曝光、软目标减弱的排序约束，以及训练后段变化和选点移位，都是与结果相容的解释。KuaiRand 的 LogLoss 改善和 d0 排名下降可同时出现，反映不同目标的变化。当前结果来自 M3oE fixed1、两数据集、固定 E10 教师与单一剂量，Train 关系保持与 Validation 泛化分别需要各自的读出。[C0 机制诊断](C0_ANALYSIS.md) · [阶段判定](../../artifacts/three_directions_20261128/C1/decisions.json)

若未来有独立依据继续追问软目标的作用，可固定历史配对身份、抽取顺序、曝光剂量和 seed，对比软 margin 与硬标签，并在预定的共同 epoch 比较排序、LogLoss 和同一批 Train 关系的保持情况。这个对照可区分重复回放所共有的监督与软目标的增量。C4 的注册触发条件要求 C3 相对两父臂同时通过选点门槛和六格保护，本轮 C4 状态为 `NOT_TRIGGERED`。[C1 证伪设计](C1_EXECUTION_DESIGN.md#11-falsification-and-disposition) · [已登记判定](../../artifacts/three_directions_20261128/C1/decisions.json)
