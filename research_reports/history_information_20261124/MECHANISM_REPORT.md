# 机制报告 — 共同二阶矩、历时预条件与排序关系

[主报告](FINAL_REPORT.md) · [数值附录](NUMERICAL_APPENDIX.md) · [诊断规范](../../research_plans/26-11-24_history_information/04_MECHANISM_DIAGNOSTICS.md)

## D1 来源、支持和时变预条件

H0 的所有 17,250 个训练步都具有合法 PAIR 事件。固定 24 条条件中，MovieLens 实际出现并抽到 12 条，KuaiRand 出现并抽到 20 条。MovieLens 的标签嵌套产生三种联合标签，5,000 步均有 12 条活动边界。KuaiRand 有 9,774 步具 20 条活动边界、2,476 步具 16 条。静态采样同时保留均匀质量和自然质量，所选来源格与训练流量的差额给出了辅助梯度曝光边界。

| 数据 | 域 | 训练自然占比 | PAIR t0 次数 | PAIR t1 次数 | PAIR 域占比 |
| --- | --- | ---: | ---: | ---: | ---: |
| MovieLens | d0 | 19.49% | 585 | 549 | 22.68% |
| MovieLens | d1 | 39.82% | 956 | 988 | 38.88% |
| MovieLens | d2 | 40.69% | 965 | 957 | 38.44% |
| KuaiRand | d0 | 8.61% | 731 | 760 | 12.17% |
| KuaiRand | d1 | 86.93% | 4,683 | 4,492 | 74.90% |
| KuaiRand | d2 | 4.46% | 808 | 776 | 12.93% |

同状态探针在 40 锚点分别对六格求梯度，得到 `STAR_SKIP` 与 `EXPERT` 各 240 条格记录，共 480 条。原始 L2 与自然格质量缩放 L2 并列，覆盖支持量、与 Q/DQ/U 对齐及 H0 步内积可用 `D1_SOURCE` 的数据、seed、epoch、cell 和组定位。探针反映当时模型上指定 pair 的局部导数，训练路径经过全部成功更新。 [逐格探针与同状态原始诊断](evidence/S1/diagnostics.jsonl.gz) · [支持统计](pair_support.csv)

下表按五 seed 等权汇总各 seed 的阶段均值。BCE 与 WD 列分别是 signed 二阶交叉成分的坐标均值，后二列是整组时变方向与当前方向的相对 L2 差。正式分析以 `source_cell=all_steps` 的分组统计为源，格条件详见 D1 数表。

| 数据 | 组 | 阶段 | BCE–PAIR signed 均值 | WD–PAIR signed 均值 | 负坐标占比 | 归一化 signed 均值 | $\|\widehat U-D\widehat Q\|/\|D\widehat Q\|$ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| MovieLens | STAR_SKIP | E1–10 | 1.62e−11 | −3.87e−16 | .301 | .00950 | .713 |
| MovieLens | STAR_SKIP | E26–50 | 6.90e−10 | −1.16e−15 | .232 | .00929 | .105 |
| MovieLens | EXPERT | E1–10 | 3.95e−12 | −5.41e−15 | .355 | .00911 | .832 |
| MovieLens | EXPERT | E26–50 | 6.98e−11 | −6.97e−15 | .308 | .00986 | .144 |
| KuaiRand | STAR_SKIP | E1–10 | 2.67e−12 | −1.20e−17 | .345 | .01228 | .544 |
| KuaiRand | STAR_SKIP | E26–50 | 1.15e−10 | −2.22e−16 | .274 | .01663 | .067 |
| KuaiRand | EXPERT | E1–10 | 1.88e−12 | −2.98e−16 | .367 | .00590 | .540 |
| KuaiRand | EXPERT | E26–50 | 7.32e−11 | −3.32e−15 | .173 | .01951 | .081 |

C1 的交叉分解同时保留正、负坐标。以 ML STAR_SKIP 为例，E26–50 的 BCE–PAIR RMS 为 5.81e−8，WD–PAIR 为 1.43e−11，说明此处主要改变 BCE 与辅助损失共同形成的二阶尺度。C2 的方向差在早期更大，两数据和两组均在晚期收敛到更小的相对差。H0 组内相对差的下降与 ML C2 的晚期相对 AUC 上升并存，方向差规模的时间序列提供了运算证据，排序价值由完整轨迹裁定。 [H0 D1 分阶段表](../../artifacts/history_information_20261124/S1/D1_SUMMARY.csv) · [C1/C2 D1 分阶段表](../../artifacts/history_information_20261124/S2/D1_SUMMARY.csv)

## D2 同状态完整函数响应

每数据、每 seed 在 E5/E20/E35/E50 重放真实 H0 步，共 40 锚点。模型、optimizer、EMA 三个更新后哈希全部精确相等。每个锚点使用 11 个方向分支、三个固定面板 `P_TRAIN/V_A/V_B` 和两个读出。分支包括四个候选、各自同状态匹配半径分支、H0 及两个 M9 分母参照。40 个参考步、400 个反事实步和 2,640 个面板评价产生 15,840 条格记录。面板保留分层包含率、加权 AUC、概率 ties、LogLoss 和 logit margin。记录同状态的真实组跨度，对 H0 轨迹的后续状态执行隔离校验。

以下为两个 Validation 面板的平均每格纠正与破坏数。每格使用 8,192 对，对两数据分别汇总 5 seed×4 锚点×2 面板×6 格，即每分支和读出 240 条格记录。

| 数据 | online 分支 | 平均纠正 | 平均破坏 | 平均净变化 |
| --- | --- | ---: | ---: | ---: |
| MovieLens | C1_Z050 | .5583 | .5792 | −.00000254 |
| MovieLens | M_C1_Z050 | .5292 | .5500 | −.00000254 |
| MovieLens | C2_A100 | .4708 | .5750 | −.00001272 |
| MovieLens | M_C2_A100 | .0292 | .0375 | −.00000102 |
| KuaiRand | C1_Z050 | 1.3917 | 1.6875 | −.00003611 |
| KuaiRand | M_C1_Z050 | 1.3667 | 1.7292 | −.00004425 |
| KuaiRand | C2_A100 | .7542 | .8625 | −.00001322 |
| KuaiRand | M_C2_A100 | .0625 | .0583 | +.00000051 |

M_C1 的局部关系变化接近 C1。C2 调整更多 pair，尺度匹配后的 H0 方向改变更少。全部表内净效应绝对值小于 $5\times10^{-5}$，fixed EMA 的同锚点单步变化更小。该干预支持“方向和尺度可分辨”，完整轨迹提供各阶段的净排序差。 [D2 锚点、尺度与面板](evidence/S1/diagnostics.jsonl.gz) · [M 的完整轨迹尺度检查](../../artifacts/history_information_20261124/S3b/M_ACTUAL_DISPLACEMENT_CHECK.json)

## D3 恒等式和数值验收

共同二阶矩的同轨迹差 $V_0-V_{sep}$ 与 signed $2\lambda b h$ 的 $.999$ 递推一致。C2 的相对方向差对应事件核加权的 $(D_j-D_t)h_j$。固定分母或首个事件使差为零。独立 FP64 小张量检查的最大方向误差为 $7.77\times10^{-16}$。H0 与候选零点复用父入口，真实两数据首尾批的模型、optimizer、EMA 验收哈希一致。诊断只读模型和状态，原生 None 与活动零梯度分别保留。 [检查收据](../../artifacts/history_information_20261124/IMPLEMENTATION_CHECKS.json) · [独立实现审查](IMPLEMENTATION_REVIEW.md) · [数值检查源码](../../experiments/history_information_20261124/kernel_checks.py)

## D4 关系保留、纠正与交互

D4 对同一数据、seed 和 cell 固定 32,768 个 Validation pair，比较依据保存的完整验证概率。固定关系样本以概率 ties 半分计分，净变化等于纠正减破坏的评分差。集合 Jaccard 比较两个变更的关系重叠，起点均为父方法 online，终点分别为候选 online 和父方法 fixed EMA。

| fixed EMA 的候选−B1 | 数据 | 端点 | 平均纠正 | 平均破坏 | 平均净变化 |
| --- | --- | --- | ---: | ---: | ---: |
| H0 | KuaiRand | E10 | 1,192.0 | 1,127.7 | +.001963 |
| H0 | KuaiRand | E25 | 2,626.9 | 2,665.8 | −.001188 |
| H0 | KuaiRand | E50 | 3,165.2 | 3,332.0 | −.005090 |
| C2_A100 | MovieLens | selected | 465.1 | 460.2 | +.000149 |
| C2_A100 | MovieLens | E50 | 1,100.0 | 1,066.8 | +.001011 |
| C2_A100 | KuaiRand | selected | 3,044.2 | 3,087.4 | −.001318 |
| C2_A100 | KuaiRand | E50 | 3,169.6 | 3,185.9 | −.000497 |

每行为五 seed 六格等权的关系样本描述。ML C2 的 selected 纠正和破坏集合重叠均值为 .395/.327，KR 为 .264/.182。KR C2_A100−B1 在 E50 的固定 pair 净变化为 −.000497，完整 Validation AUC 差为 +.000151，效应靠近零且分属两个估计口径。ML 同一比较的 E50 净纠正 +.001011 与完整 AUC +.000910 同向。分格原因、各阶段的保持/遗忘、读出交互与支持量来自原始记录。 [S1 关系明细](../../artifacts/history_information_20261124/S1/D4_COMPARISONS.csv) · [S2 关系明细](../../artifacts/history_information_20261124/S2/D4_COMPARISONS.csv) · [完整指标的配对交互](NUMERICAL_APPENDIX.md#部署选点与同阶段交互)

## 对完整算法的含义

证据最强的结果是更新运算得到预期干预，以及 EMA 在两数据上的同轨迹增益。短辅助核在本轮验证集未覆盖 C2，提供了时间跨度有作用的证据。ML 的 C2 在冻结 Test 上与尺度、近邻和同 PAIR 对照形成不同的 AUC/LogLoss 组合，KR 的 M/P 控制覆盖部分验证集增量。把长期候选价值分配给历史关系内容需要同 pair、同曝光的干预，主报告将它作为下一项可区分的问题。 [完整控制决策](../../artifacts/history_information_20261124/S3b/DECISIONS.json)
