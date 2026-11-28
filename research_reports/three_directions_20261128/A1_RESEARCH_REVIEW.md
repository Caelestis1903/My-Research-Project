# A1 独立研究评审

**裁决 PASS。** MovieLens 和 KuaiRand 的干净配对与高斯配对均为 `NO_GO_VALIDATION`，两个数据集的 A2 均为 `NOT_TRIGGERED`。以 40 条 M3oE fixed1 轨迹的 2,000 份逐轮记录核对 240 个 Validation 端点和 80 个独立选点，重算 2,376 行五种子配对统计及 24 项 selected 比较。选中轮次、同轮 Macro LogLoss、逐格保护和阶段判定一致。10 项停止控制复查均未触发，Test 访问记录为零。[执行设计](A_EXECUTION_DESIGN.md) · [运行报告](A1_RUNNER_REPORT.md) · [分析](A1_ANALYSIS.md) · [阶段判定](../../artifacts/three_directions_20261128/A1/decisions.json)

- **STRONG｜预设终点。** P0−B0 和 GP0−G0 在两个数据集、两种读出的 selected Macro AUC 均值共八项均为负，Macro LogLoss 均值均上升。MovieLens 的六格保护通过，AUC 区间上界均低于各自的实用门槛。KuaiRand 的四项比较均未通过六格保护。GP0−G0 的固定 EMA 和 Online AUC 分别下降 0.007549 和 0.008414，两个读出均为五个 seed AUC 下降且 LogLoss 上升。KuaiRand 的 P0−B0 AUC 区间仍包含 +0.002，淘汰判定来自预设均值门槛和六格保护。
- **STRONG｜训练阶段。** 固定 EMA 下，两套数据的两种配对损失在 E10 均为五个 seed AUC 上升，四项区间下界均高于零。两种读出在 selected 和 E50 的比较均值转负。KuaiRand 固定 EMA 的 P0 在第 21–29 轮选中，B0 有三个 seed 到第 50 轮才达到窗口内最高值。后续引用时将判定文件中的 “common-epoch acceleration” 表述为“第 10 轮同轮次 AUC 优势”，把结论限定在本次 50 轮观察窗口。
- **MEDIUM｜局部权衡。** KuaiRand 固定 EMA 的 P0 提高了较小的 d0 两项任务 AUC，最大域 d1 两项均下降，最小域 d2 一升一降，Traffic 和 Pooled AUC 均下降。Online 的 d0t0 方向相反。KuaiRand 的 GP0 在两个读出上六格均下降，降幅最大的 d0t1 位于规模居中的域，其 G0 基线 AUC 排六格第三。MovieLens 最小域 d0 的干净配对两格均下降。规模、基线难度及任务效应应分别解释。
- **WEAK｜机制归因。** 差中差同时涉及 BCE 背景和配对输入，四项区间均跨零。若另行检验机制，固定高斯 BCE、配对 ID、剂量和种子，在 D0 对比干净与独立加噪的配对输入，可直接检验配对噪声的贡献。A2 比较的是噪声共享方式，其父方法门槛在两套数据上均未达到。

[逐种子选点](../../artifacts/three_directions_20261128/A1/per_seed_selected.csv) · [配对汇总](../../artifacts/three_directions_20261128/A1/paired_summary.csv) · [数据审计](../../artifacts/three_directions_20261128/DATA_AUDIT.json)
