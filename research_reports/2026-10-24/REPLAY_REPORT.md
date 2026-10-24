# 有限 I3/I5 重放判断

> 本次按“仅上传实验报告”发布。配套CSV/JSON、配置、日志、源码、原始数据、预测和模型权重均未纳入本次提交；正文中的运行工件路径指原服务器归档。数值、统计口径和研究结论保持原报告内容。
> 原始报告SHA256：`3d5897fccea79b90491489b9d1e556b86a50b9e7be83a0ee6625e12e692e9572`。

## 1. 回答对象

复用校准BASE轨迹预定的ML8/KR10个严格按时间排列的状态，seed2022。每状态先独立计算真实Adam位移的一阶预测，再作7个完整回滚H1分支。支持40960个自然fit行，风险P_update每域256行。前2/3为开发，后1/3为固定比较；仅限同一保存轨迹，非完整训练反馈或跨seed证据。

## 2. 原始主要比较

| 数据 | I3周期(开发选) | 周期评价regret | event评价regret | never评价regret |
|---|---|---:|---:|---:|
|movielens|periodic_20|0.000111232|0.000059023|0.000059023|
|kuairand|periodic_100|0.000076939|0.000279775|0.000076939|

| 数据 | I5策略 | 真实branch预算(含BASE) | 评价regret | 累计实际重放成本s |
|---|---|---:|---:|---:|
|movielens|uniform_budget2_ridge|2|0.000008821|5.627714|
|movielens|two_fidelity_budget2_ridge|2|0.000020613|5.629789|
|movielens|two_fidelity_budget2_rank2|2|0.000020613|5.657765|
|movielens|two_fidelity_budget4_ridge|4|0.000000000|7.143863|
|movielens|single_fidelity_budget7_ridge|7|0.000000000|6.166011|
|kuairand|uniform_budget2_ridge|2|0.000000000|8.032396|
|kuairand|two_fidelity_budget2_ridge|2|0.000015461|8.018897|
|kuairand|two_fidelity_budget2_rank2|2|0.000000472|8.021330|
|kuairand|two_fidelity_budget4_ridge|4|0.000007978|10.304394|
|kuairand|single_fidelity_budget7_ridge|7|0.000000000|8.790910|

所有策略、所有1/2/4/7预算、动作、paid回调和calibrator状态保存在 ../../independent_replay/{dataset}/I3_REPORT.json、I5_REPORT.json。完整oracle成本与实际生成成本另有EXECUTION_RECEIPT；未把向量切片当成真实独立scalar更新实现。

## 3. 机制链

廉价预测在真实HF标签之前独立生成，并验证完整状态恢复；没有使用已执行HF位移作为免费低保真先验。不同策略只可通过付费回调取得相应动作标签，未查询的oracle只用于事后评分。当前仍未建立改变训练反馈后的收益链。

## 4. 强简单对照

I3公开never/EMA/random及开发前缀选出的最佳20/50/100周期。I5公开low-only、uniform、random、无low特征的single-fidelity、two-fidelity/rank2。ML two-fidelity预算4虽达到零regret，但真实7branch single-fidelity也为零且总成本更低；KR uniform预算2在评价段已为零，two-fidelity并无稳定额外收益。只比较真实branch数量会漏掉低保真计算成本。

## 5. 异常与证据

**WEAK**：单一探索seed、ML末段3锚点/KR末段4锚点，锚点不是新模型重复数。真实效用尺度约1e-4，零regret只针对有限候选和保存状态。ML完整完成但dispatcher崩溃未捕获进程退出码，使用完整收据证明工件完成，exitcode保持unknown。KR中断3完整锚点保留并重新执行全部10状态；已测部分下界与保守整体耗时上界分别记账。

## 6. 决定

当前事件刷新版本和两保真分配版本不晋级到短程真实反馈或正式ranking比较。它们未超过适当简单对照形成重复的质量—成本优势；未执行的真实反馈问题保持OPEN，不宣称所有状态刷新/低保真机制无效。

## 7. 下一有限问题

在当前7个动作规模先使用直接或简单均匀测量。若真实更大规模任务出现可验证成本瓶颈，可另开有限campaign研究低保真，而不人为制造无意义来源。I3应先证明有能被廉价特征提前预测的实际regret增长；当前缓存动作已足够竞争。

## 8. 收据

INDEPENDENT_REPLAY_DESIGN.json在新HF标签前冻结；逐anchor行ID存ROWS.npz，完整快照SHA与恢复digest见EXECUTION_RECEIPT。总体reserve含额外BASE校准、完整和中断重放、启动失败的保守费用，见EXTENSION_RESERVE_LEDGER.json；Q_confirm/test均未使用。
