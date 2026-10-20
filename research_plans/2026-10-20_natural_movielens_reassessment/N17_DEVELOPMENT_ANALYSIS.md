# N17 H100训练与开发记录（未读test）

**本版范围：64个正式H100配置全部完整；H200、N2/N3及test闭锁尚待后续规定流程。** 原16pilot单列N17_PILOT_ANALYSIS，不混入正式四seed。本文件不报告测试效果，也不根据开发结果改变候选或预算。

## 完整预算核对

formal .0003/.001/.01×双q×四seed共24，quota .01×双q×四seed共8：每配置100轮、2000主更新、80,016,700曝光。matched-budget与matched-N各16：每配置100宏轮/轮、500更新、18,750,000曝光。64配置合计原始训练receipt均COMPLETE_TRAINING_TEST_DEFERRED，100轮有限轨迹与每轮六cell selection指标一致，earliest maximum选点及累计更新/曝光与receipt一致。没有使用不完整seed组。

## selection-only q差异

Δ=SB−SN，数值是**曾用于选模的selection半集**；不能作为独立测试确认。每行两种读数各4seed，所有点态t95 AUC区间均跨零。

|人口/预算，LR|selected ΔAUC（正/负seed）|final ΔAUC（正/负seed）|
|---|---:|---:|
|完整自然，.0003|+.000400（3/1）|−.001189（2/2）|
|完整自然，.001|−.000261（2/2）|+.000547（2/2）|
|完整自然，.01共同/own|+.000481（3/1）|+.002420（4/0）|
|完整支持匹配曝光，.0003|−.000388（1/3）|−.000032（1/3）|
|完整支持匹配曝光，.001|−.000284（2/2）|+.000575（3/1）|
|匹配N，.0003|+.000057（2/2）|+.001811（3/1）|
|匹配N，.001|+.000097（3/1）|+.000571（2/2）|
|完整自然quota，.01|−.000458（1/3）|−.000727（1/3）|

## .01正式共同LR的阶段背景

|q/读数|selection AUC|selection LL|完整训练经验目标J|完成epoch均值|
|---|---:|---:|---:|---:|
|SN selected|.752435|.539590|.439341|13.00|
|SB selected|.752917|.540163|.438487|13.50|
|SN final|.713541|.787096|.315269|100|
|SB final|.715961|.796934|.318188|100|

SN选轮13/13/14/12，SB14/14/14/12。训练J继续下降、selection风险上升与排名下降并存，说明按训练J定义的平台门控和验证选点回答不同问题；H200若按J规则触发仍须执行，不能因这里已过selection峰值取消。已冻结H200门控实际触发：SN平均rho=.030705、SB=.020324，两个q均4/4正且均超过.005；全部8个共同LR配置继续到200轮。主代理从原完整训练J独立复算八个rho（残差<1e−12），保留[原门控](analysis/n17_development/plateau_frozen.json)、[追加配置](analysis/n17_development/H200_manifest.json)与[逐seed复算](analysis/n17_development/plateau_rule_recalculation.csv)。H100原件保留，H200尚未完成，不报告其末轮效果。

**STRONG**：当前完整预算、选点身份与有限开发记录事实。**WEAK**：开发半集中的差异代表总体泛化、或quota的gap变化由计数噪声单独造成。这里的每臂selected是各自验证峰值，点态区间未修正选择和多重性；common/own均.01为同一组模型，不算两次复现。逐cell/test、同状态机制及多预算比较仍按原冻结流程完成。

来源：[逐seed开发读数](analysis/n17_development/seed_selection_readouts.csv)、[完整训练/selection轨迹](analysis/n17_development/training_and_selection_trajectories.csv)、[四seed配对表](analysis/n17_development/selection_q_contrast_summary.csv)、[源文件SHA](analysis/n17_development/source_index.csv)。分析器没有打开任何test工件，状态JSON明确test_read=false、recipes_changed=false。
