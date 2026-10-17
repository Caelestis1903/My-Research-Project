# 机制分析：IN PROGRESS

**STRONG，条件算术身份：** 固定预测下只改变评价权重，足以解释calAdam E50宏AUC与test流量加权cell AUC的相反符号。此项不识别域内人口漂移的原因。

**MEDIUM，历史重复与反例：** 四seed下阶段/选轮/LR影响方向；校准改变LL/Brier而保持排序。不同指标和时钟不能合并为统一“均衡采样有害”。现有结果也不支持统一小域收益/大域损伤：该E50例子是两个小域下降、大域改善。

**尚待本轮证据：** 对称LR与独立种子后是否残留实用q效应、域计数贡献、C幅度/方向、瞬时Adam非线性及共享拓扑。N2/N3的工程smoke不是这些机制的科研结果。不能把固定point trace与泛化差直接等同，也不能预设Adam/课程有效或小域等于困难域。

竞争解释包括评价权重和人口、LR/有限预算、排序与概率校准、域计数组成/高阶矩、共享表示及optimizer映射。按设计以N1和N2/N3反事实分别区分。一般importance sampling、EMS-LR与MIS已有直接文献，当前不能宣称方法创新；八篇原文差异与假设边界见LITERATURE_COVERAGE.md。

原Q1–Q46语义保留；Q47–Q52进度见question_registry_working.json。最终需逐项记录稳定现象、反例、证据强弱、未排除解释和最小后续，目前不作终局裁决。

**WEAK，尚仅调参描述：** 本轮tuning的H10与H100共同赢家在两优化器均不同；这支持按完整预算冻结配置的必要性，但n=2且用同一selection选点，不能作为独立泛化或q机制确认。两个H100赢家都在上边界，相邻范围不确定性保留，不追加无限LR网格。


## N2 固定点里程碑（2026-10-17，整轮仍进行中）

16个resolved锚已完成并验证，原14成功+2验证修正重试；原FAILED保留。详见 [N2_FIXEDPOINT_ANALYSIS.md](N2_FIXEDPOINT_ANALYSIS.md) 及 N2_fixedpoint_summary/receipt.json。全参数IID SB/SN trace比2.008914–2.584112，quota后1.994669–2.550850；预测域计数份额仅0.001297%–0.105936%。该固定点结论不能替代完整quota训练的长期性能对照。Adam q平均更新作用尚无法判定；未识别出超出MC尺度的总体均值漂移。N3开发仍运行，独立确认、条件N4、N5-C、全局test freeze及最终统计尚未闭合。
