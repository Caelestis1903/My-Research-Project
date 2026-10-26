# 2026-10-16 科研执行入口

**Stage2D 正式训练16/16完成，逐行数值核验和科学分析完成。**

最新报告：[执行报告](stage2d_execution_20261016/execution_report.md)、[实验结果](stage2d_execution_20261016/experiment_results.md)、[机制分析](stage2d_execution_20261016/mechanism_analysis.md)。机器状态见 [execution_status.json](stage2d_execution_20261016/execution_status.json)，逐track判断见 [scientific_decision.json](stage2d_execution_20261016/scientific_decision.json)。

SB−SN validation-selected macro AUC四seed均值−0.010406，4/4负；EB−EN为−0.006547，1/4正。两者无实用收益且guard失败，因此不升级条件性边界训练或新方法开发。Sample的负效应本身稳定，**不能写成整个Track A“没有效应所以STOP”**；Adam-specific性能因果仍未识别。

原Stage3B/4/4B的历史材料依赖仍为INCOMPLETE/BLOCKED，不用新IID研究替代。可执行的预定矩阵及分析已完成，不能声称所有条件阶段都跑过。用户纠正的上轮全局禁训解释已撤销，旧 `execution_20261016/` 的数值/取证保留，其全局停止判断仅为历史快照。
