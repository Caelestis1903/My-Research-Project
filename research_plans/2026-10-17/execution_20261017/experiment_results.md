# 实验结果：IN PROGRESS

N0已完成，N1正在运行，尚无本轮完整独立确认结果。当前可用的科学结果均来自已有v3固定模型复算，不作为新训练重复。

calAdam E50：SB−SN宏AUC−.009074，train-p加权cell AUC+.001016，test-p加权+.003108。三个域内两任务均差−.014657/+.003442/−.016007，说明同一预测的评价权重足以翻转符号。宏指标保持预注册，不作事后替换。

672个仅selection拟合的正斜率校准全部成功，2016行before/after读数完整；AUC前后最大变化0，test AUC复算与历史表最大差2.22e−16。宏AUC选轮test的336个cell中279个LL改善；cell不是独立seed。

统一AULC1..20与固定E10/E20/E50显示阶段反例，E100仅实际完成的seed有资格。N0/EVALUATION_DECOMPOSITION.md给数值与边界；N0/readout_ledger.csv、calibration_cell_metrics.csv、split_populations.csv及用户簇区间给全部源表。图见N0/evaluation_decomposition.png及PDF。

新训练test未用于任何选择；N1的LR赢家和科学结论须等H100完整pilot及独立确认，不从当前前缀推断。
