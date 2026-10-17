# 实验结果：IN PROGRESS

N0与32条N1试跑已完成，32条独立确认正在运行，尚无本轮完整独立确认结果。N0来自已有v3固定模型复算；新试跑只作调参，不作为八seed独立确认。

calAdam E50：SB−SN宏AUC−.009074，train-p加权cell AUC+.001016，test-p加权+.003108。三个域内两任务均差−.014657/+.003442/−.016007，说明同一预测的评价权重足以翻转符号。宏指标保持预注册，不作事后替换。

672个仅selection拟合的正斜率校准全部成功，2016行before/after读数完整；AUC前后最大变化0，test AUC复算与历史表最大差2.22e−16。宏AUC选轮test的336个cell中279个LL改善；cell不是独立seed。

统一AULC1..20与固定E10/E20/E50显示阶段反例，E100仅实际完成的seed有资格。N0/EVALUATION_DECOMPOSITION.md给数值与边界；N0/readout_ledger.csv、calibration_cell_metrics.csv、split_populations.csv及用户簇区间给全部源表。图见N0/evaluation_decomposition.png及PDF。

新训练test未用于任何选择。全部试跑完成后，H100共同/各自赢家Adam均.01、SGD均1.0，故正式确认只需32条。H10共同赢家则分别.001/.3，显示本两seed tuning配置的预算依赖；均为网格条件结果，不宣称全局LR最优。原始响应、128行horizon读数和图见N1_pilot_surface，独立确认结果尚待完成。
