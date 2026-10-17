# N2首批身份检查失败：诊断记录（IN PROGRESS）

实际失败：N2geometry_adam_SN_seed2033_E40，CUDA:1，原首draw的domain-partition forward梯度与full-batch forward梯度残差超过既定1e−6+1e−4||g||检查界。日志只保存AssertionError，未保存残差值，故目前不能认定是数值舍入，也不能认定聚合有错。原目录/日志/队列FAILED状态全部保留。其余健康运行继续；冻结n17_geometry.py保持不变，避免源码漂移使其他运行失效。

## 最小诊断规格

使用完全相同checkpoint、seed、SN首draw stream和40960行。只比较：同一prediction graph上的full mean与三个domain mean的计数加权和；另forward分域与整批的float32/float64结果。输出预测差、梯度残差、原检查界和行SHA。不训练、不调LR/种子/预算、不读取test、不增加MC或独立训练样本。

实现experiments/n17_identity_recheck.py，CPU先作算术oracle；CPU不等于CUDA失败数值的重现，必要时仍须在有空闲GPU槽位时复核实际失败。不得先放宽界或将失败替换为更有利seed。

## 实现/自审

主agent按上述规格完成实现；核对原Sampler stream、相同初始model/state、SN权重为1、full/per-domain均按两task平均BCE、相同graph retain/free、float64目标与模型dtype一致、allow_unused零坐标和原参数顺序。AST PASS；independent_review=false。CPU工程检查授权RUN，科学判定待实际数值。此诊断不改变正式N2算法或统计阈值。

CPU单draw实际结果：[{"dtype": "torch.float32", "gradient_norm": 0.0583442714384013, "original_tolerance": 6.83442714384013e-06, "same_graph_residual_norm": 4.654378010732188e-09, "separate_forward_residual_norm": 8.662729557845505e-08, "max_prediction_difference": 5.960464477539063e-08, "loss": 0.4876028001308441, "elapsed_seconds": 21.392046689987183}, {"dtype": "torch.float64", "gradient_norm": 0.05834497293300389, "original_tolerance": 6.834497293300389e-06, "same_graph_residual_norm": 9.224111420865454e-18, "separate_forward_residual_norm": 1.6886951280960474e-16, "max_prediction_difference": 5.551115123125783e-17, "loss": 0.4876027900604121, "elapsed_seconds": 40.700037240982056}]
同图聚合通过，分域/整批float32在CPU亦通过原检查界，float64残差约1.69e−16；暂未复现CUDA失败。下一步在第一批N2完成空出GPU时，只暂停机械队列的准入（不停止任何训练/固定点子进程），在空槽复核同一draw后恢复准入；记录该机械事件。尚不修改正式检查界或算法，不将CPU通过虚报为CUDA原因已查明。
