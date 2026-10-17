# N2首批身份检查失败：诊断记录（IN PROGRESS）

实际失败：N2geometry_adam_SN_seed2033_E40，CUDA:1，原首draw的domain-partition forward梯度与full-batch forward梯度残差超过既定1e−6+1e−4||g||检查界。日志只保存AssertionError，未保存残差值，故目前不能认定是数值舍入，也不能认定聚合有错。原目录/日志/队列FAILED状态全部保留。其余健康运行继续；冻结n17_geometry.py保持不变，避免源码漂移使其他运行失效。

## 最小诊断规格

使用完全相同checkpoint、seed、SN首draw stream和40960行。只比较：同一prediction graph上的full mean与三个domain mean的计数加权和；另forward分域与整批的float32/float64结果。输出预测差、梯度残差、原检查界和行SHA。不训练、不调LR/种子/预算、不读取test、不增加MC或独立训练样本。

实现experiments/n17_identity_recheck.py，CPU先作算术oracle；CPU不等于CUDA失败数值的重现，必要时仍须在有空闲GPU槽位时复核实际失败。不得先放宽界或将失败替换为更有利seed。

## 实现/自审

主agent按上述规格完成实现；核对原Sampler stream、相同初始model/state、SN权重为1、full/per-domain均按两task平均BCE、相同graph retain/free、float64目标与模型dtype一致、allow_unused零坐标和原参数顺序。AST PASS；independent_review=false。CPU工程检查授权RUN，科学判定待实际数值。此诊断不改变正式N2算法或统计阈值。

CPU单draw实际结果：[{"dtype": "torch.float32", "gradient_norm": 0.0583442714384013, "original_tolerance": 6.83442714384013e-06, "same_graph_residual_norm": 4.654378010732188e-09, "separate_forward_residual_norm": 8.662729557845505e-08, "max_prediction_difference": 5.960464477539063e-08, "loss": 0.4876028001308441, "elapsed_seconds": 21.392046689987183}, {"dtype": "torch.float64", "gradient_norm": 0.05834497293300389, "original_tolerance": 6.834497293300389e-06, "same_graph_residual_norm": 9.224111420865454e-18, "separate_forward_residual_norm": 1.6886951280960474e-16, "max_prediction_difference": 5.551115123125783e-17, "loss": 0.4876027900604121, "elapsed_seconds": 40.700037240982056}]
同图聚合通过，分域/整批float32在CPU亦通过原检查界，float64残差约1.69e−16；暂未复现CUDA失败。下一步在第一批N2完成空出GPU时，只暂停机械队列的准入（不停止任何训练/固定点子进程），在空槽复核同一draw后恢复准入；记录该机械事件。尚不修改正式检查界或算法，不将CPU通过虚报为CUDA原因已查明。

## CUDA实际复核与最小修复规格

CUDA:1同一checkpoint/40960行draw复现：float32 separate-forward residual1.8875222e-5>原tol6.8345037e-6，same-graph residual5.3332540e-9；预测差1.3113022e-6。float64 separate-forward5.9357870e-17、same-graph7.8234075e-18。支持float32不同batch形状前向的数值差异解释；未定位具体算子/激活边界，不将其写成已证明具体CUDA内核bug。原错误跨两个forward计算图作严格梯度身份检查，对此锚不成立；同图聚合代数成立。CUDA工程进程exit0后已恢复调度，仅暂停准入约48秒，任何健康子进程均未停止。另SN_seed2034_E40出现同一原检查失败，仍保留，不换seed。

DESIGN（正式重试前冻结）：
1. 原n17_geometry.py完全不变，健康N2/N3的SHA guard继续有效。隔离n17_geometry_identity.py，除身份检查和来源元数据外，与原文件逐字相同。
2. 所有原梯度函数、随机流、完整人口mu、四臂512draw、moments/Adam映射/输出数学不变。新检查只在原首draw上用一个full-batch graph，对直接p/q加权loss和域均值系数组合的loss分别求导；仍用原tol，不放宽。
3. 原partition-forward梯度与full-forward梯度的差异逐块记录，不能将其误标PASS。结果解释时相对实际MC噪声尺度报告这一单draw差异；单draw不提供整个bank的数值误差上界。若误差量级影响结论，再作最小必要复核。
4. 实现后按反向移除新增helper/元数据、恢复旧检查片段，必须逐字恢复原完整文件；记录SHA与差异，主agent自审PASS后才RUN。readout只接受该明确的validation-only等价证明，不普遍忽略源版本。
5. 只对实际失败条件在新目录重做，原目录和FAILED不覆盖；以resolved manifest记录旧任务到重试的映射。第一批正常结果不重跑。
6. 新重试队列启动时短暂停原队列准入，等所有重试已准入且CUDA PID可见（无PENDING）后恢复原队列；随后重试队列无新准入，避免双调度器竞争。每卡总compute<=5含外部任务，现有健康子进程不动。若协调失败，先暂停重试准入，再恢复原队列并记录。

此修复只纠正验证器，不是新增科学算法/调参或MC扩展。IMPLEMENT/REVIEW/RUN与实际结果后续追加，当前尚未重试。

## IMPLEMENT / REVIEW（正式重试前）

已创建隔离n17_geometry_identity.py，原文件未修改。diff仅新增same_graph_identity helper、一行来源元数据、替换原跨forward断言调用。反向移除此三项后与原整个文件逐字相同，原所有估计器函数、main抽样/统计/映射步骤恢复SHA完全一致；证明见N2_validation_only_equivalence.json。

主agent逐项核对：同图参数顺序/unused零填充、两task BCE、p/q与计数系数、retain_graph、原阈值不变、未消费sampler随机数或改变参数、跨forward误差逐块保留、原源文件SHA guard不受影响。AST及全文反向等价检查PASS；independent_review=false。自审PASS，只批准必要工程smoke，smoke通过才准正式同配置重试。不同源SHA不默认为算法不同/相同，汇总须引用该明确证明。

工程smoke已真实通过：CPU、192行工程人口、4臂各2draw、B512，COMPLETE_ENGINEERING，同图身份通过且输出/序列化/来源检查完成；不作研究结果。readout新增显式validation-only等价证明入口，其他base/quota源仍必须一致，并相对实际MC尺度记录首SN-IID draw的跨forward误差（旧结果只有全向量上界）。正式重试只含两个原FAILED锚，N2_identity_retry_manifest及resolved manifest保留替代映射；N2_IDENTITY_RETRY_AUTHORIZATION.json已授权。AST/主agent逐式自审PASS。尚未启动正式重试。
