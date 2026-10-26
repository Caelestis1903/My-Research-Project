# N1 实现报告

实现文件：experiments/n17_common.py、n17_train.py、n17_queue.py、n17_checks.py；历史实现均保留。

复用历史fixed_model、Sampler和prediction/六cell指标代码。新runner保持同训练loss/单optimizer/物理batch/初始化/采样流；局部改变为分离新输出目录、有限模型快照、逐epoch sampler RNG替代完整row列表。保留全部selection/diagnostic预测和每轮经验J；未计算新训练test。源hash、数据/初值、执行commit与真实配置每run保存。

队列逐卡检查显存并包含外部计算进程及尚在导入的子进程，最多5计算进程/卡。每任务独立退出记录，一个失败不终止独立任务；队列不决定科学分支。

必要检查：AST语法；CPU三步loss/一步更新身份/配对/完整state恢复；随后一个完整GPU epoch测实际显存和时长。检查产物见preflight目录。本报告不分析科研结果。
