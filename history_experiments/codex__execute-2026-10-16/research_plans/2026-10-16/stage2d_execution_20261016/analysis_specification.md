# Stage2D 分析规格（结果前冻结）

主代理进行科学分析；自动脚本只算表和数值核验。primary为SB−SN、EB−EN的validation-selected六cell宏AUC；不同alpha的Equal−Sample仅interaction，test为forensic。不按阶段观察修改阈值或arm。

先核验16run源码/config/data/init、49×50main/50arch步数、同q两目标的全batch row配对、完整state/产物hash。逐行验证selected val/test labels/domains/row IDs与原CSV及split范围，重算每cell AUC/LL并对CSV，复算严格earliest validation选点。参数状态在同epoch/不同轨迹时只能描述差异，不能因范数不同证明m/v因果。

并列：四seed配对差、均值/SD/范围/正向及达到+.003/.005数量；逐domain/task/cell；worst；训练p权重/实际split行频率权重（都是cell AUC加权，不是pooled AUC）；LogLoss；selected/common/final/same-epoch。基线绝对表现与域规模、正例数分别报告，不把大小定义为难度。

数据抽样不确定性：1000次同cell/标签配对row bootstrap，同一组行权重用于全部四seed的模型。得到给定四个模型平均cell效应的条件区间；四seed不当4倍独立测试样本。不同task分别分层，不造macro置信区间；无user/time聚类、无选择修正与多重比较修正。seed变异另列，不用bootstrap行数代替独立训练seed。

实用和guard按design预先冻结。若有同目标稳定实用收益且guard通过，评估是否足以触发Stage6的独立setting及需排除的竞争解释；若收益不存在，按原2D联合STOP条件和E1单独门作判断。仅有局部state差不等于方法收益；也不能因为“还没有新方法”提前终止本16run验证。

逐项审阅稳定现象、反例、STRONG/MEDIUM/WEAK、竞争解释、规模主因、小/大域trade-off、baseline/model/stage和下一项最有区分力实验。最后明确哪些track真实STOP、哪些历史依赖INCOMPLETE、哪些条件NOT_TRIGGERED；未出现整轮终止条件时继续执行授权范围内必要步骤。
