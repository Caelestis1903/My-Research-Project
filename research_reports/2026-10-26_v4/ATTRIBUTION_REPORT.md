# 归因：需求可读、函数形成与完整增量

## H1：可预测需求

ML的PF OOF需求策略在两个seed均胜常量，六格均非负，P1通过；KR完整h版本在两个seed都输常量和LOGIT，P1失败。证据分别是**MEDIUM**的有限开发支持/反例，不能用ML替代KR，也不能把KR结果解释为所有条件需求不存在。LOGIT在KR仍有两seed LL收益，但伴随AUC下降。详细数值和不确定性在[P1报告](P1_DEMAND_REPORT.md)。

## H2：需求跨支持指导内容形成

ML责任确实变化，DFD每行/任务两项和2最大误差约5.96e-8；source0/task0的责任SD为.5488/.5703（seed5101/5102），不是恒定权重或全OFF。来源函数独立受各自source标签监督，初始化微扰/row plan在匹配臂相同，新的Adam状态、缺源None语义均经实际验收。责任质量、ESS、clip率、优化器step/moment摘要完整保留在原始JSON和[支持表](tables/COST_formation_support.csv)。

责任两项相关−1由和2约束必然产生，不能当互补知识证据。ESS是加权支持集中度，不是独立样本数。正权weighted BCE在充分容量/优化下仍有相同source Bayes目标；函数身份和非零权重并不保证不同功能。

对UNIFORM的开发平均LL收益只有.0000274，GEOM/SHUFFLE未被稳定超过。因此特定需求形成的独立收益为**WEAK/本版未获支持**。不能把source外推错、容量有限、优化路径或凸包约束中的任何一项指定为已经识别的根因。尤其Q/outer固定μ分层缺失，外层需求迁移原因保留为MISSING信息，未用ETA_DIRECT微调输出替代。

## H3：完整方法的实际价值

ML完整流水线未胜两个强直接概率头；最小域task0的LL代价还超保护门槛。H3本版为**VERSION_NOT_SUPPORTED**，有限反例证据**MEDIUM**。DFD胜LOCAL、LOCAL_CAP与常量读取仍可保留，普通来源监督/条件读取、追加标签拟合或概率范围约束已足以竞争。P3与P4均NO_START，不声称已验证标签预算桥接、pooled来源必要性、外部领先或四seed重复性。

## 规模、难度与阶段

ML P1三域LL/AUC均改善。P2相对NATIVE，最小域0的LL在两个seed恶化，较大域1/2改善；相对H_DIRECT，小域与最大域的AUC都可改善而中域下降，LL三个域都变差。没有稳定可推广的“小域改善/大域下降”规律。KR最小域2并非最低基线AUC域。规模与域语义/基率/支持混杂，作为主因的证据**WEAK**。

四个原生模型的epoch24在test LL均差于自己的合法selected；ML Q/legacy的晚期AUC却升而validation/test下降。不同角色的人口支持与模型阶段提供竞争解释，不能把晚期较稳等同最佳质量，也不能据观察断言Adam或共享机制的普遍根因。

## 最有判别力的下一项问题

先在真正未暴露的自然窗口锁定比较H_DIRECT、ETA_DIRECT与普通UNIFORM来源读取，维持目标标签与完整流程计费；检验来源函数是否仍有强简单方法之外的增量。若没有，停止进一步细分形成。当前不再追加K、元学习、正则、seed或人工领域比例搜索。该问题是未来建议，本轮未执行。
