# 07｜有限五 seed 超参数计划

返回 [执行入口](00_EXECUTION_RUNBOOK.md)。选择规则统一见 [09](09_EVALUATION_AND_DECISION.md)。每个候选都完成相应五seed预算，不进行单seed性能预筛。

## 1. 搜索菜单

| 算法 | 新超参及候选 | 依据 | 每数据搜索预算 |
|---|---|---|---|
| A_MATCH | direction_alpha={.5,1} | 完全当前方向与保留部分历史，匹配同一步实际范数 | 2×2anchor×5seed×6epoch=120epoch |
| A_RAW | beta1=0固定 | 普通Adam对照；检测是否无需手工norm层 | 条件触发2×5×6=60epoch；无额外beta网格 |
| B_SPLIT | H=ML{20,40,80}/KR{69,138,277}；scope={S,all}；mode={logit,probability} | 源半衰期邻域、共享来源、两种函数均值 | 共享5条50epoch父轨迹，12个score曲线；12次V前向/epoch |
| B_SHARED_ONLY | 无 | 最简单共享参数EMA读出 | 每epoch1次V前向，无SGD |
| C | H固定40/138，scope S，mode logit；s={.5,1} | 收缩与不收缩；不再搜域任务rho | 2个收缩配置×预测/参数、加2个GLOBAL控制，五seed完整曲线 |
| C内部动作 | {0,.25,.5,.75,1,a_e} | 有界凸混合，包含固定B动作 | 用P_A拟合，不是Test/Validation超参网格 |
| B_DISTILL | lambda={.25,.5}；lr={.001,.003} | 保留至少一半硬标签；原lr的.1/.3，5epoch小步迁移 | 4×5×5=100epoch；LABEL两个LR50epoch；Fteacher赢家25epoch |
| AB | 全部继承A与父B赢家 | 有独立来源才组合 | 无额外SGD，无新搜索 |
| Gaussian迁移 | 继承同dataset Clean赢家 | 边界评价 | 不重新搜索 |

若Gaussian继承的主赢家为KD_B/LABEL/Fteacher，三学生共用主赢家LR；两个软教师臂沿用Clean KD赢家的非零λ，LABEL λ=0，教师族不重选。此映射由09固定，不构成新HPO。

A起点E20固定；E35仅短程边界，不把它作为隐藏时机超参。F的rho、Adam原lr/beta2/eps/WD、Gaussian κ/替换比、KD温度T=1、KD长度5epoch是有来源的固定设计值，不为每一常量展开新搜索。KD新增LR已经两点五seed比较；C pair数4096是观察预算，不按性能调。

## 2. 每阶段选参

A在两anchor末两epoch五seed效用上选alpha，门和E20条件见09；不是根据某seed局部虚拟步最大值选择。继续完整训练的是一个matched配置，RAW用于更简单实现对照。

B每配置每seed自行在完整V曲线合法选点，再比较五seed均值；不能先对每seed选不同H/scope再平均。S与all、logit与prob的前向/指标配对，若S/all相近，按规定简单优先次序选。

C各seed内部P_A更新alpha属于算法运行状态；全体五seed共同选择s，不能在Test逐格选最优动作。H不跟随最优B搜索结果改变，固定源主H控制计算；同时报告与同H B及最佳B的差别。

KD四参数组合和两个LR LABEL先全部完成；只冻结一对lambda/lr并补同参数Fteacher控制。学生F是主选择读出，online独立报告。Fteacher并非不受控的原F：它从相同初值、相同步数与LR训练，区分新历史soft目标与普通自蒸馏。

## 3. 没有必要再追加的搜索

本轮不搜索更多rho、分组半衰期、神经门控、pair采样策略、Hessian指标、任意Adam重置、更多历史状态K、蒸馏温度或多阶段学习率。若现有网格没有足够动作或价值，按09结束该版本；不将缺乏机制依据改写为搜索范围不足。

数据路径、eval microbatch与并发数属于执行设置；可在不改变预测定义/训练batch的前提下适配，并记录工程身份。模型状态、预测dtype、loss归约或数学动作有变化时必须登记新科学attempt，不能装作调度变化。

## 4. 冻结与可追溯性

每个选择输出`HPO_SELECTION.json`：候选全部五seed的U/cell/LL、可行性、并列规则、获选配置、配置hash、best epoch/state hash和拒绝/未触发理由。相同源轨迹上的多readout不增加seed数。完成选择后RAM仅保留正式端点、KD初值/教师、父F/O及必要对照；保留所有标量曲线，释放无消费者参数。
