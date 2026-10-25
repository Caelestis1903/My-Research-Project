# 必要消融与外部参照

## 完整性与比较身份

四条旧aux=0和六条外部方法现均完整到24epoch，未沿用旧native等价超时。它们从核实的完整模型/实际optimizer/控制器/RNG续接，保留原源码与fork身份；不是新seed独立重复，也不因新目录而成为新算法。旧aux=.1四条完整结果不重训。外部方法为项目适配的AutoIFS、STEM和STGU，不声称原论文benchmark完整复现。原始作者版本及适配边界见审查和实现报告。

## 外部方法完整seed3101结果

以下保留旧桥接的AUC-primary原表。LPS按AUC进行匹配seed比较；RHD的LL-primary需要下一节的单独STGU LL选点表。不能用两seed内部均值对一个外部seed宣称稳定领先。

| 数据 | 方法 | selected epoch | Q AUC | Q LL | epoch24 AUC | epoch24 LL |
|---|---|---:|---:|---:|---:|---:|
|movielens|AUTOIFS|23|0.81210252|0.47189906|0.81224289|0.47187494|
|movielens|STEM_DOMAIN_TASK|15|0.80587965|0.47884098|0.80923776|0.47876308|
|movielens|STGU_DOMAIN_ADAM|14|0.80902458|0.47596239|0.81470996|0.47727722|
|kuairand|AUTOIFS|23|0.70290432|0.46367482|0.70492985|0.46197461|
|kuairand|STEM_DOMAIN_TASK|24|0.69645368|0.46436802|0.69645368|0.46436802|
|kuairand|STGU_DOMAIN_ADAM|14|0.69785866|0.46636806|0.70316682|0.46783378|

在selected匹配seed3101，L4相对AutoIFS的AUC在ML约−.004568、KR−.003437，LL均较差（ML+.004423、KR+.002431）；相对STEM AUC均较高，但KR LL反而+.001738。外部pilot无法支持L4总体领先。单seed结论强度WEAK，不以外部模型结果替代L2/L3/FEATURE的内部归因。

## STGU LL选点修正

旧桥接的primary_best对全部外部方法使用validation AUC。RHD主要比较要求LL；本轮从原有ll_best完整检查点读出，在epoch≥12的validation LL最小、平局最早规则下核验身份，再做零训练更新评分，旧AUC输出完整保留。未用Q选择检查点，未改模型/数据/训练。正式fresh STGU直接采用LL规则。

| 数据 | view | R6 epoch | STGU LL epoch | R6−STGU AUC | R6−STGU LL |
|---|---|---:|---:|---:|---:|
|kuairand|selected|24|14|+0.00513315|-0.00210580|
|kuairand|endpoint|24|24|-0.00017501|-0.00357152|
|movielens|selected|12|12|+0.00020495|+0.00072462|
|movielens|endpoint|24|24|-0.00103461|+0.00552072|

ML selected的STGU LL点为epoch12，R6比它LL高.00072462；KR R6比STGU LL低.00210580。共同epoch24在ML R6 LL高.00552072，KR低.00357152而AUC略低.00017501。校准与排序、选点与训练终点均有区别。以上仅seed3101，效力结论WEAK；选择和检查点身份核验为STRONG。

## 组织×aux二乘二

以同seed、同数据和合法独立选点比较，K1 aux0−aux.1的AUC在ML+.0000705、KR−.0093026，LL在ML+.0002458、KR+.0043503；K4为AUC ML−.0028696、KR−.0012904，LL ML+.0016014、KR+.0002979。aux帮助大小依组织和数据变化，ML K1排序近似持平，不能概括成aux普遍有害。

K4−K1的AUC在aux0为ML−.0020761、KR−.0010930，在aux.1为ML+.0008640、KR−.0091052，组织与aux存在描述性交互。单seed且旧版本，证据WEAK；不能直接外推为新LPS形成必须使用aux。完整selected/24及全部六格见旧析因附录，未把旧ML22/10和KR9/9当完整终点。

## 不能省略的内部对照

RHD同时对R0–R5比较，BEST_SIMPLE由validation锁定，不以Q上的弱对手替代。LPS同时对L2/L3/FEATURE，并用L1/L5及同bank常量诊断解释粒度与条件性。MONO保持每格排序、FEATURE能改变排序，额外目标监督、校准与来源读取不混为一种机制。P0固定状态机会、额外head拟合及动作使用率都不能替代完整轨迹。

条件CAL/OWNER/GEOM归因、R7历史对照及KR新seed复评另见主线报告和正式报告。所有外部LL修正的checkpoint SHA和六格输出另列附录，原始AUC表不会被静默改写。

## 完整冻结后基准与新seed边界

两历史角色全部89条逻辑轨迹的selected/终点均已评分；老aux纯析因和STGU LL选择transport的开发Q表保持其原身份，不将历史AUC-primary STGU的test LL列当成公平RHD主比较。fresh STGU在KR4101–4104的完整50epoch中直接按validation LL选点，test selected平均LL为.43581552，HB为.43834780（HB−STGU+.00253228，仅1/4有利）；终点HB则4/4 LL更低，差均值−.01779577。不能宣称外部方法全局领先或HB全局领先。

单seed开发test上，L4相对AutoIFS selected AUC在ML/KR分别−.00149423/−.01401558，LL均较高；相对STEM AUC也均较低，LL均较高。legacy的L4−STEM AUC却两数据为正、KR LL仍较坏。它们是完整项目适配的单seed参考（WEAK），不是原论文benchmark全复现。AutoIFS真实search/rewind/reuse和所有必要费用均保留。

P2 CAL/OWNER/GEOM的完整两个历史角色数值与反向终点见LPS主报告及exposed附录；这些单seed不替换冻结主算法。完成触发包后，本版没有未跑完的规定训练；历史叶候选/部分成本/缓存访问缺项作为证据边界保留。
