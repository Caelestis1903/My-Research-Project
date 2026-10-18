# 最终科研分析：条件效应、机制反证与适用边界

**状态：COMPLETE。** 必做及触发训练、原始读数、数值核验、分析和报告均完成。120条H100训练轨迹无训练失败；72条唯一正式轨迹、7200个逐epoch模型的test读取全部成功。N2原两个验证断言失败保留，隔离验证修正后的同seed重试已通过；不是换seed补结果。原Q1–Q46历史条目不改写，新轮登记Q47–Q53。遵循start.md §11，由主agent承担所有非机械工作，independent_review=false；机械runner仅启动/监控/记录原始结果。

## 最重要结果

1. **KR主比较未完成统计确认。** Adam selected宏AUC SB−SN=−.005203，t95[−.012006,+.001600]，7/8负，Holm p=.171875；SGD=+.000837，[−.004309,+.005984]，5/8正，Holm p=.6953125。均不能写成显著普遍劣化/改善或等效。Adam bootstrap95为[−.010924,−.000367]而t95跨0，保留推断差异，不切换到有利区间；主协议仍以原t/符号-Holm判定。该selected宏结论证据**WEAK/INCONCLUSIVE**。
2. **条件阶段现象重复。** KR Adam AULC1..20=−.007480，[−.010031,−.004930]，8/8负；AULC1..100=−.005324，[−.009196,−.001453]。ML固定人口的selected宏差common=−.010392、own=−.010875，均8/8负、pointwise区间均在−.003以下；宏LL均恶化。**MEDIUM**：独立优化seed支持这些固定配置/阶段的条件边界，次级读数不升级为主Holm确认。
3. **晚期反例不能忽略。** ML own-LR E100宏AUC=+.021145，[+.015579,+.026710]，8/8正；相同组selected却为负，早期AULC1..20=−.040981。own比较SN .001、SB .0003，包含LR与进度变化；不能据此宣称纯q收益。E100宏收益通过数值门槛，但小域d1两task的同时非劣下界−.005968/−.009874低于−.003。保护未通过表示证据不足，不证明实际损伤；此时六cell点均值均为正。
4. **计数主导的固定点解释被反驳。** 16锚ALL trace的SB/SN：IID 2.008914–2.584112，quota 1.994669–2.550850；IID计数项仅占预测trace的.001297%–.105936%。**STRONG（限本固定点）**：域内梯度分布差异保留了q方差差。全程quota的q差仍负，但quota×q交互区间跨0，不能称quota确定扩大/缩小差距，也不能将quota当纯trace开关。
5. **N3未授权长期拓扑。** 冻结embedding/E10独立确认，两路径full均4/4正但只有约+3e−5；四个相关MCSE达标，full及方向残差均小于.0001，SN方向残差仅2/4正。不补draw，不重选，N4未触发。**MEDIUM**的局部实际界结论；长期共享必要性、唯一方向归因及Adam历史必要性仍未识别。

## 冻结的正式比较

本轮固定Sample经验目标及原M3OE fixed_model计算图：SN的q=p，SB的q=(1/3,1/3,1/3)，IID用p/q修正；routing detach、固定fusion、B40960、WD1e−5保留。训练、调参、机制和评价估计对象分别报告。主不确定性单位是8个配对训练seed，N3每路径是4seed；多个q、task、epoch或draw均不增加独立n。历史test已被过去研究使用，本轮按全局冻结后统一读取，不称全新hidden holdout。

共同LR：KR Adam .01/SGD1.0，common/own赢家相同，轨迹复用；ML共同.001，own SN .001/SB .0003。均只称预设网格内赢家，KR上边界与ML SB下边界不外推连续最优。selection在1..100最早宏AUC最大轮选点，然后评价固定test。

| group | SN | SB | delta | t95 | pos_neg |
| --- | --- | --- | --- | --- | --- |
| N1_common_adam | 0.667714 | 0.662511 | -0.005203 | [-0.012006, +0.001600] | 1/7 |
| N1_common_sgd | 0.648212 | 0.649049 | +0.000837 | [-0.004309, +0.005984] | 5/3 |
| N2_quota_adam | 0.667628 | 0.658437 | -0.009191 | [-0.015795, -0.002588] | 2/6 |
| N5_common_adam | 0.733318 | 0.722926 | -0.010392 | [-0.014599, -0.006185] | 0/8 |
| N5_own_adam | 0.733318 | 0.722444 | -0.010875 | [-0.014447, -0.007303] | 0/8 |

N1 own与common为相同模型，不重复展示、不增加n。以下列出全部预定阶段宏差，完整CI/符号/原始seed见FINAL_STAGE_MACRO_AUC.csv和FINAL_performance/seed_summary.json。

| group | E10 | E20 | E50 | E100 | selected | AULC1_20 | AULC1_100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N1_common_adam | -0.008530 | -0.009708 | -0.007246 | -0.007422 | -0.005203 | -0.007480 | -0.005324 |
| N1_common_sgd | +0.002001 | -0.003312 | +0.000743 | -0.000877 | +0.000837 | -0.001693 | +0.000113 |
| N2_quota_adam | -0.007071 | -0.007078 | -0.009556 | -0.009643 | -0.009191 | -0.008474 | -0.007134 |
| N5_common_adam | -0.013281 | -0.009284 | -0.007296 | +0.002114 | -0.010392 | -0.008830 | -0.005166 |
| N5_own_adam | -0.064555 | -0.005438 | +0.006352 | +0.021145 | -0.010875 | -0.040981 | +0.000389 |


主Holm族只有KR common-eta Adam/SGD的selected宏AUC两个比较；各组t95、全部2^8符号随机化、20,000次配对seed bootstrap均保存。符号随机化依赖零假设下符号交换性，t/bootstrap受n=8限制。其余固定轮、AULC、cell、加权读数、quota和ML均为预设次级pointwise，未获得全族显著性保证。delta_global=.003；区间跨0不等于等效，只有整个对应区间在±delta内才描述实际等效。AULC为离散epoch均值。

## AUC、loss、任务与模型状态

selected宏LL差（正为恶化）：

| group | SN | SB | delta | t95 | pos_neg |
| --- | --- | --- | --- | --- | --- |
| N1_common_adam | 0.439815 | 0.447272 | +0.007457 | [+0.004913, +0.010002] | 8/0 |
| N1_common_sgd | 0.444526 | 0.448554 | +0.004028 | [-0.004448, +0.012504] | 5/3 |
| N2_quota_adam | 0.444782 | 0.450989 | +0.006207 | [-0.000571, +0.012985] | 7/1 |
| N5_common_adam | 0.530695 | 0.551004 | +0.020309 | [+0.005769, +0.034849] | 7/1 |
| N5_own_adam | 0.530695 | 0.551235 | +0.020539 | [+0.004762, +0.036317] | 6/2 |

KR Adam selected训练J差+.006271，t95[+.003210,+.009332]，8/8正；宏LL也8/8恶化。SGD J差+.000680，区间跨0，AULC1..100宏AUC区间[−.001527,+.001754]在±.003内，但其selected仍不精确，不能混同估计对象。

KR Adam大域d1两task AUC差−.005673/−.004919，均8/8负，pointwise区间不跨0；最小域d2两task−.010055/−.010996，7/8负。d0/t0均值+.005627但4正4负、区间宽，d0/t1均值−.005203且5正3负，显示尾部和task差异。SGD d0/t0+.011187、7/8正，d2/t0−.006112、7/8负；这些是次级cell边界，不构成统一小域收益。

ML共同LR下最小域d1（N3000）两task约−.031801/−.025664；own约−.032892/−.034238。最大域d2（N150000）共同LR仅约+.000509/+.000880。完整baseline AUC/LL、test人口和标签率在FINAL_domain_context，不能用相对差定义内在难度。

## 规模、难度与取舍

| group | macro | train_p_cell | test_p_cell |
| --- | --- | --- | --- |
| N1_common_adam | -0.005203 | -0.005055 | -0.005277 |
| N1_common_sgd | +0.000837 | -0.000951 | -0.001638 |
| N2_quota_adam | -0.009191 | -0.004793 | -0.003571 |
| N5_common_adam | -0.010392 | -0.000482 | -0.012557 |
| N5_own_adam | -0.010875 | +0.000744 | -0.013586 |

上表都是同一选定预测的cell加权AUC，非pooled AUC。ML own宏均值负而train-p加权均值+.000744，其t95[−.000649,+.002138]在±.003内；test-count加权却−.013586。KR SGD宏略正而两种加权均值负。N0历史calAdam E50已精确显示同预测权重可翻转符号，详见METRIC_AND_POPULATION_DECOMPOSITION。

严格“两小域均改善且大域下降”在selected的seed计数：

| group | strict_both_small_gain_large_loss | either_small_gain_large_loss | reverse_both_small_loss_large_gain | practical_strict |
| --- | --- | --- | --- | --- |
| N1_common_adam | 0 | 6 | 0 | 0 |
| N1_common_sgd | 1 | 4 | 0 | 0 |
| N2_quota_adam | 1 | 1 | 0 | 0 |
| N5_common_adam | 0 | 1 | 2 | 0 |
| N5_own_adam | 0 | 2 | 3 | 0 |

KR Adam严格0/8，SGD1/8，按±.003实际界的严格模式均0/8；“至少一个小域改善”是更弱事件，不能替代前者。ML更常见小域下降/大域改善的反向模式。**WEAK/未识别：域大小是主因。** 三个自然域共变了标签率、输入、p、baseline、共享路径，两个task在同一N上还可异向；最小域不直接等于最困难域。计数波动很小不等于支持规模完全无作用，本轮没有独立操纵真实支持N或自然难度。

## 竞争解释及三种研究判定

- 评价权重与训练/评价人口：**STRONG（固定预测算术）**，可解释聚合符号，不单独识别域内人口因果。
- LR、外部预算与selection：**MEDIUM**，ML own晚期大幅改善与早期/selected下降并存；H200原rho规则未触发，不证明已收敛。
- 域计数/梯度分布：固定点排除计数主导为**STRONG（限定锚）**；长期quota交互不确定，域内分布/高阶矩/依赖及累计路径仍竞争。
- Adam即时均值漂移与历史：**WEAK/INCONCLUSIVE**；原生对比/MC RMSsum约.48–.62，固定分母后不明确消失，RMS不是CI。线性SGD理论身份不把有限bank非零范数变成显著漂移。未识别历史必要性；无动量SGD的历史反例原样保留。
- 共享C：只确认小且依状态/度量的局部作用；对长期唯一中介说法的证据为**WEAK**，N4未运行。课程与动量机制没有从这些结果得到有效性确认。

**GO-EFFECT（限定范围）：** 预定早期轨迹、固定ML人口及多个cell/度量有重复的条件差异；KR selected主Holm未通过，不能扩大。
**GO-MECHANISM（限定范围）：** 固定点域计数主导解释被实测反驳；局部C四格、阶段反例可操纵地量化，但长期必要性/唯一机制仍未识别。
**NO_GO-METHOD（本轮）：** 所有已列读数均没有同时满足预设宏收益及七项保护条件的比较，也没有被触发的新方法候选训练/新增方法依据。它不否定上述条件效应，也不证明未来方法均无效。

## 最有辨别力的下一实验

优先证伪“ML晚期优势只来自较小LR/较慢进度”：在同一固定ML人口、原模型上，用**新的独立优化seed做q×LR(.0003,.001)完整2×2**，固定预先选定外部steps/epochs，分别保留固定轮与selection读数、宏/六cell/LL。当前正式矩阵缺SN .0003的独立八seed格；若SB的晚期优势在同LR下不存活，支持LR/阶段解释；若同LR重复保持，则纯LR解释受反驳。它不操纵N，也不证明唯一机制。此为下一轮预注册建议，本轮不据test倒回开分支。

Q53另登记SGD primary的精度缺口：[−.004309,+.005984]同时覆盖实际伤害与收益。当前不加seed追显著；下一轮先固定精度目标和预算。

## 复核入口

- GLOBAL_MODEL_AND_PROTOCOL_FREEZE.json、FINAL_COMPARISONS.json、PRETEST_TRAINING_STAGE_CLOSURE.json：test前冻结，提交889e012。
- FINAL_TRAINING_RUN_LEDGER.csv：120条实际轨迹；TEST_READOUT_VALIDATION.json、TEST_RUN_LEDGER.csv：72条原始读数、43200 cell行。
- test_metadata/*/test_prediction_index.json与TEST_PREDICTION_ARTIFACT_INDEX.json：7200预测与模型SHA；二进制留原产物盘，不上传Git。
- FINAL_performance、N2_quota_performance、FINAL_method_guards、N2_absolute_quota_method_guards、FINAL_domain_context：完整seed表和保护界。
- FINAL_figures、N0/evaluation_decomposition.png及同名PDF、N2_figures、N3_figures：PNG/PDF已实际生成与视觉查看。设计/阶段授权保留签署时状态；最终状态以本报告与execution_report为准。
