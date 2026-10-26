# 实验结果总表

状态：COMPLETE。全部真实产物，不补造未运行条件。120条H100训练（48 tuning、72 formal）、16有效N2固定点锚、40个N3局部锚，72 formal×100 epoch test全部完成。统计单位为每组8训练seed，局部为每路径4seed；同轨迹别名和MC draw不增加n。

## selected test宏AUC

| group | SN | SB | delta | t95 | pos_neg |
| --- | --- | --- | --- | --- | --- |
| N1_common_adam | 0.667714 | 0.662511 | -0.005203 | [-0.012006, +0.001600] | 1/7 |
| N1_common_sgd | 0.648212 | 0.649049 | +0.000837 | [-0.004309, +0.005984] | 5/3 |
| N2_quota_adam | 0.667628 | 0.658437 | -0.009191 | [-0.015795, -0.002588] | 2/6 |
| N5_common_adam | 0.733318 | 0.722926 | -0.010392 | [-0.014599, -0.006185] | 0/8 |
| N5_own_adam | 0.733318 | 0.722444 | -0.010875 | [-0.014447, -0.007303] | 0/8 |

KR两主比较Holm p分别.171875/.6953125，均未通过；其余组为次级pointwise。Adam的t95跨0与bootstrap负区间并列保留。N1 own等于common。

## selected宏LL

| group | SN | SB | delta | t95 | pos_neg |
| --- | --- | --- | --- | --- | --- |
| N1_common_adam | 0.439815 | 0.447272 | +0.007457 | [+0.004913, +0.010002] | 8/0 |
| N1_common_sgd | 0.444526 | 0.448554 | +0.004028 | [-0.004448, +0.012504] | 5/3 |
| N2_quota_adam | 0.444782 | 0.450989 | +0.006207 | [-0.000571, +0.012985] | 7/1 |
| N5_common_adam | 0.530695 | 0.551004 | +0.020309 | [+0.005769, +0.034849] | 7/1 |
| N5_own_adam | 0.530695 | 0.551235 | +0.020539 | [+0.004762, +0.036317] | 6/2 |

## 全部固定阶段宏AUC差

| group | E10 | E20 | E50 | E100 | selected | AULC1_20 | AULC1_100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N1_common_adam | -0.008530 | -0.009708 | -0.007246 | -0.007422 | -0.005203 | -0.007480 | -0.005324 |
| N1_common_sgd | +0.002001 | -0.003312 | +0.000743 | -0.000877 | +0.000837 | -0.001693 | +0.000113 |
| N2_quota_adam | -0.007071 | -0.007078 | -0.009556 | -0.009643 | -0.009191 | -0.008474 | -0.007134 |
| N5_common_adam | -0.013281 | -0.009284 | -0.007296 | +0.002114 | -0.010392 | -0.008830 | -0.005166 |
| N5_own_adam | -0.064555 | -0.005438 | +0.006352 | +0.021145 | -0.010875 | -0.040981 | +0.000389 |

ML own E100+.021145、8/8正与selected−.010875、8/8负并存，禁止只选一个作普遍结论；q及LR共同改变。七项保护没有任何数值联合通过，E100 own失败来自d1两task宽区间，不是所有cell点均值损伤。

## quota四臂

| effect | mean | t95 | pos_neg |
| --- | --- | --- | --- |
| IID_SB_minus_SN | -0.005203 | [-0.012006, +0.001600] | 1/7 |
| quota_SB_minus_SN | -0.009191 | [-0.015795, -0.002588] | 2/6 |
| quota_interaction | -0.003988 | [-0.010310, +0.002333] | 3/5 |
| SN_quota_minus_IID | -0.000085 | [-0.003289, +0.003118] | 4/4 |
| SB_quota_minus_IID | -0.004074 | [-0.010089, +0.001941] | 3/5 |

绝对四臂、所有cell/LL/J/时钟见N2_quota_performance；交互区间跨0。固定点域计数份额小而SB/SN trace仍约2倍，不能用trace直接预测AUC。

## 来源与完整数据

FINAL_SELECTED_ENDPOINTS.csv、FINAL_STAGE_MACRO_AUC.csv、FINAL_SELECTED_CELL_METRICS.csv为可读汇总；FINAL_performance/paired_seed_endpoints.csv及seed_summary.json保存全原始seed、两区间、符号随机化。FINAL_domain_context给各域大小、标签率、baseline和取舍计数。N0历史评价/校准、N2原始矩、N3每draw四格分开存放，不能混为独立重复。详细解释见FINAL_ANALYSIS、METRIC_AND_POPULATION_DECOMPOSITION、Q_LR_BUDGET_RESPONSE、DOMAIN_COUNT_COUNTERFACTUAL、C_LOCAL_AND_TOPOLOGY_ANALYSIS和QUESTION_REGISTRY_FINAL。

主Holm族只有KR common-eta Adam/SGD的selected宏AUC两个比较；各组t95、全部2^8符号随机化、20,000次配对seed bootstrap均保存。符号随机化依赖零假设下符号交换性，t/bootstrap受n=8限制。其余固定轮、AULC、cell、加权读数、quota和ML均为预设次级pointwise，未获得全族显著性保证。delta_global=.003；区间跨0不等于等效，只有整个对应区间在±delta内才描述实际等效。AULC为离散epoch均值。
