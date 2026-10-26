# C局部作用与条件拓扑判定

状态：N3全部完成、通过核验；N4按原门槛未触发。此为已闭合的机制结果；整轮N5和统一test也已完成。主agent设计/实施/自审/分析，independent_review=false。

## 设计与一次冻结

Adam .01、KR固定人口，开发2032–2035×两原生路径×E10/E40共16锚、32draw；整体C、五子块norm×direction四格及P完整参考。N3_DEVELOPMENT_ANALYSIS.md给全量开发、LL/J、六cell及实际更新范数；不是选最大后当独立复制。一次冻结C_embedding/E10、full和direction residual预期均为正，原SHA713c82008518d85936ca8df007047aff92209dc8a3e7352637c7fc44ab0e508c。开发need64=false，未重新提名。SGD只按预设重复整体C16锚，见N3_SGD_CONTROL_ANALYSIS.md，不据其重选子块。

独立训练seed2036–2039、两路径共8锚，同一冻结模块/阶段与原生状态，独立成对draw bank，各32draw。N3_confirmation_validation.json核对8/8完整身份、来源和读数。重复单位为4训练seed，每条路径单独判断；两路径不是8个独立训练seed，draw也不是训练重复。diagnostic人口早已用于资源触发，不称新holdout。

## 独立确认全部门槛

SN/SB每路径分别要求full−SB和full−norm方向残差均达到.0001、预期正方向、至少3/4同号，且四seed平均MCSE≤.000025。

| effect | mean | positive | negative | seed_t95_low | seed_t95_high | mc_se | effect_rule_pass | precision_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SN_full | 0.00003252 | 4 | 0 | 0.00000185 | 0.00006320 | 0.00001955 | False | True |
| SN_direction | 0.00001480 | 2 | 2 | -0.00005343 | 0.00008302 | 0.00001780 | False | True |
| SB_full | 0.00002817 | 4 | 0 | -0.00000105 | 0.00005739 | 0.00001238 | False | True |
| SB_direction | 0.00001408 | 3 | 1 | -0.00002368 | 0.00005185 | 0.00001185 | False | True |

四项MC精度均通过，因此不触发32→64；四项实际量级均未达到.0001，SN方向残差仅2/4正。N1重要残差为真，但独立局部联合门槛为假，故**N4 NOT_TRIGGERED**。这不是拓扑阴性实验，未进行的tied/真实untied/随机untied不能记作失败。

**MEDIUM（限定点位、pointwise）**：完整替换在两个独立路径均4/4正，但均值仅约3e−5；SN full的t95在0上方，SB full的区间略跨0。所有四个相应pointwise区间均完整位于±.0001内，可称这些状态的独立局部效应在该实际界内近似消失；不是联合多重检验后的普遍等效保证，更不是证明embedding/C长期无用。方向残差符号比完整替换不稳定，不能把全更新的微小收益全归于方向。未改候选、实际界、方向或独立seed。

## 损失、目标及反例

负LL/J表示改善，J为完整训练人口经验BCE；不能用batch loss或AUC替代。

| anchor_arm | case | estimand | metric | mean | positive | negative | seed_t95_low | seed_t95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | C_embedding_factorial | direction_residual | logloss | -0.00002361 | 1 | 3 | -0.00009451 | 0.00004729 |
| SB | C_embedding_factorial | direction_residual | train_J | -0.00003202 | 0 | 4 | -0.00006539 | 0.00000135 |
| SB | C_embedding_full | vs_SB | logloss | -0.00003124 | 0 | 4 | -0.00009784 | 0.00003537 |
| SB | C_embedding_full | vs_SB | train_J | -0.00004199 | 0 | 4 | -0.00007360 | -0.00001038 |
| SN | C_embedding_factorial | direction_residual | logloss | -0.00001399 | 2 | 2 | -0.00028243 | 0.00025445 |
| SN | C_embedding_factorial | direction_residual | train_J | 0.00000743 | 2 | 2 | -0.00012124 | 0.00013609 |
| SN | C_embedding_full | vs_SB | logloss | -0.00007289 | 0 | 4 | -0.00013269 | -0.00001309 |
| SN | C_embedding_full | vs_SB | train_J | -0.00011055 | 0 | 4 | -0.00016780 | -0.00005331 |

完整embedding替换的两路径J和LL点均值均改善；SN LL/J的pointwise区间在负侧，SB J也在负侧，SB LL区间跨0。方向残差的SN J点均值略恶化且符号2/2。训练目标、概率loss与排序AUC不互作证明；一步替换不是长期方法。六cell反例、原始seed与实际范数见N3_confirmation_summary/candidate_cell_counterexamples.csv、cell_seed_estimands.csv及actual_block_update_norms.csv。

## 整体C、SGD与长期边界

Adam开发整体Cfull在SN路径略正、SB路径略负，相关t95均跨0；其方向残差四组均值均为负。SGD整体Cfull四组均值小、pointwise t95落在±.0001内，但E40 SB方向残差区间越过负实际界；并非所有方向读数等效。不同优化器的eta和训练状态不同，不能当只切换Adam归一化的纯因果对照。同bank仿射SGD及固定滞后分母对照另见N2_FIXEDPOINT_ANALYSIS.md，Adam总体瞬时均值漂移仍INCONCLUSIVE。

实际坐标数/落地范数、极小shared STAR更新和相同预测SHA反例全保留。N2较大q方差比与小局部AUC作用共存，方差差不自动推出性能损害、共享C唯一中介或历史必要性。当前对这些普遍因果说法的证据为**WEAK**，尚未识别；不启动shape干预来制造所需主张。

## 条件闭合与后续

CONDITIONAL_BRANCH_DECISIONS_PRETEST.json在新test前固定范围：N4未触发；N5-A/B/D及shape不承担本轮所作的总q干预/固定点计数判断，纯顺序、重复独立中介、自然难度与shape归因均保留未知。N5-C固定ML人口按触发完成16对称pilot及24去重正式轨迹（8优化seed）。没有新方法候选训练不构成所有未来方法无效。所有test已在必做训练与选择闭合后统一读取；其最终表现见FINAL_ANALYSIS，N3局部统计仍仅使用原diagnostic，不与test混合。
