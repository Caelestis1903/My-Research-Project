# N3 SGD整体C局部对照

状态：COMPLETE_VALIDATED，16/16原始锚无失败；N3_SGD_control_validation.json逐文件SHA与配置/锚/源码/32draw/六cell完整性核验PASS。主agent分析/自审，independent_review=false。未读本轮新模型test。

两优化器均触发N1.6后，按照N3.1只在SGD重复整体C四格，不筛SGD子块、不重新选择Adam候选。原SGD共同eta1.0，2032–2035×SN/SB原生路径×E10/E40共16条件锚，每锚32成对draw；训练重复单位是4seed。这个对照与Adam的模型状态、训练路径和学习率不同；不能把跨优化器差异当仅切换更新映射的纯因果效应。同bank单位eta仿射SGD对照另见N2_fixedpoint_summary/SGD_affine_mapping_control.csv。

## 完整四格、方向残差与交互

以下为diagnostic macro AUC，正值改善，t95为四训练seed的pointwise区间。native SB相对自身为0，绝对性能、vs_SN和vs_anchor均在全表保留；交互不是归因百分比。

| epoch | anchor_arm | case | estimand | mean | positive | negative | seed_t95_low | seed_t95_high | mc_se |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | SB | C_direction | vs_SB | 0.00000271 | 3 | 1 | -0.00001428 | 0.00001970 | 0.00000557 |
| 10 | SB | C_factorial | direction_residual | 0.00000311 | 3 | 1 | -0.00001215 | 0.00001836 | 0.00000529 |
| 10 | SB | C_factorial | norm_direction_interaction | 0.00000039 | 2 | 2 | -0.00000700 | 0.00000778 | 0.00000216 |
| 10 | SB | C_full | vs_SB | 0.00000389 | 2 | 2 | -0.00001025 | 0.00001804 | 0.00000649 |
| 10 | SB | C_norm | vs_SB | 0.00000079 | 2 | 2 | -0.00000438 | 0.00000595 | 0.00000396 |
| 10 | SN | C_direction | vs_SB | -0.00000269 | 3 | 1 | -0.00002122 | 0.00001585 | 0.00000539 |
| 10 | SN | C_factorial | direction_residual | -0.00000180 | 3 | 1 | -0.00002080 | 0.00001721 | 0.00000533 |
| 10 | SN | C_factorial | norm_direction_interaction | 0.00000089 | 3 | 1 | -0.00000465 | 0.00000643 | 0.00000253 |
| 10 | SN | C_full | vs_SB | -0.00000388 | 1 | 3 | -0.00001391 | 0.00000615 | 0.00000627 |
| 10 | SN | C_norm | vs_SB | -0.00000208 | 1 | 3 | -0.00001916 | 0.00001500 | 0.00000342 |
| 40 | SB | C_direction | vs_SB | -0.00005914 | 0 | 4 | -0.00013648 | 0.00001820 | 0.00003299 |
| 40 | SB | C_factorial | direction_residual | -0.00003859 | 1 | 3 | -0.00011985 | 0.00004267 | 0.00003147 |
| 40 | SB | C_factorial | norm_direction_interaction | 0.00002055 | 4 | 0 | -0.00000558 | 0.00004667 | 0.00000654 |
| 40 | SB | C_full | vs_SB | -0.00001891 | 1 | 3 | -0.00007303 | 0.00003522 | 0.00003264 |
| 40 | SB | C_norm | vs_SB | 0.00001968 | 4 | 0 | -0.00000821 | 0.00004758 | 0.00001014 |
| 40 | SN | C_direction | vs_SB | -0.00002388 | 1 | 3 | -0.00009257 | 0.00004481 | 0.00002455 |
| 40 | SN | C_factorial | direction_residual | -0.00003072 | 1 | 3 | -0.00009211 | 0.00003068 | 0.00002441 |
| 40 | SN | C_factorial | norm_direction_interaction | -0.00000683 | 1 | 3 | -0.00002028 | 0.00000661 | 0.00000603 |
| 40 | SN | C_full | vs_SB | -0.00000173 | 1 | 3 | -0.00003864 | 0.00003517 | 0.00002563 |
| 40 | SN | C_norm | vs_SB | 0.00002898 | 4 | 0 | -0.00002240 | 0.00008036 | 0.00000771 |

**MEDIUM（限于这些预定状态、未作多重校正的pointwise界）**：整体C完整替换的四组均值均远小于.0001，四个相应t95均完整位于±.0001内，可描述为“这些条件的整体C完整替换在该实际界内近似消失”。这不表示所有C方向读数等效，更不表示长期C拓扑无作用。E40 SB方向残差区间越过−.0001，direction-only四seed均负但其CI跨零；保留此反例和MCSE约3.1e−5的精度限制。没有按对照结果追加子块筛选或64draw；既定补draw规则只针对已冻结的Adam独立候选。

SGD E10整体Cfull在SN/SB路径点均值约−3.88e−6/+3.89e−6，E40约−1.73e−6/−1.89e−5。Adam对应开发均值SN略正、SB略负；双方都不支持“整体C方向收益跨路径稳定”的泛化主张。跨优化器差异还混合eta、原生状态与训练阶段，因此证据不能唯一指向Adam非线性或历史必要性。

## LL与完整训练目标

负LL/J为改善；J使用原始完整训练人口经验BCE目标，不用AUC替代。

| epoch | anchor_arm | split | metric | mean | positive | negative | seed_t95_low | seed_t95_high | mc_se |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | SB | diagnostic | logloss | -0.00000194 | 1 | 3 | -0.00000986 | 0.00000598 | 0.00000533 |
| 10 | SB | train | train_J | -0.00000480 | 0 | 4 | -0.00001573 | 0.00000613 | 0.00000090 |
| 10 | SN | diagnostic | logloss | -0.00000367 | 1 | 3 | -0.00001433 | 0.00000700 | 0.00000574 |
| 10 | SN | train | train_J | -0.00000466 | 0 | 4 | -0.00001124 | 0.00000192 | 0.00000129 |
| 40 | SB | diagnostic | logloss | 0.00004276 | 4 | 0 | -0.00001140 | 0.00009692 | 0.00007366 |
| 40 | SB | train | train_J | -0.00009208 | 0 | 4 | -0.00018825 | 0.00000409 | 0.00004138 |
| 40 | SN | diagnostic | logloss | 0.00000866 | 2 | 2 | -0.00007136 | 0.00008868 | 0.00005752 |
| 40 | SN | train | train_J | -0.00008237 | 0 | 4 | -0.00015315 | -0.00001160 | 0.00004671 |

所有整体Cfull的J点均值均改善且每组4/4负；E40 SN的pointwise t95在负侧。E40 SB的diagnostic LL均值却恶化且4/4正，CI仍跨零。训练目标、验证loss和排序AUC不可互换；这些一步替换没有构成长期方法。

## 实施控制与保留限制

整体C坐标661968，ALL1614254，与Adam使用同参数空间。没有方向未定义draw。实际/目标范数的最大相对误差为9.21979765e-05；逐draw实际范数、目标误差、与native SB的prediction SHA身份计数均在actual_block_update_norms.csv与seed_summary保留，不按坐标数推测影响。

全六cell AUC/LL反例见whole_C_cell_counterexamples.csv和cell_seed_estimands.csv；不能将task或路径当新训练重复，也没有操纵域大小/难度。32draw在固定状态下量化条件采样，不能变成独立训练样本数。

Adam唯一候选仍为原SHA的C_embedding/E10，必须完成2036–2039独立8锚确认。N4只依原N1.6与该独立门槛决定，SGD对照不能替代或修改门槛。quota16长程与N5-C继续按既定触发执行。
