# N3 开发结果与一次候选冻结

状态：16/16开发锚完成且通过N3_development_validation.json；独立确认尚未运行，SGD整体C对照仍在执行，N4尚未触发。新模型test仍封存。本文件是开发阶段分析，不是整轮最终结论。按start.md §11由主agent实施/自审/分析，independent_review=false。

固定Adam η=.01，四训练seed2032–2035、SN/SB双原生路径、E10/E40，16个条件锚。每锚32成对draw，native optimizer一步，每draw从同θ/m/v/step重置，不是连续32步训练。AUC/LL使用完整validation两半，J扫完整训练人口。独立重复单位是四个训练seed，不是512个draw，也不将两路径或两个阶段当额外训练seed。以下均为diagnostic宏AUC，正AUC为改善、负LL/J为改善。

## 一次提名

按预先写定的五子块×两阶段最大跨路径完整替换均值绝对值规则，冻结 **C_embedding、E10、完整替换与方向残差预期符号均为正**。跨路径完整替换均值为+0.0000299721，低于delta_local=.0001。没有添加开发显著性或实际量级提前停止门槛；仍须其余2036–2039、双路径共8锚独立确认。候选文件保持原SHA，不依据确认或后续读数重选。

| anchor_arm | case | estimand | mean | positive | negative | seed_t95_low | seed_t95_high | mc_se |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SB | C_embedding_factorial | direction_residual | 0.00000633 | 2 | 2 | -0.00004935 | 0.00006200 | 0.00001392 |
| SB | C_embedding_full | vs_SB | 0.00001059 | 2 | 2 | -0.00003925 | 0.00006043 | 0.00001443 |
| SN | C_embedding_factorial | direction_residual | 0.00000695 | 2 | 2 | -0.00007406 | 0.00008796 | 0.00001713 |
| SN | C_embedding_full | vs_SB | 0.00004935 | 3 | 1 | -0.00001609 | 0.00011480 | 0.00001898 |

四个相关路径/效应的MCSE均≤.000025，故不触发开发32→64。训练seed区间仍宽，MC精度不等于训练seed效应已确认。当前候选完整替换两路径均值分别约+.00004935/+.00001059，方向残差约+.00000695/+.00000633；均低于局部实际界，区间跨零。**WEAK：没有获得候选的正向独立机制证据。** 开发筛选后的逐点区间只作描述，不用其声称正式等效；独立确认将依原N3.6检查两路径的实际界、方向、3/4及MC精度。

## 整体C四格与交互

以下是相对native SB的signed变化，交互为F(full)−F(norm)−F(direction)+F(SB)，不将效应相加解释成归因百分比。native SB这一格的相对差为定义上的0；实际绝对指标及vs_anchor/vs_SN均保留在主汇总表。

| epoch | anchor_arm | case | estimand | mean | positive | negative | seed_t95_low | seed_t95_high | mc_se |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | SB | C_direction | vs_SB | -0.00008381 | 2 | 2 | -0.00034371 | 0.00017609 | 0.00003745 |
| 10 | SB | C_factorial | norm_direction_interaction | 0.00000198 | 3 | 1 | -0.00001673 | 0.00002069 | 0.00000397 |
| 10 | SB | C_full | vs_SB | -0.00006271 | 2 | 2 | -0.00027414 | 0.00014872 | 0.00003662 |
| 10 | SB | C_norm | vs_SB | 0.00001912 | 3 | 1 | -0.00002894 | 0.00006719 | 0.00000317 |
| 10 | SN | C_direction | vs_SB | -0.00004290 | 1 | 3 | -0.00025944 | 0.00017363 | 0.00004423 |
| 10 | SN | C_factorial | norm_direction_interaction | -0.00000558 | 2 | 2 | -0.00002769 | 0.00001653 | 0.00000683 |
| 10 | SN | C_full | vs_SB | 0.00007011 | 4 | 0 | -0.00003484 | 0.00017506 | 0.00004331 |
| 10 | SN | C_norm | vs_SB | 0.00011860 | 4 | 0 | 0.00001845 | 0.00021874 | 0.00000623 |
| 40 | SB | C_direction | vs_SB | -0.00004044 | 1 | 3 | -0.00017974 | 0.00009886 | 0.00004591 |
| 40 | SB | C_factorial | norm_direction_interaction | 0.00000151 | 3 | 1 | -0.00000895 | 0.00001196 | 0.00000306 |
| 40 | SB | C_full | vs_SB | -0.00002008 | 2 | 2 | -0.00015356 | 0.00011340 | 0.00004553 |
| 40 | SB | C_norm | vs_SB | 0.00001886 | 3 | 1 | -0.00001298 | 0.00005070 | 0.00000246 |
| 40 | SN | C_direction | vs_SB | -0.00004861 | 1 | 3 | -0.00013694 | 0.00003971 | 0.00004049 |
| 40 | SN | C_factorial | norm_direction_interaction | 0.00000558 | 3 | 1 | -0.00000995 | 0.00002112 | 0.00000682 |
| 40 | SN | C_full | vs_SB | 0.00005106 | 2 | 2 | -0.00007054 | 0.00017267 | 0.00003914 |
| 40 | SN | C_norm | vs_SB | 0.00009409 | 4 | 0 | 0.00003853 | 0.00014965 | 0.00000560 |

整体C完整替换在SN路径E10/E40均值略正，SB路径均值略负；全部相应训练seed区间跨零且跨局部实际量级的一侧。E10 SN虽4/4正，SB仍2正2负。方向残差的点均值在四组均为负，不能将完整替换的正均值自动归为方向收益。**WEAK：这些数据不支持跨路径稳定的整体C正向归因，也不证明C没有作用。** 整体C的MCSE高于候选的精度，原冻结补充规则只针对唯一候选，不为搜索阳性展开所有块的64draw矩阵。

## 训练目标、损失和六cell反例

| anchor_arm | split | metric | case | estimand | mean | seed_t95_low | seed_t95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SB | diagnostic | logloss | C_embedding_factorial | direction_residual | -0.00000323 | -0.00007081 | 0.00006435 |
| SB | train | train_J | C_embedding_factorial | direction_residual | -0.00002850 | -0.00003508 | -0.00002192 |
| SB | diagnostic | logloss | C_embedding_full | vs_SB | -0.00001142 | -0.00005655 | 0.00003370 |
| SB | train | train_J | C_embedding_full | vs_SB | -0.00003412 | -0.00004224 | -0.00002599 |
| SN | diagnostic | logloss | C_embedding_factorial | direction_residual | -0.00011692 | -0.00032560 | 0.00009177 |
| SN | train | train_J | C_embedding_factorial | direction_residual | -0.00002567 | -0.00010513 | 0.00005379 |
| SN | diagnostic | logloss | C_embedding_full | vs_SB | -0.00005320 | -0.00014703 | 0.00004063 |
| SN | train | train_J | C_embedding_full | vs_SB | -0.00008770 | -0.00013091 | -0.00004448 |

完整embedding替换的训练J点均值改善，两路径四seed区间均在负侧；diagnostic LL点均值也改善但区间跨零，macro AUC效应很小。不能用训练J改善替代排序收益，也不把一步替换称为有长期收益的方法。该模块由AUC筛选，所有这些结果仍是开发性读数。

六cell完整替换相对SB的点估计如下；每格的训练seed区间、符号、方向残差与MCSE见candidate_cell_counterexamples.csv，不将12个格子视为独立模型重复。

| anchor_arm | domain | task | AUC_delta | positive | negative | LL_delta |
| --- | --- | --- | --- | --- | --- | --- |
| SB | 0 | 0 | 0.00001573 | 3 | 1 | -0.00000143 |
| SB | 0 | 1 | -0.00000452 | 1 | 3 | -0.00000147 |
| SB | 1 | 0 | 0.00000840 | 4 | 0 | -0.00002313 |
| SB | 1 | 1 | 0.00000962 | 3 | 1 | -0.00002228 |
| SB | 2 | 0 | 0.00001963 | 3 | 1 | -0.00000805 |
| SB | 2 | 1 | 0.00001468 | 3 | 1 | -0.00001218 |
| SN | 0 | 0 | 0.00007956 | 4 | 0 | -0.00000966 |
| SN | 0 | 1 | 0.00002492 | 3 | 1 | -0.00000786 |
| SN | 1 | 0 | 0.00005345 | 4 | 0 | -0.00011946 |
| SN | 1 | 1 | 0.00005688 | 4 | 0 | -0.00010266 |
| SN | 2 | 0 | 0.00003682 | 2 | 2 | -0.00003208 |
| SN | 2 | 1 | 0.00004449 | 4 | 0 | -0.00004747 |

这项局部干预没有操纵域大小或自然难度。不同task、anchor路径与训练阶段须保留，不能把小域直接定义为困难域，也不能由当前小局部效应推导小域收益/大域受损的一般取舍。

## 参数数与实际干预

| block | coordinates |
| --- | --- |
| C | 661968 |
| P | 863238 |
| R | 6168 |
| E | 82880 |
| C_embedding | 44752 |
| C_shared_STAR | 279040 |
| C_skip | 140032 |
| C_star_mlp | 131840 |
| C_experts | 66304 |
| ALL | 1614254 |

物理控制检查：全16锚的最大实际/目标范数相对误差为2.24813663e-05，位于seed2035、SB路径E40的C_shared_STAR_norm。该case的32个draw预测SHA均与native SB相同，但实际块更新非零；这是特定状态的功能敏感性反例，不证明该模块全程无用或确定激活原因。目标缺失/零的相对误差保留未定义。**STRONG（仅实施身份与已测物理控制）**：锚参数未变、源/配置/检查点SHA通过，所记录范数控制达到原更新的数值量级；这不是算法收益结论。

## 证据边界和已授权下一步

N2全参数trace的SB/SN约2倍，与这里小且路径不稳定的一步AUC作用并存。方差差异不自动等于性能损害或均值漂移，更不证明共享C为唯一中介。N3独立确认是当前最直接的证伪检验：固定这个模块/阶段/正方向，不能转而选新的开发候选。若不满足规则，N4记录未触发，不当作所有共享拓扑无效；若满足，必须执行已设计的长期拓扑矩阵，即使没有方法收益。

产物：N3_development_summary/conditional_summary.csv、seed_estimands.csv、cell_conditional_summary.csv、candidate_cell_counterexamples.csv、actual_block_update_norms.csv和parameter_counts.csv。原始macro/cell/draws/config/receipt按锚保留，N3_development_validation.json逐文件SHA固定。N3_confirmation_manifest.json引用独立2036–2039的原生E10检查点及本候选SHA。确认采用原队列PENDING=0后的既有队列准入，不修改健康训练源码、不暂停运行中的实验。所有新test须等全部规定/触发训练及选择闭合后统一解封。
