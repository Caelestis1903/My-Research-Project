# P0 机制复核

P0 已完成，退出码 0；本次缓存诊断内部墙钟 10.697 秒。以旧模型状态做冻结前向和小型预测头拟合，没有新增完整主训练轨迹。以下数值由本目录的原始机器工件聚合，主 agent 负责科学解释。

## 独立核验与状态身份

48 条旧核心的 selected/epoch24 共 96 份预测，六格、macro、task-pooled 指标重新计算与旧机器表最大绝对差为 0；所有 row_id 与 Q_explore 一致，三个源码归档逐文件 hash 一致。关于身份与复算的一致性，证据为 STRONG；这不是算法有效性的强证据。

HIST/ridge/NN 相对 STATIC 的两数据×两 seed×两个 checkpoint 共 24 对状态，模型、真实 optimizer、梯度、torch/CUDA RNG 摘要全部相同，见 `artifacts/mdmt_next_v3/v3_0/p0/old_rhd_state_identity.csv`。这为这些已评价快照没有额外训练路径提供 STRONG 证据，不外推到所有时刻或所有状态算法。

## R1：旧拒绝链

下表只统计 10.5–24.0 的 28 个控制期事件，不与旧报告仅后 14 事件的 gamma 计数混用。历史门在每条 trace 上拒绝了全部 28 次非回退提议；state 有时 gamma>0，但实际动作仍不改变历史动作。收缩只在部分 ML 事件改变候选，不能把全部失败归因于收缩，也不能把 gamma 非零当成实际控制。

| dataset | seed | policy | events | shrink_changes | history_rejections | gamma_positive | actual_changes_vs_history |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kuairand | 3101 | HIST_SHRINK | 28 | 0 | 28 | 0 | 0 |
| kuairand | 3101 | NN_RESIDUAL | 28 | 0 | 28 | 0 | 0 |
| kuairand | 3101 | RIDGE_RESIDUAL | 28 | 0 | 28 | 11 | 0 |
| kuairand | 3102 | HIST_SHRINK | 28 | 0 | 28 | 0 | 0 |
| kuairand | 3102 | NN_RESIDUAL | 28 | 0 | 28 | 5 | 0 |
| kuairand | 3102 | RIDGE_RESIDUAL | 28 | 0 | 28 | 9 | 0 |
| movielens | 3101 | HIST_SHRINK | 28 | 2 | 28 | 0 | 0 |
| movielens | 3101 | NN_RESIDUAL | 28 | 2 | 28 | 3 | 0 |
| movielens | 3101 | RIDGE_RESIDUAL | 28 | 2 | 28 | 10 | 0 |
| movielens | 3102 | HIST_SHRINK | 28 | 6 | 28 | 0 | 0 |
| movielens | 3102 | NN_RESIDUAL | 28 | 6 | 28 | 5 | 0 |
| movielens | 3102 | RIDGE_RESIDUAL | 28 | 6 | 28 | 7 | 0 |

此结果将瓶颈定位到已保存的选择链，证据 STRONG。关于取消门槛是否会提高最终泛化，证据仍 WEAK；v3 不据此预设放松门槛必然有效。

## R2：严格前缀的简单历史机会

表中为公共 BASE 状态上后 14 个事件的 P_cal 宏收益相对 STATIC，正数有利。动作只由此前 P_update 决定；FREEZE 不在自身 12 个选择事件上评分。EMA 与 EMA_WINDOW32 在这些实际动作上相同，保留两列以明示窗口敏感性结果。

| dataset | seed | CUM | EMA | EMA_WINDOW32 | FREEZE | STATIC |
| --- | --- | --- | --- | --- | --- | --- |
| kuairand | 3101 | 0.00012511 | 0.00008154 | 0.00008154 | 0.00002938 | 0.00000000 |
| kuairand | 3102 | 0.00005790 | 0.00006674 | 0.00006674 | 0.00006936 | 0.00000000 |
| movielens | 3101 | 0.00026035 | 0.00011987 | 0.00011987 | 0.00031497 | 0.00000000 |
| movielens | 3102 | 0.00048823 | 0.00012200 | 0.00012200 | 0.00040321 | 0.00000000 |

CUM/EMA/FREEZE 四条 trace 平均均正，支持同状态机会，证据 MEDIUM；它们没有改变公共 BASE 轨迹，因此不能据此声称完整闭环有效。H5 只有两份旧 epoch10 快照，8 条记录分别包含不同策略，不是 8 个独立训练状态，也不是四条完整 H5 轨迹。

## L1：树已经生长，部署深度仍为 0

10 个可得快照的六棵树均实际生长，非根节点也有不同于根动作的通过项。因此“结构阶段从未找到分区”被这些快照推翻。全部部署 cap=0，其中 8/10 的深度 2 P_cal 原始 LL 更低；原实现按最佳深度的 jackknife SE 选择最浅合格深度，说明最终共同深度选择实际抑制了这些已存在的分区。

| dataset | seed | checkpoint | selected_depth | grown_trees | nonroot_different_action | depth2_minus_depth0_LL |
| --- | --- | --- | --- | --- | --- | --- |
| movielens | 3101 | S1_epoch10 | 0 | 6 | 3 | 0.00002241 |
| movielens | 3101 | primary_best | 0 | 6 | 26 | -0.00072935 |
| movielens | 3101 | epoch24 | 0 | 6 | 18 | -0.00037313 |
| movielens | 3102 | primary_best | 0 | 6 | 11 | -0.00007823 |
| movielens | 3102 | epoch24 | 0 | 6 | 21 | -0.00052053 |
| kuairand | 3101 | S1_epoch10 | 0 | 6 | 7 | -0.00000462 |
| kuairand | 3101 | primary_best | 0 | 6 | 5 | 0.00000289 |
| kuairand | 3101 | epoch24 | 0 | 6 | 15 | -0.00002223 |
| kuairand | 3102 | primary_best | 0 | 6 | 12 | -0.00006720 |
| kuairand | 3102 | epoch24 | 0 | 6 | 12 | -0.00018409 |

这是快照限定的 STRONG 结构/选择事实；更深树是否提高 Q_explore 或完整训练没有被这里证明。缺少全部候选、原始未收缩叶风险和中间刷新表，仍不能量化每一层独立责任。现有证据已足以排除“没有长树”这一解释，本轮不追加三种可选替换来挑更有利的旧结果。

## L2：完整容量和可读性

固定各自 epoch10 native/reader，逐来源、逐目标、逐块和成对响应均已保存。下表是六个有向 source→target 对先各自平均两任务后的描述性均值；这些 pair 不是独立 seed。ACTUAL_RHO 是旧责任聚合，FULL_SUM 是直接相加的反事实读取，不是新方法的完整训练。

| dataset | K | variant | mean_delta_LL | mean_delta_AUC | beneficial_source_target_pairs |
| --- | --- | --- | --- | --- | --- |
| kuairand | 1 | ACTUAL_RHO | -0.00042289 | 0.00020631 | 5 |
| kuairand | 1 | FULL_SUM | -0.00042289 | 0.00020631 | 5 |
| kuairand | 4 | ACTUAL_RHO | 0.00000126 | -0.00000765 | 2 |
| kuairand | 4 | FULL_SUM | 0.00003163 | -0.00004928 | 2 |
| movielens | 1 | ACTUAL_RHO | -0.00006029 | 0.00007819 | 4 |
| movielens | 1 | FULL_SUM | -0.00006029 | 0.00007819 | 4 |
| movielens | 4 | ACTUAL_RHO | 0.00009568 | 0.00015169 | 3 |
| movielens | 4 | FULL_SUM | 0.00187554 | 0.00034056 | 2 |

旧 K4 全块直接相加在 ML 平均 AUC 上升而 LL 恶化，不能将其解释成单纯“可读容量越大越好”；改变求和也改变幅度。旧 K1 的 owner 风险明显受益，K4 的 owner/target 作用更小且依赖域。目标响应并非处处不可读：K4 ACTUAL_RHO 在 ML 3/6、KR 2/6 有向对的两任务平均 LL 有改善，单块/成对也有混合收益。因此未满足“owner 有用但 target 完全不可读”这一小型额外 receiver 对照触发条件，不追加该拟合。单 seed、早期固定状态的可迁移性解释证据 WEAK。

## L3：单调校准、额外条件预测与来源几何

MONO 与无 foreign key 的 34 维修正使用相同旧 P_label 和 P_cal 选择；原 CAL42 和原概率选择器来自保存的原策略。全部在 Q_explore 评分，完整六格及预测位于本地工件。下表保留四种风险与排序值。

| dataset | K | macro_auc_MONO | macro_auc_NO_FOREIGN_KEY34 | macro_auc_ORIGINAL_CAL_OFF42 | macro_auc_OWN_OFF | macro_logloss_MONO | macro_logloss_NO_FOREIGN_KEY34 | macro_logloss_ORIGINAL_CAL_OFF42 | macro_logloss_OWN_OFF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kuairand | 1 | 0.69176266 | 0.69217179 | 0.69261436 | 0.69176266 | 0.46746667 | 0.46822965 | 0.46814607 | 0.47062722 |
| kuairand | 4 | 0.68335374 | 0.68392558 | 0.68364479 | 0.68335374 | 0.46903226 | 0.46948380 | 0.46966281 | 0.47131154 |
| movielens | 1 | 0.80814068 | 0.80904653 | 0.80903112 | 0.80814068 | 0.47464318 | 0.47390782 | 0.47390724 | 0.47567823 |
| movielens | 4 | 0.80719957 | 0.80823081 | 0.80824999 | 0.80719957 | 0.47572258 | 0.47466717 | 0.47466397 | 0.47629291 |

MONO 的 cell AUC 保持不变，而条件修正可以改变排序。ML K4 的无 key 修正 AUC 0.808231 与旧 CAL42 的 0.808250 接近，LL 0.474667 与 0.474664 接近；KR K4 无 key 修正也没有被 CAL42 压倒。因而不能将旧 CAL 的所有价值解释为来源几何，也不能将其全部称为单调校准。额外特征/标签是有力竞争解释，在这些冻结状态上证据 MEDIUM；跨 seed 的完整训练结论仍 WEAK。原 A7/A51 概率选择的完整数值另留 calibration_summary.csv，不把局部最优列代替预定对照。

## 规模、阶段和下一步

自然 fit 行数 ML 124759/254879/260495，KR 138225/1395118/71561。没有固定域身份独立操纵规模，关于规模是主要解释的证据 WEAK。小域≠困难域，AUC 与 LL 又受不同基率/排序/校准因素影响；完整 P1 将按 BASE 六格、两任务、seed、selected 和共同终点逐项检验取舍。

本轮最有判别力的下一步仍是已经冻结的完整简单历史/PB 闭环，以及 L4 对 L2/L3/FEATURE 的完整比较。P0 的正效用、非零分区或可读残差均不替代这两项最终检验。

原始输入/缺失范围见 ARTIFACT_AUDIT；输出 hash、实际命令与成本见本地 `p0/STATUS.json`、`p0/EXECUTION_REPORT.json`、`p0/cost_components.jsonl`。未用旧 test 或 Q_confirm 选择算法。
