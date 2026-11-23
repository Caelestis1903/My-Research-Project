# 02 — M1–M8 全候选机制、变体与可证伪假设

## 总原则

本轮不对八个机制家族排序。M1–M8 全部纳入，每个家族的以下变体全部注册。机制变体可以得到 NO_GO，但不得因主观优先级在执行前排除。

## M1：共享更新诱导的专有梯度一致性塑形

设共享参数 U、专有参数 P，六个 domain-task cell 的损失为 L_c。定义：
g_c(P)=∇_U L_c(U,P)，g_J=∇_U L_joint(U,P)。

三个互不等价的目标：
1. M1-A 方向一致性：L_cos = Σ_c w_c[1−cos(g_c,g_J)]。
2. M1-B 幅度感知一致性：L_mag = Σ_c w_c ||g_c−α_c g_J||²/(||g_J||²+ε)，α_c 固定由训练 batch 的 cell loss mass 计算，不使用 Validation。
3. M1-C 成对冲突能量：L_pair = Σ_{c<k} w_ck max(0,−cos(g_c,g_k))²，并记录与 g_J 的关系。

每种目标均通过二阶自动微分更新 P；U 保持 requires_grad=True 构造 g_c，二阶反传后显式清空 U.grad=None，再执行 native Adam step，使 U 参数与 U 的 Adam m/v/step bitwise unchanged。

处理/控制：
- 对应 consistency objective；
- P-BCE-only step + joint step；
- within-tensor coordinate-shuffled gradient control：对每个 cell、每个 U parameter tensor 使用独立固定 permutation，保留每个向量范数；不得所有 cell 共用同一 permutation。
- two-step joint baseline；
- consistency-only readout。

假设：P 的变化可以塑造 U 接收的 cell gradient geometry。该几何变化是否改善排序是独立问题，不预设成立。

## M2：样本×任务条件影响

对 row i、source task t、target cell c，使用 Train-only calibration pair margin m_c。计算同一 anchor Adam state 下：
ΔU_full = virtual_adam(g_full)
ΔU_without(i,t) = virtual_adam(g_full−g_i,t)
ΔU_i,t = ΔU_full−ΔU_without(i,t)
s(i,t,c)=∇_U m_c · ΔU_i,t。

正分数表示 row-task 对当前共享更新的一阶预测提高正负样本 margin。该量是 local optimizer-step marginal attribution，不称为精确因果效应或完整 influence function。

变体：
1. M2-A task/cell-conditioned influence weighting：按 s 的 task-specific aggregate 形成权重，domain-task 内均值归一，保持总 loss mass。
2. M2-B influence-ranked scheduling：每个 optimizer step 分别为 task0/task1 构造独立 row microbatch，按各自 task-specific score 排序；两个 task 的梯度在同一参数状态下合并后执行一次 Adam step。每个 task 的全部 row 在 epoch 内恰好出现一次，保持每任务 exposure、每步 row 数、optimizer steps 和 loss mass。自然顺序为 control。
3. M2-C negative-impact suppression：对负向影响行降低权重，正向影响行补偿，保持 domain-task loss mass。
4. M2-D worst-cell/cell-response weighting：先保留六维 s(i,t,c)，主 action 使用该 row-task 对六个 target cells 的最小预测 margin response 形成保护性权重；另报告等权平均聚合的敏感度版本。所有 source domain-task 的总 loss mass 显式匹配。

控制：自然 uniform、task-agnostic score、within-domain shuffled score、随机权重分布匹配。不得退化为 loss、gradient norm 或域频率排序。

## M3：阶段性与组合式训练影响

M3 不依赖 M2 是否阳性，全部执行。

变体：
1. M3-A stagewise attribution：对 E10/E25/E40 的同一 row/task panel 计算影响分数、pair-margin response 和符号翻转。
2. M3-B score-defined group interaction：每个 anchor 使用 Train-only M2 score 构造高/低影响的互斥 A/B pools，执行五个 group arms。
3. M3-C random matched group interaction：使用相同 domain、joint-label support 和 pool size 构造随机匹配 A/B pools，执行相同五个 arms。
4. M3-D stage×group interaction：对 M3-B 与 M3-C 的同一 group pools 在 E10/E25/E40 分别计算，检验交互项随阶段变化。

固定 batch size n：
I_AB = ΔU(A/B)−0.5ΔU(A-only)−0.5ΔU(B-only)。
同时报告函数输出与 ranking margin 的 interaction residual。A、B 是两个互斥 row pools，各自至少有 n 个不同 row。A-only/B-only 分别使用 n 个不同 row；A/B 各用 n/2；A/A 与 B/B 分别从对应 pool 取 n/2 个 row 并重复一次。五组总 row slots 均为 n，Adam 起点和步数相同。

## M4：域任务条件有害特征依赖

不把相关性或 HSIC 下降直接称为因果或有害依赖。按真实 feature schema 分组，逐组报告 domain/task 条件结果。

变体：
1. M4-A conditional dependence map：计算 feature-group embedding 与 shared/task representation、logit/margin 的条件依赖统计，按 domain/task 分层。
2. M4-B conditional permutation intervention：在 domain × observed joint-label strata 内置换目标 feature group，比较 logit、pair margin、AUC 变化；这是依赖诊断，不是因果证明。
3. M4-C task-conditioned residualization：对被测 feature group 的可预测表示分量做 task-conditioned residualization，配 matched sham control。
4. M4-D task-conditioned HSIC penalty：对指定 representation/feature-group dependence 添加 task-conditioned HSIC penalty，配 matched sham penalty/control。
4. M4-D task-conditioned HSIC penalty：对指定 representation/feature-group dependence 添加 task-conditioned HSIC penalty，配 matched sham penalty/control。

所有 feature groups 由 DATA_MANIFEST 的实际字段映射生成。domain_indicator 不作为可干预特征。用户/物品实体 ID 不做任意置换；对这些字段只做依赖测量或合法 masking 检查。

## M5：语义约束的多标签保持增强

M5 不限于审计。所有可实现变换均进入真实增强诊断，但标签保持必须被测量，不得先验宣称。

变体：
1. M5-A sparse field masking/dropout：使用模型真实支持的 missing/OOV/mask 路径；逐字段检查 mask 语义，不得假设 OOV index 自动等于 mask。
2. M5-B embedding perturbation：稀疏/稠密 embedding 的小幅扰动，Gaussian+EMA 作为已知内部参照，新增噪声形式单独标记。
3. M5-C within-stratum categorical substitution：仅在相同 domain、观察到的联合标签支持和预先定义的上下文匹配条件下替换类别值；记录替换前后频率和标签一致性。
4. M5-D dense feature perturbation：只对真实 dense feature 执行，在原 domain/semantic bin 内扰动；无 dense field 的数据集标记该变体 NOT_APPLICABLE。
5. M5-E multi-view consistency：原始样本与每种通过字段合法性审计的增强视图共同训练，两个任务标签保持原值，增加 prediction consistency 项。

MovieLens 的嵌套任务标签关系必须保持；KuaiRand 按实际联合标签支持处理。user_id、movie_id、domain_indicator 和不可解释的离散 ID 不得任意替换。每个变换记录输入合法性、标签联合分布变化、预测一致性、训练效果。

## M6：输出敏感度条件化控制

对每个 cell 的 calibration pair margin m_c，计算共享/专有参数组 Jacobian J_c=∇_θ m_c。给定原始更新 Δθ_0，预测响应为 J_cᵀΔθ_0。

变体：
1. M6-A sensitivity scaling：依据六格预测响应缩放整体更新，避免预测 worst-cell 大幅受损。
2. M6-B worst-cell trust region：求解 min_δ ||δ−Δθ_0||²，使每格 J_cᵀδ ≥ −τ。
3. M6-C parameter-group response gating：对 U/P 及运行时确认的参数组估计输出敏感度，执行组级缩放或门控。

控制：global norm clipping、random scale、无敏感度的相同预算缩放。代理预测与实际 pair/AUC 变化分开报告。

## M7：跨 seed 分布影响归因

M7 使用 M2 的 row-task influence 和其他机制的局部响应，但作为独立机制家族完整评价。

变体：
1. M7-A distributional attribution：同一 row/task/cell 的五 seed score 分布，报告均值、SD、分位数、正负方向比例。
2. M7-B robust influence weighting：score = mean(score_seed)−λ·SD(score_seed)，λ 取预注册轻量网格；对照只用均值。
3. M7-C leave-one-seed-out transfer：用四个 seed 估计样本影响规则，在留出的第五 seed 上检验排序预测和更新方向；轮换五次。

五 seed 经验分布不等同总体分布。禁止把 row 数或 anchor 数作为独立 seed。

## M8：多任务冲突控制器

M8 与 M1 区分：M1 改变 P 以塑造 cell gradients；M8 直接控制共享更新中的任务/cell gradient aggregation。

变体：
1. M8-A PCGrad reference：按六 cell gradient 执行标准 pairwise projection，作为历史负向方法的复核控制，不宣称新颖。
2. M8-B conflict-aware weighting：根据六 cell pairwise cosine/conflict matrix 求非负、受上下界约束的 cell weights，保持总 loss mass。
3. M8-C common-descent projection：将联合梯度投影到各 cell gradient 的共同下降可行方向，若不存在共同方向则求最小冲突残差；完整报告 solver 与约束。
4. M8-D conflict-aware cell gating：根据冲突矩阵对 cell contribution 做连续门控，设定最低权重和总质量约束，不允许静默丢弃任务。

控制：canonical joint gradient、uniform cell weights、random weights、历史 PCGrad 对照。所有动作保留每个 cell 的权重与有效监督量。

## 五组跨机制交互

全部运行，不依赖单机制先通过：
- X1 M1×M2：P consistency + task-specific row-task weighting；
- X2 M2×M3：sample-task weighting + group interaction action；
- X3 M4×M5：task-conditioned dependency intervention + label-consistency-constrained augmentation；
- X4 M6×M8：output-sensitivity trust region + conflict-aware aggregation；
- X5 M1×M8：private-gradient shaping + shared-gradient conflict controller。

每个组合均有两个单机制 arm、组合 arm 和预算匹配 control。禁止扩成全部八机制全排列。


## 三点轻量超参网格（全部 trainable variants）

S1 使用每个变体的 canonical default 运行全量局部诊断；S3 对全部 31 个训练动作变体执行三点轻量搜索。每个设置均跑两数据集、五种子，不允许按单 seed 选择。预注册网格：

| Family | Parameter | Three values |
|---|---|---|
| M1 | consistency inner-step LR multiplier | 0.1 / 0.3 / 1.0 |
| M2 | row-task weight amplitude α | 0.05 / 0.10 / 0.20 |
| M3 | group pool size | 64 / 128 / 256 rows |
| M4 | dependence penalty λ or conditional permutation rate | λ: 1e-4 / 1e-3 / 1e-2; permutation rate: 0.05 / 0.10 / 0.20 |
| M5 | transform strength / mask or replacement probability | 0.01 / 0.05 / 0.10 for continuous/embedding strength; 0.05 / 0.10 / 0.20 for categorical mask/substitution probability |
| M6 | worst-cell predicted margin tolerance τ | 0.05 / 0.10 / 0.20 in normalized margin units |
| M7 | robust-score risk coefficient λ | 0.25 / 0.50 / 1.00 |
| M8 | conflict activation threshold | 0.00 / 0.10 / 0.20 |

若某个变体使用两个相关参数，按表中对应三点单参数 sweep，其他参数固定 canonical default；不得做全笛卡尔积。M3 group size 受真实 label support 约束时，使用最大可用值并记录实际 n，不补造样本。X1–X5 使用其组成 family 在 S3 五 seed 综合结果中选出的配置，不另做组合超参网格。

S3 结束后，每个变体使用五 seed Validation aggregate 选择一个配置并 freeze；所有通过 gate 的变体都进入 S4，不按变体间收益排名删减。
