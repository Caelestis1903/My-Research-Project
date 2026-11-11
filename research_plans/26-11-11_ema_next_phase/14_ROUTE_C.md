# 路线 C：用域任务信息组合两块共享历史

返回：[唯一执行入口](00_EXECUTION_RUNBOOK.md)｜依赖：[路线 B](13_ROUTE_B.md)｜门值：[评价与决策](09_EVALUATION_AND_DECISION.md)｜资源与恢复：[自动执行](08_AUTOMATED_EXECUTION.md)。

**执行选择：在 B 的两个共享历史重心上，比较固定权重、训练可见的全局权重和六格收缩权重。** C 的主 H 固定为 MovieLens 40 步、KuaiRand 138 步，S-only、logit 组合；不等待 B 的 H 搜索、不追加条件 H/rho 搜索。C 同时报告相对同 H 的 B 和相对 Validation 冻结最佳 B 的差异。研究继续门统一由 [09](09_EVALUATION_AND_DECISION.md) 定义，本文件不增加显著性或正向 seed 数门槛。

## 1. 证据、关键问题与更简单替代

11/05 的响应方向 EMA 确实运行，但相对固定 F 的 selected Test AUC 增量仅 ML `+0.000000104`、KR `−0.000004946`；分组/全局 EMA 也没有稳定 selected 增量。11/09 A0 中，持续性预测器虽然改变部分预测误差，GLOBAL/TASK0/TASK1 的实际动作却与简单身份/阶段模型完全相同；30 条静态任务政策均选择 Fast，任务政策相对统一政策增量为零。A1 及后续闭环没有运行。[11/05 报告](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/research_reports/2026-11-05_aug_history/REPORT.md)、[11/09 A0 报告](https://github.com/Caelestis1903/My-Research-Project/blob/1391affa5f08556c79b4cb2bfad15228a8b9129f/research_reports/2026-11-09_ema_adaptive/FINAL_REPORT.md)。

最新归因支持共享表示的重要性、EMBED recent/old 正交互及固定面板的历史预测组合价值。这给出了新的操作对象，但没有预先确立六格一定需要不同权重。[最新归因报告](https://github.com/Caelestis1903/My-Research-Project/blob/5ef45b18b973ed11be6fbb5f654562349998de60/research_reports/2026-11-10_ema_causal_attribution/FINAL_REPORT.md)。

本路线依次回答：

1. 两块共享历史的**统一预测组合**是否已足够？
2. 训练可见的域任务排序信息，是否能选择在下一 epoch 有效的不同组合？
3. 条件组合是否优于同样训练可见、同一动作空间的全局组合？
4. 若参数插值即可达到相同收益，是否应采用更便宜的参数版本？

C 不重复旧 Fast/Slow 的未来收益回归，不增加十二组 EMA，不预设梯度冲突或负迁移。域规模不进入系数、损失或进入条件。“小域不是困难域”直接继承。

## 2. 两个共享历史状态及预测

设 optimizer step 为 t，固定参数 EMA 为 F_t，S 为来源图确认的共享坐标，N 为其余坐标。H 只使用上述两个主值。令

\[
h_t=\min(H,t),\quad a_t=\rho^{h_t},\quad
\mu^O_t=F_{t-h_t,S},\quad
\mu^R_t=\frac{F_{t,S}-a_t\mu^O_t}{1-a_t}.
\]

t=0 不作除法；从第一步开始使用 B 的有界历史实现、数值重构与检查。rho、EMA 初始化和更新时钟完全沿用源协议。两态的 N 都固定为同刻 F_{t,N}，不使用旧私有参数：

\[
z^R_t(x)=f_{\rm logit}(x;\mu^R_t,F_{t,N}),\qquad
z^O_t(x)=f_{\rm logit}(x;\mu^O_t,F_{t,N}).
\]

logit 获取接口沿用 B；不得用截断后的概率反变换替代真实 pre-sigmoid 值。固定 B 的读出为

\[
z^B_t=(1-a_t)z^R_t+a_tz^O_t.
\]

条件动作只改变混合系数：

\[
z^C_{t,c}(\alpha)=(1-\alpha)z^R_{t,c}+\alpha z^O_{t,c},\qquad
p^C_{t,c}=\sigma(z^C_{t,c}),\quad c=(d,k).
\]

不增加历史模型副本；六格仅有六个标量。S 的坐标映射复用真实 registry，domain embedding 的专属行仍在 D 组，不能把所有 embedding 坐标都改称 S。来源实现：[group_registry.py](https://github.com/Caelestis1903/My-Research-Project/blob/40705dfe0bb4fe614bdca835c0bae567348c87d0/experiments/aug_history_20261105/group_registry.py)。

## 3. 训练探针、唯一 pair 与拟合规则

### 3.1 保留来源训练人口

P_A/P_B使用来源固定训练面板和行身份，P_A拟合、P_B审计。人口锁定为：拟合/pairwise风险/面板AUC使用rank_x、rank_y、rank_row_ids；自然LL使用natural_x、natural_y和相应有效行，继承源readouts.pulse_rows定义。两者的行仍照常参加原始训练 Loader；不删除训练样本，不重新切分 train/validation/test，不改变自然样本目标。策略拟合的六格均衡风险不替代 online 的自然样本 BCE。

每个 `(dataset, seed, panel, cell)` 独立建立固定 pair 表。正/负行先按原始 row_id 去重、排序，分别得到 I^+、I^-；对笛卡尔积的唯一 `(positive_row_id, negative_row_id)` 计算确定性 BLAKE2b-128。hash 输入使用规范 UTF-8 JSON：study=`EMA_NEXT_20261111_C_PAIR_V1`、dataset、seed、panel、domain、task、positive_row_id、negative_row_id。字段顺序固定、无空白、整数不用科学计数。按 `(hash,positive_row_id,negative_row_id)` 排序，取

\[
K_c=\min(4096, |I^+_c||I^-_c|)
\]

个唯一 pair，**无放回**。来源面板的每格正负支持是有界的；不能用随机有放回抽样补满，也不能用取模产生重复 pair。pair 表只创建一次，在所有 epoch、动作、s 和参数/预测对照间共用。保存 row/pair hash、K_c 与唯一性检查结果；向量化索引计算全部 margin，不逐 pair 做模型前向。

P_A 或 P_B 的 source row/hash 不符为数据审计错误。合法退化格若没有两类，标记 `NO_VALID_PAIR`；P_A 该格的局部系数回退全局，P_B 该格风险记 NA 并交 [09](09_EVALUATION_AND_DECISION.md) 判断门是否有完整输入，不伪造零收益。若 P_A 全部格退化，所有系数使用当时 a_e，禁止以缺失数据产生晋级。

### 3.2 有限动作、全局拟合与收缩

设 epoch e 结束的 step 为 t_e，a_e=a_{t_e}。候选集合为

\[
\mathcal A_e=\operatorname{unique}\{0,0.25,0.5,0.75,1,a_e\}.
\]

浮点去重容差 10^{-12}，升序排列。给定固定 pair (i^+,j^-)，定义

\[
m_{e,c,r}(\alpha)=(1-\alpha)(z^R_{i^+}-z^R_{j^-})
 +\alpha(z^O_{i^+}-z^O_{j^-}),
\]

\[
R^A_{e,c}(\alpha)=\frac1{K_c}\sum_{r=1}^{K_c}
\operatorname{softplus}[-m_{e,c,r}(\alpha)].
\]

模型前向仍为源 FP32；风险归约使用 FP64，稳定 softplus，无新增温度。有效六格的均衡风险给出全局系数；完整面板时就是六格均值：

\[
\alpha^G_e=\arg\min_{\alpha\in\mathcal A_e}
\frac1{6}\sum_c R^A_{e,c}(\alpha),\qquad
\hat\alpha_{e,c}=\arg\min_{\alpha\in\mathcal A_e}R^A_{e,c}(\alpha).
\]

退化格按上节记录与回退，不静默使用不同科学人口。候选风险距最小值不超过 10^{-12} 时视为平分；依次选距离 a_e 最近、距离 0.5 最近、数值最小者。

六格方案仅有两个预设候选：

\[
\alpha^{C,s}_{e,c}=(1-s)\alpha^G_e+s\hat\alpha_{e,c},
\qquad s\in\{0.5,1\}.
\]

收缩后不投回 grid。s=0 是全局控制，不另列新算法；两个 s 都完成五 seed，统一按 Validation 选 s，不逐 seed 选 s。不增加神经门控、ridge、风险校准温度或基于域规模的正则。

## 4. 严格的一 epoch 前瞻时钟

epoch e 的 P_A 系数只在 **epoch e+1** 使用。epoch 1 尚无上一 epoch 系数，三类预测读出都用当前 a_1；同系数参数版退化为 F。epoch 1 完整评价完成后才拟合供 epoch 2 使用的系数。

在 epoch e≥2，使用上一 epoch 冻结的绝对系数 α_{e−1}，配合**当前**两态 μ^R_{t_e}/μ^O_{t_e} 与 F_{t_e,N}；即使 warmup 中 a_e 改变，也不偷偷把已拟合 α 重映射为 a_e。记录 `fit_epoch=e−1`、`apply_epoch=e`、`a_fit` 和 `a_apply`。拟合过程不读取 Validation 或 P_B 标签。

每 epoch 的顺序固定为：

1. 完成该 epoch 原 online 训练、Adam、F 更新与 B 两态维护。
2. 取上一 epoch 系数（e=1 取 a_1），流式计算本 epoch 六种完整 Validation 曲线与 P_B 前瞻审计；执行最早严格最大选点。
3. 在 P_A 的当前两态 logit 上计算所有 grid 风险，生成下一 epoch 系数。
4. 原子提交本 epoch 结果和 pending 系数。selected 快照保存实际已用系数，不能保存刚生成的 pending 系数代替。

P_B 的“未来风险”是 P_A(e−1) 决策在 P_B(e) 当前两态上的实际风险，不是拟合出来的预测值。P_B AUC使用该面板rank人口所有有效行的实际概率预测；pairwise risk使用同rank人口固定唯一pair，是额外机制量，不替代AUC。PARAM的风险和AUC由对应参数模型的真实前向计算，不能使用PRED线性混合margin代替。

## 5. 六种完整曲线与参数替代

改变 α 后，参数重心一般不再等于 F：

\[
\theta^P_{t,S}(\alpha)=(1-\alpha)\mu^R_t+\alpha\mu^O_t,
\qquad\theta^P_{t,N}=F_{t,N}.
\]

匹配参数读出为 `f_logit(x;θ^P_t(α))`，使用和对应预测版**完全相同的上一 epoch 系数**。其用途是区分“更合适的参数位置”和“非线性输出组合”；不能将 MATCH 写成对所有参数组合充分调优的上界。

| 曲线 ID | 系数 | 读出 | 相对 B 的额外完整 batch 等效前向/epoch |
|---|---|---|---:|
| `C_GLOBAL_PRED` | α^G_{e−1} | 两态 logit 混合 | 0，共用 B 两态预测 |
| `C_GLOBAL_PARAM` | 同上 | 单一插值参数模型 | ≤1 |
| `C_CELL05_PRED` | α^{C,.5}_{e−1,c} | 六格 logit 混合 | 0 |
| `C_CELL05_PARAM` | 同上 | 六格匹配参数插值 | ≤2 |
| `C_CELL10_PRED` | α^{C,1}_{e−1,c} | 六格 logit 混合 | 0 |
| `C_CELL10_PARAM` | 同上 | 六格匹配参数插值 | ≤2 |

六条曲线从 epoch 1 开始逐 epoch 记录完整 Validation，各自独立保留最早严格最大 selected 状态；不能等 s 冻结后从最后的参数补算历史曲线。两种预测候选只增加向量运算，三种参数对照合计最多增加5个完整 batch 等效前向/epoch。

六格参数读出按已知 domain 拆分当前评估 batch，每域两任务分别用对应 α_{d,k} 的 S 参数前向，取对应任务输出，恢复原 row 顺序。每个样本最多前向两次，等效处理量≤2F；实际为至多6次子 batch 调用，要单列调用和重构开销。不能写成“一次前向”，也不需要6个常驻模型。完全相同的系数和状态可直接复用同一预测字节。

固定 α=a_t 时，参数插值应按 B 的数值容差退化 F；全局参数版具有单次推理优势。若全局参数版在 [09](09_EVALUATION_AND_DECISION.md) 的成本/收益比较中已足够，优先保留它。若六格参数版与六格预测版相当，报告“域任务条件参数读出有效”，不把收益归为预测集成独有机制。

## 6. 核心接口与伪代码

接口置于 [06](06_CODEX_IMPLEMENTATION.md) 指定的新实验包，复用 B 的 bank、logit adapter、functional readout 和源 evaluator：

```text
build_unique_pairs(panel_rows, labels, identity) -> PairTable
fit_c_weights(z_recent, z_old, pairs_A, a_fit) -> CWeightState
mix_c_logits(z_recent, z_old, domain_ids, used_weights, s) -> [N,2]
readout_c_parameters(bank, batch, used_weights, s) -> [N,2]
observe_c_epoch(bank, epoch, used_weights, evaluator, pairs_B) -> CRecords
```

```text
pending = None
for epoch in 1..50:
    run_source_epoch_without_C_feedback()
    bank = B_shared_two_centroids(H=dataset_fixed_H)
    used = fixed_a_for_all_cells(bank.a) if epoch == 1 else pending
    assert used.fit_epoch < epoch or epoch == 1

    for source_validation_batch in full_validation_stream:
        zr, zo = B_two_logits(bank, batch)       # 同H、同S两态
        stream_metrics(C_GLOBAL_PRED, mix(zr, zo, used.global))
        stream_metrics(C_CELL05_PRED, mix(zr, zo, used.cell05))
        stream_metrics(C_CELL10_PRED, mix(zr, zo, used.cell10))
        stream_metrics(C_GLOBAL_PARAM, parameter_readout(bank, used.global))
        stream_metrics(C_CELL05_PARAM, cell_parameter_readout(bank, used.cell05))
        stream_metrics(C_CELL10_PARAM, cell_parameter_readout(bank, used.cell10))
    commit_full_validation_and_update_selected_RAM(bank, used)
    audit_P_B_future_actions(bank, used, fixed_pairs_B)

    zr_A, zo_A = B_two_logits(bank, P_A)
    pending = fit_c_weights(zr_A, zo_A, fixed_pairs_A, bank.a)
    commit_epoch_scalar_hash_receipt(epoch, used, pending)
```

P_A/B 观测和 Validation 前向必须使用独立观测 RNG/恢复源 RNG，不改变下一训练 batch 顺序或增强噪声。C 对 online 参数、Adam m/v、梯度、F 更新完全只读；不将 pairwise loss 反传到模型。**本路线首先是历史读出算法，蒸馏时才通过学生目标影响训练**；C 教师蒸馏只能按 B 已定义的条件分支执行，不另创损失和温度。

## 7. 状态、恢复、成本与结果记录

每个 selected endpoint 保存两块 S、当时 F_N、所用 α、H、a、数据/布局/epoch hash；全局参数读出可保存其已物化单模型状态。所有模型、优化器、历史参数和 selected 状态只驻 RAM guardian，遵守来源 `start.md`，**禁止参数/checkpoint 写盘**。六条曲线允许按完全相同 state hash 去重，不能引用会继续更新的可变 tensor。

磁盘仅保存实验标识、配置、系数、指标、pair/row身份、hash和运行日志。worker 恢复从存活 guardian 的已提交状态开始；guardian 状态丢失，hash 不能恢复参数，按 [08](08_AUTOMATED_EXECUTION.md) 重放该 seed 的源轨迹和系数时钟，并记录真实成本。不假设已释放的历史 RAM 仍可用。

本路线 H 固定、共享 B 训练轨迹；新增成本是小面板拟合、最多5F/epoch参数对照和 selected RAM。完整 V raw logits 按当前 epoch 分批计算；源 evaluator 在内存保留本 epoch 必需的 score/label 并完整排序得到 AUC，不能平均 batch AUC。完成本 epoch 指标后释放，不保存所有 epoch 的全量数组。每 seed 六条 endpoint 的 payload、guardian RSS和额外前向时间必须实际记录。

必要输出：六格/Macro AUC与LL、源Traffic/Pooled、五seed配对差、同H B及最佳B对比、GLOBAL/CELL和PRED/PARAM匹配差；另记每次拟合的grid风险、α_G/α_c、实际与全局不同的格、P_B未来风险/AUC、实际排序翻转、a_fit/a_apply及缺失格。排序翻转用固定唯一 pair 的符号（含平局）逐条比较，pair只作机制观测，不作为额外独立样本。

## 8. 决策边界

必须分别报告四层：六格数值不同；排序动作不同；上一 epoch 训练信号在下一 epoch 有正效用；完整 selected 评价超过全局策略。不能跳过中间层，亦不因最后一层未通过而抹掉前面已成立的事实。

[09](09_EVALUATION_AND_DECISION.md) 统一规定五seed Validation 对全局增量（主门为≥0.0001）、P_B未来效用与实际排序变化等可计算条件。任何正向 seed 数、CI和相关系数均只作描述，不新增4/5或5/5门。完整算法对发表模型的+0.003资格与这里的组件研究门分别记账。

若条件版不超过全局版，停止本轮复杂分域控制；若全局参数版已达到成本收益要求，保留一次前向替代；若只有函数组合胜出，再按 B 的既定部署/蒸馏决策处理。失败后不追加更多层门控、指标、H或rho。C的停止不会否定固定EMA已成立的收益，也不会把未执行的蒸馏反馈写成失败。
