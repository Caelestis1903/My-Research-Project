# 09 — 指标、配对统计与结果分析规范

## 1. 主评价对象与独立单位

正式性能的独立重复单位是训练seed：**2022、2023、2024、2025、2026，共5个**。同seed的不同方法配对；seed用于训练随机性重复，不替代新数据人口；同轨迹online/fixed EMA、不同epoch、六格、诊断pair和恢复attempt均不增加n。两数据分别分析，跨数据的平均只用于06预定的方法形式选择，不构成n=10的同分布显著性检验。

机器枚举统一为`dataset=movielens|kuairand`、`readout=online|fixed_ema`、`split=train_probe|validation|test`、`endpoint=selected|E10|E25|E50`。面板V_A/V_B另有`panel_id`，不可冒充完整Validation。原始表中`fixedEMA`等历史名字通过显式映射归一，保留`source_readout_name`。

所有新增训练指标当前状态为`UNRUN`。本目录evidence下的结果是既有训练的历史重分析，不填入新候选完成率，也不与未来重建同一轨迹的结果叠加。

## 2. 六格与正式指标

令域d∈{0,1,2}，任务t∈{0,1}，格c=(d,t)。每行同时有两任务标签，预测概率为p，域内正负数为n⁺、n⁻。

### 2.1 AUC

每格按原始模型输出的概率排序，使用与冻结源码相同的`roc_auc_score`。等价定义为：

\[
AUC_{dt}=\frac{1}{n^+_{dt}n^-_{dt}}\sum_{i:y_i=1}\sum_{j:y_j=0}
\left[\mathbf1(p_i>p_j)+\tfrac12\mathbf1(p_i=p_j)\right].
\]

模型概率保留原浮点输出，再转float64计算指标；不先转logit、不加打破并列的抖动、不为AUC做epsilon裁剪。sigmoid饱和0/1可能产生ties，正式评价保留它们。logit margin是独立解释量。

| 指标 | 精确定义 | 用途 |
| --- | --- | --- |
| Macro AUC | 六格AUC等权平均，\(\frac16\sum_{d,t}AUC_{dt}\) | 主指标、选checkpoint、方法门 |
| Traffic AUC | \(\frac12\sum_t\sum_d\frac{n_d}{N}AUC_{dt}\)；源码键`sample_weighted_cell_auc` | 自然流量加权的格内排序 |
| Pooled AUC | 每个任务单独合并三个域算AUC，再对两个任务平均；源码键`pooled_task_mean_auc` | 包含跨域正负比较的排序 |
| Domain Macro AUC | 域内两个任务AUC平均 | 域层损益 |
| Task Macro AUC | 一个任务的三个域AUC等权平均 | 任务层损益 |
| Worst-cell AUC | 六格绝对AUC最小值 | 辅助描述；不替代逐格差值保护 |

Traffic AUC与Pooled AUC不是同一指标；Pooled不把两个任务的标签拼成一个二分类任务。Traffic权重从当前评价split的真实n_d/N计算，不用Train权重替代。

### 2.2 LogLoss

严格复用冻结概率BCE口径：

\[
\ell(y,p)=-\left[y\max(\log p,-100)+(1-y)\max(\log(1-p),-100)\right].
\]

使用`log1p(-p)`并保留原始0/1语义；不另换1e−7/1e−15裁剪。格内平均得到LL_dt；Macro LL为六格等权平均，Traffic LL为自然n_d/N加权，Pooled LL为全行两任务平均。LL在AUC-selected同一模型上评价，不另为LL选checkpoint。ΔAUC>0表示改善，ΔLL<0表示改善，二者分列裁定。

指标源码：[evaluate.py](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/experiments/conditional_information_momentum_20261102_v3/vendor/ema02/evaluate.py)，其实际调用的[metrics.py](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/experiments/conditional_information_momentum_20261102_v3/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31/metrics.py)，后者blob为`8d50d5e3261558cc885c91b9a52c0bfb35cab0a6`。

### 2.3 标签支持与缺项

记录每格count、positive、negative、positive rate。某格缺一类时AUC为null，状态`NO_LABEL_SUPPORT`，相关Macro也为null；不填0.5，不自动改成剩余格平均。正式完整Validation/Test自然人口本应具备六格支持；若不具备，先定位分割/预处理身份并技术暂停相应配置。诊断小面板缺类只使该面板格不可估计，不取消完整训练。

## 3. 选点、选择与同轨迹读出

每次真实更新后更新一次非反馈EMA；每个完整epoch结束评价online和EMA。每个seed、每个读出独立选择E1–E50中**完整Validation Macro AUC严格最大且精确并列最早**的epoch。使用未四舍五入值；新值只有严格`>`旧best才替换。

selected记录绑定`epoch + model_state_hash + buffers_hash + config_hash + data_hash + readout`。其他指标、六格、Test都使用这个模型。E10/E25/E50是固定进度端点，不用于替换selected。AutoIFS的E1–E50指reuse阶段，search10单独记成本。

同一个冻结配置同时报告online与EMA。对家族选择的`selection_readout`按06/08一次冻结；若另行展示专为EMA选择的参数，标记`EMA_DEPENDENT`，不把不同配置的两读出混在交互公式内。

Validation同时承担轻量选参，因此其“最佳配置CI”是选择后描述。所有配置的完整结果都展示；不把这个CI当作独立确认集推断。未来Test是历史已暴露holdout，按照08冻结后评估并注明这一身份；现有协议仍可支持本轮方法取舍。

## 4. 五seed配对统计

对候选A与参照B，每个seed先求\(\delta_s=M_{A,s}-M_{B,s}\)，然后计算：

\[
\bar\delta=\frac15\sum_s\delta_s,\qquad
s_\delta=\sqrt{\frac14\sum_s(\delta_s-\bar\delta)^2},\qquad
CI_{95}=\bar\delta\pm2.776445\frac{s_\delta}{\sqrt5}.
\]

绝对值表展示均值±样本SD（ddof=1），差值表展示配对均值、配对SD、t区间、正/零/负seed数、逐seed差。CI区间宽度表征这五次训练的变化，不覆盖所有可能人口；两数据的差异按其具体人口解释。

配对数必须为5才能满足正式五seed门。技术缺项保留缺失行与原因，不以四个seed平均冒充完整结果；同seed恢复后用通过身份验收的唯一轨迹。恢复attempt不能按性能择优。

**5/5同向是明确的一致性证据，但不自动等于双侧p<.05。** 五个非零同号差的双侧精确sign-test概率为2/32=.0625；不通过单侧/双侧切换美化结果。行动规则依据预定均值增量、六格保护和完整对照，不以是否跨某个p阈值决定继续追加seed。

本轮默认不进行大规模p值筛选。若论文表需要显著性标记，先冻结主检验族：每数据、每家族代表在其唯一selection_readout上对B1/H0的比较，共至多8项，采用Holm校正并同时保留原始/校正p；六格、阶段、面板与机制日志作为描述性分析，不按几百个组合挖显著性。正式+.003门按点估计定义，不要求区间下界也超过+.003；两者分别报告。

## 5. 必须形成的配对比较

| 比较族 | 对象 | 研究判断 |
| --- | --- | --- |
| 新操作增量 | 每个C1/C2网格点−H0 | 原辅助历史之外的增量 |
| 同监督增量 | 候选−B1 | 超出普通PAIR Adam |
| 总体增量 | 候选−B0 | 相对自然强训练的净收益 |
| EMA自身贡献 | 同配置EMA−online | 非反馈读出的收益 |
| EMA交互 | 下述差中之差 | 互补、重叠、干扰与阶段范围 |
| 简单解释 | C或S代表−M/P，原C−S | 尺度、短核、近邻优化思想是否覆盖 |
| 组合 | C12−C1、C12−C2、C12−C1−C2+H0 | 组合的实际净增量 |
| 作用域 | (Cstar_N−H0_N)−(Cstar_SD−H0_SD) | 同作用域父参照校正后的差异 |
| 骨干迁移 | HG_Cstar−HG_H0/B1/B0 | 原生强模型上的相同改动 |
| Gaussian交互 | (G_Cstar−G_H0)−(Cstar−H0) | 既定增强是否改变历史增量 |
| 正式模型 | 最终冻结候选−M3oE/HyperGate/AutoIFS | 强方法门与逐格代价 |

不合适的跨身份配对只进入独立背景表，不进入这张主比较表。所有差先在相同seed上计算，再聚合；差中之差也先在seed内完成，不能组合四个独立SD近似配对不确定性。

### 5.1 EMA交互的两种估计对象

固定候选C与父参照P、相同参数配置，定义：

\[
\Delta_O=AUC(C,O)-AUC(P,O),\quad
\Delta_E=AUC(C,E)-AUC(P,E),\quad
I=\Delta_E-\Delta_O.
\]

`I>0`表示EMA背景下的候选增量更大，`I<0`表示增量减小。`I<0`同时EMA自身正向并不等于EMA有害。I≈0也不证明两者因果独立。

- **部署流程交互：** 四个端点分别按各自Validation-selected epoch取值，回答完整选点与读出流程。
- **共同进度交互：** 四个端点都取E10、E25或E50，回答相同训练进度下的作用关系。

两种表分开，不用部署交互声称固定时刻机制。历史evidence仅已有E10/E50；未提供的E25不插值、不伪造。新训练才补共同E25。

### 5.2 六格保护

对每个保护参照B逐格计算五seed平均差\(\bar\delta_{dt}\)，保护线是全部六格均≥−.001。判定使用完整精度，图表中1e−3单位显示不得改变门。分别记录`worst_mean_cell_delta`、受损cell及n_pos/n_neg、基率、域权重。

S2安全配置检查参照为B0/B1/H0的同读出；控制后还报告对M/P及对应S的逐格差。正式强方法门对固定发表名单中Test均值最高的模型按同人口selected端点检查Macro≥+.003及六格保护。发表名单先冻结，不能看候选Test后删去更强模型；也不把各seed不同最强模型拼成一个不存在的基线。

Macro均值改善而某格越线时，保留净改善和具体损害，标签为trade-off或条件结果。Traffic/Pooled/LL可揭示损益机制，但不替代六格主门。

## 6. 排序信息、面板和来源诊断

04的D2面板采用按域×joint label抽取的行。未加权面板AUC属于该面板分布；若估计原人口格内AUC，用行包含概率\(\pi_i\)的逆权重计算加权Mann–Whitney：正负对权重为\(1/(\pi_i\pi_j)\)，分母为两侧权重和的乘积。该归一化是有限面板的Hájek型估计，不宣称精确无偏。报告n与包含概率；面板指标不用于正式选点。

固定pair分数\(s=I(p^+>p^-)+.5I(p^+=p^-)\)。记录分数增加、减少、完全正确关系保留/遗忘及ties，净值为\(mean(s_{new}-s_{ref})\)。遍历全部正负对时它精确等于AUC差；8,192/32,768抽样pair时只估计该差，不要求样本误差严格为0。概率ties和logit margin分开统计。

EMA与候选的纠正集合可以计算交并集/Jaccard；分母为0时写null和空集原因，不定义为1以制造完全重叠。六格各自计算，不把pair当作独立训练重复。历史正确关系保留图仅解释信息变化，实际算法不读取Validation梯度、不用这些指标调步长。

来源日志保留raw条件ID、w/q、活动k、local k_p、native τ_p、PAIR覆盖、signed cross的BCE×PAIR与WD×PAIR部分、\(D_t\hat Q-\hat U\)、组方向/范数以及finite-step净排序。相关性描述使用“伴随/对应”；同状态替换支持局部运算后果；完整full50与控制支持算法层结论。没有直接被干预的因果环节不升级成已证明机制。

## 7. 最小数据文件合约

全部CSV使用UTF-8、完整数值精度、无DataFrame行号；JSON遵循标准null/boolean，不写NaN。每次原子提交后可校验checksum。

| 文件 | 必需字段 |
| --- | --- |
| `runs.csv` | study_id, logical_run_id, attempt_id, config_id, algorithm_id, dataset, seed, backbone, stage, state, source_commit, config_hash, data_hash, scope_hash, started_at, ended_at, physical_replay, replay_reason |
| `curves.csv` | logical_run_id, epoch, native_steps, pair_events, readout, split, metric_name, cell, value, n, n_positive, n_negative, state_hash, evaluation_seconds |
| `endpoints.csv` | logical_run_id, seed, config_id, selection_readout, readout, split, endpoint, selected_epoch, model_hash, data_hash, cell, macro_auc, cell_auc, macro_logloss, cell_logloss, traffic_auc, pooled_auc, source_type |
| `paired.csv` | comparison_id, dataset, seed, readout, endpoint, cell, candidate_id, reference_id, candidate_value, reference_value, delta, identity_match |
| `summary.csv` | comparison_id, dataset, readout, endpoint, cell, n, mean_delta, sd_delta, ci95_low, ci95_high, positive, zero, negative, threshold, gate_result |
| `config_selection.csv` | family, dataset, grid_config, selection_readout, mean_val_auc, six_cell_safe, parent_deltas, selected, tie_rule, selection_timestamp |
| `decisions.json` | rules_version, family_representatives, selection_readouts, Cstar_id, Cstar_params_by_dataset, stage_gates, triggered_arms, not_triggered_arms, reason_refs |
| `diagnostics.jsonl` | run/state/anchor/panel IDs, split, parameter_group, source_id, counts/clocks, scalar quantities, branch_id, reference_branch, null_reason |
| `costs.csv` | logical_run_id, attempt_id, train_seconds, eval_seconds, diagnostic_seconds, epoch_count, global_batches, optimizer_calls, autograd_sweeps, train_forwards, eval_forwards, max_gpu_allocated_bytes, max_process_gpu_bytes, peak_host_ram_bytes |
| `state_receipts.jsonl` | run_id, state_kind, guardian_key, schema_version, epoch, step, put_hash, get_hash, bytes, owner, available, verified_at |

`endpoints.csv`一行一格时Macro/Traffic/Pooled重复值必须相同，聚合先按模型端点去重；不把六行Macro当六个观测。指标长表转换保留一对一键。模型哈希只写摘要，模型/optimizer/EMA/梯度张量不落磁盘。

## 8. 报告强度与停止条件

沿用仓库STRONG/MEDIUM/WEAK作为**具体主张**的证据等级，而非对整条路线一次定级。已验证代数关系可标确定；稳定五seed完整增量及匹配控制支持经验强结论；局部干预支持相应局部因果；相关日志作机制线索。标签附一句对应证据和适用范围。

所有默认配置完成并按08做完触发控制后，即可作方法/组件/关闭决策。区间跨0时给出量级、方向一致性和是否影响选型；不自动启动更多seed、更多小面板、更多时间核。12的报告必须同时给出“当前结论—核心证据—适用边界—下一研究决定”。
