# 阶段与最终报告模板

本文件是执行后填数的报告schema。当前规划不填充新实验结果；历史证据已在01/12记录。科学报告按研究问题组织，不按运行日期堆叠。

## 1. S0 来源与复用报告

~~~text
SOURCE_AUDIT
main_snapshot:
source_files_and_hashes:
data_population_and_split_hashes:
raw_csv_verified:
model_parameter_layout_hash:
source_loss_and_optimizer:
selection_rule:
historical_test_exposure:
reuse_rows:
state_available_now:
missing_controls:
technical_blockers:
approved_new_module_paths:
independent_review_verdict:
~~~

附复用矩阵：每个dataset/arm/seed，原始来源、完整50epoch、是否有selected Validation/Test、模型hash、能复用数值/模型/optimizer的哪一层、需要新增科学训练还是STATE_REPLAY。给出P00/P10缺失三seed清单，别把目录存在等同实验完成。

## 2. A 核心研究报告

### 当前结论

用一段话回答：辅助pair是否有增量；历史包是否有独立增量；动态抽样改变了什么；是否进入简单历史对照。

### 核心结果表

| 数据 | 对比 | n | Validation Macro ΔAUC | 六格最小Δ | ΔLL | 历史Test线索 | 决定 |
|---|---|---:|---:|---:|---:|---|---|
| ML/KR | A10−A00 | 5 | 执行后填写 |  |  |  |  |
| ML/KR | A20−A10 | 5 |  |  |  |  |  |
| ML/KR | A30−A10 | 5 |  |  |  |  |  |

附五seed全部绝对六格AUC/LL及selected epoch；说明同pair身份、λ、RNG、WD/v、参数覆盖是否匹配。若A21/A22触发，明确普通Adam、单份长m、双通道单辅助h、24条件h的性能/状态大小比较。

### 结论层次

- 经验现象：哪些数据集/读出有增量；
- 更新干预：哪些参数/历史贡献造成排序变化；
- 机制解释：条件表、双通道或统一长历史谁获得支持；
- 研究决定：保留哪个版本、停止哪些版本、下一合法任务。

若换λ，必须展示A40−A41；不能以A40−旧A10冒充历史效应。若历史Test曾有坏格而新Validation没有支持修正，直接写当前选参未解决该历史代价，不用Test继续搜索。

## 3. B 核心机制报告

### 当前结论

分别回答：是否存在实际跨格排序损害；EMA缓解了哪些影响；真实域任务身份能否胜匹配伪分组；train信息能否支持动作。

### 更新效用表

| 数据/阶段 | 组 | 源q→目标c | JOINT−LOO online | JOINT−LOO EMA | 同范数 | 伪分组 | n |
|---|---|---|---:|---:|---:|---:|---:|

附40锚点原始表，但科学汇总n=5。ZERO_DATA/ONLY只解释条件更新效用，不作为最终算法性能参照。

### 真实脉冲与函数组合表

| 数据 | 训练选源规则 | 逻辑步数 | online终点ΔAUC | F终点ΔAUC | 六格最小Δ | 概率/Logit平均 | 结论 |
|---|---|---:|---:|---:|---:|---|---|

必须注明一次LOO之后恢复正常联合训练；段内1380逻辑步与额外1340实际调用分账。有限窗口参数/预测平均与原始全历史F分列。

### 三种解释裁定

| 解释 | 支持证据 | 反向证据 | 适用范围 | 是否改变算法选择 |
|---|---|---|---|---|
| 跨格有害更新 |  |  |  |  |
| 身份无关噪声/尺度 |  |  |  |  |
| 历史函数组合 |  |  |  |  |

可以多项并存，不填唯一贡献百分比。不能用(1−ρ)代数衰减替代实际排序结果。

## 4. 候选与HPO报告

记录全部预注册配置及其五seed结果，含未选值；说明较优配置来自哪个Validation指标与保护条件。对η报告ADAPT/STATIC/UNIFORM三个完整控制，对Aλ报告普通Adam匹配控制。

~~~text
FROZEN_SELECTION
candidate_id:
dataset_specific_parameters:
searched_values_and_all_seed_results:
selected_by_validation:
increment_vs_strong_internal_control:
increment_vs_simple_alternative:
cell_guard_status:
test_used_for_selection: false
selected_epoch_and_model_hashes:
remaining_open_questions:
~~~

## 5. 最终主表

| 数据 | 方法/角色 | n | 六格AUC | 六格LL | Macro AUC | Macro LL | Traffic | Pooled | source |
|---|---|---:|---|---|---:|---:|---|---|---|

发表模型、内部控制、候选三种角色单列。配对增量表至少含：最终候选−最强发表模型、最终候选−强内部参照、最终历史−同监督普通Adam、最终自适应−静态/统一简单方案。包含EMA的总收益不得重复计入新组件。

FINAL_REPORT结论需回答用户五问：

1. 条件历史是否有普通Adam未提供的独立作用？
2. 共享EMA收益是否包含可利用的跨域任务更新部分？
3. 哪种实际更新机制值得保留，或为何停止当前候选？
4. 新增AUC与六格代价分别是多少？
5. 当前证据已经足以接受、排除和结束哪些具体问题？

## 6. 执行/恢复/成本附录

登记每条物理attempt、有效完整轨迹、同seed重放、虚拟Adam、脉冲和extra-forward；给出资源测量上下文。不能把worker时间和墙钟混称，也不能把历史RAM验收当作当前状态。

失败记录：故障位置、已提交epoch、RAM现场验收、是否可从同状态续训、实际重放成本、科学数据是否受影响。技术暂停与候选性能停止分别统计。

## 7. 上下文回填与审查

每个最终实验报告附完整数表与source manifest；更新 EXP_AI_CONTEXT.md、Global Summary、Stable Findings、Open Questions、MDMT_ALGORITHM_METRICS_LEDGER.md 以及适用实验索引。被关闭的旧问题保留历史证据，新问题必须说明为何改变研究决策。

最终独立审查逐项检查：公式与源码、5seed完整性、配对身份、Test冻结、保护参照、结果/机制层次、非重复预算、恢复与状态实际可用性。审查结论与修改记录随报告发布，最后回读远端提交。
