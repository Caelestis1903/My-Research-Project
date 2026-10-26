# REVIEW REPORT — AC-V9

日期：2026-10-20。独立review，范围仅8条AC-V9。已读核心设计、实现报告、actual v9.py/v9check.py/freeze_configs.py、原v3/v2/audit代码与检查工件。未运行正式训练或研究分析，未改coder代码。

## 1. Verdict

**PASS WITH FIXES；当前不得RUN。** V9恢复checkpoint/观测的一致性阻塞项须修复并复查。S5/S6和AC2D的独立放行不受本报告影响。

## 2. Requirement-by-requirement verification

|要求|actual核对|结论|
|真实原producer|归档v3_train与原commit `0f83141bcdf033200fba678965ec808e69983e8c` Git blob相同；wrapper继承原fit/step，固定网络仍原v2.fixed_model|通过|
|网络/优化器|固定四fusion buffer，原routing detach/网络保留；唯一SGD lr=.3/momentum0/WD1e-5，无DARTS或新增梯度控制|通过|
|人口/loss/sampler|统一自然bundle800167行；p实际重算，SN q=p、SB q=uniform；双任务BCE mean按p/q固定batch分母；原IID-WR sampling/ORIGINAL order，无corruption/duplication|通过|
|预算|SN/SB×四seed共8，50完整epoch、1000main步，batch40960，尾批21927；原完整empirical objective/exposure/repeat观测保留|通过|
|原选模|原audit.frozen_split按domain×joint-label内sha256(20261016:row_id)排序取半；selection50014/diagnostic50007；选点仅selection宏AUC最早严格最大；未替换full-validation或test选点|通过|
|附加指标/test|额外full-validation指标从原每epoch完整预测读取，不改变选点；原fit不test，wrapper先selection_frozen_natural再评价selected/final及E10/20/30/40/50；零更新；新表epoch-1注明原1起始语义|通过|
|数学基本检查|actual checker以21真实行显式逐域loss、raw梯度、独立SGD实际参数变化与新step相等；SN/SB最终JSON误差全0；检查source stamp与当前381项一致|通过|
|冻结/日志|8配置source/hash与frozen配置一致；独立输出；最终receipt刷新为完成工件SHA，run_manifest/sha_index不相互递归；恢复时工件与CSV一致性不足|见V1/V2|

## 3. Bugs/deviations found

### V1 — MAJOR / BLOCKING：last_resume后的checkpoint提交窗口未修复

位置：`v9.py:39–51` 完全继承原v3恢复；归档 `v3_train_20261016.Runner.fit` 顺序为save last_resume，然后selected_state/common checkpoint/epoch_model。若进程在该窗口中断，原恢复从saved epoch+1继续，未补写该epoch工件。saved selected/history可以指向新epoch，而selected_state仍为旧epoch或缺失；最后测试将错评或因缺文件失败。final/common文件也可能缺失。

最小修复仅在wrapper恢复生命周期中：从原子last_resume state核对/重建该saved epoch应有的selected_state（当state.selected==state.epoch且epoch>=1）、原COMMON checkpoint和原每epoch model快照；初始化state需保留原init checkpoint。复用saved state/model，不执行任何optimizer.step，不更改原选模或预算；核对selected文件epoch与冻结选择一致后再test。不能用final替代缺失selected。

### V2 — MAJOR / BLOCKING：未清理不完整epoch追加的观测

位置：原v3 resume只恢复state/sampler/batch-chain，wrapper无CSV截断。中断发生在新epoch训练或validation日志后、last_resume提交前时，下一次完整重跑该epoch会追加重复batch/optimizer/selection/trajectory行，破坏实际步数、曝光和阶段分析。

最小修复：在wrapper恢复后，依据last_resume的**原生1起始epoch**，对原训练CSV `batch_trace.csv`、`optimizer_observations.csv`、新 `optimizer_full_observations.csv`、`metrics_selection.csv`、`trajectory.csv` 保留epoch<=saved_epoch，清除后续不完整epoch的行。新wrapper的0起始metrics_epoch/test/aggregates可维持fit后已有完整重建逻辑，不混用其epoch口径。原预生成batches/validation预测会在重跑相应epoch时覆盖，不需改变sampler或训练顺序。

## 4. Corrections made

未改coder代码；V1/V2交coder修复，之后独立复查。当前不写passing gate。

## 5. Remaining risks

当前fresh无中断路径的原训练数学、选模和测试顺序通过；单batch检查不覆盖上述恢复窗口。已有检查没有正式epoch训练。test历史重复使用，最终科学结论须标探索性复现。

## 6. Whether experiments are safe to run

**当前否。** 修复V1/V2后仅可签8条AC-V9的独立gate `review_gate_v9.json`；不改现有AC2D或S5/S6 gate。
