# 最终报告与证据索引

**COMPLETE**：120条H100训练，72条正式test轨迹、7200逐epoch模型/预测，全部完成必要核验。正式训练seed2032–2039；tuning2030/31；N3开发2032–35、独立确认2036–39。本页为本轮结果入口，原规划文件保留其制定时语境。

## 研究结论

|报告|内容|
|---|---|
|[FINAL_ANALYSIS.md](FINAL_ANALYSIS.md)|主结论、反例、STRONG/MEDIUM/WEAK证据、竞争解释及最有辨别力的下一实验|
|[experiment_results.md](experiment_results.md)|AUC/LL/J、阶段、任务和种子汇总|
|[mechanism_analysis.md](mechanism_analysis.md)|机制反证、未识别解释、效应/机制/方法三层判定|
|[execution_report.md](execution_report.md)|实际矩阵、资源、异常修复、恢复和交付|
|[METRIC_AND_POPULATION_DECOMPOSITION.md](METRIC_AND_POPULATION_DECOMPOSITION.md)|权重、人口、baseline、历史校准与不确定性单位|
|[Q_LR_BUDGET_RESPONSE.md](Q_LR_BUDGET_RESPONSE.md)|双q对称LR、外部预算、selection与晚期反例|
|[DOMAIN_COUNT_COUNTERFACTUAL.md](DOMAIN_COUNT_COUNTERFACTUAL.md)|固定点四臂、全程quota交互与绝对lift|
|[C_LOCAL_AND_TOPOLOGY_ANALYSIS.md](C_LOCAL_AND_TOPOLOGY_ANALYSIS.md)|局部开发/SGD对照/独立确认及N4未触发|
|[LITERATURE_COVERAGE.md](LITERATURE_COVERAGE.md)|原文干预对照及本轮结论覆盖边界|
|[QUESTION_REGISTRY_FINAL.json](QUESTION_REGISTRY_FINAL.json)|原Q1–Q46逐字段保留，新Q47–Q53及终态|

## 数值与可视化

- FINAL_NUMERIC_FINDINGS.json、FINAL_SELECTED_ENDPOINTS.csv、FINAL_STAGE_MACRO_AUC.csv、FINAL_SELECTED_CELL_METRICS.csv：可读数值，来源代码FINAL_REPORT_TABLES_SOURCE.py。
- FINAL_performance/：8配对训练seed原始差、t95、符号随机化、bootstrap及主Holm族。
- N2_quota_performance/：四臂、交互及两个单臂绝对lift；FINAL_method_guards/、N2_absolute_quota_method_guards/：宏收益与六cell/LL保护界。
- FINAL_domain_context/：每域训练规模、test规模、标签率、AUC/LL baseline及严格/宽松取舍事件。
- FINAL_figures/：预算曲线、六cell和权重敏感性PNG/PDF；N0/evaluation_decomposition.png及同名PDF、N2_figures/、N3_figures/补充机制图。PNG均已实际查看。
- N2_fixedpoint_summary/、N3_development_summary/、N3_SGD_control_summary/、N3_confirmation_summary/：完整局部原始汇总与一次冻结的候选/独立门槛。

## 轨迹与冻结证据

- FINAL_TRAINING_RUN_LEDGER.csv、completed_artifact_index.json、run_metadata/：120条实际训练，48调参不读test。
- GLOBAL_MODEL_AND_PROTOCOL_FREEZE.json、FINAL_COMPARISONS.json、TEST_manifest.json：统一test前冻结提交889e012，56配对比较、72唯一正式轨迹、7200模型。
- TEST_READOUT_VALIDATION.json、TEST_RUN_LEDGER.csv、test_metadata/：72条成功读数及43200有限cell行；TEST_PREDICTION_ARTIFACT_INDEX.json记录预测/模型SHA，读数器实际逐模型核对并逐预测存SHA。
- TEST_launch.json、TEST_manifest.status.json、TEST_mechanical_observations.jsonl：实际启动、原生退出与机械记录；原生状态不由报告回写。
- CONDITIONAL_BRANCH_DECISIONS_PRETEST.json、PRETEST_TRAINING_STAGE_CLOSURE.json：H200/N4/N5未触发分支的冻结依据；未触发不是阴性。
- REGISTRY_PRESERVATION_CHECK.json：固定Git原46对象深比较完全一致；原计划声明SHA与原始Git blob SHA不符，两者保留，不改科学字段。
- GLOBAL_FREEZE_PROVENANCE_VERIFICATION.json：数值源码及冻结证据来源；LITERATURE_COVERAGE_PRETEST_FROZEN.md保存提交889e012的原文学术对照原版本，最终报告另加读数说明。
- execution_status.json、FINAL_DELIVERY_MANIFEST.json：最终状态与交付文件SHA。

## 产物位置与复用

实际checkout：/home/xzf/Code/phase0-3-20261017-git。分支：codex/execute-2026-10-17-run，origin同名分支。最终交付commit见Git HEAD及交付消息。

模型、逐样本预测与原始trace：/home_new/xzf_phase0_3_n17_artifacts_20261017，经本目录runs符号链接引用。Git收代码、配置、必要CSV/JSON及PNG/PDF；大二进制保留本地产物盘。原工作区/home/xzf/Code/phase0-3的未跟踪内容与无HEAD状态保留。Python使用既有phase0-3-env。

所有非机械设计、实现、自审、分析由主agent完成（start.md §11，independent_review=false）；机械runner只执行/监控/记录。当前没有待完成实验；建议的下一轮q×LR完整交叉及Q53精度设计均未被伪称已运行。
