# 实验执行报告：规定流程已完成

全部规划V0–V14、已触发分支、冻结test与数值核验、问题闭合及14章报告已完成。唯一总分类为 **BOUNDARY-CONFIRMED**；方法决定 **NO_GO**。最终远程提交须以Git实际push及ls-remote回读为准，不以“已启动”替代完成。

本轮使用隔离目录`/home/xzf/Code/phase0-3-20261016-v3-git`，分支`codex/execute-2026-10-16-v3`，保护原工作区未提交改动与缓存。原始start.md为GB18030，已保存[source_start_utf8.md](source_start_utf8.md)；其§11优先约束为主控完成设计/实现/自审/分析，子agent只机械执行进程，故independent_review=false。没有伪称独立review。

## 环境和执行约束

使用既有`phase0-3-env`：Python3.13、PyTorch2.9/CUDA12.6、2×RTX3090、32CPU、约64GiB RAM。无必要依赖重装或数据重下载。原始数据完整性KR SHA a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f、ML SHA f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4；每source config另含完整人口/编码/预处理指纹。训练种子2022–2025，q/模型/任务/初值/目标/超参数按冻结设计。没有根据test改训练条件。

持久Popen/start_new_session队列保存真实PID/argv/日志/状态；每GPU≤5且显存<90%，所有GPU前序状态纳入调度，CPU阶段按2前缀/8summary等资源约束。未使用ionice/nice或进程优先级变动。完整模型/预测/批次留本地artifact盘；GPU最终状态与真实退出记录见[状态快照](final_actual_GPU_queue_state_archive.json)。直接工具分析未捕获OS子PID时如实记null和真实tool session，不伪造PID，见[记录](primary_direct_analysis_execution_receipts.json)。

|阶段|已完成内容|实际范围|证据|
|---|---|---|---|
|V0|阅读/历史纠错/冻结定义|规划d18f194；旧Equal/late-rescue不复活|[V0_decision.json](V0_decision.json)|
|V1|同目标与完整图梯度审计|16θ×512batch/q；1,614,254坐标及exact控制|[V1_decision.json](V1_decision.json)|
|V2|固定参考轨迹|Adam.01/SGD.1各四seed两q，固定轮/早期/选轮独立|[V2_fixedSGD_and_interaction_sensitivity_analysis.md](V2_fixedSGD_and_interaction_sensitivity_analysis.md)|
|V3|Selection规则与基线|7家族28对112000次选择检查；冻结test完成|[V3_final_baseline_selection_analysis.md](V3_final_baseline_selection_analysis.md)|
|V4|校准/2B/足预算/学习速度|16pilot；正式训练与延长；2448hit/censor，20等步记录|[V4_final_budget_and_learning_speed_analysis.md](V4_final_budget_and_learning_speed_analysis.md)|
|V5|域内采样与顺序|40最终训练4200臂轮205800批；350来源22前缀完成|[V5_SGD_final_scientific_obligations_primary_certificate.json](V5_SGD_final_scientific_obligations_primary_certificate.json)|
|V6|谱/块/任务/稀疏时间与矩遥测|320几何行、8完整任务梯度；254快照756800矩行刷新|[V6_geometry_observational_analysis.md](V6_geometry_observational_analysis.md)|
|V7|共享/私有块局部反事实|12局部×7条件；C AUC筛选阳性，联合loss门槛失败，条件完整训练未触发|[V7_local_gate_primary_decision.json](V7_local_gate_primary_decision.json)|
|V8|名义规模/输入困难度/标签曝光|2x/4x精确别名；三强度4seed；576单元/384稀疏范数；3300物理batch文件|[V8_final_diagnostic_obligations_primary_certificate.json](V8_final_diagnostic_obligations_primary_certificate.json)|
|V9|第二环境|ML四seed50；宏观负效应未稳定复现，深层机制分支未触发|[V9_final_cross_environment_analysis.md](V9_final_cross_environment_analysis.md)|
|V10|文献/目标/AdaTask|9原文审计；同目标组件4seed50/100/50/50|[V10_final_vs_calAdam_analysis.md](V10_final_vs_calAdam_analysis.md)|
|V11|完整状态与影子反事实|最终250/150/100/200；12最终+27历史配对、156000选择检查|[V11_final_diagnostic_report_primary_review.json](V11_final_diagnostic_report_primary_review.json)|
|V12|六CF因果闭环|CF1–6全部完成，结构NI与条件未触发明确|[V12_final_causal_synthesis.md](V12_final_causal_synthesis.md)|
|V13|十项方法化判断|NO_GO|[V13_final_method_decision.md](V13_final_method_decision.md)|
|V14|终局问题/报告/归档|46问题全部终态、19新问题六字段、14章；推送核对后任务结束|[FINAL_REPORT.md](FINAL_REPORT.md)|

## 实际预算和停止条件

V4 native Adam50/100/50/50、SGD B0/B2各50/50/50/100；V5 Adam50/100/100/50；V5 SGD50/50/50/350；V8三强度50/200/50/350；V10 AdaTask50/100/50/50；V11三状态共同250/150/100/200。按train最近5/前5相对改善≤.005或selection确认过拟合停止，两臂及预定匹配组共同延长；实际350并非硬上限，seed25此前300不满足而继续。未用diagnostic/test决定预算。实用平台不等于全局收敛。

V7不是全部阴性：C AUC在10/40轮四seed≥50%衰减，第二阶段LL/J方向冲突，原联合条件完整门槛未满足。其长期架构对照按条件未触发，并非遗漏必做。额外2026–2029未触发，因为核心关闭不依赖±.003关键null等效。所有NI结论均已有规定最小实验及结构/联合干预限制。

## 冻结test、核验与归档

[test全局冻结](test_readout_frozen.json)先于本轮任何test指标，固定46问题定义、14初始/最终cohort、所有producer/config/model/源SHA、四选择规则及窗口。156来源目录、79个不同paired分析目录、160个cohort-pair记录有重叠，不能当独立seed。按字节完全相同config/model分组，120原生来源实际13500模型轮次test推断，36精确前缀引用2200已推断模型轮次；所有156来源94200条epoch×cell记录（含复用）已归档。

首个AdaTask23 SN100→50前缀全部50×6单元从实际test标签与预测以独立AUC/LL公式核验，最大误差8.33e−17。[全14组数值重建](analysis/test_all_cohorts_primary_verification/receipt.json)核验748376数值，最大误差5.773159728050814e-15，包含全部选择重采样、实际n1–4/预算资格、final别名、所有原生预测hash和既定交互。历史test已暴露，不声称pristine holdout。

[156来源归档核验](final_test_archive_primary_verification.json)、[原始本地产物hash索引](final_test_local_raw_hash_index.json)、[精确主控核验源码](primary_final_audit_source_archive.json)。上传CSV/JSON/图PDF/PNG和可复现源码，不上传数据集、模型权重、预测NPZ、冗余日志或缓存。Q34 raw155.7MiB本地SHA索引、compact1016组归档。

## 异常、修复与未执行分支

早期V4首次启动的工程故障及最小修复保留于[V4_attempt1_postmortem](V4_attempt1_postmortem.json)和[V4_startup_fix_review](V4_startup_fix_review.json)，没有覆盖历史或把失败标完成。主控test前缀独立校验首次因DataFrame列顺序比较失败；修正校验的列对齐后全部300单元PASS，未改冻结实验源码、数据或指标，也未重复推断/前缀生产。14组汇总首次在JSON输出边界因NumPy int64类型失败，原冻结统计源码保持原SHA；独立兼容wrapper仅将标量转为Python原生类型，14个失败目录完整保留，重跑实际全部exit0，42份原有CSV/JSON逐字节相同；见[修复核验](test_summary_serialization_repair_actual_primary_verification.json)。正式最终test任务及汇总修复重跑均记录实际exit0；队列完成状态回读，持久launcher的returncode保持null，不伪称直接等待到队列OS退出码。动态Q28–Q46共19项完整保留。

没有剩余规定必做实验。未触发条件分支、结构不可识别和更强未来假设逐项写明；不以“future work”掩盖当前必做缺口。最终验收以[FINAL_REPORT](FINAL_REPORT.md)、[问题全集](question_registry.json)、Git实际远程验证记录为准。
