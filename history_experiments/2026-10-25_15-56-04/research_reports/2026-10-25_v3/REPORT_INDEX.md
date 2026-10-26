# MDMT-NEXT-v3 完整实验报告索引

本目录对应 `start.md` 指向的 `research_plans/26-10-25_v2/`。已完成全部已触发规定实验：69条24epoch及20条50epoch逻辑轨迹，178个历史角色评分、388个预测器检查点角色记录和51项统一推理测量。最终方法冻结于2026-10-25T12:02:26.282158Z。

## 研究结果

- [总报告](FINAL_REPORT.md)
- [RHD 完整闭环](RHD_CLOSED_LOOP_REPORT.md)
- [LPS 形成与读取](LPS_FORMATION_REPORT.md)
- [消融与外部参照](ABLATION_AND_BASELINE_REPORT.md)
- [锁定新 seed 复评](FORMAL_LOCKED_REEVALUATION.md)
- [成本与信息访问](COST_REPORT.md)
- [尚未解决的问题](OPEN_QUESTIONS.md)

## 可核验流程

- [原始工件审查](ARTIFACT_AUDIT.md)
- [规范澄清](SPEC_RESOLUTION.md)
- [设计规格](DESIGN_SPECIFICATION.md)
- [独立实现审查](IMPLEMENTATION_REVIEW.md)
- [P0 机制复核](P0_MECHANISM_REVIEW.md)
- [条件设计补充](CONDITIONAL_DESIGN_ADDENDUM.md)
- [条件独立审查](CONDITIONAL_IMPLEMENTATION_REVIEW.md)
- [完整数值附录](APPENDIX_INDEX.md)

原始数据、逐行预测、完整checkpoint和源码归档保留在服务器。公开目录只包含报告、数值附录及报告图，明确区分已暴露开发评价、新seed随机性和真正新增独立数据。

## 完整数值与身份入口

- [Q开发角色全部数值索引](APPENDIX_INDEX.md)
- [历史test总体及逐seed](APPENDIX_EXPOSED_BENCHMARK_TEST_EXPOSED_METRICS.md)
- [历史test全部六格](APPENDIX_EXPOSED_BENCHMARK_TEST_EXPOSED_SIX_CELLS.md)
- [历史test配对差与统计](APPENDIX_EXPOSED_BENCHMARK_TEST_EXPOSED_PAIRS_AND_STATS.md)
- [历史test域/任务/六格对照](APPENDIX_EXPOSED_BENCHMARK_TEST_EXPOSED_CELL_DOMAIN_TASK.md)
- [历史test检查点及角色身份](APPENDIX_EXPOSED_BENCHMARK_TEST_EXPOSED_IDENTITIES.md)
- [Legacy-confirm总体及逐seed](APPENDIX_EXPOSED_Q_LEGACY_CONFIRM_EXPOSED_METRICS.md)
- [Legacy-confirm全部六格](APPENDIX_EXPOSED_Q_LEGACY_CONFIRM_EXPOSED_SIX_CELLS.md)
- [Legacy-confirm配对与统计](APPENDIX_EXPOSED_Q_LEGACY_CONFIRM_EXPOSED_PAIRS_AND_STATS.md)
- [Legacy-confirm域/任务/六格对照](APPENDIX_EXPOSED_Q_LEGACY_CONFIRM_EXPOSED_CELL_DOMAIN_TASK.md)
- [Legacy-confirm检查点及角色身份](APPENDIX_EXPOSED_Q_LEGACY_CONFIRM_EXPOSED_IDENTITIES.md)
- [正式四seed主要比较及图](APPENDIX_FORMAL_COMPARISON_SUMMARY.md)
- [预定检查点质量与累计已记录成本](APPENDIX_QUALITY_COST_OBSERVATIONS.md)
- [验证AUC/LL次别名身份](APPENDIX_VALIDATION_SECONDARY_ALIASES.md)
- [协议、源码与实际父状态](APPENDIX_PROTOCOL_AND_PARENT_IDENTITIES.md)
- [STGU验证LL选点校正](APPENDIX_STGU_LL_SELECTION_CORRECTION.md)
- [成本与信息访问索引](APPENDIX_COST_INDEX.md)
- [全部51项统一推理](APPENDIX_INFERENCE_COMBINED_NUMERICAL.md)
- [P0完整数值](APPENDIX_P0_NUMERICAL.md)
- [固定状态删除/同bank/无标签群组](APPENDIX_LPS_FIXED_STATE_DIAGNOSTICS.md)
- [形成元步执行](APPENDIX_LPS_FORMATION_EXECUTION.md)
- [1152元步](APPENDIX_LPS_META_STEPS.md)
- [来源权重完整汇总](APPENDIX_LPS_SOURCE_WEIGHTS.md)
- [CAL裁剪与GEOM执行](APPENDIX_CAL_AND_GEOM_DIAGNOSTICS.md)
- [发布范围及服务器工件](PUBLICATION_SCOPE.md)

设计/编码/独立审查文档保留其当时的准入范围与历史时态；是否最终完成以本索引、总报告及最终收据为准。公开数值保留阴性和矛盾，不合并开发n=2与正式n=4。
