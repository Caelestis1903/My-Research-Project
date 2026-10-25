# 完整数值附录索引

本附录仅转录完成包的机器聚合数值，不作算法效果判断。质量数值固定显示10位小数，缺失或不可定义值显式为 `NaN`。`selected` 是各预测器按预登记验证指标、eligible≥12且平局取最早epoch的独立选点；`endpoint` 是同一预登记训练终点（开发24epoch，若请求正式manifest则为其50epoch）。Q_explore为已暴露开发评价，不是新未见确认。

统计单位为训练seed。配对差统一为候选减参照；AUC正差、LL负差对应有利方向。`favorable_gain`已统一为越大越有利。seed级Student t95%区间在n=2仅作开发描述；n=1无区间。仅actual formal阶段元数据对应的统计才属于锁定复评，实际n逐表列出；formal与开发、P2归因的seed分别分组，不能合成n=6。域、任务、cell与epoch不增加独立重复数；不报告p值。

完成包计数：

| logical_trajectories | local_extra_predictors | predictor_checkpoint_rows | six_cell_rows | validation_selected_rows | old_aux01_reference_checkpoint_rows |
| --- | --- | --- | --- | --- | --- |
| 89 | 8 | 194 | 1164 | 97 | 8 |

本附录覆盖所请求的实际manifest，不推断未触发P2或正式矩阵。每项完整checkpoint/source身份、原始绝对指标、配对值与seed统计均可由下列附录独立审阅。

- [APPENDIX_RUN_REGISTRY_AND_IDENTITIES.md](APPENDIX_RUN_REGISTRY_AND_IDENTITIES.md)
- [APPENDIX_PER_SEED_METRICS.md](APPENDIX_PER_SEED_METRICS.md)
- [APPENDIX_SIX_CELLS_KUAIRAND.md](APPENDIX_SIX_CELLS_KUAIRAND.md)
- [APPENDIX_SIX_CELLS_MOVIELENS.md](APPENDIX_SIX_CELLS_MOVIELENS.md)
- [APPENDIX_VALIDATION_SELECTION.md](APPENDIX_VALIDATION_SELECTION.md)
- [APPENDIX_PAIRED_SEED_VALUES.md](APPENDIX_PAIRED_SEED_VALUES.md)
- [APPENDIX_SEED_STATISTICS.md](APPENDIX_SEED_STATISTICS.md)
- [APPENDIX_CELL_CONTRASTS_KUAIRAND.md](APPENDIX_CELL_CONTRASTS_KUAIRAND.md)
- [APPENDIX_CELL_CONTRASTS_MOVIELENS.md](APPENDIX_CELL_CONTRASTS_MOVIELENS.md)
- [APPENDIX_DOMAIN_TASK_CONTRASTS.md](APPENDIX_DOMAIN_TASK_CONTRASTS.md)
- [APPENDIX_OLD_AUX_FACTORIAL.md](APPENDIX_OLD_AUX_FACTORIAL.md)
- [APPENDIX_SIZE_BASELINE_AND_WORST_CELLS.md](APPENDIX_SIZE_BASELINE_AND_WORST_CELLS.md)
- [APPENDIX_PROVENANCE.md](APPENDIX_PROVENANCE.md)
