# Stage2D 续执行索引

16/16正式训练完成、全部exit0；400项产物hash、完整配对流及32份selected预测指标重算通过，1000次逐cell条件row bootstrap完成。主代理科学分析完成；不是仅启动、仅写方案或仅数值oracle。

- [执行报告](execution_report.md)：协议、实际过程、验证、全计划任务状态及未完成历史依赖。
- [结果报告](experiment_results.md)：AUC/LL、域/任务/seed、selection/common/final、曝光和条件区间。
- [机制分析](mechanism_analysis.md)：STRONG/MEDIUM/WEAK、反例、竞争解释、规模/trade-off与条件判断。
- [冻结设计](design_specification.md)、[分析规格](analysis_specification.md)、[自审](review.md)、[提前结束纠正](continuation_gate_reassessment.md)。
- [当前执行状态](execution_status.json)、[科学判断](scientific_decision.json)、[全计划清单](plan_execution_ledger.json)。producer receipt中的PENDING_MAIN_AGENT为其生成时的分工快照；后续主代理结论以scientific_decision为准。
- [统一selected schema](analysis/unified_selected_run_schema.csv)、[配对差](analysis/selected_paired_differences.csv)、[seed统计](analysis/seed_summary.csv)、[逐cell原始指标](analysis/selected_cell_metrics.csv)。
- [数值核验](analysis/receipt.json)、[状态/选点表核验](analysis/readout_receipt.json)、[条件bootstrap](analysis/bootstrap_receipt.json)、[复现说明](reproduce.md)。

正式run在 `runs/seed{2022,2023,2024,2025}/{SN,SB,EN,EB}/attempt1/`；关键config/manifest/metrics/batch fingerprints及validation_selected/test_selected NPZ随Git归档。原始数据、权重/优化器state、common/final完整预测、完整日志留本机，路径与hash见 `local_artifact_index.csv`。不上传大体积checkpoint或原始数据集。

SB相对强SN在本设置有稳定损伤，EB相对EN无稳定实用收益；条件升级及E1不满足。没有伪称整个Track A无效，也没有把历史原件缺失写成科学NO。Git归档确认见 `publication_status.json`。
