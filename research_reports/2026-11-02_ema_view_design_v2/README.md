# EMA02V2 完整研究记录

36条正式轨迹、1,800 epoch、62,100次optimizer更新全部完成，144个冻结test端点评价完成，无训练或评价失败。本轮停止新增视图候选，保留A0＋固定EMA。

- [研究报告](REPORT.md)：结论、逐seed结果、全部域任务、阶段差异、机制解释和成本。
- [实验代码](../../experiments/ema_view_design_20261102_v2/)；执行commit `e3ca50380fff0535d16b73888cc6e462391021ca`。
- [设计](DESIGN_SPECIFICATION.md)、[SVD设备补充](DESIGN_ADDENDUM_SVD_PLACEMENT.md)。
- [核心实现报告](CODER_CORE_REPORT.md)、[执行实现报告](CODER_EXECUTION_REPORT.md)、[SVD实现报告](CODER_SVD_REPORT.md)。
- [独立审查](REVIEW_REPORT.md)、[最终增量审查PASS](SVD_REVIEW_REPORT.md)、[runner记录](RUNNER_REPORT.md)。
- [S1阶段判断](S1_VALIDATION_ANALYSIS.md)、[备选分支判断](S2B_VALIDATION_ANALYSIS.md)、[冻结前最终决定](FINAL_VALIDATION_DECISION.json)。
- [验证原值](tables/validation_metrics.csv)、[test原值](tables/test_metrics.csv)、[汇总](tables/endpoint_summary.csv)、[配对差/SD/区间](tables/paired_effects.csv.gz)。
- [逐epoch完整验证与训练损失](tables/epoch_metrics.csv.gz)、[逐格差](tables/cell_effects.csv)、[最小配对格差](tables/minimum_cell_gains.csv)、[EMA收益](tables/ema_effects.csv)、[机制日志汇总](tables/diagnostics_by_run.csv)。
- [完整标量原始记录](RAW_SCALAR_RECORDS.tar.xz)、[SHA256](RAW_SCALAR_RECORDS.tar.xz.sha256)、[归档收据](RAW_ARCHIVE_RECEIPT.json)。
- [冻结清单](raw/selection_frozen.json)、[Stage0](raw/STAGE0_REPORT.json)、[派发/成本/原始执行收据](raw/)。
- [来源收据](SOURCE_RECEIPT.json)、[执行身份](EXECUTION_IDENTITY.json)、[40文件远端blob核验](REMOTE_EXECUTION_VERIFICATION.json)。

从仓库根用项目Python环境运行 `python -m experiments.ema_view_design_20261102_v2.run --help`。本轮准确派发文件和命令保留在raw及归档中；真实CSV位置和SHA256见报告。模型训练状态全程保存在RAM，评价完成后已释放；复现从原数据重新训练。

报告中的新方法重复单位为两个开发seed（2022/2024）。确认和额外窗口未触发，阶段停止决定先于统一test冻结。
