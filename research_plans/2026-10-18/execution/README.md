# 2026-10-18 六方向实际执行

本目录是实际执行产物，规划原文保留在上一级。

- [综合研究报告](FINAL_REPORT.md)：六方向判定、全部正负证据、证据强度与最小后续。
- [P1](P1/analysis.md) / [P2](P2/analysis.md) / [P3](P3/analysis.md) / [P4](P4/analysis.md) / [P5](P5/analysis.md) / [P6](P6/analysis.md)：逐方向分析，各有独立 `derivation.md`。
- [设计](DESIGN_SPECIFICATION.md)、[实现报告](CODER_IMPLEMENTATION_REPORT.md)、[自审](REVIEW.md)、[原始拟合报告](P3_RUNNER_RAW_REPORT.json)、[科学必要检查](SCIENTIFIC_ACCEPTANCE.json)。
- [文献原文核查](LITERATURE_AUDIT.md)、[复现命令](REPRODUCE.md)、[执行清单](EXECUTION_CHECKLIST.md)、[环境和谱系](provenance.json)。
- `theory_data/`：支持/识别/预算/代理 oracle、原始 schema 和时间/遮蔽/审计数据。
- `P3_training/seed*/`：四种子拟合配置、系数、原始预测、逐 cell 指标、运行收据。
- `measurements/seed*/features_*.npz`：P3 原始冻结特征、teacher、标签与 row_id。
- `measurements_functional/seed*/`：正式 P4 Gram、P6 样本矩和计时原始数据。
- `readouts/`：可由原始数据重算的逐 cell、宏/人口、成本和方差读数。
- `archive/` 和原 `measurements/` 的 P4/P6：修订前定义与读数，保留追溯，**风险方向结论以 functional 版本为准**。
- [研究图 PNG](RESEARCH_RESULTS.png) / [PDF](RESEARCH_RESULTS.pdf)：误差棒为四个 seed 标准差，非置信区间。

本轮完成四种子冻结头新拟合，没有进行新 backbone/采样器端到端训练。
所有条件触发与停止均有证据，不把未触发训练写为已失败。当前无候选
通过独立新方法的继续开发门槛；这不否定整个研究问题。
