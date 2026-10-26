# RCF-v1实验交付入口

先读[最终科研报告](FINAL_REPORT.md)。主线34个数据×方法条件各有5201–5204四个独立训练seed，共136条方法完成证据；独立SCALE两数据各2022–2025四seed。Q门槛未通过的S4为NO_START，未把开发seed冒称独立锁定确认。

八份正式报告：

| 文件 | 内容 |
|---|---|
| [FINAL_REPORT.md](FINAL_REPORT.md) | 结论、全部关键反例、证据强度和具体行动 |
| [SOURCE_INCREMENT_REPORT.md](SOURCE_INCREMENT_REPORT.md) | 固定内容、读取、直接/本地强对照及角色边界 |
| [FUNCTION_FORMATION_REPORT.md](FUNCTION_FORMATION_REPORT.md) | 形成、S/R/K/c、K1/初始化、交叉功能和OOF/refit |
| [SCALE_REPORT.md](SCALE_REPORT.md) | 独立Adam尺度桥、无增强四seed、历史增强与人口权衡 |
| [COST_REPORT.md](COST_REPORT.md) | 共享/独立范围、参数、真实计时和无法补造的缺测 |
| [DATA_AND_IDENTITY.md](DATA_AND_IDENTITY.md) | 自然角色、模型/目标/选点、环境与分版本身份 |
| [IMPLEMENTATION_REVIEW.md](IMPLEMENTATION_REVIEW.md) | 设计—实现—独立审查—运行链及修订历史 |
| [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) | 未区分解释及最有判别力的后续实验 |

[表格说明](tables/README.md)解释seed级区间、Macro/pooled、支持、阶段和成本口径。[figures/](figures/)提供同表生成的PNG与独立PDF。大CSV以无损`.csv.gz`发布，服务器仍保留原CSV；路径、原始/压缩SHA见`PUBLICATION_FILE_MANIFEST.json`。

正式验收入口为`CORE_FOUR_SEED_ANALYSIS_RECEIPT.json`、`SUPPORT_FOUR_SEED_ANALYSIS_RECEIPT.json`、`CROSSOVER_FOUR_SEED_ANALYSIS_RECEIPT.json`、`POPULATION_FOUR_SEED_ANALYSIS_RECEIPT.json`、`COST_FOUR_SEED_ANALYSIS_RECEIPT.json`和`SCALE_FOUR_SEED_ANALYSIS_RECEIPT.json`。`TRAINING_COMPLETION_MANIFEST.json`索引136训练收据；`ARTIFACT_INDEX.json`、`EXPORT_ACCEPTANCE.json`和`CLEANUP_RECEIPT.json`提供真实保留与清理状态。实际Git提交和blob验证见`PUBLICATION_RECEIPT.json`。

其余DESIGN/CODER/REVIEW/RECOVERY/VERIFY文件是带版本的过程证据；其中当时的NOT_RUN/待运行状态不代表最终状态，也不能用后续完成倒写旧收据。旧两seed分析已SUPERSEDED，仅用于实现数值差异/恢复历史，科学结论只用完整四seed。加载训练源码、冻结源码与后加观察器身份分别保留。

代码入口在[新实验目录](../../experiments/mdmt_rcf_20261026/README.md)。实际工作区是`/home/xzf/Code/phase0-3`；协作名称`/root`不是路径。大规模逐行预测、φ、δ、请求和日志留在服务器，不随Git上传。模型/Adam/RNG/checkpoint没有落盘；未来新前向或未导出的表示分解可能需要重建prefix。
