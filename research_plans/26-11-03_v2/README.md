# 四候选五 seed 补全与正式模型比较方案

入口：根目录 `start.md` → `buchong.md`。本轮完成四候选补全实验，并交付下一阶段已发表模型统一比较方案。

按以下顺序阅读：

1. [补全前的真实成绩与 seed 覆盖](INITIAL_EVIDENCE.md)、[逐 seed 原值](INITIAL_SEED_VALUES.csv)。
2. [设计规格](DESIGN_SPECIFICATION.md)、[40 条冻结矩阵](PHASE1_MATRIX.csv)、[复用审计](DESIGN_REUSE_AUDIT.json)：10 条复用、30 条补跑；种子仅 2022–2026。
3. [实现与直接运行命令](../../experiments/mdmt_five_seed_completion_20261103/README.md)、[独立审查](../../research_reports/2026-11-03_mdmt_five_seed_completion/REVIEW_REPORT.md)。
4. [完整结果与研究裁定](../../research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md)、[逐 seed 六格数值](../../research_reports/2026-11-03_mdmt_five_seed_completion/NUMERICAL_APPENDIX.md)。
5. [下一阶段已发表模型比较协议](PHASE2_PUBLISHED_MODEL_PROTOCOL.md)：统一人口、预处理及 validation Macro AUC 选点；保留 HyperGate、AutoIFS 必要训练流程。

评价分别记录完整方法相对普通 A0 online 的总收益，以及相对 A0 固定 EMA 的组件增量。五 seed 平均 Macro AUC 差≥.003 为总体达标；六格各自平均差≥−.001 为逐格约束；平均 LogLoss 差<0 单独裁定。方向数及置信区间不增加淘汰门，1–4 seed 不终止矩阵。

所有训练完成后统一冻结选点，再评价新增180个测试端点。模型状态只驻RAM；标量归档和结果复算见报告目录的 `RAW_ARTIFACTS.md`。主目录 [算法指标台账](../../MDMT_ALGORITHM_METRICS_LEDGER.md) 保留不同协议分组及原始来源。

最新规范来源为 main 固定提交 `18d18bad6afdf84bc7c13aa8f303cc3f32019b12`，见 [来源索引](source_main/SOURCE_INDEX.json)。
