# 2026-10-16 最终研究产物索引

B1重建8/8完成，400状态/800预测核验通过；主agent科学分析已完成，历史exact缺口及条件未执行项明确保留。

入口：[执行报告](execution_report.md)、[结果报告](experiment_results.md)、[机制分析](mechanism_analysis.md)、[全部条件节点判定](final_scope_decisions.md)、[B1完整分析](B1_reconstruction_20261016/scientific_analysis.md)。

B1机器数据在 `B1_reconstruction_20261016/analysis/`、`interpretation/`；原始配置/manifest/选点/dense哈希索引及epoch指标在 `run_metadata/`。C2.1及历史审计数据保留各自范围。模型、数据集和完整逐行预测留本地，不上传。最终发布状态见 `publication_status.json`。

以下为先前阶段的索引/恢复说明；最终状态以上述入口与机器状态为准。不要重跑旧build_reports覆盖最终报告。

> 2026-10-16续执行更新：用户最新禁止子 agent；此前两项派发已中止，之后由主 agent 顺序执行。新 B1 独立重建规格、实现与自审见 `continuation_design_spec.md`、`continuation_implementation_review.md`。原 S6 exact gate和缺失原件未被重建替代。动态状态见 `B1_reconstruction_20261016/scheduler_status.json`（尚不存在时不代表已启动）。此更新优先于下方旧轮次状态。

# 2026-10-15 execution evidence (updated 2026-10-16)

Overall **PARTIAL**. No new formal research training completed. User dataset root `/home/xzf/data`; continuation is primary-agent only, with earlier delegated work interrupted per the latest user instruction. The active goal is not complete.

Continuation: B1 full CPU engineering and analysis processing passed; actual eight-run training remains pending GPU admission. C2.1's eight fixed-theta numeric cases and primary-agent scientific analysis are complete, with zero training: [C2 numeric analysis](C2_numeric_oracle_20261016/numeric_analysis.md). This does not trigger C2 training or close historical provenance gaps. The earlier history-audit `build_reports.py` does not contain these continuation sections; preserve current reports when regenerating history tables.

- [Execution report](execution_report.md): task matrix, environment, fixes, actual runs, recovery commands and remaining requirements.
- [Results](experiment_results.md): per-seed/cell AUC and loss, bootstrap, selected/grid boundary, figure.
- [Mechanism analysis](mechanism_analysis.md): scoped evidence, contradictions, size/difficulty, competing explanations and falsification.
- `execution_status.json`: machine-readable package states; `publication_status.json`: actual Git delivery status.
- `artifact_inventory.json`, `missing_artifacts.json`, `artifact_recovery_search.json`: hashed accessible inputs and explicit missing provenance.
- `dataset_preflight_*.json`, `dataset_prediction_binding.json`, `environment.json`: real data paths, hashes, split/label/schema and environment.
- [Callgraph](runner_objective_optimizer_callgraph.md), [audit specification](audit_design.md), [self-review](implementation_review.md).
- `audit_evidence.py`, `dataset_preflight.py`, `analyze_tables.py`, `bootstrap_ao04.py`, `make_figures.py`, `build_reports.py`, `run_data_tests.py`: reproducible evidence processing. Source input files retain their existing repository paths; full clone plus `/home/xzf/data` required to rerun.

Input snapshots are indexed, not guaranteed immutable by their local path. New evidence outputs are versioned, datasets/weights not duplicated. `output_checksums.json` verifies the deliverable snapshot except publication/checksum metadata. Publication identity is determined from Git commit and remote branch, not from an embedded self-referential commit hash.
