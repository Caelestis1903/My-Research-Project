# 新实验工件索引（阶段性）

服务器根路径 `/home_new/xzf_natural_movielens_reassessment`。独立分支 `codex/natural-movielens-reassessment-2026-10-20`。历史来源及缺失的大型旧工件另见HISTORICAL_EXPERIMENT_AUDIT；下列索引只描述本次新运行，不表示全部研究族已闭环。

## 已完成core与V2

- [completed_core_artifacts.csv.gz](analysis/artifact_index/completed_core_artifacts.csv.gz)：120 core与16 V2，共136个完整配置、5472个producer已记录文件；绝对路径、完成时SHA256、当前存在/大小及对应source manifest SHA。导出时0条路径缺失。
- [completed_core_manifests.jsonl.gz](analysis/artifact_index/completed_core_manifests.jsonl.gz)：上述136个原始run manifest内容快照，连同来源路径/SHA，含源代码/数据身份、配置、状态及预算。
- [completed_core_manifest_sources.json](analysis/artifact_index/completed_core_manifest_sources.json)：136个原始manifest路径及本次独立计算的SHA。
- [index_status.json](analysis/artifact_index/index_status.json)：机器可读范围与待补族。

导出器复用训练producer完成时的文件哈希，不重新读取全部重型张量；列名明确为producer_recorded_sha256。当前路径存在并非独立重算其全部bytes。预测级指标重算、来源manifest/metrics复核分别由各阶段分析执行，见analysis/core和analysis/v2，不把工件索引代替性能验证。压缩表使用确定性gzip mtime=0。

原始每个core run的模型/optimizer、所用epoch snapshot、selected/final测试预测、批次身份和状态遥测均位于 `runs/<run_id>/`。V2位于 `V2/<optimizer>/<arm>/seed<seed>/`。N17/AO04/N19/CM19训练及额外零更新评价/机制的索引将在相应闭环后追加；其当前运行状态以统一RUN_LEDGER为准。

## 既有证据入口

冻结数据、数组/row-ID身份：[DATA_PROTOCOL_AUDIT.md](DATA_PROTOCOL_AUDIT.md)与data_manifest。

历史672个ML配置：[HISTORICAL_ML_RUN_CONFIG_INDEX.csv](HISTORICAL_ML_RUN_CONFIG_INDEX.csv)，保留原配置SHA及新迁移配置字段，相关字段修正见HISTORICAL_INDEX_FIELD_CORRECTION.md。

历史研究问题与逐文件SHA：analysis/question_registry；各历史数值汇总的来源清单保留于对应analysis子目录。旧重型checkpoint/预测当前缺失的情况始终公开，不因新索引而称已恢复。

## 已完成V9全轨迹零更新评价

[只读工件表](analysis/artifact_index/v9_full_readout_artifacts.csv.gz)与[8份原receipt快照/SHA](analysis/artifact_index/v9_full_readout_receipts.json)，单列8任务、408工件（400预测及8指标表），每任务50模型评价、optimizer_updates=0，不增加独立训练数。完整预测重算2400cell及公共点一致性见ACV9_PERFORMANCE_ANALYSIS。模型原hash已在core及专门freeze中绑定。

## 台账恢复与机械调度证据

[LEDGER_REPAIR.md](LEDGER_REPAIR.md)及analysis/ledger_repair保存原V9事件行、修复前台账gzip、恢复映射/源码/SHA、两条旧监控曝光量修正，以及H200复制计数导致无效profile的审计。未知退出码不写成0；训练、只读和失败attempt分开。当前RUN_LEDGER仍由机械agent维护，最终完整交付前持续更新。
