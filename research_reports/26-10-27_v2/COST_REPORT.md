# 成本账与参数汇总

状态：`COMPLETE_COSTED_DEVELOPMENT_MATRIX`；快照：2026-10-27T10:16:02.180190+00:00。本报告整理现有记录，不启动训练或追加实验。

修订矩阵目录 `artifacts/mdmt_relation_basis_v2/revision1`。ML采用R1来源 `72f3e825dac422784fd575bd24450e7858e99ce138f6c8261b3ae7636b8b13fc`，KR采用R2来源 `26708434c81106dd424d52b4de258bee8aad7b35640b1d50b8699290694a2a3c`；两版训练源码身份均为 `29d7018f037c06af567d58646c6f81d022cab101a0e87c86c99af295e751cd98`。已收集 48/48 个完整 run 成本收据、8/8 个共同前缀、960/960 条适配 epoch 数据。每个已收集 run 的 epoch 恰为 1–20 各一次；未完成项逐条列于 COST_LEDGER.json。READY 为 8/8，历史 E 导出完成 8/8。

## 修订矩阵的实际计时

| 数据 | 完整 run | 唯一前缀 | 前缀秒（只计一次） | 适配秒 | 功能诊断秒 |
|---|---:|---:|---:|---:|---:|
| movielens | 24/24 | 4/4 | 277.348 | 1725.734 | 41.632 |
| kuairand | 24/24 | 4/4 | 846.991 | 1468.732 | 33.819 |

已记录前缀与适配合计 4318.804 秒；加单独功能诊断为 4394.255 秒。这是各进程计时的组件和，GPU 并发下不等于整个矩阵经过的日历时间。

前缀含 native24、source12、T0 标准化/教师检查和数据缓存。适配计时含每轮 V 评分、选择、日志和 RAM 状态捕获；不含末尾 V/Q 预测导出、功能诊断与 latency 测量。每方法 `single_method_with_prefix_seconds_view_only` 可展示单方法依赖成本，但整个六方法矩阵只计该前缀一次。

参数、所用物理GPU、逻辑设备、selected/final epoch、global steps、峰值显存、每批来源调用、MAC范围及 raw-X/cached 的 batch1/256 P50/P95 见 RUN_PARAMETERS_AND_COSTS.csv。relation 与 residual_function 参数是 adaptation 总数的子集；audit-only teacher 参数不算总部署参数。

Latency 每种路径 50 次 warmup、200 次同步测量。原始收据只有 P50/P95，无独立 latency 总秒；不以分位数反算算法总时间。MAC 是 nn.Linear 部分清单，原收据已明确排除 embedding、LayerNorm、activation、STAR 矩阵乘及部分显式 einsum；显式残差与低秩关系项另列。所有方法 `sparse_execution=false`。

从每run diagnostics.json 写出至 costs.json 写出的文件mtime间隔，累计观察到 1173.011 秒；此间隔含部署benchmark、MAC计数与成本收据生成，列于参数成本表。它是文件时间戳观测，不作为精确纯算法计时。

## 原失败/中止 attempt 与重放

| attempt | 状态 | 精确进程秒 | 停止前ps快照秒 | 已完成组件收据秒 | 最终落盘epoch数 |
|---|---|---:|---:|---:|---|
| revision0 ML/2022 | FAILED | 67.83427999300147 | — | 59.764 | native:24, source:12, B_REFERENCE:20, DIRECT:20, READ:20 |
| revision0 ML/2023 | ABORTED | 未记录 | 154 | 155.206 | native:24, source:12, B_REFERENCE:20 |
| revision0 ML/2024 | ABORTED | 未记录 | 115 | 94.183 | native:24, source:12, B_REFERENCE:11 |
| revision0 ML/2025 | ABORTED | 未记录 | 16 | 22.783 | native:24 |

精确进程秒、组件收据秒与 epoch 秒彼此重叠，不能相加。ps 快照是停止前观测而非最终耗时；例如 seed2025 的最后持久化 native 日志已达24轮，停止前快照仍是15轮/16秒。原四 attempt 全部保留，不纳入修订矩阵48条独立训练结果。

用失败 worker 的精确总秒、其余中止 worker 的不重叠组件收据与尚未覆盖的完整 epoch 计时，原四 attempt 已记录投入至少 370.321 秒；每项的计算边界均在账本，未计时的中止开销没有填0。

| 诊断重放 | 状态 | native/source/READ组件和秒 | 整体诊断秒 |
|---|---|---:|---:|
| seed2022 | DIAGNOSTIC_OUTPUT_SERIALIZATION_FAILED | 39.781 | 未记录 |
| seed2022_v2 | REPRODUCED | 39.256 | 40.94693444400036 |

两次重放均使用原 ML seed2022 身份，分别因输出序列化失败和完成定位而记录。组件和与整体诊断秒重叠，不相加。另有诊断 import 失败和 revision1 detached 启动失败，两者日志/初始化目录保留但没有最终总计时；不填0。

## KR R1失败归档与R2重跑

| attempt | 精确worker总秒 | 共同前缀秒（已含于总秒） | native轮数 | source轮数 | 完成适配 |
|---|---:|---:|---:|---:|---|
| KR/2022 R1 | 255.443537 | 185.231807 | 24 | 12 | B_REFERENCE:20 |
| KR/2023 R1 | 497.100701 | 327.373921 | 24 | 12 | B_REFERENCE:20 |
| KR/2024 R1 | 242.213810 | 223.113301 | 24 | 12 | B_REFERENCE:20 |
| KR/2025 R1 | 240.327865 | 224.357024 | 24 | 12 | B_REFERENCE:20 |

四份失败worker及其B_REFERENCE20轮、源码身份、FAILED/epoch/预测/stdout均由归档索引保留在 revision1/attempts/KR_seedNNNN_R1_failed。其精确进程总秒累计 1235.085913，包含前缀、B_REFERENCE训练、导出与失败parity流程；不能再叠加表内组件秒。R2原位重跑是修订矩阵的实际KR训练，归档失败轨迹不算额外seed，也不复用为成功前缀。

合并原4attempt、两次诊断重放与KR R1四失败，已记录重复投入至少 1686.135 秒；缺最终计时的启动/import失败另列而不填0。此成本与成功矩阵组件和分开。

## 等待、导出与存储

每 worker 的 runner 启动、READY、E释放、完成时点及文件 mtime 观测保存在 COST_LEDGER.json。READY 后 RAM 驻留等待不属于算法训练时长；COMPLETE 的 worker wall 含这段等待，不可直接作训练成本。账本另提供 worker wall减READY→release观测等待的近似active wall，仍含导出、benchmark、E与缓存等流程。最终 V/Q 导出、数据准备/加载和 E 评分缺少独立组件秒时，记录为未分离，不能用漏记等同0。

| worker | E释放(+08:00) | COMPLETE(+08:00) | worker总秒（含等待） | READY→释放观察秒 | 减等待近似active秒 | 释放→完成观察秒 |
|---|---|---|---:|---:|---:|---:|
| movielens/2022 | 2026-10-27T18:11:43+08:00 | 2026-10-27T18:11:51.356108+08:00 | 2947.528 | 2718.426 | 229.102 | 8.356 |
| movielens/2023 | 2026-10-27T18:12:01+08:00 | 2026-10-27T18:12:34.672155+08:00 | 2967.183 | 1617.405 | 1349.778 | 33.672 |
| movielens/2024 | 2026-10-27T18:11:44+08:00 | 2026-10-27T18:11:52.152184+08:00 | 2878.691 | 2692.480 | 186.211 | 8.152 |
| movielens/2025 | 2026-10-27T18:12:02+08:00 | 2026-10-27T18:12:35.824260+08:00 | 2882.496 | 1515.940 | 1366.556 | 33.824 |
| kuairand/2022 | 2026-10-27T18:10:52+08:00 | 2026-10-27T18:11:16.748776+08:00 | 1042.811 | 299.499 | 743.312 | 24.749 |
| kuairand/2023 | 2026-10-27T18:10:53+08:00 | 2026-10-27T18:11:21.221212+08:00 | 1028.355 | 289.227 | 739.129 | 28.221 |
| kuairand/2024 | 2026-10-27T18:10:54+08:00 | 2026-10-27T18:11:23.133398+08:00 | 918.195 | 241.092 | 677.103 | 29.133 |
| kuairand/2025 | 2026-10-27T18:10:55+08:00 | 2026-10-27T18:11:22.505337+08:00 | 899.576 | 239.360 | 660.216 | 27.505 |

八worker完整进程计时合计 15564.834 秒，其中READY→释放观察到 9613.428 秒；相减的近似active wall为 5951.406 秒。释放→完成观察间隔合计 193.613 秒，含轮询/排队、E缓存、评分与导出。以上口径彼此重叠，不能再与训练组件和相加；E释放/完成以 E_EVALUATION_LEDGER.json 的实际记录优先，READY以runner记录和READY文件mtime为据。

E账本状态 `COMPLETE`：8 worker完成、96 份selected/final指标、96 份预测文件、576 个域任务格。48run保持ML24份R1、KR24份R2；37份共同训练源码字节等价继续使用已独立审查的原证据，没有追加模型运行。

当前数据/预测 .npy/.npz 逻辑大小 32,626,285,887 字节，按 inode 去重的唯一文件大小 32,626,285,887 字节；逐文件大小与mtime在账本。RAM 模型、optimizer 和 checkpoint 未写磁盘。

汇总命令：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/26-10-27_v2/aggregate_costs.py`。完整计时边界见账本 measurement_scopes，原始证据路径随每项保留。
