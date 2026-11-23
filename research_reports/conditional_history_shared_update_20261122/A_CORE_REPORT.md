# A 核心归因结果

在同一自然样本对目标下，24 条件历史相对普通 Adam 的 Validation Macro AUC 平均增量在 MovieLens 和 KuaiRand 分别为 +0.000161 和 +0.000172。正向种子分别为 4/5 和 2/5，两个配对区间均跨零。按预注册门进入 A21、A22 的五种子比较。

以训练种子为单位配对。所有行均为固定 EMA，各自取 Validation 六格 Macro AUC 首次严格最大值对应的 epoch，LogLoss 取同一 epoch。下表中 Δ 为左臂减右臂；AUC 越大越好，LogLoss 越小越好。

| 数据集 | 比较 | ΔAUC | 配对 95% 区间 | 正向种子 | ΔLogLoss | 最差格及 ΔAUC |
|---|---|---:|---|---:|---:|---|
| movielens | 固定 PAIR − clean | +0.000082 | [-0.000753, +0.000918] | 3/5 | -0.000492 | d1t1 -0.000429 |
| movielens | 条件历史 − 相同 PAIR 的 Adam | +0.000161 | [-0.000185, +0.000507] | 4/5 | -0.000806 | d2t0 -0.000174 |
| movielens | 动态 PAIR − 固定 PAIR | -0.000029 | [-0.000519, +0.000462] | 2/5 | -0.000137 | d2t0 -0.000410 |
| kuairand | 固定 PAIR − clean | +0.003067 | [-0.006549, +0.012682] | 3/5 | -0.001664 | d1t0 -0.001835 |
| kuairand | 条件历史 − 相同 PAIR 的 Adam | +0.000172 | [-0.003135, +0.003479] | 2/5 | +0.001085 | d0t0 -0.003979 |
| kuairand | 动态 PAIR − 固定 PAIR | -0.000401 | [-0.006309, +0.005506] | 2/5 | -0.000234 | d2t0 -0.003282 |

**WEAK · 条件历史的独立贡献。** MovieLens 上 4 个种子的排序改善，均值较小。KuaiRand 上仅 2 个种子排序改善，LogLoss 平均增加 0.001085，条件历史相对相同 PAIR 的普通 Adam 还使 d0t0 的平均 AUC 下降 0.003979。两数据集同向的宏均值尚不足以把差异归因于条件历史表。

**WEAK · 样本对监督和动态采样。** KuaiRand 上 A20 相对干净基线的宏增量为 +0.003238，其中固定样本对加普通 Adam 已达到 +0.003067。动态采样的宏均值在两数据集均低于固定采样，且其配对区间较宽。简单历史对照可区分监督目标、普通历史长度和条件索引带来的增量。

逐种子的绝对六格 AUC/LogLoss、任务/域汇总、选点 epoch、模型哈希和配对差见 [端点表](../../artifacts/conditional_history_shared_update_20261122/tables/A_CORE/endpoints.csv)、[逐种子配对](../../artifacts/conditional_history_shared_update_20261122/tables/A_CORE/paired_seed.csv) 与 [统计汇总](../../artifacts/conditional_history_shared_update_20261122/tables/A_CORE/paired_summary.csv)。本节对应 12 条新训练和 28 条历史训练的 Validation 记录。
