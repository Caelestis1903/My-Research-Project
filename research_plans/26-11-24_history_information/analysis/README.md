# 历史验证集交互项：独立重算

[reproduce_validation_interaction.py](reproduce_validation_interaction.py) 仅使用 Python 标准库，从固定的历史端点 CSV 重算 A10/A21 的验证集 AUC 配对差值。脚本只读取 `--source` 指定文件并写出两个结果 CSV；不联网、不启动训练、不导入仓库训练模块，也不读取人工评审文件。

## 固定来源与运行命令

| 项目 | 固定值 |
|---|---|
| 仓库 | `Caelestis1903/My-Research-Project` |
| 证据提交 | `358d3a465747eda6af1d55acc937759fa651a10b` |
| 原始文件 | [`artifacts/conditional_history_shared_update_20261122/tables/A_HISTORY/endpoints.csv`](https://github.com/Caelestis1903/My-Research-Project/blob/358d3a465747eda6af1d55acc937759fa651a10b/artifacts/conditional_history_shared_update_20261122/tables/A_HISTORY/endpoints.csv) |
| 原始 Git blob | `d420cdf0043f88b66a6bb886e0c73d72175023a8` |
| 数据行数 | 6,480，不含表头 |
| 运行环境 | 脚本兼容Python 3.8或以上、无第三方依赖；仓库执行使用指定phase0-3-env |

在已经包含该提交的本地仓库根目录执行。`git show` 直接导出固定提交的文件字节，不修改工作区，也不需要运行训练：

```bash
git show 358d3a465747eda6af1d55acc937759fa651a10b:artifacts/conditional_history_shared_update_20261122/tables/A_HISTORY/endpoints.csv > /tmp/history_information_endpoints.csv
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_plans/26-11-24_history_information/analysis/reproduce_validation_interaction.py \
  --source /tmp/history_information_endpoints.csv \
  --output-dir /tmp/history_information_reanalysis
```

默认先核对 `SHA1(b"blob " + str(len(raw_bytes)).encode("ascii") + b"\0" + raw_bytes)`，只有结果与固定 blob 一致才继续。重新序列化 CSV、改变换行符或删去 `config` 列都会改变字节，因此不能通过原始 blob 校验。输出文件已存在时程序拒绝覆盖；再次运行应指定新的输出目录。

## 重算口径

输入保留两数据集 `movielens`、`kuairand` 的 `validation`，两个 arm `A10`、`A21`，五个 seed `2022–2026`，三个 checkpoint `selected`、`E10`、`E50`，以及 `online`、`fixed_ema` 两种 readout。`selected` 使用历史表中已选定的端点，不重新选 epoch。原表中的其他 arm 和预计算汇总行不进入差值计算。

每个端点必须具备 `d0t0,d0t1,d1t0,d1t1,d2t0,d2t1` 六格，共 **720 条六格端点记录**。`macro` 由这六格 AUC 等权平均重新计算，不使用源表的 `macro`、traffic 或 pooled 行。每个数据集、checkpoint、cell 和 seed 都计算以下五个差值，正值表示该差值方向上的 AUC 增益：

| `contrast` | 定义 |
|---|---|
| `A21_minus_A10_online` | `A21_online − A10_online` |
| `A21_minus_A10_fixed_ema` | `A21_fixed_ema − A10_fixed_ema` |
| `EMA_gain_A10` | `A10_fixed_ema − A10_online` |
| `EMA_gain_A21` | `A21_fixed_ema − A21_online` |
| `interaction` | `EMA_gain_A21 − EMA_gain_A10`，等价于上述两个 A21−A10 差值的 EMA 读数减 online 读数 |

先在相同 seed 内做配对差值，再跨五个 seed 汇总。均值、样本标准差和区间分别为 `mean(delta)`、`stdev(delta)`（`ddof=1`）及 `mean ± 2.7764451051977987 × sd / sqrt(5)`，对应自由度 4 的双侧 95% t 区间。`positive_seeds` 为严格满足 `delta > 0` 的 seed 数；零不计为正值。三个 checkpoint 与各 cell 分开汇总。

## 输出和完整性检查

输出文件与 [`../evidence/`](../evidence/) 的对应表保持字段和行序一致：

| 输出文件 | 行数 | 字段 |
|---|---:|---|
| `validation_interaction_seed.csv` | 1,050 | `dataset,seed,split,checkpoint_rule,cell,contrast,delta_auc,source_commit,source_blob` |
| `validation_interaction_summary.csv` | 210 | `dataset,split,checkpoint_rule,cell,contrast,n,mean_delta_auc,sd_delta,ci95_low,ci95_high,positive_seeds,evidence_type` |

脚本检查全表端点键 `(dataset,arm,seed,checkpoint_rule,readout,split,cell)` 唯一；两数据集 × 两 arm × 五 seed × 三 checkpoint × 两 readout × 六格的配对矩阵必须完整。目标记录出现未知 seed、checkpoint、readout 或 cell 时直接报错，不会静默过滤。AUC、logloss 和计数必须有效；每格均须有正负样本，`positive + negative = support`，同一数据集与 cell 的支持数及正负样本数必须在所有目标端点间一致。所有检查完成后才写出结果。

## 本地机械核对与已完成验证

仅在使用已解析、去掉 `config` 列或重新序列化的副本检查计算过程时使用：

```bash
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_plans/26-11-24_history_information/analysis/reproduce_validation_interaction.py \
  --source /path/to/compact_endpoints.csv \
  --output-dir /tmp/history_information_mechanical_check \
  --skip-source-blob-check
```

该模式仍要求 6,480 行、完整配对矩阵与上述数值检查，但将汇总表的 `evidence_type` 明确写为 **`local_mechanical_check`**，并在终端报告 `source_authentication=SKIPPED` 及输入的实际 Git blob。seed 表中的 `source_commit/source_blob` 仅标识预期历史来源，不认证所提供副本的来源。默认通过原始字节核验的模式写为 `historical_reanalysis`。

已用本次工作中保存的 6,480 行紧凑 JSON 转成临时 CSV，完成本地机械核对。与现有两张 evidence 表逐行比较，行数、行序、字段、配对身份、`n=5` 和 `positive_seeds` 全部一致；`evidence_type` 按上述模式规则有意不同。六格重算宏平均后，各数值字段的最大绝对差如下：

| 字段 | 最大绝对差 |
|---|---:|
| `delta_auc` | `3.3306690738754696e-16` |
| `mean_delta_auc` | `1.1102230246251565e-16` |
| `sd_delta` | `1.903316913798303e-16` |
| `ci95_low` | `3.030345072096985e-16` |
| `ci95_high` | `1.6975895515691786e-16` |

差异均小于 `1e-14`，来自宏平均重算及浮点求和。已确认默认模式拒绝该重序列化副本，并确认重复键、缺少一格、缺少一个 arm 的整组 seed、支持数不一致均在输出前失败。现有 evidence 文件未修改。本次核对验证了计算复现；原始 CSV 的字节认证由默认模式在取得固定文件后执行。
