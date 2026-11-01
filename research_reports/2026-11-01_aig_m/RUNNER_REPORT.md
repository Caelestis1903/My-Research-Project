# AIGM01 实验执行记录

## 执行身份与配置

- 执行代码 commit：`2cec25ab5b32411855a13ab3143678a5e980ea3f`；模型、优化器和配置身份见 `selection_frozen.json`、`formal_configuration_frozen.json`、`source_manifest.json`。
- 完整运行目录：`artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720`；Stage0 目录：`artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720_check`。
- 环境：Python 3.13.5、PyTorch 2.9.0+cu126、CUDA runtime 12.6、两张 RTX 3090。全程由单一 RAM coordinator session 完成，结束码0；`state_release.json`记录 `released=true`、`storage=RAM_ONLY`，释放发生在统一test及数表完成后。
- Stage0使用CUDA `cuda:1`；正式all使用设备`0,1`，batch size 40960。开发阶段使用seed 2022、2024；S5正式阶段覆盖seed 2022–2026，允许RAM缓存复用。训练epoch为50。

## 各阶段执行状态与机械门记录

| 阶段 | 完成/计划 | 失败 | 数据集阶段记录 |
|---|---:|---:|---|
| Stage0 | 两数据集各19次真实更新 | 0 | PASS；细节与数值见 [STAGE0_REPORT.md](STAGE0_REPORT.md) 及Stage0收据 |
| S1 | 16/16 | 0 | MovieLens选C/H2，selected validation macro AUC两seed均值0.7598499757933823；KuaiRand选C/ema995，均值0.6943158354968332 |
| S2 | 12/12 | 0 | MovieLens选G1/H2，均值0.7597041309889003，`promoted=false`；KuaiRand选G2/ema995，均值0.693205794223638，`promoted=false` |
| S3 | 8/8 | 0 | MovieLens选S95/H2，均值0.7596503161575956，`promoted=false`；KuaiRand选S99/ema995，均值0.6934750069446153，`promoted=false` |
| S4 | 8/8 | 0 | MovieLens及KuaiRand均为`selected=null`、`promoted=false` |
| S5 | 12个新训练run完成；另有8次RAM缓存复用 | 0 | 最终逐数据集门和状态见下表及`decision.json` |

上述selected均值与门字段照录阶段JSON。各阶段50 epoch训练明细、逐run配置和成本见`trial_ledger.csv`、`costs.csv`及完整attempt归档。

S5最终机械状态（`decision.json`）：

| 数据集 | 臂 | status | development gate | test gate |
|---|---|---|---:|---:|
| MovieLens | EMA | GO | true | true |
| MovieLens | augmentation | NO_PROMOTION | false | false |
| MovieLens | AIG | STOP_DEVELOPMENT | — | — |
| KuaiRand | EMA | GO | true | true |
| KuaiRand | augmentation | TRADEOFF | false | false |
| KuaiRand | AIG | STOP_DEVELOPMENT | — | — |

## 汇总指标与原始记录

完成收据`completion.json`记录：开发独立run 44个；独立训练run总计56个；不使用缓存的名义数量64；缓存复用8；训练总计2800 epoch、96600 optimizer steps；正式输出30个逻辑model-seed记录；统一test已执行；wall time 8504.296677483013秒。收据还记录RAM snapshot tensor合计9576073648 bytes。

复制到报告目录的顶层数值CSV/JSON位于[`raw_attempt_tables`](raw_attempt_tables/)（含完整及Stage0两个attempt的顶层CSV/JSON）。其中：

- `test_metrics.csv`保留每正式输出的每域×任务AUC、logloss、样本数和正例数，并含macro、domain、task及pooled汇总字段。
- `selected_metrics.csv`保留selected/E10/E50的validation及test macro AUC、macro logloss。
- `validation_metrics.csv`、`validation_epoch_metrics.csv`、`epoch_metrics.csv`、`gate_stats.csv`、`gate_epoch_summary.csv`位于raw_attempt_tables中的完整attempt子目录；`view_stats.csv`、`batch_trace.csv`位于压缩包内各个run目录。
- `paired_effects.csv`、`effect_summary.csv`、`arm_summary.csv`及各阶段`*_decision.json`记录逐域/任务配对原值、汇总值和门状态。

## 日志、错误与缺失项

- 完整attempt压缩包：[raw_attempts_2026-11-01_aig_m.tar.gz](raw_attempts_2026-11-01_aig_m.tar.gz)。包内包含Stage0与all attempt原始CSV/JSON、stdout/stderr、batch/view/split行序数值日志；未包含模型权重或checkpoint文件。
- 文件清单及逐文件SHA-256：[raw_attempts_2026-11-01_aig_m_SHA256SUMS.txt](raw_attempts_2026-11-01_aig_m_SHA256SUMS.txt)；压缩包SHA-256见[sha256 sidecar](raw_attempts_2026-11-01_aig_m.tar.gz.sha256)。
- `failure_ledger.csv`存在且仅含schema header `stage,dataset,trial,seed,device,attempt,oom,error`；失败数0。没有缺失的授权run。
- 所有数值记录及日志均保留在原attempt目录。
