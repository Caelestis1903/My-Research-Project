# 原始标量、来源与复建

归档包含3900个文件：本轮30条训练的1600个epoch记录、batch日志、配置/数据/源码锁、选点与V/Test端点、成本和调度日志，以及实际使用的历史复用输入。每个发布端点的source文件均核对原checksum。文件路径相对HISTORY_ROOT；数据数组缓存与guardian认证文件不在发布集合中。

压缩包为9771576字节，GitHub按2MiB保存为`raw_training_scalars.tar.gz.part000`至`.part004`。完整文件列表、每文件SHA256及各块校验见[RAW_ARTIFACTS_MANIFEST.json](RAW_ARTIFACTS_MANIFEST.json)。拼接后的SHA256：

```text
8b136d2b2cb70ba4f61f8afd0dbbee098363078c20e95342d034d85f49a9c66e
```

从已检出的仓库根目录拼接并核对：

```sh
cat research_reports/2026-11-03_overnight_AB/raw_training_scalars.tar.gz.part* > research_reports/2026-11-03_overnight_AB/raw_training_scalars.tar.gz
(cd research_reports/2026-11-03_overnight_AB && sha256sum -c raw_training_scalars.sha256)
```

恢复标量到独立历史目录：

```sh
REPLAY_HISTORY_ROOT=/path/to/scalar_history
mkdir -p "$REPLAY_HISTORY_ROOT"
tar -xzf research_reports/2026-11-03_overnight_AB/raw_training_scalars.tar.gz -C "$REPLAY_HISTORY_ROOT"
```

直接复建分析图表和阶段差表：

```sh
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-03_overnight_AB/analyze.py > research_reports/2026-11-03_overnight_AB/ANALYSIS_VALUES.json
```

该命令读取已发布ENDPOINTS.csv，要求全部比较具备2022–2026五seed，输出两张图、STAGE_EFFECTS.csv与完整派生数值。原始配置和全部CLI启动契约见[00执行入口](../../research_plans/26-11-03_overnight_AB/00_EXECUTION_RUNBOOK.md)。本次prepare使用HISTORY_ROOT=/home/xzf/Code/phase0-3，执行树=/home/xzf/Code/phase0-3_overnight_ab；实际环境和家族源码锁随归档保留。归档中的PREPARED记录实际运行路径。

## 固定上游代码

模型adapter已随新包交付；其实际外部依赖及blob/SHA256见[vendor清单](../../experiments/overnight_ab_20261103/vendor/sota2526/SOURCE_MANIFEST.json)。上游Git根及固定版本：

| 来源 | 仓库 | 固定commit | 本次目录 |
|---|---|---|---|
| HyperGate parent | https://github.com/XuHwang/HyperGate | fdc82e3f3feee23694a8123f91092615f977900e | /tmp/mdmt_sota_sources_2526/HyperGate |
| RecStudio | https://github.com/XuHwang/RecStudio | f08cca607e998e8be4c5e915a393346baffa71f6 | /tmp/mdmt_sota_sources_2526/RecStudio |
| AutoIFS | https://github.com/ChaohuaYang/AutoIFS | 65fc27cdfad1eb6d7a4a2cc69cd31f61cc8e3d7e | /tmp/mdmt_sota_sources_2526/AutoIFS-pinned |

自然CSV来源保持AGENTS.md中的迁移路径；raw SHA与预处理/行号/六数组身份在归档DATA_MANIFEST中。原始M3/history/pair依赖使用报告登记的冻结源码，发布合并保留已有补评报告。
