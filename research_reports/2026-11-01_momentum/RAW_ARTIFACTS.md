# 原始训练工件

[RAW_RESULTS.tar.gz](RAW_RESULTS.tar.gz) 收录正式 `attempt_20261101_1340` 的全部578个原始文件，包括40run的stdout/stderr、batch/view/epoch/cell指标、模型身份哈希、运行统计、来源/数据清单和split行号。

- 文件大小：14707814字节。
- SHA256：`8847164d9585ec11b8a95eb1a4f7a6c8e152acfa9882481c857cc046b1e8634d`。
- 逐文件大小和SHA256：[raw_artifact_index.json](raw_artifact_index.json)。
- 数据集原始CSV、模型权重及checkpoint文件不在包中；模型状态按任务要求仅保留RAM，已释放。

在仓库根目录恢复工件：

```bash
mkdir -p artifacts/gradient_aug_momentum_20261101
tar -xzf research_reports/2026-11-01_momentum/RAW_RESULTS.tar.gz -C artifacts/gradient_aug_momentum_20261101
```

然后使用[README](README.md)中的分析/绘图命令复算。报告目录同时保留可直接读取的完整CSV副本。Stage0真实CUDA收据是[stage0_checks.json](stage0_checks.json)，CPU公式、语义和恢复收据分别单列。
