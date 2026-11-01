# AIGM01 Stage0执行记录

- 执行代码commit：`2cec25ab5b32411855a13ab3143678a5e980ea3f`
- Stage0运行目录：`artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720_check`
- CUDA：RTX 3090×2；PyTorch 2.9.0+cu126，CUDA runtime 12.6；Python 3.13.5。
- 最终状态：PASS；`test_evaluated=false`，`disk_weight_files=false`，状态存储RAM_ONLY。
- MovieLens：batch 40960，19次真实更新，44个U组参数差分精确为零，四shadow非干扰、RAM恢复后下一步一致，最大显存1,262,873,600 bytes。
- KuaiRand：batch 40960，19次真实更新，44个U组参数差分精确为零，四shadow非干扰、RAM恢复后下一步一致，最大显存1,356,083,200 bytes。
- GAM01源检查在两数据集的预测/loss/梯度/Adam一步最大差均为0；四shadow与RAM续训检查PASS。来源manifest SHA256：`d297bea130cae8fedaeef47ae35239fd9af4f4d7f3076416eaf50670bcdae02c`，与传入执行commit匹配。
- 启动时GPU0已有外部PID826260，显存17656/24576 MiB、功耗229.6/350 W；GPU1为18 MiB、25.2 W。外部进程保持运行。
- Stage0 stdout确认两份数据identity PASS。仅有PyTorch旧TF32设置API弃用提示。

原始检查收据：`attempt_20261101_1720_check/stage0_checks.json`、`environment.json`、`source_manifest.json`、`data_manifest.json`。下一阶段完整运行目录为 `artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720`。
