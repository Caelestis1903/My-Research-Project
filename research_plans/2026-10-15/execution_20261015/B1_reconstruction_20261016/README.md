# B1 独立重建测量

本目录不替代历史 S6 原始工件。正式矩阵为 KuaiRand/natural/M3OE/Sample，F/DH×2022–2025，50 epochs；实际状态以 `scheduler_status.json` 和各 run 的 `progress.json` / 真实进程为准。工程检查不计正式 run。

用户最新要求严禁子 agent，后续均由主 agent执行，review是SELF_REVIEW。设计与复核位于上一级 `continuation_design_spec.md`、`continuation_implementation_review.md`。原计划的exact历史identity与独立复核要求未满足，不冒充完成。

启动前必须让 `engineering_verification.json` 的PASS与当前worker SHA256一致；源码变化会使结果失效，需重新检查。完整状态包括main/private/arch Adam、RNG、模型温度属性和架构迭代器重放信息，`resume_verified=false`明确保留恢复验证边界。

运行命令（从项目根目录）：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home/xzf/miniconda3/envs/phase0-3-env/bin/python experiments/b1_reconstruction_20261016.py verify
/home/xzf/miniconda3/envs/phase0-3-env/bin/python -u experiments/b1_reconstruction_scheduler_20261016.py
```

调度器应持久化运行。检查当前PID与子进程后才能决定是否重启；仅有状态文件/锁文件不足以证明进程仍活着。原attempt不会自动覆盖，遇到失败先诊断，保留证据。

每卡独立准入；单卡总显存≤90%，每卡最多5个本study进程。首次用20,000 MiB保守预留，取得完整epoch实测峰值后使用峰值×1.15+512 MiB。即使另一卡超过90%，合格卡仍可启动。外部用户任务不终止、不修改。

`runs/seed*/{F,DH}/dense/epoch*.pt`为完整epoch状态；`dense/validation_epoch*.npz`、`test_epoch*.npz`保留逐行预测；选择先在`selection_frozen.json`固定，再评test。训练原始日志/weights不推到Git。推送关键表格、状态、报告及复现代码；本地保留完整状态供重分析。

正式完成后主agent应逐项审计8/8配对、50epoch state hash、两split预测、R0–R4真实选点、same-epoch与selected时序分解、domain/task AUC与LL、seed变异与配对bootstrap、时间依赖限制。根据原规划判定后续条件，完成三份报告和关键数据推送后才可关闭总目标。

完整CPU工程流程已验证（`full_pipeline_verification.json`），其小子集原始输出在本地 `engineering_full_pipeline/`，均为非正式数据。分析程序及规格见 `experiments/b1_analyze_20261016.py` 和上一级 `b1_analysis_spec.md`；正式矩阵完成后执行：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home/xzf/miniconda3/envs/phase0-3-env/bin/python experiments/b1_analyze_20261016.py
```

该入口不接受缺失或仅部分完成的正式矩阵，缺少时不会生成正式报告。工程分析必须显式传 `--engineering-fixture <fixture_root>`，结果写入 `engineering_analysis/` 并带ENGINEERING_ONLY标签。Git只保存工程验证receipt和代码，不把工程预测/weights作为正式关键结果发布。完整科学分析仍需主agent依据真实正式结果完成。
