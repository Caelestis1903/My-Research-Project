# 06｜Scheduler & Resources

## 1. 有限队列

固定阶段，不设置无限循环：
1. S0 source/model/resource audit。
2. S0 implementation + CPU tests。
3. Independent implementation review。
4. S0 real GPU/RAM smoke。
5. S1 10 baseline trajectories + diagnostics.
6. S1 independent analysis/review and decision.
7. If GO: S2 30 new trajectories at most (J_ACC/P→U/U→P × 2 datasets × 5 seeds).
8. S2 independent review and freeze.
9. If GO: S3 one-shot Test and final report.
10. Close all child agents and guardian jobs.

任何未注册 arm 不得 dispatch。所有 job 使用 logical_run_id = SHA256(canonical config JSON)；attempt_id 不进入科学身份。恢复只允许同一逻辑 run 同源重放，不能挑选更好 attempt。

## 2. 环境

执行时重新核对：
- Python executable 与 version；
- PyTorch/CUDA/cuDNN；
- GPU 型号、显存、功耗、当前利用率；
- host RAM、CPU；
- 数据路径和 SHA256；
- 当前仓库 commit 与 dirty status。

历史路径和版本只是审计线索，不代表本次执行机器已经具备这些条件。

## 3. GPU/RAM 准入

沿用项目现行逐卡动态准入：
- 每卡当前显存使用率 <85%；
- 每卡当前功耗 <85%；
- host RAM <90%；
- 启动后重新测量，检查完整进程实际驻留和 CUDA usable capacity。
不增加固定并发数上限，不以预计功耗替代实测，不因资源紧张静默缩小 batch、改精度或改变训练人口。

资源不满足时 PAUSED_RESOURCE；不杀外部作业。

## 4. RAM-only 状态

需要保留：
- S1 每 dataset×seed 的 E10/E25/E40 anchors；
- online、EMA、完整 Adam state；
- RNG、loader、row cursor；
- S1 每个 fork 的起始/结束标量和状态 hashes；
- S2 所有训练 arm 的 selected 状态及恢复状态。

每个状态保存前后都做 get/hash readback；父进程独立验收模型/optimizer/EMA identity。状态丢失时：
- 记录失败 attempt 与成本；
- 仅按原配置重放；
- 不增加 seed 数；
- 不以重跑后更好结果覆盖原结果；
- 缺失完整状态不能只凭旧收据声称可恢复。

## 5. 成本

每条 run 记录：
- worker wall time、实际 GPU time；
- forward/backward 数、optimizer step 数；
- peak allocated/reserved memory；
- 每 epoch 秒数；
- validation 时间；
- fork 数、虚拟/真实更新数；
- RAM 状态字节数；
- 失败/暂停/恢复成本。

并行 worker 时间不可相加称为独占 GPU 时长。机制诊断的虚拟步骤和正式训练的 optimizer 更新分开记账。

## 6. 禁止资源/调度行为

- 不启动独立后台 Codex 会话。
- 不恢复已完成子 agent。
- 不使用旧 driver/旧 runner 的未审查入口。
- 不因 GPU 空闲而自动扩大矩阵。
- 不在终态后启动新阶段。
- 不把 OOM、环境不兼容或 RAM owner 丢失写成科学阴性。
