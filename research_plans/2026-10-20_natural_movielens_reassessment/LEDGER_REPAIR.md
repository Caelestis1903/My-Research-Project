# V9训练/只读任务的台账更正

## 问题及影响范围

2026-10-20 09:52 UTC主代理导出快照时发现，8个V9只读评价与原训练均使用`<run_id>_attempt01`；机械协调器旧`save_row`仅按attempt_id替换，覆盖了6条完整训练attempt和2条失败attempt。两个成功attempt02仍在。因而仅按台账汇总误报core114/120；原run manifest、模型、预测和采样/预算凭据没有被改写。已完成的预测级性能复算不受影响。

错误快照未提交/未准入，服务器`primary_analysis_ledger_audit/`保留原导出及INVALID标记。报告目录先恢复最后有效已提交快照，恢复台账后才重新导出。

## 恢复依据与处理

机械执行代理在N17本批64配置完整闭环、parent PID153678无child的安全阶段边界停止该parent，未中断训练进程。根据原coordinator START/EXIT事件与独立producer凭据恢复8行：

- 6个完整attempt01：50epoch、1000main steps、40,008,350曝光；OS退出码保留UNKNOWN_MONITOR_LOST。
- 2个零epoch失败attempt01：原exitcode=1，后续成功attempt02保留。
- SN2022原退出事件缺失，由原receipt/progress/冻结模型hash确定完整训练；completed_utc未知留空，不由mtime或wall_seconds推算。
- 原8个diagnostic只读记录全部保留。
- 两个既有attempt02的examples原为旧monitor占位值0；仅在原receipt确认完整曝光后更正为40,008,350，并绑定真实receipt路径。

修复前后台账SHA及逐attempt来源/状态/缺失字段均在[恢复映射](analysis/ledger_repair/v9_ledger_attempt_recovery_20261020T101836Z.json)及[曝光量更正](analysis/ledger_repair/v9_attempt02_examples_correction_20261020T102210Z.json)。[原V9事件](analysis/ledger_repair/v9_original_coordinator_events.json)保留原始行号和完整源文件SHA；修复前台账已确定性压缩归档，源路径及原bytes SHA见[索引](analysis/ledger_repair/archive_index.json)。恢复不冒充找回了不存在的failed attempt模型bytes。

## 防止覆盖及当前状态

机械保存键改为`(execution_kind, attempt_id)`，后续diagnostic/control/test增加类型namespace；旧历史ID保留。改动仅为调度/记账，研究producer、网络、优化器、训练预算及选点规则不变。[修复后协调器源码](analysis/ledger_repair/published_coordinator_after_ledger_fix.py)与[一次性恢复脚本](analysis/ledger_repair/repair_v9_ledger_attempts.py)均归档。新唯一parent PID298126复核既有receipt后继续队列，不以重训修补证据。

10:20 UTC重导快照：core120/120，V2 16/16，N17 80/80（16pilot+64正式H100），共216逻辑配置完整。原8个V9训练对应10个实际训练attempt及8个独立只读任务；不能把同run_id的两个类型合为一行。后续H200条件配置另计，不将延长前缀当独立数据/seed。

当前共8个零epoch失败attempt：V2三个输出目录冲突、V9两个共享缓存读取异常、N17三个共享split临时文件竞争。全部保留，成功同seed完整重跑已闭环；见[完整失败原字段](analysis/ledger_repair/zero_epoch_failed_attempts.csv)。失败启动不当科研阴性，也不增加四seed比较样本量。

## H200启动显存profile的另一个计数边界

H200复制H100目录，初始progress已有100轮/2000步；新的协调器把该复制值误当本阶段首个完整epoch，记录GPU baseline=peak 34/34与15/15 MiB、delta下限256 MiB。该值已标[INVALID_COPIED_H100_COUNTERS](analysis/ledger_repair/H200_profile_audit_20261020T1026Z.json)，不当作真实训练显存测量。当前8任务5/3槽位，采用同架构H100已测1318/1320 MiB的保守参照核算，等待真实≥101轮测量；后续N17_DIAGNOSTIC是独立profile组。没有因误记profile修改研究producer、停止child或把复制计数当新训练完成。启动I/O状态[快照](analysis/ledger_repair/H200_io_wait_snapshot_20261020T1028Z.json)单列，不当模型效果。

## N17诊断共享split准备的机械互斥（11:12安全边界）

N3_SN_seed2022_E040_bank0 attempt01在准备阶段因共享 `movielens_seed2022.json.tmp` 的并发replace失败，exit=1，失败attempt保留；attempt02已完整。这里的2022是固定人口split seed，不随优化seed改变。修复只在runner入口以fcntl独占锁包住原 `n17_bootstrap.prepare()`，函数返回即释放；未改变原采样、模型、优化器、draw、预算或研究producer文件。后续N17测试也使用相同入口。

[wrapper原件](analysis/n17_runner_repair/n17_prepare_lock_runner.py)、[操作审计](analysis/n17_runner_repair/N17_prepare_lock_operational_audit_20261020.json)保留锁路径、SHA和四个producer源文件与冻结manifest的逐项一致性。主代理阅读wrapper后确认只增加准备互斥；机械worker做了语法、命令适配和help导入检查，没有因此新增研究实验。

只暂停旧parent PID298126，不向健康child发信号；全部10个既有child完成后成为Z状态，再替换parent。10份原receipt全部有效、test_read=false；独立子进程退出码无法从新监控取得，台账写UNKNOWN_MONITOR_LOST、完成时间留空，不虚构0或把观察时间写成真实完成时间。[安全边界原件](analysis/n17_runner_repair/N17_diagnostic_race_drain_20261020.json)、[全部恢复记录与压缩前台账来源SHA](analysis/n17_runner_repair/source_index.json)。边界时48项基础诊断中12项完成、36未启动，1次失败后已成功重试；不是研究阴性或48项全部完成。
