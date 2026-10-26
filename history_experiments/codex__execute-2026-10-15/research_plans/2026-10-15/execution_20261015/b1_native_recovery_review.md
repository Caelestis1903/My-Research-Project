# B1 首attempt原生中止：诊断、准入校准与有界重试

主agent设计/实现/SELF_REVIEW，2026-10-16；未派发新子agent。

## 已确认的失败与范围

原调度器PID953415在GPU1 used=15/24576 MiB时启动seed2022/F，PID975570约30秒后exit=-6。唯一输出是`munmap_chunk(): invalid pointer`；没有failure.json、初始化identity、epoch状态或指标。源调用顺序先完成constructor才进入fit，因此没有可计为正式结果的epoch。原调度器已实际终止并进入NEEDS_FAILURE_REVIEW；这次恢复依据真实退出事件与缺失进程，不依据观察超时。

原native abort根因仍是UNRESOLVED_NOT_REPRODUCED，不能写成OOM、科研阴性或已修复的模型bug。原日志、terminal状态与事件须归档保留，不覆盖。

## 实际诊断

在原GPU1/原环境独立完成CUDA Adam clone oracle→真实KuaiRand loader，随后完整原CUDA constructor；两者PASS，actual init hash仍为9cf5f785…。这只能说明这些路径的独立运行通过，不能抹除首次native fault。

新增`b1_runtime_profile_20261016.py`采用原40960 batch、DH原控制更新，执行2次main+1次arch，无正式fit/评估。每0.5秒采样单卡总显存，超90%仅终止本诊断进程。原模型/优化器更新公式均复用冻结实现。实测peak reserved=2145386496 bytes（2046 MiB），allocated=2023244288；基线2911 MiB时总显存观测峰值5277 MiB。`cuda_runtime_profile_receipt.json`保存source绑定、真实batch及操作计数。

## 最小恢复实现

- 冻结科学算法`b1_reconstruction_20261016.py`保持原SHA1552aa5aec80c4b1118ee45928ac9a566de50878c1a5f84b2b4199878353fb7f；原CPU full-pipeline与梯度验证仍适用，不改变batch、模型、目标、seed、epoch或梯度控制。
- 新增`b1_reconstruction_attempt_20261016.py`只负责真实attempt id、phase日志、faulthandler与提交/source元数据。constructor/fit/write仍调用同一冻结DenseRunner。普通异常保留failure.json；native中止保留stderr堆栈，由调度器记录真实exit。
- 调度器在源码/完整batch profile匹配时以peak_reserved×1.25+768 MiB准入（本次3325.5 MiB），每卡独立、最多5进程；之后按实际epoch峰值取更保守值。原20GB只在无实测profile时使用。校准不修改科学定义。
- supervisor必须确认frozen worker、launcher和scheduler均与记录Git commit bytes一致。failed attempt1归档到`failed_attempts/seed2022_F/attempt1`，新的F为attempt2，其他seed/arm仍为attempt1。普通失败仍先记录/诊断，禁止覆盖或自动无限重试。

复核结论：允许在保留失败现场、语法检查、提交实际wrapper/scheduler且确认旧进程均终止后，执行一次记录明确的attempt2及其余独立任务。独立CPU/GPU诊断通过与有界重试不构成原生根因已解决的证据。若再发生native中止，读取新增phase/堆栈并诊断；有效任务输出保留，不能为指标改变协议。
