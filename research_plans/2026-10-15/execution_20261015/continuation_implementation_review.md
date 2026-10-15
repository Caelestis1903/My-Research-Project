# 2026-10-16 主 agent 实现与复核记录

本记录为 SELF_REVIEW；不代表独立 REVIEW PASS。用户最新要求全程由主 agent 执行，已中止此前派发的 designer 和 reviewer；两者未交付正式规格/独立审查结论。后续不派发子 agent。旧文档的“未使用子 agent”是旧执行轮次的记录。

设计依据为 `continuation_design_spec.md`，在新增实现之前已保存。实现新增 `experiments/b1_reconstruction_20261016.py` 和 `experiments/b1_reconstruction_scheduler_20261016.py`，原 S6/S5 训练实现及 F0 production控制文件未改。

## 代码审查

1. 新 runner 继承 S6，复用实际 Sample loss 和 F/DH controlled step；主循环与原 S1/S6 顺序一致：main epoch → train评估 → persistent arch update → validation → 最优点更新。50 epochs与原超参保持一致。新增 dump在validation之后，不增加训练forward。
2. 完整 main Adam包含 C/P，arch Adam独立；动态检查参数覆盖且不交叉。序列化真实 state_dict，含 m/v/step/param_groups；未激活参数通过param_groups显式登记，无状态不伪造。
3. 保存原始RNG、架构迭代器创建RNG与已消耗batch数、每epoch模型普通Python温度属性、训练计数/历史和真实初始化hash。保存迭代器恢复recipe；未完成全训练恢复等价验证，`resume_verified=false`。
4. 每epoch参数/随机数在dump前后不变；工程检查还逐项比较磁盘optimizer内容与内存真值。新两臂初值及同epoch累计main/arch batch hash逐项配对，错误立即停止。
5. R0–R4仅从validation历史计算，严格最大值并列最早；冻结文件先写，再评估test。逐epoch重新评估validation并检查与原记录数值一致，保存val/test的row IDs、labels、local domains、predictions。历史test仍是retrospective。
6. 原 serializer会把历史initial pin写入输出；新serializer明确改回实测hash，另保留historical pin字段，并重算config hash。不以历史常量替代真实状态。
7. 本地scheduler只操作新study目录。逐卡独立准入，每卡上限5进程，加入预计新增峰值检查总显存≤90%；未知峰值使用20,000 MiB保守预留。每5秒监测，超过上限则停止本study最新worker，保留失败与现有状态，不触碰其他用户任务。并发实际数量由显存约束决定，5只是上限。
8. scheduler用独占文件锁防止重复启动；现有未完成attempt不自动重跑/覆盖。进程退出以真实return code为准；失败后等待已有有效worker结束，再进入诊断状态。

## 验证证据及解释限制

详见 `B1_reconstruction_20261016/engineering_verification.json`，只有其中source_sha256与当前源码一致的PASS才可用于启动。CPU工程fixture使用真实数据小批次、原完整M3OE与原控制更新，检查固定分母BCE/梯度、Adam公式与实际步进、磁盘完整状态、RNG不变、persistent iterator下一batch重放、F/DH实测初值一致。它不是正式训练，也未验证CUDA大batch峰值或整段恢复。

当前实测seed2022初值 `9cf5f785fbc1fc87cd5a8eb67afb3e94e9feb05d4511331b900d8b8f0e386730`；历史S6 pin为 `381a86db191ec55764195800b70e2fea10d23f9b055dee7f5d50ce951d8f2ae4`。两者不同是当前实测事实；具体差异源尚未因果定位，不能凭猜测归为操作系统或PyTorch。新F/DH一致只支持新study内配对，不能恢复历史identity。

因此B1原exact gate保持未满足；新重建是单独测量证据。A0/A1/A2缺失原件继续列INCOMPLETE。没有新数据可证明课程、动量或某类域普遍有效；不修改既有STRONG/MEDIUM/WEAK结论边界。B2/C1/C2/D1/E1仍按原规划条件逐项判断。

复核结论：允许在工程验证与磁盘/显存准入均实际通过后，执行已授权的独立重建测量；不得称为历史exact复现、独立复核或全部实验完成。后续必须用8/8实际完整输出做科学分析，保存未完成与矛盾证据，最后推送报告和关键数据。
