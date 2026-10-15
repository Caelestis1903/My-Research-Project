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

## 追加：完整流程与分析验证

`experiments/b1_full_pipeline_fixture_20261016.py` 在隔离CPU进程中使用真实数据的类别覆盖小子集，贯通两臂各50epoch、validation选点、逐epoch val/test、完整serializer。生产worker源码未改变。两臂共100份完整state和200份预测文件的哈希复核通过，结果见 `B1_reconstruction_20261016/full_pipeline_verification.json`。fixture输出均标识 `engineering_fixture=true`、`formal_valid=false`；不能计为正式run，也不支持任何方法效果结论。

初次CPU fixture在原S6 writer的CUDA协议检查处停止；通过fixture专用adapter复用serializer，并明确记录CPU与小子集偏离，未改生产设备合同。数值分析初次执行在最终receipt的相对路径转换处停止；修复路径归一化后完整重跑通过。失败产物保留，见 `engineering_failure_history.json`。

分析规格 `b1_analysis_spec.md` 在正式训练开始前保存。`experiments/b1_analyze_20261016.py` 默认要求8/8正式完成，逐个验证400份full state及800份预测，完整split覆盖/真实数据label绑定、真实选点、批次配对、AUC/LL独立复算、双锚点时序恒等式、每cell/seed及固定模型均值的paired bootstrap、校准和row-order窗口。它只产出数值，不自动签发机制结论或后续GO。

在两臂CPU fixture上，分析实际检查100份state和200份预测并贯通表图生成，receipt的scope是ENGINEERING_ONLY，formal_runs_completed=0。默认缺少正式矩阵时确实拒绝且没有创建正式analysis目录（`analysis_scope_gate.json`）。bootstrap的tie/weight AUC用100个独立sklearn加权用例复核，残差低于1e-14（`analysis_oracle.json`）。图表已检查工程标签和坐标。没有将工程数据或小批次曲线用于科学结论。

上述证据扩大了工程覆盖，仍未测得CUDA大batch峰值、未完成4seed正式矩阵，也未验证整段resume；这些限制保持。调度器真实PID及心跳须继续检查，不能从这份静态报告推断训练已开始。

## CUDA正式启动后的在线检查

后续CUDA完整batch测量及首attempt恢复过程见 `b1_native_recovery_review.md`。新的PID983855调度器在GPU0超过90%时，独立向GPU1发出5个正式任务，均进入fit并保存epoch0。首个seed2022/F/attempt2的真实state按SHA读取并核对：C80/P54/A4个参数tensor，主Adam完整134个state、arch完整4个state，每参数step分别49/1，与主/arch执行计数一致；六个验证cell的AUC/loss及train objective有限。seed2022/2023的F-DH epoch0初始化、main/arch累计batch hash均相同。这里只核验了实际首epoch，不代表400份最终状态全部通过。

数值分析器追加严格的FULL-history状态全覆盖及每参数step断言，以满足原分析合同；生产算法、launcher和scheduler未编辑。该检查在既有两臂50epoch工程数据的全部100份state/200份预测上重新贯通，在线epoch0亦通过。默认正式分析入口对当前仅完成部分epoch的运行继续拒绝，未生成正式analysis结果。新receipt绑定当前分析源码；本次增强验证不改变科学定义、选点或模型更新。
