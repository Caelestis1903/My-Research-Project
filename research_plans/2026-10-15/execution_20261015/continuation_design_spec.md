# B1 dense measurement reconstruction：执行规格

2026-10-16，主 agent 编写。用户最新要求严禁子 agent，已停止此前派发的两个工作；后续设计、实现、复核、运行、分析均由主 agent 顺序执行。自审不能满足原规划的独立复核证据等级，必须在结论中保留这一限制。旧报告的“从未使用子 agent”仅描述旧轮次。

## 研究问题与边界

优先研究原规划 B1：KuaiRand Sample F/DH 的 validation/test 非目标反转是否在每 epoch 的完整测量中持续，并区分同 epoch policy difference 与各自选点时序。原 S6 原始 checkpoint/optimizer/预测在当前项目、project.zip、同用户封存与远程 tracked tree 中未恢复；因此新实验是独立重建配对，不能登记为历史 exact replication，不能填补 A2 的缺失原件。

新研究身份 `B1_reconstruction_20261016`。原规划、旧 F0 hold/control/authorization、历史 run 不改写；不使用 F0 production queue 或为其签发 GO。用户完整实验授权覆盖这个独立的测量实现及运行，不能据此声称旧 F0 已合格。

## 冻结协议

- KuaiRand `/home/xzf/data/kuairand/processed/kuai_rand.csv`，SHA256 `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`；natural 原全局有序 80/10/10，target local d1，任务不改。
- M3OE，experts=4，embedding=16，fcn=[512,256,256,64]，batch=40960，50 epochs（0–49），workers=0；main Adam lr=.01，arch Adam lr=.001，betas=(.9,.999)，weight_decay=1e-5。
- F/DH × seeds 2022/2023/2024/2025 共8 runs。复用 S6 原 loss、梯度控制和 main/arch loop；P 始终 FULL，DH history FULL/current TARGET，不能改为新的归一化目标。
- 按实际初始化/数据/schema/源码/RNG/batch fingerprints登记。历史 init pin仅作比对，不能把常量冒充实际测量。配对只在新 study 内证明；与历史差异披露。
- 不提前停训；validation six-cell macro AUC 最大、并列最早；R0 macro，R1 target，R2 non-target，R3 worst-domain，R4 evaluation-row weighted。全部 validation 轨迹先冻结，之后才访问 test；test 为 retrospective forensic，不用于方法/超参选择。

## 实现合同（新增文件，不改变历史 runner）

新增 `experiments/b1_reconstruction_20261016.py`：继承 S6Runner，在 epoch 边界保存真实完整 model/main Adam（含 private）/arch Adam、RNG、loader迭代信息、批次历史、训练计数与选点上下文。完整持久化不能只存 m/v 的范数。为避免把无法准确恢复的 iterator 称可恢复，应保存实际训练 loader generator/迭代器位置或明确重放验证；未经恢复验证不得标注 resume verified。

每 epoch保存 full state与验证指标；训练结束冻结选点后逐 epoch test预测，保留 row IDs/labels/local domain。原 S6 的 sparse outputs可继续保留，但新 manifest 必须使用实际 initial hash并单独标注重建身份。原 invariant、source drift检查保留。

新增调度入口（可同文件子命令）：持久化进程，逐卡读取显存、分别准入。每卡最多5本研究运行，现有总显存和预计新增峰值不超过90%；冷启动先单进程测量峰值。未知峰值保守排队，不通过缩小 batch/模型改变实验。其他用户任务保持原样。磁盘预留完整状态空间；不足时等待或报告，不丢弃要求的epoch状态。

## 验证与复核

依次进行 syntax/basic check、固定输入loss/梯度及Adam实际更新oracle、完整状态覆盖、dump前后RNG/参数/optimizer不变性、同 seed 初值与批次配对检查。必要小规模检查仅作工程验证，不能计入8正式runs。用户要求完整实验的验证允许这些检查。

主 agent 复核设计与实现并形成实现报告及复核记录；明确 SELF_REVIEW，绝不标独立 PASS。任何数值不变量、参数分组、状态覆盖或配对错误都阻止正式运行，先修复。若旧控制文件实际直接禁止独立新路径，应保留原件并记录具体scope，不绕过执行代码中的拒绝。

## 分析与停机

逐seed、逐epoch、逐domain×task AUC/LogLoss、macro/N/T/weighted/worst并列。记录train loss、selected/common/final；same-epoch Δ和selected timing decomposition分开。固定RNG的同row配对分层bootstrap与seed异质性分开；不把4seed×N视独立样本。报告时间依赖与多次test开发限制。

实用阈值沿原规划 +.003、强阈值+.005，至少3/4seed同向仅作探索门，不替代统计检验。反转消失、时序主导或不确定性很大时保留反证并停止该线；不自动触发新方法。B2/C1/C2/D1/E1仍各按原条件判定，不以完成B1代替全部目标。最终须如实更新三份报告、机器状态与关键数据，并推送已授权的专用分支；完成状态需逐项审计。
