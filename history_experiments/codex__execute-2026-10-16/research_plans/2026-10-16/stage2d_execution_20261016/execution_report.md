# Stage2D 续执行报告

2026-10-16，主代理。正式训练 **16/16 COMPLETE、退出码全部 0**，数值核验通过。此前 `../execution_20261016/` 的“停止所有新训练”解释已由 [continuation_gate_reassessment.md](continuation_gate_reassessment.md) 纠正；旧报告及其数值保留为历史快照。本报告不把缺失的历史实验原件标为恢复成功，也不把本次新研究叫作历史 exact replay。

## 实际执行与身份

研究 `STAGE2D_KR_IID_20261016`，KuaiRand-natural / M3OE，SN/SB/EN/EB × seeds 2022–2025。训练代码冻结于 `66c4cb33d4c3677a38f65010e06b6e7a3cacfc17`，worker SHA256 `115c37e1e865a8e2da7802bcf7a90c1edecc3eec5dabfe783e165457b7706b91`。全部 run 的实际配置、命令、GPU、PID、起止时间及退出码见 `queue_status.json`、`queue_events.jsonl` 和 `runs/seed*/ARM/attempt1/`。

原始计划来源、AGENT.md、start.md 编码/指令冲突、环境和历史工件恢复见前次 [执行报告](../execution_20261016/execution_report.md)。本次按用户要求优先执行 start.md§11：主代理完成设计、实现、自审和科学分析，运行代理只做进程/日志机械监督；**没有独立 reviewer PASS**，所有 config 如实保存 `independent_review=false`。设计完成后才实现；工程核验与自审完成、源码提交后才启动正式训练。

复用 `phase0-3-env`：Python 3.13.5、Torch 2.9.0+cu126、CUDA 12.6；两张 RTX3090。数据 SHA256 为 `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`。原工作区和历史结果未覆盖，发布使用独立 worktree `/home/xzf/Code/phase0-3-20261016-git`、分支 `codex/execute-2026-10-16`。

## 冻结协议

详见 [设计规格](design_specification.md)、[分析规格](analysis_specification.md)。Sample alpha=p_train，Equal alpha=(1/3,1/3,1/3)；Natural q=p_train，Balanced q=(1/3,1/3,1/3)，weight=alpha/q。两种 q 都为域内均匀、有放回 IID，使用相同 uniform 流耦合。同 seed 的 SN/EN、SB/EB 有相同初始化与每批 row IDs。这个 SN 是新 IID 协议的 Sample/Adam 对照，与旧 shuffle-S1 执行身份不同。

模型为现有 StageS1M3OEAdapter，embedding16、4专家、fcn=[512,256,256,64]，保留现有四个 learned fusion scalars；没有恢复已删除的 V1/V2 参数。C/P main Adam lr=.01，A_arch Adam lr=.001，betas=(.9,.999)、eps=1e-8、coupled WD=1e-5；无 clip、scheduler、early stop。

每 run 50 epochs，每 epoch main 曝光 2,006,132 行、49 updates，另有一个 40,960 行 arch batch。每 run 2450 main/50 arch updates；全矩阵共 800 epochs、39,200 main/800 arch updates。各臂处理数量、更新数、模型和评估日程相同；记录 wall time，不声称并发环境下实际耗时相同。

每 epoch 全量 validation；以 six-cell macro AUC 严格大于时更新选点，平局最早。先写 `selection_frozen.json`，再读取已开发 test，仅用于 forensic。保留 selected、common 9/19/29/39/49、final 的完整模型/优化器/RNG/sampler state 和相应预测。Test 不参与选点、阈值、重跑或方法确认。

## 执行结果与最小验证

| seed | SN selected epoch | SB selected epoch | EN selected epoch | EB selected epoch | 状态 |
|---|---:|---:|---:|---:|---|
| 2022 | 42 | 22 | 22 | 8 | 四臂正常完成 |
| 2023 | 25 | 45 | 7 | 6 | 四臂正常完成 |
| 2024 | 29 | 36 | 17 | 6 | 四臂正常完成 |
| 2025 | 41 | 47 | 28 | 5 | 四臂正常完成 |

Epoch 为 0-based。队列从 07:55:04 UTC 到 09:08:18 UTC，约 73.2 分钟；各 run wall time 1112.0–1640.0 秒。没有正式 run 失败、重试、NaN 或 OOM。工程 attempt1 和修订记录保留，最终正式源码的 CPU 完整小流程及 CUDA 40,960 batch profile 通过；正式训练未因分析发现而改变配置。

68 次机械监控中，GPU0/1 的最大已记录显存比例为 81.52%/85.27%，本研究每卡最多 1/5 个进程；准入独立检查每卡90%上限，未终止外部作业。每进程 Torch reserved peak 为1262 MiB，队列按2345.5 MiB预留启动余量。监控是离散采样，不假称逐微秒连续证明。

`analysis/receipt.json`：16份 config/源码核验、400项 producer artifact hash 全部一致；8组同q配对各2500批记录吻合；同seed四臂 init一致；50轮/2450+50步完整；严格 earliest validation argmax 与保存选点一致。32份 selected val/test NPZ 逐行绑定原CSV row IDs、labels、domains和split范围后重算六cell AUC/LogLoss，与CSV差小于1e-10。原始checkpoint等全部纳入producer hash，没有重写原manifest以补齐字段。统一selected schema另由 `analysis/unified_selected_run_schema.csv` 导出。

`analysis/readout_receipt.json` 记录common epoch、选点时序、曝光、theta/m/v描述表；条件row bootstrap的限制与完成状态见 `analysis/bootstrap_receipt.json`。这两个receipt只声明数值/表格完成，科学判断见后两份报告。

## 全计划清单及停止边界

| 阶段 | 本轮最终处理 |
|---|---|
| WP1/WP2、A0/A1/A2 | 可访问工件审计和法证完成；缺失原件逐项登记，未当作全局禁训 |
| 2A | 15篇原始来源 novelty matrix / Q1–Q4完成，未作全球创新认证 |
| 2B/2C | 8个新增零训练case通过；严格限定有限pool、固定theta和投影坐标 |
| 2D | 新增16/16正式训练完成并核验；同目标Balanced未通过收益/guard门 |
| 3 | 复用既有B1八run的轨迹、选点和timing法证；本次2D另提供相同epoch对照 |
| 3B | **INCOMPLETE**：原S6-R2逐行预测/原身份仍缺，未用新研究替代 |
| 4 | **BLOCKED/NOT_RUN**：原S5 artifacts未恢复，A0/A1门未通过；不是科学NO |
| 4B | **NOT_RUN_DEPENDENCY**：无合法原轨迹shadow-state来源；未拼接异轨迹m/v |
| 5 | 可恢复B1/2022两个state及取证范围完成；全epoch/全seed证据仍不完整；无B3/B4/B5新训练 |
| 6 | **NOT_TRIGGERED**：冻结的实用收益及guard未通过，optimizer-specific性能因果机制未确认；没有为解释阴性追加边界网格 |
| 7/E1 | **NO_GO**：相对强Sample对照无稳定改善，独立setting/新holdout/分量因果等门也未齐；不开发新方法 |

**不能把整个Track A写成“没有效应所以STOP”。** Sample下存在稳定负向selected效应，state差也并非消失；原§7的联合STOP和§18“无稳定selected effect”不因此成立。本次可执行的预定四臂矩阵已完成；停止的是无实用收益的候选/方法开发升级，保留optimizer-specific因果问题未识别。历史依赖仍为INCOMPLETE/BLOCKED，不伪称所有规划训练均已完成。

原件恢复路径已在任务执行期间向用户征询；当前可访问范围内没有新增来源。没有等待用户再次发送“继续”来启动本可执行的训练。若以后获得原S5/S6材料，按原Stage3B/4依赖继续；不应重跑本16项来替代缺失历史。

## 交付与复现

三份续执行报告、`analysis/` CSV/JSON/图、每run config/manifest/metrics/batch指纹、32份选中点预测及监督记录归档到研究分支。原始数据集、所有模型/优化器checkpoint、重复common/final全量预测及完整stdout保留本机，hash和准确路径见run_manifest和本机产物索引。运行/重算及恢复边界见 [reproduce.md](reproduce.md)；当前没有经过验证的断点恢复CLI，不能只因state完整就声称支持无缝恢复。

Git最终提交与远端核对结果见 `publication_status.json`。发布并不改变各科学节点的NO/INCOMPLETE/NOT_TRIGGERED含义。
