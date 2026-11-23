# 00 — 全候选夜间执行手册

## 1. 执行原则

本任务不是优先级筛选。M1–M8 和五组预注册交互均是正式候选，全部执行。禁止：
- 按主观价值先后排序；
- M1 失败后停止 M2–M8；
- M2 无阳性信号就跳过 M3；
- 将 M4/M5 降级为只读审计；
- 排除 M8；
- 只对 Top-K 机制做独立确认或短程训练。

每个家族按其适用诊断器执行。机制的时间尺度、输入对象和训练动作不同，不要求所有机制使用同一类实验，但要求每个家族都有实际干预/诊断、对照和独立结果。

## 2. 权威上下文

执行前读取本目录全部方案、实际第一阶段报告：
research_reports/2026-11-24_open_mechanism_discovery/FINAL_RESEARCH_REPORT.md
最新 main 上下文：
experiments/reports/EXP_AI_CONTEXT_V2.md
以及 D2 分支 26-11-23_v2 的设计、S0 manifests、S1_RUN、SCOPE_DECISION、ANCHOR_INDEX 和原始结果。

第二阶段提示词指定的 26-11-23_multi_mechanism_discovery 路径及三个支持文件在当前 main 未找到。该差异已登记，禁止伪造读取记录。

## 3. 固定协议

- MovieLens、KuaiRand；M3oE fixed1；3 domains × 2 tasks；4 shared experts；FCN [512,256,256,64]。
- seeds 2022、2023、2024、2025、2026。
- baseline 50 epochs；batch size 40960；保留自然尾批。
- 数据划分、预处理、loss reduction、Adam、EMA、learning rate、weight decay、RNG、DataLoader 顺序以 D2 已核验的 effective source/config 为准。
- Validation 六格 Macro AUC 选点；Test 在所有算法与超参冻结后才开放。
- 固定 EMA 是既有组件和强内部参照，不作为新机制，不重新证明其有效性。
- 小域不是困难域；域规模不得作为困难度代理或机制权重。
- 正式发表基线 M3oE、HyperGate、AutoIFS；CLEAN、固定 EMA、Gaussian+EMA 和机制控制组属于内部对照。

## 4. 自动执行状态机

~~~text
INIT
 -> S0_SOURCE_AND_PROTOCOL_AUDIT
 -> S0_ALL_FAMILY_FEASIBILITY
 -> S0_INDEPENDENT_IMPLEMENTATION_REVIEW
 -> S0_RESOURCE_SMOKE
 -> S1_BASELINE_TRAJECTORIES
 -> S1_ALL_M1_TO_M8_LOCAL_DIAGNOSTICS
 -> S1_ALL_FIVE_CROSS_FAMILY_INTERACTIONS
 -> S2_ALL_VARIANTS_VB_CONFIRMATION
 -> S3_ALL_TRAINABLE_VARIANTS_SHORT_TRAJECTORY
 -> S3_FAMILY_AND_VARIANT_DECISIONS
 -> S4_ALL_QUALIFIED_VARIANTS_FULL_TRAINING
 -> FINAL_REPORT_AND_CONTEXT_HANDOFF
~~~

任一 family 的技术阻塞只影响该 family/variant，不阻塞其他 family。任何科学 NO_GO 不得提前终止其他机制。不得以 S1 的结果删减 S2 或 S3 的注册候选。

## 5. S0：全家族可行性审计

S0 不训练，但必须逐家族验证输入、算子和数据对象真实存在：
- M1：二阶梯度图与 P-only Adam step；
- M2：逐 row-task 梯度、虚拟 Adam 和 Train-only ranking calibration；
- M3：阶段锚点、固定样本组与组合干预；
- M4：各数据集真实 feature schema、条件依赖测量和可执行干预；
- M5：所有预注册变换的字段支持、变换前后字段合法性与多标签一致性；
- M6：cell ranking margin Jacobian/JVP/VJP 与信任域求解；
- M7：五 seed 对齐的样本影响分数和留一 seed 计算；
- M8：六 cell 梯度、冲突矩阵、投影和权重求解。

每个 family 输出独立 FEASIBILITY.json。若一个具体变换不适用，标记 NOT_APPLICABLE 并继续该 family 的其他变体。若算子、资源或状态语义不成立，标记 TECHNICAL_BLOCKED；不得静默换成不同机制。

## 6. S1：统一基线和全部局部诊断

### 6.1 Baseline / anchors

建立 2 datasets × 5 seeds = 10 条 50-epoch baseline。先对 D2 的 10 条 J_ORIG trajectory 和 30 anchors 做 live readback。只有 model、Adam、EMA、step、RNG、sampler、loader cursor、source/config/data hashes 全部一致才复用；否则从冻结源重建。

每条轨迹保留 E10、E25、E40 三个完整状态。每个 anchor 都能独立 fork。

### 6.2 Validation panels

每个 dataset/seed/anchor 建立 V_A 与 V_B 两个互斥 panel，按 domain×task 分层，保留自然行权重、row IDs、正负 pair IDs。V_A 是筛选读数，V_B 是独立确认读数。所有候选在设计时就注册，所有候选均评价 V_A 和 V_B；V_B 不能被用于训练或超参选择。

Train-only calibration panel 与训练优化 batch 不重叠。M2/M7 的样本影响分数只能由 Train calibration 计算。

### 6.3 全量执行规则

执行 04_EXPERIMENT_MATRIX.md 中的全部 2,190 个局部 fork slots。M1–M8 不按结果触发；M3 永远运行；五组交互永远运行。各 arm 从相同 anchor 状态克隆，保留完整状态和 action receipt。

M1 的二阶图、M2 的逐样本虚拟 Adam 等高成本操作可以按 anchor 分批、按 cell 顺序计算，但不得通过降低精度、缩小 batch 或改用一阶代理改变机制定义。

## 7. S2：全部变体的 V_B 独立确认

S2 不是“只确认 S1 胜者”。所有 S1 注册变体都必须在 V_B 上计算相同的预测、排序和 cell-level 读数。若变体是训练动作，使用其冻结 endpoint；若变体是纯归因诊断，使用冻结分数在 V_B 上检验预测方向。

所有候选的 S1 与 S2 结果同时保留。S1 阴性但 S2 阳性、S1 阳性但 S2 阴性均如实记录，不按单一面板删候选。

## 8. S3：所有可训练变体的短程轨迹

所有具有可执行训练动作的注册变体均运行 5-epoch continuous trajectory，不要求先进入 S1/S2 Top-K。S1/S2 结果用于解释，但不作为 S3 的删减条件。纯测量型变体继续完成其跨阶段/跨 seed 诊断，不强行伪装成训练算法。

每个短程 run 从 E25 anchor 恢复，执行 E1/E3/E5 readouts，保存完整 model/Adam/EMA/RNG/loader state。所有匹配控制与候选使用相同数据 exposure 和 batch schedule。

## 9. S4：全部合格变体的完整训练

S4 不设候选数量上限。所有同时满足：
- 实际动作执行正确；
- S2 有独立机制/预测响应；
- S3 有持续或可解释的阶段性响应；
- 相对匹配预算控制存在可重复差异；
- cell protection gate 通过；
的候选变体均单独进入 2 datasets × 5 seeds × 50 epochs 完整训练。

不能只选平均收益最高的一个，也不能因候选数量多而只跑 Top-K。若多个变体通过，全部跑。超参以五 seed 综合 Validation 结果选择并冻结。Test 在各候选方法、超参、checkpoint 规则冻结后统一评价。

## 10. 夜间自主运行与失败恢复

- 所有 family/variant 在启动前写入不可变 RUN_MANIFEST。
- Scheduler 使用 family-round-robin + dataset/seed/anchor interleave，不设置研究优先级。
- 单个任务失败不停止队列。
- 技术失败最多两次重试；科学 NO_GO 不重试。
- 每个 fork 原子写入状态、metrics、cost、receipt。
- 不设任意墙钟截止；本次夜间窗口未完成时，保持 QUEUED/CHECKPOINTED 状态，下一次自动续跑。
- 不因资源紧张改变 batch、precision、optimizer、数据、模型或方法。先降低并发。
- 所有失败、取消、OOM、NOT_APPLICABLE、技术阻塞均保留。

## 11. 全局禁止事项

不重新证明固定 EMA；不重复固定 U→P/P→U scope sweep；不将小域称为困难域；不将代理指标当作 AUC；不将 anchor/fork/pair 当独立 seed；不提前使用 Test；不静默删掉阴性候选；不把一个机制家族的 NO_GO 外推至其他家族。
