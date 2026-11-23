# 00 — Codex 执行手册

## 1. 总目标

在统一 M3oE fixed1 状态与评估基础上，联合诊断：
- M1：私有参数能否通过改变各 cell 对共享参数的梯度方向，使共享更新更协调；
- M2：同一训练样本对不同任务/cell 的局部排序影响是否存在可重复的异质性；
- M3：M2 识别的样本组是否存在阶段依赖或非加性组合影响。

M4/M5 仅做数据/语义可行性审计。所有阶段只使用 Train 与 Validation；Test 保持封存。

## 2. 权威来源与冲突裁决

执行前读取：
1. 本目录 README 与 00–11 全部文档。
2. main 的 experiments/reports/EXP_AI_CONTEXT_V2.md。
3. main 的 research_reports/2026-11-24_open_mechanism_discovery/FINAL_RESEARCH_REPORT.md。
4. 分支 26-11-23_v2 的 D2 设计、D2 S0 manifests、S1_RUN.json、SCOPE_DECISION.json、ANCHOR_INDEX.json。
5. 相关源码：D2 分支的 experiments/mdmt_shared_private_coordination_20261123/；fixed1 authority 由 D2 DESIGN_SPECIFICATION.md 所列 v3/EMA02/GA31 依赖链确定。

若 prompt 指定的 26-11-23_multi_mechanism_discovery 报告路径仍不存在，记录 SOURCE_PATH_MISMATCH；继续使用实际存在的 2026-11-24 报告。不得自行生成缺失的第一阶段支持文件。

## 3. 强制协议

- 数据集：MovieLens、KuaiRand。
- 模型：M3oE fixed1，3 domains × 2 tasks，4 shared experts，FCN [512,256,256,64]。
- seeds：2022、2023、2024、2025、2026。
- 基线完整轨迹：50 epochs，batch size 40960，保留自然尾批。
- Validation 六格 Macro AUC 选点；Test 不加载、不评分、不用于调参。
- Adam、EMA、学习率、weight decay、数据划分、预处理、loss reduction、RNG 和 DataLoader 顺序全部沿用 D2 已核验的实际有效配置。不得根据字段名猜测配置；以 D2 SOURCE_MANIFEST、DATA_MANIFEST、实际 adapter 和 fixed1 canonical source 为准。
- 正式算法性能门槛：五 seed 配对平均 AUC +0.003；每个 domain-task cell 的平均下降不得低于 −0.001；LogLoss 独立报告。
- 小域不是困难域；禁止使用域大小作为困难度或诊断分组代理。
- 固定 EMA 是既有组件和强内部参照，不作为新机制。

## 4. 状态机

~~~text
INIT
  -> S0_SOURCE_AUDIT
  -> S0_MODEL_DATA_AUDIT
  -> S0_IMPLEMENTATION_REVIEW
  -> S0_RESOURCE_SMOKE
  -> S1_BASELINE_STATE
  -> S1_M1_M2_SCREEN
  -> S1_M3_CONDITIONAL
  -> S1_ANALYSIS_FREEZE
  -> S2_CONFIRMATION
  -> S3_SHORT_TRAJECTORY
  -> S3_DECISION
  -> [S4_NEW_METHOD_SPEC_REVIEW -> S4_FULL_TRAINING]
  -> FINAL_REPORT
~~~

S0 任一身份/恢复/实现关键项失败，状态为 TECHNICAL_BLOCKED，不得进入 S1。S1/S2/S3 科学 gate 未通过为 NO_GO；样本支持不足或确认集失效为 INCONCLUSIVE。

## 5. S0：来源与协议审计（零训练）

必须生成：
- SOURCE_RECEIPT.json：分支、commit、blob SHA、源文件 hash、依赖闭包；
- PROTOCOL_LOCK.json：每数据集 effective config；
- DATA_IDENTITY.json：raw data hash、split row IDs、train/validation hashes、预处理；
- PARAMETER_GROUPS.json：参数名、shape、numel、requires_grad、forward support、gradient support、optimizer membership；
- MODEL_FIDELITY.json：source vs adapter forward/probability/loss parity；
- FEATURE_SEMANTICS_AUDIT.json：M4/M5 字段审计；
- RESOURCE_AUDIT.json：GPU、显存、RAM、CPU、其他进程和预计准入；
- IMPLEMENTATION_REVIEW.json：独立审查结果。

固定1必须确认四个 fusion constants 是不可训练常量；若出现 optimizer-visible fusion scalar 或与 D2 manifest 不同的 trainable parameter grouping，阻塞。

参数分组必须从运行时模型导出，禁止只按名称字符串猜测。U 为真实共享参数组；P 为全部可训练专有参数组；FROZEN/OTHER 另列。P 内 gate/tower/domain_expert/task_expert 的实际结构依赖与梯度 support 必须记录。若组重叠或 support 不是预期，先审查，不得强行假定六个互斥私有头。

## 6. S1：统一基线与多机制筛选

### 6.1 基线状态

目标为 2 datasets × 5 seeds 的 10 条 50-epoch 轨迹，锚点 E10/E25/E40。先检查 D2 的 10 条 J_ORIG 轨迹和 30 个 anchor 是否可从当前受控状态服务完整读回。只有当 model、EMA、Adam、step、RNG、sampler、loader cursor、config、source、data hash 全部吻合时才复用。

D2 S1 的 SCOPE_DECISION 为 NO_GO，不等于 D2 anchor RAM 状态当前仍可访问。若只有 endpoint 数值/哈希索引而没有完整状态读回，必须从冻结来源确定性重建，不得把 endpoint 当作可恢复状态。

### 6.2 锚点面板

每个 seed、数据集、anchor 从同一 Validation row population 划分互不重叠的 V_A 与 V_B。按 domain×task 分层，保留原始行权重和行 ID。固定正负样本对只用于排序诊断；pair count、row count、缺类情况均写入 manifest。V_A 用于候选筛选，V_B 在 S2 独立确认。不得把同一行/样本对跨 panel 重复。

训练数据另划定 label-blind calibration panel；它只从 Train 中抽取，不进入优化 batch。M2 影响分数仅由 Train calibration 计算，不能使用 Validation 来决定训练样本权重。

### 6.3 M1 筛选

每个 anchor 使用相同的两批训练样本 A/B 和同一完整 Adam/EMA 状态 fork。M1 的核心处理不是固定 U→P，而是先在 U 冻结时调整 P，使各 cell 的共享梯度更一致，再执行同一后续联合更新。

对每个 cell c，计算：
g_U,c = ∇_U L_c(U,P)
g_U,J = ∇_U L_joint(U,P)

私有一致性目标：
L_cons(P) = Σ_c w_c [1 − cosine(g_U,c, g_U,J)]
其中 w_c 在诊断阶段固定为六格等权；同时记录自然样本权重下的辅助版本，但不改变主目标。计算 g_U,c 时只使用 Train batch；不得用 Validation 梯度优化 P。

在 P 更新前后记录 L_cons、六个 cell 的共享梯度 cosine、相对方向残差、梯度范数、P 位移和六格预测变化。二阶梯度只通过 autograd HVP/反向图计算，不显式构造 Hessian。

处理组：
- M1-0：joint baseline 两步；
- M1-1：P-consistency step（U 冻结）→ joint step；
- M1-2：P-BCE step（U 冻结）→ joint step，控制额外 P 更新；
- M1-3：P-consistency 目标的 cell 对应关系打乱 → joint step，控制 cell-specificity；
- M1-4：consistency 更新后不执行 joint step，仅测 P 直接改变输出的效果。

M1-0、M1-1、M1-2、M1-3 的 optimizer.step 次数均为两次；所有 arm 从同一 anchor 状态克隆。M1-4 是机制读数，不参与预算公平主比较。更新 U 冻结时必须确保 U 的参数、Adam m/v/step 完全不变；P 更新只允许改变 P 的状态。若二阶计算超出资源门槛，允许 S0 将 M1 标为 TECHNICAL_BLOCKED，不允许静默换成一阶近似后仍称 ConsMTL-style。

### 6.4 M2 筛选

M2 不按 loss 或梯度范数排序样本。对 Train calibration pair 的 cell-specific ranking margin m_c，计算样本 i、任务 t 的局部预测影响：
s(i,t,c) = − ∇_θ m_c · Δθ(i,t)
其中 Δθ(i,t) 是从 anchor Adam 状态对该行任务 t 梯度执行的虚拟单步更新；只计算需要的向量积，不持久化全 Hessian。该量是局部一阶预测，不称为真实因果影响。

每域每 seed 每 anchor 预先抽取固定 label-blind 样本 panel，按真实存在的标签组合和任务支持记录分层。若某组合不存在（例如 MovieLens 嵌套标签导致的结构零），标记 NOT_APPLICABLE，不补造样本。

处理组：
- M2-0：均匀自然目标联合更新；
- M2-1：按 task-specific predicted ranking impact 加权，权重在每个 domain-task 内归一到均值 1，并 clip 到冻结范围；
- M2-2：把 task-specific score 跨任务平均成 task-agnostic score，权重总质量匹配；
- M2-3：在 domain 内打乱 score，保留权重分布与总质量。

所有组保持相同 batch rows、样本暴露、总 loss mass 和 optimizer step 数。M2 只在 V_A 读取局部排序变化；不使用 test。另以有限真实 fork 验证 top/bottom 样本组的预测方向，代理指标与实测 AUC 分开。

### 6.5 M3 条件性组交互

仅当 M2 在至少两个 seed、至少两个 anchor 发现方向可复现的候选样本组，才触发 M3。按冻结规则选择两个不重叠组 A/B，每组相同样本数。比较 A-only、B-only、A/B 等量混合以及同组重复的 exposure-matched control。定义：
I_AB = ΔU(A∪B) − 0.5ΔU(A) − 0.5ΔU(B)
所有 arm 具有相同 batch size、optimizer step 和 Adam 起点；A∪B 为固定总 batch 中各半，不得以 2 倍样本量对比单组。若样本支持不足或配对失衡，M3 为 INCONCLUSIVE，不扩大搜索。

M3 在 E10/E25/E40 分别诊断，stage effect 只作为条件现象，不自动触发 curriculum。

## 7. S2：独立确认

S1 完成后冻结：
- 最多一个 M1 consistency candidate；
- 最多一个 M2 task-specific influence candidate；
- 如满足触发条件，最多一个 M3 group pair。

S2 只在 V_B 上确认；所有候选、超参数、对照、读数和门槛在解封 V_B 结果前冻结。五个 seeds 是统计单位，V_B 的多个 cell/anchor 是 seed 内重复测量。若只有一个 seed 或一个 anchor 出现信号，不晋级。

## 8. S3：短程轨迹

仅对 S2 通过的候选执行。每个候选独立从同一确定性基线恢复，连续运行 5 epochs，保留 E1/E3/E5 轨迹与完整 optimizer/EMA/RNG。每个候选 2 datasets × 5 seeds。对照包括标准 joint training、相同更新预算的控制动作、候选动作。短程轨迹用于判断累积/衰减/符号翻转，不等同 50-epoch 正式性能。

## 9. S4：完整算法晋级（条件阶段）

S3 通过后，Codex 必须先生成新算法 DESIGN_SPECIFICATION，定义可部署动作、信息流、复杂度和超参数；独立 Reviewer PASS 后才能执行。若没有任何候选通过，S4=NOT_TRIGGERED。

正式训练：2 datasets × 5 seeds × 50 epochs；Validation 六格 Macro AUC 选点；Test 只在算法、参数和 checkpoint freeze 后进行一次性评价。正式发表基线 M3oE、HyperGate、AutoIFS；内部控制 CLEAN、固定 EMA、Gaussian+EMA 和候选机制最小控制组分开列示。

## 10. 终止与恢复

每个 run 采用原子状态：ADMITTED → RUNNING → CHECKPOINTED → COMPLETE_VERIFIED / FAILED_RETRYABLE / TECHNICAL_BLOCKED。每个 anchor/fork 需独立 owner、state hash、config hash、source hash、data hash。恢复必须校验 model/Adam/EMA/RNG/loader 状态及下一批 row IDs。

失败重试最多两次，且只对技术失败；科学 NO_GO 不重试。OOM 先释放本 run 中间对象、降低并发，不得改变 batch size、模型或方法配置来伪造可比性。所有失败、取消、重跑保留在 ledger。

## 11. 禁止事项

- 不重复 D2 静态 U→P/P→U scope sweep。
- 不把 M1 consistency objective 与普通 P-BCE step 混为一谈。
- 不用 Validation 或 Test 梯度驱动训练动作。
- 不将样本梯度相似度称为真实 influence/causal effect。
- 不把六格、anchor、fork、sample panel 计为独立 seed。
- 不将诊断 NO_GO 外推到整个机制家族。
- 不修改基础协议，不提前解封 Test。
