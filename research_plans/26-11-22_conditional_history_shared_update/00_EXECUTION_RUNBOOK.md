# MDMT 条件历史与共享更新研究：Codex 执行入口

## 1. 交付与任务范围

本目录是完整研究设计，研究快照为 main@875215c1e509ab55c1b6632d4f5d13e8b94782da。实施时先读本文件，再读 [工程规范](06_IMPLEMENTATION_SPEC.md)、[实验矩阵](05_EXPERIMENT_MATRIX.md)、[决策规则](09_DECISION_RULES.md) 与 experiment_manifest.json。最终规划审查见 [13](13_PLAN_REVIEW.md)。

本次规划交付只新增本目录，不实现算法、不启动训练。后续 Codex 执行本方案时完成 DESIGN → IMPLEMENT → 独立 REVIEW → 受监督 RUN → ANALYZE → 报告与交付。研究范围是条件辅助梯度、梯度历史和共享参数更新；不修改推荐主干，不开展独立损失函数或选择性共享架构研究。

**当前研究判断：** 固定 EMA 的既有收益已接受；P01 在 KuaiRand 相对固定 EMA 有五种子宏 AUC 正增量但违反两格保护。缺少的是 P01 相对同样条件样本对的普通 Adam 的五种子独立比较，以及真实共享更新如何影响其他任务排序的直接证据。

## 2. 规范优先级与本轮边界

1. 本次用户附件及其明确五种子、自然人口、验证选点、最终 Test 和交付要求优先。
2. 继承 main 的 AGENTS.md、start.md 中当前会话监督运行、独立 coder/reviewer、每卡当前资源准入、RAM_ONLY 状态和科学证据记录规则。
3. 旧 C04/C05 的停止请求、无限循环段落、旧四种子 runner、旧两种子探索门只适用于其历史任务。本轮执行到本目录的最终报告即结束，不启动 C06，也不恢复旧 autonomous driver。
4. 对本轮新生成的冻结候选，附件已要求验证选点后 Test；按本轮 FINAL_FREEZE 流程执行，不再添加“等待用户手动结束旧循环”的条件。旧 C01–C05 封存 Test 不因本方案自动开放。
5. 研究者已固定“小域不是困难域”；本轮不把域规模当困难度，不开展该命题的实验。
6. start.md 禁止模型、optimizer、检查点等训练张量落盘。完整训练状态只保存在当前会话监督的 RAM 中；磁盘保留代码、配置、数据身份、标量、预测评价记录与报告。明确的输入扰动属于本次获准研究，原始人口、划分及 loader 采样比例保持不变。

## 3. 首轮必须执行的队列

| 顺序 | 执行内容 | 新科学训练 | 状态重建 | 完成条件 |
|---|---|---:|---:|---|
| S0 | 来源、配置、数据、复用记录与实现审查 | 0 | 0 | SOURCE_READY、REVIEW_PASS |
| S1 | A10=P00、A30=P10 缺失种子补全 | 12 | 0 | 两数据每臂均有 2022–2026 |
| S2 | A 的五种子独立收益判断 | 0 | 0 | A_GATE.json |
| S3 | B 的 clean 轨迹及核心机制干预 | 0 | 10 | B_CORE_REPORT；不依赖 A 晋级 |
| S4 | 按 A/B 实证门执行有限候选及轻量搜索 | 条件触发 | 条件触发 | 分支明确完成或停止 |
| S5 | 冻结候选、统一最终评价、报告与提交 | 默认复用已完成训练 | 必要时恢复冻结状态 | FINAL_REPORT、FINAL_REVIEW、提交回读 |

S1 的 12 条来自：2 数据集 × 2 臂 × 缺失 seeds {2023,2025,2026}。旧 P00/P10 的 {2022,2024} 已有匹配记录。A00 clean 与 A20=P01 各十条已有五种子数值，不重复运行来确认已知总收益。

S3 的十条是恢复真实模型、Adam 和 EMA 锚点所需的同源重放；旧状态已释放或仅剩 selected 模型，不能提供原优化器。对 B 属于新增机制数据，对训练性能统计不增加 seed 数。若执行时发现同身份完整 RAM 状态且能现场核验，则逐条扣除重放量；不能以历史存活收据代替现场验收。

**默认完整物理轨迹 22 条，每条 50 epoch。** 其余候选只有对应门通过才生成任务，详见 [矩阵](05_EXPERIMENT_MATRIX.md)。A_GATE 为停止时仍必须执行 S3。

## 4. 实施步骤

### S0：来源恢复与工程准备

- 从 [来源与复用登记](12_SOURCE_AND_REUSE_REGISTER.md) 恢复指定文件；优先 main，只有 main 缺失的历史报告、实现及原表才读取固定来源 SHA。
- 审查五种子 endpoints、原两种子 endpoints 的身份，核对训练/数据/选点/readout/模型 hash；登记复用到 REUSE_MANIFEST.json。
- 当前已核实跨表 96 条 A0/P01 端点的模型 hash 和 AUC 完全一致，LogLoss 最大舍入差约 5.6e−17。Codex 的复核只验证文件身份和已有桥接结论，不重新搜索实验。
- 创建新的 experiments/conditional_history_shared_update_20261122/ 模块；保留旧 v3 科学内核，替换它硬编码两种子和旧调度门的外层 orchestration，不改旧结果。
- 执行 [必要正确性检查](06_IMPLEMENTATION_SPEC.md)；独立 reviewer 检查源码、设计和 coder 报告，通过后启动科学训练。
- 缺 CSV、身份不符、显存不足记 TECHNICAL_BLOCKED，不能记科学 NO_GO。准备数据目录缺失不等于原始 CSV 不可用，按冻结预处理恢复并校验。

### S1–S2：方向 A 归因

完成 P00/P10 补全后，输出每 seed 的 selected Validation 六格、Macro 和 LogLoss，分别计算：

- A10−A00：固定条件样本对的增量；
- A20−A10：给定相同辅助目标，条件历史更新包的增量；
- A30−A10：动态条件抽样的增量。

阶段决定只用 Validation；历史已暴露 Test 作为历史线索保留，不用于参数选择。A20−A10 正均值触发 A21/A22 简单历史对照，两个数据集各自保留结论，不要求二者同时正才继续。A10 有收益而 A20 没有独立增量时，保留条件监督结果，停止把本版作为条件动量贡献推进。

### S3：方向 B 核心机制

对两数据、五种子，流式捕获 E5/E20/E35/E50 最后一个真实 batch 更新前的状态，共 40 个锚点。执行真实六格梯度分解、JOINT/ZERO_DATA/ONLY/LOO、等尺度读出、两组身份无关的随机损失项分组、EMA 对照。达到短段触发条件时，只从训练面板选出源，进行一次 LOO 脉冲及后续正常联合更新。

模型深拷贝与 optimizer.load_state_dict 必须复用真实 per-parameter step 和 None mask；主训练轨迹、RNG、loader 和 EMA 不受诊断污染。B 的单步、短段、完整训练结论分别登记。

### S4：有证据才创建候选

- A 历史增量成立：比较 A21 单份辅助历史、A22 单份较长 Adam 历史；接近时采用更简单者。
- A 有正增量且六格受损：先搜索原 λ=.05 与 λ=.025，新增λ同时配普通Adam；仍有已定位的共享子组代价，才运行两种历史作用域限制。
- B 发现真实且有训练期预测信息的跨格更新损害：运行 B_ADAPT、B_STATIC、B_UNIFORM，η∈{.25,.5}，每配置两数据五种子。
- A、B 均有独立收益且操作位置/信息互补：只运行一套冻结联合配置；否则不组合。
- 进入最终候选后，才运行冻结 Gaussian D0 背景下的组件增量补充；增强控制身份不匹配则暂停该补充表，不借用不同覆盖。

### S5：冻结、Test 与完整交付

冻结以下内容：候选方法、所有超参数、主比较、同 seed 对照、selected epoch 和模型 hash、六格保护参照、统计脚本版本。随后一次性评价冻结的最终主表，连同必要组件对照一并输出；不根据 Test 重新调参、改作用域或启用新候选。

已暴露历史 Test 的来源必须标明。新训练仍用同一历史 Test 人口，因此最终属于冻结配置下的历史数据集比较，不声称获得新用户/新时间窗口确认。

最终交付包括完整运行身份、数表、阶段决定、机制报告、性能表、成本与失败恢复记录；更新六个研究上下文/台账对应条目。运行结果按 start.md 的新日期实验分支规则发布；若用户明确在已有执行分支启动，则回到该分支。此次规划本身提交 main。

## 5. 命令入口

以下为本轮**待实现 CLI 合约**，不是仓库已有脚本；接口实现并经审查后可直接运行：

~~~bash
PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
PLAN=research_plans/26-11-22_conditional_history_shared_update
$PY -m experiments.conditional_history_shared_update_20261122.cli audit --plan "$PLAN"
$PY -m experiments.conditional_history_shared_update_20261122.cli check --plan "$PLAN"
$PY -m experiments.conditional_history_shared_update_20261122.cli run --plan "$PLAN" --stage A_CORE
$PY -m experiments.conditional_history_shared_update_20261122.cli analyze --plan "$PLAN" --stage A_CORE
$PY -m experiments.conditional_history_shared_update_20261122.cli run --plan "$PLAN" --stage B_CORE
$PY -m experiments.conditional_history_shared_update_20261122.cli advance --plan "$PLAN"
$PY -m experiments.conditional_history_shared_update_20261122.cli freeze --plan "$PLAN"
$PY -m experiments.conditional_history_shared_update_20261122.cli evaluate --plan "$PLAN"
$PY -m experiments.conditional_history_shared_update_20261122.cli report --plan "$PLAN"
~~~

run/advance 必须在当前前台会话受监督运行，advance 只展开本目录已定义分支，不自行设计下一轮研究；禁止 detached AI/旧自动循环 driver。freeze 在矩阵已结案、审查通过和状态可用时执行。用户停止时结束当前原子操作、保全 RAM 和已提交标量，报告 USER_STOPPED；不把未完成实验登记为负结果。
