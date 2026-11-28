# MDMTRec 三方向深化研究与 Codex 执行总方案

> 版本：2026-11-28；基准分支：main；执行范围：研究方案、工程实现、训练、分析和上下文更新。本文档是执行入口。

## 0. 结论与决策

本轮将三个研究问题拆成不同层级，按 **A → B → C** 推进，但允许 A、B 的准备工作并行：

1. **A：排序目标导向的数据增强**进入第一优先级。核心不是再增加一种噪声，而是检验 Gaussian view 是否能向 M3oE 提供与 Macro AUC 对齐、且超出已有 BCE 增强的 pairwise 排序监督。先做最小 2×2 组件矩阵，再决定是否保留 Gaussian。
2. **B：PAIR 历史梯度的新鲜度校正**作为第二优先级。A21 的积极信号支持继续研究辅助 PAIR 梯度，但 EXP082 的 C1/C2 已结束，不能再围绕其超参数延伸。新的问题是：同一 PAIR 样本在相邻共享参数状态下的梯度差，能否为当前 PAIR 梯度提供可用校正。首选 MARS-inspired same-batch correction，严格称为本项目适配机制，不称为原版 MARS。
3. **C：历史有效排序关系记忆**先进行 selected checkpoint 离线可行性审计，再决定是否训练。先确认历史正确 pair 关系是否在 selected checkpoint 上真实存在、是否发生可量化的遗忘、是否与 selected AUC 有关；只有现成训练轨迹/selected 权重和合法 Train pair 可用时才进入小型记忆训练。C 不与 A/B 自动组合。
4. 独立文献探索发现两个值得记录的替代启发：AdEMAMix 的长短梯度历史混合；Backpropagated Output Momentum（BOM，2026 预印本）把历史信息保存在输出残差空间、再通过当前网络投影回参数空间。BOM 不是已确认的顶会正式论文，也不直接纳入首轮训练；作为 B/C 的后续替代研究候选。
5. 固定 EMA 保留为所有适用实验的强内部读出与对照。普通 EMA 是否有效不再重证。A21 与参数 EMA 是不同对象：前者是 PAIR 来源梯度历史进入优化过程，后者是完整模型参数的非反馈式读出。

## 1. 任务背景和现有证据

### 1.1 已确认的结果

- 固定参数 EMA 在两数据集的冻结训练轨迹上有稳定的 selected Test AUC 收益。EXP082 补充证据：80 条冻结轨迹中 77 条 EMA Test AUC 高于同轨迹 online；MovieLens 平均 EMA−online +0.004533，KuaiRand +0.004816。相对同轮 M3oE online，EMA 的 Test 增量分别为 +0.003849 和 +0.004616。该结果是固定 EMA 的强证据，后续不重复证明。
- G-D0 + fixed EMA 的 Test Macro AUC：MovieLens 0.779357（相对 M3oE +0.004167），KuaiRand 0.679180（相对 M3oE +0.007274）。MovieLens 六格保护通过，但 Macro LogLoss 增加 0.000931；KuaiRand 仍低于 HyperGate 0.002932，且有六格保护问题。Gaussian 的整体收益不等于其相对 clean+EMA 的独立增量。
- EXP082 的 A21 相对同 PAIR Adam + fixed EMA：MovieLens Test ΔAUC −0.000083；KuaiRand +0.004062，五 seed 均正，95% 配对区间约 [0.000469, 0.007656]。KuaiRand Validation ΔAUC −0.001989，故该 Test 不能被包装为已通过正常 Validation 选点的算法晋级。它提供的是值得保留的历史辅助梯度信号。
- EXP082 C1（来源二阶矩交叉项解耦）和 C2（历史时刻预条件方向累积）均已结束为 NO_ACTIONABLE_INCREMENT。C2 相对 B1+EMA 的 MovieLens Test 差 −0.000190，相对 H0 −0.000107497。不得继续做 ζ/α/β 无目标扫描。
- 自适应 EMA、参数组时间尺度、收益预测动作、梯度投影、任务视图残差历史、NQR、早晚剂量、轴向扰动、格质量和完成权重等已有大量负向或不稳定结果。新实验必须明确干预对象和最简单匹配替代，不重复这些问题。
- 小域不是困难域；域规模不得作为难度代理，也不安排任何检验该命题的实验。

### 1.2 main 读取审计

main 最新上下文 `experiments/reports/EXP_AI_CONTEXT_V2.md` 对先前 Equal-objective history-carrier / target-rescue 方法线给出 STOP METHOD LINE；禁止重启 m/v 拆分、ratio boundary 或该路线的方法开发。此次 A/B/C 不是该旧 Equal 路线的延伸：A 检验排序目标增强，B 只检验 EXP082 提供的 PAIR auxiliary gradient freshness（必须用 Sample objective 与当前 A21 实现身份），C 检验 Train-only 历史排序关系。不得将 C13/C14 的 Equal 历史机制结论当成 B 的正向证据；B 的唯一正向起点是用户本次提供的 EXP082 A21 对照结果。若无法证明 A21 PAIR 路径与旧 STOP line 在 objective、参数 scope、采样器上有明确区别，则暂停 B 并先做 source identity review。

main 实际存在：
- `mdmt_research_context/01_AI_EXPERIMENT_INDEX.md`
- `mdmt_research_context/02_AI_EXPERIMENT_DETAILS.md`
- `mdmt_research_context/03_EXPERIMENT_RESULTS.md`
- `mdmt_research_context/04_HUMAN_RESEARCH_REVIEW.md`（仓库 AGENTS 明确要求 AI 排除该文件；本方案执行时也排除）
- `experiments/reports/EXP_AI_CONTEXT_V2.md`
- `research_plans/26-11-24_history_information/` 全套规划
- `experiments/conditional_history_shared_update_20261122/` 与 `experiments/conditional_information_momentum_20261102_v3/` 代码。

提示词列出的 `EXP_AI_CONTEXT.md`、`MDMT_ALGORITHM_METRICS_LEDGER.md` 和独立 EXP082 文件未在当前 main 根目录找到。EXP082 数值仅以本任务输入中明确给出的内容作为外部补充，不声称其已合并至 main。当前 main 的 EXP 结果索引截至 EXP081；Codex 执行前需记录 main HEAD、文件 SHA 和新增提交状态。

### 1.3 核心科学问题

- A：增强视图是否改变正负样本的相对排序监督，并在 selected checkpoint 上产生超过 G-D0+EMA 的 AUC 增量？
- B：A21 保存的 PAIR 梯度在共享参数移动后是否仍有当前优化价值；用同一批 pair 在相邻共享参数状态下的梯度差校正，能否比 A21 进一步提高 selected AUC？
- C：历史训练中已形成的正确 pairwise 关系是否在后续共享训练中被遗忘；保留这些关系是否能改善 selected checkpoint，而不是仅仅增加样本曝光或重复当前 pairwise loss？

## 2. 阅读顺序和执行入口

Codex 必须先读：
1. `experiments/reports/EXP_AI_CONTEXT_V2.md`
2. `mdmt_research_context/01_AI_EXPERIMENT_INDEX.md`
3. `mdmt_research_context/02_AI_EXPERIMENT_DETAILS.md` 中 EXP069–081、EMA 归因、Gaussian、A21 相关段落
4. `mdmt_research_context/03_EXPERIMENT_RESULTS.md` 中对应结果表及原始来源索引
5. `research_plans/26-11-24_history_information/01_EVIDENCE_AND_RESEARCH_DECISION.md`、02、05、06、07、08、09、11、13
6. `experiments/conditional_history_shared_update_20261122/history_runner.py`、`history_controls.py`、`experiments/conditional_information_momentum_20261102_v3/objectives.py`、`optimizers.py`、`boundary_sampler.py`
7. `m3oe/MDMTRec/models/multi_domain/model.py`、`m3oe/MDMTRec/trainers/ctr_trainer.py`、`m3oe/examples/run_movielens.py`、`run_kuaishou.py`

禁止读取 main 中的 04 人类阅读文件；禁止扫描其他研究分支。EXP082 只用本方案开头所列补充数值，不把未合并代码当作可复用实现。

## 3. 总体阶段和依赖

统一训练协议：Sample/natural objective、M3oE fixed1、MovieLens-1M 与 KuaiRand-Pure、seeds 2022–2026、50 epoch、batch 40960、FP32。不得回退到 Equal objective，不得将 domain size 作为困难度或权重代理。所有新增 pairwise objective 复用 canonical `boundary_sampler.sample` 的 2048 pair、selected condition 与 w/q 重要性修正。

| 阶段 | 内容 | 训练预算 | 进入条件 | 通过后 |
|---|---|---:|---|---|
| S0 | 源码/数据/历史结果身份审计 | 0 | 立即 | 冻结数据、代码、pair sampling 和 baseline 身份 |
| A0 | A 方向 pair sampler、noise view、梯度剂量的固定小批工程检查 | 0 科学轨迹 | S0 通过 | 进入 A1 |
| A1 | clean/Gaussian × BCE/Pairwise 最小矩阵 | 每 arm×数据×5 seed×50 epoch | A0 通过 | 选 A 的父方法/候选 |
| A2 | A 的条件触发消融（pair-level independent/shared noise） | 最多 2 个候选×数据×5 seed×50 epoch | A1 至少一个候选对直接父方法 ΔAUC≥+0.001，六格保护通过 | 冻结 A 或关闭 |
| B0 | B 的相邻共享状态同 batch 梯度校正实现和代数测试 | 0 科学轨迹 | S0 通过；可与 A1 并行实现 | 进入 B1 |
| B1 | PAIR Adam、A21、MARS-inspired 校正及最简控制 | 每 arm×数据×5 seed×50 epoch | B0 通过；A21 可用训练实现已核对 | 选 B 候选 |
| B2 | 校正系数/是否只校正历史源的有限确认 | 最多 2 个候选×数据×5 seed×50 epoch | B1 对 A21 ΔAUC≥+0.001 且六格保护通过 | 冻结 B 或关闭 |
| C0 | selected checkpoint 的历史 pair 离线审计 | 0 新训练；最多现成模型 inference | 可取得合规 checkpoint / 预测缓存 | 决定 C1 是否启动 |
| C1 | 历史关系记忆训练 | 每 arm×数据×5 seed×50 epoch | C0 发现 selected 之前存在可重复的历史关系遗忘，且关系记忆与当前 pairwise 可区分 | 评估 C |
| S4 | A/B/C 单独通过后的组合 | 每组合×数据×5 seed×50 epoch | 至少两个机制分别通过改进门、存在明确互补假设 | 最多一个组合，不跑全组合 |
| S5 | 冻结、Test、总报告和上下文更新 | 0 新训练 | Validation 配置冻结并取得现有权限 | 完成 |

A 与 B 的算法实现和小批测试可以并行；正式训练队列按资源与 S0 审计结果调度。C 的离线审计可以并行，但 C1 训练依赖 C0。不得把 A+B+C 预先组合。

## 4. 最终决策标准

- 新机制相对原始 M3oE 的准入门槛：Δ selected Test Macro AUC ≥ +0.002。
- 对已有机制的改进相对直接父方法的门槛：Δ selected Test Macro AUC ≥ +0.001。
- 逐格平均 AUC 保护线：−0.001。
- LogLoss 在 AUC selected 的同一 checkpoint 上独立报告和裁定，不得另选 LL epoch。
- 五 seed 配对均值、标准差、95% 配对区间、正向 seed 数均报告；不强制五个 seed 全正。
- 外部竞争基线：M3oE、HyperGate、AutoIFS。每个数据集按已冻结的统一协议使用其最强正式发表基线。
- 任何 Test 只在全部 Validation 决策完成、候选配置冻结后执行。历史暴露 Test 必须保留历史身份标签。

## 5. 输出文件

本目录七个文件共同构成 Codex 执行合同：
- 01：背景与文献
- 02：算法数学定义
- 03：矩阵与阶段触发
- 04：工程实现规格
- 05：运行手册
- 06：指标和决策规则

Codex 只负责按文档实现、运行、分析和更新上下文。算法设计和科学门槛已在本方案冻结；如源码审计发现某个接口或数据列不存在，先写出具体阻塞及其对实验定义的影响，不得猜测替代字段或自行改变科学问题。

## 6. 交付要求

最终在 main 新增本目录所有文件，并更新必要索引（不覆盖历史证据）。执行完成后报告 commit、文件清单、阶段结果、候选保留/关闭决定、selected Test AUC/LogLoss、六格保护、配对统计、成本和下一步决策。
