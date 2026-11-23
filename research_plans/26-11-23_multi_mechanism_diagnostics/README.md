# 多机制联合诊断实验规划

## 研究定位

本目录把第一阶段开放式机制发现报告转化为 Codex 可执行的实验计划。研究主线是：**共享更新如何改变域任务专有参数对共享梯度的塑形作用，以及这种条件性变化能否被样本—任务影响信号解释。**

本计划不是固定 U→P / P→U 顺序实验的复刻，不重新证明固定 EMA，不以小域作为困难度代理，也不把局部诊断当作完整算法性能结论。

## 关键来源身份与差异声明

- 本轮实际第一阶段报告：research_reports/2026-11-24_open_mechanism_discovery/FINAL_RESEARCH_REPORT.md。
- 第二阶段提示词指定的 research_reports/26-11-23_multi_mechanism_discovery/00_RESEARCH_REPORT.md 在当前 main 未找到；其三个支持文件 01_LITERATURE_SEARCH_AUDIT.md、02_CANDIDATE_MECHANISM_REGISTER.csv、03_SOURCE_AND_HISTORY_MAPPING.md 也未找到。
- 因此本方案以当前 main 已实际交付的第一阶段报告为研究输入，不伪称读取了不存在的文件。第一阶段支持文件不作为必要前提；缺失情况记录在 10_SOURCE_AND_REUSE_REGISTER.md。
- 最新常驻上下文为 experiments/reports/EXP_AI_CONTEXT_V2.md（blob 2dd249cceee10b4d51391d3a3c27aef296a5a283，当前 main 研究状态基准）。
- D2 实验来源为分支 26-11-23_v2；其相对 main@f66bf17cba2980618e6f2edd1c7381b73c35cac5 ahead 1 / behind 0。S1 已完成并 NO-GO；不得重复固定 scope / 固定顺序动作。

## 本次纳入的机制

| ID | 处理 | 本轮角色 |
|---|---|---|
| M1 共享梯度一致性塑形 | 纳入 | 主候选，第二阶私有参数诊断 |
| M2 样本×任务条件影响 | 纳入 | 独立机制诊断，与 M1 共用状态与样本面板 |
| M3 阶段性/组合式影响 | 有条件纳入 | 仅对 M2 有明确异质性信号的有限样本组执行 |
| M4 域任务条件有害特征依赖 | 仅可行性审计 | 不训练、不做因果主张 |
| M5 语义合法多标签增强 | 仅可行性审计 | 无合法变换定义则停止 |
| M6 输出响应敏感度 | 辅助测量 | 不作为独立候选 |
| M7 跨 seed 分布归因 | 统计策略 | 不另建算法 |
| M8 通用任务冲突控制器 | 排除 | 与既有投影/任务选择工作重叠 |

## 执行顺序

1. S0：来源、协议、参数分组、数据语义、计算资源审计；不训练。
2. S1：建立或精确恢复五 seed × 两数据集基线轨迹；E10/E25/E40 锚点；执行 M1/M2 联合筛选和有限 M3 交互诊断。
3. S2：使用独立 Validation panel B 对少数冻结候选确认；不使用 Test。
4. S3：仅对通过 S2 的候选做 5-epoch 连续短程轨迹；若无候选通过则停止。
5. S4：只有在机制证据明确且形成独立算法规格、经独立审查后，才进入五 seed、两数据集、50 epoch 正式算法训练。S4 是条件阶段，不允许 Codex 自行发明未冻结的算法。
6. M4/M5 的可行性审计结果可单独触发后续新方案，但不自动并入本轮训练。

## 文件导航

- 00_EXECUTION_RUNBOOK.md：状态机、阶段门、运行顺序和恢复。
- 01_RESEARCH_CONTEXT.md：第一阶段结论、最新上下文与 D2 事实。
- 02_MECHANISM_SELECTION_AND_HYPOTHESES.md：纳入/排除机制及数学定义。
- 03_SHARED_INFRASTRUCTURE_AND_STATE_SNAPSHOT.md：基线、状态快照和隔离。
- 04_EXPERIMENT_MATRIX.md：实验组、对照、种子、锚点和预算。
- 05_IMPLEMENTATION_SPEC.md：模块、接口、伪代码和测试。
- 06_METRICS_AND_STATISTICS.md：指标、统计和多重比较。
- 07_SCHEDULER_AND_RESOURCES.md：资源准入、成本、重试与恢复。
- 08_DECISION_RULES.md：PASS / NO_GO / TECHNICAL_BLOCKED / INCONCLUSIVE。
- 09_REPORT_TEMPLATES.md：机器可读与人类报告模板。
- 10_SOURCE_AND_REUSE_REGISTER.md：来源身份、复用和去重。
- 11_PLAN_REVIEW.md：独立方案审查及实际修正。

## Codex 执行边界

本方案授权执行 S0–S3 诊断流程。S4 完整算法训练仅在 S3 通过且独立算法规格审查完成后触发；S4 的算法定义、超参数、基线和 Test freeze 必须在新的阶段门记录中冻结。不得把本目录当作对 Test 的预先授权。
