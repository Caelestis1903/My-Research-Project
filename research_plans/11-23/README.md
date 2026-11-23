# 11-23｜MDMT 共享表示—域任务专有路径协调更新

**状态：DESIGN_ONLY；本目录只授权 Codex 执行本方案定义的有限实验。** 本方案不代表算法已经实现、实验已经启动或任何候选已经取得收益。

## 核心研究问题

在 M3oE 的多域多任务计算图中，共享表示更新会同时改变多个域任务预测路径的输入；不同域任务路径可能对该变化产生不同响应。当前联合 Adam 更新没有显式区分“共享表示变化”与“对应专有路径在变化后完成适配”的贡献。

本研究检验：**共享表示更新之后，域任务专有预测路径的适配是否带来额外排序收益；若存在，是否能将其组织成有独立增量的 MDMT 协调更新算法。**

核心概念使用“共享表示—域任务专有预测路径响应失配”，不预设“适配滞后”已成立。专有路径不是六个独立完整模型头：M3oE 具有共享专家、域专家、任务专家、每个域任务对的 Gate 和 Tower。

## 文件导航

| 文件 | 用途 |
|---|---|
| [00_EXECUTION_RUNBOOK.md](00_EXECUTION_RUNBOOK.md) | Codex 执行顺序、阶段门、终止条件 |
| [01_RESEARCH_CONTEXT.md](01_RESEARCH_CONTEXT.md) | 研究背景、历史实验边界、来源入口 |
| [02_THEORY_AND_HYPOTHESES.md](02_THEORY_AND_HYPOTHESES.md) | 核心假设、数学对象、竞争解释 |
| [03_CAUSAL_EXPERIMENT_DESIGN.md](03_CAUSAL_EXPERIMENT_DESIGN.md) | 共享更新与专有路径适配的受控干预 |
| [04_EXPERIMENT_MATRIX.md](04_EXPERIMENT_MATRIX.md) | 完整训练矩阵、种子、数据、臂定义 |
| [05_IMPLEMENTATION_SPEC.md](05_IMPLEMENTATION_SPEC.md) | M3oE 参数分组、优化器状态、训练接口 |
| [06_SCHEDULER_AND_RESOURCES.md](06_SCHEDULER_AND_RESOURCES.md) | 资源、恢复、并发、成本 |
| [07_METRICS_AND_STATISTICS.md](07_METRICS_AND_STATISTICS.md) | 指标、选点、配对统计、性能门 |
| [08_REPORT_TEMPLATES.md](08_REPORT_TEMPLATES.md) | 报告、决策、交付物模板 |
| [09_CODEX_HANDOFF.md](09_CODEX_HANDOFF.md) | Codex 必须遵守的工程任务合同 |

## 唯一主方向

Domain-Task Response-aware Shared–Private Coordination（域任务响应感知的共享—专有协调更新）。

不并行启动 D1/D3/D4；不重新研究固定 EMA；不将 Gaussian、PAIR、来源关系网络或自适应 rho 加入本轮主方法。

## 固定实验边界

- 数据：MovieLens、KuaiRand 自然人口。
- 模型：M3oE fixed1，3 domains × 2 tasks。
- Seeds：严格为 2022、2023、2024、2025、2026。
- 训练：50 epochs、batch 40960、FP32、自然尾批。
- Optimizer：沿用当前已审查 M3oE 协议的 Adam 配置；不得在实现时自行改变。
- 读出：online 与固定 EMA 分开报告；EMA 是内部强参照/读出，不是本轮新算法。
- 选点：Validation 六格 Macro AUC 最早严格最大值。
- Test：所有方法、超参和 selected checkpoint 冻结后一次性评价。
- 正式模型基线：M3oE、HyperGate、AutoIFS；内部控制不得称为正式发表基线。
- 不允许将小域视为困难域。

## 执行边界

本方案是有限 DAG，不是无限循环。Stage 0、Stage 1 必须完成；Stage 2 仅在 Stage 1 的预注册机制门通过后执行；Stage 3 仅在 Stage 2 的验证门通过后执行。任一阶段触发 NO-GO，保存完整结果与决策报告后结束本目录任务，不自动发明新算法或追加搜索。
