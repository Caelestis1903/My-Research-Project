# 26-11-28 Three Directions — Codex Execution Entry

本目录是 2026-11-28 MDMTRec 三方向深化研究的完整执行合同。先读 `00_MASTER_PLAN.md`，再按以下顺序读取：

1. `01_RESEARCH_BACKGROUND_AND_LITERATURE.md` — 现有实验事实与相关文献。
2. `02_ALGORITHM_DESIGN.md` — A/B/C 算法公式、变量、更新顺序和控制。
3. `03_EXPERIMENT_MATRIX.md` — 实验矩阵、复用、阶段触发和停止条件。
4. `04_CODEX_IMPLEMENTATION_SPEC.md` — 真实代码接口、工程模块和测试要求。
5. `05_EXECUTION_RUNBOOK.md` — 运行队列、资源、恢复、Test 授权。
6. `06_METRICS_AND_DECISION_RULES.md` — selected checkpoint、统计、AUC 门槛和决策规则。

## 决策摘要

- A：先比较 clean pairwise 与 Gaussian pairwise，复用现有 PAIR sampler 和 GA31-R3 原始 batch RNG identity。
- B：在 A21 collapsed-history 原实现上加入同 batch、相邻共享参数状态梯度差校正；必须严格保持 A21 的 `w/q`、pair clock、λ 和 optimizer injection 公式。
- C：先做 selected checkpoint 历史正确排序关系离线审计，只有发现 selected 前可重复的关系遗忘且具有独立信息时才训练 memory。
- A/B/C 独立评价，只有两个机制分别通过后才允许一个组合。
- 不修改模型结构，不启动任何训练。本目录只定义 Codex 后续执行工作。

## 来源状态

- 工作基准为 main。
- EXP082 数值是用户本次任务文本提供的未合并补充证据，不声称其已在 main。
- 任何源代码、数据、checkpoint 或 RNG identity 无法核实，均按对应阶段 PAUSE，不得猜测补齐。
