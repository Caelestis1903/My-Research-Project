# MDMTRec 历史训练信息：机制研究与算法实验规划

**研究决策：继续历史信息路线，以“历史方向如何经过共享优化器状态转化为有效排序更新”为主问题，推进两个可完整训练的候选。** 固定参数 EMA 的收益及主要作用位置作为既有事实；A21 提供辅助来源历史的积极线索。本轮考察来源二阶耦合和历史预条件顺序，直接连接机制解释、算法定义与完整性能验证。

- 交付日期：2026-11-24。
- 证据基点：`main@358d3a465747eda6af1d55acc937759fa651a10b`。
- 当前交付状态：研究、文献、理论、规划及历史端点重分析完成；模型训练由后续 Codex 执行任务启动。
- 执行入口：[00_EXECUTION_RUNBOOK.md](00_EXECUTION_RUNBOOK.md)。机器清单：[experiment_manifest.json](experiment_manifest.json)。

## 核心方案

| 路线 | 改变的具体运算 | 对应问题 | 默认完整训练 |
| --- | --- | --- | --- |
| C1 来源二阶耦合调节 | 保留 A21 的 BCE/PAIR 一阶历史，插值 mixed 平方与来源平方和 | 混合二阶矩是否改变了有用辅助历史的有效步长？ | ζ=.5、1；两数据×五 seed×50 epoch |
| C2 预条件辅助历史 | 在每个历史时刻先使用当时的 Adam 分母，再累积辅助方向 | 当前预条件器是否重新扭曲了此前形成的辅助方向？ | α=.5、1；两数据×五 seed×50 epoch |

两个零点均为原 A21。两条路线共享原 PAIR 监督、自然数据、S_DENSE 范围和优化预算。它们都有完整训练资格，局部诊断负责解释和选择对照。每条轨迹同时维护 online 与非反馈 fixed EMA；两个读出只计一条训练轨迹。

本次已利用旧端点补算 Validation 的 A21×EMA 交互：[逐 seed 差](evidence/validation_interaction_seed.csv)、[汇总](evidence/validation_interaction_summary.csv)。[复算脚本与说明](analysis/README.md)可以从固定原CSV重现这两份表。这项重分析不消耗新训练，核心发现见 [01 §3](01_EVIDENCE_AND_RESEARCH_DECISION.md#3-本次已完成的历史端点重分析)。

## 文件导航

| 文件 | 用途 |
| --- | --- |
| [00 执行手册](00_EXECUTION_RUNBOOK.md) | Codex 顺序、阶段输入输出、命令合约、停止与恢复 |
| [01 证据与研究决策](01_EVIDENCE_AND_RESEARCH_DECISION.md) | EXP001–081 相关结论、数值、重复边界、当前研究选择 |
| [02 文献与理论](02_LITERATURE_AND_THEORY.md) | 原始文献综合、数学推导、可迁移的机制 |
| [03 竞争假设与优先级](03_HYPOTHESES_AND_PRIORITIES.md) | 科学问题、可观测预期、鉴别方式与算法后果 |
| [04 机制诊断](04_MECHANISM_DIAGNOSTICS.md) | 同状态干预、时间运算、MDMT 排序证据、复用方案 |
| [05 算法定义](05_CANDIDATE_ALGORITHMS.md) | C1/C2、控制、时钟、伪代码及退化关系 |
| [06 完整矩阵](06_EXPERIMENT_MATRIX.md) | 必做/条件臂、搜索、训练数量、读出与比较 |
| [07 实现规范](07_IMPLEMENTATION_SPEC.md) | 对接现有代码、数据/状态/RNG/参数支持与最小验收 |
| [08 决策规则](08_DECISION_RULES.md) | 筛选、简化、修订、论文资格与 Test 冻结 |
| [09 指标与统计](09_METRICS_AND_STATISTICS.md) | 配对 seed、六格、AUC/LogLoss、交互与统计口径 |
| [10 资源与调度](10_RESOURCES_AND_SCHEDULER.md) | GPU/RAM 准入、状态保存、成本与恢复 |
| [11 来源与复用](11_SOURCE_AND_REUSE_REGISTER.md) | 固定提交、实现路径、数表、真实状态可用性 |
| [12 报告模板](12_REPORT_TEMPLATES.md) | 阶段报告、完整报告、决策与上下文更新 |
| [13 交付审查](13_PLAN_REVIEW.md) | 本规划的独立审查、修改记录和静态验收 |

## 预算与研究终点

默认新增科学轨迹 **40 条**；B0/B1/H0 的状态与遥测按需复用或重建，最多另外 **30 条**。默认物理上界 **70 条、3,500 epoch、120,750 次主更新**。条件模块包括尺度/短历史/近邻方法控制、候选组合、作用域、HyperGate 迁移和 Gaussian 组合；全部触发时为 **240 条**（含最多30条父轨迹重建），详细计数见06。历史正式模型身份若不匹配，最多另列20条 HyperGate/AutoIFS 基线恢复，计入修复成本。

完成本轮后给出以下之一：`METHOD_READY`、`CONDITIONAL_METHOD`、`EMA_DEPENDENT_COMPONENT`、`GENERAL_OPTIMIZER_COMPONENT`、`AUXILIARY_ONLY` 或 `NO_ACTIONABLE_INCREMENT`。结论按数据集分别形成。已有 EMA 经验收益持续保留；一个新候选失败结束的是该候选的本轮定义。
