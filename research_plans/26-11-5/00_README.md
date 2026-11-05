# 域任务增强与参数历史聚合：A/B/C 完整交付

日期：2026-11-05。状态：**DESIGN_COMPLETE / IMPLEMENTATION_NOT_STARTED / TRAINING_NOT_RUN**。

## 本次任务及优先级

本目录将三条路线一次性设计完成，交给 Codex 实现、验收与分阶段运行。B 优先实验；A、C 是完整的独立方案，不是 TODO，也不因 B 优先而删除。本次只提交规划、公式、接口、配置和命令合同，不提交正式算法实现、不运行正式训练。

用户补充优先于 `prompt(10).md` 中冲突条款：

1. 全部规划和实现方案直接提交 **main / research_plans/26-11-5**；不使用上传文件中的旧规划分支和旧目录。
2. 每条被执行的路线都必须对其新增有效超参数做有限搜索。**每一个参加优劣比较的候选配置，都使用相同的五个 seed 完整结果；禁止单 seed/双 seed 预筛。**目标是找到已比较集合中的较优值，不声称全局最优。
3. 上传文件后半部分重复的旧“重新探索三至四路线”文字作为背景；实际交付以其前半部分明确的 A/B/C 三路线工程化任务为准，不重新发散或替换路线。

## 阅读入口

直接交给 Codex：[07_FINAL_CODEX_ENTRY.md](07_FINAL_CODEX_ENTRY.md)。

| 文件 | 用途 |
|---|---|
| [01_RESEARCH_CONTEXT.md](01_RESEARCH_CONTEXT.md) | 固定证据、实际代码语义、已知结论与新假设 |
| [02_THREE_ROUTE_COMPARISON.md](02_THREE_ROUTE_COMPARISON.md) | 机制、假设与简单替代 |
| `route_A/01...04` | 实际路径分组 EMA：数学、接口、搜索与运行 |
| `route_B/01...04` | 增强响应定向 EMA：数学、接口、搜索与运行 |
| `route_C/01...04` | 历史漂移条件增强：数学、接口、搜索与运行 |
| [03_STAGED_EXECUTION_PLAN.md](03_STAGED_EXECUTION_PLAN.md) | B→决策→A/C 的实际阶段顺序 |
| [04_DECISION_TREE.md](04_DECISION_TREE.md) | 有限状态决策及机器输出 |
| [05_SHARED_EVALUATION_PROTOCOL.md](05_SHARED_EVALUATION_PROTOCOL.md) | 选点、调参、五 seed、六格保护、test 冻结 |
| [06_RESOURCE_AND_REUSE_PLAN.md](06_RESOURCE_AND_REUSE_PLAN.md) | 轨迹、估计器、RAM、评价与重放预算 |
| [08_HYPERPARAMETER_SEARCH.md](08_HYPERPARAMETER_SEARCH.md) | 每路线候选值、顺序、停止条件 |
| [09_SHARED_ENGINEERING_CONTRACT.md](09_SHARED_ENGINEERING_CONTRACT.md) | 数据、RNG、状态、恢复、接口和命令合同 |
| [10_SOURCE_INDEX.json](10_SOURCE_INDEX.json) | 固定提交和源码/报告路径 |
| [11_CONSISTENCY_REVIEW.md](11_CONSISTENCY_REVIEW.md) | 交付审查、已修订问题、未测项 |
| [12_RELATED_WORK_AND_DERIVATION.md](12_RELATED_WORK_AND_DERIVATION.md) | 原论文启发及本次推导边界 |
| [experiment_manifest.json](experiment_manifest.json) | 全部候选、依赖、seed、搜索预算与阶段规则 |

## 三条路线都已给出确定算法

A：按真实梯度路径形成 12 个活动参数组；主要版本用只读训练探针更新各组半衰期，配套固定分组与全局自适应替代。在线训练仍为原 Gaussian+Adam。

B：从同一个旧 Adam 状态计算各增强项的边际参数位移，形成子空间；子空间内外使用不同 EMA 半衰期。在线训练不变，首轮运行。

C：在线模型与固定 EMA 的输入响应差生成任务条件协方差；保持每字段期望扰动能量不变，比较抑制/增强漂移方向。它会改变在线训练，必须独立训练。

## 预算摘要（两数据集均执行时）

B：10 条在线训练，15 个影子估计器/轨迹，150 个估计器-seed 实例。

A：独立运行同样 10 条在线训练、15 个影子估计器/轨迹。默认在 B 完成后触发，因此通常需要重放 10 条；不能用 B 最后一个 checkpoint 生成 A 的历史。

C：4 个有限搜索候选+2 个匹配机制对照，每配置每数据集五 seed，共 **60 条新轨迹**；普通 Gaussian 内部对照优先复用 B，完全独立执行且无合法结果时另加 10 条，共 70 条。

默认最低 10 条（仅 B）；B+A 为 20 条；B+C 为 70 条；A/B/C 全部顺序执行为 80 条。没有预设三条必须全跑。

## 继承边界

小域不是困难域；本目录不安排证明或否定这一结论的实验。不按样本量设平滑或增强强度。五 seed 不要求全部正向；正式 baseline 仅 M3oE、HyperGate、AutoIFS 等已发表模型。Gaussian+EMA 和 A/B/C 都是内部方法或控制。没有测试结果的参数只称候选/工程默认值，不能称“已找到较优值”。
