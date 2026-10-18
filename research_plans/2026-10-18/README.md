# 2026-10-18｜MDMT 前沿创新研究方案

本目录汇集 **2026 年 10 月 18 日** 对 My-Research-Project 的证据审查、近年论文排重与六个独立创新候选。本研究面向推荐系统与机器学习顶会/顶刊，不以硕士论文工作量为准。

## 文件

- [MDMT_FRONTIER_RESEARCH_REPORT.md](./MDMT_FRONTIER_RESEARCH_REPORT.md)：完整研究报告。包含仓库证据与反例、论文批判、六个候选方案、三个重点方案的数学设计、必要实验决策树、参考文献 60 项及出处。此文件由同日 39 页研究报告转为适合 GitHub 阅读的 Markdown，保留正文、表格、公式与文献链接。
- [START_RESEARCH_PROMPT.md](./START_RESEARCH_PROMPT.md)：面向 Codex/研究执行代理的**可直接复制**的启动指令；要求六个方向全部进行实质性尝试。
- [DIRECTION_TRACKER.md](./DIRECTION_TRACKER.md)：六方向完成标准与研究结论登记模板。

## 六项正式研究方向（全部必做独立判定）

| ID | 问题 | 关键排重对象 |
|---|---|---|
| P1 | 有限支持下的域任务条件稳定性与信息保留 | SAF、DVD |
| P2 | 跨域任务约束能否增加缺失标签/迁移决策的可识别信息 | ExtraDebias、WCP-DR、HiDe-ESMM、Feasible Fusion |
| P3 | 固定部署预算下共享模型能否同时满足各 cell 收益保护 | DSMTL、AutoIFS、压缩/低秩方法 |
| P4 | 比完整 DT 展开更低的测量成本能否恢复决策相关交互 | 全 cell 梯度、ANOVA、低秩与 task affinity |
| P5 | gate/specificity 代理的函数等价、解释语义和决策影响 | D3、常规表示不变性/验证效用 |
| P6 | 成本约束下的域任务联合采样与方向风险 | EMAIS/EMS、MIS/OMIS、经典 importance sampling |

前三项可优先深化，但 **P4–P6 不得以优先级低为由省略**。所有方向至少完成：相关工作排重、问题可证伪化、理论或已有数据检验、一个能够改变研究决策的最小关键检验、结论与原始证据归档。没有现实可行的训练对象，或已被理论/文献直接覆盖时，可以以证据化 NO-GO 结束，不强制浪费 GPU。

## 历史约束

- “小域≠困难域”为研究前提；不得重新设计大规模实验来否定“小域天然困难”。
- Adam 不是同目标采样效应的普遍必要条件；S5 Equal 历史结果与 N17 Sample/q 结果不可混写。
- 不把 N17 selected 主比较阴性写成“已证平衡采样总是有害”，不把 N4 未触发当成 N4 已执行。
- 不预设重加权、梯度修正、动量或课程有效；不把 DBO-inspired 与真实 D3，magnitude-only 与完整 GradCraft 混同。
- 只为科学结论、算法正确性、方法有效性与创新覆盖做必要检查，拒绝冗余 hash/测试/无限种子或 Epoch。

## 原始证据与执行

报告定格 main `98247ebf1acaf6f9cfbb823441391c1c83bbfb05` 和执行分支 `codex/execute-2026-10-17-run@5550afbc57f76e0b94c5783db16f4c1796933541`；历史评估含重复使用的测试集，新的方法结论应冻结独立确认规则。按 [START_RESEARCH_PROMPT.md](./START_RESEARCH_PROMPT.md) 启动执行；不把此目录视为已经完成的新训练或已经证明有效的算法。

**文献与仓库证据记录的是截至 2026-10-18 的快照，正式研究过程中须核查直接竞争的最新工作，但无需无边界扩展文献检索。**
