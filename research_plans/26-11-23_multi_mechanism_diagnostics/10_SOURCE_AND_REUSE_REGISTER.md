# 10 — 来源、文献与历史复用登记

## 1. 第一阶段报告

实际存在：
research_reports/2026-11-24_open_mechanism_discovery/FINAL_RESEARCH_REPORT.md

第二阶段提示词指定的：
research_reports/26-11-23_multi_mechanism_discovery/00_RESEARCH_REPORT.md
及 01_LITERATURE_SEARCH_AUDIT.md、02_CANDIDATE_MECHANISM_REGISTER.csv、03_SOURCE_AND_HISTORY_MAPPING.md

在当前 main 文件检索中未找到。Codex 必须将其记录为 SOURCE_PATH_MISMATCH，不得伪造读取收据。若执行环境中用户另行提供该文件，需先比较内容身份并通过 reviewer 决定是否替代。

## 2. 最新 main context

- experiments/reports/EXP_AI_CONTEXT_V2.md，blob 2dd249cceee10b4d51391d3a3c27aef296a5a283。
- 当前研究状态核心 C1–C14、A1–A24。
- 重点：Equal/Sample objective contingency、Sample 下 S6 STOP、固定 EMA 不重证、小域不是困难域。

## 3. D2 来源

Branch: 26-11-23_v2
Main base: f66bf17cba2980618e6f2edd1c7381b73c35cac5
D2 S1: COMPLETE_NO_GO_S1
D2 source:
- research_reports/mdmt_shared_private_coordination_20261123/DESIGN_SPECIFICATION.md
- research_reports/mdmt_shared_private_coordination_20261123/DESIGN_ERRATUM.md
- artifacts/mdmt_shared_private_coordination_20261123/S0/SOURCE_MANIFEST.json
- artifacts/mdmt_shared_private_coordination_20261123/S0/DATA_MANIFEST.json
- artifacts/mdmt_shared_private_coordination_20261123/S0/PARAMETER_MANIFEST.json
- artifacts/mdmt_shared_private_coordination_20261123/S0/MODEL_FIDELITY.json
- artifacts/mdmt_shared_private_coordination_20261123/S1_RUN.json
- artifacts/mdmt_shared_private_coordination_20261123/SCOPE_DECISION.json
- artifacts/mdmt_shared_private_coordination_20261123/ANCHOR_INDEX.json

D2 S1 的 10 trajectories / 30 anchors / 570 forks / 0 Test evaluation 可核验。D2 当前状态快照为 RAM guardian receipts；本轮执行时必须实时读回，不能假定远端仍保留。

## 4. 相关历史研究

- conditional_history_shared_update_20261122：A/B history study；A21 未获得两个数据集的最终算法资格；Gaussian supplement 因噪声流 identity mismatch 技术暂停。
- FINAL_REVIEW.md：对 90 selected endpoints、五 seed pairing、Test freeze 和计算表的独立 PASS；不授权追加实验。
- 该研究的历史收益与当前 M1/M2 研究不同，不得将 A21 当作 M1/M2 baseline 或新发表基线。

## 5. 文献映射

M1：ConsMTL CVPR 2025，通用 MTL；只提供机制先例，不直接等同推荐排序方法。
M2：DTJS ICDE 2025，通用 MTL；样本任务联合调度先例。
M4：MT-CCL TNNLS 2025；只作为条件特征依赖研究先例。
M7：Distributional Training Data Attribution NeurIPS 2025；统计归因启发。
M3：Interaction-Aware Influence Functions ICLR 2026；组交互先例；Influence Dynamics and Stagewise Data Attribution ICLR 2026；阶段归因先例。
M8：MTCCM 2026 Web API recommendation；问题域相邻但不作为本轮候选。

## 6. 明确不复刻的历史方向

- 固定 EMA / adaptive rho / 分组 EMA 时间尺度；
- 通用梯度投影、PCGrad / GradCraft；
- 固定 U→P / P→U 与 D2 P scope sweep；
- 一般域权重、Equal/Sample、loss mass、quota、sampling schedule；
- 一般 easy-to-hard curriculum；
- 一般 HSIC / 去相关 / 因果对比；
- 仅换 ranking loss；
- 以小域身份或域大小定义困难度。

## 7. 复用规则

可以复用 D2 的纯函数、数据身份校验、fixed1 adapter、state serialization conventions、metrics and reporting utilities，前提是记录 source hash 并重新独立审查。不可复用 D2 的 arm 结果作为 M1/M2 证据，也不可将 D2 的 S1 NO_GO 改写为新机制 NO_GO。
