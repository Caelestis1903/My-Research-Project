# 10 — 来源、历史复用与去重登记

## 1. 第一阶段报告

实际存在：
research_reports/2026-11-24_open_mechanism_discovery/FINAL_RESEARCH_REPORT.md

第二阶段提示词指定的 research_reports/26-11-23_multi_mechanism_discovery/00_RESEARCH_REPORT.md 及 01_LITERATURE_SEARCH_AUDIT.md、02_CANDIDATE_MECHANISM_REGISTER.csv、03_SOURCE_AND_HISTORY_MAPPING.md 在当前 main 未找到。记录 SOURCE_PATH_MISMATCH，不得伪造读取收据。

## 2. 最新 main context

experiments/reports/EXP_AI_CONTEXT_V2.md。
其核心 C1–C14、A1–A24 作为当前事实来源。特别遵守 Equal/Sample objective contingency、S6 STOP、小域不是困难域、固定 EMA 不重证。

## 3. D2

Branch 26-11-23_v2；base main@f66bf17cba2980618e6f2edd1c7381b73c35cac5。
读取：
- research_reports/mdmt_shared_private_coordination_20261123/DESIGN_SPECIFICATION.md
- DESIGN_ERRATUM.md
- artifacts/mdmt_shared_private_coordination_20261123/S0/SOURCE_MANIFEST.json
- DATA_MANIFEST.json
- PARAMETER_MANIFEST.json
- MODEL_FIDELITY.json
- S1_RUN.json
- SCOPE_DECISION.json
- ANCHOR_INDEX.json

D2 S1 的 10 trajectories / 30 anchors / 570 forks / 0 Test evaluation 可核验。D2 RAM state 必须 live readback。

## 4. 历史研究覆盖

历史实验包括但不限于：
- Equal/Sample 与域目标对照；
- 固定 EMA、Gaussian+EMA、adaptive rho；
- PCGrad、GradCraft；
- 样本权重、quota、采样顺序、curriculum；
- task-view / ranking objective；
- conditional history A/B；
- D2 fixed U/P scope 与更新顺序。

每个新 variant 在 RUN_MANIFEST 中登记历史重叠标签：
EXACT_REUSE_CONTROL、RELATED_BUT_DIFFERENT、NEW_MECHANISM、NOT_COMPARABLE。
不得把相同轨迹在多个 context files 的重复记录当作独立证据。

## 5. 文献映射

M1：ConsMTL CVPR 2025，通用 MTL 机制先例。
M2：DTJS ICDE 2025，数据与任务联合调度先例。
M3：Interaction-Aware Influence Functions ICLR 2026；Influence Dynamics and Stagewise Data Attribution ICLR 2026。
M4：MT-CCL TNNLS 2025、SAF、ORCA 相关依赖研究。
M5：Taesar WWW 2026 及数据增强/语义保持相关研究。
M6：输出敏感度、局部 Jacobian 与 trust-region 机制。
M7：Distributional Training Data Attribution NeurIPS 2025。
M8：MTCCM 2026 Web API recommendation；PCGrad/任务冲突控制作为相关先例。

跨领域方法只能作为机制先例，不能写成已在 MDMTRec 证明有效。

## 6. 复用规则

可复用 D2 fixed1 adapter、数据 identity、state serialization、metrics、scheduler 的纯函数，必须记录 blob/hash 并重新审查。不得把 D2 S1 的 NO_GO 当作本轮新机制证据。不得复用历史 Test 结果作为当前新候选的 Test。
