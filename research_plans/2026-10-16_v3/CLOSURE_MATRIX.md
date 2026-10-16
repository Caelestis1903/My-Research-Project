# 2026-10-16 v3：问题—实验—结论闭环矩阵

本文件的目的，是防止执行者只完成“主故事”而遗漏旁支问题。

| ID | 当前已知 | v3 必做动作 | 终局允许状态 |
|---|---|---|---|
| Q1 | v2 已观察 SB-SN | V1 exact objective + V2 trajectory | RESOLVED / NULL |
| Q2 | selected 与 fixed epoch 不同 | V3 selection decomposition | RESOLVED / SELECTION-DEPENDENT |
| Q3 | SGD 也有负效应 | V4/V11 optimizer comparison | RESOLVED / INTERACTION |
| Q4 | Adam-specific decomposition 未触发 | 仅 interaction 显著时做 m/v/history | FALSIFIED / RESOLVED |
| Q5 | cyclic 同时改变多个因素 | V5 factorial controls | RESOLVED / MULTI-FACTOR |
| Q6 | C/R distance 较大 | V7 intervention | RESOLVED / ASSOCIATION |
| Q7 | 仅 3 domain，size association 不足 | V8 duplication manipulation | RESOLVED / BOUNDARY |
| Q8 | difficulty 与 domain 混杂 | V8 controlled corruption | RESOLVED / SENSITIVITY / NOT_IDENTIFIABLE |
| Q9 | 只有 6 domain-task cells | V8 cell audit + identifiability | RESOLVED / NOT_IDENTIFIABLE |
| Q10 | selection uncertainty 尚不完整 | V3 bootstrap + selection rules | RESOLVED |
| Q11 | 第二环境 Adam 已有初步结果 | V9 SGD + paired environment | RESOLVED / BOUNDARY |
| Q12 | D3/SAF/AdaTask 尚未完整判定 | V10 mechanism-equivalence audit | RESOLVED |
| Q13 | novelty gap 未闭合 | V10 + V12 | RESOLVED / COVERED |
| Q14 | method gate 未执行 | V13 | GO / NO-GO |
| Q15 | objective identity 尚需严格数值证明 | V1 | RESOLVED / IMPLEMENTATION BUG |
| Q16 | early effect vs overfitting 不清 | V2 | RESOLVED |
| Q17 | SGD LR 在边界 | V4 | RESOLVED / OPTIMIZER-BOUNDARY |
| Q18 | same multiset order effect 未单独识别 | V5 | RESOLVED |
| Q19 | repetition 未单独识别 | V5 | RESOLVED |
| Q20 | coverage 与 repetition 混杂 | V5 | RESOLVED / BOUNDARY |
| Q21 | covariance geometry 仅关联 | V6 | RESOLVED / ASSOCIATION |
| Q22 | shared block necessity 未证明 | V7 | RESOLVED |
| Q23 | optimizer × sampling interaction 不确定 | V11 | RESOLVED |
| Q24 | cross-environment boundary 不清 | V9 | RESOLVED / BOUNDARY |
| Q25 | 最新文献覆盖不完整 | V10 | RESOLVED |
| Q26 | D3/SAF objective equivalence 不清 | V10 | RESOLVED |
| Q27 | AdaTask 是否能消除 effect | V10 | RESOLVED |
| Q28+ | 执行中动态出现 | 动态扩展实验 | 任意数量，全部报告 |

## 终局硬规则

- 任何 BLOCKING question 不能以“future work”结案。
- 若实验显示问题不可识别，必须证明当前设计为什么不可识别，并记录可识别它所需的最小设计。
- 若问题不影响核心结论，可以保留为 OPEN，但必须进入 Final Report 的 Newly Exposed Questions。
- 不允许删除、合并、重命名问题以降低数量。
