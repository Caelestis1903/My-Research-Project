你现在执行科研项目的 Phase 2.5：Experimental Evidence Reconstruction。

当前输入包含本研究此前已经完成的全部阶段级实验分析报告，例如：

- Stage S0
- Stage S1
- Stage S2
- Stage S3
- Stage S4
- Stage S5
- Stage S5-ML
- Stage S6
- Stage S7
- Stage S7B
- Stage S8
- 以及重要的 trajectory / reanalysis 报告

不同实验阶段之间存在明确的时间顺序。

后期实验可能：

- 支持早期假设；
- 限制早期假设的适用范围；
- 削弱早期假设；
- 推翻早期假设；
- 为早期现象提供新的替代解释。

因此：

**不得把所有阶段报告中的结论视为同时成立。**

---

# 0. 本阶段唯一目标

本阶段不进行文献分析。

不寻找 research gap。

不设计新算法。

不寻找创新点。

不判断论文投稿价值。

唯一目标是：

> 从全部历史实验报告中重新建立一个严格、可追踪、去除过时解释的“当前实验事实数据库”。

最终回答：

1. 到目前为止究竟做过什么实验；
2. 实际观察到了什么；
3. 哪些观察具有稳定证据；
4. 哪些只是单数据集、单配置或弱证据；
5. 哪些历史假设后来被削弱或推翻；
6. 哪些问题目前仍未解释。

---

# 1. 基本分析原则

严格区分以下四层。

## Observation

实验直接观察到的结果。

例如：

某种设置下 Domain A AUC 相比 baseline 上升 X。

Observation 不包含机制解释。

---

## Pattern

多个实验重复观察到的现象。

只有在多个 seed、dataset、configuration 或重复实验中具有一定一致性时，才能升级为 Pattern。

---

## Interpretation

基于 Observation / Pattern 对现象进行的解释。

Interpretation 不等于已经证明。

---

## Hypothesis / Mechanism

关于产生该现象原因的可验证假设。

除非存在直接机制实验支持，否则不得标记为 Established Mechanism。

---

# 2. 禁止预设

不得预先假设：

- M3OE 存在稳定负迁移；
- 小域一定是困难域；
- 大域一定受损；
- Equal objective 一定错误；
- Sample objective 一定正确；
- Sample 已经解决所有问题；
- gradient conflict 是根因；
- parameter sharing 是根因；
- optimizer history 是根因；
- HT / DH / T 某一种模式一定更好；
- 某个方法改善是因为缓解 overfitting；
- 某个方法改善属于真正 optimization improvement；
- 训练阶段动态一定有研究价值。

全部结论必须来自已有实验。

---

# 3. 首先建立 Experiment Index

阅读全部实验报告。

建立完整 Experiment Index。

每一个实验或实验组赋予唯一编号：

EXP001
EXP002
EXP003
...

记录：

- Experiment ID
- Stage
- Dataset
- Model / Backbone
- Objective
- Training setting
- Compared conditions
- Seeds
- Metrics
- Report source
- Experiment purpose

注意：

“Experiment purpose”记录当时为什么做这个实验，

但不意味着当时的假设后来仍然成立。

---

# 4. 建立 Experimental Evidence Ledger

这是本阶段最核心的输出。

不要按照 S0 → S8 写流水账。

按照“独立实验事实”组织。

每一个 Evidence 建立唯一编号：

E001
E002
E003
...

至少包含以下字段。

---

## Evidence ID

---

## Observation

只描述直接实验现象。

尽可能包含：

- baseline
- comparison setting
- AUC / loss 等指标变化
- domain/task/cell
- epoch / selected checkpoint
- dataset

不要混入原因解释。

---

## Source Experiments

对应 EXP ID。

---

## Source Reports

对应具体阶段报告。

---

## Dataset

---

## Objective

例如：

Equal / Sample / 其他。

---

## Model / Backbone

---

## Seeds

记录：

- seed 数量
- 哪些 seed 支持
- 哪些不支持

如果报告没有给出，则写 UNKNOWN。

---

## Effect Direction

例如：

Increase / Decrease / Mixed / No clear effect

---

## Effect Magnitude

如可以确认，记录具体数值。

不能确认则 UNKNOWN。

不得自行估计。

---

## Robustness

评价：

### Strong
多 seed / 多配置重复出现，方向稳定。

### Medium
存在重复证据，但仍有明显条件依赖。

### Weak
单 seed、单配置、小幅变化或不稳定。

---

## Scope

判断该 Evidence 当前只支持到什么范围：

- single run
- single seed
- single configuration
- single dataset
- multiple datasets
- cross-model
- general

禁止过度外推。

---

## Current Status

只能使用以下状态：

### ESTABLISHED
直接实验事实反复成立。

### SUPPORTED
证据较好，但仍有限定条件。

### TENTATIVE
存在迹象，但证据不足。

### DATASET_SPECIFIC
主要只在特定数据集成立。

### CONFIGURATION_SPECIFIC
依赖特定 objective / model / training setting。

### CONTRADICTED
后续实验存在明确反证。

### SUPERSEDED
旧解释被后来的证据替代。

### UNRESOLVED
现象存在，但解释尚不清楚。

---

## Current Interpretation

基于目前全部实验，可以合理接受的最低强度解释。

注意使用：

“supports”
“suggests”
“is consistent with”

而不是无证据地写：

“proves”
“demonstrates the mechanism”

---

## Alternative Explanations

列出仍然可能解释该现象的其他因素。

例如：

- objective weighting
- dataset imbalance
- overfitting
- checkpoint selection
- stochastic variance
- training duration
- domain/task interaction
- parameter sharing
- optimizer dynamics

这里只根据实验报告记录，不自行发明复杂理论。

---

## Evidence Strength

Strong / Medium / Weak

并简短说明理由。

---

# 5. 建立 Hypothesis History

单独重建整个项目的假设演化。

每一个重要历史假设建立：

H001
H002
H003
...

记录：

## Initial Hypothesis

当时提出了什么解释？

---

## Origin

由哪些 Observation 产生？

---

## Tests Performed

哪些实验用于检验它？

---

## Supporting Evidence

哪些 Evidence IDs 支持？

---

## Contradicting Evidence

哪些 Evidence IDs 反对？

---

## Evolution

记录：

提出
→ 支持
→ 修正
→ 削弱
→ 推翻

的过程。

---

## Current Status

仅使用：

### ACTIVE
仍有较强支持。

### PARTIALLY_SUPPORTED
只有部分成立。

### WEAKENED
后续实验明显削弱。

### FALSIFIED
已有明确反证。

### SUPERSEDED
已经被更合理的解释替代。

### OPEN
仍无法判断。

---

## Current Allowed Claim

根据目前证据，论文/汇报中最多可以说到什么程度。

---

## Claims That Should No Longer Be Made

明确写出历史上曾经可以提出、但现在不能继续直接使用的结论。

这一字段非常重要。

---

# 6. 建立 Observation → Interpretation → Mechanism 链

对于所有核心现象，建立如下结构：

Observation
↓
Candidate Interpretation
↓
Candidate Mechanism
↓
Supporting Experiment
↓
Contradicting Experiment
↓
Current Confidence

例如：

E0XX
↓
Interpretation A
↓
Mechanism A1
↓
Sx supports
↓
Sy contradicts
↓
Current confidence: Weak

目的：

防止把“实验效果”直接等同于“机制成立”。

---

# 7. 分析 Dataset Dependence

针对所有重要 Evidence，比较：

- KuaiRand
- MovieLens
- Ali-CCP（如果已有）
- 其他数据集

判断：

### Cross-dataset replicated

### Partially replicated

### Dataset-specific

### Cannot determine

特别禁止：

只因为两个数据集结果不同，就立即解释为什么不同。

先记录差异。

机制解释留在 Unresolved Questions。

---

# 8. 分析 Objective Dependence

专门比较：

- Equal
- Sample
- 其他 objective

回答：

哪些现象：

### Equal-only

### Sample-only

### Appears under both

### Objective-sensitive but unclear

### Insufficient evidence

不得把 Equal 下发现的机制直接外推到 Sample。

反之亦然。

---

# 9. 分析 Training-Stage Dependence

如果已有 epoch trajectory / selected checkpoint / early stopping 等分析：

严格区分：

### Early-stage effect

### Mid-training effect

### Late-stage effect

### Final / selected checkpoint effect

### Overfitting-related effect

### Cannot distinguish

尤其检查：

某种“改善”是否只是：

- 延缓后期退化；
- 改善 checkpoint selection；
- 减少 overfitting；

还是：

- 在正常训练区域真正获得更优 optimization trajectory。

如果证据不足，写：

UNRESOLVED。

不得自行选择更有利的解释。

---

# 10. 分析 Metric Dependence

不要只看 overall AUC。

如果报告包含：

- per-domain AUC
- per-task AUC
- six-cell macro
- worst-cell
- overall / weighted overall
- loss
- selected epoch
- final epoch

则分别记录。

特别寻找：

### Aggregate improvement with subgroup degradation

### Target-domain improvement with non-target degradation

### Target-domain degradation with macro improvement

### Mean improvement with worst-cell deterioration

这些都应分别成为 Evidence。

不得只用一个 overall 指标覆盖内部 trade-off。

---

# 11. 建立 Contradiction Register

专门记录所有实验间矛盾。

每一个矛盾赋予：

C001
C002
...

记录：

## Evidence A

## Evidence B

## Difference in setting

例如：

- dataset
- objective
- seed
- model
- epoch
- checkpoint
- training mode

## Can the contradiction currently be explained?

Yes / Partially / No

## Current interpretation

如果无法解释：

UNRESOLVED

不要强行统一所有实验。

---

# 12. 建立 Stable Findings

只有满足足够证据标准的结论才能进入 Stable Findings。

每一条必须包含：

- Finding
- Evidence IDs
- Scope
- Evidence strength
- Known exceptions

Stable Finding 必须是保守陈述。

例如：

不要写：

“Sample 解决了 MDMT 的负迁移。”

如果现有证据实际上只支持：

“在当前 KuaiRand / M3OE 实验中，相比 Equal，Sample 显著减弱此前观察到的 domain-level trade-off。”

就只能写后者。

---

# 13. 建立 Open Questions

只保留真正仍然没有被现有实验回答的问题。

每一个 Open Question 记录：

- Question ID
- Scientific question
- Why unresolved
- Relevant Evidence IDs
- Competing explanations
- What evidence would resolve it

注意：

本阶段只记录需要什么类型的证据。

**不要设计下一阶段具体算法。**

---

# 14. 建立 Dead Ends / Closed Threads

单独记录已经不值得继续基于原假设推进的研究线程。

可能包括：

- 已被 falsified 的假设；
- 仅由 Equal artifact 支撑的现象；
- 无法稳定复现的现象；
- effect size 极小且不稳定；
- 已有实验直接证明无效的方法。

每个 Dead End 记录：

- Thread
- Why it was initially considered
- Evidence against it
- Current status
- Whether any narrower version remains open

目的不是否定历史工作。

而是避免 Phase 3 重新把已经失败的研究方向“复活”。

---

# 15. 不进行 Literature Matching

即使你知道某种现象可能对应：

- negative transfer
- gradient conflict
- task dominance
- optimizer imbalance
- overfitting

也不要查文献。

不要判断 novelty。

不要使用 Phase 2 的 Research Gap 来重新解释实验。

Phase 2.5 必须保持：

**experiment-only reconstruction。**

---

# 16. 最终输出文件

至少生成以下文件。

## 1. Experimental_Experiment_Index.md

全部实验索引。

---

## 2. Experimental_Evidence_Ledger.md

完整 Evidence Ledger。

这是 Phase 3 最主要的实验输入。

---

## 3. Experimental_Hypothesis_History.md

全部重要假设的演化、支持、削弱、推翻过程。

---

## 4. Experimental_Contradiction_Register.md

实验之间所有重要矛盾。

---

## 5. Experimental_Stable_Findings.md

只包含当前仍可以安全使用的实验结论。

---

## 6. Experimental_Open_Questions.md

真正未解决的问题。

---

## 7. Experimental_Dead_Ends.md

已被显著削弱、推翻或不值得继续按照原逻辑推进的方向。

---

## 8. Experimental_Global_Summary.md

高度凝练总结整个实验项目。

必须回答：

### Q1
目前最强的实验事实是什么？

### Q2
哪些早期结论后来被修正？

### Q3
哪些现象只在特定 dataset/objective/setting 下成立？

### Q4
哪些机制目前真正有证据？

### Q5
哪些所谓机制其实仍然只是 hypothesis？

### Q6
哪些问题仍然 unresolved？

### Q7
哪些旧研究线程不应在 Phase 3 中被重新当成有效前提？

---

# 17. Traceability Audit

完成后进行检查。

随机抽查重要 Evidence。

确认：

- Evidence 是否确实来自对应报告；
- 数值是否正确；
- dataset/objective/seed 是否正确；
- Observation 是否混入了 Interpretation；
- 后期结论是否覆盖了已经过时的早期判断；
- 是否把单数据集结果误写成 general conclusion；
- 是否把 performance improvement 误写成 mechanism proof。

发现错误必须修正。

---

# 18. Completion Criteria

只有以下全部满足，Phase 2.5 才算完成：

- 全部阶段报告均被处理；
- 全部主要实验进入 Experiment Index；
- 所有重要 Observation 都进入 Evidence Ledger；
- 旧假设与当前结论已经明确分离；
- Contradicted / Superseded 结论被明确标记；
- Stable Findings 都能追踪到实验；
- 所有重要 unresolved questions 被记录；
- 没有进行 literature matching；
- 没有提出新的算法；
- 没有为了后续研究方向选择性解释实验。

完成后，仅简短报告：

1. 共整理多少个 Experiment；
2. 共形成多少条 Evidence；
3. Strong / Medium / Weak 各多少；
4. 多少个历史 Hypothesis 被保留、削弱、推翻；
5. 多少个 Stable Findings；
6. 多少个 Open Questions；
7. 是否已经具备进入 Phase 3 的条件。