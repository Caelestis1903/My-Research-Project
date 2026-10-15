你现在进入 **Stage S8 — Optimizer/Trajectory Final Kill Test**。

本阶段不是 method-design stage，而是对 S5–S7 所形成的 optimizer-history / gradient-dynamics 研究主线进行最后一次严格因果诊断与 GO/STOP 裁决。

你必须以 falsification 为第一原则。不得为了延续当前研究方向而解释实验结果，不得预设 HT、DH、T、gradient utility、optimizer history 或 trajectory mechanism 有效。

---

# 0. 必须先阅读的已有实验材料

开始设计前，完整阅读并继承以下已有阶段结论：

- `stage_S5_analysis.md`
- `stage_S5ML_analysis.md`（如存在）
- `stage_S6_analysis.md`
- `stage_S6_phaseA_analysis.md`（如存在）
- `stage_S7_analysis.md`
- `stage_S7B_analysis.md`
- `EXP_AI_CONTEXT_V2.md`（如存在）
- S6/S7 对应原始 CSV、run reports、manifest、checkpoint 与 optimizer-state dumps

禁止只阅读摘要。

如果报告结论与原始数据存在冲突，以原始实验数据和可复核代码为准，并在最终报告中显式指出。

---

# 1. 当前已经成立的事实

以下内容视为 S8 的背景事实，但不得进一步扩大解释。

## 1.1 S7-A

在 Sample objective 下：

1. MovieLens 出现明显的 late-stage deterioration：
   - validation-selected epoch 大约位于 7–12；
   - epoch 19–49 后性能持续恶化；
   - late stage 出现稳定的 Domain×Task harmful-gradient structure；
   - validation-gradient utility \(U_k\) 随训练后期转负；
   - 可以定位 source→target harmful contribution。

2. KuaiRand 不复制 MovieLens 的 performance consequence：
   - 即使 late checkpoint 也可以出现负 \(U_k\)；
   - 但 validation performance 仍继续改善。

因此：
- negative \(U_k\) 不是性能下降的充分条件；
- gradient phenomenon 是真实的，但具有 dataset/regime conditionality；
- 不允许直接把它解释为 universal MDMT failure。

## 1.2 S7-B

在 same-θ、same shared optimizer moments、same batch sequence 下：

- late-stage TARGET-current manipulation 可以立即改变部分 harmful-gradient metrics；
- late effect 明显强于 early；
- 但是 5-step continuation 内没有出现 validation improvement；
- MovieLens late anchor target AUC 反而出现轻微下降；
- 因此“即时 raw-gradient improvement → validation improvement”的机制没有得到支持。

S7-B 留下的主要未排除解释是：

> S6 的 late DH/T residual 可能来自较长时间尺度上的 optimizer-state accumulation / optimization trajectory / regularization / over-training dynamics，而不是单步 gradient geometry 的直接作用。

但该解释当前仍未被证明。

---

# 2. S8 唯一核心科学问题

本阶段只回答：

> **S6 中观察到的 DH/T late-stage benefit，是否可以在完整 optimizer state、严格 same-state 的条件下，通过较长 continuation 稳定改变 optimization trajectory，并最终提高“最佳 validation performance”；还是它只能够改善已经严重过拟合的 late checkpoint？**

必须区分：

### 情况 A：仅 late rescue

例如：

- F 在 epoch39 很差；
- T/DH 从 epoch39 continuation 后比 F 好；
- 但仍然没有超过 F 原来的 validation-selected best checkpoint。

这种结果不得解释为一个有价值的新方法。

它只能说明：

> manipulation can partially rescue an already over-fitted model.

### 情况 B：best-performance improvement

例如：

- baseline best AUC = epoch8；
- T/DH continuation 最终形成新的 trajectory；
- best validation AUC 稳定超过原 baseline best；
- ≥3/4 seeds；
- 同时不能由单个 seed 或 checkpoint selection artifact 驱动。

只有这种结果，才能继续讨论 optimizer/trajectory mechanism 的方法价值。

---

# 3. 实验设计原则

## 3.1 不开发新算法

本阶段禁止：

- 新 gradient surgery；
- 新 adaptive weighting；
- 新 loss；
- 新 gating/expert module；
- 新 regularizer；
- 新 curriculum；
- 新 momentum variant；
- 根据结果临时设计额外方法。

只能使用 S6/S7 中已经定义、能够明确复现的 branch。

优先比较：

- F baseline；
- T branch；
- DH branch（如果 S6 的语义与实现可以严格恢复）。

如果 HT 是理解 S6 mechanism 所必需，可以加入，但必须说明其唯一诊断目的。

---

# 4. 最重要的实验改进：完整 optimizer-state continuation

S7-B 的重要限制是 private optimizer moments 未被保存，step≥1 使用了 fresh-Adam private state。

S8 必须尽可能消除此限制。

对 continuation anchor 必须保存和恢复：

1. 全部 model parameters：
   - shared parameters；
   - private/domain-specific parameters；
   - task-specific parameters。

2. 完整 Adam state：
   - shared parameter first moment \(m_t\)；
   - shared parameter second moment \(v_t\)；
   - private parameter \(m_t,v_t\)；
   - task/domain-specific optimizer state；
   - optimizer step counter；
   - scheduler state。

3. RNG states：
   - Python；
   - NumPy；
   - PyTorch CPU；
   - CUDA；
   - dataloader/batch sampler。

4. Batch order / batch fingerprints。

5. learning-rate scheduler / early-stopping state。

目标是实现：

> F/T/DH branches 在 branch point 之前完全 identical，在 branch point 之后仅允许目标操纵产生差异。

必须提供自动 validity test。

---

# 5. Dataset 与 seeds

至少：

## MovieLens
主诊断数据集。

seeds：

- 2022
- 2023
- 2024
- 2025

## KuaiRand
negative/control dataset。

同样：

- 2022
- 2023
- 2024
- 2025

不要因为 MovieLens 更容易出现现象而删除 KuaiRand。

Ali-CCP 本阶段默认不加入，除非已有完全成熟的数据接口并且计算成本不会显著扩大实验。

S8 首先是 kill test，不是 benchmark expansion。

---

# 6. Anchor 设计

不能只使用 epoch39。

MovieLens 每个 seed 至少包含三类 anchor。

## A. selected / near-selected anchor

使用该 seed 实际 validation-selected checkpoint。

如果必须统一 epoch，可以增加 common early anchor，但不能取代 per-seed selected checkpoint。

这一 anchor 是整个 S8 最重要的现实性测试：

> manipulation 在模型正常应该停止训练的位置是否有价值？

## B. middle anchor

建议：

- epoch19 左右；

或者根据 S7 performance trajectory 选择“已经开始 deterioration、但尚未 severe overfit”的阶段。

## C. late anchor

建议：

- epoch39；

用于复现 S6/S7 已发现的 late phenomenon。

可选增加 epoch49，但不得因为增加更多 late anchor 而稀释 selected-anchor 结果的重要性。

KuaiRand 使用对应的 common checkpoints/control anchors。

---

# 7. Continuation horizon

S7-B 的 5-step window 太短，无法检验 long-horizon trajectory hypothesis。

S8 必须至少包含：

- very-short：5 steps（用于与 S7-B 对照）
- short：约 0.25–0.5 epoch
- medium：1 epoch
- longer：2 epochs

如计算允许，可继续至：

- 5 epochs

但不要无止境训练。

对于不同 branch：

> 必须使用相同的 continuation batch sequence。

---

# 8. 主要评价指标

本阶段最重要的指标不是 \(U_k\)。

评价优先级严格如下。

## Primary endpoint

### 8.1 Best validation performance

对每个 seed、branch、anchor：

记录 continuation 期间：

- best target-domain AUC；
- best macro AUC；
- best six-cell / worst-cell（若已有定义）；
- 对应 checkpoint；
- validation LogLoss/BCE。

必须和：

> baseline F 从正常完整训练过程中得到的原始 best-validation checkpoint

进行比较。

主要统计量：

\[
\Delta BestAUC =
BestAUC_{branch}
-
BestAUC_{F,\ original}
\]

而不是：

\[
AUC_{branch@epoch39}
-
AUC_{F@epoch39}
\]

后者只代表 late rescue。

---

# 9. Secondary endpoint：trajectory

记录：

\[
AUC(t),\quad BCE(t)
\]

比较 F / T / DH continuation trajectory。

需要判断：

1. manipulation 是否延迟 deterioration；
2. 是否只是降低 overfitting slope；
3. 是否形成新的、更高 validation optimum；
4. 还是只是使 late model “没那么差”。

必须明确区分这四种结果。

---

# 10. Optimizer-level diagnostics

本阶段不能继续只分析 raw gradient。

至少记录：

## 10.1 raw gradient

\[
g_t
\]

包括 S7 已使用的：

- gradient norm；
- cosine；
- \(U_k\)；
- \(H_{i\rightarrow k}\)。

这些是 secondary diagnostics。

## 10.2 Adam effective update

必须计算实际 Adam update：

\[
\Delta \theta_t
=
-\eta_t
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
\]

重点分析：

- effective update norm；
- effective-update cosine；
- Domain×Task source contribution（若数学上可严谨定义）；
- effective update 与 validation gradient alignment；
- raw gradient alignment 与 effective-update alignment 的差异。

核心问题：

> S7-B 中 raw gradient metric 改善却 validation 不改善，是否因为 Adam state 改变了真正执行的 update geometry？

---

# 11. History-state diagnostics

如果 HT/DH/T 的定义允许，必须明确区分：

### Current effect
当前 batch gradient source 改变。

### History effect
Adam \(m/v\) 历史积累改变。

### θ effect
长期 trajectory 导致参数位置改变。

最终需要尝试回答：

\[
Current
\rightarrow
History
\rightarrow
EffectiveUpdate
\rightarrow
\theta
\rightarrow
Validation
\]

哪一个连接有证据，哪一个没有。

禁止把相关性写成因果。

---

# 12. 关键分析：selected anchor vs late anchor

必须单独制作表格：

| dataset | seed | anchor | branch | original F best AUC | continuation best AUC | Δbest | final AUC | late rescue? | new best? |

然后按：

- selected；
- middle；
- late

分别聚合。

重点回答：

> branch 的优势是不是随着 anchor 越晚越明显？

如果：

\[
late \gg middle \gg selected\approx0
\]

那么最自然解释应优先是：

> overfitting rescue / trajectory stabilization

而不是：

> universally better MDMT optimization.

---

# 13. 必须进行的 single-domain / practical-value sanity check

如果已有 single-domain baseline，使用已有结果。

若成本较低，可以补充。

需要判断：

> manipulation 是否真正缩小正常 selected checkpoint 下的 joint-training performance gap？

如果没有任何正常 checkpoint performance gap，则不得因为 late-stage 指标漂亮而开发方法。

---

# 14. GO / STOP 判据

必须预注册并严格执行。

## GO：允许继续 optimizer/trajectory 主线

只有满足大部分以下条件，尤其是第 1 条，才能 GO：

1. **Best validation performance improvement**
   - T/DH 相对原始 F best checkpoint 有稳定提升；
   - ≥3/4 seeds 同方向；
   - 不是只比较 late checkpoint。

2. selected 或 middle anchor 也可以产生有意义的 benefit，而不是只有 severe-overfit late anchor。

3. effective Adam update 提供比 raw gradient 更直接、稳定的机制证据。

4. MovieLens 中 effect magnitude 不是由单一 seed 驱动。

5. KuaiRand 至少不会出现明显性能损害。

6. trajectory difference 可以重复，并且时间顺序与 hypothesis 一致。

满足这些条件后，才允许进入：

> history-aware / trajectory-aware MDMT optimization method design。

---

# 15. STOP 条件

出现任一核心 STOP 条件时，应停止当前主线。

## STOP-1

T/DH 只能改善 late overfitted checkpoint，但：

\[
BestAUC_{T/DH}
\le
BestAUC_{F,original}
\]

则判定：

> useful primarily as overfitting rescue, not a new MDMT optimization problem.

## STOP-2

selected checkpoint continuation：

- 无稳定收益；
- 或 2/2 / seed-split；
- 或平均变化接近噪声。

同时 late rescue 明显。

则优先判定：

> late-stage overfitting phenomenon。

## STOP-3

完整 optimizer-state + long-horizon continuation 后，S6 late residual 不再稳定复现。

则 S6 residual 不足以支撑 mechanism。

## STOP-4

raw gradient / effective-update statistics 改善仍无法对应任何 validation benefit。

则停止基于这些 statistics 开发 correction method。

## STOP-5

结果主要由某一个 seed 驱动。

---

# 16. 如果 STOP，不允许 main agent 自动设计 S9 gradient method

这是硬约束。

如果 S8 STOP：

不要继续：

- gradient correction；
- PCGrad-like method；
- stage-aware U weighting；
- harmful-pair masking；
- adaptive source gradient suppression；
- HT/DH 新组合。

下一阶段研究问题转为：

> **Sample objective 在 extreme domain imbalance 下，是否存在 selected-checkpoint minority-domain under-optimization / joint-training performance gap？**

下一阶段必须从真实 performance gap 出发，而不是从 gradient statistic 出发。

---

# 17. STOP 后下一研究问题的初步诊断方向

仅在 S8 STOP 后提出，不在 S8 中大规模执行。

建议下一问题定义为：

\[
Gap_{d,t}
=
Perf^{joint}_{d,t}
-
Perf^{single/oracle}_{d,t}
\]

研究：

\[
Domain\ imbalance
\rightarrow
joint-training\ gap
\]

是否存在稳定关系。

候选 ratio：

- 1:1:1
- 3:1:1
- 10:1:1
- 25:1:1
- 50:1:1

要求：

- selected checkpoint；
- 4 seeds；
- Sample objective；
- 与 single-domain / balanced control 比；
- 先证明 performance problem，再做 gradient/representation/optimizer mechanism diagnosis。

不得预设：

- small domain = difficult domain；
- Sample 一定忽视小域；
- gradient 一定是原因。

---

# 18. Agent 分工

若采用多 Agent：

## Main agent

职责：

- 阅读历史报告；
- 控制研究问题；
- 检查实验是否回答预注册问题；
- 最终 GO/STOP 裁决；
- 禁止亲自改代码；
- 禁止为了延续方向重新定义成功标准。

## Designer

设计：

- exact optimizer-state continuation；
- anchor；
- branch；
- validity checks；
- effective-update measurement；
- statistical aggregation。

不得设计新算法。

## Coder

严格实现 designer specification。

不得修改实验问题。

## Reviewer

重点检查：

1. branch 是否真正 same-state；
2. private moments 是否恢复；
3. scheduler/step counter 是否一致；
4. batch replay 是否一致；
5. best-AUC comparison 是否错误地只与 late F 比；
6. 是否存在 data leakage；
7. effective Adam update 公式是否与实际 optimizer implementation 一致。

## Runner

只运行并记录。

禁止分析。

---

# 19. 最终报告必须包含

报告命名建议：

`stage_S8_analysis.md`

必须包括：

### A. Execution validity
exact-state restore、optimizer state、batch replay、RNG、scheduler 检查。

### B. Original baseline best performance
每 seed 原始 F best checkpoint。

### C. Selected-anchor continuation
这是最重要部分。

### D. Middle-anchor continuation

### E. Late-anchor continuation

### F. Best-AUC comparison
禁止只报告固定 late epoch。

### G. Raw gradient vs effective Adam update

### H. Current / History / θ / Validation causal-chain evidence

### I. MovieLens vs KuaiRand

### J. Seed consistency

### K. Counterexamples

### L. Competing explanations
至少包括：
- ordinary overfitting；
- optimizer-history；
- regularization；
- learning-rate schedule；
- checkpoint-selection artifact。

### M. GO / STOP table

### N. Final verdict

最终裁决只能选择类似：

**A. LONG-HORIZON OPTIMIZER EFFECT SUPPORTED**

或

**B. LATE OVERFITTING RESCUE ONLY**

或

**C. OPTIMIZER/GRADIENT MAINLINE FALSIFIED**

不得使用模糊结论。

---

# 20. 最重要的科研纪律

本阶段最重要的问题不是：

> “还能不能找到一个现象把梯度方向继续做下去？”

而是：

> “这一现象是否能够提高正常训练协议下最终会被选择的模型性能？”

如果答案是否定的，即使：

- late AUC 提升很大；
- \(U_k\) 很漂亮；
- harmful pair 很稳定；
- HT/DH/T 差异明显；

也必须停止 method-design 主线。

S8 的价值在于给 S5–S7 一个清晰、可复核、不会继续漂移研究问题的最终裁决。