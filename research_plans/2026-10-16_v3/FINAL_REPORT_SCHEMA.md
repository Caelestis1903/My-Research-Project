# 2026-10-16 v3：终局报告规范

## 1. Executive conclusion

第一屏必须回答：

> 目前最终知道什么？

最多用一页总结：

- core phenomenon；
- mechanism；
- boundary；
- selection/overfitting status；
- optimizer status；
- novelty status；
- method decision。

---

## 2. Historical correction

必须明确：

### S5–S8 已经证明

- 哪些是 Equal-specific；
- 哪些属于 Adam history；
- 哪些属于 late-overfitting rescue；
- 哪些路线已经 STOP。

不得把 v3 新结果重新解释成旧问题。

---

## 3. Objective equivalence

必须给：

[
J(	heta)
]

以及：

[
G^*(	heta),quad
E[hat g_{SN}],quad
E[hat g_{SB}].
]

明确：

- objective 是否相同；
- estimator 是否相同；
- estimator covariance 是否不同。

---

## 4. Trajectory vs selection vs overfitting

必须分别给：

### Trajectory

[
Delta(t)
]

### Selection

[
Delta_{selected}
]

### Overfitting

每个 seed 的 onset。

### Learning speed

AULC / time-to-threshold。

结论不能混在一句“最终 AUC 更低”里。

---

## 5. Sampling decomposition

必须回答：

- exposure；
- repetition；
- coverage；
- order；
- batch composition。

每一项都必须有：

- control；
- observation；
- decision；
- residual uncertainty。

---

## 6. Gradient mechanism

如果 geometry 被保留，必须展示：

- mean gradient；
- covariance trace；
- eigen-spectrum；
- effective rank；
- anisotropy；
- temporal correlation；
- block-wise geometry。

必须解释为什么 geometry 与 downstream metric 相关。

---

## 7. Shared/private

必须明确区分：

> correlation

和：

> necessity evidence。

没有 intervention 时不能称 necessary mediator。

---

## 8. Domain/task factors

至少报告：

- size；
- difficulty；
- positive rate；
- task exposure；
- domain exposure；
- per-cell performance。

只有在可识别时才写 causal。

---

## 9. Optimizer

必须最终回答：

### Adam

是否必要？

### SGD

是否同样存在？

### interaction

是否稳定？

### state decomposition

是否有理由继续？

如果没有，明确关闭 Adam-specific route。

---

## 10. Cross-environment

表格：

| Question | KuaiRand | Second environment | Final status |
|---|---|---|---|
| SB-SN exists | | | |
| pre-overfit | | | |
| repetition | | | |
| order | | | |
| geometry | | | |
| shared block | | | |

---

## 11. Existing work

逐项：

| Work | Same objective? | Same intervention? | Same mechanism? | Covers result? |
|---|---|---|---|---|
| D3 | | | | |
| SAF | | | | |
| AdaTask | | | | |
| domain-order works | | | | |
| optimizer-memory works | | | | |
| other discovered works | | | | |

---

## 12. Final decision

只能选择：

### MECHANISM-CONFIRMED

核心 mechanism 有反事实证据。

### EFFECT-CONFIRMED_MECHANISM-UNRESOLVED

effect 稳定，但没有唯一 mediator。

### BOUNDARY-CONFIRMED

只在明确条件成立。

### OVERFITTING_ONLY

没有独立的有价值早期 effect。

### OPTIMIZATION-SPEED_ONLY

只是有限预算速度差异。

### SELECTION-DEPENDENT

主要是 checkpoint selection。

### COVERED_BY_PRIOR_WORK

现有工作已覆盖。

### NULL

现象无法稳定复现。

### NOT_IDENTIFIABLE

已经完成最小实验，但当前实验设计无法识别。

不得为了“必须有创新”而强行选择 positive outcome。

---

## 13. Unlimited new-question section

最终报告必须有：

# Newly Exposed Questions

所有 Q28+ 必须逐条列出。

每个问题：

- 原始 observation；
- why unresolved；
- whether it blocks core conclusion；
- completed experiments；
- current boundary；
- next minimal experiment。

**不得限制问题数量。**

如果出现 1 个、10 个、50 个新问题，都原样报告。

---

## 14. Final scientific statement

最后必须用一段文字回答：

> 这项研究到底发现了什么，以及哪些东西没有发现。

必须明确：

- confirmed；
- falsified；
- boundary；
- unexplained；
- existing-work overlap；
- method/no-method decision。

不得用“future work”掩盖当前本应完成的关键闭环。
