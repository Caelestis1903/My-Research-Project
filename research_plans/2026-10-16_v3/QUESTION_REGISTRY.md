# 2026-10-16 v3：问题登记表

## 使用规则

本表不是固定长度列表。

**新问题数量无限制。**

实验过程中出现任何会改变解释、边界、创新判断或后续实验选择的问题，都必须新增 Q 编号，不得覆盖旧问题。

原始问题只有在以下状态之一时才算闭环：

- RESOLVED；
- FALSIFIED；
- BOUNDARY；
- NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN。

若状态为 NOT_IDENTIFIABLE，必须记录已经完成的最小必要实验以及为什么当前设计无法识别。

---

## A. v2 延续的原始问题

| ID | 问题 | v3 主要阶段 | 关闭标准 |
|---|---|---|---|
| Q1 | Objective-equivalent sampling 是否真的改变 training trajectory？ | V1,V2 | exact objective audit + common/pre-overfit trajectory |
| Q2 | 是否只是 checkpoint-selection artifact？ | V3 | common epoch + selection bootstrap + frozen test |
| Q3 | 是否依赖 Adam？ | V4,V11 | optimizer interaction / SGD replication |
| Q4 | 若 optimizer-specific，是否为 m/v/history？ | V11 | 仅 interaction 明确时触发；否则 FALSIFIED/NOT_REQUIRED |
| Q5 | 非 Adam 机制是 exposure/repetition/coverage/order/geometry 中哪一个？ | V5,V6 | 至少完成单因素 intervention |
| Q6 | shared/private parameter 是否必要？ | V7 | local + conditional full-training intervention |
| Q7 | domain size 是否解释 effect？ | V8 | size manipulation + sensitivity |
| Q8 | domain difficulty 是否解释 effect？ | V8 | controlled difficulty perturbation |
| Q9 | task/domain imbalance 是否解释 effect？ | V8 | cell-level audit + causal identifiability judgment |
| Q10 | validation selection 是否改变结论？ | V3 | selection rule sensitivity |
| Q11 | 是否跨环境成立？ | V9 | KuaiRand + second environment |
| Q12 | SAF/D3/AdaTask 是否覆盖？ | V10 | objective/mechanism-equivalence audit |
| Q13 | 是否存在未被已有工作覆盖的 mechanism gap？ | V10,V12 | literature + counterfactual synthesis |
| Q14 | 是否值得方法化？ | V13 | GO-METHOD criteria |

---

## B. v3 新增的关键问题

| ID | 问题 | 阶段 | 关闭标准 |
|---|---|---|---|
| Q15 | SN/SB 的“same objective”在代码和 empirical gradient 上是否严格成立？ | V1 | exact/MC gradient audit |
| Q16 | 现象是否主要发生在 overfitting 之后？ | V2 | independent diagnostic validation |
| Q17 | effect 是否只是 learning-speed / finite-budget effect？ | V4 | LR + batch + extended-budget |
| Q18 | 在相同 sample multiset 下，仅改变 optimizer-step 顺序是否改变结果？ | V5 | same-batch-set order replay |
| Q19 | repetition 是否是 exposure effect 的必要组成？ | V5 | WR vs PERM |
| Q20 | coverage 是否独立于 exposure/repetition？ | V5 | cumulative coverage + controlled sampler |
| Q21 | estimator covariance 的 anisotropy 是否比 scalar variance 更有解释力？ | V6 | spectrum + block + mediation |
| Q22 | shared C/R 是否是必要 mediator 而非相关参数块？ | V7 | intervention |
| Q23 | optimizer effect 与 sampling effect 是否存在真实 interaction？ | V11 | paired interaction |
| Q24 | 第二环境失败时，boundary 究竟来自哪里？ | V9 | boundary decomposition |
| Q25 | existing literature 是否已经覆盖“same objective + different q + MDMT trajectory”机制？ | V10 | literature audit |
| Q26 | D3/SAF 的 objective change 是否使其不能作为 mechanism-equivalent prior work？ | V10 | objective audit |
| Q27 | AdaTask 的 optimizer-state intervention 是否能消除 sampling effect？ | V10 | objective-preserving comparison |

---

## C. 动态问题模板

新问题统一追加：

### QN：<问题标题>

- **Origin**：由哪个实验产生；
- **Observation**：原始数据是什么；
- **Why it matters**：为什么会影响当前结论；
- **Candidate explanations**：所有竞争解释；
- **Blocking**：是否阻塞核心结论；
- **Minimal discriminating experiment**；
- **Status**；
- **Evidence**；
- **Conclusion**；
- **Remaining uncertainty**；
- **Next action**。

不得把多个不同机制问题为了减少编号而合并。

---

## D. Final closure rule

终局报告中：

1. Q1–Q27 必须全部有状态；
2. 所有 Q28+ 也必须全部登记；
3. 若某个新问题阻塞核心结论，必须继续实验；
4. 若某个新问题不阻塞核心结论，也必须原封不动报告；
5. 不设置“最多 N 个问题”的限制；
6. 不得为了形成漂亮的 final story 删除矛盾问题。
