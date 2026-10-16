# 2026-10-16 v3：文献边界与创新审计计划

## 1. 目的

本文件不是预先宣布“有创新”，而是规定如何证明：

> 当前 Objective-Equivalent Sampling + MDMT trajectory 问题是否已经被已有工作覆盖。

## 2. 必须覆盖的文献簇

### A. MDMT / multi-domain recommendation

重点：

- domain balancing；
- domain-aware weighting；
- shared/private representation；
- MoE routing；
- domain-specific optimization。

### B. Sample weighting / adaptive weighting

重点：

- sample-level weights；
- importance weighting；
- adaptive sample selection；
- variance reduction；
- objective modification。

### C. Importance sampling / stochastic optimization

重点：

- unbiased estimator；
- finite-batch variance；
- effective sample size；
- minibatch covariance；
- optimizer interaction。

### D. Training order / domain order

必须搜索近期关于：

- domain order；
- data mixing；
- sequential domain learning；
- same multiset / different order；
- training trajectory path dependence。

### E. Optimizer memory

必须搜索：

- Adam state；
- momentum；
- optimizer-memory-induced order dependence；
- minibatch perturbation propagation；
- finite-horizon optimizer dynamics。

### F. Task-wise optimizer

重点：

- AdaTask；
- task-wise Adam state；
- task-specific adaptive learning rate；
- optimizer state separation。

---

## 3. 当前已知工作只作为候选，不作为预设结论

至少建立以下 comparison：

### D3

需要回答：

> D3 是不是改变 objective 的 domain weighting？

若是，则与本问题：

[
J_{same},quad q_{different}
]

不是同一个 intervention。

### SAF

需要回答：

> SAF 的 sample weighting 是否改变 expected objective？

如果改变，则不能仅凭“sample weight”三个字判定已覆盖。

### AdaTask

需要回答：

> AdaTask 是否在 objective 不变时直接改变 optimizer-state pathway，并能否消除 SN/SB effect？

如果不能，则其 mechanism 与当前问题不同。

### Domain-order literature

如果已有工作证明：

> same data multiset + different order → different trajectory，

则 v3 不能把“order dependence”本身作为创新。

可能的 gap 必须进一步收缩到：

> objective-equivalent domain sampling in MDMT changes the stochastic estimator and its interaction with shared/routing structure.

但这个表述仍必须经过实验验证。

---

## 4. 近期工作审计

实验启动时必须重新搜索 2026-10-16 前的论文和预印本。

至少包含关键词：

- "objective equivalent sampling" optimizer;
- "importance sampling" Adam minibatch;
- "domain sampling" multi-domain training;
- "domain order" multitask;
- "training order" domains;
- "optimizer memory" shuffle order;
- "Adam minibatch perturbation";
- "task-wise optimizer" multitask;
- "sample weighting" multi-domain recommendation.

如果发现新论文与核心 mechanism 同构，立即新增 literature question，不等到最终论文写作阶段才处理。

---

## 5. Novelty decision

### Covered

如果已有工作已经同时解释：

1. same expected objective；
2. different sampling q；
3. trajectory difference；
4. MDMT/shared-routing consequence；
5. downstream performance；

则：

> 该机制不作为新颖性 claim。

### Partially covered

如果已有工作解释其中一部分，例如：

- order dependence；
- optimizer memory；
- importance-sampling variance；

但没有解释 MDMT-specific interaction，则只能把未覆盖部分作为候选 gap。

### Not covered

只有在：

- mechanism-equivalent；
- objective-equivalent；
- counterfactual-supported；
- cross-environment/boundary documented；

之后，才能声明新的 mechanism gap。

---

## 6. Literature result must be negative-friendly

最终报告必须允许：

> “实验发现已有工作已经覆盖该现象，因此没有创新。”

这是有效科研结果，不得为了保留论文故事而降低 equivalence 标准。
