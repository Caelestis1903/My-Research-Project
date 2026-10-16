# 2026-10-16 v3：统一实验与统计协议

## 1. Primary endpoint hierarchy

所有实验必须同时保存：

### Primary scientific endpoints

1. pre-overfit validation AUC；
2. pre-overfit AULC；
3. common-epoch SB-SN；
4. selected test AUC（selection 冻结后一次读取）。

### Secondary

- validation LogLoss；
- test LogLoss；
- domain/task AUC；
- train objective；
- convergence speed；
- parameter displacement；
- gradient geometry。

禁止只用 selected AUC 讲完整故事。

---

## 2. Seed protocol

默认：

- primary experiment：4 seeds = 2022, 2023, 2024, 2025；
- mechanism diagnostic：4 seeds；
- 若关键 null 结论对 4 seeds 不足以区分“真 null”和“低 power”，扩展到 8 seeds；
- 诊断 batch 的 512 次采样不能冒充 512 个 independent training seeds。

每个 primary contrast 必须保存四个 seed-level (Delta_s)。

---

## 3. Statistical reporting

必须报告：

- mean；
- SD；
- median；
- 4 seed values；
- sign consistency；
- paired bootstrap CI；
- paired permutation/randomization check；
- common-epoch effect；
- selected effect；
- effect at pre-overfit window。

4 seeds 太少时，不得把 p-value 当作唯一结论。

尤其不得把：

> 4/4 同方向

写成“严格显著”。

应写：

> seed-consistent evidence，随后结合 CI、effect size 和独立复现实验判断。

---

## 4. Multiple epochs

epoch 1..50 是 trajectory data，不允许因为某个 epoch 恰好显著就把它当作 primary discovery。

预注册 readout：

- 5；
- 10；
- 15；
- 20；
- 30；
- 40；
- 50；
- pre-overfit AULC。

---

## 5. Test leakage

Test 只能：

1. 在训练完成后；
2. selection rule 冻结；
3. LR 冻结；
4. 方法超参数冻结；
5. 分析问题冻结；

之后读取。

不得因为 test 结果决定下一次实验条件。

若确实发现 test 暗示问题，必须重新登记为新问题并在新的 frozen experiment 中验证。

---

## 6. Checkpoint selection

定义：

[
t^*=argmax_t AUC_{val}(t)
]

tie 时：

1. earliest epoch；
2. 若仍相同，按预注册次级规则。

必须保存：

- selected epoch；
- selected validation metric；
- all epoch metrics；
- selection bootstrap；
- selected effect。

同时报告：

[
Effect_{fixed}(t)
]

和：

[
Effect_{selected}.
]

---

## 7. Overfitting protocol

不能把“epoch 20 后下降”自动定义为 overfitting。

必须同时观察：

- training objective；
- validation AUC；
- validation LogLoss；
- diagnostic validation；
- test only after freeze。

若 validation AUC 下降但 training objective 也停止改善，则不能简单称 overfitting。

若 training objective 持续下降而 independent diagnostic validation 持续下降，才有较强 overfitting 证据。

---

## 8. Optimization budget

比较 SN/SB 时必须保持：

- total examples；
- optimizer updates；
- epochs；
- model initialization；
- split；
- preprocessing。

LR calibration 单独记录，不能把 calibration runs 混入 primary seeds。

---

## 9. Objective equivalence tolerance

固定 (	heta) 后：

[
G^*=sum_dp_dG_d.
]

对于 estimator mean：

[
ar G_q=rac1Ksum_{k=1}^{K}hat g_{q,k}.
]

必须给出：

[
|ar G_q-G^*|
]

和 Monte Carlo CI。

若 difference 超出预先计算的 MC uncertainty，则 objective equivalence audit FAIL。

---

## 10. Same-multiset order protocol

对一个 epoch：

[
S={b_1,ldots,b_T}
]

保存每个 batch 的 exact row-id set。

构造：

[
S_A=(b_1,ldots,b_T)
]

和：

[
S_B=(b_{pi(1)},ldots,b_{pi(T)}).
]

要求：

- batch 内容完全相同；
- 每个 row 出现次数完全相同；
- domain exposure 完全相同；
- batch size 完全相同；
- 只改变 batch 顺序。

这是 order intervention 的硬条件。

---

## 11. Repetition protocol

对于每个 q：

### WR

domain 内 with replacement。

### PERM

domain 内固定随机 permutation，耗尽后重新 permutation。

必须报告：

- exact domain quota；
- unique count；
- repeat histogram；
- first-seen time；
- cumulative coverage。

---

## 12. Gradient probe protocol

每个固定 (	heta)：

- ≥512 batches；
- 相同 batch size；
- 相同 parameter checkpoint；
- 不更新参数。

记录：

- (g)；
- block gradients；
- exact (G^*)；
- norm；
- cosine；
- covariance；
- eigen-spectrum；
- effective rank；
- temporal correlation。

必须明确：

> 诊断 covariance 的维度、projection size、rank 上限。

不能把 projection covariance 冒充全模型 covariance。

---

## 13. Causal language rules

### 可以写

- “consistent with”；
- “supports”；
- “necessary-condition evidence”；
- “associated with”；
- “boundary”.

### 只有满足反事实标准才可以写

- “mediator candidate”；
- “mechanistically explains”。

### 禁止

没有 intervention 时写：

> “X causes Y.”

---

## 14. Engineering scope

只修会改变 scientific conclusion 的工程问题。

必须保留：

- failed attempt；
- device initialization issue；
- invalid run；
- aborted run；
- data audit discrepancy。

但不要为了形式上的测试覆盖率增加大量与研究结论无关的 defensive tests。

---

## 15. Reproducibility artifact

每个实验至少保存：

- command；
- code commit；
- config；
- seed；
- dataset fingerprint；
- data split fingerprint；
- sampler fingerprint；
- selected epoch；
- metrics；
- raw trajectory；
- summary；
- analysis script hash。

---

## 16. Result integrity

任何失败、反向 seed、异常曲线都必须保留。

不得：

- 删除 inconvenient seed；
- 根据结果重新选择 seeds；
- 根据 test 修改 selection；
- 把失败实验从 registry 中移除；
- 将 exploratory result 写成 preregistered primary result。
