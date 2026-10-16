# 2026-10-16 v3：运行矩阵与执行顺序

## 0. Run naming

统一：

`<phase>__<dataset>__<optimizer>__<condition>__seed<seed>`

每个 run 必须记录 code commit 和 config hash。

---

## 1. Tier A：核心闭环，必须完成

### A1 Objective audit

KuaiRand：

- fixed checkpoints：epoch 1/5/10/20/40；
- SN/SB；
- ≥512 diagnostic batches；
- exact domain gradient；
- Adam 与 SGD 至少各覆盖关键 checkpoint。

输出：

- exact objective gradient；
- estimator mean；
- covariance；
- bias；
- block geometry。

### A2 Overfitting / trajectory

KuaiRand：

- Adam；
- SGD；
- SN/SB；
- seeds 2022–2025；
- epoch 1..50；
- 若必要延长至 100+。

输出：

- full trajectory；
- overfit onset；
- pre-overfit AULC；
- common epoch；
- selected epoch；
- selection uncertainty。

### A3 Optimizer calibration

Pilot：

- SGD LR 0.03/0.1/0.3/1；
- Adam LR 0.003/0.01/0.03/0.1；
- 2 seeds；
- validation only。

Final finalist：

- 4 seeds。

### A4 Repetition

KuaiRand：

- SN-WR；
- SN-PERM；
- SB-WR；
- SB-PERM；
- SGD 4 seeds；
- Adam 4 seeds作为 interaction sensitivity。

### A5 Order

KuaiRand：

- SN 原始 batch order；
- SN same-multiset permuted order；
- SB 原始 batch order；
- SB same-multiset permuted order；
- 每个 4 seeds。

注意：这里必须是 exact same batch contents，只改变 batch order。

### A6 Batch/optimization budget

至少：

- current batch (B_0)；
- (2B_0) 或等价 accumulation；
- 4 seeds；
- 样本预算一致。

### A7 Cross environment

第二环境：

- SN/SB；
- Sample objective；
- primary optimizer；
- 4 seeds。

优先使用当前项目已经存在并经过工程验证的 MovieLens/domain configuration，不随意换新模型。

---

## 2. Tier B：机制闭环，按 Tier A 结果触发，但属于 v3 必须覆盖的候选机制

### B1 Gradient geometry

- epoch 1/5/10/20/40；
- SN/SB；
- ≥512 batches；
- C/P/R/E；
- exact gradient reference。

### B2 Shared/private intervention

先 local one-step：

- C blocked；
- R blocked；
- P/E blocked作为 control。

若某 block 是强候选，再做 full training privateized control。

### B3 Domain size

- original；
- small-domain 2x duplication；
- small-domain 4x duplication。

保持 empirical row distribution 不变。

### B4 Difficulty

- clean；
- mild corruption；
- strong corruption。

只作为 sensitivity / controlled stress test，不宣称自然 difficulty 的完全因果识别。

### B5 AdaTask

只有在实现能够严格保持 Sample objective 时执行正式 comparison。

至少：

- SN/SB；
- 4 seeds；
- selected + common epoch；
- optimizer-state diagnostics。

---

## 3. Tier C：最终 novelty / method gate

### C1 Literature audit

搜索：

- objective-equivalent sampling；
- importance sampling；
- domain balancing；
- domain order；
- batch order；
- optimizer memory；
- Adam state；
- task-wise optimizer；
- sample reweighting；
- MDMT/MoE recommendation。

### C2 D3/SAF

先 objective audit，再决定是否需要 full reproduction。

### C3 Method candidate

只有 GO-METHOD 后才设计。

不得在此之前预设具体算法。

---

## 4. Suggested execution order

[
V0
ightarrow
V1
ightarrow
V2
ightarrow
V3
ightarrow
V4
ightarrow
V5
ightarrow
V6
ightarrow
V7
ightarrow
V8
ightarrow
V9
ightarrow
V10
ightarrow
V11
ightarrow
V12
ightarrow
V13
ightarrow
V14
]

其中：

- V1/V2/V3/V4 是第一优先级；
- V5 是当前最重要的 mechanism isolation；
- V6/V7 只有在 V5 后仍存在 unexplained effect 时升级；
- V8 用于 domain/task explanation；
- V9 判断边界；
- V10 判断 novelty；
- V11 防止重新误回 Adam root-cause；
- V12 才做最终 causal synthesis；
- V13 才允许方法化。

---

## 5. Minimum completion rule

不得因为某一阶段得到漂亮结果就跳过其他未关闭的 blocking question。

如果：

> exposure 已经解释了 effect，

则后续复杂 Adam-state 实验可以按 Stop-6 正式关闭。

但必须把“为什么没有继续做”写进 receipt。

---

## 6. Required artifacts per phase

每个 phase 必须产生：

- design receipt；
- run manifest；
- raw result；
- analysis table；
- decision receipt；
- question updates。

最终：

- `FINAL_REPORT.md`
- `FINAL_QUESTION_REGISTRY.md`
- `FINAL_LITERATURE_BOUNDARY.md`
- `FINAL_DECISION.json`
- 所有 raw data / hashes。
