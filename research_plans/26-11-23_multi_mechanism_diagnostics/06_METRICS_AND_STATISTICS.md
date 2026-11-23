# 06 — 全机制指标、统计与多重比较

## 1. 共同指标

每个 dataset × seed × variant × anchor × arm 记录：
- 六格 AUC、六格 LogLoss、support/positive/negative counts；
- six-cell Macro AUC / Macro LogLoss；
- domain macro、task macro、worst cell；
- pooled AUC 仅作补充；
- V_A 与 V_B 分开报告；
- pairwise margin、pair correctness、correction/destruction/tie；
- 参数组位移、gradient geometry、optimizer state；
- forward/backward/optimizer steps、EMA ticks；
- GPU/CPU/wall time、peak VRAM、host RAM。

## 2. M1 指标

L_cos、L_mag、L_pair；六格 g_U,c 与 g_U,J cosine；pairwise cosine matrix；gradient norm/direction residual；P/U displacement；P-only step 后 U/Adam unchanged receipt；pair margin 与 AUC response。

## 3. M2 指标

row-task-cell influence score；full/leave-one-out virtual Adam displacement；score distribution；score 与实际 pair-margin delta 的 calibration；task-specific vs task-agnostic/shuffled；weights、loss mass、row exposure。

## 4. M3 指标

E10/E25/E40 score sign and magnitude；A/B/A-B/A-A/B-B outcomes；parameter/function interaction residual；group membership hash；阶段×组交互。

## 5. M4/M5 指标

M4：feature-group dependence、conditional permutation effect、representation/logit/margin change、task/domain heterogeneity、regularizer/sham comparison。

M5：变换合法性、字段有效率、label-pair consistency、domain preservation、prediction consistency、augmentation exposure、六格 AUC/LogLoss。标签保持率不等于语义因果证明。

## 6. M6/M7/M8 指标

M6：J_c、predicted margin delta、trust-region constraint residual、step scaling、worst-cell response 与实际结果差异。

M7：五 seed score distribution、mean/SD/quantile、sign stability、robust score、leave-one-seed-out held-out prediction。

M8：pairwise conflict matrix、projection residual、cell weights、active set、gradient norm、共同下降可行性、实际 cell outcomes。

## 7. 统计单位

五个 seed 是独立重复单位，使用同 seed 配对。anchors、forks、cells、pairs、rows、feature groups 都是 seed 内重复测量。S1 的三个 anchor 先在 seed 内按预注册方式汇总，再形成五 seed paired statistics；同时保留 anchor-specific table。

报告：
- per-seed values；
- paired mean delta；
- sample SD；
- 95% paired t interval (df=4)；
- positive/negative seed counts；
- 六格保护线；
- 每个 anchor 的异质性；
- costs。

不要求所有 seed 同向；不得用一个 seed 单独否决。不得把 30 anchors 写成 n=30。

## 8. 多机制比较

本轮八个 family 都是预注册候选，不设 primary/secondary 排序。
- Family-level confirmatory contrasts：八个 family primary contrasts 使用 Holm correction。
- Family 内变体比较：报告所有变体，探索性检验使用 Benjamini–Hochberg q-values。
- 五组 cross-family interaction 单独成族并校正。
- 机制代理指标与 AUC 主结果分开校正和解释。
- 所有失败、负向、无差异和 NOT_APPLICABLE 结果进入同一汇总表。

不得根据结果事后改变 family primary contrast、筛选门槛或报告范围。超参选择在每个 variant 内基于五 seed Validation 综合结果完成；不得使用单 seed 选择。

## 9. 机制与算法阈值

局部机制诊断不强制 +0.003。需要：
- 动作正确执行；
- 预期对象出现可测变化；
- V_A/V_B 方向及条件边界如实报告；
- 不被单个 seed/anchor 驱动；
- 与实际 pair/AUC 响应关联。

S4 正式算法：
- 五 seed paired Macro AUC mean gain ≥ +0.003；
- 每格 mean delta ≥ −0.001；
- Validation 六格 Macro AUC 选点；
- LogLoss 独立评价；
- M3oE、HyperGate、AutoIFS 正式比较；
- Test 只在 freeze 后评价。

## 10. AUC 与 LogLoss

AUC 是排序，LogLoss 是概率质量。不同向时保留两种结论，不互相替代。任何仅改善 calibration / LogLoss 的机制不得写成 AUC 收益。
