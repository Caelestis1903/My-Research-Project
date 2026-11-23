# 06 — 指标与统计规范

## 1. 共同结果指标

每个 seed / dataset / arm / anchor：
- 六格 AUC、六格 LogLoss；
- Macro AUC（六格等权平均）；
- Macro LogLoss（六格等权平均，单独报告）；
- domain macro、task macro、worst cell；
- pooled AUC 仅作为额外描述，不替代六格 Macro；
- validation selected epoch（仅完整训练阶段）；
- pairwise margin、pair correctness correction/destruction/tie；
- 参数位移、梯度一致性和 optimizer state 变化；
- forward/backward/optimizer step 数、GPU time、peak memory。

## 2. M1 专属

- L_cons before/after；
- 六个 g_U,c 与 g_U,J cosine；
- pairwise cosine matrix；
- normalized direction residual；
- per-group gradient norm；
- P displacement、U unchanged proof；
- consistency gain 与 cell AUC/pairwise changes 的 seed-paired relationship。

L_cons 降低属于机制信号，不等于 AUC 收益。

## 3. M2 专属

- per-row/task/cell score 分布；
- score 与实际局部 pair-margin change 的校准；
- task-specific vs task-agnostic score 的差异；
- top/bottom group 的有限真实 fork 方向；
- label support、row counts、score ties；
- score 与 loss、gradient norm、label frequency 的相关关系，仅作为代理比较。

不把 score 排名本身作为性能指标。

## 4. M3 专属

- A-only、B-only、A/B、A/A、B/B 的 paired delta；
- interaction residual；
- E10/E25/E40 的阶段差；
- 每个组的 row IDs、label support、batch size、总 loss mass。

## 5. 统计单位与估计

五个 seed 是独立重复单位；同 seed 配对。anchors/forks/cells/pairs/samples 仅是 seed 内重复观测。每个结果必须保留五个 seed 原始值。

报告：
- paired mean delta；
- sample SD；
- 95% t interval（df=4）；
- 正向 seed 数；
- 每格配对 delta；
- 每个 anchor 分别结果与预注册 anchor reduction；
- 不能将 anchor 数作为 n=15。

S1 的三个 anchors 先在每 seed 内按预注册等权平均，再形成五 seed paired summary；另外完整列出 anchor-specific effects。

## 6. 多机制比较

- M1 为唯一 primary mechanism family。
- M2 为 secondary mechanism family。
- M3 为 conditional exploratory mechanism family。
- M4/M5 feasibility only。
- S1 筛选使用 V_A；S2 确认使用 V_B。
- 每个 family 内对多个 arms 的比较采用 Holm correction；机制代理指标单独分族，不与 AUC 主检验混合。
- 不允许事后替换 primary contrast。

## 7. 晋级阈值

机制阶段不强制 +0.003。S1/S2 主要要求：
- 操作成功且状态控制通过；
- 机制信号方向达到预注册；
- 至少 4/5 seeds 方向一致作为强支持描述，但不是单独显著性规则；
- V_B 独立确认不由单一 seed/anchor 驱动；
- cell guard：mean cell delta ≥ −0.001；
- 代理改善必须对应独立 pairwise ranking 或 AUC 变化，才能进入 S3。

S4 完整算法：
- 五 seed paired Macro AUC mean gain ≥ +0.003；
- 每个 cell mean delta ≥ −0.001；
- LogLoss 单独报告；
- 正式比较 M3oE、HyperGate、AutoIFS；
- validation six-cell Macro AUC 选点，Test 只在 freeze 后评价。

## 8. AUC / LogLoss 分歧

AUC 是排序指标，LogLoss 是概率质量指标。两者不同向时保留两套结果，不以其中一个解释或替代另一个。Test 不参与筛选或机制判断。
