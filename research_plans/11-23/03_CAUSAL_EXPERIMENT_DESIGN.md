# 03｜Causal Experiment Design

## 1. S1 锚点与状态

每个 dataset × seed 标准 M3oE fixed1 Clean 轨迹训练 50 epochs。锚点为 E10、E25、E40 完成后的状态；每个锚点保留：
- online 参数/buffers；
- 完整原生 Adam state（每参数 exp_avg、exp_avg_sq、step、param group）；
- fixed EMA 参数/buffers；
- Python/NumPy/CPU/CUDA RNG；
- DataLoader permutation/cursor/generator；
- optimizer update count、epoch/batch；
- 数据/模型/配置/source hashes。

锚点从下一 epoch 的两个首批完整自然 batch A、B 建立 fork。若首批不足两个完整 batch，按同 epoch 后续最早两个完整 batch 取样并登记。诊断 fork 不改变 live baseline 状态。

Validation 诊断面板固定拆分为 V_A、V_B。按 domain 对自然验证行做 deterministic keyed split，保持自然行权重，不做类别重采样。每个 cell 必须报告正负支持数；支持不足时使用全该 cell validation 人口并标注，不得伪造 AUC。

## 2. 参数组和实际干预

参数组定义见 01、05。对每个锚点建立独立 clone。所有 fork 从同一个 pre-state 开始，optimizer state 和 RNG 一致。

### 2.1 单块更新控制

- JOINT-2-AB / JOINT-2-BA：两个连续 batch 上执行完整自然联合 Adam 更新。
- JOINT-ACC：A、B 两批在同一 pre-state 上计算梯度并按真实行数累积，执行一次完整联合 Adam 更新。
- U-ONLY-ACC：同样累积 A/B 梯度，仅更新 U。
- P_CELL-ONLY-ACC：仅更新 P_CELL。
- P_AXIS-ONLY-ACC：仅更新 P_AXIS。
- P_FULL-ONLY-ACC：仅更新 P_FULL。

所有臂使用相同自然 BCE 定义；梯度累计按实际 batch 行数加权，不能简单平均不同大小 batch 的 mean loss。

### 2.2 更新顺序干预

对每个 scope (P_s\in\{P_CELL,P_AXIS,P_FULL\})：
- U→P_s：在 A 上只更新 U；重新 forward B，在更新后的 U 上只更新 P_s。
- P_s→U：在 A 上只更新 P_s；重新 forward B，在更新后的 P_s 上只更新 U。

每个方向均执行 AB 与 BA 两种 batch 映射：
- U_A→P_B 与 U_B→P_A；
- P_A→U_B 与 P_B→U_A。

这样可将 batch 难度/行序差异与更新顺序分开。报告两个映射的单独结果及平均结果，不只报最优映射。

### 2.3 状态语义

- 更新某组时，其余参数 `grad=None`，不执行 optimizer step、weight decay 或 step counter 增长。
- 显式零梯度与 None 不得混淆。
- Adam 的 m/v/step 继承 fork pre-state；每个参数仅在自身实际更新时推进 step。
- 不为每个 cell 创建新的独立模型或 optimizer；P_CELL 的六个 Gate/Tower 保留源模型原始参数共享结构。
- EMA 在机制诊断中只作为次级读出；按每个 fork 的两个 batch 时间 tick 更新，所有臂规则一致。Online 是主要机制读出。
- 每个 fork 结束后验证 live baseline 参数、optimizer、EMA、RNG、loader hash 未改变。

## 3. 评价人口

- 完整 Validation：每个 fork 结束后评价六格 AUC/LogLoss、Macro、Traffic、Pooled。
- V_A：仅用于 Stage 1 scope 排序。
- V_B：独立确认 scope 排序，不用于 Stage 1 初始选择。
- 固定 validation pair panel：在 Stage 1 开始前生成并冻结 row IDs/pair IDs，用于 pair correction/destruction 统计。
- Test：S1 完全不可访问。

每个 fork 评价至少包括：
1. 六格 AUC、LogLoss、support；
2. Macro AUC、Macro LogLoss；
3. traffic/pooled 指标（与 Macro 分开）；
4. online 与 EMA；
5. 每 cell logit shift、pair margin shift；
6. corrected / damaged / unchanged pair 数；
7. 参数组更新范数、实际 Adam 位移、更新次数；
8. 运行耗时与 forward/backward 计数。

## 4. S1 scope 选择规则

先在 V_A 对每个 scope 计算跨五 seed、跨 E10/E25/E40 的配对均值：
- U→P_s − U-ONLY；
- U→P_s − P_s→U；
- U→P_s − JOINT-ACC。

候选 scope 的排序主量是 V_A 的 Macro AUC 增量；六格平均 ΔAUC 作为保护约束。V_B 对冻结后的首选 scope 复核。不得在 V_B 上换成次优 scope。

首选 scope 必须满足：
- V_A 上 U→P_s 的 Macro AUC 均值高于 U-ONLY、P_s→U 和 JOINT-ACC；
- V_B 上 U→P_s 对 P_s→U 与 JOINT-ACC 的 Macro AUC 均值仍为正；
- 没有 cell 的五 seed 平均 AUC 低于 -0.001；
- 至少一个训练阶段（E10/E25/E40）呈现相同顺序优势，避免仅一个偶然锚点驱动；
- 不是只改善 LogLoss 而 AUC 无增量。

不以 95% CI 跨零单独否决；但必须报告 CI、逐 seed 和阶段差异。若无 scope 满足，S1 NO-GO。若两个数据集方向不同，按跨数据集等权主量选出的 scope 必须在 V_B 至少不出现双数据集负向；若出现明显 dataset interaction，保留适用边界，不宣称统一机制。

## 5. S2 训练算法

选择唯一 scope 后，训练 U→P 和 P→U 两种 block schedule。每个 epoch 中，按原 DataLoader 顺序将两个完整 batch 组成一个 cycle：
- U→P：第一个 batch 更新 U；第二个 batch 在更新后的 U 上更新选定 P scope。
- P→U：第一个 batch 更新 P scope；第二个 batch 在更新后的 P 上更新 U。
- J_ACC：两个 batch 在同一个 pre-state 上累积梯度，执行一次完整联合 Adam 更新。
- J_ORIG：保留原始每 batch 完整联合 Adam。

为了保证每个样本都参与训练：
- 只对两个完整 batch 组成的 pair 执行 block schedule；
- 每 epoch 末尾剩余的一个或两个 batch 对所有 arms 使用相同的原始联合 Adam tail rule；
- 不丢弃自然尾批，不跨 epoch 拼 batch。
- 若数据/loader batch 数与预期不同，按实际批次数生成 schedule 并写入 manifest。

所有 arms 使用相同的 batch row IDs。U/P 更新在各自 batch 上使用真实自然目标；不做 cell reweighting、不平衡采样、不额外数据增强。

## 6. 竞争解释控制

| 竞争解释 | 对照 |
|---|---|
| 额外计算/额外反向 | 每臂每 pair 两次 forward/backward；记录真实计数 |
| 单纯减少 optimizer update 次数 | J_ACC 与 J_ORIG |
| 仅是更新顺序/批次难度 | AB/BA 双映射、P→U |
| 只是多更新专有参数 | P_ONLY、P→U、U→P 对照 |
| 只是 LogLoss/calibration 改善 | 六格 AUC、pair transition、LogLoss 分开 |
| 只是一个 cell 牺牲换 Macro | 六格保护线与逐格报告 |
| EMA 时间刻度不同 | 每 consumed microbatch 一个 EMA tick；online 为主要读出 |
| 额外梯度信息改变 RNG/loader | 诊断 RNG 隔离、row IDs hash 核对 |

## 7. 不执行的干预

不做：
- Test 上的参数组选择；
- 随机/打乱域任务标签后训练；
- 通过人为劣化某个任务再“修复”；
- 为了制造 head lag 而先冻结头部长时间训练；
- 通过额外增加任务头容量制造可恢复性；
- 动态学习率/EMA rho/样本权重；
- 多轮预测器拟合与在线控制器搜索。
