# P0-4 Stage Research Review：Loss Weighting + Checkpoint Objective Control

- 阶段：P0-4（Priority 0 第四档）
- 数据：MovieLens-1M，D=3×T=2，固定 N=187,500，strong 档 ratio_50（L/M/S = 150,000/34,500/3,000，三 setting 轮换）
- Arms：M3OE baseline × 3 种训练 loss weighting（equal / sample-proportional / sqrt-frequency，总尺度归一 Σw=6 隔离 lr 混杂）× 2 种 checkpoint selection objective（equal 6-cell 均值 / sample-proportional 加权，后者纯事后重选、零新 run）
- 权重（p=0.8/0.184/0.016）：sample=(2.400, 0.552, 0.048)，sqrt=(1.851, 0.888, 0.262)
- Seeds：2022-2025；36 新 run 全 OK，oracle 36 行复用 P0-2；equal 臂 12 run 与 P0-2 ratio_50 baseline **逐位一致**（36/36 字段锚点 PASS）
- 原始报告：`experiments/reports/P0_4_RAW_REPORT.md`；汇总：`experiments/P0_4_loss_weighting/P0_4_{cells,summary,headline}.csv`
- 已知限制：selection=sample 的事后重选受早停截断，18 个 run（全部在 sample 选点列）argmax 落在最后记录 epoch（censored，equal 臂 4/12、sample 臂 0/12、sqrt 臂 2/12……总计 18/72），sample 选点行的读数是保守近似

## 1. Stage research question

L 牺牲 / S 获益 pattern 是否主要来自训练 objective（6 项等权 BCE）与 model selection objective（6-cell 等权 mean val AUC）？若 pattern 随 weighting 明显改变，必须承认此前所谓 trade-off 部分来自 objective design。

## 2. Experimental facts

headline（L/S 角色 mean Δ_transfer，n=24 cell/role，逐 cell 同 seed 配对 joint − oracle）：

| weighting | selection | Δ_L | Δ_S |
|---|---|---|---|
| equal（现状） | equal（现状） | −0.0155 | +0.0598 |
| equal | sample | −0.0083 | +0.0265 |
| sample | equal | −0.0012 | +0.0973 |
| sample | sample | −0.0008 | +0.0963 |
| sqrt | equal | −0.0053 | +0.0939 |
| sqrt | sample | −0.0029 | +0.0827 |

summary 关键列（每配置 n=72 cell）：

| weighting | selection | Δ_overall | frac_negative | overall AUC | worst-domain AUC | mean sel. epoch |
|---|---|---|---|---|---|---|
| equal | equal | +0.0210 | 34.7% | 0.7098 | 0.6425 | 7.5 |
| equal | sample | +0.0118 | 37.5% | 0.7006 | 0.6056 | 12.8 |
| sample | equal | +0.0420 | 22.2% | 0.7308 | 0.6754 | 7.8 |
| sample | sample | +0.0416 | 22.2% | 0.7304 | 0.6746 | 8.0 |
| sqrt | equal | +0.0391 | 29.2% | 0.7279 | 0.6743 | 6.3 |
| sqrt | sample | +0.0347 | 25.0% | 0.7235 | 0.6671 | 8.5 |

M 角色：equal/equal +0.0186 → sample/equal +0.0301。对照参照：P0-2 同档 comp1-on-equal overall AUC 0.7271、L 残余 −0.0034。

## 3. Main observations

**Strong evidence**
- 训练 loss weighting 是 L 牺牲的主控变量：equal → sample 使 Δ_L 从 −0.0155 移动到 −0.0012（修复 92%），sqrt 居中（−0.0053），三臂单调。变化幅度（0.014）远超判定阈值 0.005。
- weighting 改变不是"拆东墙补西墙"：sample/equal 下 overall AUC 0.7098→0.7308、worst-domain 0.6425→0.6754、Δ_overall +0.021→+0.042 全面改善；S 获益不降反升（+0.060→+0.097，S 的 oracle 太弱，小权重 joint 信号仍远好于单域训练）。
- checkpoint selection objective 是次级但真实的因素：equal 臂内换 sample 选点使 Δ_L −0.0155→−0.0083（修复 47%），代价是 overall AUC 降 0.009（选点目标与评估口径错配的正常代价）。

**Moderate evidence**
- 在 sample/sqrt 训练臂内，选点口径的影响大幅缩小（Δ_L 变化 ≤0.0024）：训练 objective 与选点 objective 存在交互，训练侧修正后选点侧基本无害化。
- "trade-off"框架本身瓦解：sample/equal 下三角色 Δ 全部 ≥0（L −0.0012≈0，M +0.030，S +0.097），不存在有意义的此消彼长。

**Weak evidence**
- sample 选点的 censored 比例 25%（equal 臂），其读数系统性偏保守；sample 选点在 sample/sqrt 臂内 censored=0，结论不受污染。

**Speculation**
- 等权 BCE 下 L 牺牲的本质可能是：小域项把共享参数拉向对小域有利的区域，等权时该拉力与大域相同；sample weighting 恢复样本贡献的真实比例后，拉力平衡、joint 训练对 L 回到中性、对小域仍是纯增益。

## 4. Competing hypotheses

- H1：L 牺牲 / S 获益主要来自训练 objective 的等权设计。
- H2：pattern 主要来自 checkpoint selection 的等权 val 口径。
- H3：pattern 是数据/优化层固有，对 objective design 稳健（零假设）。

## 5. Evidence supporting each hypothesis

- H1：训练 weighting 单调修复 92%（equal−0.0155 → sqrt−0.0053 → sample−0.0012）；三臂唯一差异是 loss 权重。
- H2：固定 equal 训练，仅换选点口径修复 47%；mean selected epoch 从 7.5 移到 12.8。
- H3：（无支持证据）

## 6. Evidence against each hypothesis

- H2：在 sample/sqrt 训练臂内选点口径几乎无影响（≤0.0024），说明选点不是独立主因，而是等权训练的"共犯"——等权训练把模型推向偏袒小域的轨迹，等权选点再从轨迹中挑出偏袒小域的 checkpoint。
- H3：被证伪——pattern 随 weighting 大幅单调移动。
- H1：残余 ~0.001 的 L 负值与 sqrt 臂不完全修复说明训练 weighting 不是全部；但残余已在噪声边缘。

## 7. Major confounds

- sample 选点列 18/72 run censored（早停截断），其 Δ_L=−0.0083 是保守估计；真实在线 sample 早停可能进一步收窄，本阶段不补跑。
- arch update 通道（每 epoch 用 unmasked batch BCE 更新架构权重）本身近似 sample-proportional，三臂一致；_fix_optim_grouping 下该通道只影响 `_weight_` 参数，属已知残余混杂。
- selection=sample 的 overall/worst-domain 读数用了与选点不同的评估口径（等权 mean），跨列比较时注意口径错配是设计意图。
- 本阶段只有 baseline 臂；comp1 × weighting 交互未测。

## 8. What has been falsified?

- **"L 牺牲是数据/优化层固有、与 objective design 无关"——证伪。** 训练 loss weighting 单变量修复 92%，训练+选点合计修复 ~95%。
- **"存在真实的 small-large trade-off"——基本证伪。** sample/equal 配置下三角色全部获益（L≈0、M +0.030、S +0.097），overall 与 worst-domain 同时改善；此前观察到的 trade-off 主要是等权 objective 的人造产物。按 promt.md 要求明确承认：此前所谓"M3OE trade-off"（及其后修正的"一般 MDMT 优化现象"）**大部分来自 objective design**。
- **"comp1 是修复 L 牺牲的必要手段"——被削弱。** sample weighting（纯数据侧改动）把 L 修复到 −0.0012 且 overall 0.7308，优于 comp1-on-equal（−0.0034 / 0.7271）。

## 9. What remains unresolved?

- comp1 在 sample weighting 之上是否仍有增量（comp1 的价值重估需要 comp1×weighting 2 配置，~24 run）。
- sample 选点在线早停（非事后）的真实读数。
- 残余 −0.001 级 L 负值与 arch update 通道的贡献。
- 结论限于 strong 档 ratio_50 与 ML-1M。

## 10. Research direction decision

**Modify hypothesis**：研究问题第三次修正——"多域联合训练中最大域被系统性牺牲"的本质是**等权 training/selection objective 在样本占比悬殊下的系统性错配**，而非数据或架构的固有 trade-off。等权 BCE 使小域在 scalar objective 中获得与其样本量不符的相对权重（3,000 样本域与 150,000 样本域同权），把共享参数拉离大域最优；sample-proportional weighting 恢复比例后，joint training 对所有域变为近似纯增益。后续论文叙事的核心应从"trade-off 及其缓解"转向"objective 错配的诊断与修正"，comp1 的定位需在修正后的 objective 之上重新评估。

## 11. Next-stage recommendation

Priority 0 四档全部完成，研究问题已稳定。建议补一个低成本封口实验后进入 Priority 1：

- **P0-5（建议，~24 run）**：comp1 × {equal, sample} 交互（sample 臂 comp1 与 equal 臂 comp1 已有数据复用），回答"comp1 在修正后的 objective 上是否仍有增量"。若增量消失，comp1 的论文定位降为"等权 objective 下的补丁"；若仍在，comp1 是独立机制，值得 Priority 1 机制实验。此结果直接决定 P1-1（momentum 机制仪表化）的必要性。
- P0-5 完成后进入 P1-2（Curriculum Life-or-Death），其结论同样需要在新 objective 口径下重估。
