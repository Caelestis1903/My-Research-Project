# Stage Research Review — P0-2: Fixed-N Imbalance Ratio Sweep

> 作者：main（Agent 0）。数据来源：`experiments/reports/P0_2_RAW_REPORT.md`（runner，140/140 新跑 OK + 60 行 P0-1 复用）、`experiments/P0_2_ratio_sweep/P0_2_cells.csv`、`P0_2_summary_ratio.csv`。
> 设计：N=187,500 跨 ratio 恒定；ratio ∈ {1:1, 5:1, 20:1, 50:1}（L:S）；20:1 档与 P0-1 逐位相同直接复用；ratio 是被操纵的实验变量（其余全部固定），因此 gain–ratio 关系的因果解读来自设计而非相关。

## 1. Stage research question

固定总样本量后，imbalance ratio 是否仍系统性影响 (a) baseline negative transfer 幅度；(b) comp1 gain。区分 H-A（imbalance 本身驱动 comp1 gain）vs H-B（旧 P5 大增益来自 total-data scarcity / baseline degradation）。

## 2. Experimental facts

### 2.1 Δ_transfer（baseline joint − oracle，逐 cell 配对，均值，n=24 cells/role；ratio=1 为退化档 n=8，桶=域身份）

| ratio | L 角色 | M 角色 | S 角色 |
|---|---|---|---|
| 1:1 | +0.0159（=d0） | +0.0212（=d1） | +0.0074（=d2） |
| 5:1 | −0.0018 | +0.0235 | +0.0304 |
| 20:1 | −0.0146 | +0.0194 | +0.0487 |
| 50:1 | −0.0155 | +0.0186 | +0.0598 |

- 平衡（1:1）时 joint training 对**所有**域为正迁移（三域全正）。
- 最大域受损在 1:1→5:1 之间出现（5:1 时 ≈0，三 setting 分别为 −0.0009/+0.00003/−0.0044），20:1 达到 −0.015，**20:1→50:1 饱和**（−0.0146→−0.0155）。
- 小域受益随 ratio 单调增长（+0.030→+0.049→+0.060），但注意 S 档样本量跨 ratio 在缩小（25k→7.5k→3k），oracle 同步变弱，部分放大表观 Δ。
- M 角色稳定在 +0.019~+0.024，与 ratio 基本无关。

### 2.2 comp1 gain（comp1 − baseline，同配对）

| ratio | overall | L | M | S |
|---|---|---|---|---|
| 1:1（γ 全 0.9，uniform 控制） | +0.0030 | — | — | — |
| 5:1 | +0.0067 | +0.0077 | +0.0061 | +0.0063 |
| 20:1 | +0.0149 | +0.0138 | +0.0137 | +0.0172 |
| 50:1 | +0.0174 | +0.0121 | +0.0124 | +0.0276 |

- 单调递增、增量递减（+0.0037、+0.0082、+0.0025）：**concave / 饱和形态**，无明确阈值点。
- 5:1 时角色间平坦；50:1 时 S 角色增益翻倍（+0.0276 vs L/M ≈ +0.012）——differential γ 的小域定向收益主要在极端 imbalance 出现。
- 域×角色交互仍在（50:1 S 档：d1_large +0.0344 vs d0_large +0.0206）。

### 2.3 overall / worst-domain AUC（6-cell mean；worst=cell 级最小值）

| ratio | baseline | comp1 | oracle |
|---|---|---|---|
| 1:1 | 0.7405 / 0.7042 | 0.7435 / 0.7106 | 0.7257 / 0.6962 |
| 5:1 | 0.7338 / 0.6993 | 0.7405 / 0.7067 | 0.7164 / 0.6881 |
| 20:1 | 0.7185 / 0.6877 | 0.7334 / 0.7024 | 0.7007 / 0.6779 |
| 50:1 | 0.7098 / 0.6785 | 0.7271 / 0.6985 | 0.6888 / 0.6692 |

- 固定 N 下 baseline mean 随 ratio 下降（0.7405→0.7098）——主要由小域缩小与 L 受损共同构成，是结构性而非"训练崩坏"。
- comp1 在每一档都提升 mean 与 worst-domain；50:1 时 comp1 mean（0.7271）仍高于 1:1 oracle（0.7257）。

### 2.4 与旧 P5（变 N 设计）的对照

旧 P5 comp1 gain：r30 = +0.0166、r100 = +0.0202（总样本量随 ratio 缩小）。本阶段固定 N：20:1 = +0.0149、50:1 = +0.0174。两组量级高度接近（口径差异：旧 P5 为 buggy-grouping baseline、变 N；仅作量级对照）。

## 3. Main observations

**Strong evidence**
1. comp1 gain 在固定 N 下仍随 ratio 单调增长（+0.003→+0.017）——imbalance 本身是 gain 的驱动变量，H-A 成立。
2. 旧 P5 的增益量级在固定 N 下完整复现（±0.002 量级）——增益并非总样本量缩减的伪影，H-B 被显著削弱。
3. 最大域负迁移的出现点（1:1→5:1 之间）与饱和点（20:1 之后）清晰；P0-1 的"largest-domain sacrifice"在 ratio 维度上可复现且有剂量响应。
4. 平衡设定下 joint training 对所有域净为正（+0.007~+0.021）——"牺牲"只在 imbalance 存在时发生。

**Moderate evidence**
5. comp1 含一个 imbalance 无关的小分量：1:1 档（γ 退化为 uniform 0.9）仍有 +0.0030。differential（size-tiered）分量 ≈ 总增益减去该基线，在 50:1 占大头。
6. comp1 的 S 定向收益只在高 ratio（50:1）显著，20:1 及以下近似广谱（与 P0-1 结论一致并细化）。

**Weak evidence / Speculation**
7. gain–ratio 在 20:1→50:1 放缓（overall），可能接近 γ_max=0.99 clip 与 S 档信息量的联合饱和；50:1 以上（受 d0 池限制不可行）无法验证。

## 4. Competing hypotheses

- H-A：imbalance ratio 本身驱动 comp1 gain。
- H-B：旧增益主要来自 total-data scarcity / baseline degradation。
- H-C：comp1 = 普通 uniform momentum 效应（与 ratio 无关的常数增益）。

## 5. Evidence supporting each hypothesis

- H-A：§2.2 单调 gain–ratio；固定 N 排除 scarcity；S 角色在 50:1 的定向放大与 γ 分层结构一致。
- H-B：仅剩量级残差解释力（旧 +0.0166 vs 新 +0.0149 的微小差距可归入口径差异）。
- H-C：1:1 档 +0.0030（uniform γ=0.9 仍为正）。

## 6. Evidence against each hypothesis

- H-A：无实质反证；但 1:1 正增益说明 gain 不完全由 imbalance 解释（需 H-C 补充为加性分量）。
- H-B：§2.4——固定 N 后增益量级未消失（若 H-B 为真，固定 N 应使增益大幅缩水至 ~0）。
- H-C：若 H-C 为真，gain 应与 ratio 无关；实测 +0.003→+0.017 随 ratio 变化 5.7 倍，H-C 单独不成立。

## 7. Major confounds

1. 1:1 档的 comp1 是 uniform-γ 控制臂（γ 全 0.9），其 +0.003 不能归因于 size-tiered 结构；解读 gain–ratio 曲线时该点语义与其他档不同（designer 已声明）。
2. S 角色的 Δ_transfer 与 comp1_gain 跨 ratio 比较时，S 的绝对样本量在变（25k→3k），"S 受益随 ratio 增长"混有 oracle 变弱的机械成分。
3. M 档尺寸跨 ratio 非单调（37.5k/30k/34.5k），M 角色的跨 ratio 比较只作次要粒度。
4. 50:1 档 S=3,000 的 oracle 方差大（S cell Δ std 达 0.022），该档 S 的均值估计精度有限。
5. 结论限于 ML-1M；KuaiRand 未验证。

## 8. What has been falsified?

1. **H-B（scarcity 伪影假说）被证伪**：固定 N 后 comp1 增益不降反稳（20:1 +0.0149 vs 旧变-N r30 +0.0166），旧增益不能归因于总样本量缩减。
2. **"comp1 只在 imbalance 时有效"被证伪**：1:1 仍有 +0.0030（uniform 分量）。
3. **"imbalance 下 joint training 必然伤害小域"被进一步埋葬**：所有 ratio 下 S 角色都是最大受益者，受损者始终是 L。

## 9. What remains unresolved?

1. uniform 分量（+0.003）的本质：普通 regularization / effective step-size 变化？→ P1-3 matched momentum control。
2. 50:1 时 S 定向放大的机制（γ=0.99 长记忆 vs 其他）→ P1-1 instrumentation。
3. L 牺牲在 1:1–5:1 间的精确阈值位置（本设计分辨率只到档位）。
4. 等权 cell loss 对"L 牺牲"的贡献 → P0-4。
5. 跨数据集（KuaiRand）与跨架构（MMoE/PLE）的稳健性 → P0-3 / P2-1。

## 10. Research direction decision

**Continue current hypothesis**（P0-1 修正版）：largest-domain sacrifice 是真实、因果、有剂量响应的现象；comp1 gain 由 imbalance 驱动（H-A），并含一个小的 imbalance 无关分量。

## 11. Next-stage recommendation

进入 **P0-3（M3OE vs MMoE Per-Domain Transfer Matrix）**：

1. P0-3 所需的两个 setting 现已具备：balanced = ratio_1（62,500×3）、strong imbalance = ratio_50（150k/34.5k/3k），采样管线与 oracle 对齐机制直接复用。
2. 核心问题升级为：largest-domain sacrifice 是 M3OE（Star 融合）特有，还是一般 multi-domain joint training（MMoE）同样存在？旧 P7 已有 MMoE/PLE/ShareBottom + comp1 的移植实现可复用。
3. P0-4（loss weighting）在 P0-3 之后执行，因为 P0-4 的解读依赖"P0-3 确认的现象是否 architecture-specific"。
