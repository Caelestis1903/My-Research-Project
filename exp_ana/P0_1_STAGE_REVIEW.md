# Stage Research Review — P0-1: Fixed Total Sample Size + Domain Size Role Reversal

> 作者：main（Agent 0）。数据来源：`experiments/reports/P0_1_RAW_REPORT.md`（runner，60/60 OK）、`experiments/P0_1_role_reversal/P0_1_raw_dedup.csv`。
> 本文所有 Δ 均由 main 从 runner 报告的逐 cell 原始数值计算（runner §4.3 的聚合口径不符 Δ_transfer(d,t) 定义，未采用；其 §3 原始数据完整可用）。

## 1. Stage research question

在固定总训练样本量（N=187,500，三 setting 完全相同）、val/test 全局固定的前提下，人为轮换三域的大小角色（L=150,000 / M=30,000 / S=7,500，20:4:1），检验 domain size 本身是否因果性决定：(1) 哪个 domain 遭受 negative transfer；(2) 哪个 domain 从 joint training 获益；(3) comp1（per-domain differential EMA momentum）对哪个 domain 最有效。

## 2. Experimental facts

### 2.1 Δ_transfer(d,t) = AUC_joint_baseline(d,t) − AUC_oracle(d,t)（4 seeds 均值）

| Setting | 域(角色) | t0 | t1 |
|---|---|---|---|
| A (d0_large) | d0 (L) | **−0.0080** | **−0.0076** |
| A | d1 (M) | +0.0238 | +0.0266 |
| A | d2 (S) | +0.0372 | +0.0487 |
| B (d1_large) | d0 (S) | +0.0533 | +0.0445 |
| B | d1 (L) | **−0.0144** | **−0.0145** |
| B | d2 (M) | +0.0117 | +0.0061 |
| C (d2_large) | d0 (M) | +0.0260 | +0.0223 |
| C | d1 (S) | +0.0501 | +0.0583 |
| C | d2 (L) | **−0.0196** | **−0.0233** |

按角色聚合（每角色 6 个 domain×task 观测的均值）：

| 角色 | Δ_transfer 均值 | 符号一致性 |
|---|---|---|
| L（最大域） | **−0.0146** | 6/6 为负（跨 3 域 × 2 任务） |
| M | +0.0194 | 6/6 为正 |
| S（最小域） | +0.0487 | 6/6 为正 |

逐 seed 层面同样一致：L 角色的负 Δ 在全部 4 seeds × 3 settings 中稳定出现（runner §3 数据可逐 seed 复核）。

### 2.2 comp1 gain(d,t) = AUC_comp1 − AUC_baseline（同 setting 配对，4 seeds 均值）

| Setting | 域(角色) | t0 | t1 | 域均值 |
|---|---|---|---|---|
| A | d0 (L) | +0.0077 | +0.0065 | +0.0071 |
| A | d1 (M) | +0.0089 | +0.0016 | +0.0052 |
| A | d2 (S) | +0.0098 | +0.0043 | +0.0071 |
| B | d0 (S) | +0.0225 | +0.0273 | +0.0249 |
| B | d1 (L) | +0.0142 | +0.0173 | +0.0158 |
| B | d2 (M) | +0.0167 | +0.0230 | +0.0199 |
| C | d0 (M) | +0.0153 | +0.0165 | +0.0159 |
| C | d1 (S) | +0.0231 | +0.0164 | +0.0198 |
| C | d2 (L) | +0.0176 | +0.0195 | +0.0186 |

- **36/36 setting×cell 全为正**（无一个 cell 受损）。
- 按角色聚合：L +0.0138，M +0.0137，S +0.0173 —— 角色间差异很小。
- 按域×角色看存在强交互：d0 的 gain 随其体型减小单调上升（L +0.007 → M +0.016 → S +0.025）；d2 恰好相反，作为 S 时 gain 最小（+0.007）、作为 M/L 时 +0.019；d1 在 M 档凹陷（+0.005）。
- comp1 后 L 角色的 Δ_transfer 从 ≈−0.015 收窄到 ≈−0.001（6 个 L cell：−0.0003/−0.0011/−0.0001/+0.0028/−0.0020/−0.0038），即 comp1 几乎抹平大域负迁移，但它对 M/S 域的提升幅度并不小于 L 域。

### 2.3 域的固有难度（oracle，L 角色同 150k 样本量对比）

oracle t0/t1：d0 = 0.7503/0.7623；d1 = 0.7523/0.7736；d2 = 0.7225/0.7427。同量数据下 d2 稳定最低 → 域难度是身份（分布）属性，不是 size 属性。

### 2.4 其他事实

- joint best_epoch ∈ [4,10]，oracle best_epoch ∈ [3,31]（oracle 收敛更慢/更晚）。
- 双 orchestrator 事故：55 个任务被两棵树各跑一次，重复行 per_dt_test_auc **逐位相同**（同 seed 同子集确定性复现），去重后 60/60 OK；该事故不构成数据质量问题，反而提供了确定性佐证。
- P0-1 baseline 使用 `--fix_optim_grouping 1`，与旧 P5 baseline（B10 buggy grouping）数值不可直接比（designer 裁定，已记录）。

## 3. Main observations

**Strong evidence**

1. **Negative transfer 跟随"最大域角色"移动，与域身份无关**：三个 setting 中，负迁移恰好落在当期最大域上（d0/d1/d2 各中一次），6/6 cell 符号一致，且 4 seeds 稳定。旧 P5 自然分布下"相对最大域 d1 负迁移（−0.0215）"的历史观察与此一致——当时 d1 恰好是最大域。
2. **Transfer 方向对 size role 呈单调剂量响应**：S (+0.049) > M (+0.019) > 0 > L (−0.015)，三个域上重复三次均成立。joint training 在本设定下系统性地"从最大域向中小域迁移"。
3. **comp1 gain 不跟随 size 角色**：角色均值几乎平坦（L/M/S ≈ +0.014/+0.014/+0.017），36/36 cell 为正；comp1 是广谱改善，而非"定向补偿小域或大域"的机制。

**Moderate evidence**

4. comp1 gain 存在 domain×role 交互：d0 越小 gain 越大，d2 相反。γ_d 公式（S→0.99、L→0.90）随角色分配，但 gain 排序与 γ 排序不一致（d2 拿最大 γ=0.99 时 gain 反而最小）→ comp1 的收益不由"分配给该域的 EMA 记忆长度"单调决定。
5. 域固有难度（d2 最难）是身份属性：同量数据 oracle 对比下 d2 稳定最低（~0.72 vs ~0.75），且 d2 作为 L 时负迁移幅度最大（−0.020/−0.023 vs d0 的 −0.008）。

**Weak evidence / Speculation**

6. 小域 S 的大正 Δ（+0.05）可能部分反映 oracle 在小样本下本就偏弱（7.5k 样本），joint 训练的共享结构带来正则化收益——这是"size causality"的另一种表述，不冲突，但提示 Δ_transfer 的"获益"端与"受损"端机制可能不对称。

## 4. Competing hypotheses

- **H1（size causality，强形式）**：transfer 方向与 comp1 gain 都由 size role 决定。
- **H2（domain identity）**：受损/获益由域身份（分布、难度）决定，与大小无关。
- **H3（交互/修正版 size causality）**：transfer 方向由 size role 因果决定；comp1 gain 为广谱效应，其幅度由域身份×角色交互调制。

## 5. Evidence supporting each hypothesis

- H1：§2.1 的 L/M/S 角色模式（方向部分）；S 角色 comp1 gain 角色均值略高（+0.0173）。
- H2：d2 固有难度身份绑定（§2.3）；d2 作为 L 受损最重；comp1 gain 的域×角色交互（§2.2）。
- H3：方向完全跟随角色（支持 size）；comp1 gain 角色间平坦、域间交互显著（§2.2）；γ 排序与 gain 排序解耦。

## 6. Evidence against each hypothesis

- H1（强形式）：被 §2.2 否定——comp1 gain 不跟随角色（L 与 M 均值相同，d2 在 S 档 gain 最小且 d2 在 S 档拿最大 γ）。
- H2：被 §2.1 否定——没有任何域"无论大小始终受损"；每个域恰好只在担任 L 时受损。d2 虽最难，但它在 M/S 角色下仍是正迁移最大受益者之一。
- H3：目前无直接反证；其"comp1 广谱"论断与旧 P6（增益几乎全部来自 per-domain momentum 本体）一致。

## 7. Major confounds

1. **L 角色池耗尽不对称**：d0 担任 L 时用掉其训练池的 ~96%（150,000/155,930），而 d1/d2 担任 L 时只用 ~47%。d0-as-L 的子集代表性高于其他域的 L 档；但负迁移在三域上一致出现，该不对称不改变方向性结论。
2. **等权 cell loss 混杂**：训练目标是 6-cell 等权 BCE 求和，最大域在 scalar objective 中并无更大权重，却承载更多独特样本——"L 受损"可能部分由 objective 设计（而非纯粹优化动力学）造成。**这正是 P0-4 要分离的混杂。**
3. **Early stopping 口径**：joint 用 6-cell 等权 mean val AUC 做 model selection，天然偏向保护小/弱 cell；oracle 用 2-cell mean。两侧选择目标不完全对称，Δ_transfer 含此口径差。
4. **S 档 oracle 偏弱**（7.5k 样本）放大正 Δ 的表观幅度（见观察 6）。
5. runner §4.3 聚合口径错误（6-cell 均值 vs 2-cell 均值），本报告未采用该节数字；不影响原始数据可用性。

## 8. What has been falsified?

1. **"小域遭受 negative transfer"（small = victim）被证伪**：最小域 S 角色在 3/3 settings、6/6 cells 全部是 joint training 的最大受益者；受损的恰是最大域。
2. **"negative transfer 绑定域身份"（强 H2）被证伪**：负迁移跟随角色轮换，每个域都在且仅在担任最大域时受损。
3. **"comp1 是定向补偿某个 size 角色的机制"被证伪/削弱**：comp1 广谱为正（36/36），角色间增益平坦；其价值更接近"全面改善 + 顺带抹平大域负迁移"。
4. **"d2 难是因为小/稀疏"被进一步否定**：d2 在同量 150k 数据下依然最难（身份属性）。

## 9. What remains unresolved?

1. L 域受损的微观机制：是等权 objective（→ P0-4）、梯度干扰（→ P1-1 插桩）、还是 shared 表示被多数域平均化？
2. comp1 广谱增益的机制（H-M1~H-M5 未区分，留待 P1-1/P1-3）。
3. comp1 gain 的域×角色交互从何而来（为何 d2 的响应曲线与 d0/d1 相反？）。
4. 本结论是否在 KuaiRand 上成立（本阶段仅 ML-1M）。
5. imbalance ratio 的剂量响应（本阶段只有一个 ratio 20:1）→ P0-2。

## 10. Research direction decision

**Modify hypothesis**。

- size causality：**supported，但方向与论文原叙事相反**——不是"小域受损、大域受益"，而是"最大域被牺牲、中小域受益"。
- 论文框架仍可以 size trade-off 为核心，但必须改写为 **"largest-domain sacrifice / size-role-asymmetric transfer"**：受损端是大域，受益端是中小域；且 comp1 的定位应从"补偿小域"改为"广谱增益（含抹平大域牺牲）"。
- domain identity 假设：不被否定而是被重新定位——它决定**难度基线与受损/受益的幅度**（d2 最难、作为 L 受损最重），但不决定方向。

## 11. Next-stage recommendation

按总规划进入 **P0-2（Fixed-N Imbalance Ratio Sweep）**，理由：

1. P0-1 已在单一 ratio（20:1）确立方向性因果；P0-2 的 1:1 / 5:1 / 20:1 / 50:1 固定-N 扫档可直接检验"L 受损幅度与 S 受益幅度是否随 ratio 单调/饱和/存在阈值"，并区分 Hypothesis A（imbalance 本身导致 comp1 gain）vs B（旧结果主要来自总样本量变化）。
2. P0-1 已建成的 role-reversal 采样管线与对齐 oracle 可被 P0-2 直接复用（只需改 role_sizes），边际成本低。
3. 建议 P0-2 固定诊断价值最高的角色配置：保留三 setting 轮换（本阶段已证明角色轮换是核心鉴别手段），每 ratio 跑同样的 60-run 矩阵；1:1 档（balanced）同时为 P0-3 提供 mild/balanced setting。

**等待用户确认后开启 P0-2。**
