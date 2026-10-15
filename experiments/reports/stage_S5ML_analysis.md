# Stage S5-ML ANALYSIS REPORT — Main Agent

**Scope.** S5-ML 是 Stage S5（KuaiRand, history-dominant Pattern A）的 MovieLens 跨数据集复现扩展（规格：`promt/S5_add.md`；设计：`experiments/reports/stage_S5_design.md`）。四臂定义与 S5 逐位一致：F = FULL history + FULL current；T = TARGET history + TARGET current；HT = TARGET history + FULL current；DH = FULL history + TARGET current。条件：MovieLens `d2_large`、Equal objective、M3OE、seeds 2022–2025、50 epochs、natural 域比例（未做任何 ratio 操纵）。

**Dataset / target identity（§十二.1–.2）.** domain 定义沿用项目已审计配置：d0={1,18}，d1={25}，d2={35,45,50,56}；t0=rating>3，t1=rating>4。target = d2（local_domain 2，physical `ages_35_45_50_56`，Large role），依据 `stage_S3_protocol.TARGETS[("movielens","d2_large")]` 与 S2 正式报告；primary target task = d2t0，secondary = d2t1。数据/系数表/初始化均由 hash 钉死并与 S2/S3 正式产物一致（`MOVIELENS_DATA_SHA256=f060ecb8…`、`EQUAL_TABLE_SHA256=6e7c8177…`、四 seed `INITIAL_PARAMETER_SHA256`）。

**Execution fidelity.** Phase A（F/T × 4 seeds）与 Phase B（HT/DH × 4 seeds）共 16/16 canonical runs 全部 OK；wave 内双进程并发证据、per-run protocol gate、四臂 `PAIR_FIELDS` + `batch_order_hash` 跨相配对审计、promotion journal 全部通过；moment gate（epoch 0/1 × m/v × 4 seeds）**8/8 全通过**（无 KuaiRand 的 15/16 限定）。执行史：wave-1 attempt1 曾因 write 路径 protocol/observed 键不对称的确定性 bug 中止（训练本身完整），经 coder→reviewer 修复后按 S5 先例归档 `.superseded_wave1` 并重跑；reviewer Round 3 修复了 report 联合读取的两个 bug。以上均为工程事件，不涉及任何算法/配置/seed 变更。数据来源：`experiments/stage_S5ML_outputs/stage_S5ML_{runs,raw_results,paired_results,aggregates,probe,gradient_steps,moment_audit}.csv`（16 runs / 672 raw / 1008 paired / 1600 probe / 4800 gradient_steps / 28 moment_audit）。

**Conventions.** 沿用 S5：practical threshold 0.003；3/4-seed 与多数 common-epoch 稳定性规则；AUC 差 = left − right；LogLoss 差为负表示有利。KuaiRand 对照值引自 `experiments/reports/stage_S5_analysis.md`（target = d1/tab1）。

---

## 1. Phase A replication gate：PASSED（§十二.3）

d2t0 selected ΔAUC(T−F)：+0.015716 / +0.024326 / +0.017056 / +0.020533（seeds 2022–2025），**mean = +0.0194，4/4 为正**（≥ +0.003 且 ≥3/4）；跨 seed 均值的 common-epoch 轨迹 5 个预注册观察点中 4 个为正（ep9 +0.0135、ep19 +0.0034、ep29 −0.0010、ep39 +0.0039、ep49 +0.0001）；无单 seed 驱动（幅度 0.016–0.024 同量级）。LogLoss 一致：d2t0 selected ΔLogLoss(T−F) 4/4 为负（−0.0031 ~ −0.0189）。secondary target d2t1 selected：+0.0078 / +0.0233 / +0.0030 / +0.0229，mean = **+0.0143，4/4 为正**，完整报告未隐藏。

**结论：MovieLens 中存在稳定、达 practical magnitude 的 F→T target rescue（RQ1 = YES）。** 幅度约为 KuaiRand 的一半（+0.019/+0.014 vs +0.041/+0.042）。

## 2. 四臂核心数字（selected checkpoint，4-seed mean ΔAUC）

| Contrast | d2t0 | d2t1 | KuaiRand d1t0/d1t1（对照） |
|---|---|---|---|
| T−F | **+0.019408** (4/4) | **+0.014270** (4/4) | +0.0414 / +0.0423 (4/4) |
| HT−F | **+0.018114** (4/4) | **+0.014149** (4/4) | +0.0411 / +0.0405 (4/4) |
| DH−F | +0.005859 (3/4) | +0.003583 (2/4) | +0.0009 / +0.0008 (2–3/4) |
| HT−T | −0.001294 (|·|<0.003 in 3/4) | −0.000121 (4/4 within) | −0.0003 / −0.0017 |

Aggregate recovery ratio（mean(HT−F)/mean(T−F)）：d2t0 = **93.3%**，d2t1 = **99.2%**。DH recovery：d2t0 = **30.2%**，d2t1 = **25.1%**（KuaiRand：HT ≈ 96–99%，DH ≈ 2%）。

DH−F 的 seed-level 分布（d2t0）：+0.0012 / −0.0016 / **+0.0083** / **+0.0155** —— seeds 2022/2023 在 ±0.003 内（≈F），seeds 2024/2025 明显超出。d2t1：+0.0035 / −0.0003 / **−0.0079** / **+0.0191** —— 方向跨 seed 不稳定。

## 3. History-dominant 判据（§八）：**PARTIAL REPLICATION**

| 判据 | 结果 |
|---|---|
| 1. T−F ≥ +0.003 | ✓（+0.0194 / +0.0143） |
| 2. \|HT−T\| < 0.003 | ✓（mean −0.0013 / −0.0001） |
| 3. \|DH−F\| < 0.003 | **✗**（mean +0.0059 / +0.0036；4 seed 中 2 个明显超出） |
| 4. seed-level 大部分一致 | 部分：HT≈T 在 4/4 seed 成立；DH≈F 仅 2/4 |
| 5. common epochs 不出现完全相反关系 | 部分：ep9 两因子关系与 KuaiRand 一致（HT−F +0.0105 ≈ T−F +0.0135，DH−F −0.0037）；但 ep39/ep49 DH−F 转为稳定为正（+0.0049 / +0.0098），而 T−F 衰减至 ≈0 |

逐条对应 §八 的 PARTIAL 定义："HT 明显接近 T，但 DH 有中等幅度"、"selected 成立但 common epochs 不稳定"、"recovery pattern 方向一致但幅度明显弱于 KuaiRand"。**不得标记为完整复现。**

## 4. 与 KuaiRand 的三个结构性差异（§十二.12–.13）

1. **DH 臂不再贴合 F。** KuaiRand 中 target-current 经 FULL history 施加时价值为零（recovery ≈2%，|R_D| ≤ 0.0027 贯穿全部 7 个 checkpoint 视图）；MovieLens 中 DH 在 selected 恢复约 25–30% 的 rescue，且在 final/common-late 稳定高于 F（d2t0 final DH−F mean +0.0098，4/4；d2t1 +0.0103，4/4）——**在 MovieLens 上，current-update 的 target 方向成分不可忽略**，尤其当 T 自身的 rescue 在训练后期衰减之后（final 排序：DH > F ≈ T > HT）。
2. **Rescue 的时间结构不同。** KuaiRand 的 T−F 在 final 仍保持 +0.031/+0.032；MovieLens 的 T−F 与 HT−F 都集中在早-中期（selected epochs：F=4/8/6/4，T=9/14/10/7，HT=7/9/8/10），final 衰减至 ≈0 甚至略负。MovieLens 该设定下所有臂在 epoch ~10 后快速过拟合（F final d1t0 LogLoss 2.1–2.5），rescue 是一个**早期窗口现象**。HT 的 selected epoch 仍跟随 T（history 因子）而非 F，这一点与 KuaiRand 的"选择时刻跟随 history"一致；DH 的 selected epoch（4/35/43/6）不再 F-like。
3. **Non-target trade-off 的指向不同（§十二.10）。** MovieLens 上 T 的代价落在 **d0**（selected d0t0 mean −0.0120、d0t1 mean −0.0259，均 0/4 为正），而 **d1 反而是受益者**（d1t0 +0.0382、d1t1 +0.0311，均 4/4）。KuaiRand 中 T 伤 d0、d2 晚期受益；这里"救 target、伤 non-target"只对 d0 成立，不得强行对齐成 KuaiRand 形态。

## 5. Rescue/damage 可分离性：再次复现且更干净（§十二.11）

HT−T（selected，4-seed mean ΔAUC）：d0t0 **+0.0219**、d0t1 **+0.0264**、d1t0 +0.0066、d1t1 +0.0046 —— **四个 non-target cell 全部 4/4 为正**。即 HT 在完整保留 target rescue（93–99%）的同时，把 T 对 d0 的损伤不仅消除而且转为相对 F 的正收益（HT−F d0t0 mean +0.0099）。KuaiRand 中 HT 对 d0 的修复在 selected 只有 ~80% 且有 seed-2024 例外；MovieLens 上可分离性更干净。DH−F 在 non-target 上 seed 方向混杂（d0t1 mean −0.0110、d1t1 mean −0.0176，均无稳定符号），不支持"DH 保护 non-target"的解读。

## 6. AUC / LogLoss 一致性（§十二.9）

Target cell selected 上 LogLoss 与 AUC 同向：T−F 4/4 有利，HT−F 4/4 有利且 3/4 seed 幅度大于 T（d2t0 mean：T −0.0099 vs HT −0.0102）。DH−F LogLoss 方向混杂（+0.028 / +0.036 / −0.022 / −0.001）。非 target 的 LogLoss 在 final 出现大尺度分化（各臂均显著优于 F 的后期过拟合），与 KuaiRand 的 d2 晚期纹理类似，属于 metric-dependent 现象，不改变 target 结论。

## 7. 三个 RQ 的裁决（§十二.3–.8、.14）

- **RQ1（MovieLens 是否存在可拆解的 F→T rescue）：YES。** mean +0.019/+0.014，4/4 seeds，双 task，AUC/LogLoss 一致。
- **RQ2（rescue 是否仍由 persistent Adam history 携带，HT≈T、DH≈F）：PARTIAL。** HT≈T 半侧**完整复现**（recovery 93–99%，|HT−T| 远低于阈值，4/4 seeds，且 HT 的 selected epoch 跟随 T）；DH≈F 半侧**未复现**（DH 在 selected 恢复 25–30%，在 final 稳定高于 F 且 4/4）。history 因子仍是主导载体，但 current-update 因子在 MovieLens 上有非可忽略的贡献，且其贡献随训练阶段增长。
- **RQ3（是否说明不依赖 KuaiRand 极端域比例）：仅能有限支持。** 允许的表述：**"history-dominant 的核心半侧（HT≈T）在两个域比例结构明显不同的数据集上均可复现；但完整 Pattern A（含 DH≈F）未在 MovieLens 温和比例下复现。"** 不得声称普遍 MDMT 机制。

## 8. 情况分类与竞争解释（§十）

本结果落在**情况 A 与情况 B 之间，按纪律归为情况 B 为主导**："target rescue 可以跨 dataset 出现，但其 optimizer-state carrier 的分解依赖数据结构/域构成"。保留的竞争解释（本阶段无法区分）：

- **imbalance severity**：KuaiRand 为"一超大域+两小域"极端结构，MovieLens d2_large 为温和结构；DH 偏差可能反映温和比例下 current-direction 有更多残留价值。
- **rescue 的时间局部性**：MovieLens 的 rescue 是早期窗口现象，DH 的晚期优势可能本质是"抗过拟合"而非"rescue 载体"——两者在 final checkpoint 上不可分。
- **target role / task 难度结构**：d2t1 seed-level 不稳定（2024 为负、2025 +0.019）提示 task 难度与因子分解存在交互。

**不得**将 DH 偏差解释为"Adam history 不重要"——HT≈T 的 93–99% recovery 恰恰说明 history 仍是充分载体；应解释为"current 因子在 KuaiRand 上的≈零贡献本身可能是极端域比例的特例"。

## 9. 十六问逐条（§十二）

1. 配置：MovieLens d2_large / M3OE / Equal / natural ratio / seeds 2022–2025 / 50 epochs。2. target d2（S2 稳定 Equal 退化 cell），t0 primary、t1 secondary。3. F/T rescue 稳定复现：是。4. 未在 Phase A 停止。5. HT 恢复 T 的 rescue：是（93–99%）。6. DH 恢复 rescue：部分（25–30% selected，晚期稳定高于 F）。7. HT≈T 成立、DH≈F 不成立 → 未形成完整 Pattern A。8. selected 上 HT≈T 一致；final/common-late 上关系改变（DH 上升、T 衰减）——**结论具有 checkpoint 依赖性**。9. AUC/LogLoss 在 target selected 上一致。10. non-target damage 复现于 d0（T 0/4），d1 为受益者。11. HT 的 rescue/damage 可分离性复现且更干净（non-target 4/4 HT≥T）。12. 一致：history 主导、可分离性；不一致：DH 非零、rescue 时间局部性、d1 获益方向。13. 温和域比例下 history 载体依然主导，但 current 因子的零贡献是 KuaiRand 特例的可能上升。14. 不足以认为 S5 完整机制"非 KuaiRand 特例"——只有 HT 半侧外推成功。15. 是否进入 Sample-objective verification：见 §10。16. 不能外推：普遍性、m/v 子成分、任何方法性（HT 不是已验证方法）、final-checkpoint 下的因子关系。

## 10. 阶段决策（§十三）

**判定：PARTIAL replication。** 按规格走"部分复现"分支：先分析差异来源，再决定后续。建议的下一阶段候选（需新一轮 designer 阶段裁决，本报告不设计）：

- **首选**：针对 DH 偏差的归因——imbalance severity 是否为 DH≈F 成立的触发条件（需要独立的 ratio 操纵阶段，规格 §四 已预留）；
- **候选**：rescue 时间局部性与过拟合的纠缠（DH 晚期优势是 rescue 还是正则化）；
- **暂缓**：S6 Sample-objective verification 与 m/v 子拆解——在 DH 偏差的来源未定位前，进入这些阶段会把"KuaiRand 的零 current 贡献"错误地当作已确立前提。

**Boundary.** 全部结论条件于 MovieLens d2_large / M3OE / Equal / natural / seeds 2022–2025 / 50 epochs。HT 不被声称方法。两个数据集的事实在规格允许的表述内："history-dominant 的 HT≈T 半侧跨数据集复现；完整 Pattern A 未复现。"

---

*Prepared by the main Agent from `stage_S5ML_outputs` collector products only（Phase A queue `b14dd7c4…` + Phase B queue `7fb1ddbb…`，16/16 canonical runs，pairing valid，moment gate 8/8）。执行事实：runner 两轮交付报告；设计：`promt/S5_add.md` + `experiments/reports/stage_S5_design.md`；审查：reviewer Rounds 1–3（均 PASS）。*
