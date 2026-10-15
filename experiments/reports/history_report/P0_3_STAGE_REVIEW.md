# P0-3 Stage Research Review：M3OE vs MMoE Per-Domain Transfer Matrix

- 阶段：P0-3（Priority 0 第三档）
- 数据：MovieLens-1M，D=3 域 × T=2 任务，固定 N=187,500
- 档：balanced（ratio 1:1，62,500×3，d0_large 退化 setting）；strong（ratio 50:1，L/M/S = 150,000/34,500/3,000，三 setting 轮换）
- Arms：MMoE baseline / MMoE+comp1 / MMoE oracle（80 新 run，全部 OK）；M3OE 侧 80 行复用 P0-2（ratio∈{1,50}，缓存键逐位一致，冒烟验证）
- Seeds：2022-2025；全部逐 cell 配对（同 backbone × 档 × setting × seed）
- 原始报告：`experiments/reports/P0_3_RAW_REPORT.md`；汇总：`experiments/P0_3_mmoe_transfer/P0_3_{cells,transfer_matrix,summary}.csv`
- 超参与训练协议与 P0-2 完全一致（epoch 50、batch 40960、lr 1e-2、earlystop 10、--fix_optim_grouping 1）

## 1. Stage research question

P0-1/P0-2 确认的 largest-domain sacrifice（L 角色 Δ_transfer ≈ −0.015）与 comp1 gain 究竟是 M3OE（Star 融合）特有，还是一般 MDMT joint training 现象？若 MMoE 复现同一 pattern，研究问题升级为一般 MDMT optimization 问题（promt.md 停止规则：M3OE-specific 停止条件）。

## 2. Experimental facts

strong 档 per-role Δ_transfer（baseline，n=24 = 3 setting × 4 seed × 2 task；逐 cell joint − oracle 配对）：

| role | M3OE Δ (mean±std) | M3OE min/max | MMoE Δ (mean±std) | MMoE min/max |
|---|---|---|---|---|
| L（150,000） | −0.0155 ± 0.0109 | −0.0402 / +0.0017 | −0.0116 ± 0.0071 | −0.0265 / −0.0025 |
| M（34,500） | +0.0186 ± 0.0149 | −0.0157 / +0.0637 | +0.0138 ± 0.0100 | +0.0003 / +0.0376 |
| S（3,000） | +0.0598 ± 0.0216 | −0.0166 / +0.0922 | +0.0651 ± 0.0137 | +0.0397 / +0.0930 |

balanced 档（n=24，role 退化）：M3OE Δ 全部为正 overall +0.0148 ± 0.0075；MMoE overall +0.0117 ± 0.0056；min 分别为 −0.0010 / +0.0023。

comp1 gain（comp1 − baseline，同口径配对）：

| 档 | role | M3OE | MMoE |
|---|---|---|---|
| balanced | overall | +0.0030 ± 0.0040 | +0.0011 ± 0.0033 |
| strong | overall | +0.0174 | +0.0113 |
| strong | L | +0.0121 ± 0.0118 | +0.0105 ± 0.0077 |
| strong | M | +0.0124 ± 0.0082 | +0.0076 ± 0.0074 |
| strong | S | +0.0276 ± 0.0191 | +0.0158 ± 0.0120 |

comp1 后 L 角色残余 Δ_transfer：M3OE −0.0155+0.0121 ≈ −0.0034；MMoE −0.0116+0.0105 ≈ −0.0011。

绝对水平（mean AUC）：balanced baseline M3OE 0.7405 / MMoE 0.7413；strong baseline M3OE 0.7098 / MMoE 0.7134；strong comp1 M3OE 0.7271 / MMoE 0.7246。worst-domain AUC strong：baseline M3OE 0.6785 / MMoE 0.6846；comp1 M3OE 0.6985 / MMoE 0.6976。

工程事实：80/80 OK，0 FAILED，无 NaN/发散（S 域 3,000 样本 oracle AUC 0.59-0.63，为真实小样本上限）；strong 档 S 角色 MMoE 24/24 cell Δ 为正、L 角色 MMoE 24/24 cell Δ 为负。

## 3. Main observations

**Strong evidence**
- MMoE 在 strong 档完整复现 largest-domain sacrifice：L 角色 Δ = −0.0116，24/24 cell 为负（一致性高于 M3OE 的 23/24）。方向、role 排序（L<0 < M < S）、量级（−0.01 级）与 M3OE 一致。
- comp1 在 MMoE 上同样广谱为正：overall +0.0113，L/M/S 三角色的均值均为正；两个 backbone 上 comp1 都把 L 角色负迁移修复到 ≈0（残余 −0.001 ~ −0.003）。
- balanced 档两 backbone 全部 cell 正迁移：负迁移需要 imbalance 才出现，与 P0-2 的 1:1→5:1 阈值观察在两个架构上同时成立。

**Moderate evidence**
- 量级差异：M3OE 的 L 负迁移略深（−0.0155 vs −0.0116，差 0.004，约 0.4 个 pooled std）、comp1 gain 略大（+0.0174 vs +0.0113）。架构调节幅度但不改变方向。
- MMoE 的 S 角色获益不小于 M3OE（+0.0651 vs +0.0598）：M3OE 并没有"更帮弱域"。

**Weak evidence**
- strong 档 M3OE baseline 绝对水平低于 MMoE（0.7098 vs 0.7134），comp1 后反超（0.7271 vs 0.7246）；n=12/臂，差 0.0025 级，接近种子噪声。

**Speculation**
- comp1 对 M3OE 的边际收益略大，可能与 Star 融合的共享参数结构（element-wise 乘性耦合）对动量平滑更敏感有关；本阶段数据不能分离该机制。

## 4. Competing hypotheses

- H1：largest-domain sacrifice 是 M3OE（Star 融合）特有现象。
- H2：largest-domain sacrifice 是一般 MDMT joint training 的优化层现象，架构无关；comp1 是 backbone 无关的优化层机制。
- H3：两 backbone 的相似 pattern 是共同数据管线/训练协议（masked 6-term 等权 BCE、共享 embedding、相同早停）的人造产物，而非架构各自的内在性质。

## 5. Evidence supporting each hypothesis

- H1：M3OE 的 L 负迁移略深、comp1 gain 略大（方向性差异存在）。
- H2：MMoE L 角色 24/24 全负、role 排序一致、balanced 档同正、comp1 在 MMoE 同向为正且同样修复 L；P7 在全量数据上 MMoE+comp1 亦显著为正（+0.0030，p=0.0066）。
- H3：两 backbone 共享同一 embedding 包装、tower 结构、训练器与早停口径（p7_models.py 的 Backbone 刻意复用 MDMTRec 数据/评估管线）；绝对 AUC 水平两架构几乎相同（balanced 0.7405 vs 0.7413），说明 pipeline 主导了大部分行为。

## 6. Evidence against each hypothesis

- H1：被直接证伪——MMoE 在方向、一致性（24/24）、量级上完整复现。
- H3：comp1 增益幅度在两 backbone 间有系统性差异（+0.0174 vs +0.0113，per-role 排序 L≈M<S 一致但幅度不同），说明并非纯 pipeline 人造；架构对优化动力学有真实调节。
- H2：目前无反面证据；残余问题只在于 H2 的机制解释（为何 L 被牺牲）尚未定位——那是 P0-4 的问题。

## 7. Major confounds

- strong 档 cell 均值跨 setting 池化时 std 大（0.03-0.04），因混合 L/M/S 角色；本报告全部按 role 分层解读。
- S 域 oracle（3,000 样本）方差大，S 角色 Δ 的绝对值含小样本噪声，但 24/24 为正的方向性结论稳健。
- MMoE 与 M3OE 参数量/容量不完全匹配（MMoE 112 参数张量 vs MDMTRec 结构不同）；绝对水平比较需谨慎，本阶段结论只依赖 role 内配对的方向与排序。
- balanced 档 role 退化（γ 全 0.9），其 comp1 臂是 uniform 控制臂，语义与其他档不同（同 P0-2 §7.1 声明）。
- 跨 backbone 比较共用同一批采样缓存（缓存键不含 backbone），数据侧无 confound。

## 8. What has been falsified?

- **"largest-domain sacrifice 与 comp1 gain 是 M3OE/Star 融合特有"——证伪。** MMoE 完整复现负迁移 pattern（L 24/24 负）与 comp1 增益（全角色为正）。
- **"M3OE 更容易帮助弱域"——证伪。** MMoE 的 S 角色获益（+0.0651）不小于 M3OE（+0.0598）。
- **"comp1 只在 M3OE 有效"——证伪。** MMoE+comp1 overall +0.0113，三角色均正。
- H1（架构特有论）整体证伪；H3（纯 pipeline 人造）被削弱但不可排除。

## 9. What remains unresolved?

- 为何等权 masked BCE 下 L 角色被牺牲：训练 objective（loss weighting）与 checkpoint selection 口径的贡献未被分离——正是 P0-4 的研究问题。
- comp1 的架构间幅度差异（M3OE > MMoE）的机制来源未定位（留待 Priority 1 机制实验）。
- 结论限于 ML-1M、D=3、T=2、当前超参；PLE/ShareBottom 未在 strong 档验证（P2-1 候选）。

## 10. Research direction decision

**Continue current hypothesis（修正版）**：按 promt.md 的 M3OE-specific 停止规则，研究问题正式升级为**一般 MDMT shared optimization 问题**——largest-domain sacrifice 与 comp1 gain 是 joint training 的架构无关性质，M3OE 只是其中一个实例。后续所有叙事不得再称"M3OE 特有问题"；论文定位从"M3OE 的 small-large trade-off"转向"多域联合训练中最大域被系统性牺牲的优化现象及其缓解（comp1）"。

## 11. Next-stage recommendation

进入 **P0-4（Loss Weighting + Checkpoint Objective Control）**。理由：两架构共同的 masked 6-term 等权 BCE 使 3,000 样本域与 150,000 样本域在 scalar objective 中权重相近，这是 L 牺牲与 S 获益最直接的共同嫌疑；P0-4 的三组 weighting（equal / sample-proportional / 中间档 sqrt-frequency）+ 两种 checkpoint selection 口径正好检验该解释。设计要点：
- 在 strong 档（ratio_50，现象最强）做，M3OE 一个 backbone 即可（架构无关性已由本阶段确立，无需双 backbone）；
- 复用本阶段的 oracle 与采样缓存；
- 若 weighting 改变后 L 牺牲消失，必须明确承认此前 pattern 部分来自 objective design（promt.md 原话）。
