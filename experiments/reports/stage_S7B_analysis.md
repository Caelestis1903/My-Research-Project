# Stage S7-B 分析报告 — Same-θ Early/Late Causal Continuation

阶段性质：causal diagnosis（same-θ 短程反事实）。本报告由 main agent 在 runner 完成全部 16 个 continuation run 后撰写。
数据基础：`experiments/stage_S7B_outputs/stage_S7B_{steps,geometry,validation}_raw.csv`（160 / 46,080 / 1,728 行）、`experiments/stage_S7B_analysis_tables/B0–B6`。

术语与口径：
- T-branch 语义 = S6 DH-arm 的 current 操纵（history=FULL，仅 current gradient source 切换为 target-only）；F-branch = S6 F arm 原样。两臂同 θ、同 shared moments、同 batch 序列、同 private 起点。
- 几何/utility 量为**单 actual-batch** 口径（该步真实更新 batch），与 S7-A 的 4-batch 平均口径不同，跨阶段只比 pattern 不比绝对值。
- diff 一律为 T−F。cell 索引 task-major（cell=task*3+domain）；MovieLens target cells = 2,5；KuaiRand target cells = 1,4。
- 已知近似：private 参数 fresh-Adam（S6 未 dump private moments）；continuation 与原 F 轨迹的同一性仅在 step 0 成立（V0 认证），step≥1 为已声明近似下的同状态反事实。GO/STOP 判读完全建立在 branch 间对比上。

---

## A. Execution validity

- 16/16 run（2 数据集 × 4 seeds × 2 anchors）全部 `validate_run` ok=true，无 crash、无 REVIEW 旗标、无 abort。
- **V0（重建正确性）**：16/16 run 的 step-0 L_full diff = 0.0、shared delta 相对差 ~1e-11（阈值 1e-3）——同 θ 同 batch 同 moments 下 F-branch 与 S6 原轨迹逐位一致。
- **V3（相同初始条件）**：16/16 run step-0 post-update 双 branch private 参数逐位相等（diff = 0.0）；B0 独立复核：step 0 双 branch 的 geometry/validation/steps 全部数值列 F−T 最大差 = 0.0。
- **batch 重放**：16/16 run 全部 batch 指纹与 S6 源 run 逐位匹配（3×(anchor+1) RNG 消耗公式经全部组合验证）。
- V2（private fresh-Adam 近似的外推限制）：step≥1 F-vs-S6 的 |L_full diff| 最大 0.17，全部低于 0.3 tripwire，如实记录于 run report。
- 数据说明：聚合 validation 行数 1,728 = 108/run × 16，与 per-run schema 一致（designer §8 的 1,152 为算术笔误，108 行/run 契约以 validator 为准）。
- macro scope 的 val_bce_raw 列为 NaN（源数据如此）；MovieLens 部分 batch 小域计数极小（d1=383），该步 cell 梯度噪声偏大——均为数据属性。

## B. Q1 — same-θ 下 late stage 的 TARGET current 是否立即改善 harmful-gradient metric？

**部分成立，但不稳定。**

- min_U_k（最坏 cell 的 validation utility）step1–4 平均 diff（T−F）：MovieLens anchor39 四 seeds = **+0.042 / +0.0074 / +0.0085 / +0.169**，4/4 为正；anchor9 为 +0.0022 / −0.0005 / +0.0008 / −0.0012（近零、方向混合）。即同 θ 下仅切换 current source，晚期有害 utility 立即部分缓解——harmful 结构**确实部分是 current-gradient 的即时属性**，不是纯粹的 trajectory 化石。
- 但逐时步看（B5/B6）：MovieLens anchor39 的 min_U diff 逐步符号震荡（s1 −0.0015 → s2 +0.1618 → s3 −0.0150 → s4 +0.0819），不是稳定单调的改善；量级由 seed2025（+0.17）与 2022（+0.04）主导，2023/2024 接近零。
- 逐 target cell：U_diff_cell2 anchor39 为 +0.020/−0.0035/+0.0006/+0.109（3/4 正）；U_diff_cell5 为 +0.018/−0.0085/−0.0121/+0.077（**2/2 符号分裂**）。
- 有害质量 Σ_{H<0}H 的 diff：anchor39 为 −0.0094/+0.018/+0.017/+0.037（3/4 有害质量下降）。

裁决：gradient 层面的即时响应存在（4/4 seeds min_U 平均为正），但逐步不稳定、per-cell 符号不一致、量级 seed 偏斜。**强度：MEDIUM。**

## C. Q2 — 该改善是否在 early stage 明显较弱？

**成立（gradient 层面）。** MovieLens 4/4 seeds 的 |min_U diff| 与 |sum_neg_H diff| 均为 anchor39 > anchor9（B4 summary 4/4）；early anchor 的梯度效应近零。即 current-source 操纵的梯度效应具有明确的 **late-stage specificity**——与 S7-A 的"有害结构晚期才出现"自洽。

值得记录的反面观察（validation 层面）：anchor9 step1 的 target AUC diff 为 4/4 seeds 微正（均值 +6.6e-5）；anchor39 step5 的均值为 −0.001226，但逐 seed 是 3/4 为负、1/4 为正（seed2025 +0.002359），不是 4/4 微负。early/late 的均值方向对比存在，但晚期 seed 一致性不足，且方向与"DH 晚期受益"相反（见 D 节）。**强度：STRONG（gradient 层面 4/4）。**

## D. Q3 — gradient change 是否先于 validation performance change？

**不成立——validation 层面没有出现改善。**

MovieLens anchor39 的 target_domain validation 轨迹（4-seed mean，B6）：
- AUC diff：s1 −5.8e-5 → s2 −4.3e-4 → s3 +1.2e-4 → s4 −1.2e-3 → s5 −1.2e-3；
- val BCE diff：s3 +0.0085、s4 +0.017、s5 +0.011（4/4 seeds 在 s3–s4 为正=T 更差的方向上）。

即：在 gradient utility 指标响应（且方向为正）的同时，5 步窗口内 target-domain validation **没有形成稳定改善**。step5 AUC 均值为 −0.001226，但只有 3/4 seeds 为负（2022 −0.001714、2023 −0.003059、2024 −0.002489、2025 +0.002359）；LogLoss 均值为 +0.011130，3/4 seeds 变差，seed2024 为 −0.002735。step4 LogLoss 均值 +0.017005 且 4/4 变差。均值整体不利，但逐 seed 并非一致恶化，故不能称为 4/4 同号或排除噪声。

这是预注册 GO 条件第 4 条（"gradient utility 改善之后出现 validation trajectory 改善"）的**直接反证**。**强度：MEDIUM**（4 seeds、AUC 与 LogLoss 均值方向不利，但 step5 各指标均有 1 个 seed 反向，窗口仅 5 步；支持"未见稳定改善"，不足以断言必然有害）。

## E. Q4 — KuaiRand 是否缺少同样的 early/late contrast？

**成立。** KuaiRand：
- gradient 层面：U_diff_cell1/cell4 的 early/late 对比仅 2/4 seeds |a39|>|a9|，方向混合（anchor39：2023 大幅为负、2025 为正）；无稳定对比结构。
- validation 层面：两个 anchor 的 target AUC diff 均为微小正（a9 +6e-4、a39 +1.2e-3 @step5），与 MovieLens 的 a39 转负形成对照；target_domain_auc_diff 的 |a39|>|a9| 为 0/4。
- 与 S6 完全一致：KuaiRand 无 late DH rescue，S7-B 中也无 late-stage 特异响应。**强度：STRONG（4/4 层面的一致 null 对照）。**

## F. 与 S6 late DH residual 的对照判决

S6 观察到 MovieLens DH−F 在 epoch39/49 的 target AUC 优势（+0.0105/+0.0081，积累于 ~10 个 epoch）。S7-B 的 same-θ 判决：

1. harmful gradient 结构**可以**被 current-source 切换立即改变（B 节）——它不是纯粹的轨迹化石；
2. 但该改变在短程窗口内**不产生** validation 改善（D 节），甚至方向相反；
3. 因此 S6 的 late DH benefit 更一致于**长程累积效应**（数十步的 trajectory/regularization/over-training 通道），而非"每步 current gradient 的即时几何改善累积成 validation 收益"这一机制。

## G. GO / STOP 判定（grad.md §16）

GO 五项条件核查：

| # | 条件 | 结果 |
|---|---|---|
| 1 | same-θ 下 TARGET-current 稳定改变 harmful gradient structure | **部分满足**：min_U 4/4 平均为正，但逐步震荡、per-cell 2/2 分裂、量级 seed 偏斜 |
| 2 | late 比 early 更明显 | 满足（4/4，gradient 层面） |
| 3 | ≥3/4 seeds 同方向 | 边缘满足（min_U 4/4；关键 per-cell 与 pair 指标 2/2–3/4） |
| 4 | gradient utility 改善之后出现 validation trajectory 改善 | **不满足**：5 步内未见稳定改善；step5 AUC 3/4 为负、LogLoss 3/4 变差 |
| 5 | MovieLens 与 KuaiRand 差异与 S6 一致 | 满足（KuaiRand null 对照成立） |

**条件 4 明确失败，条件 1/3 仅边缘满足 → 不进入 method design。**

STOP 条款命中：「gradient structure 变化但 validation performance 不变化」。

## H. 最终裁决（主 Agent）

**停止 gradient-method 主线。**

综合 S7-A + S7-B 的最终结论：

1. Sample objective 下确实存在真实、可重复、stage 依赖、MDMT 特异的 Domain×Task gradient 结构（S7-A：STRONG）；其中 harmful validation-aligned utility 在 MovieLens 晚期过拟合 regime 中与性能恶化同步（STRONG），且可被 current-source 操纵即时改变（S7-B：MEDIUM）。
2. 但该 gradient 现象**不构成可方法化的独立问题**：same-θ 下消除它并不改善 validation trajectory（S7-B 条件 4 失败）。S6 的 MovieLens late DH residual 最一致的解释是**长程 trajectory / regularization / over-training 通道的累积效应**，而不是每步 gradient geometry 的即时作用。
3. KuaiRand 在两个阶段均表现为干净的 null 对照（无恶化、无 rescue、无 stage 特异响应），排除了 universal MDMT mechanism 的可能。

**科研含义**：grad.md §21 的问题——"Sample objective 解决了 Equal 的聚合问题之后，是否还剩下一个真实、可重复、与性能相关、具有 MDMT 特异性的 gradient dynamics 问题"——答案是：**现象真实存在，但它不是性能问题的独立根因；它是 MovieLens 过拟合 regime 的伴随症状。不值得在其上开发 gradient correction 方法。**

**溯源更正（2026-10-09）**：复核 `stage_S7B_validation_raw.csv` 后，修正了 anchor39 的 seed 一致性描述。此前将 step5 target AUC 写为 4/4 为负；原始数据实际为 3/4 为负、seed2025 为正。均值及“未观察到稳定 validation 改善”的裁决不变，但逐 seed 证据强度相应降低。

## I. 后续方向建议（供研究总览决策，非本阶段执行内容）

- S6 late DH residual 若仍值得追踪，下一个可证伪对象应转向**长程通道**：optimizer-state 历史（S6 已有的 moment dumps）、过拟合 regularization、或学习率 schedule——而非 gradient surgery。
- MovieLens 的 F-arm 过拟合本身（macro LogLoss ×1.5、selected epoch 7–12 远早于 50 epoch 训练终点）提示：late-stage 现象研究的前提是训练协议允许模型进入深度过拟合区；若研究目标是部署相关结论，validation-selected checkpoint 处的现象（S7-A 显示 selected 处 U_k 接近零、无害）才是更相关的对象。
- S7 全部产物（S7-A：8 probe run + 3 CSV；S7-B：16 continuation run + 3 CSV + 分析表）已固化，可支撑后续任何复审。

## 附：执行链记录

designer（k3-256k max，2 轮裁决：validity 分层 V0–V3、3× RNG 常数、2 并发；V3 前提修正）→ coder（MiniMax M3，4 新文件，3 smoke，2 验证 run，修复 6 项）→ reviewer（k3-256k high，修复 8 项，含 MovieLens 部分 batch 权重分母 bug 与 non_target_weighted 归一化 bug；Ready: YES）→ runner（MiniMax M3，8 waves × 2 并发，16/16 OK）→ main（本报告）。
