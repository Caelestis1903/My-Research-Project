# P0-5 Stage Research Review：comp1 × Loss Weighting 2×2 交互（Priority 0 封口实验）

- 阶段：P0-5（P0-4 的封口实验，Priority 0 收尾）
- 设计：weighting {equal, sample} × comp1 {off, on}，M3OE，strong 档 ratio_50（L/M/S=150,000/34,500/3,000，三 setting 轮换，seeds 2022-2025）
- 复用：equal/off 与 equal/on（P0-2 ratio_50，24 run）、sample/off（P0-4，12 run）、oracle（P0-2，36 行）；新增：sample/on 12 run（全 OK）
- 数据完整性：anchor check PASS（P0-5 vs P0-2 comp1 config 12/12 逐字段一致；P0-4 vs P0-2 12/12）；checkpoint 统一 equal 选点口径；mismatch 0/288 cell
- comp1×weighting 实现语义：loss 项先加权再进 backward_accumulate，exactly-once weighting 经 coder 推导与 reviewer 独立复核成立；γ_d 机制不变
- 原始报告：`experiments/reports/P0_5_RAW_REPORT.md`；汇总：`experiments/P0_5_comp1_x_weighting/P0_5_{cells,summary,headline}.csv`

## 1. Stage research question

comp1（per-domain momentum）在修正后的 objective（sample-proportional weighting）上是否仍有增量？即：P0-2 观察到的 comp1 gain（+0.0174）有多少是在补偿等权 objective 错配，有多少是独立的优化层机制？该答案决定 comp1 的论文定位与 Priority 1 机制实验的必要性。

## 2. Experimental facts

2×2 主表（n=72 cell/格，逐 cell 同 setting 同 seed 配对）：

| weighting | comp1 | overall AUC | worst-domain | Δ_L | Δ_M | Δ_S |
|---|---|---|---|---|---|---|
| equal | off | 0.7098 | 0.6425 | −0.0155 ± 0.0023 | +0.0186 | +0.0598 |
| equal | on | 0.7271 | 0.6750 | −0.0034 ± 0.0009 | +0.0310 | +0.0874 |
| sample | off | 0.7308 | 0.6754 | −0.0012 ± 0.0008 | +0.0301 | +0.0973 |
| sample | on | 0.7363 | 0.6819 | −0.0001 ± 0.0006 | +0.0327 | +0.1098 |

comp1 gain（配对，n=12 配对；SE 按 cell 计 n=24）：
- gain(equal) = +0.0174（overall AUC）
- gain(sample) = +0.0054（0.736252 − 0.730845）
- interaction = gain(sample) − gain(equal) = −0.0120
- gain(sample) 的 per-role 分解：L +0.0011，M +0.0026，S +0.0125（增益重心从 L 移到 S）
- frac_negative：34.7% → 27.8%（equal+comp1）→ 22.2%（sample）→ 15.3%（sample+comp1）
- mean selected epoch：7.5 / 7.3 / 7.8 / 8.8（sample+comp1 略晚）

## 3. Main observations

**Strong evidence**
- 交互显著：interaction = −0.0120，|interaction| ≥ 0.005 阈值；comp1 增益的约 69%（0.0120/0.0174）在 objective 修正后消失——这部分是对等权 objective 错配的补偿，不是独立机制。
- comp1 在 sample 上仍有残余增量 +0.0054（恰好越过 0.005 阈值）：worst-domain +0.0064、Δ_S +0.0125、Δ_overall +0.0054，方向一致为正——中间情形成立，两解释按量级拆分归因。
- 最佳配置易主：sample+comp1（0.7363）> sample（0.7308）> equal+comp1（0.7271）> equal（0.7098）。objective 修正的贡献（+0.021）大于 comp1 的贡献（最大 +0.017）。

**Moderate evidence**
- comp1 残余增量的重心从 L（equal 臂 +0.0121）转移到 S（sample 臂 +0.0125）：objective 修正接管了"L 修复"功能后，comp1 的剩余价值主要体现在小域——与 γ_d 给小域更长记忆（γ_S=0.99）的机制定性一致。
- sample+comp1 下 Δ_L = −0.0001 ± 0.0006：largest-domain sacrifice 在修正 objective + comp1 后完全消失（统计上与 0 无差异）。

**Weak evidence**
- mean selected epoch 在 sample+comp1 下推迟到 8.8（vs 7.3-7.8），样本量小，方向性观察。

**Speculation**
- comp1 残余机制可能与"objective 错配补偿"不同源：若只是补偿，sample 臂应归零；+0.0054 的残余提示 temporal smoothing 对小域有独立的正则化价值（P1-3 的 matched-momentum 设计可直接检验）。

## 4. Competing hypotheses

- H_a：comp1 的 equal 档增益主要是等权 objective 错配的补丁，修正后冗余。
- H_b：comp1 是独立于 objective 修正的优化层机制。
- H_c（中间）：两者并存——大部分增益是补丁，残余部分是独立机制（对小域的 temporal smoothing）。

## 5. Evidence supporting each hypothesis

- H_a：interaction −0.0120，69% 增益消失；P0-4 已证明 objective 是 L 牺牲主因，comp1 在 equal 臂的增益恰好集中于 L 修复（+0.0121）。
- H_b：gain(sample) = +0.0054 > 0 且跨 12 配对方向一致（frac_negative 继续下降、worst-domain 继续上升）。
- H_c：以上两条同时成立；增益重心的 L→S 转移直接支持"补丁部分（L）被 objective 接管，独立部分（S）保留"。

## 6. Evidence against each hypothesis

- H_a：gain(sample) 未归零（+0.0054，越阈值），纯补丁论不成立。
- H_b：增益衰减 69%，"完全独立机制"不成立。
- H_c：残余增量的机制归属（temporal smoothing vs 有效步长变化）未被本阶段分离——那是 P1-3 的问题。

## 7. Major confounds

- gain(sample) 恰好落在 0.005 阈值边缘（+0.0054），n=12 配对的 SE 约 ±0.002 级，残余增量的存在性结论稳健但其精确量级不确定。
- γ_d 与 w_d 同由 p_d 驱动（designer 已论证：w 定义 objective、γ 定义 optimizer，正交轴；行内 objective 固定，归因干净）。
- comp1×sample 下 S 域 momentum 幅度被 w_S=0.048 压缩 ~20 倍，S 仍获 +0.0125——杠杆缩小但方向未变，说明 comp1 对 S 的作用不依赖梯度幅度绝对值。
- 结论限于 ratio_50、per_domain_only 形态、ML-1M。

## 8. What has been falsified?

- **"comp1 是修复 L 牺牲的独立必要机制"——证伪。** 其 equal 档增益的 69% 与 objective 修正冗余；objective 修正（零成本的 loss weighting）才是主修复手段。
- **"comp1 完全是 objective 补丁"——证伪。** 修正后仍有 +0.0054 残余增量，集中于 S 域。

## 9. What remains unresolved?

- 残余 +0.0054 的机制：temporal smoothing（H-M1/M2）vs 有效 momentum 幅度（H-M5）——P1-3 的 matched momentum control 可直接回答，且现在只需解释小残余而非整个 +0.0174。
- Curriculum（comp2）在修正后 objective 下的 keep/drop——P1-2 必须在 sample weighting 口径下重做，equal 口径下的任何既有 curriculum 结论都已失效。
- 残余 L 负值（−0.0001）已统计消失，largest-domain sacrifice 问题本身宣告闭合。

## 10. Research direction decision

**Modify hypothesis（Priority 0 最终版）**：完整叙事链定型——(1) 等权 6-term BCE 在样本占比悬殊时系统性错配 objective，制造最大域牺牲的假象（P0-4，修复 92%）；(2) 该现象架构无关（P0-3）；(3) comp1 的大部分历史收益是补偿该错配的补丁（P0-5，69%）；(4) comp1 保留一个集中于小域的残余优化机制（+0.0054）；(5) 修正 objective + comp1 后 largest-domain sacrifice 完全消失（Δ_L = −0.0001）。论文核心应从"trade-off 缓解"转向"objective 错配诊断 + 轻量修正"，comp1 降级为可选的增量组件。

## 11. Next-stage recommendation

Priority 0 闭合。进入 Priority 1，但顺序按证据价值重排（main 决策，理由如下）：

- **先做 P1-2（Curriculum Life-or-Death）**：curriculum 仍是论文声称的核心组件，且其全部既有证据来自 equal 口径，必须在 sample weighting 下重判 keep/drop；这是论文结构决策，优先级最高。注意 curriculum 与 comp1 的交互也需在修正口径下重估（P0-5 已改变 comp1 的定位）。
- **P1-3（Matched Momentum Control）其次**：范围收窄为解释 +0.0054 残余——若 matched uniform EMA 即可获得该残余，comp1 的 domain-conditioned 结构无独立价值，可整体降为补丁。
- **P1-1（机制仪表化）降级为按需**：仅当 P1-3 显示残余机制为 domain-conditioned 特有时才值得做全量梯度轨迹仪表化。
