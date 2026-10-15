# P1-2 Stage Research Review：Curriculum Life-or-Death（修正口径重判）

- 阶段：P1-2（Priority 1，curriculum keep/drop 决策实验）
- 口径：sample weighting（w=3p，Σw=6，修正后 objective）——equal 口径下的 asymmetric transfer 已被 P0-4 证明是 objective 人造，curriculum 必须在干净口径下证明独立价值
- 设计：6 臂 = curriculum {none(A), forward κ0=+0.5(B), reverse κ0=−0.5(C)} × comp1 {off(0), on(1)}，strong 档 ratio_50（L/M/S=150000/34500/3000，三 setting 轮换），seeds 2022-2025，72 run 全 OK
- 机制核实（reviewer 复核）：κ_d = κ0·log(p_max/p_d)/log(p_max/p_min)；forward = 前半程压低小域自身 fusion weight（sigmoid(e−κ_d(1−ρ))），迫使小域借跨域信息；ρ=1 后与原始 MDMTRec 逐位一致——curriculum 只重塑训练轨迹，不改模型容量
- 回归锚点：A0 vs P0-4 w_sample、A1 vs P0-5 sample+comp1 共 24 格 config + metrics 逐位一致（PASS 24/24）
- 原始报告：`experiments/reports/P1_2_RAW_REPORT.md`；汇总：`experiments/P1_2_curriculum/P1_2_{paired,per_domain,peak_epoch,grad_norm,rho_sanity}.csv`

## 1. Stage research question

Curriculum（comp2）在修正后 objective 口径下是否值得保留为论文核心组件？判定规则（promt.md P1-2 实例化，12 配对单元）：KEEP 需满足 ①稳定性能增益（|median Δ|≥0.002 且方向一致）②或稳定可解释的动态变化（peak epoch 中位平移≥2 或 grad norm 轨迹在 ρ<1 半程方向性分离）③或与 comp1 的可重复交互；四条 DROP 条件同时成立则 DROP。

## 2. Experimental facts

最终性能配对差（n=12，median [IQR]，符号一致率）：

| 配对 | overall AUC | worst-domain AUC |
|---|---|---|
| B0−A0（forward−none） | +0.0008 [−0.0005, +0.0024]，7/12 | +0.0012 [−0.0003, +0.0023]，8/12 |
| C0−A0（reverse−none） | +0.0011 [−0.0003, +0.0017]，8/12 | +0.0012 [−0.0001, +0.0024]，8/12 |
| C0−B0（reverse−forward） | +0.0004 [−0.0001, +0.0010]，9/12 | +0.0003 [−0.0010, +0.0015]，6/12 |
| B1−A1（comp1 层 forward−none） | +0.0003 [−0.0017, +0.0011]，7/12 | −0.0009 [−0.0021, +0.0003]，3/12 |
| C1−A1 | +0.0004 [−0.0017, +0.0014]，7/12 | −0.0001 [−0.0032, +0.0017]，6/12 |
| 交互 (B1−A1)−(B0−A0) | −0.0010 [−0.0019, +0.0010]，5/12 | −0.0015 [−0.0043, +0.0002]，4/12 |

peak epoch 配对差（中位数）：B0−A0 全角色 0；C0−A0 0~0.5；comp1 层 B1−A1 −1~+1。无任何 ≥2 epoch 的稳定平移。

gradient norm 轨迹：epoch 0 的 shared grad norm A0=5.63e-2 vs B0=5.54e-2（几乎重合）；逐 epoch 轨迹各臂均值带全程重叠，ρ<1 半程无方向性分离。

rho sanity：B/C 臂 ρ 0→~0.9 线性（早停截断，符合设计）、κ_d 符号正确（forward 小域 +0.5、大域 0）、A 臂恒 1。

## 3. Main observations

**Strong evidence**
- curriculum 对最终性能无稳定影响：全部配对 |median| ≤ 0.0012（低于 0.002 阈值），IQR 全部跨 0，符号一致率 0.58-0.75（正向噪声水平）。
- reverse 与 forward 无实质区别：C0−B0 median +0.0004，reverse 若有任何倾向反而是略优——方向性操纵不产生方向性结果，说明该机制在此口径下无杠杆。
- 动态通道同样为空：peak epoch 无平移、grad norm 轨迹重合、前半程（ρ<1，curriculum 实际起作用的窗口）无可识别分离。
- 与 comp1 无交互：交互项 median 为负（−0.0010）且符号率 5/12，方向都不稳定。

**Moderate evidence**
- comp1 层（A1/B1/C1）peak epoch 整体晚于 comp1-off 层（11-12.5 vs 7-9）——这是 comp1 的效应，与 curriculum 臂无关（同层内三臂 peak 几乎相同）。

**Weak evidence / Speculation**
- 无。

## 4. Competing hypotheses

- H1：curriculum 在修正口径下有独立价值（KEEP）。
- H2：curriculum 的全部历史证据来自 equal 口径的 objective 错配环境；错配修复后机制冗余（DROP）。
- H3：curriculum 有效但 κ0=0.5 太弱（参数问题而非机制问题）。

## 5. Evidence supporting each hypothesis

- H2：四条 DROP 条件全部满足（见 §8）；P0-4/P0-5 已证明 equal 口径的 trade-off 主要是 objective 人造，curriculum 的历史增益（若有）与 comp1 一样大概率是补丁效应。
- H3：无法排除——本实验按设计不扫 κ0。

## 6. Evidence against each hypothesis

- H1：KEEP 三条无一满足。
- H3：ρ=0 时 κ_d 对小域的偏移（sigmoid 1.0→0.62）幅度不小，机制在前半程确实实质性改变了前向（test_equivalence 单测确认 rho=0 改变 forward）；但轨迹与终点均无效应——"太弱"辩护与"前半程有实质干预却零效应"的观察相矛盾，H3 支持很弱。

## 7. Major confounds

- 早停使多数 run 在 ρ 到达 1 之前停止（rho_max 0.88-0.92）：curriculum 的"完整课程"从未跑完，严格说测的是"curriculum 的前半程干预 + 早停"组合；但早停是全部臂共享的标准协议，且 ρ 调度设计本就是前半程生效。
- grad_norm_shared 在 comp1 臂是优化器实际施加的 EMA 平滑梯度（Σm_d），与 comp1-off 臂的瞬时梯度口径不同；臂间比较只在同 comp1 层内有效——本报告遵守该口径。
- 结论限于 κ0=±0.5、T_c=total/2、strong 档、ML-1M。

## 8. What has been falsified?

**DROP Curriculum——四条 DROP 条件逐条核验**：

| 条件 | 证据 | 成立 |
|---|---|---|
| 最终性能无稳定差异 | 全部 6 组配对 \|median\| ≤ 0.0012 < 0.002，IQR 跨 0 | ✓ |
| peak epoch 无稳定变化 | 配对中位差 ≤ 1，无 ≥2 平移 | ✓ |
| gradient dynamics 无稳定变化 | shared/分域 norm 轨迹各臂重合，ρ<1 半程无分离 | ✓ |
| reverse 与 forward 无区别 | C−B median +0.0004，方向率 9/12 但量级为噪声级 | ✓ |

KEEP 三条均不满足。**Curriculum 不再作为论文核心组件**（promt.md：禁止为保原论文设想强行保留）。H1 证伪；H2 成立；H3 仅存理论可能但无支持证据。

## 9. What remains unresolved?

- comp1 残余 +0.0054 的机制归属（P1-3 matched momentum control 待做）。
- curriculum 在 equal 口径下的历史表现是否纯补丁（exploratory 探针未跑，判定不依赖它；cut）。
- 若未来换数据集/域数（P2-2 D=5），curriculum 是否仍无效——本结论不约束该情形，但举证责任在"保留方"。

## 10. Research direction decision

**Reject current hypothesis（curriculum 作为核心组件）**。论文组件清单修正为：sample-proportional loss weighting（主修正，+0.021）+ comp1（可选增量，+0.0054 集中于小域）。Curriculum 从核心组件中移除；若论文需交代，以 P1-2 作为消融阴性结果如实报告（含 reverse 对照）。

## 11. Next-stage recommendation

进入 **P1-3（Matched Momentum Control）**，目标收窄为解释 comp1 的 +0.0054 残余：三臂——current comp1（γ_d domain-conditioned）、matched uniform EMA（匹配有效记忆长度）、matched Adam beta1 control。若 matched uniform 即得同等残余，comp1 的 domain-conditioned 结构无独立价值，可进一步简化论文叙事；若 comp1 显著优于 matched 控制，残余是 domain-conditioned 特有，再考虑是否值得 P1-1 全量机制仪表化。setting 用 strong 档 ratio_50 + sample weighting（残余效应所在口径），seeds 2022-2025。
