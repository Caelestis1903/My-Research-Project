# Stage S6 — Final Analysis：Sample Objective Residual History Gate

- 分析者：Main Agent（k3-256k）
- 日期：2026-09-29
- 研究问题（promt/S6.md）：将训练 objective 从 Equal 改为 Sample 后，S5/S5-ML 观察到的 Adam persistent-history mediated target rescue 是否仍存在稳定、practical magnitude 的 residual？
- 关联文档：`promt/S6.md`（设计）、`experiments/stage_S6_design.md`（实现规格）、`experiments/stage_S6_phaseA_analysis.md`（MovieLens Phase A 详细分析）、runner 报告 `experiments/stage_S6_outputs/reports/stage_S6_run_report.md` 与 `experiments/stage_S6ML_outputs/reports/stage_S6ML_run_report.md`。

---

## 1. 实验配置与执行摘要

| 项 | KuaiRand（S6） | MovieLens（S6ML） |
|---|---|---|
| setting / target | natural / d1（tab1，最大域 ~86.9%） | d2_large / d2（~80%） |
| tasks | d1t0=is_click, d1t1=long_view | d2t0=rating>3, d2t1=rating>4 |
| Phase A arms | F, T | F, T |
| seeds | 2022–2025 | 2022–2025 |
| epochs | 50 | 50 |
| objective | sample（w_d=1，a_d=p_d） | 同左 |
| run 状态 | 8/8 成功（seed2024 因瞬时 CUDA 硬件错误重试 1 次成功） | 8/8 成功，0 重试 |
| SAMPLE_TABLE pin | `30c68c9b…` 与 S3 审计一致 | `e0616863…` 与 S3 审计一致 |
| 审计 | config_lock_audit 96/96 OK（reviewer 裁决 condition 字段 FAIL 为 audit 期望错误并修复）；pairing 全等；checkpoint 纪律通过 | STRUCTURALLY_VALIDATED；pairing 全等 |
| 唯一实验变量 | objective Equal→Sample（其余逐项锁定，config_lock_audit 证实） | 同左 |

Phase B（HT/DH）：**未触发**（两个数据集 Phase A 均 STOP），见 §4。

---

## 2. Phase A Primary Results：R_T^Sample = T − F

### 2.1 KuaiRand（target d1）

**Selected checkpoint ΔAUC：**

| seed | d1t0 | d1t1 |
|---|---|---|
| 2022 | −0.00334 | −0.00362 |
| 2023 | −0.00031 | −0.00162 |
| 2024 | −0.00433 | −0.00452 |
| 2025 | +0.00536 | +0.00660 |
| **mean** | **−0.00066** | **−0.00079** |
| 正向 seed | 1/4 | 1/4 |

**Final ΔAUC：** d1t0 mean −0.00163（0/4 正向）；d1t1 mean −0.00189（0/4 正向）。

**Common-epoch ΔAUC（4-seed mean，全部 0/4 正向）：**

| ckpt | d1t0 | d1t1 |
|---|---|---|
| common9 | −0.0021 | −0.0021 |
| common19 | −0.0020 | −0.0024 |
| common29 | −0.0016 | −0.0018 |
| common39 | −0.0013 | −0.0013 |
| common49 | −0.0016 | −0.0019 |

**Selected ΔLogLoss：** d1t0 mean +0.00242（2/4 变差）；d1t1 mean +0.00054（2/4 变差）。方向混杂。

### 2.2 MovieLens（target d2）（详见 stage_S6_phaseA_analysis.md）

- selected ΔAUC：d2t0 **+0.00190**（2/4 正向）、d2t1 **+0.00256**（2/4 正向）
- final ΔAUC：d2t0 +0.00017、d2t1 −0.00207
- common-stage：mid-training（c19–c39）以负向为主，无稳定正向观察点
- selected ΔLogLoss：2/4 seed 明显变差（d2t0 最差 +0.0178）

---

## 3. Phase A Gate 判定（S6.md §7.3，预注册标准）

| 判据 | KuaiRand d1t0 | KuaiRand d1t1 | MovieLens d2t0 | MovieLens d2t1 |
|---|---|---|---|---|
| ① selected mean ≥ +0.003 | ✗（−0.0007） | ✗（−0.0008） | ✗（+0.0019） | ✗（+0.0026） |
| ② ≥3/4 seeds positive | ✗（1/4） | ✗（1/4） | ✗（2/4） | ✗（2/4） |
| ③ LogLoss 无稳定反向 | ✗（混杂） | ✗（混杂） | ✗ | ✗ |
| ④ common-stage 多数观察点为正 | ✗（0/5） | ✗（0/5） | ✗ | ✗ |
| ⑤ 非单 seed/checkpoint 驱动 | —（唯一正 seed 2025 为孤点） | 同左 | ✗ | ✗ |

**KuaiRand = STOP。MovieLens = STOP。**

纪律性表述（S6.md §7.3）："不显著"不等于"等于零"。MovieLens selected mean 为正但低于阈值且方向不稳；KuaiRand mean 为微负。两者共同的稳健事实是：**Sample 下不存在达到 practical magnitude、跨 seed 稳定的 F→T target rescue**；观察到的逐 seed 摆动（±0.005 量级）落在 seed 自然方差范围内。

---

## 4. 分支裁决（S6.md §8 / §17）

两个数据集均 STOP → **Case A**。

**最终裁决：A. STOP METHOD LINE**

正式结论（按 S6.md Case A 规定措辞收束）：

> **Equal objective 会诱发可被 persistent optimizer history 携带的优化动力学效应（S5/S5-ML 已证实为真实机制），但该效应在 Sample objective 下未达到稳定 practical residual。Adam-history mechanism 是真实机制，但不再自动意味着需要新方法。**

因此：不进入 Phase B；不做 m/v factorization；不做 ratio/stage boundary experiment（作为方法前置研究失去依据）；不把 Adam-history 发展成新 optimizer。

---

## 5. Equal vs Sample 跨 objective 对照（S6.md §10）

| dataset | task | Equal（S5/S5-ML anchor）selected T−F | Sample（S6）selected T−F | 幅度变化 |
|---|---|---|---|---|
| KuaiRand | d1t0 | ≈ +0.041（稳定正向） | −0.0007（1/4 正向） | 消失并微翻负 |
| KuaiRand | d1t1 | ≈ +0.042（稳定正向） | −0.0008（1/4 正向） | 消失并微翻负 |
| MovieLens | d2t0 | +0.0194（4/4 正向） | +0.0019（2/4 正向） | −90% |
| MovieLens | d2t1 | +0.0143（4/4 正向） | +0.0026（2/4 正向） | −82% |

三种解释区分（S6.md §10 要求）：
- **objective 改变了 phenomenon magnitude：成立且是主解释。** 两个数据集一致，幅度缩水 82%–102%。
- **objective 改变了 mechanism decomposition：无法检验**——Sample 下无稳定 residual，Phase B 未触发，R_H/R_D 无定义。
- **objective 仅改变 convergence/selected epoch：不成立为主解释。** selected epoch 分布（KuaiRand F: 24–47，T: 27–49；MovieLens F: 7–12，T: 6–12）无系统性偏移能解释 rescue 消失；common-stage 固定 epoch 对照（控制收敛差异）下 rescue 同样不存在。

---

## 6. Training-stage 结构（S6.md §11）

1. **Sample 下 rescue 并非"更早消失"，而是全训练阶段均未成形**：KuaiRand 全部 5 个 common 观察点 0/4 正向（均值 −0.0013 ~ −0.0024）；MovieLens mid-training 以负向为主。Equal 下 KuaiRand 的 rescue 是稳定全程现象，Sample 将其全阶段抹平。
2. **history contribution 是否仅存于 early training：问题失去前提**——Sample 下任何阶段都无稳定正向 target rescue 可归属。
3. **DH late-stage 上升问题：未触发**（Phase B 未运行）。
4. **overfitting onset**：MovieLens F 臂在 ~epoch 29 后整体过拟合崩塌（d1t0 LogLoss 至 1.2–1.4），T 臂平稳——但该 T 优势是全局正则化副产品而非 target rescue（详见 phaseA 分析 §4.2）；KuaiRand 两臂均无此崩塌。
5. **checkpoint shift**：无证据表明 selected epoch 跟随 history factor 系统性移动。

## 7. Non-target / Global outcomes（S6.md §12）

**KuaiRand selected（T−F mean ΔAUC）**：d0t0 −0.0181、d0t1 −0.0306、d2t0 −0.0164、d2t1 −0.0139、macro **−0.0134**（4/4 负向）。T 的 target-only 更新对两个小域造成稳定损伤，target 本身无收益——**trade-off 存在但 rescue 侧为空**。

**MovieLens selected**：d0t0 −0.0240、d0t1 −0.0316（4/4），d1t0 +0.0059、d1t1 +0.0172（稳定正向），macro −0.0047（4/4 负向）。损伤集中于最小域 d0。

两数据集一致：**Sample 下 T 臂没有 target rescue 可言，却保留了（甚至重排了）non-target 损伤结构**——损伤沿域大小向小域集中。T 臂在任何意义上都不是 deployable direction，本阶段也从未以此为目标。

## 8. 逐条回答 S6.md §16 的 18 个问题

1. **KuaiRand Sample 下 T−F 是否存在？** 否。selected mean −0.0007/−0.0008，1/4 正向，common-stage 全负。
2. **MovieLens Sample 下 T−F 是否存在？** 不存在稳定 practical residual（+0.0019/+0.0026，2/4 正向，低于阈值）。
3. **magnitude 相比 Equal 缩小多少？** KuaiRand 从 +0.041/+0.042 到 ≈0（微负）；MovieLens 缩小 82–90%。
4. **是否达到 0.003 practical threshold？** 两数据集四组 target 测量均未达到。
5. **是否 seed-stable？** 否（KuaiRand 1/4，MovieLens 2/4）。
6. **是否 task-stable？** 否。
7. **是否 checkpoint-stable？** 否（common-stage 与 selected 不一致，KuaiRand common 全负）。
8. **AUC/LogLoss 是否一致？** 否（多处 AUC 微正伴随 LogLoss 变差）。
9. **是否仍存在 non-target damage？** 是——T 臂对 non-target 小域的损伤在 Sample 下依然稳定存在（KuaiRand d0t1 −0.031；MovieLens d0t1 −0.032），这进一步说明 T 臂语义是"牺牲全域保 target"，而 Sample 下连"保 target"都不再成立。
10. **Sample 是否基本消除了 Equal 下的 history effect？** 是。两个数据集一致。
11. **HT 是否仍≈T？** 未检验（Phase B 未触发，gate 纪律禁止在 STOP 后做 factorization）。
12. **DH contribution 在 Sample 下是多少？** 未检验，同上。
13. **KuaiRand/MovieLens 的 DH discrepancy 是否仍存在？** 未检验；MovieLens Equal 下的 DH≉F 异象在 Sample 下无对应物可比较。
14. **rescue 与 damage 是否仍可分离？** 未检验；且 rescue 侧为空使该问题失去意义。
15. **residual 是否大到值得继续研究？** 否。
16. **是否值得进入 ratio/training-stage boundary experiment？** 否（作为 history 机制研究的前置已不成立）。
17. **是否值得进入 m/v factorization？** 否。
18. **是否仍存在发展新方法的必要性？** 否。Equal 下的 history 效应是 Equal 这一失配 objective 的下游病理；Sample 这一简单 objective correction 已将其消除至 practical threshold 以下。

## 9. Competing explanations 与 evidence strength

**主解释（最简约）**：Equal（a_d=1/3）人为压低了 dominant domain 的有效权重，制造了目标域的"欠训练"状态，persistent Adam history 操作实质上是部分还原该权重；Sample（a_d=p_d）本就给予 dominant domain 其经验频率权重，失配不存在，history 操作自然失去效应空间。该解释同时兼容：(a) 两数据集 target 均为最大域、(b) S6.md 既有事实 2（Sample 缓解 Large-cell degradation）、(c) 本阶段 rescue 消失、(d) T 臂 non-target 损伤在两种 objective 下持续存在。

**备选解释 1（history effect 真实但缩到阈值以下）**：不能证伪（MovieLens mean 微正），但 seed 方向不稳、common-stage 不支持、KuaiRand 微负，证据强度不足以支持继续投入。

**备选解释 2（Sample 改变了训练动力学使四臂语义不可比）**：被 config_lock_audit 与 pairing audit 否定——除 objective 外所有条件逐位锁定，F/T 配对完全配对。

**Evidence strength**：强。两数据集 × 4 seeds × 2 tasks × 7 checkpoint 层次、预注册阈值、完全配对的实验控制；方向一致（无稳定 practical residual）。局限：seed 数为 4；Phase B 未触发故 mechanism decomposition 在 Sample 下无数据（这是 gate 设计的预期行为而非缺陷）。

**对 S5/S5-ML 结论的回溯影响**：S5 的 Pattern A（HT≈T、DH≈F）与 S5-ML 的 partial replication 结论在其 Equal objective 语境下仍然成立、不受本阶段影响；本阶段改变的是其**外推范围**——该机制是 Equal pathology 的 downstream mechanism，不是 objective-independent 的普遍现象。

## 10. 最终裁决与后续建议

**裁决：A. STOP METHOD LINE。**

- Adam-history 研究线以"真实但 objective-contingent 的机制"收束；
- 不进入 Phase B、不做 m/v 拆解、不做 ratio/stage boundary、不做方法化；
- 若未来撰写报告/论文，推荐表述为：「persistent optimizer state 的跨域效应是失配训练目标（Equal）诱发的真实优化动力学现象；恢复样本频率目标（Sample）后该现象消失至实践阈值以下。这一负结果本身界定了该机制的边界条件：它是 objective-induced pathology，而非普适的跨域优化现象。」
- 实验资产（S6/S6ML 全部 16 个 run、审计链、config_lock 机制）保留可复用；若未来在新 setting（target 为小域、非 natural ratio）重提该问题，需作为新阶段重新立项而非本阶段延续。

## 11. 输出文件清单（S6.md §18 对照）

| 要求 | 状态 |
|---|---|
| stage_S6_design.md | ✅ `experiments/stage_S6_design.md` |
| stage_S6_phaseA_run_report.md | ✅ runner 产物：`experiments/stage_S6_outputs/reports/stage_S6_run_report.md`、`experiments/stage_S6ML_outputs/reports/stage_S6ML_run_report.md` |
| stage_S6_phaseA_analysis.md | ✅ `experiments/stage_S6_phaseA_analysis.md`（MovieLens）；KuaiRand gate 分析并入本文件 §2.1/§3 |
| stage_S6_phaseB_run_report.md | 未触发（双 STOP，gate 纪律禁止 Phase B） |
| stage_S6_analysis.md | ✅ 本文件 |
