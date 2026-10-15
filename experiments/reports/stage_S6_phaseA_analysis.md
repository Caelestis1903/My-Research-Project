# Stage S6 — Phase A Analysis（MovieLens-1M / d2_large / Sample objective）

> 2026-10-08 状态与表述更正：这是 2026-09-28 的 MovieLens Phase A 历史报告，正文中“KuaiRand 未运行”仅描述当时状态。KuaiRand Phase A 后续已完成，本次 HT/DH 也已补齐；当前完整矩阵与审计限制请读 [stage_S6_analysis.md](stage_S6_analysis.md)。MovieLens 的域大小为 d0=18.4%（中等）、d1=1.6%（最小）、d2=80%（最大），因此下文把 d0 写为“最小域”的解释错误。seed 2024 的 selected target ΔAUC 是负值，不应写成“AUC 正而 LogLoss 变差”。这些更正不改变原 selected T−F gate 的 STOP 判定。Sample 下所有 optimizer 分量效应均消失的宽泛外推由完整四臂报告进一步限定。

- 分析者：Main Agent（k3-256k）
- 日期：2026-09-28
- 范围声明：本文件仅覆盖 **MovieLens（S6ML）Phase A**。KuaiRand（S6）Phase A 按用户指令暂未启动，本阶段最终裁决（STOP / CONDITIONAL / ROBUST）需待 KuaiRand Phase A 完成后作出。
- 数据来源：`experiments/stage_S6ML_outputs/stage_S6ML_paired_results.csv`（168 行，pairing_valid 全 True）、`stage_S6ML_raw_results.csv`（336 行）、`stage_S6ML_summary_manifest.json`（STRUCTURALLY_VALIDATED）、runner 报告 `experiments/stage_S6ML_outputs/reports/stage_S6ML_run_report.md`（即 S6.md 第十八节所要求的 phaseA run report 之 MovieLens 部分）。
- 实验执行：8/8 run 成功，0 retry，0 crash；SAMPLE_TABLE_SHA256 = `e0616863…3463` 与 S3 审计 pin 一致；同 seed F/T pairing 全等；checkpoint selection frozen-before-test；final == common49 数值相等；nonfinite 为空。

---

## 1. Primary contrast：R_T^Sample = T − F（target domain d2）

### 1.1 Selected checkpoint（ΔAUC, T−F）

| seed | d2t0 | d2t1 |
|---|---|---|
| 2022 | −0.00109 | −0.00042 |
| 2023 | +0.00626 | +0.00683 |
| 2024 | −0.00316 | −0.00332 |
| 2025 | +0.00559 | +0.00713 |
| **mean** | **+0.00190** | **+0.00256** |
| 正向 seed 数 | 2/4 | 2/4 |

selected epoch：F = {7,8,8,12}，T = {8,12,10,6}（per seed 2022–2025）。

### 1.2 Final checkpoint（= common49，ΔAUC, T−F）

| seed | d2t0 | d2t1 |
|---|---|---|
| 2022 | +0.00180 | +0.00203 |
| 2023 | +0.00201 | −0.00654 |
| 2024 | −0.01018 | −0.00528 |
| 2025 | +0.00703 | +0.00152 |
| **mean** | **+0.00017** | **−0.00207** |
| 正向 seed 数 | 3/4 | 2/4 |

### 1.3 Common-epoch trajectory（ΔAUC T−F，4-seed mean）

| checkpoint | d2t0 mean | d2t0 正向 seed | d2t1 mean | d2t1 正向 seed |
|---|---|---|---|---|
| common9 | +0.00198 | 3/4 | −0.00092 | 1/4 |
| common19 | −0.00299 | 1/4 | −0.00002 | 3/4 |
| common29 | −0.00519 | 0/4 | −0.00160 | 2/4 |
| common39 | −0.00231 | 1/4 | +0.00207 | 3/4 |
| common49 | +0.00017 | 3/4 | −0.00207 | 2/4 |

共同阶段无稳定正向结构：mid-training（c19–c39）以负向为主，无任何观察点达到 4/4 或稳定 3/4 同方向且幅度达标。

### 1.4 LogLoss（selected，Δ = T−F，负值为 T 更优）

| seed | d2t0 | d2t1 |
|---|---|---|
| 2022 | +0.00006 | −0.01116 |
| 2023 | +0.00447 | +0.00333 |
| 2024 | +0.01776 | +0.01254 |
| 2025 | −0.02485 | −0.01644 |
| **mean** | **−0.00064** | **−0.00293** |

AUC 与 LogLoss 方向不一致：selected 上 d2t0 AUC mean 为正（+0.00190）但 2/4 seed 的 LogLoss 明显变差（2023 +0.0045、2024 +0.0178），d2t1 亦然。AUC 正向主要由 2023/2025 两个 seed 驱动。

---

## 2. Phase A GO/STOP Gate 判定（S6.md §7.3，MovieLens 独立判定）

| 判据 | d2t0 | d2t1 |
|---|---|---|
| ① selected mean ΔAUC ≥ +0.003 | ✗（+0.00190） | ✗（+0.00256） |
| ② ≥3/4 seeds positive | ✗（2/4） | ✗（2/4） |
| ③ LogLoss 无稳定反向 | ✗（2/4 seed 明显变差） | ✗（2/4 seed 明显变差） |
| ④ common-stage 多数观察点为正 | ✗（5 点中 2 点为正且不稳定） | ✗（5 点中 1 点为正） |
| ⑤ 非单 seed / 单 checkpoint 驱动 | ✗（正向集中于 2023/2025 与 selected） | ✗（同左） |

**判定：MovieLens = STOP。**

满足 STOP 主情形中的多条：两个 target task 的 selected mean 均 < +0.003；seed 方向不稳定（2/4）；common-stage 不复现；AUC/LogLoss 存在系统性不一致。按预注册标准，MovieLens 不进入 Phase B（不跑 HT/DH）。

按 S6.md 纪律要求明确表述：这不等于"效应为零"。selected mean 为正（+0.0019/+0.0026），但幅度低于 practical threshold，方向跨 seed 不稳定，且落入 seed 间自然波动范围（逐 seed Δ 从 −0.0032 到 +0.0071 摆动）。结论应表述为：**Sample objective 下 MovieLens 未观察到稳定、达到 practical magnitude 的 F→T target rescue residual。**

---

## 3. 与 Equal（S5-ML anchor）的跨 objective 比较

| 指标 | Equal（S5-ML） | Sample（S6ML Phase A） | 变化 |
|---|---|---|---|
| d2t0 selected T−F | +0.0194（4/4 正向） | +0.00190（2/4 正向） | 缩小约 90%，方向失稳 |
| d2t1 selected T−F | +0.0143（4/4 正向） | +0.00256（2/4 正向） | 缩小约 82%，方向失稳 |

magnitude reduction 是压倒性的：Equal 下明确、稳定、超 strong threshold 的 target rescue，在 Sample 下缩水到 practical threshold（0.003）以下且 seed 方向各半。这属于 S6.md 第十节所区分的「objective 改变了 phenomenon magnitude」的情形；由于 Sample 下无稳定 residual，mechanism decomposition（HT/DH）对该数据集不再触发。

---

## 4. Training-stage 结构观察（S6.md 第十一节）

1. **Sample 下 rescue 并非"更早消失"，而是全阶段均未成形**：Equal S5-ML 曾有明显 early-window 正向行为；Sample 下 common9 仅 d2t0 弱正（+0.0020，3/4），d2t1 已为负，mid-training（c19–c39）两臂差异以负向为主。没有任何训练阶段出现稳定正向 target rescue。
2. **晚期主导信号是 F 的整体崩塌而非 T 的 target rescue**：F 臂在 c29 之后宏观性能持续恶化（macro AUC 从 c9 的 ~0.727–0.734 跌至 c49 的 ~0.688–0.702；d1t0 LogLoss 恶化至 1.20–1.43），而 T 臂全程平稳。final/common49 处 T−F 的 macro AUC 差异约 +0.02（4/4 seed 正向），但该差异遍及所有 non-target cell（d0/d1 全部大幅为正），target d2 反而最小且方向不稳。这是「T 的 target-only 更新意外充当了抗过拟合正则」的全局效应，不是 target rescue，二者不得混淆。
3. **checkpoint shift**：T 的 selected epoch（均值 9.0）与 F（均值 8.75）接近，无证据表明 selected 点跟随某一机制因子系统性移动。
4. **overfitting onset**：Sample 下 F 臂在 ~epoch 29 后开始明显过拟合崩塌；该现象是 F 臂自身轨迹属性，与臂间对比无关。

---

## 5. Non-target / Global outcomes（S6.md 第十二节）

### 5.1 Selected checkpoint（T−F，4-seed mean ΔAUC）

| cell | mean ΔAUC | 方向 |
|---|---|---|
| d0t0 | −0.0240 | T 显著更差（4/4） |
| d0t1 | −0.0316 | T 显著更差（4/4） |
| d1t0 | +0.0059 | T 更好（3/4） |
| d1t1 | +0.0172 | T 更好（4/4） |
| d2t0 (target) | +0.0019 | 不稳（2/4） |
| d2t1 (target) | +0.0026 | 不稳（2/4） |
| macro | −0.0047 | T 更差（4/4） |

selected 上 T 的 target-only 更新对 d0（最小域）造成稳定且大幅的损伤（−0.024/−0.032），对 d1 有稳定收益，对 target 本身反而无稳定收益；macro 层面 T 全面更差。**Sample 下 T 不存在"rescue target"可言，却保留了 non-target 损伤结构（且方向沿域大小重排：损伤集中到最小域 d0）。**

### 5.2 Final checkpoint

F 臂晚期崩塌使所有 cell 上 T−F 转为大幅正向（d0t1 最小约 +0.005，d1t0 最大约 +0.053），macro T−F ≈ +0.020。该结果反映的是 F 的过拟合而非 T 的优势机制，见 §4.2。

---

## 6. 对 S6.md 第十六节问题的逐条回答（MovieLens 部分）

2. **MovieLens Sample 下 T−F 是否仍存在？** 不存在稳定 practical residual。selected mean +0.0019/+0.0026，低于 0.003 阈值，2/4 seed 正向。
3. **magnitude 相比 Equal 缩小多少？** d2t0 约 −90%（+0.0194→+0.0019），d2t1 约 −82%（+0.0143→+0.0026）。
4. **是否达到 0.003 practical threshold？** 否，两个 target task 均未达到。
5. **是否 seed-stable？** 否（2/4）。
6. **是否 task-stable？** 否（final/common49 上 d2t1 mean 为负）。
7. **是否 checkpoint-stable？** 否（selected 弱正，common19/29 为负，final 近零）。
8. **AUC/LogLoss 是否一致？** 否（selected 上 2023/2024 两个 seed AUC 正而 LogLoss 明显变差）。
9. **是否仍存在 non-target damage？** 是，且结构改变：selected 上损伤集中于最小域 d0（−0.024/−0.032，4/4 稳定），而非 Equal 下的泛化 non-target cost。
10. **Sample 是否基本消除了 Equal 下的 history effect？** 就 MovieLens 的 target rescue 而言：是。幅度缩水 ~82–90% 且方向失稳。
11–14.（HT/DH、recovery ratio、rescue/damage separability）MovieLens STOP，不进入 Phase B，不适用。
15. **residual 是否大到值得继续研究（MovieLens）？** 否。按预注册 gate，MovieLens 侧不支持继续机制分解。

---

## 7. 当前分支状态（S6.md 第八节）

- MovieLens：**STOP** → 不做 HT/DH。
- KuaiRand：未运行（用户指令暂缓）。
- 因此当前既不能裁决 Case A（双 STOP），也不能裁决 Case C（双 GO）。待 KuaiRand Phase A 完成后：
  - KuaiRand STOP → Case A（STOP METHOD LINE）；
  - KuaiRand GO → Case B（CONDITIONAL RESIDUAL，仅 KuaiRand 进 Phase B）。

## 8. Counterexamples / competing explanations / evidence strength

- **Counterexample to "history effect 是 objective-independent"**：本结果就是最强反例——同一数据集、同一 backbone、同一臂定义，仅切换 objective，target rescue 从 +0.019/+0.014（4/4）跌至阈值以下且方向失稳。与 S6.md 既有事实 2（Sample 缓解 Large-cell degradation）一致。
- **Competing explanation 1（target 即最大域）**：MovieLens d2 占约 80% 样本，Sample 下其 effective coefficient a_d=p_d≈0.8，本就是训练主导信号；Equal 下的"rescue"可能本质上是把被人为压低的 dominant-domain 权重还原。Sample 完成该还原后，history 操作再无额外可挖掘空间。该解释与数据一致，且比"history residual 存在但太小"更简约。
- **Competing explanation 2（晚期 F 崩塌污染 final 判读）**：final/common49 的 T−F 全局正向由 F 过拟合驱动，若误读为"T 的 rescue 在晚期回归"将得出错误结论；common-stage 轨迹（c19–c39 负向）否定了该解读。
- **Evidence strength**：中-强。4 seeds、双 task、7 checkpoint 层次一致地指向"无稳定 target residual"；主要局限是 seed 数（4）与单数据集，且 KuaiRand 半侧尚缺。

## 9. 运维备注

- `stage_S6ML_config_lock_audit.csv` 未生成：设计文档 §4.5–4.8 将 S6ML report 定义为 S5ML 的对称克隆（S5ML 本无此 audit），reviewer 认定非阻塞；S6ML 侧的锁定保证由 orchestrator 硬 pin（dataset/coefficient/init hash）与 pairing audit 承担。如需补齐，应由 designer 显式修订设计。
- coder 验证残迹已隔离于 `experiments/stage_S6_coder_verification_debris/`（KuaiRand 侧），不影响本结果。

## 10. 下一步建议（提交用户裁决）

1. 按用户节奏决定是否启动 KuaiRand（S6）Phase A；其结果将决定最终裁决落入 Case A 还是 Case B。
2. MovieLens 侧按预注册 gate 不做 Phase B。
3. 若 KuaiRand 也 STOP，按 S6.md Case A 收束主结论：「Equal objective 会诱发可被 persistent optimizer history 携带的优化动力学效应，但该效应在 Sample objective 下未达到稳定 practical residual」——Adam-history 是真实机制，但不再自动意味着需要新方法。
