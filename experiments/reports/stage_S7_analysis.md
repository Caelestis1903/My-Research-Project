# Stage S7-A 分析报告 — Sample Objective 下 MDMT Domain×Task Gradient Dynamics 诊断

阶段性质：diagnostic / falsification。本报告由 main agent 在 runner 完成全部 8 个 probe run 后撰写。
数据基础：`experiments/stage_S7_outputs/stage_S7_gradient_raw.csv`（13,824 行）、`stage_S7_performance_raw.csv`（288 行）、`experiments/stage_S7_analysis_tables/T1–T12`。

术语约定：cell = task×3+domain（task-major）。MovieLens target=d2（cells 2,5；share 0.8）；KuaiRand target=d1（cells 1,4；share 0.869）。U_k 与 H_{i→k} 为局部一阶近似量，本报告不做因果声称。ConflictMass 仅为 geometry statistic。`target_weighted_*` 为全总体份额加权（S1 命名沿袭），`target_domain_*` 为 target 域两 cell 简单均值。

---

## A. Execution validity

- 8/8 probe run（2 数据集 × 4 seeds）全部通过 `validate_probe_run`：行数 1728/run、|cos|≤1+1e-9、重构恒等式 g_S=Σw_i g_i 与 Σ_i H_{i→k}=U_k 在容差内（逐 run 重构残差最大 2.23e-7；聚合层恒等式最大误差 7.21e-9）。
- 全部 run 的 `initial_parameter_state_hash` 与 S6 源 manifest 逐位匹配；模型源 hash 匹配；batch 指纹跨 checkpoint 逐 bit 一致；RNG digest 前后相等（probe 无 RNG 消耗）。
- 16 个源 run 的 selected_epoch（KR: 30/47/41/24；ML: 7/8/8/12）均不在 common epochs，selected 全部单独 probe（selected_coincide=0），复用路径未触发——已在 run report 记录。
- 已知架构事实：MDMTRec gate 输入 detach，部分模块（如 shared_scalars、KR epoch49 的 star_mlp dead-ReLU）出现零梯度，如实记录，非实现错误。
- 无 runtime failure。reviewer 修复前的 seed2022 旧产物隔离于 `_s7_review_old/`，未进入聚合。

## B. Absolute performance trajectory

**MovieLens（F arm）——单调 late-stage deterioration，4/4 seeds：**

| checkpoint | val target_domain AUC | val target_domain LogLoss |
|---|---|---|
| epoch_9 | 0.7397 | 0.548 |
| epoch_19 | 0.7359 | — |
| epoch_29 | 0.7292 | — |
| epoch_39 | 0.7211 | — |
| epoch_49 | 0.7130 | 0.673 |

early(9/19)→late(39/49)：AUC −0.0208（4/4 同号，robust），LogLoss +0.125（4/4，robust）。macro LogLoss 从 ~0.58 膨胀至 0.83–0.91——典型的 dominant-domain 过拟合 regime。注意：validation-selected epoch 为 7/8/8/12，即 selected 模型本质上是 early 模型，epoch39/49 已远超最优点。

**KuaiRand（F arm）——单调改善至 epoch49，4/4 seeds：**
val target_domain AUC 0.6842→0.7131（+0.0191，4/4 robust），LogLoss −0.0089（4/4 robust）。selected epoch 24–47，无 late deterioration。

两数据集性能轨迹方向相反，与 S6 观察一致，构成后续对照的基础。

## C. Gradient norm dynamics

- MovieLens：全部 6 个 cell 的 shared-gradient norm 晚期显著增长（early→late 全部正 diff；cells 0/1/2/3 为 4/4 seeds，cells 4/5 为 3/4）。极端例：2023 的 cell2（t0,d2）norm 从 0.015 增至 0.218（14.6×）。Σ_i w_i·N_i 4/4 seeds 增长（+0.029 ~ +0.176）。
- KuaiRand：target 域 cells 1/4 的 norm 晚期**下降**（3/4），小域 cells 0/3 与大梯度 cell 2/5 增长（4/4）。两数据集 norm 结构变化方向不同。
- MovieLens 的 norm 增长与过拟合 sharpening 同时发生；norm 大不等于有害，但它是 U_k 尺度变化的重要混杂（见 L 节）。

## D. Domain×Task cosine matrices

- 稳定结构（全部 seeds、全部 checkpoints）：**same-domain cross-task cosine 显著高于一切 cross-domain pair**。KuaiRand：0.67–0.80 vs cross-domain ≈ −0.07~+0.58（均值近 0.1–0.2，符号随 checkpoint 波动）；MovieLens：0.10–0.45 vs cross-domain ≈ −0.07~+0.29。
- 即梯度相似性按 **domain** 组织而非按 task 组织——这是 MDMT 二维结构的直接证据，task-level 聚合会抹掉该结构。
- cross-domain cosine 无稳定 stage 趋势（符号在 epoch 间翻转），不支持"跨域冲突随训练单调恶化"的 simple story。

## E. Sample-gradient alignment

A_{d,t}=cos(g_{d,t}, g_S) 结构上由 w_{d,t}=n_d/(2B) 主导：大域 cell 与 g_S 高度对齐，小域 cell 对齐度低但不稳定。MovieLens 中 d2（share 0.8）两个 cell 主导 g_S 方向；KuaiRand 中 d1（share 0.869）主导。无超预期的 stage 反转现象。（逐 cell 数值见 T4。）

## F. Validation-gradient utility（S7 核心）

**MovieLens：U_k 全面转负且与性能恶化同步。**
- 6 个 cell 的 early→late U_k diff 全部为负（均值 −0.008 ~ −0.012；cells 2/3/4/5 为 3/4 或 4/4 seeds 同号；cell4 4/4 robust）。
- epoch49 快照：cell2（target t0）4/4 seeds 为负；cell5（target t1）3/4 为负；2022 全部 6 个 cell 为负。
- 时序：U_k 从 epoch19–29 开始转负并单调加深（min U_k：e9 −0.0014 → e29 −0.0123 → e39 −0.0224 → e49 −0.0256），与 val target AUC 从 epoch9 起的单调下降时间对应。

**KuaiRand：U_k 在 epoch49 也转负，但无性能后果。**
- early→late 均值层面 target cells 1/4 的 U_k 仍为正（0.0186→0.0059 / 0.0137→0.0058），但 **epoch49 快照上 cells 1/4 在 3/4 seeds 转负**（2022: −0.0086/−0.0075；2023: −0.0633/−0.0522；2024: −0.0356/−0.0299；2025 仍为正）。
- 然而 KuaiRand val target AUC 在 epoch39→49 继续上升（4/4）。**负的一阶 utility 未转化为实现的性能损害**——这是对"负 U_k ⟹ 性能损害"映射的重要反例：一阶近似基于 raw gradient step，实际更新经 Adam 预调节，且 checkpoint 间轨迹由数千步累积而成，单点 U_k 符号不是充分条件。

**结论：U_k 比 cosine conflict 具有显著更强的性能相关性（MovieLens 中符号、时序、seed 一致性全部对齐），但该映射是条件性的，不是普遍的。**

## G. Source→target harmful contribution matrices

MovieLens late stage 稳定有害的 source→target pair（T11 robust 或 4/4 同号）：

| pair | 类型 | early→late diff | seeds |
|---|---|---|---|
| 0→0 (t0d0→t0d0) | self | −0.0018 | 4/4 robust |
| 0→3, 0→5 | cross-domain cross-task | −0.0014 / −0.0011 | 4/4 robust |
| 2→3, 2→5 (t0d2→t1 cells) | 跨任务 | −0.0025 / −0.0037 | 4/4 robust |
| 2→4 | cross-domain cross-task | −0.0050 | 3/4 robust |
| 5→4 (t1d2→t1d1) | cross-domain same-task | −0.0064 | 3/4 robust |

- 最大单一有害 pair（epoch49，3/4 seeds 的 top-1/2）：**5→2，target 域内部跨任务**（same-domain cross-task），2023 达 −0.0381（占该 checkpoint 负 H 总质量的 18%）。
- 结构性观察：late-stage 有害源集中于**主导域 d2 自身的两个 cell（2 和 5）**及 cell0；受害方集中于 task1 cells（3,4,5）。即 MovieLens 晚期的核心结构是 **dominant-domain 梯度对其他 cell（尤其跨任务）的 harmful contribution 随过拟合加深而增长**。
- 模块分层（T9）：5→2 在 epoch49 的有害贡献集中于 **skip_conn（−0.0245）与 embedding（−0.0124）**——共享表示层，而非 expert 层。
- KuaiRand epoch49 的 top 有害 pair 同属"target 域内部跨任务 + self"家族（4→1、1→1、4→4），定性结构相似，但无量级相当的性能后果。

## H. Early vs middle vs late comparison

- U_k（MovieLens）：early≈0/微正 → middle 转负 → late 深度为负，单调加深，4/4 层面方向一致（min U_k 单调：−0.0014 → −0.0123 → −0.0256）。
- ConflictMass 无 early/late 系统差异：MovieLens 0.0158→0.0284 但 2/2 seeds 分裂（not robust）；KuaiRand 反而下降（0.0130→0.0096，3/4）。**geometry conflict 本身不是 stage-dependent 量**。
- 性能：MovieLens 从 epoch9 即开始单调恶化（即"early"epoch9 已略过顶点；selected epoch 7–12 更早）；KuaiRand 全程改善。
- gradient norm：MovieLens 全面增长，KuaiRand 分裂（小域涨、target 域缩）。

## I. MovieLens vs KuaiRand comparison

| 维度 | MovieLens | KuaiRand |
|---|---|---|
| val target AUC 轨迹 | 单调恶化（4/4） | 单调改善（4/4） |
| target cell U_k 转负时点 | epoch19–29 起，持续加深 | 仅 epoch49 快照，3/4 seeds |
| 负 U_k 的性能后果 | 同步恶化 | 无（继续改善） |
| cell norm late 变化 | 全部增长 | target 域缩小 |
| ConflictMass | 略升（2/2，不 robust） | 下降（3/4） |
| 过拟合 regime | 严重（macro LogLoss +0.28~0.33） | 无 |
| top 有害 pair 家族 | target 域内跨任务 5→2 + 主导域外溢 2→{3,4,5} | target 域内跨任务 4→1 + self |

关键区分点：**两数据集的差别不在于"是否存在负 utility"（两者 epoch49 都有），而在于负 utility 的持续性/时序深度与是否处于过拟合 regime**。H4 的强形式（"KuaiRand 不出现相同 pattern"）被部分证伪；其弱形式（"KuaiRand 不出现与性能损害同步的持续 harmful regime"）成立。

## J. Seed consistency

- 4/4 同号：MovieLens 性能恶化（AUC/LogLoss）、norm 增长（cells 0–3）、H pairs 0→0/0→3/0→5/2→3/2→5、U_k cell4 diff；KuaiRand 性能改善。
- 3/4：MovieLens U_k cells 2/3/5、H pairs 2→4/5→4、epoch49 KuaiRand target-cell 负 U_k。
- 2/2（不一致）：ConflictMass 两数据集、KuaiRand U_k cells 1/4 的 early→late diff。
- 无单 seed 驱动的主要结论；所有核心结论均 ≥3/4 seeds 支持。

## K. Counterexamples

1. **KuaiRand epoch49 负 U_k 无性能后果**（3/4 seeds target cells U_k<0 而 AUC 继续上升）——直接否定"负 U_k 是性能损害的充分条件"。
2. **ConflictMass 与性能脱钩**：KuaiRand conflict 下降、MovieLens conflict 上升但 seed 分裂；geometry conflict 不具备性能解释力。
3. MovieLens 2024/2025 部分 direction-only 指标未恶化（如 5→4 在 2024 平坦）——方向性恶化是 3/4 而非 4/4 现象。
4. MovieLens 2023 epoch49 的 cell1 U_k 为 +0.033（少数正 cell），说明负 utility 非全 cell 均匀。

## L. Competing explanations

1. **过拟合症状说（最强竞争者）**：MovieLens 晚期 gradient 结构变化与严重过拟合（macro LogLoss ×1.5）完全同步；过拟合使 validation loss landscape 陡峭化、gradient norm 全面增长，harmful H 的增长可能部分只是 norm 尺度效应。T12 证据：Σw_iN_i 4/4 增长；剥离 norm 后的 direction-only 量（w_i·cos(g_i,g_k^val)）对关键 pair 仍在 3/4 seeds 恶化（5→4 在 2023/2025 符号翻转、2022 加深）——**现象不能仅由 norm 变化解释，但 norm 增长贡献了相当部分**。
2. **Adam 预调节 / trajectory 说**：U_k 是 raw-gradient 一阶近似；KuaiRand 反例表明实际更新（Adam 归一化 + 长期轨迹）可以吸收负 utility 而无实现损害。S6 的 DH 残余效应是否经由此通道，S7-A 无法区分——这正是 S7-B same-θ continuation 要回答的。
3. **小域困难说**：norm 最大的是小域 cell（ML 的 d1 cells 1/4，share 仅 1.6%），但按 w_i 加权后其有害贡献微小；有害源实际是主导域。"小域=问题源"不被支持。
4. **task-level 说**：cosine 结构按 domain 组织（D 节），有害 pair 混合 same-domain cross-task 与 cross-domain；task-level 聚合无法定位 5→2（target 域内跨任务）这一最大有害 pair。Domain×Task 分解提供了额外信息。

## M. Evidence-strength table

| # | 问题 | 裁决 | 强度 |
|---|---|---|---|
| 1 | Sample 下存在 MDMT-specific harmful gradient interaction？ | 存在，但条件化（MovieLens late-stage / 过拟合 regime） | **STRONG**（4/4 seeds、多 pair 可定位、同步性能恶化） |
| 2 | 是否只有普通 conflict 而无性能相关？ | 否。cosine conflict 确无性能相关性；但 U_k 在 MovieLens 有符号+时序+seed 三重对齐 | **MEDIUM**（KuaiRand 反例限制其充分性） |
| 3 | Domain×Task 优于 task-only？ | 是（结构按 domain 组织；最大有害 pair 5→2 只能在该分解下定位） | **MEDIUM** |
| 4 | 稳定 stage dependence？ | MovieLens U_k/性能：early→late 单调、4/4 | **STRONG**（限 MovieLens；ConflictMass 无 stage 依赖性） |
| 5 | MovieLens late DH 能被 gradient dynamics 解释？ | 一致但未证明；过拟合混杂未分离 | **WEAK-MEDIUM**（需 S7-B） |
| 6 | KuaiRand 为何无对应结果？ | 无过拟合 regime；负 utility 仅出现于 epoch49 快照且无实现后果 | **MEDIUM** |
| 7 | 支持进入 same-θ causal continuation？ | 是 | **STRONG** |
| 8 | 已足够支持设计新算法？ | 否 | **STRONG** |
| 9 | 缺失的最关键证据 | same-θ 下 TARGET-current 是否立即改变 harmful 结构（区分 current-gradient 作用 vs trajectory/过拟合症状）；norm vs direction 的定量分解；Adam 预调节通道的作用 | — |

## N. GO / STOP decision（S7-A → S7-B）

逐条核对 §10 GO 条件：

1. gradient statistic late 明显强于 early：**满足**（U_k；ConflictMass 不满足但 U_k 是主指标）。
2. ≥3/4 seeds 同方向：**满足**（3/4–4/4）。
3. 与 validation AUC/LogLoss 恶化时间对应：**满足**（MovieLens 单调同步）。
4. 可定位稳定 source→target relation：**满足**（5→2、2→{3,4,5}、0→{0,3,5} 等，4/4 或 3/4 robust）。
5. KuaiRand 不完全复制：**满足（弱形式）**——无持续 harmful regime、无性能后果；但 epoch49 快照负 U_k 存在，需在 S7-B 中作为 control 继续使用。
6. 不能仅由 norm 变化解释：**部分满足**——direction-only 指标 3/4 seeds 仍恶化，但 norm 增长是实质混杂，S7-B 必须控制。

STOP 条件核查：无一条完全触发（最接近的是 #3"两数据集 pattern 相似"，但时序深度与性能后果的区分度成立）。

**S7-A 判定：GO 进入 S7-B（Same-θ Early/Late Causal Continuation）。**

---

# §20 主 Agent 最终裁决

## **B. CONDITIONAL GRADIENT PHENOMENON**

发现了一个 seed 稳定（≥3/4）、stage 依赖（early→late 单调）、性能关联（MovieLens 内同步）、MDMT 特异（Domain×Task 结构、可定位 source→target pair）的 harmful-gradient 现象；但它**条件化于 MovieLens late-stage 过拟合 regime**，不与 KuaiRand 共享其性能后果，且与 gradient norm 增长/过拟合 sharpening 部分混杂。

按裁决条款：**允许进入 same-θ causal diagnosis（S7-B）；不允许直接开发新方法。**

## S7-B 必须回答的问题（继承 grad.md §15）

1. 相同 θ（early anchor epoch9 / late anchor epoch39）下，TARGET-current 是否**立即**改变 harmful gradient structure（5→2、2→{3,4,5} 的 H 与 U_k）？
2. 该改善是否 late >> early？
3. gradient utility 改善是否先于 validation 改善？
4. KuaiRand 同 stage 对照是否缺少该 early/late contrast？
5. 关键判别：若 same-θ 后差异基本消失 → S6 late DH residual 是 trajectory/over-training 效应，停止 gradient-method 主线；若稳定存在 → 才允许讨论 gradient correction。

## 附加声明

- 本阶段未修改任何 S1–S6 文件；S7 全部产物为新增文件。
- `_s7_review_old/` 为 reviewer 保留的修复前对照证据，可择机清理。
- 所有 U_k/H 结论均为局部一阶近似；未做任何因果声称；MovieLens 现象不得推广为 universal MDMT mechanism。
