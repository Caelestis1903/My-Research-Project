# Stage S8 分析报告 — Optimizer/Trajectory Final Kill Test

阶段性质：final kill test / falsification。本报告由 main agent 在 runner 完成全部 8 个 base run 与 24 个 continuation run 后撰写。
数据基础：`experiments/stage_S8_outputs/`（`stage_S8_best_auc_table.csv` 288 行、`stage_S8_trajectory.csv` 36,936 行、`stage_S8_effective_update.csv` 1,116 行、`stage_S8_base_validity.csv` 8 行）、`stage_S8_summary_manifest.json`、`stage_S8_run_report.md`；实现规格 `experiments/stage_S8_design.md`；reviewer 报告 `experiments/reports/S8_REVIEW.md`。

术语与口径：branch = F/T/DH（S6 语义逐字复用：F=FULL/FULL，T=TARGET/TARGET，DH=FULL history/TARGET current；private 参数恒用 FULL loss 更新）。macro AUC = 六 cell validation AUC 无权均值（全项目 checkpoint selection 同口径）。`delta_best = continuation_best_macro_auc − f_original_best_auc`，其中 `f_original_best_auc` = 原 S6/S6ML canonical F run 的 `selected_validation_macro_auc`（冻结参照）。primary horizon = h_2ep（ML 10 steps / KR 98 steps）。practical threshold 0.003、strong 0.005、≥3/4 同向（项目惯例，预注册于设计 §11）。

---

## A. Execution validity

**Phase 1（base 重训，8/8 V-BASE ok）**：8 个 S8-BASE F-arm 重训与原 S6/S6ML canonical F run：train batch fingerprint 全部逐位一致（ML 250/250、KR 2450/2450）；逐 epoch validation macro AUC 最大差 = 0.0；`selected_epoch` 全部一致（ML 7/8/8/12、KR 30/47/41/24）；`initial_parameter_state_hash` 全部一致。三个 full-state dump（selected/19/39）逐 run 存在并记录 sha256。

**Phase 2（continuation，24/24 validate_run ok）**：
- V0：24/24 的 anchor 点 macro AUC 与源 `metrics_epoch.csv` 差 ≤ 2.2e-16（位级）；F 分支 step-1 `L_full` diff = 0.0、delta rel diff ≤ 7.5e-8（阈值 1e-3）。
- V3（same-state）：24/24 三分支在 step-0 的全部参数 + 双 optimizer 全部 state（**含 private m/v**，S8 相对 S7-B 的关键修复）两两逐位相等。
- V4（batch replay）：24/24 的全部 main batch 与 arch batch（含 ML 跨 arch iterator 代边界）fingerprint 与 S8-BASE 逐位一致。
- V5（状态恢复完整性）：shared step == bpe×(anchor+1)、arch step == anchor+1、private state 100% 覆盖，24/24 通过。
- V2（轨迹漂移）：24/24 `v2_review_flagged=False`；F 分支逐步 `L_full` diff 实测为 0.0（本机 CUDA 确定性成立），远好于 S7-B 的 0.17——**S7-B 的 fresh-Adam private 近似限制已被消除**。

**执行期偏差的如实记录**（均不改变科学口径）：
1. orchestrator `_launch_base` 的目录预创建 bug 导致 2 次秒级失败启动（无数据产生，日志留存于 `stage_S8_failed_attempts/`）；runner 用 helper 绕行，S8 源文件未改。
2. KuaiRand manifest 的 `dataset` 标签为 `KuaiRand-Pure`，`validate_base_run` 只规范化 `-1m`；runner 在训练完成后、校验前将该标签改写为 `kuairand`（数据标签修复，非代码/算法修改），改写后 V-BASE ok 且与 canonical 0 差。
3. 自动聚合器 `_all_base_run_dirs()` 少探一级目录导致 `base_validity.csv` 需人工转录填充；`summary_manifest.json` 的 `base_run_count` 由 0 更正为 8。聚合 CSV 的 continuation 部分由脚本正常产出并经本报告独立重算核对。
4. reviewer 的两个验证性 continuation 产物保留在 `experiments/_s8_review_cont_ml2022*/`，在正式 raw results 之外，未进入聚合。

**结论：F/T/DH 在 branch point 前完全 identical、branch point 后仅目标操纵不同，三分支共享完全相同的 continuation batch 序列——grad.md §4 的核心要求达成。**

## B. Original baseline best performance（冻结参照）

| dataset | seed | selected_epoch | F original best macro AUC |
|---|---|---:|---:|
| movielens | 2022 | 7 | 0.7313917 |
| movielens | 2023 | 8 | 0.7293003 |
| movielens | 2024 | 8 | 0.7319634 |
| movielens | 2025 | 12 | 0.7306116 |
| kuairand | 2022 | 30 | 0.6823676 |
| kuairand | 2023 | 47 | 0.6945312 |
| kuairand | 2024 | 41 | 0.6924488 |
| kuairand | 2025 | 24 | 0.6829632 |

## C. Selected-anchor continuation（最重要部分）

**MovieLens：null。** h_2ep 的 `delta_best`：F −0.000206、T −0.000144、DH −0.000144（4-seed mean）。paired contrast：T−F = +0.000062（3/4 正）、DH−F = +0.000062（3/4 正）——比 practical threshold 0.003 小约 50 倍，且 T 与 DH 数值几乎逐位相同（在 selected anchor 上两分支轨迹尚未分化）。final-step contrast 反而为负（T−F −0.0018，0/4 正）。唯一的 `new_best=True` 是 seed2023（+4.1e-4），且 F 分支同 seed 同样为 True（+4.2e-4）——轨迹噪声水平的微小上翘，不是分支效应。

**KuaiRand：null。** `delta_best`：F +0.000619、T +0.000174、DH +0.000687；paired T−F −0.000445（1/4 正）、DH−F +0.000068（2/4 正）。KR 2025 三分支同现 +3.4e-3 的 `new_best`（F 亦为 True），反映 KuaiRand selected epoch 后继续训练仍在改善的数据集属性，与分支无关。

**裁决：在模型正常应该停止训练的位置，T/DH 相对 F 无任何实际量级的好处。grad.md §6.A 的"现实性测试"失败。**

## D. Middle-anchor continuation

MovieLens `delta_best`：F −0.008799、T −0.007670、DH −0.006777——**所有分支、所有 seed（12/12）的 continuation best 都低于 F original best**。paired DH−F = +0.002023（4/4 正，sd 0.000811），T−F = +0.001130（2/4 正）。DH−F 达 4/4 同向但均值低于 practical threshold；且它是"少恶化"，不是"新 optimum"。

KuaiRand：paired T−F −0.005319（0/4 正）、DH−F −0.001373（1/4 正）——T 在 KuaiRand middle anchor 上有实际量级的损害。

## E. Late-anchor continuation

MovieLens `delta_best`：F −0.027948、T −0.026916、DH −0.021122——**12/12 全部大幅为负**。即使表现最好的 DH continuation（2025：0.72233）也比该 seed 的 F original best（0.73061）低 0.0083。

paired contrasts（h_2ep，continuation best）：T−F +0.001032（2/4）、DH−F **+0.006826（4/4，sd 0.007318）**；final-step：T−F +0.002095（4/4）、DH−F **+0.010732（4/4，sd 0.010259）**。

**但 magnitude 高度 seed 偏斜**：final-step DH−F 逐 seed = +0.00278 / +0.00216 / +0.01438 / +0.02361——两个 seed（2024/2025）贡献均值的 ~89%，2022/2023 仅 +0.002 量级。在 **target-domain** 口径下 rescue 更弱：final-step DH−F = +0.002566（3/4），best 口径仅 +0.000441（3/4）——S6 报告的 late target AUC 优势（+0.0105/+0.0081）在完整 optimizer-state、same-batch continuation 下**未能在 target-domain 指标上以同等量级复现**；macro 口径的 rescue 很大部分是两个 seed 的 non-target/macro 波动。

KuaiRand late：paired T−F（best）−0.003744（0/4 正）、final-step T−F −0.010239（0/4 正）——T 有明确损害；DH−F ≈ −0.0001~−0.0009（1/4 正）为 null。**KuaiRand 继续保持干净的对照角色。**

## F. Best-AUC comparison（primary endpoint，禁止视角之外的总表）

grad.md §12 主表口径（h_2ep；完整 288 行见 `stage_S8_best_auc_table.csv`）：

| dataset | anchor | branch | delta_best mean | sd | new_best/4 |
|---|---|---|---:|---:|---:|
| movielens | selected | F/T/DH | −0.000206 / −0.000144 / −0.000144 | ~0.0004 | 1 / 1 / 1 |
| movielens | middle | F/T/DH | −0.008799 / −0.007670 / −0.006777 | ~0.005 | 0 / 0 / 0 |
| movielens | late | F/T/DH | −0.027948 / −0.026916 / −0.021122 | ~0.006 | 0 / 0 / 0 |
| kuairand | selected | F/T/DH | +0.000619 / +0.000174 / +0.000687 | ~0.0019 | 2 / 1 / 2 |
| kuairand | middle | F/T/DH | −0.004478 / −0.009797 / −0.005851 | ~0.006 | 1 / 0 / 1 |
| kuairand | late | F/T/DH | −0.000663 / −0.004407 / −0.000806 | ~0.003 | 2 / 0 / 2 |

MovieLens 上 T/DH 在 **任何 anchor、任何 seed** 都没有形成超过 F original best 的 new best（唯一 +4e-4 的 seed2023 三分支共享，F 相同）。KuaiRand 上的少量 `new_best=True` 同样在 F 分支出现（数据集属性，非分支效应）。

**primary endpoint 裁决：ΔBestAUC 的 GO 条件（T/DH 稳定超过 F original best、≥3/4 seeds 同向）在 24 个 (dataset × anchor) 组合的任何一个上都不成立。**

## G. Raw gradient vs effective Adam update

设计 §10.2 的核心问题——"S7-B 中 raw gradient 改善而 validation 不改善，是否因为 Adam state 改变了真正执行的 update geometry"——得到直接的定量回答：

1. **Adam 预调节系统性地吸收 raw-gradient 方向分歧。** 同一 θ 上（DH 分支，pre-step state 完全相同）：MovieLens middle 的 raw cos(g_F,g_T) = 0.759，而 effective-update cos(Δ_F,Δ_T) = 0.846；late：0.863 → 0.909。raw cosine 把实际执行更新的几何分歧**高估**了一档。这确认 S7-B 用 raw-gradient 口径观察到的"harmful structure 即时改变"不能线性映射到实际参数更新。
2. **T 分支的 applied update norm 系统性更小**（ML late：T 1.14 vs F 1.26；KR late：T 0.90 vs F 0.95）——target-only 梯度在 Adam 归一化后产生更小的 shared 步长，这是 T 轨迹分化的主要 update-level 特征。
3. **eff_align（applied update 与 validation gradient 的 cosine）全面微弱（|·|≤0.02）且大多接近零/为负**；唯一可记录的信号是 ML middle 的 DH 分支 `eff_align_full` 在 2/4 seeds（2022/2023 +0.016）为正而 F 为负——但在 rescue 最大的 late 2024/2025 上 DH 的 eff_align **并不优于 F**（−0.0003/−0.0042 vs F −0.0019/−0.0049）。**effective-update alignment 与 validation rescue 在空间（哪个 seed）上不对应。**

裁决：raw 与 effective 两个口径的 alignment 统计都与最终 validation 结果脱钩——与 S7-B 的发现一致并延伸到长程窗口。

## H. Current / History / θ / Validation 因果链证据

| 链节 | 证据 | 状态 |
|---|---|---|
| Current → History | T 提交 g_T 进 moments（构造保证）；T 的 delta norm 系统性更小；分支间 m/v 轨迹分化 | 成立（构造性） |
| History → EffectiveUpdate | 同 θ 同 state 下 effective cosine 显著高于 raw cosine；Adam state 改变执行几何 | 成立（实测，G 节） |
| EffectiveUpdate → θ | 三分支 θ 轨迹在窗口内持续分化（`theta_distance_to_F_shared` 单调增长） | 成立（实测） |
| θ → Validation（best 口径） | 仅 late anchor、仅 2/4 seeds 出现窗口内 macro 上升；target-domain 口径仅 3/4 且均值 +0.0004~+0.0026；**任何条件下不超 F original best** | **不满足预注册标准** |

禁止把相关性写成因果：late anchor 的 macro rescue 与 DH 轨迹在时间上的先后关系成立，但 (a) 它不是 target-domain 口径的稳定现象，(b) 它达不到"正常协议会被选择的模型性能"这一唯一有方法学意义的门槛。

## I. MovieLens vs KuaiRand

| 维度 | MovieLens | KuaiRand |
|---|---|---|
| selected anchor T/DH−F | null（+6e-5） | null（−4e-4 / +7e-5） |
| middle anchor DH−F（best） | +0.0020（4/4） | −0.0014（1/4） |
| late anchor DH−F（best / final） | +0.0068 / +0.0107（4/4，2 seed 驱动） | −0.0001 / −0.0002（1/4） |
| late anchor T−F（best / final） | +0.0010 / +0.0021 | **−0.0037 / −0.0102（0/4，损害）** |
| delta_best（任何分支任何 anchor） | 全部 ≤ +4e-4 | 全部 ≤ +3.4e-3 且 F 同现 |

KuaiRand 不存在过拟合 regime（S7-A），S8 中也不存在任何 rescue 对象；T 在 KuaiRand 的 non-target 损害与 S6 完全一致地复现。两数据集的对照结构（ML late 有 rescue 对象、KR 没有）与 S6/S7 自洽。

## J. Seed consistency

- 4/4 同向：ML middle/late 的 DH−F（best 口径）；ML late final-step DH−F 与 T−F；KR middle/late 的 T 损害。
- 量级 seed 偏斜：ML late macro rescue 的 ~89% 均值来自 2024/2025（+0.0144/+0.0236 vs +0.0028/+0.0022）；target-domain 口径 2025 单 seed 占主导。
- selected anchor 的全部结论为 null，无 seed 方向冲突。

## K. Counterexamples

1. **ML late 2022/2023**：同为 late anchor，DH−F 仅 +0.0012~+0.0013（best 口径）——"late rescue"在两个 seed 上接近 null，说明 rescue 不是 late anchor 的必然属性。
2. **target-domain 与 macro 口径脱钩**：S6 的 late DH 效应以 target AUC 报告；S8 的 macro rescue（+0.0107）在 target-domain 口径缩到 +0.0026（final）/+0.0004（best）——rescue 的主要载体不是 S6 效应发生的指标。
3. **ML selected seed2023 的 +4e-4 new_best 在 F 分支同样出现**——任何"continuation best 超过 original best"的孤立案例都不是分支效应。
4. **DH middle 的 eff_align_full 优势 seed（2022/2023）与 late macro rescue 优势 seed（2024/2025）不相交**——optimizer-level 统计与 validation 结果无空间对应。

## L. Competing explanations

1. **ordinary overfitting / over-training（最强，且被本阶段数据直接支持）**：DH/T 的优势严格单调于 anchor 的 lateness（selected +0.00006 → middle +0.0020 → late +0.0068/+0.0107），且在无过拟合的 KuaiRand 完全缺失。grad.md §12 预注册的判读规则（late ≫ middle ≫ selected≈0 ⟹ 优先解释为 overfitting rescue / trajectory stabilization）逐字适用。
2. **optimizer-history 机制**：History→EffectiveUpdate 链节真实存在（G 节），但它产生的 trajectory 差异只转化为"少恶化"，不转化为 best-performance improvement。作为机制存在，作为方法价值被否决。
3. **regularization / trajectory stabilization**：T 的更小 update norm（G 节）等价于一种隐式步长收缩/正则化，与"降低过拟合斜率"的轨迹形态一致（late anchor F 的 best_step 集中在 1–4，DH 推到 5–10）。
4. **learning-rate schedule**：本项目 lr 恒定，不存在 schedule 混杂；排除。
5. **checkpoint-selection artifact**：primary endpoint 用逐步 best 而非固定 epoch，已排除固定 checkpoint 比较假象；且 F original best 参照是逐 seed 的 validation-selected 冻结值。排除。

## M. GO / STOP table

### GO 条件（grad.md §14）

| # | 条件 | 结果 |
|---|---|---|
| 1 | T/DH 相对 F original best 稳定提升，≥3/4 seeds 同向 | **失败**：24 个组合无一成立；ML 全部 anchor 的 delta_best ≤ +4e-4 且 T/DH new_best 0/4（middle/late） |
| 2 | selected/middle anchor 也有意义 benefit | **失败**：selected null（~6e-5）；middle 仅"少恶化"，best 仍全部低于 F original best |
| 3 | effective update 提供比 raw gradient 更直接稳定的机制证据 | **失败**：effective 口径修正了 raw 的高估（G.1），但 eff_align 与 validation rescue 无空间对应（K.4） |
| 4 | 效应非单 seed 驱动 | **失败**：late macro rescue ~89% 来自 2/4 seeds；target 口径更偏 |
| 5 | KuaiRand 无明显性能损害 | **失败**：T 在 KR middle/late 有实际量级损害（final −0.0102，0/4 正） |
| 6 | trajectory difference 可重复且时序一致 | 部分成立（DH−F 4/4 同向于 middle/late），但形态为"少恶化"，与 hypothesis 要求的"新的更高 optimum"不符 |

### STOP 条件（grad.md §15）

| # | 条件 | 触发 |
|---|---|---|
| STOP-1 | T/DH 只改善 late overfitted checkpoint，BestAUC_T/DH ≤ BestAUC_F,original | **触发**（ML late DH best −0.021 vs F original best；rescue 真实存在但量级差一个数量级） |
| STOP-2 | selected continuation 无稳定收益 + late rescue 明显 | **触发**（selected ~6e-5 vs late +0.0107，相差两个数量级） |
| STOP-3 | 完整 state + 长程后 S6 late residual 不再稳定复现 | **部分触发**：target-domain 口径的 S6 residual（+0.0105/+0.0081）未以同等量级复现（S8 best 口径 +0.0004）；macro 口径的残余由 2 seeds 承载 |
| STOP-4 | raw/effective 统计改善无法对应 validation benefit | **触发**（G.3/K.4：eff_align 与 rescue seed 不对应） |
| STOP-5 | 结果主要由某一个 seed 驱动 | **部分触发**（late macro rescue 由 2024/2025 两个 seed 主导；2025 单 seed 在 target 口径主导） |

**6 项 GO 条件 5 项失败 1 项部分；5 项 STOP 条件 2 项完全触发、2 项部分触发、1 项触发。**

## N. Final verdict

# **B. LATE OVERFITTING RESCUE ONLY**

S6 的 DH/T late-stage benefit 在完整 optimizer state、严格 same-state、same-batch、长程（2-epoch）continuation 下表现为且仅表现为：**对已严重过拟合的 late checkpoint 的部分救援**（macro 口径 4/4 同向但由 2/4 seeds 承载；target-domain 口径弱且不稳定）。在模型正常会被选择的 selected checkpoint 上，操纵没有任何实际量级的效果（~6e-5，practical threshold 的 1/50）；continuation 的 best validation performance 在任何 dataset × anchor × branch × seed 组合上都没有稳定超过 F 原始 best。**该现象不能提高正常训练协议下最终会被选择的模型性能，不具备方法化价值。**

按 grad.md §20 的纪律：即使 late macro rescue 达到 +0.0107、即使 DH−F 在 middle/late 4/4 同向，结论不变。

## 执行链记录

designer（k3-256k max；确认 S6 无 private moments → Phase 1 重训方案；V-BASE/V0–V5 分层 validity）→ coder（MiniMax M3；5 新文件；ML seed2022 base 完整训练通过）→ reviewer（k3-256k high；修复 14 项含跨 epoch 崩溃、arch 语义错误、V0/V2 静默失效、f_original 参照源错误；Ready: YES）→ runner（MiniMax M3；严格单进程串行；8/8 base + 24/24 continuation，全部硬门位级通过）→ main（本报告）。Phase 3（5-epoch 扩展）未启动——primary endpoint 已决定性失败，扩展窗口不可能逆转"不超 F original best"的裁决（所需弥补的差距 −0.021 比已观察 rescue 大一个数量级），启动它只会稀释结论。

## S8 STOP 后的研究状态（按 grad.md §16/§17，仅记录方向，不在本阶段执行）

- **禁止**：gradient correction、PCGrad-like、stage-aware U weighting、harmful-pair masking、adaptive source suppression、HT/DH 新组合等一切 S9 gradient method 设计。
- **下一研究问题**（grad.md §17 预定义）：Sample objective 在 extreme domain imbalance 下，是否存在 selected-checkpoint minority-domain under-optimization / joint-training performance gap（`Gap_{d,t} = Perf_joint − Perf_single/oracle`，ratio sweep 1:1:1→50:1:1，selected checkpoint，4 seeds，先证 performance problem 再做机制诊断）。
- **重要的既有背景**：S1 已在 Sample + SD-reference 下 STOP 普遍 residual NT（0/72 稳定 practical NT），S2 的 Sample−SD 亦无 ≤−0.003 组——即目前**没有已知的 selected-checkpoint 性能缺口**等待被解决。下一阶段若启动，必须先以新协议（极端 ratio、per-domain 细粒度）重新证明 gap 存在，不得预设 small domain = difficult domain。

## 产物清单

- 设计：`experiments/stage_S8_design.md`；审查：`experiments/reports/S8_REVIEW.md`
- 代码：`experiments/stage_S8_{protocol,base_run,run,orchestrator,report}.py`（新增；S1–S7 文件零修改）
- 原始：`experiments/stage_S8_base_raw_results/`（8 runs）、`experiments/stage_S8_raw_results/`（24 runs，含逐 run replay_audit.json）
- 聚合：`experiments/stage_S8_outputs/stage_S8_{best_auc_table,trajectory,effective_update,base_validity}.csv`、`stage_S8_summary_manifest.json`、`stage_S8_run_report.md`
- runner 辅助与失败尝试存档：`experiments/_runner_s8_*.py`、`experiments/stage_S8_runner_*.json`、`experiments/stage_S8_failed_attempts/`
