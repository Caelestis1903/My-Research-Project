# Stage S8 Reviewer 审查报告（实现审查 + 修复 + 验证）

日期：2026-10-08。审查对象：Coder 交付的 S8 全部实现（`stage_S8_protocol.py` / `stage_S8_base_run.py` / `stage_S8_run.py` / `stage_S8_orchestrator.py` / `stage_S8_report.py` + ML seed2022 基座试跑产物）。审查基准：`experiments/stage_S8_design.md`（设计冻结）、`promt/grad.md` §18 Reviewer 清单。

## 一、总结论

- Coder 初版实现中，**protocol / base_run / orchestrator 三个文件审查通过，无需修改**；ML seed2022 基座试跑真实有效（与 S6ML canonical 的 250/250 train + 50/50 arch 指纹逐位一致、逐 epoch macro AUC diff=0.0、selected_epoch=7 一致、init hash 一致）。
- **`stage_S8_run.py` 初版的正式路径存在阻断级 bug**（跨 epoch 必崩、arch 更新顺序与公式错、V0/V2 静默失效等 10 项），**`stage_S8_report.py` 有 3 处校验门空转**。以上已在授权范围内全部修复。
- 修复后验证：V-BASE 校验 ok 且逐 epoch 门真实执行（50 epoch，max diff=0.0）；三种 smoke 全过；**两次真实正式延续（ML seed2022 selected/7 与 middle/19，h=2ep）均 exit 0 且 `validate_run` ok=true**，全部硬门数值达到位级一致（详见 §四）。
- **裁决建议：S8 实现已具备正式跑（Runner `--phase cont`）条件。** 正式矩阵启动与否由主 agent 裁决。注意：端到端验证只覆盖了 ML（基座试跑 + 两个正式延续）；KR（bpe=49）基座与延续均未实跑，逻辑路径相同但单 run 时长约一个数量级更高。Runner 跑 KR 首个 task 时 orchestrator 会在 task 内自动 validate，fail-closed。

## 二、grad.md §18 Reviewer 清单逐项证据

| # | 检查项 | 结论 | 证据 |
|---|--------|------|------|
| 1 | branch 是否真正 same-state | 通过 | `branch_state_full` 深拷贝 + 逐参数 `torch.equal` 硬断言 + `v3_per_pair` 三对（F/T/DH）optimizer state 逐张量比对；smoke-restore `v3_equal=True`；正式 run audit `v3_per_pair.*.all_equal=true`，`validate_run` 强制该门 |
| 2 | private moments 是否恢复 | 通过 | full-state dump 含 main/arch optimizer 完整 state_dict；`private_state_coverage` 硬门（ML=全部 private 参数有非空 m/v/step）；smoke-restore `private_coverage=True` |
| 3 | scheduler/step counter 是否一致 | 通过 | `shared_step_observed==expected=bpe*(anchor+1)`（ML selected7：40/40）、`arch_step==anchor+1` 硬断言；本实现无 lr scheduler（Adam 恒定 lr，与 S1 一致） |
| 4 | batch replay 是否一致 | 修复后通过 | 初版正式路径跨 epoch 必崩（详见 §三.1）；修复为流式逐 epoch 重演（每 epoch 3 draws：main iter 创建 1 + 首个 next 1 + train_records iter 1），V4 硬门逐步对拍。实证：selected7 run 覆盖 epoch 8/9（10/10 main 指纹 + 2/2 arch 指纹全 match）；middle19 run 覆盖 epoch 20/21，其中 epoch 20 跨 arch iterator "代"边界（20//5=4 新代），指纹仍全 match |
| 5 | best-AUC 比较是否只与 late F 比 | 修复后通过 | 初版 `f_original_*` 错取自 S8-BASE retrain manifest；修复为设计 §9.11 要求的**原 S6 canonical manifest**（`proto.source_run_dir`），运行时硬核对（selected_epoch 相等、AUC 差 ≤1e-6、anchor_class=selected 时 pin==canonical selected_epoch），report 的 `validate_run` 与 `_best_table_rows` 也改读 canonical 并复核 |
| 6 | data leakage | 通过 | 延续只读 anchor 时刻的 full-state dump；未来 batch 来自 S8-BASE batch_trace 的指纹对拍（数据即源轨迹本身，非泄漏）；val 梯度/指标仅作诊断记录，从不安装进 optimizer |
| 7 | effective Adam update 公式 vs 实际 optimizer | 通过 | `adam_deltas` 复算 torch Adam 公式，`assert_vector_match` 对 main（shared+private）与 arch 步均做 V1 硬门；实测 F/T applied residual ≈1.3e-6（fp32 量级），arch 步同门 |

## 三、发现并修复的 bug 清单（全部已修复并验证）

### `experiments/stage_S8_run.py`

1. **正式路径无法跨 epoch（阻断）**：`replay_main_batches` 从单个 iterator 连续 `next()` h_max_steps 次（ML h=10 > bpe=5，KR h=98 > bpe=49），StopIteration 必崩；且完全无 per-epoch RNG 记账。→ 重写为流式逐 epoch 重演：`setup_seed+build_model` 后预消耗 3*(anchor+1) 次 iter draw，随后每 epoch `iter(train_loader)`→bpe 个 next→epoch 末 `iter(train_records_loader)`，逐步 V4 指纹硬门（失配即 abort 并写审计）。
2. **arch 更新顺序错（阻断）**：初版在全部 main steps 跑完后才补做所有 epoch 的 arch update；源轨迹是 epoch e 末 arch update 影响 epoch e+1 的 main batches（arch 参数参与 forward）。→ 按 epoch 交错：每 epoch 末立即做三分支 arch update。
3. **arch batch 重建公式错**：初版用 `3*e` draws 且不跳代内 offset。→ 新增 `arch_batch_for_epoch`：epoch_start=(e//bpe)*bpe，`preserve_rng_state` 内预消耗 3*(epoch_start+1) draws、建 iter、先 next 再跳 offset=e%bpe 个（与 smoke-replay 已验证公式一致）。
4. **arch 指纹失配只 `pass`**：→ V4 硬门，失配 abort（`v4_arch_batch_fingerprint_mismatch_epoch_*`）。
5. **arch update 分支语义错**：初版所有分支一律安装 g_F。→ 新增 `_arch_controlled_step`，镜像 S6 `_controlled_step(kind="architecture")`：F 装 g_F；T 装 g_T；DH 装 g_F+step+覆写 θ←θ_before+Δ_T；含 None-grad fail-closed、arch Adam state 非空断言、RNG 不变硬断言、`assert_vector_match` V1 门。
6. **V0/V2 实质失效（阻断）**：`_epochs_metrics_lookup` 对 per-cell 型 metrics_epoch.csv 返回空 dict（已实证 len=0）→ step-0 AUC 门与 epoch 末 tripwire 全静默跳过；V0 的 delta 取自 batch_trace 中不存在的 `delta_observed_step_norm` 列 → NaN 恒过；step0 fail 算了但从不 abort。→ 修复：`_epochs_metrics_lookup` 按 epoch 分组 + 一致性断言；step-0 anchor AUC 门（≤1e-6）fail 即 abort；V0 step-1 用 base `gradient_step.csv`（kind=main）的 L_full/delta_observed_step_norm 内联对拍，fail 即 abort；V2 逐步 tripwire。
7. **f_original 参考源错**：→ 见 §二.5。
8. **validation 轨迹错位一步**：初版 diagnostics 里的 val metrics 是 step 前状态却标为 post_step step=k。→ 拆分：diag（pre-step，仅 train 侧 + 采样步 val grads）→ controlled step → post-step `_full_validation_metrics`+aggregate；step-k 行记录 step-k 后状态；best-tracking 基于 post-step macro AUC（主 endpoint 修正）。
9. **`eff_align_*` 四列永远 NaN**（stub）：→ 删除 stub，新增 `_validation_full_target_grads`（按 `loss_components` sample 语义：full=全体 val 样本 per-example mean-BCE 均值；target=masked 和除以全体 N）与 `_eff_alignment`；`controlled_continuation_step` 返回 fp64 Δ 向量；采样步实算 cos/dot。
10. **杂项**：best pth 文件名空格（`"{} {}.pth"`→`continuation_best_{branch}.pth`）；step 行 `rng_digest` 填 continuation_after → 改逐步 digest；F 行 `theta_distance_to_F_shared` 0.0→NaN（设计 §6.2"仅 T/DH 行"）；`_aggregate_validation` 硬编码 1/3 domain shares → 用 `bundle.domain_shares`；epoch_end_post_arch 行缺 6 个 per-cell 行且 step=-1 无 epoch 信息 → 补 9 行、step=该 epoch 最后全局步；`smoke_onestep_ok` 恒 True → 改为 fingerprint_match ∧ ¬v0_fail；删除 `compute_eff_align`/`_best_so_far` 死代码与 `theta_history` 全量快照（改为仅存最优一份 CPU state，KR h=98 下避免内存堆积）；`reconstruct_anchor_full` 的 `__wrapped__` 怪表达式 → 占位 0（调用方覆写）；新增 `anchor+h_max_epochs>49` 硬检。
11. **RNG 断言语义**：`controlled_continuation_step` 与 `_arch_controlled_step` 步边界 RNG digest 不等即 raise（S6 同强度）；新增"epoch 内逐步不变"断言（`rng_within_epoch_invariant`），iterator draw 只发生在 step/epoch 边界。

### `experiments/stage_S8_report.py`

12. **`validate_base_run` gate 2 空转**：同样按 domain/task 过滤导致空 dict 静默通过。→ 按 epoch 分组（per-cell 行 macro 重复），空 dict 本身报 reason。
13. **`validate_run`**：finite 检查含 `theta_distance_to_F_shared` 会误杀 F 行 NaN → 改为仅 T/DH 行检查；`rng_state_unchanged` 门与正确的逐 epoch RNG 消耗矛盾 → 改为 `rng_within_epoch_invariant` 门；新增 v0 anchor 门、arch 指纹全 match 门、validation 行数==3*9*(1+h_max_steps+h_max_epochs) 与 geometry 行数==3*|valgrad_steps|*288 的基数硬校验；新增 f_original canonical 复核门（含 selected anchor pin 核对）。
14. **`_best_table_rows`**：f_original 改读原 S6 canonical manifest（audit 兜底）。

## 四、修复后验证结果（全部实测）

| 验证 | 命令 | 结果 |
|------|------|------|
| V-BASE（ML seed2022） | `stage_S8_report.py --base-run-dir stage_S8_base_raw_results/movielens/seed2022/joint` | ok=true，gate 2 真实比较 50 epoch，max diff=0.0 |
| smoke-restore | `--smoke-restore` | ok=True，shared_step=40/40，private_coverage=True，v3_equal=True |
| smoke-replay | `--smoke-replay` | ok=True，5 main + 1 arch 指纹全 match |
| smoke-onestep | `--smoke-onestep` | ok=True，F/T residual≈1.3e-6，fingerprint_match=True，v0_fail=False |
| 正式延续 selected7（h=2ep，epoch 8/9） | `--anchor-class selected --out-dir _s8_review_cont_ml2022` | exit 0（120s）；`validate_run` ok=true |
| 正式延续 middle19（h=2ep，epoch 20/21，跨 arch 代边界） | `--anchor-class middle --out-dir _s8_review_cont_ml2022_mid` | exit 0；`validate_run` ok=true |

正式 run 关键数值（selected7）：

- V0 anchor macro AUC：observed==expected==0.7313916750519242，diff=0.0（位级一致）。
- V0 step-1：L_full diff=0.0，delta rel diff=3.28e-9（阈值 1e-5/1e-3）。
- V2 逐步（9 步）：L_full diff 全 0.0 —— F 分支与源轨迹位级重合。
- arch 指纹：epoch 8/9 全 match（middle19：epoch 20/21 全 match，含跨代）。
- epoch 末 post-arch F macro AUC vs base：diff ≤ 2.2e-16。
- RNG：逐步不变断言全过；step 1–5 digest 相同、step 6 起变化（epoch 边界 draw），符合预期链。
- 产物基数：steps 30 / validation 351 / geometry 5184 / best 12，与冻结 schema 精确一致；eff_align_* 采样步为实值、非采样步 NaN；F 行 theta 距离 NaN、T/DH 实值；best pth 文件名 `continuation_best_{F,T,DH}.pth`。
- f_original 链：canonical（stage_S6ML F seed2022）selected_epoch=7、AUC=0.7313916750519242，与 S8-BASE crosscheck diff=0.0，pin 核对通过。

验证产物保留在 `experiments/_s8_review_cont_ml2022/` 与 `experiments/_s8_review_cont_ml2022_mid/`（在 `stage_S8_raw_results` 之外，不会被聚合器误收）。

## 五、遗留事项 / 给主 agent 的建议

1. **正式跑就绪**：建议 Runner 可以启动 `--phase cont`（orchestrator 串行、每 task 内 validate、fail-closed）。本会话未动正式矩阵。
2. **KR 未做端到端延续试跑**：KR 路径与 ML 同一代码（bpe=49），但单 run 时长显著更高（98 步 × 3 分支 × 每步全量 validation）。若主 agent 想更稳，可让 Runner 先跑 1 个 KR selected task 观察（orchestrator 顺序本来就是 ML 先行）。
3. V2/tripwire 在两次 ML 试跑中全部 0.0（位级），说明 CUDA 确定性在这台机器上成立；若 KR 试跑出现 V0 超阈，先区分确定性差异与真 bug，再交主 agent 裁决阈值。
4. `stage_S8_run.py` 中 `validation_cell_gradients`/`shared_module_of` 等 import 未使用（沿自 Coder 初版），无功能影响，未清理。
