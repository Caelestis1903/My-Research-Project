# 05 — Codex 实现规格：M1–M8 全家族

## 1. 新目录

源码：experiments/multi_mechanism_diagnostics_20261123/
工件：artifacts/multi_mechanism_diagnostics_20261123/
报告：research_reports/multi_mechanism_diagnostics_20261123/

不得修改原始 M3oE source、D2 source 或历史报告。复用纯函数必须记录来源 blob/hash。

## 2. 公共模块

- protocol.py：effective config、seed、dataset、split。
- source_adapter.py：fixed1 canonical model/data/Adam/EMA adapter。
- parameter_registry.py：运行时 U/P/FROZEN 参数和 support。
- state_store.py：完整 state save/load/hash/readback。
- panels.py：Train calibration、V_A、V_B、pair IDs。
- ranking.py：六格 AUC/LogLoss、pair margins、correction/destruction/ties。
- virtual_adam.py：不污染 live state 的 native Adam virtual step。
- scheduler.py：family-round-robin、worker ownership、retry、resume。
- statistics.py：paired seed、CIs、Holm/BH、family/variant reports。
- reports.py：CSV/JSON/Markdown outputs。
- cli.py：audit、smoke、baseline、screen-all、confirm-all、short-all、full-all、report。

## 3. 八个机制适配器

- m1_private_gradient_shaping.py
- m2_row_task_influence.py
- m3_stage_group_interaction.py
- m4_conditional_feature_dependency.py
- m5_multilabel_augmentation.py
- m6_output_sensitivity_control.py
- m7_distributional_attribution.py
- m8_task_conflict_controller.py
- cross_family_interactions.py

每个 adapter 必须实现同一 receipt contract，但保留各自真实输入、作用对象和数学操作。不能为了统一接口而把不同机制都转成简单 loss weighting。

## 4. 核心接口

~~~python
def load_protocol(dataset: str, seed: int) -> Protocol: ...
def capture_state(runtime, parent_id: str) -> StateReceipt: ...
def restore_state(receipt: StateReceipt) -> RuntimeState: ...
def run_variant(anchor, variant_spec, controls) -> ForkReceipt: ...
def evaluate_panels(model, panel_a, panel_b) -> PanelMetrics: ...
def run_short_trajectory(anchor, variant_spec, epochs=5) -> TrajectoryReceipt: ...
def run_full_trajectory(protocol, variant_spec, epochs=50) -> FullRunReceipt: ...
def analyze_all_registered_variants(manifest, receipts) -> DecisionLedger: ...
~~~

## 5. M1 implementation

每个 variant 计算六格 g_U,c 与 natural joint g_U,J，构造对应 differentiable objective。通过 create_graph=True 计算 ∇_P L_cons。U 保持可微；backward 后 U.grad=None，再执行 native Adam step。随后按该 arm 定义执行同 batch joint update。记录 gradient geometry、P/U displacement、Adam state transitions 和 pair-margin responses。

M1-C 的 pairwise conflict energy 不能用 cell-ID permutation 或所有 cell 共用同一坐标 permutation 作为 control；必须对每个 cell 独立执行 within-tensor coordinate-shuffled control。

## 6. M2 implementation

每个 dataset/seed/anchor 从 canonical Train batch B 中抽取最多 128 rows/domain（总计最多 384 rows），按 observed joint-label support 分层；Train-only calibration pairs 独立保留，不进入 optimizer batch。未评分 row-task 权重保持 1。每个 row-task BCE gradient contribution 使用 canonical reduction scale 1/(2B)，B 为原始 batch row 数；求和须重建完整 batch gradient。

对每个 sampled row-task：
1. 按 source canonical loss reduction 计算 BCE gradient contribution；
2. 计算 full batch native Adam virtual displacement；
3. 从同一 anchor state 计算 leave-one-row-task-out virtual displacement；
4. ΔU_i,t = ΔU_full − ΔU_without(i,t)；
5. 对 Train-only calibration positive-negative margin 计算 ∇_U m；
6. s(i,t,c)=∇_U m_c · ΔU_i,t。

正分数表示 row-task 对该更新的一阶预测提高该 margin。对 task/cell score 的聚合、权重、排序和 loss mass 规则必须逐一记录。M2-B 为每个 optimizer step 分别构造 task0/task1 row microbatch，按各自 task score 排序，在相同参数状态下计算两个 task gradients 后合并并执行一次 Adam step；每个 task 的 row 在 epoch 内恰好出现一次。M2 不得退化为 loss 或 gradient norm 排序。

## 7. M3 implementation

stagewise attribution 在 E10/E25/E40 的同一冻结 panel 上重复。Group interaction 使用固定 A/B row pools（各至少 n 个不同 row），按 domain × observed joint-label strata 保持相同配额比例。A-only/B-only 各取 n 个不同 row；A/B 各取 n/2；A/A、B/B 各取 n/2 后重复一次。五组总 row slots 固定为 n，optimizer state 和 step 数一致。记录 parameter delta、logit、pair margin 和 AUC interaction residual。

## 8. M4 implementation

从 DATA_MANIFEST 的真实字段 schema 构造 feature groups。计算 domain/task conditional dependence; 执行 within-domain × observed joint-label conditional permutation；执行 task-conditioned residualization/HSIC intervention。每种操作都保留 sham control。禁止把相关性下降直接报告为因果效应。

## 9. M5 implementation

每个变换有独立 transformer、validity checker、label-pair checker 和 receipt：
- sparse field mask/dropout；
- embedding perturbation；
- within-stratum categorical substitution；
- dense perturbation within original domain/semantic bin；
- multi-view consistency training。

OOV index 不自动视为 mask。domain indicator 不变。user_id/movie_id 不任意替换。每种增强保留标签联合分布、原始 domain、feature schema、预测一致性和有效样本数。无合法字段的变体写 NOT_APPLICABLE，不阻塞其他 M5 变体。

## 10. M6 implementation

对六格 pair margin 求 Jacobian/JVP/VJP。实现：
- sensitivity-based scalar step scaling；
- worst-cell constrained trust-region projection；
- parameter-group response gating。

QP/solver 需记录约束、收敛状态和残差。对照 global norm clipping、random scale、unmodified update。实际 AUC 与预测敏感度分开报告。

## 11. M7 implementation

同一 row/task 的 score 在五 seed 间对齐。输出 empirical distribution、mean、SD、quantiles、sign stability。实现 mean−λSD robust score，并执行 leave-one-seed-out 五轮外推。留出 seed 不参与其自身 score rule fitting。M7 结果不宣称总体分布估计。

## 12. M8 implementation

实现 PCGrad reference、conflict-aware simplex weighting、common-descent projection、conflict-aware cell gating。每个 controller 输入六格梯度与冲突矩阵，输出更新向量、cell weights、projection residual 和每格实际监督质量。权重非负、总量受控，不得静默丢弃任务。

## 13. Cross-family implementation

执行 X1–X5 五组预注册组合。组合 arm 使用已有单机制 arm receipts 作为 component reference；只有 combined 和 matched control 需要新增 fork，避免重复计算。组合定义与单机制变体完全冻结。

## 14. Output contract

全局：
RUN_MANIFEST.json、VARIANT_REGISTER.csv、SEED_MANIFEST.csv、BASELINE_MANIFEST.csv、SNAPSHOT_MANIFEST.csv、ADMISSIONS.jsonl、EXITS.jsonl、FORK_ENDPOINTS.csv、CELL_METRICS.csv、PAIR_TRANSITIONS.csv、M1–M8 family result tables、CROSS_FAMILY_INTERACTIONS.csv、COSTS.csv、DECISION_LEDGER.json。

每个 run：
run_manifest.json、config.json、source_receipt.json、parent_state.json、batch_identity.json、action_receipt.json、metrics.json、costs.json、COMPLETE_VERIFIED.json 或 failure.json。

## 15. Mandatory tests

- U/P/FROZEN coverage、support、overlap；
- source/adapter model and loss parity；
- six-cell output ordering；
- Adam None/zero semantics；
- M1 U unchanged / P updated / second-order finite gradient；
- M2 row-task gradient reconstruction and virtual Adam non-mutation；
- M3 group size, exposure and optimizer parity；
- M4 permutation strata and feature-group identity；
- M5 label-pair/domain preservation and transform validity；
- M6 Jacobian finite-difference check and solver feasibility；
- M7 leave-one-seed-out no leakage；
- M8 projection/weight constraints and total gradient mass；
- cross-family component/combined identity；
- state save/restore prediction and next-batch parity；
- V_A/V_B disjointness and Test sealing；
- atomic-write and retry recovery；
- metric/statistic recomputation.

独立 Reviewer 必须审查所有八个 adapter 与 cross-family adapter。单个 adapter 未通过时隔离其 family，不阻塞其余 family。
