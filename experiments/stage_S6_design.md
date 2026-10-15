# Stage S6 设计文档 — Sample Objective Residual History Gate

设计者：Agent 1（designer）。本文件是 Stage S6 的唯一权威实现规格。
科研定义以 `promt/S6.md` 为准；本文件不修改任何科研标准，只定义实现。
Coder 必须仅凭本文件 + 现有代码完成实现，不得自行发明任何算法、权重、采样或调度。

---

## 1. Experiment objective

判定 S5（KuaiRand/Equal）与 S5-ML（MovieLens/Equal）中已确认的
「persistent Adam history mediated F→T target rescue」在把训练 objective
从 Equal 切换为 Sample 后是否仍存在稳定、达到 practical magnitude 的 residual。

两阶段 gate：

- **Phase A**：只跑 F / T 两臂 × 2 数据集 × 4 seeds（共 16 个 run）。
- **Phase B**：仅当某数据集 Phase A 判定为 GO / STRONG GO 时，才对该数据集
  补跑 HT / DH（每数据集 8 个 run）。PARTIAL GO / STOP 不自动触发 Phase B，
  PARTIAL GO 情形上交 Main Agent 裁决（`promt/S6.md` 第八节只定义了 GO/STOP 分支）。

唯一允许改变的变量：`objective_weighting: equal → sample`。
其余一切（target identity、architecture、optimizer 超参、data split、exposure、
初始化、optimizer grouping、checkpoint selection、current/history manipulation
实现、DataLoader 顺序、RNG 配对）逐项锁定，核验方式见 §3 与 §7。

---

## 2. Mathematical definition

### 2.1 Objective 定义（项目已审计口径，不得重新解释）

记 batch 内样本 i 属于域 d_i ∈ {0,1,2}，任务 t ∈ {0,1}，
BCE(p_it, y_it) 为逐样本逐任务二值交叉熵。

**Sample（S6 唯一使用的 objective）**

```
w_d = 1                              （loss coefficient）
L_full^Sample = mean_i( mean_t( BCE(p_it, y_it) ) )
effective domain coefficient a_d = p_d = n_d / N   （由普通打乱串联采样自然实现）
```

**Equal（S5/S5-ML anchor，仅用于对照，S6 不运行）**

```
w_d = 1 / (3 p_d)
L_full^Equal = mean_i( w[d_i] * mean_t( BCE(p_it, y_it) ) )
effective domain coefficient a_d = 1/3
```

实现位置（已存在、已被 S3 审计、必须原样复用）：

- `experiments/stage_S3_gradient_access.py:103` `loss_components(...)`：
  `objective == "sample"` 分支使用 `weights = torch.ones(3)`（第 130–132 行），
  逐样本 BCE 不加域权重直接取均值。
- `experiments/stage_S3_run.py:96` `_sample_objective_table(equal_table)`：
  生成 Sample 系数表（w_d=1、a_d=p_d、cell=a_d/2、population_mean_weight=1.0、
  double_weighting=False、sampling="ordinary_shuffled_concatenated_loader"）。
- `experiments/stage_S3_run.py:56-63` `OBJECTIVE_NAMES/OBJECTIVE_FORMULAS`：
  sample → `sample_population_mean_task_bce` / `mean_i(mean_t(BCE(p_it,y_it)))`。

Sample 的 target/non-target 分解（与 Equal 同一函数、同一语义）：

```
L_target = mean_i( 1[d_i = target] * mean_t(BCE) )     （batch 内频率加权，≈ p_target * mean_BCE_target）
L_non    = L_full − L_target
equality_residual = L_full − (L_target + L_non) = 0    （精确成立，逐 step 审计）
```

注意：**Sample 下 L_target 不做 1/(3 p_d) 放大**，其 effective coefficient 就是
batch 经验频率。这正是 S3 审计过的 Sample 定义；禁止「为对齐 Equal 而重归一化
target 梯度」——那属于新方法，违反 `promt/S6.md` 第五节。

### 2.2 四臂定义（与 S5/S5-ML 逐字同义）

history = persistent Adam optimizer state（m = exp_avg，v = exp_avg_sq，
step/bias-correction 计数 τ；本阶段不拆 m/v）。current = 当前 shared update
的梯度来源。每个 batch 只有一次 `optimizer.step()`：

| arm | history source (g_H) | current source (g_D) | 实现语义 |
|-----|----------------------|----------------------|----------|
| F   | FULL                 | FULL                 | `optimizer.step()` 提交 g_F，无 post-step overwrite |
| T   | TARGET               | TARGET               | `optimizer.step()` 提交 g_T，无 post-step overwrite |
| HT  | TARGET               | FULL                 | 提交 g_T 进 m/v/τ，然后用 pre-step 状态计算的只读反事实 Δ_F 覆写 shared 参数：`θ ← θ_before + Δ_F` |
| DH  | FULL                 | TARGET               | 提交 g_F 进 m/v/τ，然后覆写 `θ ← θ_before + Δ_T` |

实现位置：`stage_S5_run.py` `S5Runner._controlled_step`（KuaiRand）/
`stage_S5ML_run.py` `S5MLRunner._controlled_step`（MovieLens）。g_F/g_T 由
`gradients_one_forward(losses["full"], losses["target"], ...)` 单次前向算出；
`adam_deltas` 从 pre-step persistent state 只读计算 Δ_F/Δ_T。该机制与
objective weighting **完全解耦**：objective 只决定 `losses["full"]` 与
`losses["target"]` 的数值，四臂 dispatch 逻辑不含任何 objective 分支。
S6 对该机制 **零改动**。

### 2.3 主要 contrast

```
Phase A:  R_T^Sample = T − F                       （per cell: AUC 与 LogLoss）
Phase B:  R_H = HT − F,  R_D = DH − F,  HT − T,  DH − T
HR = mean(HT−F)/mean(T−F)   （History Recovery Ratio）
CR = mean(DH−F)/mean(T−F)   （Current Recovery Ratio）
```

---

## 3. Independent variables / Controlled variables

### 3.1 独立变量（唯一）

| 变量 | S5/S5-ML | S6 |
|------|----------|-----|
| objective_weighting | equal | sample |

### 3.2 锁定项逐项清单（含核验方式）

| 锁定项 | 锁定值（KuaiRand / MovieLens） | 核验方式 |
|--------|-------------------------------|----------|
| dataset / setting | kuairand/natural；movielens/d2_large | `_validate_args` fail-closed |
| backbone | m3oe（`m3oe/MDMTRec/models/multi_domain/model.py`，sha256 = S3 pin `a7702a64…`） | inventory + `MODEL_SOURCE_SHA256` 运行时比对 |
| target identity | KuaiRand local_domain=1（tab1），tasks t0=is_click/t1=long_view；MovieLens local_domain=2（ages_35_45_50_56），t0=rating>3/t1=rating>4 | `target_for()`；manifest `target_local_domain` |
| seeds | 2022, 2023, 2024, 2025 | `SEEDS`；key 校验 |
| epochs | **50（两数据集相同；S5 与 S5-ML `_validate_args` 均锁 epoch=50）** | `_validate_args` |
| batch_size | 40960 | 同上 |
| optimizer | Adam，main lr=0.01，arch lr=0.001，betas=(0.9,0.999)，weight_decay=1e-5 | 同上 + manifest `formal_protocol_observed` |
| optimizer grouping | shared/private/architecture 三组，partition 与 S3/S4 相同 | `gradient_partition.json` + `assert_expected_parameter_counts` |
| architecture | expert_num=4，fcn_dims=[512,256,256,64]，embedding_dim=16 | `_validate_args` expected dict |
| data split / exposure | 与 S2/S3/S5 完全一致（`ordinary_shuffled_concatenated_loader`，shuffle=True，drop_last=False，num_workers=0，无 balanced sampler） | `dataset_file_hash`、`split_row_fingerprints`、`batch_order_hash`（PAIR_FIELDS 跨臂一致） |
| data file hash | KuaiRand `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`；MovieLens `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4` | orchestrator `validate_run` 硬比对 |
| 初始化 | 逐 seed 初始参数 hash 必须与 S5/S5-ML pin **完全一致**（objective 不影响初始化）：KuaiRand 用 S4 `INITIAL_PARAMETER_SHA256`（2022=`381a86db…`，2023=`2280adf3…`，2024=`7bb9447c…`，2025=`90f375b1…`）；MovieLens 用 S5ML pin（2022=`bfa86eb4…`，2023=`43cecc5a…`，2024=`79088924…`，2025=`1d5048bf…`） | manifest `initial_parameter_state_hash` vs pin |
| checkpoint selection | unweighted validation macro-AUC 最大、并列取最早 epoch（`stage_S1_run.py:157` `select_checkpoint_epoch`），test-invisible（selection 冻结后才评 test）；shadow_patience=10 | manifest `checkpoint_selection_frozen_before_test=True`、`test_evaluated_during_training=False` |
| current/history manipulation | §2.2 机制零改动 | I1–I9 不变量（见 §7）逐 step fail-closed |
| common epochs | (9, 19, 29, 39, 49)；moment dump epochs (0, 1, 9, 19, 29, 39, 49) | manifest |
| device | cuda:0 | `_validate_args` |

### 3.3 允许且仅允许不同的字段

stage 名（"S6"/"S6ML"）、objective 三字段（objective_weighting / objective_name /
objective_formula）、coefficient_table_hash（换为 §3.4 的 Sample pin）、
formal_run_key、fresh_role/provenance_type（"fresh_s6"/"fresh_s6ml"）、
run_dir/输出路径、四个新 stage 源文件的 code hash、queue manifest 路径与 hash。

### 3.4 Sample 系数表 pin（已从 S3 审计产物逐字节复算确认）

| dataset | train_counts (d0,d1,d2) | p_d | sample coefficient_table_hash |
|---------|------------------------|-----|-------------------------------|
| kuairand/natural | 172782 / 1743898 / 89452（total 2006132） | 0.08612693481784847 / 0.8692837759429589 / 0.044589289239192635 | `30c68c9b7b3a0966d1cdeb1e6dce65647d0459dcdf946c57aa735f3b4762aec7` |
| movielens/d2_large | 34500 / 3000 / 150000（total 187500） | 0.184 / 0.016 / 0.8 | `e0616863f6970789ae92b1d2f975d497da83d89bd224af7416496f5ccbf63463` |

复算方式（designer 已执行验证，与 S3 产物
`stage_S3_raw_results/full_replacement/sample/.../objective_coefficients.json`
逐字段一致）：`_sample_objective_table(compute_equal_coefficients(counts))`。
注意两数据集的 target 域都是最大域（KuaiRand d1=86.9%，MovieLens d2=80%）。

---

## 4. Required code changes（最小修改原则：新增 8 个文件，不修改任何现有文件）

现有源码零改动。S6 复用 S5/S5-ML 的全部训练机制，按项目惯例
（S5MLRunner 之于 S5Runner 的常量替换克隆模式）新增四个 S6 文件 +
四个 S6ML 文件。

### 4.1 `experiments/stage_S6_protocol.py`（克隆 `stage_S5_protocol.py` + S5ML 相位机制）

- `STAGE = "S6"`；`RAW_ROOT = experiments/stage_S6_raw_results`；
  `OUTPUT_ROOT = experiments/stage_S6_outputs`；
  `QUEUE_PATH = experiments/stage_S6_queue_manifest.json`（Phase A）；
  `QUEUE_PATH_PHASE_B = experiments/stage_S6_queue_manifest_phaseB.json`；
  `queue_path_for_phase(phase)` 同 S5ML。
- `ARMS/ARM_HISTORY/ARM_CURRENT/COMMON_EPOCHS/MOMENT_DUMP_EPOCHS/LABELS` 原样。
- 新增 `PHASE_A_ARMS = ("F","T")`、`PHASE_B_ARMS = ("HT","DH")`；
  `matrix(raw_root, *, phase="A")` 与 `waves(tasks)` 采用 S5ML 的相位参数化版本
  （waves 只对矩阵中实际存在的臂对发波，Phase B 不重跑 F/T）。
- `key(seed, arm)` → `"s6|sample|kuairand|natural|m3oe|seed{seed}|{arm}"`。
- 新增 pin：`SAMPLE_TABLE_SHA256 = "30c68c9b7b3a0966d1cdeb1e6dce65647d0459dcdf946c57aa735f3b4762aec7"`。
  继续从 S4 protocol 引用 `KUAIRAND_DATA_SHA256`、`INITIAL_PARAMETER_SHA256`、`SEEDS`（值不变）。
- `S6_SOURCE_FILES` = 四个 S6 文件；`inventory(root)` = S4 inventory + S6 四个文件。

### 4.2 `experiments/stage_S6_run.py`（克隆 `stage_S5_run.py`，常量替换点如下）

`class S6Runner(S4Runner)`，逐点替换（这是全部差异，除此之外函数体逐字保持）：

1. import 改为 `stage_S6_protocol`（STAGE、ARMS、ARM_HISTORY、ARM_CURRENT、
   COMMON_EPOCHS、MOMENT_DUMP_EPOCHS、SEEDS、inventory、inventory_hash、key、
   SAMPLE_TABLE_SHA256）。
2. `__init__`：`args.objective_weighting = "sample"`（替换 `"equal"`）；
   `args.fresh_role = "fresh_s6"`。其余（shared_access="FULL" 仅作日志列、
   arm dispatch、validate_adam_delta_against_clone、inventory 校验）不变。
3. `_validate_args`：五元组校验改为 `("kuairand","natural","m3oe","sample","joint")`；
   epoch/lr/betas 等 expected dict 逐项不变。
4. `_output_formal_protocol`：`"objective": "sample"`；其余字段不变（epochs=50 等）。
5. `_condition_name`：`"s6_" + self.arm`。
6. `write_outputs`：`fresh_role/provenance_type = "fresh_s6"`；
   `source_code_hash` 指向 `experiments/stage_S6_run.py`；
   **新增** `"coefficient_table_hash": SAMPLE_TABLE_SHA256` 写入 config/manifest
   （KuaiRand S5 未在 run 文件 pin 此值，S6 主动补上以与 S6ML 对齐）。
7. `build_argparser`：`--dataset` choices=("kuairand",)；`--arm` 必需；其余同 S5。
8. `main()` 的 print/failure.json 中 stage 字段随 STAGE 自动切换。

不改：`_controlled_step`（四臂机制+I1–I9）、`_dump_moment_state`、
`_optimizer_state_digest`、`_moment_cosine`、`_architecture_update`、
`_check_source_drift`、字段列表（S5_FIELDS/S5_PROBE_FIELDS 原样，列名不动）。

### 4.3 `experiments/stage_S6_orchestrator.py`（克隆 `stage_S5_orchestrator.py` + S5ML 相位管线）

- 引入 `phase` 参数与 `queue_path_for_phase`（同 S5ML 第 309/461 行模式）：
  Phase A 冻结/校验 8 tasks / 4 waves（("F","T") × 4 seeds）；
  Phase B 8 tasks / 4 waves（("HT","DH") × 4 seeds）。
- `validate_run` 保持 S5 的全部检查，并新增两条 pin 比对
  （S5ML 第 195–199 行同款）：
  `m["dataset_file_hash"] == KUAIRAND_DATA_SHA256`、
  `m["coefficient_table_hash"] == SAMPLE_TABLE_SHA256`、
  `m["initial_parameter_state_hash"] == INITIAL_PARAMETER_SHA256[seed]`。
- Phase A 基数：每 run main step 行数 = `batches_per_epoch × 50`、arch 行数 = 50、
  probe 行数 = 100；`zero_target_main_steps ≤ 1% main_steps_seen` 门槛保留。
- wave 执行参数不变：`max_training_processes=2`、`max_attempts_per_wave=2`。
- Phase B 收集时按 S5ML 的 `_load_frozen_phase_queue` + 跨相位 pairing 校验
  （PAIR_FIELDS + batch_order_hash 跨相位一致）处理；Phase A 产物永不重跑。

### 4.4 `experiments/stage_S6_report.py`（克隆 `stage_S5_report.py`）

- Phase A collect：runs=8、raw=336（7 checkpoints × 3 domains × 2 tasks × 8 runs）、
  probes=800；contrast 只含 `T-F`（PAIRS 缩减为 (("T","F"),)）。
- Phase B collect：合并 Phase A 冻结队列，恢复四臂 PAIRS
  （("T","F"),("HT","F"),("DH","F"),("HT","T"),("DH","T")，与 S5 PAIRS 同序）、
  moment gate（dump epoch 0/1 处 HT 距 T 近于 F、DH 距 F 近于 T）；
  全量基数回到 16 runs / 672 raw / 1600 probes。
- 输出文件名 `stage_S6_{runs,raw_results,paired_results,gradient_steps,probe,aggregates,moment_audit}.csv`
  + `stage_S6_summary_manifest.json` + `reports/stage_S6_run_report.md`。
- 新增 `stage_S6_config_lock_audit.csv`（见 §6.3）。

### 4.5–4.8 `experiments/stage_S6ML_{protocol,run,orchestrator,report}.py`

对 `stage_S5ML_*` 做完全对称的常量替换：

- `STAGE = "S6ML"`；key → `"s6ml|sample|movielens|d2_large|m3oe|seed{seed}|{arm}"`；
  目录改 `stage_S6ML_raw_results` / `stage_S6ML_outputs` /
  `stage_S6ML_queue_manifest{,_phaseB}.json`。
- `OBJECTIVE_WEIGHTING = "sample"`；`_validate_args` 五元组
  `(movielens, d2_large, m3oe, sample, joint)`；`_output_formal_protocol`
  `"objective": "sample"`；`__init__` 第 117 行替换为 `"sample"`。
- `SAMPLE_TABLE_SHA256 = "e0616863f6970789ae92b1d2f975d497da83d89bd224af7416496f5ccbf63463"`；
  run 文件第 434 行与 orchestrator 第 196 行的 `EQUAL_TABLE_SHA256` 比对全部
  替换为 `SAMPLE_TABLE_SHA256`。
- `MOVIELENS_DATA_SHA256`、`INITIAL_PARAMETER_SHA256`、`TARGET_LOCAL_DOMAIN=2`、
  `TARGET_PHYSICAL_DOMAIN`、`TASKS` 原样保留。
- `build_argparser`：dataset choices 保持 `("movielens","kuairand")`，
  fresh_role choices 加入 `"fresh_s6ml"`。
- `S6MLRunner(S4Runner)`，`_condition_name = "s6ml_" + arm`，
  `fresh_role/provenance_type = "fresh_s6ml"`。

---

## 5. Required configuration changes

无独立配置文件。全部配置通过 §4 的常量与 CLI 参数表达：

- 每个 run 的 CLI：`--dataset {kuairand|movielens} --setting {natural|d2_large}
  --backbone m3oe --run_type joint --seed {2022..2025} --arm {F|T|HT|DH}
  --epoch 50 --batch_size 40960 --learning_rate 0.01 --learning_rate_darts 0.001
  --weight_decay 1e-5 --expert_num 4 --num_workers 0 --device cuda:0
  --shadow_patience 10 --formal_run_key <key> --run_dir <raw_root/arm/seedN/joint>
  --co_run_pair_id/--planned_peer_run_key/--worker_id/--attempt_id`（orchestrator 填充）。
- `objective_weighting` 不由 CLI 传入；由 runner `__init__` 内部钉死为 `"sample"`
  （与 S5 钉死 `"equal"` 同机制），`_validate_args` fail-closed。
- 冻结队列必须先于 RUN（reviewer PASS 后），Phase A 与 Phase B 使用各自独立
  append-only queue/state 文件。

---

## 6. Required metrics / logging

### 6.1 每个 run 的 audit 产物（与 S5/S5-ML 完全同构）

`config.json`、`run_manifest.json`、`batch_trace.csv`、`gradient_step.csv`、
`probe.csv`、`metrics_epoch.csv`、`metrics_test.csv`、`metrics_summary.csv`、
`objective_coefficients.json`（= §3.4 Sample 表）、`gradient_partition.json`、
`reachability_audit.json`、`initialization_audit.json`、`nonfinite_inventory.json`
（必须为空列表）、`predictions_test.npz`、`selected_checkpoint.pth`、
`final_checkpoint.pth`、`common_epoch_{9,19,29,39,49}.pth`、
`moment_state_epoch{0,1,9,19,29,39,49}.pth`、`optimizer_state.pth`、
`architecture_optimizer_state.pth`、`parameter_manifest.json`、`rng_state.pth`、
`process_interval.json`（status=OK）、`stdout.log`。

### 6.2 报告指标（per run × per cell × per checkpoint）

- checkpoint 标签：`selected`、`final`、`common_epoch_9/19/29/39/49`（共 7 个，
  缺一不可；禁止只报 selected）。
- 每个 (domain × task) cell（3 域 × 2 任务 = 6 cells）的 **AUC 与 LogLoss**，
  取自 `metrics_test.csv`；派生量：six-cell macro AUC/LogLoss、
  worst-domain AUC、worst-cell AUC（分析期从 per-cell raw 计算，与 S5 口径一致）。
- 主要 contrast（Phase A）：`T−F` 的 ΔAUC / ΔLogLoss，target cells
  （KuaiRand d1t0、d1t1；MovieLens d2t0、d2t1）为 primary，其余 cells 全量报告。
- Phase B 追加：`HT−F`、`DH−F`、`HT−T`、`DH−T`，HR/CR。
- 训练阶段维度：early=common9、middle=common29、late=common49 + selected + final
  （S6.md 第十一节，缺一不可）。
- batch 级：`L_full/L_target/L_non/decomposition_residual`（Sample 权重下）、
  gradient norms、applied/history/private formula residuals、RNG digest 对、
  Adam step 计数——字段与 S5 完全相同。

### 6.3 S6 新增审计：`stage_S6_config_lock_audit.csv`

对每个 S6 run，与同 dataset/seed/arm 的 S5（或 S5-ML）canonical run 逐项 diff
§3.2 锁定清单字段：dataset_file_hash、split_row_fingerprints、
split_row_id_hashes、initial_parameter_state_hash、gradient_partition_hash、
batch_order_hash（同 seed 跨臂一致即可，不要求与 S5 跨阶段一致——S5↔S6 跨阶段
比较仅作信息列）、train_domain_counts、model_parameter_manifest、
batches_per_epoch、以及 `_validate_args` expected dict 的全部训练超参。
任何 locked 字段不等 → 该 run 标记 FAIL，阻断 collect。允许不同的字段（§3.3）
也逐行列出并标注 `expected_diff=True`，其中 objective 三字段与
coefficient_table_hash 必须确实从 equal 变为 sample pin 值。

---

## 7. Audit / invariant 检查清单（Reviewer 核验用）

**Objective 核验（S6 新增重点）**

- O1：`objective_coefficients.json` 与 §3.4 pin 表逐字段一致（w_d 全 1.0、
  a_d == p_d == train_counts/total、cell = a_d/2、population_mean_weight=1.0、
  double_weighting=false、hash == SAMPLE_TABLE_SHA256）。
- O2：manifest 中所有 `objective_call_hashes` 只含 SAMPLE_TABLE_SHA256 一个值；
  `batch_trace.csv` 每行 `objective_weighting == "sample"`。
- O3：无额外 reweighting/curriculum/calibration/sampler——`loss_components`
  sample 分支为 `torch.ones(3)`；训练循环无其他 loss 路径（S3Runner.
  `_train_epoch` / `_architecture_update` 只经 `_objective_components`）。

**四臂同义（沿用 S5 不变量，零改动）**

- I1 installed==g_H（cosine ≥ 0.99999）；I2 observed step == Δ_H；
  I3 applied == Δ_D（HT/DH overwrite 后）；I4 private == AdamDelta(g_P) 且不被覆写；
  I5 Adam step 计数全参数同步 +1；I6 probe 步 optimizer.state digest 在 step()
  前后不被日志/覆写路径污染；I7 controlled step 不消耗 RNG（digest 前后相等）；
  I9 shared/arch None 梯度 fail-closed；`zero_target_main_steps ≤ 1%`。

**锁定 diff**

- L1：§3.2 表逐项核验通过；L2：config_lock_audit 无 FAIL。

**配对 / 公平性**

- P1：同 seed 四臂（Phase A 两臂）PAIR_FIELDS 全等 + `batch_order_hash` 相等
  （DataLoader 顺序由 seed 决定，objective 不参与；见 §9-P2 陷阱）。
- P2：每 run main/arch/probe 行数基数；epoch 数 = 50；batches_per_epoch 跨臂一致。

**Checkpoint 纪律**

- C1：`checkpoint_selection_frozen_before_test=True`、
  `test_evaluated_during_training=False`、selected 由 unweighted validation
  macro-AUC 产生（与 objective 无关，见 §9-P5）；final == common_epoch_49
  数值相等（`_checkpoint_state_numerically_equal`）。

**Moment gate（Phase B）**

- M1：dump epoch 0/1 处 cos(m_HT,m_T)>cos(m_HT,m_F) 且 cos(m_DH,m_F)>
  cos(m_DH,m_T)，v 同理（与 S5 report 相同实现）。

---

## 8. Phase A GO/STOP gate 判定表（逐字采用 promt/S6.md 第七节，阈值不得修改）

thresholds：practical `|ΔAUC| ≥ 0.003`；strong `|ΔAUC| ≥ 0.005`；
seed consistency 至少 3/4 同方向。**每个数据集独立判定。**
primary target tasks：KuaiRand d1t0/d1t1；MovieLens d2t0/d2t1。
每项必须同时看 selected / final / common-stage 的 AUC 与 LogLoss。

| 判定 | 条件 |
|------|------|
| **GO** | 至少一个 primary target task 同时满足：① selected mean ΔAUC ≥ +0.003；② ≥ 3/4 seeds positive；③ LogLoss 不出现稳定反向；④ common-stage 中至少多数预注册观察点仍为正；⑤ 效应不能完全由单 seed 或单 checkpoint 驱动 |
| **STRONG GO** | 两个 target tasks 均满足 GO 条件，且至少一个达到 +0.005 |
| **PARTIAL GO** | 存在 residual，但仅一个 task 稳定；或 selected 成立但 common-stage 较弱；或 magnitude 介于 weak/practical 边界 → 不自动进 Phase B，交 Main Agent 裁决 |
| **STOP** | 任一：两个 target task 均 < +0.003；seed 方向不稳定；仅 isolated checkpoint 出现；common-stage 不复现；AUC/LogLoss 明显系统性冲突；效应落入 baseline/seed variance 范围 |

**Phase B 触发条件**：某数据集 Phase A = GO 或 STRONG GO → 仅对该数据集
冻结 Phase B 队列并补跑 HT/DH（8 runs）；另一数据集 STOP 则不做 HT/DH
（S6.md 第八节 Case B）。双 STOP → 不收束任何 Phase B。

Equal anchor（只读引用，不拼接不重跑）：S5 KuaiRand T−F selected ≈
+0.041/+0.042（d1t0/d1t1）；S5-ML T−F ≈ +0.0194（d2t0）/+0.0143（d2t1），
4/4 seeds positive；CR ≈ 2%（KR）/ 25–30%（ML）。分析时从
`stage_S5_outputs` / `stage_S5ML_outputs` 聚合表读取。

---

## 9. Potential implementation pitfalls

- **P1 硬编码 equal 的路径**（每个 run 文件三处 + key + orchestrator pin）：
  `__init__` 的 `args.objective_weighting="equal"`、`_validate_args` 五元组、
  `_output_formal_protocol["objective"]`、`key()` 内嵌 `"equal"` 字符串、
  S5ML run 第 434 行与 S5ML orchestrator 第 196 行的 `EQUAL_TABLE_SHA256` 比对。
  漏改任何一处都会 fail-closed（好事）或静默跑成 equal（灾难）。§4 已枚举全部点位。
- **P2 objective 不影响 DataLoader**：采样是普通打乱串联 loader，batch 组成/顺序
  只由 seed 决定；Sample 的 a_d=p_d 由采样频率自然实现，**禁止**为 Sample 引入任何
  sampler 改动。P1 配对检查（batch_order_hash 跨臂相等）即为此守门。
- **P3 sample 权重与 target gradient 语义**：`loss_components` 中
  `L_target = mean(1[target]·BCE)` 是 batch 频率加权的「target 项」，T/DH 臂的
  g_T 即其梯度。这是 S3 审计口径，**不要**除以 p_d 或乘 w_d 去「对齐 Equal」。
  同样地 `zero_target_gradient_step`、I1–I9 语义在 Sample 下不变，门槛不变。
- **P4 `validate_coefficient_invariants` 只适用于 Equal**（它强制 p_d·w_d=1/3，
  会拒绝 Sample 表）。`loss_components` 仅在 equal 分支调用它；
  S2Runner 的 `batch_weight_audit` 会调用它，但 S3Runner._record_batch 直接调
  `S1Runner._record_batch` 绕开了 S2 的 batch audit——coder 不得改动这条继承链，
  也不得给 Sample 表套 equal 不变量。S3Runner.__init__ 先构造 equal 表再派生
  sample 表（`_sample_objective_table`），此顺序保持原样。
- **P5 checkpoint selection 与 objective 无关**：`select_checkpoint_epoch` 用
  unweighted validation macro-AUC，不读取训练 objective；无需也不许为它加
  objective 分支。确认 `metrics_epoch.csv` 的 validation 评估路径不经过
  `loss_components`（评估是 per-cell AUC/LogLoss，与训练 loss 权重无关）。
- **P6 跨 objective 数值不可拼接**：S6(Sample) 的绝对 AUC 与 S5(Equal) 不可直接
  合并分析；Equal 只作 anchor（S6.md 第十节）。config_lock_audit 中 S5↔S6 的
  跨阶段字段比对仅是信息列，不作为放行条件。
- **P7 梯度量级变化**：Sample 去掉 1/(3p_d) 放大后 g 的范数整体变小，Adam
  历史范数、delta 范数与 S5 不可比；容差校验（TOLERANCE_ABS/REL）是相对
  参考范数的，沿用即可，不要为 Sample 重调容差。
- **P8 Phase B 不重跑 Phase A**：waves() 只对矩阵中存在的臂对发波；Phase A
  canonical 产物在 Phase B 收集中以冻结队列形式只读引用（S5ML 模式）。
- **P9 过期 pin 污染**：S5 protocol import 了 S4 的 EQUAL_TABLE_SHA256 但未使用；
  S6 protocol 不得携带任何 equal 表 pin，只保留 §3.4 的 SAMPLE_TABLE_SHA256。
- **P10 目标域即最大域**：两数据集 target 都是最大域（KR 86.9%、ML 80%），
  Sample 下 target 项天然主导 L_full；这属于已知背景，不改变任何实现，
  但分析时不得把「Sample 下 F 本身就偏向 target」误读为 history residual。
- **P11 fresh_role 校验链**：S3Runner._validate_args 的 fresh_role 白名单被
  S5/S6 runner 各自的 `_validate_args` 覆盖；S6 runner 的 fresh_role
  （"fresh_s6"/"fresh_s6ml"）只出现在自身校验与 manifest 中，不要回写进
  S3 protocol 的 FRESH_* 常量。

---

## 10. 给 coder 的实现规格（逐步）

1. 创建 §4 的 8 个新文件（4 个 S6/KuaiRand + 4 个 S6ML/MovieLens），
   按 §4.1–4.8 的点位表从对应 S5/S5-ML 文件克隆并做常量替换；**不修改任何
   现有文件**。每处替换与本文件 §4 逐条对照打勾。
2. 静态核验（不训练）：
   a. `python -c "from stage_S6_protocol import *; print(matrix(RAW_ROOT, phase='A'))"`
      输出 8 个 task，key 形如 `s6|sample|kuairand|natural|m3oe|seed2022|F`；
   b. 用 §3.4 的复算式确认两个 SAMPLE_TABLE_SHA256 与本文件一致；
   c. `S6Runner._validate_args` 对 equal 参数组合 fail-closed、对 sample 放行
      （可用 dry argparse 对象单测，不启动训练）；
   d. grep 全部新文件，确认不存在 `"equal"` 字面量残留（注释/文档串除外，
      且文档串须标明指 S5 anchor）。
3. 冻结 Phase A 队列（两个数据集各自）：`freeze_queue(raw_root, queue_path,
   phase="A")` → 8 tasks / 4 waves；reviewer PASS 前不得 RUN。
4. Runner 执行 Phase A（每数据集 4 waves × 每波 F+T 两进程并发，
   seeds 2022–2025，共 16 runs）；逐 seed 保存原始产物；失败保存 failure.json
   与 retry 记录。
5. Phase A collect：`stage_S6_report.collect(phase="A")` /
   `stage_S6ML_report.collect(phase="A")`，生成 §7 全部 CSV、summary manifest、
   config_lock_audit；任何 audit FAIL 则停止并上报，不得进入判定。
6. 将 paired/aggregate 表交 Main Agent 按 §8 判定 GO/STOP。
7. 仅当某数据集 GO/STRONG GO：冻结该数据集 Phase B 队列（HT/DH × 4 seeds），
   重复步骤 4–5（phase="B"），collect 合并四臂并计算 HR/CR 输入表与
   moment gate。
8. 全程禁止：修改 S5/S5-ML/S4/S3/S2/S1 任何文件；调整阈值/容差；重跑已
   canonical 的 run；向训练循环加入任何本文件未列出的逻辑。

### 冒烟建议（可选，非 formal）

复用现有 smoke 机制（`run_s3_update_smoke(..., objective="sample")` 已存在）
验证 S6Runner 的 `_controlled_step` 在 sample 下 I1–I9 成立；formal 矩阵
不允许 smoke 捷径（`_validate_args` 已禁止 smoke_only）。

---

## 附：Expected output files / 目录结构

```
experiments/
  stage_S6_protocol.py / stage_S6_run.py / stage_S6_orchestrator.py / stage_S6_report.py
  stage_S6ML_protocol.py / stage_S6ML_run.py / stage_S6ML_orchestrator.py / stage_S6ML_report.py
  stage_S6_queue_manifest.json            # Phase A 冻结队列（KuaiRand）
  stage_S6_queue_manifest_phaseB.json     # Phase B（仅 GO 后）
  stage_S6ML_queue_manifest.json          # Phase A（MovieLens）
  stage_S6ML_queue_manifest_phaseB.json   # Phase B（仅 GO 后）
  stage_S6_raw_results/{F,T,HT,DH}/seed{2022..2025}/joint/   # §6.1 全部产物
  stage_S6ML_raw_results/{F,T,HT,DH}/seed{2022..2025}/joint/
  stage_S6_outputs/
    stage_S6_{runs,raw_results,paired_results,gradient_steps,probe,aggregates,moment_audit}.csv
    stage_S6_config_lock_audit.csv
    stage_S6_summary_manifest.json
    reports/stage_S6_run_report.md
  stage_S6_outputs/ 同构 stage_S6ML_outputs/
```

Phase A 只物化 F/T 目录（各 4 seeds）；HT/DH 目录仅在 Phase B 后出现。
报告文档（`stage_S6_phaseA_run_report.md` / `stage_S6_phaseA_analysis.md` 等，
S6.md 第十八节）由 pipeline 下游角色撰写，不在本设计与 coder 范围内。
