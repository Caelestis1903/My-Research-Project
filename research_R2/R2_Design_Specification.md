# R2 Design Specification — AO-04 五臂融合实验实现设计（Algorithm Specification）

日期：2026-10-11（Asia/Shanghai）。作者：Stage R2 Designer。状态：**设计稿，待 Independent Implementation Review**。
依据文件：`research_R1_R3/R1_Experiment_Charter_v1.2.md`（科学合同主依据，下称 Charter）、`research_R1_R3/R2_Execution_Plan.md`（下称 Plan）、`research_R1_R3/R1_to_R2_Independent_Review.md`（下称 Review，P1/P2/P3 已 CLOSED）、`research_R1_R3/R1_Architecture_Audit.md`、`research_R1_R3/R1_Baseline_Protocol.md`、执行授权书 `promt/stage_r2.md`。
行为来源代码（全部只读，不得修改）：`experiments/stage_S1_models.py`、`experiments/stage_S1_run.py`、`experiments/stage_S1_data.py`、`experiments/stage_S1_metrics.py`、`experiments/stage_S1_protocol.py`、`experiments/stage_E1_instrumentation.py`、`m3oe/MDMTRec/models/multi_domain/model.py`。

本文件不包含任何最终实现代码；所有"规格"均为对 coder 的实现指令。旧 S0–S8 全部文件只读。

---

## 1. Experiment objective

**研究问题（AO-04）**：在强 Sample-M3OE 基线上，保持全局 own/other 总系数定义不变，学习他域专家内部相对比例（B1）是否产生方向重复、达到实用量级且无禁止退化的 AUC 信号；该信号是否相对简单域级权重（B2）、固定无意义比例表（B3）和逐格 own/other 调节（B4）仍有余量。唯一主比较始终 B1−B0。允许阴性和停止投入。

**假设**：H0 / H1 / Htask / Hown / Hpost 按 Charter §1 原文，不在此重述；Hpost 本阶段保持 **disabled**。

### 1.1 MovieLens 范围扩展说明与偏差登记（最高优先级用户指令）

Charter v1.2 原文（§4）限定"只使用 KuaiRand-Pure natural"。**用户范围修订指令**：实验必须在 KuaiRand-Pure natural 和 MovieLens（`d0_large`）两个数据集上都完成，才允许主分析。本设计将 Charter 的全部规则按"每数据集独立实例化"方式镜像到 MovieLens：相同五臂 B0–B4、相同优化协议、相同判决门（阈值逐字不变）、相同 G0/G1、相同配对协议。每一处无法镜像的差异在第 11 章逐项登记为 **用户授权的范围扩展偏差（D-ML-x）**。汇总：

| 编号 | 偏差 | 处理方式 |
|---|---|---|
| D-ML-1 | Charter §4"只使用 KuaiRand"被用户指令覆盖，新增 MovieLens d0_large 数据集 | 全部合同规则按数据集独立实例化；判决门每数据集独立计算；两个数据集的 R2-A 均有效完成后才允许主分析 |
| D-ML-2 | 预算加倍：R2-A 由 6 attempt 变为 12 attempt（2 数据集 × 3 臂 × 2 seed），R2-B 由最多 4 变为最多 8；总上限由 10 attempt/500 run-epochs 变为 20 attempt/1000 run-epochs；resume 上限按每数据集队列独立计（每队列 ≤2 次，总计 ≤4 次）；smoke 上限由 15 batch 变为 30 batch（每数据集每臂 ≤3 batch、seed 2022） | 每数据集队列独立适用 Charter §9 的全部计数规则 |
| D-ML-3 | MovieLens 标签为构造标签（`1[rating>3]`/`1[rating>4]`），非 stored 标签 | 逐字复用 `stage_S1_data.py` 的构造规则，不得改写 |
| D-ML-4 | MovieLens 含 1 个 dense 特征（age，base-train min-max 缩放）与 7 个 sparse 特征；KuaiRand 34 个 sparse 特征无 dense | 逐字复用 `stage_S1_data.py` |
| D-ML-5 | MovieLens train 子集依赖 seed（角色轮转抽样 `RandomState(10*seed+physical_domain)`）；val/test 与 seed 无关 | 配对按 (dataset, seed) 进行，不受影响；详见 §11 |
| D-ML-6 | MovieLens 无 split 后 filter 步骤（KuaiRand 为 split-first/filter-second） | 逐字复用 `stage_S1_data.py` |
| D-ML-7 | MovieLens 历史期望计数、参数总数、指纹锚点与 KuaiRand 不同 | §11 给出全部期望值的来源与验收方法 |
| D-ML-8 | 历史 S1 MovieLens 已使用的 test 同样不是全新 holdout | 披露义务同 KuaiRand |

除上述登记项外，不得引入任何其他差异。MovieLens 的 setting 固定为 `d0_large`（线索来源：用户指令中的 `movielens/d0_large` 设置与 `movie_lens_role_sizes {L:150000, M:34500, S:3000}`、`movie_lens_rotation_rule RandomState(10*seed+physical_domain)`，均与 `stage_S1_protocol.py` 的 FORMAL_PROTOCOL 一致）。

---

## 2. Mathematical definition

记号：D=3、T=2、N=4；格索引 task-major `i = t*3 + d`（t0d0, t0d1, t0d2, t1d0, t1d1, t1d2）。`E_d` 为域专家 d 的输出（64 维），`H_n` 为共享专家 n，`U_t` 为任务专家 t。四个全局 raw scalar（初值全 1，对应 sigmoid(1)≈0.7310586）：

```
aD = sigmoid(_weight_exp_d.deep_weights)   # raw_aD
aT = sigmoid(_weight_exp_t.deep_weights)   # raw_aT
bD = sigmoid(_weight_bal_d.deep_weights)   # raw_bD
bT = sigmoid(_weight_bal_t.deep_weights)   # raw_bT
```

公共融合式（五臂相同）：

```
F_dt = sum_n g_dt,n(detach(emb)) * H_n + aD * R_dt + aT * Q_t
Q_t  = bT * U_t + (1 - bT) * U_(1-t)                    # T=2，无多他任务比例对象
prediction_dt = sigmoid(tower_dt(F_dt))
```

他域源模块固定顺序（j < k，rho 乘 E_j，(1-rho) 乘 E_k）：

```
d = 0: (j, k) = (1, 2)
d = 1: (j, k) = (0, 2)
d = 2: (j, k) = (0, 1)
```

五臂的域项 `R_dt` 逐字定义：

| 臂 | R_dt | 新可训练标量 | 初始化 |
|---|---|---:|---|
| B0 | `bD*E_d + (1-bD)/2 * (E_j + E_k)` | 0 | 原样 |
| B1 | `bD*E_d + (1-bD)*(rho_dt*E_j + (1-rho_dt)*E_k)` | 6 | `rho_dt = sigmoid(q_dt)`，`q = 0`（6 维，task-major） |
| B2 | 同 B1，但 `rho_td = sigmoid(q_d)`，3 个 `q_d` 两任务共享 | 3 | `q_d = 0`（3 维，按 d 排序） |
| B3 | 同 B1，但每格 rho 为固定 buffer 表 | 0 | 固定 buffer，**不初始等价**；不在 optimizer 内 |
| B4 | `b_dt*E_d + (1-b_dt)/2 * (E_j + E_k)` | 6 | `b_dt = sigmoid(raw_bD + delta_dt)`，`delta = 0`（6 维，task-major） |

B3 固定表（task-major：t0d0, t0d1, t0d2, t1d0, t1d1, t1d2；来源 NumPy2.1.3 独立 RandomState(MT19937) 10404 一次 Uniform(0.1,0.9)，直接读常数，不在训练流重抽）：

```
[0.7502780865025745, 0.8537642404691582, 0.44953240522281324,
 0.2525338669196546, 0.23434256306102627, 0.765217885741144]
```

该表无空格 UTF-8 JSON 数组的 SHA256 = `182edcbe382257dedfc725e7f0939e90edff68569dca889a1d10b918fc5a9957`。

**B4 关键约束**：`raw_bD` 必须取 `_weight_bal_d.deep_weights` 的**未 sigmoid 原值**，实际计算 `sigmoid(raw_bD + delta_dt)`；禁止把已 sigmoid 的 bD 再加 delta 当 raw 造成双 sigmoid。B4 保留原四 raw scalar 全部原路径（raw_bD 仍在 arch Adam 中照常更新，其梯度经 `sigmoid(raw_bD + delta_dt)` 回传）。初始化 delta=0 时 B4 与 B0 前向代数等价。共同 shift 冗余（`raw_bD+c, delta_dt-c` 给出同一 b_dt）是预定披露的混合 estimand，不得加 sum-zero 约束、不得冻结 raw_bD、不得改 arch 频率、不得新增正则。

B1/B2 只改 other 内部比例：own 系数 `aD*bD`、other 总系数 `aD*(1-bD)` 的代数定义不变。B4 主动允许逐格 own/other 总比例变化（`aD*(1-b_dt)`），other 内部仍均匀；不得把 B1 的"other 总质量不变"断言错套到 B4。

Sample objective（五臂相同）：`L = (1/(B*T)) * sum_i sum_t BCE(p_it, y_it)`，即 `torch.nn.functional.binary_cross_entropy(pred, target, reduction="mean"`，pred/target 形状 [B, 2]；不乘 p_d、不做 balanced sampling。batch 平均与全 epoch 样本加权两种日志分别报告。

---

## 3. Independent variables / Controlled variables

### 3.1 自变量（Independent）

- **臂**（arm）：B0 / B1 / B2 / B3 / B4。R2-A 固定 {B0, B1, B2}；R2-B 仅当追加门 YES 时 {B3, B4}。
- **seed**：{2022, 2023}。
- **数据集**（用户授权扩展）：`kuairand/natural`、`movielens/d0_large`。

### 3.2 控制变量（Controlled，全部锁定，逐字镜像 S1 正式协议）

| 变量 | 锁定值 |
|---|---|
| 主干 | StageS1M3OEAdapter 设备安全 forward；D=3/T=2/N=4；4 共享 + 3 域 + 2 任务专家；6 gate + 6 tower |
| embedding_dim / FCN | 16 / [512, 256, 256, 64] |
| expert_num | 4 |
| batch_size / num_workers / shuffle / drop_last | 40960 / 0 / True / False |
| epochs | 完整 50，无 scheduler，无实际 early stop |
| shadow patience | 10，仅 shadow 记录，不 break |
| main Adam | lr=0.01, wd=1e-5, betas=(0.9, 0.999), eps=1e-8（torch 默认），每普通 train batch 一步 |
| arch Adam | lr=0.001, wd=1e-5, betas=(0.9, 0.999), eps=1e-8，每 epoch 一个 train batch 一步（选择规则见 §5.3） |
| 四 raw scalar 初值 | 全 1 |
| B1/B2 q、B4 delta 初值 | 全 0（不消耗 RNG） |
| 选模 | 每 epoch 完整 validation 六格 macro AUC，严格 > 替换，tie 取最早 epoch；50 epoch 全部完成 |
| objective | direct per-example task-mean BCE（§2） |
| 指标实现 | `stage_S1_metrics.safe_auc`（tie-aware average rank）/ `safe_logloss`（clip eps=LOGLOSS_EPS，见 §12.10 登记） |
| device / cudnn | cuda:0；deterministic=True，benchmark=False；不声称跨 runtime 位级一致 |
| runtime | Python 3.13.5 / Torch 2.9.0+cu126 / CUDA 12.6 / cuDNN 91002 / NumPy 2.1.3 / pandas 2.2.3 / sklearn 1.6.1（已在当前 venv 核实） |
| forbidden | momentum / ema / curriculum / schedule / safe_mtl / balanced_sampler / importance_correction / adaptive_weighting；不得新增臂、seed、超参、正则、在线亲和 |

---

## 4. Required code changes

新建隔离目录 `research_R2/`。只允许在 `research_R2/` 下新建文件；`experiments/` 与 `m3oe/` 下全部旧文件**只读不改**，通过 `sys.path` 只读 import 复用其数值协议。允许 import 的旧模块：`stage_S1_data`（load_stage_s1_bundle、array_hash、split_row_fingerprint、row_ids_hash）、`stage_S1_models`（StageS1M3OEAdapter、build_model、parameter_manifest、initial_state_hash、is_architecture_parameter、estimate_forward_flops）、`stage_S1_metrics`（safe_auc、safe_logloss、cell_metrics、LOGLOSS_EPS、task_domain_cell_index）、`stage_S1_protocol`（FORMAL_PROTOCOL、COMMON_EPOCHS、EPOCHS、ROLE_SIZES、ROLE_MAPS）、`stage_E1_instrumentation`（batch_fingerprint、preserve_rng_state、rng_state_digest、write_json、write_rows_csv）。

### 4.1 `research_R2/src/ao04_v12_model.py`

- `class AO04M3OE(StageS1M3OEAdapter)`：构造签名在父类基础上增加关键字 `arm: str`（∈ {"B0","B1","B2","B3","B4"}）。`super().__init__` 之后按臂新增：
  - B1：`self.q_rho = nn.Parameter(torch.zeros(6))`；B2：`self.q_rho = nn.Parameter(torch.zeros(3))`。参数名禁止以 `_weight_` 或 `architecture.` 开头（避免被 `is_architecture_parameter` 误归）。
  - B3：`self.register_buffer("rho_fixed", torch.tensor(B3_TABLE, dtype=torch.float32))`（6 维常数；buffer 进入 state_dict 以便 strict load，但不进 named_parameters、不进任何 optimizer）。
  - B4：`self.delta_bal_d = nn.Parameter(torch.zeros(6))`（命名同样避开 `_weight_` 前缀）。
  - 新增参数/常量的构造**一律用 `torch.zeros`/常数 tensor，禁止任何随机初始化**，保证不消耗 RNG、不破坏配对。
- 覆写 `forward`：逐行复制 `StageS1M3OEAdapter.forward`（`stage_S1_models.py:142-193`）直到 `domain_features`/`task_features` 组装完毕；随后按 §2 的臂定义逐格计算域项，再走原 gate 融合与 tower。格循环索引 `i = t*3+d`，j/k 映射固定为 `{0:(1,2), 1:(0,2), 2:(0,1)}`。gate 输入保持 `emb.detach()`；STAR/gate/tower/任务项/skip/dtype/设备路径一律不改。
  - B4 的域项用 `raw_bD = self._weight_bal_d.deep_weights`（shape [1]）与 `torch.sigmoid(raw_bD + delta)`；**禁止**在 B4 域项中调用 `self._weight_bal_d()`（其返回已 sigmoid 的值，再用即双 sigmoid）。其余三处系数（aD/aT/bT）维持原调用方式。
  - 说明（非实现自由项）：父类 `_weight_bal_d()` 每次调用会递减 `tau`；softmax_type=3 下 tau 不影响输出，此行为无害但必须保留原样，不得"修复"。
- 提供查询方法（供日志与审查）：`arm_parameter_names() -> List[str]`（新增参数名）、`architecture_parameter_names() -> List[str]`（必须恰为 `["_weight_exp_d.deep_weights","_weight_exp_t.deep_weights","_weight_bal_d.deep_weights","_weight_bal_t.deep_weights"]`）、`fusion_coefficients() -> Dict[str, float]`（每格 own/other 系数、rho 或 b_dt 当前值，eval no_grad 下计算）。
- `build_ao04_model(schema, arm, seed, ...)`：先 `torch.manual_seed(seed)`（与 `build_model` 相同位置），再构造；保证 B0–B4 同 seed 下原 tensor 初始化逐位一致。

### 4.2 `research_R2/src/ao04_v12_protocol.py`

冻结常量模块（无训练代码）：臂集合、数据集集合及 setting、种子 (2022, 2023)、FORMAL_PROTOCOL 镜像、B3 表与其 JSON hash、两数据集期望计数（§5.4、§11）、两数据集各臂期望参数总数（§8.3）、guard 阈值全集（§10.3）、队列定义（§10.1）、预算常量（A 每数据集 6、B 每数据集 ≤4、50 epoch/attempt、resume 每 run ≤1 每队列 ≤2、smoke 每臂 ≤3 batch）、判决门纯函数所需的阈值常量（0.003 / 0.001 / guard 线）。该模块是一切"数值不得写死在 runner 里"的唯一来源。

### 4.3 `research_R2/src/ao04_v12_runner.py`

必须实现为**四个互相分离的组件**，禁止裸调用旧 `S1Runner.fit` / `S1Runner.write_outputs` / `_evaluate_deferred_checkpoints`：

1. **`paired_init_phase(dataset, seed)`**：在任何 B0 训练前执行一次。`setup_seed(seed)` → `load_stage_s1_bundle(...)` → 构造 B0 模型（不训练）→ 保存配对初始化工件（§8.1）。此后该 (dataset, seed) 的五臂共用此工件。
2. **`train_val_loop(arm, dataset, seed, run_dir)`**：50 epoch 训练 + 每 epoch validation + 选模 + shadow 记录 + checkpoint 捕获。逐字复刻旧 `_train_epoch`/`_architecture_update`/`fit` 中 epoch 0–49 的数值行为：
   - 每 epoch：train loader 全量一遍（shuffle=True，每 batch：main zero_grad(set_to_none=True) → forward → finite 检查 → BCE → backward → 非 arch 参数梯度 finite 检查 → main step → 参数 finite 检查）；随后一次 `_architecture_update`（规则见 §5.3）；随后 `preserve_rng_state` 包裹的完整 validation 六格评价。
   - epoch 末的全量 train eval pass（旧 `_train_epoch` 尾部以 shuffle=False 新建 loader 跑 `_cell_records` 得 train_cell_logloss）必须保留，保持数值协议与日志口径一致。
   - 选模：严格 > 更新 selected（CPU state 副本），tie 最早；shadow patience=10 仅记录首次触发 epoch；epoch ∈ {9,19,29,39,49} 及 49 捕获 checkpoint。
   - **此函数所在调用链禁止出现 test loader / test 数组 / test 指标**。train 结束后直接进 state_writer。
3. **`state_writer(...)`**：写出 §7 列出的全部训练期工件（checkpoint、optimizer/RNG、metrics_epoch.csv、batch_trace.csv、日志、run manifest）。**必须能在 test 数组完全不存在/被密封的情况下成功执行**；manifest 中不得含任何 test 性能字段（允许含 test split 的行数/指纹等 preflight 完整性字段）。每个 epoch 末另写 `last_resume.pt`（§8.4）。
4. **`final_test_evaluator(run_dirs, unlock_manifest)`**：独立入口。仅当传入全局解封 manifest（§10.5）时才执行；strict load 每 run 的 selected/final/common{9,19,29,39,49} checkpoint，对 test loader 评价并写 `metrics_test.csv` / `predictions_test.npz` / test 段 manifest 补写。selected 为主端点，final/common 仅描述。

**Test 隔离 tripwire（必须实现，供 Implementation Review 动态验证）**：
- `class SealedTestSplit`：bundle 加载后，runner 侧只持有该密封对象；它仅暴露 `row_count()`、`fingerprint()`、`class_coverage_check()`（preflight 计数/完整性/hash，对应旧 `_preflight_class_coverage` 的合法用途），任何返回特征/标签数组供 forward 的访问抛 `TestSealViolation`。
- 模块级计数器 `TEST_PREDICTION_CALLS`：只有 `final_test_evaluator` 内部会增加它；`ao04_v12_decision.py` 在生成 A 决策前断言其为 0；B 阶段启动后到全局解封前也断言为 0。
- 动态验证 fixture：synthetic spy loader + mock test loader，跑 ≤3 batch 的训练路径，断言 spy 迭代计数为 0（对应 Review P1 的验收方式）。
- 静态验证：reviewer 可 grep `ao04_v12_runner.py` 训练入口的 import 与调用链中无 `_evaluate_deferred_checkpoints`、`write_outputs`、`test_loader`、`predictions_test`。

**Nonfinite 处理**：参数/预测/loss/梯度任一 nonfinite → 立即抛错，run 标 INVALID，整个队列停止（与旧 runner 的 FloatingPointError 行为一致）。

**Resume**：仅允许从 `last_resume.pt`（完整 model + 两 optimizer + torch CPU/CUDA + numpy + python RNG + epoch/batch 位置 + arch iterator 已消耗批数 + 已累积 batch fingerprint 列表）精确续跑；每 run ≤1 次、每数据集队列 ≤2 次；不得从 selected checkpoint 接续训练。

### 4.4 `research_R2/src/ao04_v12_decision.py`

validation-only 判决门（纯函数 + 不可变 manifest 写出），规格见 §10.2–§10.5。该模块**只读**各 run 的 validation 指标与 manifest；断言 `TEST_PREDICTION_CALLS == 0`；输出不含任何 test 字段。

### 4.5 `research_R2/src/ao04_v12_hpost.py`（仅存开关，不实现扫描）

仅包含 `HPOST_ENABLED = False` 常量与一个在调用时立即 `raise NotImplementedError("Hpost is disabled in Stage R2")` 的占位函数。不得实现 grid 扫描逻辑。未来启用须另行授权、登记并 review。

---

## 5. Required configuration changes

在 `research_R2/configs/` 下放两份固定 config（训练时由 runner 读入并逐字段断言，不允许命令行覆盖任何锁定字段）：

### 5.1 `research_R2/configs/kuairand_natural_m3oe.json`

```json
{
  "stage": "R2", "design": "ao04_v12", "dataset": "kuairand", "setting": "natural",
  "backbone": "m3oe", "run_type": "joint",
  "arms": ["B0", "B1", "B2", "B3", "B4"], "r2a_arms": ["B0", "B1", "B2"],
  "r2b_arms": ["B3", "B4"], "seeds": [2022, 2023],
  "dataset_path": "D:\\CODE\\data\\kuairand\\processed\\kuai_rand.csv",
  "dataset_sha256": "a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f",
  "dataset_size_bytes": 416370411,
  "batch_size": 40960, "epochs": 50, "num_workers": 0, "shuffle_train": true, "drop_last": false,
  "learning_rate": 0.01, "learning_rate_arch": 0.001, "weight_decay": 1e-05,
  "betas": [0.9, 0.999], "eps": 1e-08, "scheduler": null,
  "embedding_dim": 16, "fcn_dims": [512, 256, 256, 64], "expert_num": 4,
  "m3oe_init": {"exp_d": 1, "exp_t": 1, "bal_d": 1, "bal_t": 1},
  "shadow_patience": 10, "shadow_breaks": false,
  "selection": {"metric": "validation_macro_auc", "rule": "strict_greater", "tie_break": "earliest_epoch"},
  "common_epochs": [9, 19, 29, 39, 49],
  "device": "cuda:0", "cudnn_deterministic": true, "cudnn_benchmark": false,
  "arch_update": {"batches_per_epoch": 1, "source": "persistent_iterator_over_shuffled_train_loader", "rng": "preserve_rng_state"},
  "expected_train_counts": {"0": 172782, "1": 1743898, "2": 89452},
  "expected_val_counts": {"0": 2692, "1": 246232, "2": 1843},
  "expected_test_counts": {"0": 2613, "1": 246284, "2": 1870},
  "expected_param_total": {"B0": 1614258, "B1": 1614264, "B2": 1614261, "B3": 1614258, "B4": 1614264},
  "b3_table_sha256": "182edcbe382257dedfc725e7f0939e90edff68569dca889a1d10b918fc5a9957",
  "hpost_enabled": false
}
```

### 5.2 `research_R2/configs/movielens_d0_large_m3oe.json`

与 5.1 逐字段相同，仅以下字段不同：

```json
{
  "dataset": "movielens", "setting": "d0_large",
  "dataset_path": "D:\\CODE\\phase0-3\\m3oe\\examples\\data\\ml-1m\\ml-1m.csv",
  "dataset_sha256": "f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4",
  "dataset_size_bytes": 75514352,
  "expected_train_counts": {"0": 150000, "1": 34500, "2": 3000},
  "expected_val_counts": {"0": 27111, "1": 35523, "2": 37387},
  "expected_test_counts": {"0": 27687, "1": 41434, "2": 30900},
  "expected_param_total": {"B0": 773474, "B1": 773480, "B2": 773477, "B3": 773474, "B4": 773480}
}
```

（MovieLens 各值来源与推导见 §11。）

### 5.3 arch optimizer 每 epoch 单 batch 的具体选择规则（从旧代码查明，逐字镜像）

来源 `stage_S1_run.py:696-733`（`_architecture_update`）：

- 维护一个**跨 epoch 持久的 iterator**，首次 arch 更新时由 `iter(self.train_loader)` 创建（train_loader 为 shuffle=True、batch 40960、drop_last=False 的 DataLoader）。
- 每 epoch 恰好 `next(architecture_iterator)` 取一个 batch；若 `StopIteration` 则重建 `iter(self.train_loader)` 再取一个。
- 整个 `next()` 调用包裹在 `preserve_rng_state()` 中：取 batch 消耗的 torch RNG 在取完后恢复，**不影响**主训练 shuffle 的 RNG 流。
- 因此 arch batch 序列 = 对 train 集的一次独立 shuffle 顺序逐批消费、耗尽后重开新 shuffle（KuaiRand：49 batch/epoch → 第 0–48 epoch 消费完一遍，第 49 epoch 用新 shuffle 的第 1 批；MovieLens：5 batch/epoch → 每 5 epoch 重开）。loss 为同一 Sample BCE；arch step 前执行 `arch_optimizer.zero_grad(set_to_none=True)` 加 `model.zero_grad(set_to_none=True)`（清掉主循环期间在 arch 参数上累积的梯度），backward 后仅 arch optimizer step。
- 新臂的 q/delta 属于 main optimizer，arch step 不写它们；四 raw 不属于 main optimizer，main step 不写它们。每步（main 与 arch）梯度全部清零规则与旧代码一致。

### 5.4 数据装载

统一调用 `load_stage_s1_bundle(dataset, dataset_path, setting, seed)`（只读 import）。该函数内置 KuaiRand 期望 train 计数断言与 MovieLens 角色计数断言；R2 runner 须额外断言 config 中的 val/test 期望计数、split 指纹、encoder/scaler hash 全部匹配（首次执行时生成并记录指纹，与 §11.6 历史锚点核对；不一致即 G0 失败，排查，不临时更换 split）。

---

## 6. Required metrics/logging

逐 epoch（每 run）记录（Charter §6/§9 逐项落实）：

1. **训练目标**：train Sample BCE 的 batch 平均与全 epoch 样本加权两种口径；train 六格 LogLoss（epoch 末全量 train eval pass）。
2. **Validation**：六格 AUC、六格 LogLoss、macro AUC、macro LogLoss、W（见下）、worst-domain、worst-cell；每格 sample_count/positive_count/positive_rate。
3. **Global 系数**：四 raw scalar 原值与 sigmoid 值（aD/aT/bD/bT）。
4. **臂特有量**：
   - B1：6 个 q_dt、6 个 rho_dt 全表、rho 均值。
   - B2：3 个 q_d、3 个 rho_d（记录其被两任务共享的事实）、均值。
   - B3：固定表（常量，记录一次即可）。
   - B4：6 个 delta_dt 全表、delta 均值、中心化 delta（delta − 均值）、raw_bD 当前值、6 个有效 b_dt、每格 own 系数 aD·b_dt 与 other 总系数 aD·(1−b_dt)。
   - 全部臂：每格 own/other 完整系数快照。
5. **饱和标记**：任一 rho 或有效 b_dt < 0.01 或 > 0.99 时记录饱和标志；**只标记，不 clamp、不改 lr**。
6. **描述性量**：专家输出尺度（每 epoch 在 validation 第一个 batch、eval + no_grad + `preserve_rng_state` 下，记录 9 个专家输出的 L2 范数均值）；6 个共享门熵（同一 probe batch 上 softmax 熵的均值）。
7. **优化审计**：main/arch optimizer 实际 step 数、每 epoch 新增参数（q/delta）梯度 L2 范数与 update L2 范数、arch step 处四 raw 梯度范数、两 optimizer 分组审计（参数数、lr、betas、wd）。
8. **曝光配对**：逐 batch fingerprint（train 与 arch 分开记录）、batch_order_hash、architecture_batch_order_hash、逐 batch domain_counts。
9. **选模与 shadow**：每 epoch macro AUC、selected epoch、shadow 首次触发 epoch（未触发记 null）、边界 censoring 说明。
10. **状态/身份**：initial_parameter_state_hash、final_parameter_state_hash、checkpoint 文件 hash、数据指纹、runtime/硬件 manifest、code hash。
11. **派生汇总口径**：macro = (1/6)·Σ六格 AUC；task 宏 = 每任务三域均值；domain 宏 = 每域两任务均值；`W = (1/2)·Σ_d p_train_d · Σ_t AUC_dt`（p_train 取实际验明的原 train 份额固定值：KuaiRand {0.08612693481784847, 0.8692837759429589, 0.044589289239192635}，MovieLens {0.8, 0.184, 0.016}），称"训练域份额加权逐格 AUC"，不称混合总体 AUC；额外 pooled task AUC 如报告须从原始预测/标签实际拼接重算、单列；LogLoss clip eps = LOGLOSS_EPS（见 §12.10 登记）。

所有入门指标（D/T/mean/guard）必须使用**原始未四舍五入**的存储值；展示表的 rounded 值不得入门（Review P3）。

---

## 7. Expected output files

```
research_R2/
├── R2_Design_Specification.md                # 本文件
├── src/
│   ├── ao04_v12_model.py
│   ├── ao04_v12_protocol.py
│   ├── ao04_v12_runner.py
│   ├── ao04_v12_decision.py
│   └── ao04_v12_hpost.py                     # 仅开关 stub
├── configs/
│   ├── kuairand_natural_m3oe.json
│   └── movielens_d0_large_m3oe.json
├── init/
│   └── <dataset>/seed<seed>/
│       ├── init_state.pt                     # B0 未训练 state_dict（原 tensor）
│       ├── init_rng.pt                       # torch CPU/CUDA + numpy + python RNG
│       └── init_manifest.json                # initial_state_hash、逐 tensor hash、数据指纹
├── runs/
│   └── <dataset>/<setting>/m3oe/<arm>/seed<seed>/attempt1/
│       ├── config.json                       # 生效 config 快照 + code/runtime/数据指纹
│       ├── metrics_epoch.csv                 # §6 逐 epoch 全量
│       ├── batch_trace.csv                   # 逐 batch fingerprint/domain_counts/objective
│       ├── fusion_log.csv                    # §6.3–6.6 臂特有量与系数轨迹
│       ├── selected_checkpoint.pth
│       ├── final_checkpoint.pth
│       ├── common_epoch_{9,19,29,39,49}.pth
│       ├── last_resume.pt                    # 每 epoch 覆盖写；含完整 resume state
│       ├── optimizer_state.pth               # 训练结束时 main optimizer
│       ├── architecture_optimizer_state.pth
│       ├── rng_state.pth
│       ├── selection.json                    # selected epoch、shadow、逐 epoch macro AUC（全精度）
│       ├── parameter_manifest.json
│       ├── run_manifest.json                 # 无 test 性能字段；含曝光 hash、状态机状态
│       ├── metrics_test.csv                  # 仅 final_test_evaluator 解封后写
│       ├── predictions_test.npz              # 同上
│       └── stdout.log
├── decisions/
│   ├── r2a_decision_kuairand.json            # 不可变 decision manifest
│   └── r2a_decision_movielens.json
├── locks/
│   └── global_test_unlock.json               # 全局 selection/config/阶段状态锁定后才生成
├── reviews/                                  # Implementation Report / Independent Review / Baseline 重建验收
└── reports/                                  # R2-A Decision Report、R2_Experimental_Analysis.md、R2_Global_Conclusion.md
```

run 状态机：PLANNED / RUNNING / COMPLETE / FAILED / INVALID / INCOMPLETE，全部留痕于 run_manifest.json；失败不隐去。目录非空拒绝覆盖（沿用旧 runner 的防覆盖语义）。

---

## 8. 初始化与配对协议

### 8.1 配对初始化（每 dataset × seed 一次）

在任何 B0 训练前执行 `paired_init_phase`：`setup_seed(seed)`（torch/cuda/numpy/python + cudnn flags）→ 加载 bundle → 构造 B0 模型 → 立即保存：`init_state.pt`（state_dict CPU 副本）、`init_rng.pt`（torch CPU、torch CUDA all、numpy `np.random.get_state()`、python `random.getstate()`）、`init_manifest.json`（`initial_state_hash`、逐 tensor SHA256、bundle 全部指纹）。MovieLens 的 train 子集在该 seed 下随之固定（`RandomState(10*seed+physical_domain)`，不消耗全局 RNG）。

### 8.2 各臂初始化

B1/B2/B3/B4 的 runner：同 seed 重新执行 `setup_seed` → 加载 bundle → 构造本臂模型（构造函数与 B0 消耗完全相同的 RNG 序列；新增参数一律 zeros/常数 buffer，零 RNG 消耗）→ 从 `init_state.pt` **strict 加载全部原 tensor**（新参数/buffer 键除外）→ 逐 tensor 断言与 init_manifest 中 B0 的 hash 相等 → 断言新参数全 0 / B3 buffer 等于固定表。**禁止**从任何已训练（selected/final/last_resume）状态初始化干预臂。

### 8.3 参数总数验收

| 数据集 | B0 | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|---:|
| KuaiRand | 1,614,258 | 1,614,264 | 1,614,261 | 1,614,258 | 1,614,264 |
| MovieLens | 773,474 | 773,480 | 773,477 | 773,474 | 773,480 |

KuaiRand 三值来自 Charter §3 与历史 manifest 双重确认。MovieLens B0=773,474 来自 Architecture Audit §5 逐模块清点与历史 `run_manifest.json`（`model_parameter_manifest.total`，seed2022/2023 一致）双重确认；B1/B4=B0+6、B2=B0+3 为代数必然（仅新增 6/3 个标量参数）。B3 的 6 元素 buffer **不计入** trainable 参数总数（named_parameters 不含 buffer），但出现在 state_dict 中。验收方法：runner 构造后调用 `parameter_manifest(model)["total"]` 与 config `expected_param_total[arm]` 精确相等断言；同时断言 `architecture_names` 恰为 §4.1 的四名集合（按名称与 `id()` 双重核验），两 optimizer 参数集交集为空、并集覆盖全部 trainable 参数（按 `id()`）。

### 8.4 曝光与状态配对

- 逐 batch fingerprint（`batch_fingerprint`）train/arch 分开记录；同 (dataset, seed) 五臂训练完成后，validity 检查要求 `batch_order_hash` 与 `architecture_batch_order_hash` 五臂两两相等。
- 任何会破坏 RNG 流一致性的行为禁止：新增参数随机初始化、训练路径上的额外 forward（probe 计算必须包 `preserve_rng_state`）、DataLoader 加 generator 等。
- `last_resume.pt` 每 epoch 覆盖写：model state、两 optimizer state_dict、torch CPU/CUDA RNG、numpy/python RNG、epoch 与 batch 计数、arch iterator 已消耗批数、已累积 fingerprint 列表、config hash。selected checkpoint 只含模型 state（可附该 epoch 两 optimizer/RNG 快照作审计），**禁止**用 selected 接续训练。

---

## 9. G0/G1 hard gates 可执行检查清单

### G0（正式 B0 训练前，两数据集各自独立执行；任一失败即停，修复后重查）

- [ ] G0-1 授权与审核：R2 执行授权书存在；本设计 PASS；隔离实现的独立 Implementation Review PASS 或全部 blocking fixes 闭环。
- [ ] G0-2 数据身份：CSV 字节 SHA256 与大小匹配 config；row/label/feature/split/encoder（及 MovieLens scaler）指纹生成并与 §11.6 历史锚点核对；期望计数（train/val/test 逐域）全部相等。
- [ ] G0-3 runtime manifest：Python/Torch/CUDA/cuDNN/NumPy/pandas/sklearn 版本与 §3.2 锁定值一致；硬件 manifest 记录完整；两数据集同机同环境。
- [ ] G0-4 构造与路由：模型输出形状 [B,2]；六格路由 task-major `i=t*3+d` 断言；j/k 顺序 {0:(1,2),1:(0,2),2:(0,1)} 断言；B0 forward 与原 `StageS1M3OEAdapter` 同 state 逐位一致（同一输入批输出 `torch.equal`）。
- [ ] G0-5 前向等价：B1/B2/B4 零初始化、eval 模式、与 B0 同一份初始化 state，在 all-domain 合成输入与真实 train 首批上比较输出：FP32 max_abs ≤ 1e-6 且相对误差 ≤ 1e-5；原 tensor 逐个 `torch.equal`。B4 六格 b_dt 初值全等 bD；B4 域内 own+other 系数和=1、乘 aD 后和=aD、两 other 系数相等。B1/B2 每格 other 两系数之和等于 B0 的 (1−bD)（代数恒等，断言数值成立）。
- [ ] G0-6 梯度归属：非退化 all-domain 合成输入下，B1/B2 的 q、B4 的 delta 得到有限非零梯度；一次 main step 实际更新 q/delta（前后值变化）且不写四 raw；一次 arch step 实际更新四 raw 且不写 q/delta；B4 在 delta=0 时 raw_bD 梯度与 B0 可比（同输入同 state 下相等到数值容差 1e-6）。B3 buffer 无梯度、不在任何 optimizer。
- [ ] G0-7 optimizer 分组：按参数 `id()` 断言——arch 组恰为四 raw 显式集合；main 组 = 其余全部 trainable（含新参数）；交集空、并集全覆盖；新参数名不匹配 `is_architecture_parameter`。
- [ ] G0-8 参数总数：§8.3 表逐臂断言。
- [ ] G0-9 test tripwire：静态——训练入口调用链无 test 依赖（grep 清单通过）；动态——synthetic spy/mock fixture 证明 train_val_loop + state_writer 全流程 test 预测/指标调用 = 0，且 test 数组缺失时 state_writer 仍能完整写出（含 last_resume）。
- [ ] G0-10 配对初始化：paired_init_phase 工件齐全；两臂（如 B0/B1）同 seed 构造后原 tensor 逐位相等、batch fingerprint 序列在 ≤3 batch smoke 中相等。
- [ ] G0-11 smoke：每启用臂 × 每数据集，seed 2022，≤3 batch 数值 smoke（非绩效），finite 通过；smoke 日志单列；**禁止**用 smoke AUC 调设计。两数据集合计 ≤30 batch。
- [ ] G0-12 B3 表身份：常量数组与 §2 逐字符一致；无空格 UTF-8 JSON SHA256 = `182edcbe…a9957`。

### G1（每数据集的正式 B1/B2 训练前；任一失败 → 该队列 INVALID，返回 Reviewer）

- [ ] G1-1 该数据集两份 B0（seed 2022/2023）完整 50 epoch 完成，状态 COMPLETE；validation 日志完整；selected 为 earliest val max（用 `select_checkpoint_epoch` 等价规则独立复算一致）。
- [ ] G1-2 完整 checkpoint/optimizer/RNG/init/曝光工件齐全且 hash 记录。
- [ ] G1-3 reload replay：strict load 每份 B0 的 selected checkpoint，在同一 validation 集重放：预测 max_abs ≤ 1e-6，六格及 macro 的 AUC/LogLoss 差 ≤ 1e-6。
- [ ] G1-4 配对核验：B0 两 seed 的 init 工件与 exposure hash 记录完备，可供 B1/B2 复制。
- [ ] G1-5 test tripwire 计数仍为 0。

---

## 10. R2-A 队列与判决门实现规格

### 10.1 固定队列（每数据集独立实例化；顺序固定，不得因性能改序/跳过）

| 顺序 | 子阶段 | 臂 | seed | 最大 epochs |
|---:|---|---|---:|---:|
| 1 | A | B0 | 2022 | 50 |
| 2 | A | B0 | 2023 | 50 |
| — | G1 检查 | — | — | — |
| 3 | A | B1 | 2022 | 50 |
| 4 | A | B1 | 2023 | 50 |
| 5 | A | B2 | 2022 | 50 |
| 6 | A | B2 | 2023 | 50 |

**并发规格（runner 最多 2 个 python 进程）**：采用固定的双臂波次调度——Wave 1: (kuairand,B0,2022) ∥ (movielens,B0,2022)；Wave 2: (kuairand,B0,2023) ∥ (movielens,B0,2023)；随后两数据集各自 G1；Wave 3: (kuairand,B1,2022) ∥ (movielens,B1,2022)；Wave 4: (…,B1,2023)×2；Wave 5: (…,B2,2022)×2；Wave 6: (…,B2,2023)×2。任一 run 失败/INVALID：暂停其所在数据集队列并排查；不得为凑数据继续该数据集的门。每 run 是独立进程、独立 seed，并发不改变任何单 run 的确定性。G1 未通过的数据集不得启动 B1/B2；另一数据集也不得以其资源补位（预算按数据集独立）。

R2-B（仅当该数据集 decision = YES）：顺序 7 B3/2022、8 B3/2023、9 B4/2022、10 B4/2023，同样两数据集波次并发；执行前再次核验已冻结实现/config 未按 A 结果改动；若 NO，该数据集这四项标 NOT SCHEDULED，余额不挪用。

### 10.2 D_s / T_s 定义

在每数据集上，用各臂 earliest validation max selected 的原始未舍入 macro AUC：

```
D_s = M_val(B1, s) − M_val(B0, s)
T_s = M_val(B1, s) − M_val(B2, s)        s ∈ {2022, 2023}
```

阈值判定严格按 `>0`、`>=`、`<` 原文执行，不加任何 tie tolerance（Hpost 的 1e-6 并列规则只属于其自身 grid 选择，与本门无关，且 Hpost 本阶段 disabled）。

### 10.3 Guards 全集（全部针对 B1−B0 配对差，两 seed）

记 Δ 为 (B1 − B0) 的配对差（按同 seed 配对）：

- G-a：每任务宏 AUC（任务内三域均值）差，两 seed 均值 ≥ −0.001（两个任务分别判）。
- G-b：W 差，两 seed 均值 ≥ −0.001。
- G-c：六格每格 AUC 差，两 seed 均值 ≥ −0.003（六格分别判）。
- G-d：六格每格 AUC 差，任一 seed 单格 ≥ −0.005（12 个格×seed 分别判）。
- G-e：六格每格 LogLoss 差，两 seed 均值 ≤ +0.010。
- G-f：macro LogLoss 差，两 seed 均值 ≤ +0.005。

追加门只检查 B1−B0 的 guards。B2（及若执行的 B3/B4）同样报告各自 guard 数值，但不作为它们的 validity 条件；B2 若自查信号，须对 B2−B0 用同一套门槛与 guards 单独判定并单列报告。

### 10.4 追加 YES 的完整布尔条件（每数据集独立判定）

```
valid       = 该数据集 6 份 A run 全部 COMPLETE、完整 50 epoch、配对证据（init/曝光 hash）齐备
guards_pass = G-a..G-f 全部通过（原始未舍入值）
practical   = (D_2022 > 0) and (D_2023 > 0) and (mean(D) >= 0.003) and guards_pass
exploratory = (D_2022 > 0) and (D_2023 > 0) and (0.001 <= mean(D) < 0.003) and guards_pass
signal      = practical or exploratory
task_gain   = (T_2022 > 0) and (T_2023 > 0) and (mean(T) >= 0.001)
DECISION    = YES  iff  valid and signal and task_gain   else NO
```

FAILED/INCOMPLETE/INVALID 不进入门（缺失不是科学阴性）。nonfinite 或泄漏等 validity 异常 → 按统一规则停队列、标 INVALID。

### 10.5 Decision manifest（不可变字段，两数据集各一份）

字段：`dataset`、`created_at`、六份 run 的 `formal_run_key` + run_manifest SHA256 + selected epoch + selected validation macro AUC（全精度）、D_2022/D_2023/mean(D)、T_2022/T_2023/mean(T)（全精度）、signal 分类（practical/exploratory/none）、G-a..G-f 每项的全部原始数值与 pass/fail、validity 摘要、DECISION（YES/NO）、理由串、预算余额（attempt/epoch/resume 计数）、`test_prediction_calls_at_decision`（必须为 0）、决策代码 SHA256。**无 test 字段**；写出后只追加只读副本，不得修改。主分析硬前提：两数据集的 R2-A 均 valid 完成（决策本身可为 NO，但 6 份 run 必须有效）。

### 10.6 最终统一 test 解封

两数据集全部已计划 run 完成、全部 selected/final/common 身份与 config、两份 decision manifest（或终止状态）锁定后，生成 `locks/global_test_unlock.json`；`final_test_evaluator` 仅此之后对 test 评价一次（selected 主端点，final/common 描述；不得据 test 择优、不得重开 B、不改阈值/seed/数据集/主法）。旧 test 历史曝光如实披露，不称全新 holdout。

---

## 11. MovieLens 专章（用户授权范围扩展的实例化方案）

### 11.1 域与任务定义

- 物理域（按 age 分组，`stage_S1_data.py:416-418`）：d0 = `ages_1_18`（age∈{1,18}），d1 = `age_25`（age=25），d2 = `ages_35_45_50_56`（age∈{35,45,50,56}）。
- 任务：t0 = `click` = `1[rating>3]`，t1 = `like` = `1[rating>4]`（构造标签，非 stored；`stage_S1_data.py:421-422`）。
- setting 固定 `d0_large`：角色映射 d0→L、d1→M、d2→S；ROLE_SIZES = {L:150000, M:34500, S:3000}。

### 11.2 数据文件与指纹

- 文件：`D:\CODE\phase0-3\m3oe\examples\data\ml-1m\ml-1m.csv`（`stage_S1_data.py:32` 的 DEFAULT_MOVIELENS_PATH + `ml-1m.csv`）。已核实存在。
- 字节身份：75,514,352 bytes，SHA256 = `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`（Baseline Protocol §3 审计值与历史 manifest `dataset_file_hash` 一致）。
- 对比锚点：KuaiRand 文件 `D:\CODE\data\kuairand\processed\kuai_rand.csv`，416,370,411 bytes，SHA256 = `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`。已核实存在。

### 11.3 split、encoder 与 rotation 规则

- 原始行序全局 80/10/10 split-first；**无 filter 步骤**（D-ML-6）。总行数 1,000,209 → base train 800,167 / val 100,021 / test 100,021。
- 特征：7 sparse（`user_id, movie_id, gender, occupation, zip, cate_id, domain_indicator`）× embedding 16 + 1 dense（`age`，按 base-train min/max 做 min-max 缩放）→ 输入维 113。`cate_id` 取 `genres` 首个 `|` 分段。
- encoder：词表仅 base train 拟合（seed 无关），OOV=0；domain_indicator 词表固定 3。
- 角色轮转抽样：仅作用于 train。每域 d 从 base-train 该域池中按 `np.random.RandomState(10*seed + d).permutation(pool)` 取前 requested 个（d0:150000，d1:34500，d2:3000）。val/test 不抽样，两 seed 完全相同。
- 标签、encoder、scaler、split 全部由 `load_stage_s1_bundle("movielens", ..., "d0_large", seed)` 逐字实现；runner 不得改写。

### 11.4 期望计数（seed 无关；G0 验收值）

| split | d0 | d1 | d2 | 合计 |
|---|---:|---:|---:|---:|
| train | 150,000 | 34,500 | 3,000 | 187,500 |
| validation | 27,111 | 35,523 | 37,387 | 100,021 |
| test | 27,687 | 41,434 | 30,900 | 100,021 |

train 份额 p_train = {0.8, 0.184, 0.016}（W 指标用）。计数来源：历史 S1 manifest（`experiments/stage_S1_raw_results/movielens/d0_large/m3oe/seed2022/joint/run_manifest.json` 的 cell_stats）+ ROLE_SIZES 代数必然；执行时以代码重算值为准，不等即 G0 失败排查。每 epoch 5 batch（⌈187500/40960⌉=5），每 run 250 main steps + 50 arch steps。

### 11.5 模型与参数量

D=3/T=2/N=4、FCN [512,256,256,64]、expert_num 4 与 KuaiRand 完全一致；仅输入维不同（113 vs 544）。B0/B3 = 773,474；B1/B4 = 773,480；B2 = 773,477（§8.3）。历史 selected epoch（仅上下文，不作验收线）：seed2022→6、seed2023→6。

### 11.6 指纹锚点（历史值，供 G0 核对；以执行时重算为准，不符即排查）

- encoder_hash = `05ab58ef6b9baf0589a3a476789006d8cc1b5c10eba53f1934c0c42d755f9e6b`（seed2022/2023 一致）；scaler_hash = `b73c4a08a40b2059f892c72edc6f659baf2acdd430b6ca1a50068bcd0dde1c1e`。
- base split fingerprints：train `3524e8f3…`，validation `c2910f69…`，test `c29ac98b…`（全值见历史 manifest `base_split_fingerprints`）。
- KuaiRand 对应锚点：encoder_hash = `58ff742e14eabc33ed001ad6ec2e4f74c6be375ab4b2a68028277b7b9bb3e33a`；split fingerprints train `d2855a27…` / val `0e8f49a9…` / test `6f5c25cf…`。

### 11.7 与 KuaiRand 协议的差异及理由汇总

完整偏差登记表见 §1.1（D-ML-1..D-ML-8）。协议层镜像关系：相同五臂定义、相同优化协议（lr/wd/betas/eps/batch/epoch/选模/arch 单 batch 规则）、相同判决门阈值与 guards、相同 G0/G1、相同配对协议、相同指标口径（W 用各自 train 份额）。无法镜像的本质差异仅有：构造标签（数据本身无 stored 标签，沿用 S1 代码定义）、dense age 特征（数据属性）、train 角色抽样依赖 seed（S1 既定协议）、无 filter 步骤（域全集无越界值）。这些均逐字继承 `stage_S1_data.py`，不构成本设计引入的新自由度。

---

## 12. Potential implementation pitfalls

1. **P1：旧 runner 的 test 耦合（最大风险）**。`stage_S1_run.py:794` 在 `fit()` 尾部调用 `_evaluate_deferred_checkpoints()`（809-810 行用 test_loader 预测）；`write_outputs()`（885-888 行）在 test 数组缺失时抛错，896-901 行才写 checkpoint。任何"包一层旧 S1Runner"的实现都会在 A 决策前触碰 test 或根本无法做 val-only 保存。必须按 §4.3 拆成四个组件；train 路径的调用链里不得出现 test。reviewer 用静态 grep + spy 动态 tripwire 验收。
2. **B4 双 sigmoid**。`self._weight_bal_d()` 返回的是 `sigmoid(deep_weights)`；B4 若写成 `sigmoid(self._weight_bal_d() + delta)` 即双 sigmoid，初始值 sigmoid(sigmoid(1))≈0.674 而非 bD，前向等价门会失败——但等价门若实现不严谨（只查形状不查数值）就会漏过。必须取 `_weight_bal_d.deep_weights` 原值。同时注意 B4 的 raw_bD 仍在 arch optimizer 中，梯度经 `sigmoid(raw+delta)` 回传是预期行为。
3. **参数误归 arch/main**。`is_architecture_parameter` 按 `_weight_`/`architecture.` 前缀判名；新参数若命名 `_weight_rho` 之类会被静默归进 arch Adam（每 epoch 一步而非每 batch 一步），且 arch 组不再恰为四 raw。必须按 §4.1 命名 + §9 G0-7 的 id() 集合断言（不能仅猜名字）。反向错误（四 raw 之一被漏出 arch 组）同样由"arch 组恰为显式四名集合"断言兜住。
4. **B3 buffer 不进 optimizer、但要进 state_dict**。用 `register_buffer`（而非 `nn.Parameter`）实现固定表；若用 Parameter 且未排除，会多 6 个 trainable、参数总数验收失败；若用普通属性，strict load selected checkpoint 会丢表。buffer 不得出现在任何 optimizer 的 param_groups。
5. **B4 共同 shift 冗余的披露义务**。`raw+c, delta−c` 给出同一 b_dt：6 个新 trainable ≠ 6 个新可辨识函数方向（最多 5 个）。实现侧不得加 sum-zero 约束/冻结 raw/改正则；日志侧必须记录 delta 均值与中心化 delta（§6.4），否则后续分析无法区分"global 快速调节"与"格异质性"两种作用。
6. **舍入值不得入门**。D/T/mean/guard 一律用全精度存储值；展示 CSV/表格的 rounded 值禁止作为门的输入。decision 模块应直接从全精度 metrics 存储（selection.json / metrics_epoch.csv 的全精度字段）读取。
7. **曝光顺序配对**。arch batch 的 `next()` 必须包 `preserve_rng_state`（否则主 shuffle 序列被污染，五臂曝光失配）；新增参数禁止随机初始化（zeros/常数不消耗 RNG）；probe 计算（专家尺度/门熵）必须 eval + no_grad + `preserve_rng_state`。验收：同 (dataset, seed) 五臂 batch_order_hash 两两相等。注意 MovieLens train 子集随 seed 变化是协议内行为，不影响同 seed 跨臂配对。
8. **Resume state 完整性**。`last_resume.pt` 必须含两 optimizer state、全部 RNG、epoch/batch 位置、arch iterator 进度；用 selected checkpoint 续训是协议违规（selected 只用于评价）。resume 次数每 run ≤1、每数据集队列 ≤2。
9. **epoch 末全量 train eval pass 不可省略**。旧 `_train_epoch` 尾部用 shuffle=False loader 全量过一遍 train 计算 train_cell_logloss；省略它日志口径与 S1 不一致（且它是 §6.1 的来源）。KuaiRand 每 epoch 因此多约 2M 样本前向，成本预估要计入。
10. **LogLoss clip epsilon 的合同-代码差异（登记，须 Reviewer 确认闭环）**。Charter §6 与 Baseline Protocol §5 文本写 "clip epsilon=1e-7"；但冻结指标模块 `stage_S1_metrics.py` 实际 `LOGLOSS_EPS = 1e-12`，且其 docstring 声明该值是 Stage S1 artifact 合同的一部分，历史两份 run 的 config 也记录 `logloss_epsilon: 1e-12`。本设计指定：**实现一律 import `stage_S1_metrics.LOGLOSS_EPS`（1e-12），不得另写常量**；理由：主端点与判决门全部基于 AUC，LogLoss guards 均为配对差（epsilon 在差值中基本抵消），且与历史工件口径一致。此差异必须在 Implementation Report 与 Reviewer 报告中显式登记闭环，不得静默选择。
11. **tau 递减不要"修复"**。`Weights.forward` 每次调用递减 tau；softmax_type=3 下与输出无关。改动它即改变 baseline。
12. **格索引与 j/k 顺序**。`i = t*3+d`（task-major）；j/k 固定 {0:(1,2),1:(0,2),2:(0,1)}；rho 乘 E_j。B1 写反 j/k 仍满足前向等价（rho=0.5 对称），是最隐蔽的错误——G0 必须加一项非对称探针：人为把 q 置为非零非均匀值（仅测试代码内，不进正式训练），验证各格系数与 §2 公式逐项一致。
13. **B3 不初始等价是特性**。B3 的固定表使初始前向与 B0 不同；不要把 B3 也拿去过 G0-5 的等价门（该门只适用 B1/B2/B4）。
14. **两数据集门的独立性**。每数据集独立判定 YES/NO；不得把两数据集指标 pooled 后过一次门，不得用一个数据集的 YES 授权另一数据集的 R2-B。主分析需两数据集 R2-A 均 valid。

---

## 附录 A：本设计引用的代码事实速查（均已实际读码核实）

- `StageS1M3OEAdapter.forward`：`experiments/stage_S1_models.py:142-193`（设备安全装配、task-major 格路由）。
- 原 M3OE 构造器与 `Weights`（softmax_type=3 sigmoid、tau 递减）：`m3oe/MDMTRec/models/multi_domain/model.py:12-48, 75-136`。
- Sample loss：`stage_S1_run.py:114-120`；两 optimizer 分组：`stage_S1_run.py:619-636`；arch 更新规则：`stage_S1_run.py:696-733`；选模：`stage_S1_run.py:735-796`（严格 >、tie 最早、`select_checkpoint_epoch` 复算断言）；test 耦合点：`stage_S1_run.py:794, 798-864, 874-905`。
- 数据：`stage_S1_data.py`（KuaiRand `_load_kuairand_base` 499-623；MovieLens `_load_movielens_base` 407-496；rotation `role_reversal_sample` 91-122）。
- 指标：`stage_S1_metrics.py`（tie-aware AUC、LOGLOSS_EPS=1e-12）。
- 协议常量：`stage_S1_protocol.py`（FORMAL_PROTOCOL、ROLE_SIZES/ROLE_MAPS、COMMON_EPOCHS=(9,19,29,39,49)、EPOCHS=50）。
- 工具：`stage_E1_instrumentation.py`（batch_fingerprint 786、preserve_rng_state 1404、rng_state_digest 1417、write_json/write_rows_csv 1447/1459）。
- 历史 manifest 锚点：`experiments/stage_S1_raw_results/{kuairand/natural,movielens/d0_large}/m3oe/seed{2022,2023}/joint/run_manifest.json`。

## 附录 B：留给 Coder 的交付要求

Coder 按本设计一次实现全部五臂与条件门（含 B3/B4，即使 B 阶段未必启动——定义与实现须在 A 结果前冻结并过审），输出 Implementation Report：数学逐条对照、逐 tensor 参数分组表（含 id 集合断言结果）、与旧代码的 diff 说明、G0 全部自检结果。Runner 只记录原始结果；性能解释归主 Agent；Reviewer 未 PASS 不得正式训练。
