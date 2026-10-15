# Stage S6-R2 — Algorithm Specification（供 coder 实现）

- 设计依据：`promt/S6_R2.md`（必须先读）。
- 本阶段为**回顾性只读分析 + evaluation-only 推理**。**禁止**任何训练/优化步；**禁止**修改任何已有文件（包括 `experiments/stage_S1_*.py`、`stage_S6_*`、`m3oe/**`、已有 raw/outputs 工件）。只允许**新建**脚本与输出文件。
- 所有结论性判定（selection robustness gate、GO/STOP、H1–H6 评级）由主 Agent 完成；本阶段产物只负责提供全部所需数字。

---

## 1. Experiment objective

判定 KuaiRand 上 DH（相对 F）在 canonical validation-selected checkpoint 上的非目标域 test 增益（≈ +0.0068，4/4 seeds 正）是最终选中解的稳健性质，还是特定 checkpoint 选择规则 / 训练时序对齐的产物。分解来源：(A) 同时刻策略差异；(B) 选择时序差异；(C) 参数分区差异；(D) 选择目标依赖。MovieLens 仅作负对照（重复 Part A、B）。

## 2. 已核实的工件事实（实现前必须再自检）

### 2.1 目录与文件

- KuaiRand：`experiments/stage_S6_raw_results/{F,DH}/seed{2022,2023,2024,2025}/joint/`（HT/T 存在但本阶段不用）。
- MovieLens：`experiments/stage_S6ML_raw_results/{F,DH}/seed{...}/joint/`，结构相同。
- 每个 run 目录含：`metrics_epoch.csv`、`metrics_test.csv`、`metrics_summary.csv`、`selected_checkpoint.pth`、`final_checkpoint.pth`、`common_epoch_{9,19,29,39,49}.pth`、`parameter_manifest.json`、`gradient_partition.json`、`predictions_test.npz`、`config.json`、`run_manifest.json`。

### 2.2 metrics_epoch.csv schema（两数据集相同，已核实）

- 每 run 300 行 = 50 epochs（`epoch` ∈ 0..49）× 6 cells；每 epoch 6 行。
- 关键列：`epoch, domain(0/1/2), task(0/1), cell(0..5), validation_sample_count, validation_auc, validation_logloss, validation_macro_auc, target_weighted_validation_auc, worst_domain_validation_auc, worst_cell_validation_auc, selected_checkpoint(0/1), common_epoch(0/1)`。
- cell 编号恒等式：`cell == task*3 + domain`；cell 标签 `d{domain}t{task}`（沿用 `s6r_common.cell_label`）。
- epoch 级聚合列（`validation_macro_auc` 等）在同一 epoch 的 6 行中重复。
- `validation_sample_count` 每 cell 跨 epoch 恒定（已核实全部 16 个 F/DH run）→ R4 权重可得。
- **没有** per-epoch validation predictions，只有聚合指标。

### 2.3 metrics_test.csv / predictions_test.npz

- test 评估仅存在于 7 个 checkpoint：`selected, final, common_epoch_{9,19,29,39,49}`（`stage_epoch` 列给出对应 epoch；final=49）。
- `metrics_test.csv` 每 checkpoint 6 行（per-cell `auc`、`logloss`、`test_sample_count`、`domain_share` 等）。
- `predictions_test.npz` keys：`row_ids, physical_domain, labels(N,2), pred_selected, pred_final, pred_common_epoch_{9,19,29,39,49}` → test 指标可在不加载模型的情况下从 npz 复算（用 `stage_S1_metrics.safe_auc`），并应与 `metrics_test.csv` 一致（断言容差 1e-9）。

### 2.4 checkpoint 与参数分区（Part D 可行性：可行）

- `selected_checkpoint.pth` 等为**纯 state_dict**（138 个 tensor 的 flat dict，key 与 `parameter_manifest.json.names` 完全一致，已核实；F 与 DH 同 seed 的 key 集合与 shape 完全一致）。
- `parameter_manifest.json`：`architecture_names`（4 个 `_weight_*` 标量参数）、`shared_names`（60 tensors / 661,848+…=661,968 params，含 `embedding.*, shared_weight, shared_bias, skip_conn.*, star_mlp.*, expert.*`）、`private_names`（74 tensors / 952,286 params，含 `slot_*, gate.*, domain_expert.*, task_expert.*, tower.*`）。三者完备且互斥（union == names）。
- **注意分歧**（必须在报告中声明）：`gradient_partition.json` 的 C 组 = shared ∪ `domain_expert.*`（20 个 tensor），P 组 = private \ `domain_expert.*`。本阶段 Part D **以 `parameter_manifest.json` 的 shared/architecture/private 语义分区为准**（与 S6_R2 §7 措辞一致）；gradient_partition 版本仅作可选敏感性附注。
- 分区定义跨全部 16 run 一致（两数据集各 8 个 F/DH run 的 `partition_hash` 唯一；manifest 已抽查一致；实现时对用到的每个 run 断言一致）。

### 2.5 模型与评估入口（evaluation-only 推理路径）

- 模型构建：`experiments/stage_S1_models.py::build_model(schema, domain_num=3, task_num=2, backbone='m3oe', expert_num=4, fcn_dims=(512,256,256,64))`（内部 `StageS1M3OEAdapter`）。config.json 中 `expert_num=4, fcn_dims=[512,256,256,64]` 与默认一致。
- 数据加载：`experiments/stage_S1_data.py::load_stage_s1_bundle(dataset, dataset_path, setting, seed)`。split 由 seed 决定（`np.random.RandomState(seed*10+domain)` 等），**必须使用对应 run 的 seed 与 setting**：
  - kuairand：`setting='natural'`，默认路径 `D:\CODE\data\kuairand\processed`（**仓库外**，已确认存在 `kuai_rand.csv`）。
  - movielens：`setting='d2_large'`（从各 run `config.json` 读取，不要硬编码），默认路径 `m3oe/examples/data/ml-1m/ml-1m.csv`（存在）。
- 评估：DataLoader(batch_size=40960, shuffle=False)，`model.eval()` + `torch.no_grad()`，前向确定性（无 dropout）。指标用 `experiments/stage_S1_metrics.py::safe_auc / safe_logloss / cell_metrics`。
- 环境：`venv/Scripts/python.exe`，torch 2.9.0+cu126，CUDA 可用（RTX 4060 Laptop）。推理量小（KuaiRand test 250,767 行），CPU 亦可接受，device 从 config 读取但允许 CPU fallback。
- **只允许 import 这些已有模块，禁止修改。**

### 2.6 target / 非目标定义与 canonical 选择规则

- KuaiRand：`target_domain=1`（tab1, size_role=L）；MovieLens：`target_domain=2`（以 `s6r_common.DATASETS` 与各 run `config.json.target_local_domain` 为准）。
- target cells = `{(d*,0),(d*,1)}`；non-target cells = 其余 4 格；non-target domains = 其余 2 域。
- canonical R0 = 对 `validation_macro_auc`（六格无权重均值）取 max，**并列取最早 epoch**（`stage_S1_run.py:157 select_checkpoint_epoch`）。
- 已核实 canonical selected epoch：KR F={30,47,41,24}、DH={40,30,48,27}（seeds 2022–2025）；ML F={7,8,8,12}、DH={8,13,8,7}。**全部不在保存网格 {9,19,29,39,49} 上**（除巧合外）。

### 2.7 关键工件缺口（已确认）

- **任意 epoch 的模型权重未保存**——只有 {9,19,29,39,49} ∪ {canonical selected} 有 checkpoint，test 指标也只有这些点。重选 epoch 落在其他位置时，**无法**获得该 epoch 的 test 表现（权重不存在，evaluation-only 推理也无米下锅；插值权重不是合法操作）。
- 因此 Part A/B/C 的 test 层覆盖必然不完整 → 处理规则见 §4.3 与 §9。

---

## 3. Mathematical definition

记号：arm ∈ {F, DH}，seed ∈ {2022,2023,2024,2025}，epoch e ∈ [0,49]，层 L ∈ {val, test}。`a^L_{arm,seed}(d,t,e)` 为 cell (d,t) 在 epoch e 的 AUC（val 层来自 metrics_epoch.csv 的 `validation_auc`；test 层仅在保存点可得）。

聚合量（每层每 epoch 由 6 个 cell AUC 现算，不信任预计算列，但与预计算列断言一致，容差 1e-9）：

- macro：`M(e) = mean_{d,t} a(d,t,e)`（六格无权重均值）
- target mean：`T(e) = mean_t a(d*,t,e)`
- non-target mean：`N(e) = mean_{d≠d*,t} a(d,t,e)`
- domain mean：`D_d(e) = mean_t a(d,t,e)`；worst-domain：`WD(e) = min_d D_d(e)`；worst-cell：`WC(e) = min_{d,t} a(d,t,e)`

### 3.1 Part A 选择规则（全部仅用 val 层；并列一律取最早 epoch）

- R0（canonical）：`argmax_e M^val(e)`
- R1（target）：`argmax_e T^val(e)`
- R2（non-target）：`argmax_e N^val(e)`
- R3（worst-domain）：`argmax_e WD^val(e)`
- R4（sample-weighted）：`argmax_e Σ_d w_d · D_d^val(e)`，其中 `w_d = (n_{d,0}+n_{d,1}) / Σ_{d',t} n_{d',t}`，`n` 为 `validation_sample_count`（每 cell 跨 epoch 恒定，取 epoch 0 的值并断言恒定）。**R4 可行**（权重信息可得；禁止发明其他权重）。

每个 (dataset, rule, arm, seed) 独立重选，得 `e*_{rule,arm,seed}`。

### 3.2 Δ 与 gate 统计量

对规则 R 与指标 X ∈ {N, T, M}（paired by seed）：

`ΔX_R(seed) = X^test_DH(e*_R,DH,seed) − X^test_F(e*_R,F,seed)`（test 不可得时用 val 层对应量，并打 `layer` 标记）

每个 (dataset, rule, X, layer) 报告：4-seed mean、SD（样本 SD，ddof=1）、逐 seed 值、positive seeds/4、|Δ|≥0.003 计数、|Δ|≥0.005 计数。

### 3.3 Part B 代数分解

e_F、e_DH = 该 seed 的 canonical R0 selected epoch。对指标 X（六格 macro、target mean、non-target mean、d0 mean、d2 mean、六个单格 cell AUC，共 10 个）与层 L：

- `ΔSel = X_DH(e_DH) − X_F(e_F)`
- F-anchor：`PolicyAtF = X_DH(e_F) − X_F(e_F)`；`TimingAlongDH = X_DH(e_DH) − X_DH(e_F)`；断言 `|ΔSel − (PolicyAtF+TimingAlongDH)| < 1e-9`
- DH-anchor：`PolicyAtDH = X_DH(e_DH) − X_F(e_DH)`；`TimingAlongF = X_F(e_DH) − X_F(e_F)`；同上断言。

**层覆盖规则**：val 层全做（所有 epoch 可得）。test 层仅当 e_F、e_DH 均在保存 checkpoint 集合时可做——已核实 KuaiRand 与 MovieLens 的 8 个 (arm,seed) 组合中 selected epoch 均不在 {9,19,29,39,49}，故 **test 层完整分解不可行**，输出中明确标注 `Existing artifacts are insufficient for test-layer cross-epoch decomposition`，仅提供：两端点 test ΔSel（selected checkpoint 已有）+ 保存网格点 {9,19,29,39,49} 上的 test Δ(e)=X_DH(e)−X_F(e) 轨迹（同时刻策略差异的 test 层近似证据）。

### 3.4 Part C 窗口统计量

对 canonical selected epoch e*（每 arm/seed），窗口 W = {e*−2,…,e*+2} ∩ [0,49]（边界截断并打 flag）。令 `V(e)=M^val(e)`：

- `drop_k = mean(V(e*) − V(e))` 对偏移 ±k 的可用侧取均值（k=1,2）
- `max_drop = max_{e∈W, e≠e*} (V(e*) − V(e))`（应为非负，若窗口内存在更大值说明并列/数据异常，记录并按最早 tie-break 说明）
- 分类（预注册操作定义）：`broad plateau` := max_drop < 0.001；`narrow spike` := drop_1 ≥ 0.002 且 drop_2 ≥ 0.004；其余 `moderate`。
- test 层：仅对窗口中落在保存网格的 epoch 报告 test N/T/M（一般为空，照常标注）。

### 3.5 Part D 参数交换（仅 KuaiRand，4 seeds）

分区（来自该 run `parameter_manifest.json`）：`S`=shared_names（60）、`A`=architecture_names（4）、`P`=private_names（74）。实现前断言：三者互斥、union == state_dict keys、F/DH 同 seed keys/shapes 一致。

对每 seed 构造并 evaluation-only 评估（test 层为主，同时记录 val 层 M/T/N 供参照）：

- `F_full`、`DH_full`（**管道自校验锚点**：其六格 test AUC 必须与 metrics_test.csv 的 selected 行一致，|Δ|<1e-4，否则中止并报管道错误）
- `C_hybrid` = DH(S∪A) + F(P)
- `D_hybrid` = F(S∪A) + DH(P)
- 可选 `E_hybrid` = DH(S) + F(A) + F(P)；`F_hybrid` = F(S) + DH(A) + F(P)

报告每个 hybrid 的六格 AUC、T、N、M；相对 F_full / DH_full 的增益保留率：`ρ_C = (N(C_hybrid)−N(F_full)) / (N(DH_full)−N(F_full))` 等（分母近 0 时标注）。

## 4. Independent variables / Controlled variables

**Independent**：选择规则 R0–R4（Part A）；anchor 方向（F/DH）×指标（Part B）；窗口偏移（Part C）；hybrid 类型（Part D）；数据集（kuairand / movielens）。

**Controlled（冻结，不得更改）**：arm ∈ {F,DH}；seeds {2022–2025}；模型结构/超参（来自各 run config.json）；数据 split（load_stage_s1_bundle 按 run seed/setting 复现）；评估协议（batch_size=40960、shuffle=False、eval()、no_grad、`stage_S1_metrics` 实现）；分区定义（parameter_manifest.json）；tie-break（最早 epoch）；epoch 范围 [0,49]。

### 4.3 重选 epoch 的 test 评估规则（核心设计决策）

对每个 (rule, arm, seed) 的重选 epoch e*：

1. e* == canonical selected epoch → 复用 `checkpoint=='selected'` 的 test 行；
2. e* ∈ {9,19,29,39} → 复用对应 `common_epoch_*` 行；
3. e* == 49 → 用 `common_epoch_49` 行（同时记录 `final` 行并断言二者每格 |ΔAUC|<1e-6 作完整性检查）；
4. 否则 → **test 不可得**：该行 test 列填 NA，`test_available=0`，**禁止**用邻近 checkpoint 顶替、禁止插值。

补充分析 **A-grid**（明确标注为敏感性变体，非主规则）：把选择域限制在保存 checkpoint epoch 集 `G(arm,seed) = {9,19,29,39,49} ∪ {e_R0(arm,seed)}` 内重选（同一规则、同一 tie-break），使每个 (rule,arm,seed) 都有 test 数字。主分析仍是全 epoch 的 **A-full**；A-grid 只用于给主 Agent 提供 test 层完整证据。两个变体都输出。

## 5. Required code changes（全部为新建文件）

新建目录 `experiments/stage_S6_R2/`，内含：

| 脚本 | 职责 |
|---|---|
| `s6r2_common.py` | 数据集 spec（复用/对齐 `stage_S6_reanalysis/s6r_common.py` 的 target_domain、cell_label、tie-break 语义，但不修改它；可 import 或复制最小常量，优先 import）；metrics_epoch/test 加载；cell→聚合量现算与一致性断言；Δ/gate 统计函数 |
| `s6r2_partA_selection.py` | R0–R4 重选（A-full + A-grid），两数据集，输出 val/test 指标表与 gate 汇总 |
| `s6r2_partB_decomposition.py` | 两种 anchor 分解 + 恒等式数值断言 + test 层可得性标注 |
| `s6r2_partC_window.py` | 窗口统计与 plateau/moderate/spike 分类 |
| `s6r2_partD_swap.py` | 仅 KuaiRand；加载数据与模型、hybrid 构造、evaluation-only 推理、自校验锚点 |
| `s6r2_report.py` | 汇总上述 CSV → 生成 `experiments/stage_S6_R2_selected_solution_analysis.md`（数据表 + 待主 Agent 填写的 verdict 占位节） |

每个脚本支持 `--dataset {kuairand,movielens}`（Part D 固定 kuairand）。允许只读 import：`stage_S1_data`、`stage_S1_models`、`stage_S1_metrics`、`stage_S6_reanalysis/s6r_common`。

## 6. Required configuration changes

无。不得新增/修改任何训练配置；不得改 `config.json`。推理 device 默认 `cuda:0`（可用时），允许 `--device cpu` 覆盖；该选择只影响速度，不改变结果（浮点差异用 1e-4 容差锚点兜底）。

## 7. Required metrics/logging

- 每脚本 stdout 进度 + 断言失败即非零退出；所有断言写入 `s6r2_integrity_report.json`（条目：macro/WD/WC 现算 vs 预计算列一致性（容差 1e-9）；npz 复算 vs metrics_test.csv 一致性（1e-9）；R0 重选 epoch == `selected_checkpoint` flag epoch（必须逐 run 相等）；final vs common_epoch_49 一致性（1e-6）；分区完备/互斥；F/DH state_dict keys/shapes 一致；Part D 锚点复现误差）。
- 所有 AUC 差值保留全精度（csv float repr），展示层再四舍五入。

## 8. Expected output files

新建目录 `experiments/stage_S6_R2_raw_results/`：

- `s6r2_partA_epochs_{dataset}.csv`：rule × arm × seed → selected_epoch、val 目标值、是否与 canonical 一致、test_available。
- `s6r2_partA_val_metrics_{dataset}.csv` / `s6r2_partA_test_metrics_{dataset}.csv`：rule × arm × seed × {六格 AUC, T, N, M, WD, WC}（test 不可得行 NA + flag）。
- `s6r2_partA_grid_epochs_{dataset}.csv`、`s6r2_partA_grid_test_metrics_{dataset}.csv`（A-grid 变体）。
- `s6r2_partA_gate_summary_{dataset}.csv`：rule × metric(N/T/M) × layer(val/test/test_grid) → mean、SD、4 个 per-seed Δ、positive/4、|Δ|≥0.003 计数、|Δ|≥0.005 计数。
- `s6r2_partB_decomposition_{dataset}.csv`：seed × metric（10 个）× layer × anchor(F/DH) → ΔSel、Policy、Timing、恒等式残差、availability 标注。
- `s6r2_partC_window_{dataset}.csv`：arm × seed → e*、窗口 5 点 val M、drop_1、drop_2、max_drop、分类、test 可得窗口点指标。
- `s6r2_partD_swap_kuairand.csv`：seed × hybrid（F_full/DH_full/C/D[/E/F]）× {六格 AUC, T, N, M}（test 层为主，附 val 层 M/T/N）+ 增益保留率 ρ。
- `s6r2_integrity_report.json`。
- `experiments/stage_S6_R2_selected_solution_analysis.md`：按 S6_R2 §12 节目录生成，数据节自动填表，executive verdict / H1–H6 / GO-STOP 节留占位符（`[MAIN-AGENT: ...]`）。

## 9. Potential implementation pitfalls（必须逐条处理）

1. **重选 epoch 无 checkpoint**：见 §4.3；这是本阶段最大的工件限制，NA 必须显式标注，禁止静默替换。A-grid 变体只是敏感性补充，不得冒充主规则。
2. **test 不得参与选 epoch**：选择函数只接收 val 层序列；代码评审点——选择逻辑与 test 数据加载在函数层面隔离。
3. **R4 权重**：只用 `validation_sample_count`（已核实恒定），断言恒定后再取；禁止用 train share 或 `domain_share` 列替代（如需对照可在附注中给出 train-share 变体，但主结果只用 val share）。
4. **MovieLens 差异**：`setting='d2_large'`（从 config.json 读，勿硬编码）、target_domain=2、selected epoch 普遍很早（7–13）；其余结构（3 域 × 2 任务 = 6 格、50 epochs、列名）与 KuaiRand 相同，已核实。
5. **分区完备/互斥校验**：Part D 前断言（见 §3.5）；并声明 manifest 与 gradient_partition 在 `domain_expert.*` 归属上的分歧，主结果用 manifest。
6. **predictions_test.npz 仅对应 7 个已保存 checkpoint**：selected/final/common 的 test 数字可直接复用 metrics_test.csv（或 npz 复算交叉验证）；**其余 epoch 无任何 test 依据**。Part D 的 hybrid 推理是新产生的 test 数字，必须用 §3.5 的 F_full/DH_full 锚点复现来证明管道等价。
7. **tie-break 一致性**：所有规则统一最早 epoch；canonical 复现断言（R0 重选 == selected flag epoch）必须逐 run 通过，否则说明理解有误，中止。
8. **聚合量现算**：T/N/M/WD/WC 一律从 6 个 per-cell AUC 现算并与预计算列断言一致（`target_weighted_validation_auc` 列语义不同，禁止当作 R1 使用）。
9. **epoch 编号**：metrics_epoch 的 epoch ∈ [0,49]；metrics_test 的 `stage_epoch` 与之同坐标；final==49 与 common_epoch_49 是两次评估行，按 §4.3-3 处理。
10. **数据路径**：KuaiRand 数据在仓库外 `D:\CODE\data\kuairand\processed`；Part D 脚本若找不到数据应报错退出而非跳过。
11. **数值容差**：同机同库复现应几乎精确；锚点容差 1e-4（防 GPU/CPU 浮点差），恒等式与 CSV 内部一致性用 1e-9。

## 10. 给 coder 的实现规格（伪代码级）

### s6r2_common.py

```
SEEDS=[2022,2023,2024,2025]; ARMS=["F","DH"]; EPOCHS=range(50)
GRID={9,19,29,39,49}
SPEC={kuairand: root=experiments/stage_S6_raw_results, target=1, setting from config,
      movielens: root=experiments/stage_S6ML_raw_results, target=2}
load_epoch_table(root,arm,seed) -> df(epoch,cell,domain,task,validation_auc,validation_sample_count)
  assert 300 rows; assert cell==task*3+domain; assert val count 恒定
agg(df_epoch) -> dict(M,T,N,WD,WC,D0,D1,D2,cells[6])  # 由 6 格现算
  assert |M - validation_macro_auc| < 1e-9 等
select_epoch(series: {epoch: obj}) -> argmax, 并列取最小 epoch
test_lookup(root,arm,seed,epoch) -> dict|None  # 按 §4.3 映射到 checkpoint 行
gate_stats(deltas: list[4]) -> dict(mean, sd(ddof=1), per_seed, pos_count, ge_003, ge_005)
```

### s6r2_partA_selection.py

```
for dataset in [kuairand, movielens]:
  for arm, seed:
    val = {e: agg(load_epoch_table(...)[e]) for e in EPOCHS}
    rules = {R0: M, R1: T, R2: N, R3: WD, R4: Σ_d w_d·D_d}  # w 来自 val counts
    for R in rules:
      e_full = select_epoch({e: rules[R](val[e])})           # A-full
      e_grid = select_epoch({e: rules[R](val[e]) for e in GRID|{e_R0}})  # A-grid
      assert R == R0 ⇒ e_full == selected_flag epoch
      写 epochs 行；val 指标行（总有）；test 指标行（test_lookup 或 NA）；grid 行同理
  汇总 gate：对 (R, X∈{N,T,M}, layer∈{val,test_full,test_grid}) 计算 Δ 与 gate_stats
```

### s6r2_partB_decomposition.py

```
for dataset, seed:
  e_F, e_DH = canonical R0 epochs（从 Part A 输出或重算并断言一致）
  for layer in [val, test]:
    for X in [M,T,N,D0,D2,6 cells]:
      try: 取 X_F(e_F), X_DH(e_DH), X_DH(e_F), X_F(e_DH)   # test 层取不到 → None
      若齐全: 计算两种分解, assert 恒等式 < 1e-9, 写行
      否则: 写 availability="insufficient" 行
  附: test 层网格轨迹 Δ(e)=X_DH(e)−X_F(e), e∈GRID（test_lookup 必有）
```

### s6r2_partC_window.py

```
for dataset, arm, seed:
  e* = canonical epoch; W = clip([e*-2..e*+2], 0, 49)
  drops vs V(e*)=M^val(e*); drop_1, drop_2, max_drop; 分类按 §3.4
  test: 对 W∩(GRID∪{e*}) 用 test_lookup 补 test N/T/M（一般仅 e* 本身）
```

### s6r2_partD_swap.py（仅 kuairand）

```
bundle = load_stage_s1_bundle("kuairand", None, cfg["setting"], seed)  # 每 seed
model = build_model(bundle.schema, 3, 2, "m3oe")  # 参数同 config
sdF = load selected F; sdDH = load selected DH
assert keys/shapes/分区校验（§3.5）
def evaluate(sd): model.load_state_dict(sd); model.eval(); no_grad;
  遍历 test_loader(shuffle=False) → per-cell safe_auc → 六格+T+N+M
  同法对 val_loader 得 val M/T/N
anchor: evaluate(sdF), evaluate(sdDH) 与 metrics_test.csv selected 行对齐 (<1e-4)，否则退出码≠0
C = {k: sdDH[k] if k∈S∪A else sdF[k]}; D = 反向; [E,F 可选]
逐 hybrid evaluate → 写 s6r2_partD_swap_kuairand.csv + 保留率 ρ
```

### s6r2_report.py

读取全部 CSV/JSON → 生成 markdown（节结构按 S6_R2 §12），表内数字含 mean±SD 与逐 seed 明细；占位节：`[MAIN-AGENT] verdict / H1–H6 evidence / GO-STOP`。

---

## 11. 设计决策摘要（供主 Agent 快速核对）

1. **重选 epoch 的 test 评估**：只复用 7 个已保存 checkpoint 的 test 行；其余显式 NA；另提供网格受限敏感性变体 A-grid 使每规则都有 test 数字。任意 epoch 的权重不存在，evaluation-only 推理无法补救。
2. **Part B**：val 层完整分解（全部 10 指标 × 2 anchors × 4 seeds × 2 数据集，恒等式断言）；test 层分解判定为 `Existing artifacts are insufficient`（所有 canonical epoch 均不在保存网格），以两端点 Δ + 网格同时刻 Δ 轨迹替代。
3. **Part C**：val 层窗口完整可做；test 层基本只有 e* 本身一点。
4. **Part D 可行**：selected state_dict 齐全、分区清单齐全、推理路径现成（import stage_S1_*），含 F_full/DH_full 锚点自校验（<1e-4）。分区以 `parameter_manifest.json` 为准，`domain_expert.*` 归属分歧已声明。
5. **R4 可行**：用 `validation_sample_count` 的域份额（已核实跨 epoch 恒定）。
6. **MovieLens 结构**：同为 3 域 × 2 任务 = 6 格、50 epoch、同 schema；差异在 `setting='d2_large'` 与 target_domain=2。
