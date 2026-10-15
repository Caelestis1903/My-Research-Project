# Stage S8 设计文档 — Optimizer/Trajectory Final Kill Test

设计者：Agent 1（designer，k3-256k max）。本文件是 Stage S8 的唯一权威实现规格。
科研定义以 `promt/grad.md`（S8 节）为准；本文件不修改任何科研标准，只定义实现。
Coder 必须仅凭本文件 + 现有代码完成实现，不得自行发明任何算法、权重、采样或调度。**禁止修改 S1–S7 的任何已有文件；S8 全部为新增文件。**

---

## 0. 可行性调查结论（必须先读，回答任务书 a–g）

以下为对现有代码/数据资产的核实结果，是全部设计决策的事实基础。

### a) S6 checkpoint 里有什么（已核实）

S6/S6ML 的 F arm run 目录（`experiments/stage_S6_raw_results/F/seed{N}/joint/`、`stage_S6ML_raw_results/F/seed{N}/joint/`）包含：

- **完整 model 参数**（`state_dict`，含 shared/private/architecture 全部参数）：`selected_checkpoint.pth`、`final_checkpoint.pth`、`common_epoch_{9,19,29,39,49}.pth`（保存点见 `stage_S1_run.py:763-764, 866-871`，`_state_cpu(self.model)`）。
- **shared + architecture 的 Adam moments**（`exp_avg`/`exp_avg_sq`/`step`）：`moment_state_epoch{0,1,9,19,29,39,49}.pth`。dump 实现在 `stage_S6_run.py:166-187`（`_dump_moment_state`），**只遍历 `(GROUP_SHARED, self.optimizer)` 与 `(GROUP_ARCH, self.arch_optimizer)`，明确不包含 GROUP_PRIVATE**——这证实了 S7-B 报告"S6 未 dump private moments"。
- **最终时刻的完整 main optimizer state**（`optimizer_state.pth`，`stage_S1_run.py:905`，`torch.save(self.optimizer.state_dict())`）：含 private 的 m/v，但**只有 epoch 49 终点**，任何中间 anchor 都不可得。
- 每个 anchor 的 private moments：**不存在于任何产物中**。
- MovieLens 每 seed 实际 selected epoch = 7/8/8/12，KuaiRand F = 30/47/41/24（`stage_S6_analysis.md` §5.3，已抽查 manifest 确认：KR 2022 selected_epoch=30、ML 2022 selected_epoch=7）。这些 selected epoch **不在** moment-dump epoch 集合 {0,1,9,19,29,39,49} 内，连 shared moments 在 selected anchor 都没有 dump。

**结论：现有 checkpoint 无法支撑 S8。任何 anchor 上的 private moments 都缺失；selected anchor 上连 shared moments 都缺失。**

### b) 必须重训 F base run（方案已锁定）

需要 **Phase 1：带完整状态 dump 的 F-arm 重训**（每 dataset × seed 一个 run，共 8 个）。重训 runner 克隆 `S6Runner`/`S6MLRunner`，仅新增只读 dump。可比性由以下机制保证（全部已存在于代码中，S7-B 已验证）：

- 同 seed：`setup_seed(seed)` + `build_model(..., seed=seed)` 的 RNG 链固定初始化（`initial_parameter_state_hash` 逐 seed pin，S6 protocol `INITIAL_PARAMETER_SHA256`）；
- 同 batch 序列：DataLoader 顺序只由 seed 决定（`shuffle=True, num_workers=0, drop_last=False`，无 sampler 改动），逐 batch `batch_fingerprint` 可与原 S6 run 逐一比对（hard gate）；
- dump 是纯只读操作、不消耗 RNG（`torch.save` + `.detach().cpu().clone()`），插入后不改变训练轨迹；
- 验证：重训 run 的逐 epoch validation macro AUC、selected epoch、final 状态与原 S6 canonical run 对比（见 §7 V-BASE）。容忍度设为 1e-6 是因为 CUDA backward 原子操作存在 ~1e-11 量级非确定性（S7-B V0 实测 step-0 delta rel diff ~1e-11），跨 250/2450 步会累积到 ~1e-9–1e-7，科学上无关但需如实声明。

### c) F/T/DH 臂的精确定义与实现位置（已核实）

2×2 因子（history source × current source）定义在 `stage_S6_protocol.py:54-56`：

```
ARM_HISTORY = {"F":"FULL","T":"TARGET","HT":"TARGET","DH":"FULL"}
ARM_CURRENT = {"F":"FULL","T":"FULL","HT":"FULL","DH":"TARGET"}
```

实现：`stage_S6_run.py:189-358`（`_controlled_step`，KuaiRand）与 `stage_S6ML_run.py` 同构函数。语义（S8 必须逐字复现）：

- 每 batch 一次 `optimizer.step()`。`gf, gt, gp = gradients_one_forward(losses["full"], losses["target"], ...)`（`stage_S4_gradient_controls.py:85-97`）单次前向算出；`adam_deltas`（同文件 100-141）从 pre-step 状态只读计算反事实 Δ_F/Δ_T/Δ_P。
- **F**：shared 安装 g_F，`step()`，无覆写。applied = Δ_F，moments ← g_F。
- **T**：shared 安装 g_T，`step()`，无覆写。applied = Δ_T，moments ← g_T。
- **DH**：shared 安装 g_F，`step()`（moments ← g_F），然后覆写 `θ_shared ← θ_before + Δ_T`（`stage_S6_run.py:250-254`）。
- **private 参数在所有臂永远用 FULL loss 的梯度 g_P 更新，从不覆写**（`stage_S6_run.py:235-237, 266`）。
- architecture 参数（4 个 `deep_weights`）每 epoch 一次、经同一 dispatch（`kind="architecture"`，names=GROUP_ARCH，private=[]），由独立 `arch_optimizer`（Adam lr=0.001）更新。
- HT 在 S8 **不运行**（grad.md §3.1：T 与 DH 足够；HT 仅在 main 明确要求时才追加，本规格不含 HT）。

### d) 训练成本（已核实实测值）

- S6 F run 全程 50 epochs 实测：KuaiRand `wall_seconds=7210.7`（≈2.0 h）；MovieLens `wall_seconds=804.9`（≈13 min）。Phase B HT/DH 同量级（KR 6.5–7.9k s，ML ~562 s）。
- S7-B continuation（5 steps × 2 branches，含全部 geometry + validation 诊断）实测：KR anchor39 = 114 s（≈11.4 s/step/branch），ML = 28–41 s（≈3–4 s/step/branch）。
- S8 单 run = 1 进程 × 3 branches × K steps。运行矩阵与时长估算见 §10：**核心矩阵（2-epoch horizon）串行总计约 16–22 h**；可选 5-epoch 扩展另需 ~15–17 h。

### e) validation 评估与 best 判定（已核实）

- 源 run 每个 epoch 末（arch update 之后）在 `preserve_rng_state()` 下用 `_cell_records(bundle, model, val_loader, device)`（`stage_S1_run.py:293-330`，`model.eval()`，shuffle=False）计算 6 cells 的 validation AUC/LogLoss，写入 `metrics_epoch.csv`（含 `validation_macro_auc` 列）。
- checkpoint selection：`stage_S1_run.py:157-164 select_checkpoint_epoch` + `fit():753-757`——**unweighted six-cell validation macro-AUC 最大，严格 `>` 实现并列取最早**；test 在 selection 冻结后才评估。
- S8  continuation 中每步记录同口径 validation 指标（复用 S7-B 的 `_full_validation_metrics`，`stage_S7B_run.py:639-688`，eval 模式、无 RNG 消耗、逐 cell AUC/LogLoss + macro/target_domain/non_target_weighted 聚合），并在 anchor step 0 与源 run 的 `metrics_epoch.csv` 对拍（V 检查），保证口径一致。
- **F 原始 best 参照**：`BestAUC_F,original` = 原 S6/S6ML canonical F run manifest 的 `selected_validation_macro_auc`（冻结产物，只读引用，collector 从 `run_manifest.json` 读取并与 S8-BASE 重训值交叉核验）。

### f) effective Adam update 测量（已核实可实现）

`stage_S4_gradient_controls.py:100-132 adam_delta` 精确复现 torch.optim.Adam（coupled weight decay：`effective = g + wd·θ`；`δ = −lr/(1−β1^t) · m_next / (√(v_next/(1−β2^t)) + ε)`，pre-step 状态只读），并有 `validate_adam_delta_against_clone` 的 clone 对拍门。S8 每步记录：observed applied delta（θ_after−θ_before，经 `assert_vector_match` 与公式对拍）、Δ_F/Δ_T 反事实、norm、cosine、以及与 validation gradient 的 alignment（cos/dot）。raw gradient 的 norm/cosine/U_k/H 沿用 S7-B instrumentation（`diagnostics_at_theta`，`stage_S7B_run.py:766-864`）。raw 与 effective 的 alignment 差异直接回答 grad.md §10.2 的核心问题。

### g) batch replay 与 RNG（S7-B 机制可直接推广到多 epoch）

S7-B 已验证 `3×(anchor+1)` RNG 消耗公式（`stage_S7B_run.py:275-367`）：源 `fit()` 每 epoch 消耗恰好 3 次全局 CPU RNG draw（main shuffled iter 2 次：RandomSampler randperm + DataLoader `_base_seed`；train_records shuffle=False iter 1 次；arch update 与 validation 在 `preserve_rng_state` 内净消耗 0）。**多 epoch 推广**：continuation 从 epoch `a+1` 开始，逐 epoch 重复 [iter(train_loader)（2 draws）→ 消耗全部 bpe 个 batch → iter(train_records_loader)（1 draw）→ arch update（净 0）→ validation（净 0）]，与源逐位同构；每 epoch 的 main/arch batch fingerprint 与源 `batch_trace.csv`（kind=train / kind=architecture）逐位 hard-gate。arch batch 的重放需要按"生成代"重建（见 §4.3 的 `arch_batch_for_epoch` 规格）。

---

## 1. Experiment objective

（引用 grad.md §2，逐字）

> **S6 中观察到的 DH/T late-stage benefit，是否可以在完整 optimizer state、严格 same-state 的条件下，通过较长 continuation 稳定改变 optimization trajectory，并最终提高"最佳 validation performance"；还是它只能够改善已经严重过拟合的 late checkpoint？**

必须区分两种结局（grad.md §2）：

- **情况 A（仅 late rescue）**：T/DH 从 late anchor continuation 后优于 F 同点，但 continuation best 仍 ≤ F 原始 validation-selected best。→ 只能说明 manipulation 可部分救援已严重过拟合的模型。
- **情况 B（best-performance improvement）**：T/DH continuation 的 best validation AUC 稳定（≥3/4 seeds）超过 F 原始 best，且非单 seed / 单 checkpoint 驱动。→ 只有这种情况才允许继续讨论 optimizer/trajectory mechanism 的方法价值。

S8 是 kill test：primary endpoint 失败即 STOP（grad.md §15 STOP-1/2/3）。本规格不预设任一方向。

## 2. Mathematical definition

### 2.1 Branch 语义（复用 S6，见 §0.c；无新算法）

history h ∈ {FULL, TARGET} = 提交进 persistent Adam m/v/τ 的梯度来源；current c ∈ {FULL, TARGET} = 实际施加到 shared/architecture 参数的 Adam delta 来源。private 参数无条件使用 g_F（full loss 的 private 梯度）。

| branch | (h, c) | shared 更新 | shared moments | private |
|---|---|---|---|---|
| F | (FULL, FULL) | θ ← θ + Δ_F | ← g_F | Adam(g_F) |
| T | (TARGET, TARGET) | θ ← θ + Δ_T | ← g_T | Adam(g_F) |
| DH | (FULL, TARGET) | θ ← θ + Δ_T（先 step() 提交 g_F 再覆写） | ← g_F | Adam(g_F) |

其中 Δ_X = `adam_delta`(θ, g_X, pre-step state) = −η·(m̂ 更新后)/(√v̂ 更新后+ε)，含 coupled weight decay（β1=0.9, β2=0.999, η_main=0.01, η_arch=0.001, wd=1e-5）。

### 2.2 Anchor 定义

anchor a = 源 F 训练在 epoch a 末（arch update 之后、validation 之前，与 `_dump_moment_state` 同一循环点）的完整状态快照。continuation 从 epoch a+1 的第一个 main batch 开始。

| anchor 类 | MovieLens（S6ML F） | KuaiRand（S6 F） |
|---|---|---|
| selected（per-seed 实际 validation-selected checkpoint） | 7 / 8 / 8 / 12（seeds 2022/2023/2024/2025） | 30 / 47 / 41 / 24 |
| middle | 19 | 19 |
| late | 39 | 39 |

selected anchor 逐 seed 从原 canonical manifest 读取并在 protocol 中 pin（见 §5），不允许统一成一个 epoch。

### 2.3 Horizon 定义（嵌套前缀，同一 run 内全部物化）

batches-per-epoch：MovieLens bpe=5，KuaiRand bpe=49（`stage_S7B_protocol.py:158`）。

| horizon | 步数 H（ML / KR） | 说明 |
|---|---|---|
| h_5steps | 5 / 5 | 与 S7-B 对照 |
| h_half | 3 / 25 | ⌈bpe/2⌉，0.5 epoch |
| h_1ep | 5 / 49 | 1 epoch（ML 与 h_5steps 重合，如实保留两行） |
| h_2ep | 10 / 98 | **core 矩阵的最大 horizon** |
| h_5ep | 25 / 245 | 可选扩展，仅当 a+5 ≤ 49（KR seed2023 selected anchor 47 排除） |

一个 run 训练到 H_max（core=2 epochs），所有更短 horizon 的 best 由轨迹前缀在 collect 阶段导出——**horizon 间共享完全相同的 batch 序列是构造性保证**。

### 2.4 Primary endpoint

对每个 (dataset, seed, anchor, branch, horizon h)：

```
BestAUC(branch, h) = max over continuation steps s ∈ {1..H(h)} of macro_auc(branch, s)
                     （严格 >，并列取最早 step；step 0 = anchor 状态不计入 primary，另列）
ΔBestAUC(branch, h) = BestAUC(branch, h) − BestAUC_F,original
BestAUC_F,original = 原 S6/S6ML canonical F run 的 selected_validation_macro_auc（冻结参照）
```

macro_auc = 六个 (domain × task) cell validation AUC 的无权均值——与全项目 checkpoint selection 同一口径。**禁止**用 `AUC@late − AUC_F@late` 替代。Secondary：target-domain AUC（target 域两 cell 简单均值）、worst-cell AUC、macro/target LogLoss、per-cell AUC；以及同 run 内 paired contrast `BestAUC_T/DH − BestAUC_F,continuation`（同 anchor 同 horizon）。

### 2.5 Effective Adam update（grad.md §10.2）

每步每 branch 记录：
- `delta_applied_norm`（observed ‖θ_t − θ_{t−1}‖，shared）、`delta_full_norm`/`delta_target_norm`（反事实）、`private_delta_norm`；
- `delta_cosine` = cos(Δ_F, Δ_T)；
- 在 val-gradient 采样步（§6.3）：`eff_align_full = cos(Δ_applied, g_val_full)`、`eff_align_target = cos(Δ_applied, g_val_target)` 及 dot 版本；同步记录 raw 版本 `cos(g_S, g_val_full)` 等，供 raw-vs-effective 直接对比；
- `m_norm_persistent`、`v_norm_persistent`（pre-step 状态范数）。

## 3. Independent variables / Controlled variables

**独立变量**（且仅有）：

| 变量 | 取值 |
|---|---|
| branch | F / T / DH（同进程、同 anchor 状态、同 batch 序列） |
| anchor | selected（per-seed）/ middle=19 / late=39 |
| horizon | §2.3（由同一轨迹的前缀导出，非独立运行变量） |
| dataset | kuairand/natural（target d1）/ movielens/d2_large（target d2） |
| seed | 2022/2023/2024/2025 |

**锁定项**（逐项核验，沿用 S6 §3.2 全部锁定）：dataset/setting/backbone（m3oe，MODEL_SOURCE_SHA256 pin）、target identity、objective=sample（SAMPLE_TABLE_SHA256 pin：KR `30c68c9b…`，ML `e0616863…`）、batch_size=40960、main lr=0.01、arch lr=0.001、betas=(0.9,0.999)、wd=1e-5、expert_num=4、fcn_dims=[512,256,256,64]、embedding_dim=16、num_workers=0、device=cuda:0、shadow_patience=10（Phase 1 重训用）、数据文件 hash、split/exposure、逐 seed 初始化 hash、optimizer grouping（shared/private/architecture 三组）。continuation 阶段无 scheduler（lr 恒定，与源一致）、无 early stopping（continuation 固定步数跑完）。

## 4. Required code changes

**只新增，不修改任何现有文件。** 共 6 个新文件 + 2 个冻结 queue manifest（orchestrator 生成）。

### 4.1 `experiments/stage_S8_protocol.py`（data-only，风格同 `stage_S7B_protocol.py`）

```text
STAGE = "S8"
SEEDS = (2022, 2023, 2024, 2025)          # 引用 S4_SEEDS 并 assert 相等
BRANCHES = ("F", "T", "DH")               # 无 HT
BRANCH_HISTORY = {"F":"FULL","T":"TARGET","DH":"FULL"}
BRANCH_CURRENT = {"F":"FULL","T":"TARGET","DH":"TARGET"}
BATCHES_PER_EPOCH = {"kuairand": 49, "movielens": 5}
DATASETS = {"kuairand":  {"setting":"natural",  "base_raw_root": experiments/stage_S8_base_raw_results 下 kuairand 子树,
                          "source_s6_dir": experiments/stage_S6_raw_results/F/seed{N}/joint},
            "movielens": {"setting":"d2_large", ..., "source_s6_dir": experiments/stage_S6ML_raw_results/F/seed{N}/joint}}
ANCHORS = {"movielens": {"selected": {2022:7, 2023:8, 2024:8, 2025:12}, "middle": 19, "late": 39},
           "kuairand":  {"selected": {2022:30, 2023:47, 2024:41, 2025:24}, "middle": 19, "late": 39}}
ANCHOR_CLASSES = ("selected", "middle", "late")
H_MAX_EPOCHS_CORE = 2                     # core 矩阵
H_MAX_EPOCHS_EXTENDED = 5                 # 可选扩展，仅 a+5<=49
HORIZON_STEPS(dataset) -> {"5steps":5, "half":ceil(bpe/2), "1ep":bpe, "2ep":2*bpe, "5ep":5*bpe}
FULL_STATE_DUMP_EPOCHS(dataset, seed) -> sorted({selected[seed], 19, 39})
阈值（沿用 S7-B）：V0_L_FULL_DIFF_ABS=1e-5, V0_DELTA_NORM_REL_DIFF=1e-3,
  V2_REVIEW_TRIPWIRE_L_FULL_DIFF=0.3, VBASE_MACRO_AUC_ABS=1e-6, S4_TOLERANCE_ABS/REL=1e-6/1e-4
路径：RAW_ROOT=experiments/stage_S8_raw_results, BASE_RAW_ROOT=experiments/stage_S8_base_raw_results,
  OUTPUT_ROOT=experiments/stage_S8_outputs
key_base(dataset, seed) = "s8base|sample|{dataset}|{setting}|m3oe|seed{seed}|F"
key_cont(dataset, seed, anchor_class) = "s8|{dataset}|seed{seed}|{anchor_class}{anchor_epoch}"
base_run_dir / cont_run_dir / full_state_filename(epoch)="full_state_epoch{e}.pth"
expected_shared_step(dataset, anchor) = bpe*(anchor+1)；expected_arch_step(anchor) = anchor+1
inventory(root) = S4 inventory + 六个 S8 源文件的 sha256（复用 S6 protocol 的 inventory 模式）
```

### 4.2 `experiments/stage_S8_base_run.py`（Phase 1：带完整状态 dump 的 F-arm 重训）

包含两个 thin subclass（克隆模式，同 S6ML 之于 S6）：

```python
class S8BaseRunnerKR(S6Runner): ...   # dataset=kuairand
class S8BaseRunnerML(S6MLRunner): ... # dataset=movielens
```

相对父类的**全部**差异：

1. `_validate_args`：五元组改为对应 dataset 的 `(...,"sample","joint")`；arm 锁死 `"F"`；`formal_run_key == proto.key_base(dataset, seed)`；其余 expected dict 逐项与 S6 相同（epoch=50、batch_size=40960 等）。
2. `__init__`：`args.arm="F"`、`args.fresh_role="fresh_s8base"`、inventory 改用 S8 inventory（25 项中 S6 四文件替换为 S8 六文件——**S6 文件仍在 inventory 中作为被 import 的依赖保留其 hash**，在 S4 inventory 基础上追加全部 S5/S6/S6ML/S8 相关源文件，具体集合由 coder 按 `stage_S6_protocol.inventory` 模式构造并写入 protocol 的 `S8_SOURCE_FILES`）。
3. **新增 `_dump_full_state(epoch)`**（核心新增，只读、零 RNG 消耗）：

```python
def _dump_full_state(self, epoch):
    if epoch not in FULL_STATE_DUMP_EPOCHS(self.args.dataset, int(self.args.seed)):
        return
    payload = {
        "epoch": int(epoch),
        "model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
        "main_optimizer": self.optimizer.state_dict(),      # 含 shared+private 全部 m/v/step
        "arch_optimizer": self.arch_optimizer.state_dict(),
        "optimizer_steps": int(self.optimizer_steps),
        "architecture_optimizer_steps": int(self.architecture_optimizer_steps),
        "rng_digest": rng_state_digest(),
    }
    torch.save(payload, os.path.join(self.run_dir, "full_state_epoch{}.pth".format(epoch)))
```

4. `_architecture_update`：先 `super()._architecture_update(...)`（父类内部已在同一点做 moment dump），随后调用 `self._dump_full_state(epoch)`——**dump 点 = arch update 之后、validation 之前，与源 moment dump 严格同一点**。
5. `write_outputs`：在 S6/S6ML 版本基础上改 stage="S8BASE"、condition=`s8base_F`、fresh_role/provenance_type="fresh_s8base"、`source_code_hash` 指向 `experiments/stage_S8_base_run.py`、manifest 增加 `full_state_dump_epochs` 与每个 dump 文件的 sha256（`full_state_dump_hashes`）。
6. `build_argparser` / `main()`：同 S6 模式；`--dataset {kuairand,movielens}` 选择 subclass；`os.makedirs(run_dir, exist_ok=False)`。

**除此之外函数体逐字继承**（`_controlled_step`、`_dump_moment_state`、fit 循环、checkpoint selection 零改动）。

### 4.3 `experiments/stage_S8_run.py`（Phase 2：三分支 continuation runner）

以 `stage_S7B_run.py` 为底稿克隆扩展。模块级列定义见 §6。核心函数规格：

1. **`replay_to_anchor(bundle, seed, anchor)`** → 返回 `(train_loader, train_records_loader)`：
   `setup_seed(seed)` → `build_model(...)`（仅为 RNG 链保真）→ 构造 `train_loader`（shuffle=True, batch 40960, num_workers=0, drop_last=False）与 `train_records_loader`（同数据 shuffle=False）→ 消耗 `3*(anchor+1)` 次 iter 创建（与 S7-B 同公式）→ 之后**流式**消费（不物化全部 batch；KR 2 epoch ≈ 98×40960 行，流式避免内存堆积）。

2. **`arch_batch_for_epoch(bundle, seed, epoch)`** → 重建源 run 在 epoch 的 architecture batch：
   arch iterator 按"代"创建：代 g = epoch // bpe，创建时机为源 epoch g*bpe 的 arch update 点，即全局 RNG 状态 = setup_seed+build_model 后消耗 `3*(g*bpe+1)` 次 draw。实现：`with preserve_rng_state():` 内 `setup_seed(seed)` → `build_model(...)` → 消耗 `3*(g*bpe+1)` 次 `iter(train_loader)` → 再建 1 个 iterator → skip `epoch % bpe` 个 batch → 取下一个。**注意 `preserve_rng_state` 内不允许调用 setup_seed 之外污染全局状态的顺序差异**——coder 须用 smoke 验证返回 batch 的 fingerprint 与 S8-BASE `batch_trace.csv` 中 `kind=="architecture"`、`epoch==epoch` 行逐位相等（hard gate，见 §7 V4）。（bpe=49 时 g 恒为 0（epoch≤48）或 1（epoch=49）；bpe=5 时 g=epoch//5。）

3. **`reconstruct_anchor_full(model, base_run_dir, anchor, device)`** → 替代 S7-B 的 `_build_adam_from_dump`：
   - `model.load_state_dict(torch.load(full_state)["model"], strict=True)`；
   - 新建 main Adam（shared+private 同序，lr/betas/wd 同前）与 arch Adam（lr=0.001），分别 `load_state_dict(payload["main_optimizer"/"arch_optimizer"])`；
   - 硬检查：shared step 全部 == `expected_shared_step`、arch step == `expected_arch_step`、**private 参数 100% 具有非空 state**（`private_state_count == len(private_names)`，这是 S8 相对 S7-B 的关键新增）；dump 文件 sha256 记入 audit。

4. **`branch_state_full(model, main_opt, arch_opt)`** → 三分支深拷贝：S7-B `branch_state`（`stage_S7B_run.py:512-566`）的直接推广，同时 rebinding 两个 optimizer；构造后对所有参数与两个 optimizer 的全部 state tensor 做跨分支 `torch.equal` 逐位断言（V3）。

5. **`controlled_continuation_step(model, optimizer, branch, batch, ...)`**：S7-B 同名函数推广到三分支语义（§2.1）：F 安装 g_F；T 安装 g_T（无覆写）；DH 安装 g_F + step() + 覆写 θ←θ_before+Δ_T。private 永远安装 g_P。每步 hard 断言（V1）：applied==Δ_applied（`assert_vector_match`）、private==Δ_P、None-gradient fail-closed、步骤边界 RNG digest 不变。

6. **主循环 `run_continuation(dataset, seed, anchor_class, out_dir, h_max_epochs=2)`**：
   - 加载 bundle（`load_stage_s1_bundle`）、构建模型、initial hash 对拍 S8-BASE manifest；
   - replay 到 anchor+1 起点；reconstruct anchor 完整状态；`branch_state_full` × 3；
   - **step 0（anchor 点）**：对每分支计算 validation metrics（step=0 行），并与 S8-BASE `metrics_epoch.csv` 的 epoch=anchor 行对拍（macro AUC，hard gate |Δ| ≤ 1e-6——同一点同一状态的 eval 必须一致）；
   - 对 e 从 anchor+1 到 anchor+h_max：
     - `iterator = iter(train_loader)`（2 draws）；
     - 逐步：取 batch → fingerprint 对拍 S8-BASE `batch_trace.csv`（kind=train, epoch=e, step=s）→ 每分支 diagnostics + controlled step → 每分支 validation metrics（逐步）→ val-gradient/geometry 诊断仅在 §6.3 采样步执行；
     - epoch 末：iterator 耗尽 → `iter(train_records_loader)`（1 draw）→ 每分支 arch update（`arch_batch_for_epoch` + 分支语义的 arch controlled step，preserve_rng_state）→ 每分支 post-arch validation（`kind="epoch_end_post_arch"` 行）→ F 分支 post-arch macro AUC 与 S8-BASE `metrics_epoch.csv` epoch=e 行对拍（V2，report-only + tripwire 1e-3）；
   - best-tracking：每分支对 `kind=="post_step"` 行按 macro AUC 严格 `>` 更新 best（并列取最早 step），内存保存 best state_dict，结束写 `continuation_best_{F,T,DH}.pth`；
   - 结束写 §8 全部产物。

7. smoke 模式（互斥，formal run 拒绝）：`--smoke-restore`（状态恢复完整性）、`--smoke-replay`（前 5 个 main batch + 第 1 个 arch batch 的 fingerprint）、`--smoke-onestep`（三分支 step-0 的 applied/private residual + 跨分支 step-0 pre-step 逐位相等）。

### 4.4 `experiments/stage_S8_orchestrator.py`

风格同 `stage_S7B_orchestrator.py`，但 **MAX_PROCESSES_PER_WAVE = 1（严格串行，单进程）**。两个子命令：

- `--phase base --task-index i`：task = (dataset, seed)，共 8 个；顺序：全部 MovieLens 先行（快，兼作管线 smoke），再 KuaiRand。每个 task 完成后立即运行 V-BASE 校验（§7），FAIL 即停止后续 wave 并上报。
- `--phase cont --task-index i`：task = (dataset, seed, anchor_class)，共 24 个；顺序：MovieLens 12 个先行，再 KuaiRand（seed-major, 再 anchor selected→middle→late）。
- `--phase cont5 --task-index i`（可选扩展）：仅 anchor+5≤49 的 task（ML 12 + KR 11 = 23 个）。
- 每个 task：launch 子进程 → `stage_S8_report.validate_run` → 追加 state JSON（append-only）。拒绝覆盖非空目录。

### 4.5 `experiments/stage_S8_report.py`

- `validate_base_run(dir)` / `validate_run(dir)`：产物存在性 + 行数基数 + audit 中全部 hard-gate 字段为真 + ABORTED 判 FAIL（同 S7-B `validate_run` 模式）；
- `collect()`：汇总为 §8 的聚合 CSV + `stage_S8_summary_manifest.json` + run report 骨架（科研分析由 main 完成，本文件只做机械聚合）。

### 4.6 `experiments/stage_S8ML_*`?

**不需要**。base runner 用 `--dataset` 分发两个 subclass（§4.2）；continuation runner 本来就是 dataset 参数化的（S7-B 模式）。这是相对 S6 四文件模式的有意简化：S6 需要双文件是因为改了训练目标的 pin；S8 不改训练语义，仅加 dump/continuation。

## 5. Required configuration changes

无独立配置文件。全部配置为 protocol 常量 + CLI：

- Phase 1：`python experiments/stage_S8_base_run.py --dataset {kuairand|movielens} --setting {natural|d2_large} --backbone m3oe --run_type joint --seed {N} --epoch 50 --batch_size 40960 --learning_rate 0.01 --learning_rate_darts 0.001 --weight_decay 1e-5 --expert_num 4 --num_workers 0 --device cuda:0 --shadow_patience 10 --formal_run_key <key_base> --run_dir <base_run_dir> --co_run_pair_id/--planned_peer_run_key/--worker_id/--attempt_id`（orchestrator 填充，单进程串行）。
- Phase 2：`python experiments/stage_S8_run.py --dataset ... --seed N --anchor-class {selected|middle|late} --out-dir <cont_run_dir> [--h-max-epochs 2]`。
- selected anchor 的 epoch pin 值由 designer 冻结在 protocol（§4.1 表）；coder 不得从 manifest 动态推导（pin 即规格）。Collector 须硬核对 pin 值 == 原 S6 manifest `selected_epoch`。

## 6. Required metrics / logging

### 6.1 Phase 1（base run）产物

S6 全部产物（config/run_manifest/batch_trace/gradient_step/probe/metrics_epoch/metrics_test/metrics_summary/checkpoints/moment dumps/optimizer_state 等）**加** `full_state_epoch{a}.pth`（a ∈ {selected_seed, 19, 39}，每 run 3 个；大小估算：KR ≈ 20 MB/个，ML ≈ 10 MB/个，总计 < 500 MB）与 manifest 中的 dump hash 表。

### 6.2 Phase 2 每 run 的 step 级 CSV（`continuation_steps.csv`）

列（每 branch × 每 step 一行，共 3×K 行）：
`stage, dataset, seed, anchor_class, anchor_epoch, branch, epoch, step(全局 continuation 步 1..K), batch_fingerprint, domain_counts, L_full, L_target, L_non, g_full_norm, g_target_norm, cosine_gf_gt, m_norm_persistent, v_norm_persistent, delta_full_norm, delta_target_norm, delta_applied_norm, delta_cosine, private_delta_norm, applied_formula_residual, private_full_residual, theta_distance_to_F_shared(仅 T/DH 行), eff_align_full, eff_align_target, eff_dot_full, eff_dot_target（后四列非采样步为 NaN）, rng_digest, validity_L_full_diff(F 行), validity_delta_norm_rel_diff(F 行)`。

### 6.3 validation 轨迹（`continuation_validation.csv`）

每 branch × 每步（step 0..K 的 post_step 点 + 每 epoch 末的 epoch_end_post_arch 点）× 9 行（6 cells + macro/target_domain/non_target_weighted 三个 scope）：列同 S7-B（`scope, domain, task, cell, val_bce_raw, auc, logloss`）+ `kind ∈ {anchor, post_step, epoch_end_post_arch}`。**validation metrics 每步都算**（primary endpoint 需要完整轨迹）；**validation gradient（U_k/H/eff_align）采样步**：step 1–5 每步、之后每 5 步、每 epoch 最后一步、以及 K。采样规则冻结在 protocol（`VALGRAD_STEPS(K)`），不改。

### 6.4 geometry CSV（`continuation_geometry.csv`）

仅在 §6.3 采样步输出，schema 同 S7-B（6×6 cells × 8 modules = 288 行/branch/步）：raw gradient norm、pairwise cosine、sample alignment、validation dot（U_k）、weighted H_{i→k}、val grad norm、source weight。

### 6.5 best 表（`continuation_best.csv`）

每 branch × 每 horizon（§2.3 全部 4/5 个，由同一轨迹前缀导出）一行：`branch, horizon, horizon_steps, best_step, best_macro_auc, best_target_auc, best_macro_logloss, best_worst_cell_auc, best_incl_anchor_macro_auc, anchor_macro_auc(step0), final_macro_auc(step K)`。附 frozen 参照列：`f_original_best_macro_auc, f_original_selected_epoch`（collector 从原 S6 canonical manifest 填入并标注来源）。

### 6.6 audit（`replay_audit.json`）

S7-B 全字段 +：`full_state_dump_sha256`、`private_state_count`（必须 == private_names 数）、`shared_step_observed/expected`、`arch_step_observed/expected`、`v3_state_bit_equality`（三分支参数+双 optimizer state 逐位相等的布尔与逐项明细）、`vbase_crosscheck`（见 §7）、`arch_fingerprint_rows`、`epoch_end_f_macro_auc_vs_source`（逐 epoch）、`h_max_epochs`。

### 6.7 validity 日志分层（写入 audit，供 run report 引用）

V-BASE / V0 / V1 / V2 / V3 / V4 / V5 各自的 pass/fail、数值与 tripwire 旗标（见 §7）。

## 7. Validity tests（自动、分层；任何 hard gate 失败 = abort 并写 failure）

- **V-BASE（Phase 1 完成后立即执行，hard）**：S8-BASE run vs 原 S6 canonical F run：
  1. 全部 train batch fingerprint 逐位相等（2450/250 个）；
  2. 逐 epoch `validation_macro_auc` 差 ≤ 1e-6（预期 ~1e-12–1e-9；CUDA backward 原子非确定性如实声明）；
  3. `selected_epoch` 相等；`selected_validation_macro_auc` 差 ≤ 1e-6；
  4. `initial_parameter_state_hash` 相等；
  5. final 参数 state hash 差：报告级记录（不作 hard gate——若 2/3 通过而此项有微差，记录为已知 fp 漂移，不影响 anchor 状态有效性，因为 anchor ≤ 47 的状态有效性由 1–3 保证）。
- **V0（continuation step 0/1 起点，hard）**：F 分支第 1 步的 `L_full` 与 S8-BASE 源行差 ≤ 1e-5、shared delta rel diff ≤ 1e-3；step-0 validation macro AUC 与源 `metrics_epoch.csv` epoch=anchor 行差 ≤ 1e-6。
- **V1（每步每分支，hard，步内断言）**：applied==Δ_applied、private==Δ_P（`assert_vector_match`，S4 容差）、None-gradient fail-closed、步骤边界 RNG digest 不变。
- **V2（轨迹漂移，report-only + tripwire）**：F 分支逐步 `L_full` diff 与逐 epoch post-arch macro AUC diff vs S8-BASE 源；tripwire：单步 |L_full diff| > 0.3 或单 epoch |macro AUC diff| > 1e-3 → `review_flagged=true`（不 abort）。预期值 ~1e-9–1e-6（远好于 S7-B 的 0.17，因为 private moments 已恢复）。
- **V3（same-state 构造，hard）**：三分支在 step 0 pre-step 的全部 model 参数、main optimizer 全部 state（含 private m/v/step）、arch optimizer state 两两 `torch.equal`；arch 参数亦逐位相等。
- **V4（batch replay，hard）**：continuation 消费的**每一个** main batch 与**每一个** arch batch 的 fingerprint 与 S8-BASE `batch_trace.csv` 逐位相等（全 horizon，不只前 5 步）。
- **V5（状态恢复完整性，hard）**：shared step == bpe×(anchor+1)、arch step == anchor+1、private state 全覆盖、model 参数 hash == dump 内 hash、initial hash == S8-BASE manifest。

## 8. Expected output files

```
experiments/
  stage_S8_protocol.py / stage_S8_base_run.py / stage_S8_run.py
  stage_S8_orchestrator.py / stage_S8_report.py
  stage_S8_base_raw_results/{kuairand,movielens}/seed{N}/joint/    # §6.1
  stage_S8_raw_results/{dataset}/seed{N}/{anchor_class}{epoch}/    # 每 run:
    continuation_steps.csv / continuation_validation.csv / continuation_geometry.csv
    continuation_best.csv / continuation_best_{F,T,DH}.pth
    replay_audit.json / process_interval.json / stdout.log
  stage_S8_outputs/
    stage_S8_base_validity.csv        # V-BASE 逐 run 结果
    stage_S8_best_auc_table.csv       # grad.md §12 主表：
                                      # dataset, seed, anchor_class, anchor_epoch, branch, horizon,
                                      # f_original_best_auc, continuation_best_auc, delta_best,
                                      # final_auc, late_rescue?(continuation_best>f_original 且
                                      #   anchor_class==late 且 final>f_original 但 best<=f_original
                                      #   的判定列按 §11 规则机械计算), new_best?(delta_best>0)
    stage_S8_trajectory.csv           # 全部 validation 行的聚合视图
    stage_S8_effective_update.csv     # §6.2 中 eff_* 与 raw 诊断的采样步聚合
    stage_S8_summary_manifest.json
    reports/stage_S8_run_report.md    # runner 事实报告骨架
```

## 9. Potential implementation pitfalls

1. **Windows/venv 重定向警告**（`Failed to find real location of ... anaconda3\python.exe`）：S6 Phase B 实测为良性环境警告，orchestrator 照常 exit 0；报告如实记录，不当失败处理。
2. **num_workers=0 必须保持**（Windows spawn pickle 失败）；任何额外 DataLoader iterator（即使 shuffle=False/num_workers=0）都消耗全局 RNG 的 `_base_seed` draw——replay 的每 epoch 3-draw 记账（main 2 + train_records 1）一次都不能漏，validation/arch 必须在 `preserve_rng_state` 内或纯 tensor 操作（S8 用 S7-B 的 `_full_validation_metrics`，直接对预载 numpy 数组分块 forward，**不建 DataLoader**，天然零消耗；每步边界 digest 断言兜底）。
3. **跨 epoch shuffle 边界**：continuation 跨 epoch 时必须严格按 [耗尽 iterator → 1 次 train_records iter → arch（净 0）→ validation（净 0）→ 新 epoch iter（2 draws）] 的源顺序推进 RNG；任何顺序调换都会使下一 epoch 的 fingerprint 全部失配（V4 会抓住）。
4. **arch batch 的"代"重建**：arch iterator 每 bpe 个 epoch 重建一次（KR：epoch 0 与 49；ML：epoch 0/5/10/...）。`arch_batch_for_epoch` 必须按代重建再 skip，不能用 continuation 当前 RNG 状态创建。fingerprint hard gate 兜底。
5. **partial batch 权重**：drop_last=False，ML 末批 23660 行、KR 末批 41492 行；`w_{d,t}=n_d/(2B)` 用实际 B（S7-B 已处理，沿用）。
6. **checkpoint/内存**：KR full-state dump ≈ 20 MB×3/run；三分支深拷贝后进程内 4 份模型+optimizer（原+3 分支）——模型仅 1.6 M 参数（KR）/0.77 M（ML），GPU 8 GB 无压力；continuation batch 必须**流式**消费，禁止物化 98–245 个 batch。
7. **KR 时长**：单个 2-epoch KR continuation ≈ 30–60 min；base 重训 ≈ 2 h/run。串行总时长见 §10；orchestrator 每个 task 独立落盘，中断后可从未完成任务续跑（目录非空拒绝覆盖，重跑前需人工归档失败目录——沿用 S5/S7-B 先例）。
8. **fp 非确定性**：CUDA backward 原子序使重训轨迹与原 run 存在 ~1e-9–1e-7 级漂移（S7-B V0 实测 1e-11/步）。V-BASE 的 1e-6 容差与 V2 的 1e-3 tripwire 已按此设定；**不得**为追求逐位一致而改 seed/改代码路径。分支间对比（F/T/DH 同进程同状态）不受此影响，是科学结论的唯一基础。
9. **selected anchor 不在 moment dump 集合内**：ML 的 7/8/12、KR 的 24/30/41/47 均 ∉ {0,1,9,19,29,39,49}；full-state dump 集合必须按 §4.1 的 per-seed 表生成，禁止四舍五入到邻近 common epoch。
10. **继承链陷阱**（S6 已踩过）：`S4Runner.__init__` 会把 `objective_weighting` 改回 `"equal"`——S8 base runner 必须沿用 S6 的 `S3Runner.__init__(self, args)` 绕行写法（`stage_S6_run.py:133-146` 注释）；`write_outputs` 中的 `source_code_hash`、`_check_source_drift` 的 inventory 都要切换到 S8 版本，否则 fail-closed 误报或静默错 pin。
11. **primary endpoint 口径**：collector 计算 `delta_best` 时参照必须是**原 S6 canonical** manifest 的 `selected_validation_macro_auc`（不是 S8-BASE 重训值、不是 late checkpoint 值）；S8-BASE 值仅作 V-BASE 交叉核验。两者若差 > 1e-6，V-BASE 已失败，collect 不得放行。
12. **step 0 是否计入 best**：primary 用 steps 1..H（不含 anchor 点）；`best_incl_anchor` 仅作敏感性列。分析期不得混用。

## 10. 运行矩阵与 wave 组织（严格单进程串行）

**Phase 1（base，8 runs）**：MovieLens × 4 seeds（≈13 min/run → ~1 h）→ KuaiRand × 4 seeds（≈2.0 h/run → ~8 h）。

**Phase 2（core continuation，24 runs，每 run 3 分支、H_max=2 epochs）**：

| dataset | runs | 每 run 步数（3 分支合计） | 每 run 估算 | 小计 |
|---|---|---|---|---|
| MovieLens | 4 seeds × 3 anchors = 12 | 30（10 步×3） | ~4–6 min | ~1 h |
| KuaiRand | 12 | 294（98 步×3） | ~30–55 min（含逐步 validation；val-gradient 采样后） | ~6–11 h |

**Phase 3（可选 5-epoch 扩展，23 runs）**：MovieLens 12（~10 min/run → ~2 h）；KuaiRand 11（排除 seed2023/selected=47；~1.3–2.3 h/run → ~15–25 h）。**仅在 core 矩阵完成、V 全绿、且 main 明确批准后启动**；不得用 Phase 3 替代 core 中任何 task。

**core 总计**：~16–21 h 串行（1–2 天）。**裁剪纪律**（grad.md §6/§7 合规）：不删任何 seed、不删 KuaiRand、不删三类 anchor；唯一允许的裁剪是不启动 Phase 3。若 core 执行中实测单 run 时长超估算 2 倍，优先把 §6.3 的 val-gradient 采样加稀（每 10 步），**不得**减少 validation metrics 频率（primary endpoint 依赖逐步轨迹）。

wave 组织：orchestrator 的 task 顺序 = [ML base ×4] → [KR base ×4（每个完成后立即 V-BASE）] → [ML cont ×12] → [KR cont ×12，seed-major、anchor 序 selected→middle→late]。任一 task FAIL/abort → 停止后续，上报，不擅自重试改参。

## 11. 预注册的分析口径（供 main 聚合；本规格不做结果判断）

1. **主表**（grad.md §12 逐字 schema）：`stage_S8_best_auc_table.csv`；按 anchor_class ∈ {selected, middle, late} 分组聚合 `delta_best`：4-seed mean、样本 SD、逐 seed 符号、≥3/4 同向计数、最大 |单 seed 贡献|（单 seed 驱动检查）。primary horizon = **h_2ep**（core 最大窗口）；其余 horizon 作为 trajectory  secondary 视图全部报告。
2. **参照链**：`f_original_best_auc` 一律取自原 S6 canonical manifest；`new_best = (delta_best > 0)`；`late_rescue` 列 = (continuation best ≤ f_original best) 且 (continuation final 或 best 优于 F 分支同 anchor 同步数 final)——两个布尔列机械计算，解释归 main。
3. **trajectory（secondary）**：AUC(t)/BCE(t) 按 (dataset, seed, anchor_class, branch) 分组，区分 grad.md §9 的四种形态（延迟恶化 / 仅降低过拟合斜率 / 新的更高 optimum / 仅"没那么差"）所需的最小数据是：逐 seed 的 best_macro_auc、达到 best 的 step、final_macro_auc、以及 F 分支同窗口轨迹——全部已在 §6 物化。
4. **raw vs effective**：在 val-gradient 采样步上，对 (branch, anchor_class) 聚合 `cos(g_S,g_val)` 与 `eff_align_full/target` 的差；逐 seed 符号计数。判读（是否回答 §10.2 核心问题）归 main。
5. **KuaiRand 对照**：与 MovieLens 完全同口径聚合，禁止因 null 预期而删减列。
6. **GO/STOP**：由 main 按 grad.md §14/§15 对聚合表逐条核查；本规格预注册的唯一统计习惯：paired 单位为 seed，n=4，阈值沿用项目（practical 0.003 / strong 0.005 / ≥3/4 同向），不引入新阈值、不做事后多重性调整。

---

## 附：给 coder 的实现顺序

1. 写 `stage_S8_protocol.py`；静态核验：`key_base`/`key_cont`/`FULL_STATE_DUMP_EPOCHS`/`HORIZON_STEPS` 输出与本文件 §4.1/§2.3 逐值一致。
2. 写 `stage_S8_base_run.py`；import/syntax 检查；dry argparse 单测 `_validate_args`（fail-closed 对错误 dataset/objective/arm）；**smoke（非 formal）**：单 seed ML 跑 1 个 epoch 截断验证 `_dump_full_state` 写出且 torch.load 往返 state 逐位相等（ coder 级 smoke，不进正式矩阵）。
3. 写 `stage_S8_run.py`；三个 smoke 模式在 ML seed2022/selected(anchor7) 上全部通过后再写 orchestrator/report。
4. reviewer PASS 前不冻结 queue、不启动 Phase 1。Phase 1 的 8 个 run 是 Phase 2 的硬前提（V-BASE 不过不得启动 continuation）。
5. 全程禁止：修改 S1–S7 任何文件；调整任何容差/阈值；把 5-epoch 扩展提前；向训练/continuation 循环加入本文件未列出的逻辑。

---

## 设计自检（相对任务书硬性约束）

- 无新算法：F/T/DH 语义逐字复用 S6 `_controlled_step`；HT 不启用。✓
- private moments 不可重构 → 最小替代 = Phase 1 重训（§0.b），已明确写出且验证机制完整。✓
- seeds 2022–2025、双数据集、三类 anchor（ML selected = 7/8/8/12 per-seed）。✓
- horizons 5 steps / 0.5 / 1 / 2 epochs 全覆盖（嵌套前缀实现，天然共享 batch 序列）；5 epochs 列为受控可选项。✓
- primary endpoint = BestAUC_branch − BestAUC_F,original（macro AUC 口径，逐步轨迹取 best，非 late-checkpoint 对比）。✓
- 已知局限如实声明：重训轨迹与原 S6 存在 ~1e-9–1e-7 级 CUDA fp 漂移（V-BASE/V2 监控）；continuation 超出 epoch 49 的 batch 序列无源指纹可对拍（已通过 a+5≤49 规则排除）。✓
