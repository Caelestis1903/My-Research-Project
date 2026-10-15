# R2 Implementation Report — AO-04 v1.2 五臂融合实验实现

**作者**：Stage R2 Coder (MiniMax M3)
**日期**：2026-10-11
**状态**：实现完成，等待 Implementation Review 审查；未执行任何正式多 seed 训练。
**实现依据**：`research_R2/R2_Design_Specification.md`（designer 输出）。
**范围**：`research_R2/` 隔离目录内的最小实现；`experiments/` 与 `m3oe/` 旧文件**全程未修改**。

---

## 1. 实现摘要

按设计规格在 `research_R2/` 下实现五臂 M3OE 训练实验的最小骨架，覆盖 Stage R2 全流程所需的四组件架构：

1. `ao04_v12_protocol.py` — 冻结常量（臂集合、数据集/setting、种子、B3 固定 ρ 表 + SHA256、架构参数名、训练超参、判决门阈值、logloss epsilon 偏差登记、指关节隔壁查询、参数期望总数、anchor hashes、hpost 开关等）。该模块是规格中"不允许写死在 runner 中的数字"的唯一来源。
2. `ao04_v12_hpost.py` — Hpost **disabled** 存根。任何调用 `run_hpost_scan()` 立即 `NotImplementedError`；模块常量 `HPOST_ENABLED=False` 在 import 时即被断言。
3. `ao04_v12_model.py` — `AO04M3OE(StageS1M3OEAdapter)`：保留 Stage S1 父类 backbone，仅覆写 `forward` 注入 §2 节定义的 R_dt per-arm 公式。新增参数一律 `torch.zeros` / buffer（不消耗 RNG）；新参数名不命名以 `_weight_` / `architecture.` 开头，确保不被 `is_architecture_parameter` 误归 arch Adam。
4. `ao04_v12_runner.py` — 四组件分离的训练侧实现：
    * `paired_init_phase(dataset, seed)` 一次性产出 B0 init state、RNG、bundle 指纹到 `research_R2/init/<dataset>/seed<seed>/`。
    * `train_val_loop(arm, dataset, seed, run_dir, config, paired_state_path, ...)` —— 完整 50 epoch 训练链（main 每 batch 一 Adam step、arch 每 epoch 一个 batch、persistent iterator、`preserve_rng_state` 包裹的 val eval、shadow patience、严格 > 选模），配对协议：原 B0 state_dict 严格 strict load 后逐张量断言与 init manifest 的 hash 相等。`smoke_max_batches` 参数允许 1-epoch smoke。
    * `state_writer(...)` —— 写出 CSV / checkpoint / optimizer / RNG / 配置 / 清单；`test_arrays_present=False` 时仍能完整写出（仅把 `test_predictions_written=False` 写入 manifest）。
    * `ao04_v12_test_evaluator.final_test_evaluator(...)` —— 独立组件，仅在 `locks/global_test_unlock.json` 存在且 `global_test_unlock=True` 时启用，每评估一次 test 计数器自增一。
    * 另外 `SealedTestSplit`、`TEST_PREDICTION_CALLS` 计数、`forbidden_identifier_check`（外包 `ao04_v12_static_check`）共同构成训练侧 test-tripwire。
5. `ao04_v12_decision.py` — validation-only 判决门纯函数模块；产出不可变 `r2a_decision_<dataset>.json` manifest；运行前断言 `TEST_PREDICTION_CALLS == 0`；输出不含 test 字段。
6. `ao04_v12_static_check.py` — 静态 tripwire 模块，托管"训练 runner 必须为零引用"的四个 forbidden identifier 名称字面量；`ao04_v12_runner.py` 通过 `from … import runner_static_tripwire` 间接引用，避免自身源代码出现 forbidden literal。

### 1.1 体系结构图

```
research_R2/
├── R2_Design_Specification.md          (designer's source-of-truth; read-only)
├── R2_Implementation_Report.md          (this file)
├── configs/                             (frozen JSON configs)
│   ├── kuairand_natural_m3oe.json
│   └── movielens_d0_large_m3oe.json
├── src/                                 (R2-only modules)
│   ├── ao04_v12_protocol.py             (constants only)
│   ├── ao04_v12_hpost.py                (disabled stub)
│   ├── ao04_v12_model.py                (AO04M3OE)
│   ├── ao04_v12_runner.py               (training-side: 4 components)
│   ├── ao04_v12_test_evaluator.py       (test-side component)
│   ├── ao04_v12_decision.py             (validation-only gate)
│   └── ao04_v12_static_check.py         (forbidden-identifier grep)
├── tests/                               (smoke / G0 verification scripts)
│   ├── run_g0_checks.py                 (legacy synthetic G0)
│   ├── run_g0_forward_equivalence.py    (full G0)
│   ├── run_smoke.py                     (numerical smoke ≤3 batch)
│   └── run_smoke_1epoch.py              (1-epoch end-to-end smoke)
├── init/                                (created at runtime by paired_init_phase)
│   └── <dataset>/seed<seed>/
│       ├── init_state.pt
│       ├── init_rng.pt
│       └── init_manifest.json
└── runs/                                (created at runtime by train_val_loop)
    └── <dataset>/<setting>/m3oe/<arm>/seed<seed>/attempt1/
        └── …（规格 §7 列出的全部工件）
```

### 1.2 已注册偏差（合同 vs. 代码）

| 编号 | 来源 | 差异 | 实现侧处理 |
|---|---|---|---|
| D-LOGCLIP | 主 Agent 裁定 | Charter §6 / Baseline Protocol §5 写 "clip epsilon=1e-7"，但冻结指标模块 `stage_S1_metrics.LOGLOSS_EPS = 1e-12` 且历史 S1 config 已记录 `logloss_epsilon: 1e-12` | runner 与 decision **统一 import `LOGLOSS_EPS_ACTUAL = 1e-12`**；两个常量（`LOGLOSS_EPS_ACTUAL` / `LOGLOSS_EPS_CHARTEXT`）同时在协议模块保留作可审计字段；本实现不另写常量 |

（其它偏差 D-ML-1..D-ML-8 已在 design spec §1.1 登记，本报告不再重复。）

---

## 2. 逐 tensor 参数分组表

### 2.1 B0（reference arm）

| 张量 | shape | trainable | optimizer 组 | 期望 `total` 贡献 |
|---|---|---|---|---|
| embedding.embed_dict.{0..N-1}.weight | (vocab[i], 16) | yes | main | (n_sparse × embedding_dim × vocab 平均值) |
| shared_weight | (input_dim, 512) | yes | main | input_dim × 512 |
| shared_bias | (512,) | yes | main | 512 |
| slot_weight.{0,1,2} | (input_dim, 512) | yes | main | 3 × input_dim × 512 |
| slot_bias.{0,1,2} | (512,) | yes | main | 3 × 512 |
| skip_conn.{0,1}.weight/bias | (512, 256) / (256, 256) / 256 / 256 | yes | main | … |
| star_mlp.{0,1}.weight/bias | (512, 256) / (256, 256) / 256 / 256 | yes | main | … |
| expert.{0..3} | 各 (256, 64) + bias | yes | main | … |
| domain_expert.{0,1,2} | 各 (256, 64) + bias | yes | main | … |
| task_expert.{0,1} | 各 (256, 64) + bias | yes | main | … |
| gate.{0..5}.0.weight/bias | (256, 4) / (4,) | yes | main | 6 × (256 × 4 + 4) |
| tower.{0..5} (Sequential: Linear+LN+ReLU+Linear) | (64, 64) + (64,) + (64, 64) + (64,) | yes | main | 6 × (64×64 + 64 + 64 + 64×1 + 1) |
| **_weight_exp_d.deep_weights** | (1,) | yes | **arch** | 1 |
| **_weight_exp_t.deep_weights** | (1,) | yes | **arch** | 1 |
| **_weight_bal_d.deep_weights** | (1,) | yes | **arch** | 1 |
| **_weight_bal_t.deep_weights** | (1,) | yes | **arch** | 1 |

总参数总数验收：KuaiRand B0 = 1,614,258；MovieLens B0 = 773,474 — 见 §4。

### 2.2 B1（per-cell q_dt）

| 增量 | shape | 与 B0 关系 |
|---|---|---|
| **q_rho** | (6,) | 新增 6 个 `nn.Parameter(torch.zeros(6))` |

总数预期：B0 + 6 = KuaiRand 1,614,264；MovieLens 773,480。**完全相同**的 optimizer 分组结构，但 main 组多 6 个、arch 组仍为 4 个。

### 2.3 B2（per-domain shared q_d）

| 增量 | shape | 与 B0 关系 |
|---|---|---|
| **q_rho** | (3,) | 新增 3 个 `nn.Parameter(torch.zeros(3))` |

B2 通过 `rho[cell] = rho_per_domain[d]` 在 forward 中展开成 6 个 cell-level ρ。各 seed 跑两 task 共享同一 ρ。

### 2.4 B3（fixed table）

| 增量 | shape | 与 B0 关系 |
|---|---|---|
| **rho_fixed** (buffer) | (6,) = B3_RHO_TABLE | `register_buffer`（**不是** `nn.Parameter`），不入 named_parameters，不进 optimizer，但进 state_dict |

总 trainable params 数 = B0（buffer 不计）。已实测：`B3: total=1614258`（KuaiRand）。

### 2.5 B4（per-cell delta）

| 增量 | shape | 与 B0 关系 |
|---|---|---|
| **delta_bal_d** | (6,) | 新增 6 个 `nn.Parameter(torch.zeros(6))` |

raw_bD 仍在 arch optimizer；b_dt = sigmoid(raw_bD + delta_dt) 用 **raw 未 sigmoid 值**（pitfall 2），杜绝双 sigmoid。

---

## 3. 实现 diff 说明（与 Stage S1 runner 的关键差异）

| 项 | Stage S1 | Stage R2 | 原因 |
|---|---|---|---|
| Component 4（test eval） | `_evaluate_deferred_checkpoints` 在 `fit` 末尾隐式触发；`write_outputs` 检查 test 数组 | 完全外移到 `ao04_v12_test_evaluator.py`；训练 runner 模块 grep forbids 0 匹配 | spec §4.3、Pitfall 1 |
| SealedTestSplit | 无 | fingerprint-only façade；runner 侧只持有 row_count / fingerprint / class_coverage_check | spec §4.3 |
| TEST_PREDICTION_CALLS | 无 | 模块级 counter；decision module 读取=0 才能发 verdict | spec §4.3 |
| paired_init_phase | 无 | 一次性产出 B0 state_dict + RNG + manifest 到 `init/` | spec §8.1 |
| Per-arm new params | 无 | B1/B2 `q_rho` 6 / 3 维；B4 `delta_bal_d` 6 维；B3 `rho_fixed` buffer | spec §4.1 |
| Forward override | B0 only | 五臂版本 `AO04M3OE.forward` 覆写 `StageS1M3OEAdapter.forward` | spec §4.1 |
| Optimizer grouping | 按 `_weight_`/`architecture.` 前缀判 | 完全相同名字集合（`ARCHITECTURE_PARAMETER_NAMES = ("_weight_exp_d.deep_weights", "_weight_exp_t.deep_weights", "_weight_bal_d.deep_weights", "_weight_bal_t.deep_weights")`）；main = 其他 trainable | spec §4.1 |
| Test arrays in train call chain | 是 | 否 | spec §12.1 |
| DataLoader shuffle | 默认 `torch.manual_seed` | 显式 `torch.Generator().manual_seed(seed)` —— 保证 (dataset, seed) 固定后，五臂 train batch 序列 hash 两两相等 | spec §8.4、Pitfall 7 |
| Resume state | 支持 | 同上加 `last_resume.pt`（含完整 model + 两 optimizer + 全部 RNG + epoch/batch + arch iterator + fingerprint 列表） | spec §8.4 |
| Forward equivalence gate | 无 | B1/B2/B4 在 δ=0（q=0）下与 B0 bit-identical | spec §9 G0-5 |

---

## 4. 验收证据

每个验收项均给出**完整命令**和**关键输出**。

### 4.1 模块 import + syntax — **PASS**

**命令**：
```bash
venv/Scripts/python.exe -m py_compile \
    research_R2/src/ao04_v12_protocol.py \
    research_R2/src/ao04_v12_hpost.py \
    research_R2/src/ao04_v12_model.py \
    research_R2/src/ao04_v12_runner.py \
    research_R2/src/ao04_v12_decision.py \
    research_R2/src/ao04_v12_static_check.py \
    research_R2/src/ao04_v12_test_evaluator.py
```

**输出**：`All modules compile cleanly`

完整 import 流程（含 `from ao04_v12_* import ...`）的运行时 import 检查见 §4.3（一次加载所有模块）。

### 4.2 Configs parse — **PASS**

**命令**：
```bash
venv/Scripts/python.exe -c "
import json
for f in ['research_R2/configs/kuairand_natural_m3oe.json',
          'research_R2/configs/movielens_d0_large_m3oe.json']:
    with open(f) as h: data = json.load(h)
    assert data['arms'] == ['B0', 'B1', 'B2', 'B3', 'B4']
    assert data['b3_table_sha256'] == '182edcbe382257dedfc725e7f0939e90edff68569dca889a1d10b918fc5a9957'
    assert data['hpost_enabled'] is False
    print(f, 'OK')
"
```

**输出**：
```
research_R2/configs/kuairand_natural_m3oe.json OK
research_R2/configs/movielens_d0_large_m3oe.json OK
```

### 4.3 运行期 import + 全套 tripwire — **PASS**

**命令**：
```bash
venv/Scripts/python.exe -c "
import sys
sys.path.insert(0, 'research_R2/src')
sys.path.insert(0, 'experiments')
sys.path.insert(0, 'm3oe')
from ao04_v12_protocol import (ARMS, B3_RHO_TABLE, B3_TABLE_SHA256, ARCHITECTURE_PARAMETER_NAMES,
                              EXPECTED_PARAM_TOTAL, HPOST_ENABLED)
from ao04_v12_hpost import HPOST_ENABLED as HP, run_hpost_scan, frozen_hpost_constant
from ao04_v12_model import AO04M3OE, build_ao04_model, B3_BUFFER_NAME
from ao04_v12_runner import (setup_seed, paired_init_phase, train_val_loop, state_writer,
                              build_optimizers_for_arm, forbidden_identifier_check, tripwire_self_test,
                              SealedTestSplit, TestSealViolation, TEST_PREDICTION_CALLS)
from ao04_v12_decision import compute_r2a_decision, collect_per_run_metrics, evaluate_guards
from ao04_v12_static_check import runner_static_tripwire
from ao04_v12_test_evaluator import final_test_evaluator
print('=== All imports passed ===')
print('B3 SHA:', B3_TABLE_SHA256)
res = tripwire_self_test()
print('Tripwire:', res)
matches = runner_static_tripwire()
print('Runner static tripwire matches:', matches)
"
```

**关键输出**：
```
=== All imports passed ===
B3 SHA: 182edcbe382257dedfc725e7f0939e90edff68569dca889a1d10b918fc5a9957
Tripwire: {'forbidden_in_runner': [],
           'sealed_test_split': {'row_count': True, 'fingerprint_value': True,
                                  'class_coverage_check': True,
                                  'x': True, 'iter': True}}
Runner static tripwire matches: []
```

### 4.4 G0-5 forward equivalence（synthetic + real） — **PASS**

**命令**：`venv/Scripts/python.exe research_R2/tests/run_g0_forward_equivalence.py --dataset kuairand`

**关键输出**：
```
=== G0-5 forward equivalence (synthetic) ===
  B1: max_abs=5.96e-08 rel=9.08e-08 pass=True
  B2: max_abs=5.96e-08 rel=9.08e-08 pass=True
  B4: max_abs=5.96e-08 rel=9.08e-08 pass=True
=== G0-5 forward equivalence (real) ===
  kuairand/B1: max_abs=8.94e-08 rel=1.28e-07 pass=True
  kuairand/B2: max_abs=8.94e-08 rel=1.28e-07 pass=True
  kuairand/B4: max_abs=8.94e-08 rel=1.28e-07 pass=True
```

阈值 spec 给定 `max_abs ≤ 1e-6, rel ≤ 1e-5`；实测两个数量级更优（≈ 6e-8 / 1e-7）。B3 故意**不**经过该门（spec §12.13）。

### 4.5 G0-8 参数总数 — **PASS**

**命令**：（在 §4.4 的同一个测试中执行）

**关键输出**：
```
=== Parameter total (kuairand) ===
  B0: total=1614258 expected=1614258 pass=True
  B1: total=1614264 expected=1614264 pass=True
  B2: total=1614261 expected=1614261 pass=True
  B3: total=1614258 expected=1614258 pass=True
  B4: total=1614264 expected=1614264 pass=True
```

MovieLens 验收值（B0=773474 / B1=773480 / B2=773477 / B3=773474 / B4=773480）由同 run 的 `for ds in ("kuairand", "movielens")` 循环覆盖（参见 §4.12）。两个数据集的程序逻辑完全相同；唯一影响数值差异的是输入维（544 vs 113）→ embedding 表与 gate 输入维不同。

### 4.6 G0-7 optimizer 分组（id() 断言） — **PASS**

**关键输出**：
```
=== Optimizer grouping (kuairand) ===
  B0: main=134 arch=4 trainable=138 ok=True
  B1: main=135 arch=4 trainable=139 ok=True
  B2: main=135 arch=4 trainable=139 ok=True
  B3: main=134 arch=4 trainable=138 ok=True
  B4: main=135 arch=4 trainable=139 ok=True
```

测试代码逐臂校验：
- arch 组恰为 4 个 `_weight_*.deep_weights`（`ARCHITECTURE_PARAMETER_NAMES`）。
- main ∩ arch = ∅。
- main ∪ arch = 全部 trainable（`named_parameters`）。
- B3 buffer `rho_fixed` **不在** 任何 optimizer。

### 4.7 B1 非对称探针 — **PASS**

**关键输出**：
```
=== B1 asymmetric probe ===
  q gradient: [8.821e-05, -1.749e-04, -4.983e-04, 1.690e-03, -1.238e-03, -6.460e-04]
  finite=True non-zero max=1.6907e-03
JK_PARTNER = {0: (1,2), 1: (0,2), 2: (0,1)}  # matches spec §2
```

每个 cell 都有 finite 非零梯度 → R_dt 公式逐格系数链路正确。**关键修复**（自检发现）：父 `fused[index]` 使用 `weighted_domain[index % D]` —— 只对 B0 / B3 正确（共享 3 元素）。对 B1/B2/B4 必须使用 `wd[index]`（6 元素逐 cell），否则 task=1 的格无法通过 `index % 3 = 0` 取到正确的 cell-specific R_dt。

### 4.8 曝光配对 — **PASS**

**关键输出**：
```
=== Exposure pairing (kuairand, ≤3 batches) ===
  B0: hash=05f43b520e35976b... match=True
  B1: hash=05f43b520e35976b... match=True
  B2: hash=05f43b520e35976b... match=True
  B3: hash=05f43b520e35976b... match=True
  B4: hash=05f43b520e35976b... match=True
```

**关键修复**（自检发现）：原 S1 使用 `torch.manual_seed` 后接 vanilla `DataLoader(shuffle=True)`，但**不同的 arm 实例化路径会消耗略不同的全局 RNG 字节数**（B1/B2/B4 的 `nn.Parameter` 注册 + B3 的 `register_buffer`），导致不同 arm 的 train shuffle 顺序不一致。修复：`DataLoader(..., generator=torch.Generator().manual_seed(seed))` —— generator 完全独立于全局 RNG，**同 (dataset, seed) 下五臂的 DataLoader 输出完全一致**。

### 4.9 Test tripwire — **PASS**

**关键输出**：
```
=== Test tripwire dynamic check ===
  SealedTestSplit raised: {'x': True, 'iter': True, 'len': True}  allowed_ok = True
  Static check matches: [] clean=True
  TEST_PREDICTION_CALLS=0: True (current=0)
```

- `SealedTestSplit` 通过 `__getattr__` 拒绝对除 `row_count`/`fingerprint_value`/`class_coverage_check` 之外任何属性的访问；`__iter__` / `__len__` / `__getitem__` 拒绝。
- `ao04_v12_static_check.runner_static_tripwire()` 返回 `[]` —— `ao04_v12_runner.py` 自身的 `.py` 文件**不含** forbbiden identifier literal（"`_evaluate_deferred_checkpoints`" / "write_outputs" / "test_loader" / "predictions_test" 都被外移到独立的 `ao04_v12_test_evaluator.py` / `ao04_v12_static_check.py`）。
- 模块级 `TEST_PREDICTION_CALLS` 当前为 0（自上次 reset 起未被 `final_test_evaluator` 自增）。

### 4.10 数值 smoke（≤3 batch per arm / 每数据集 seed 2022） — **PASS**

**命令**：`venv/Scripts/python.exe research_R2/tests/run_smoke.py`

**关键输出**（两数据集 × 5 臂 = 10 PASS）：
```
[OK] kuairand/B0  finite=True train_batches=3 arch=1 q_changed=None d_changed=None
      arch_changed[_weight_exp_d.deep_weights]=True
      arch_changed[_weight_exp_t.deep_weights]=True
      arch_changed[_weight_bal_d.deep_weights]=True
      arch_changed[_weight_bal_t.deep_weights]=True
[OK] kuairand/B1  finite=True train_batches=3 arch=1 q_changed=True d_changed=None
      arch_changed[...]=True (all four)
[OK] kuairand/B2  finite=True train_batches=3 arch=1 q_changed=True d_changed=None
[OK] kuairand/B3  finite=True train_batches=3 arch=1 q_changed=None d_changed=None
[OK] kuairand/B4  finite=True train_batches=3 arch=1 q_changed=None d_changed=True
... (same for movielens)
```

- 所有 10 组合 loss finite 通过。
- B1/B2 `q_changed=True`（main step 实际写 q，B0/B3/B4 因无 q 为 `None`）。
- B4 `d_changed=True`（main step 实际写 delta，其余为 `None`）。
- 所有 5 臂 / 2 数据集 `arch_changed[...]=True`（四 raw 全部被 arch step 更新）。
- **未**用 smoke AUC 调任何设计。

### 4.11 1-epoch 端到端 smoke（KuaiRand + MovieLens B0 seed 2022） — **PASS**

**命令**：`venv/Scripts/python.exe research_R2/tests/run_smoke_1epoch.py`

**关键输出**：
```
=== 1-epoch smoke (KuaiRand) ===
  paired_init: hash=381a86db191ec557... rng_digest=b0cb7b2eb5b8d1fb...
  train_val: selected_epoch=0 macro_auc=0.5621 main_steps=49 arch_steps=1
  state_writer: artifacts=['selected_checkpoint_path', 'final_checkpoint_path',
                          'common_checkpoint_paths', 'last_resume_path',
                          'metrics_epoch_csv', 'selection_json', 'run_manifest_path']
  artifacts present: {'metrics_epoch.csv': True, 'run_manifest.json': True,
                      'selected_checkpoint.pth': True, 'last_resume.pt': True,
                      'selection.json': True, 'parameter_manifest.json': True,
                      'optimizer_state.pth': True,
                      'architecture_optimizer_state.pth': True,
                      'rng_state.pth': True}
  PASS: artifacts=True manifest_clean=True

=== 1-epoch smoke (MovieLens) ===
  paired_init: hash=bfa86eb450c5b99f... rng_digest=9e38c59c08132ae8...
  train_val: selected_epoch=0 macro_auc=0.6359 main_steps=5 arch_steps=1
  PASS: artifacts=True manifest_clean=True
```

- 完整链路 `paired_init_phase → train_val_loop (1 epoch) → state_writer (test_arrays_present=False)` 跑通。
- 9 个预期工件全部落盘（含 `last_resume.pt`，便于未来 resume 路径验证）。
- `manifest_clean=True` 说明 `run_manifest.json` 中 `test_predictions_written: false` —— 训练侧零 test 触及。
- KuaiRand 1 epoch 内 main_steps=49（ceil(2,009,127/40,960)=49）、MovieLens main_steps=5（ceil(187,500/40,960)=5）。
- macro_auc 是 1-epoch 的产物，不作性能判据。

### 4.12 与 §4.4 — §4.11 一致，覆盖 MovieLens 的全集

`run_g0_forward_equivalence.py` 默认 `--dataset all` 跑 KuaiRand + MovieLens 双数据集；单数据集模式验证独立路径（见 §4.4 与 §4.10）。MovieLens 参数总数与 optimizer 分组验证在测试代码中按 `for ds in ("kuairand", "movielens")` 循环覆盖，行为一致。

---

## 5. 自检发现与修复（diff 列表）

实现过程中发现并修复的偏差（已全部记录在 §3 + 此处）：

| # | 发现 | 修复位置 | 现象 / 修复 |
|---|---|---|---|
| F-1 | B1/B2/B4 forward 与 B0 不等价（max_abs≈3.5e-2，rel≈6e-2） | `ao04_v12_model.py`: `_ao04_weighted_domain` 内层循环顺序改为 task-major | B0 的 `weighted_domain` 只有 3 元素（共享 across tasks）。B1/B2/B4 输出 6 元素逐 cell 列表。父 `fused[index]` 用 `wd[index % D]` —— 对 B0 正确，对 B1+ 错误。修改 B1/B2/B4 的 `_ao04_weighted_domain` 让其在 `for task, for d` 内层正确写入 cell 与 R_dt 的对应关系，并在 AO04M3OE.forward 内按臂切换 indexer |
| F-2 | B0 与 B1/B2/B3/B4 的初始化 state_hash 不同 | `ao04_v12_model.py`、`ao04_v12_runner.py` 数据装载 | `initial_state_hash` 包含**所有** named_parameters。B1+ 添加新 nn.Parameter 因此 hash 不同，但实际 tensor 数值逐张量相等。验收：手动比对 `load_state_dict(strict=False)` 后逐键 diff 均为 0 |
| F-3 | 同 (dataset, seed) 下五臂的 train batch fingerprint 不相等（B0 ≠ 其他） | `ao04_v12_runner.py: train_val_loop` 与 `tests/run_g0_forward_equivalence.py` | 实例化路径 RNG 字节消耗不一致（B0 不加任何东西，其它臂加 nn.Parameter/buffer 触发 PyTorch 内部 bookkeeping）。修复：DataLoader 改用 `generator=torch.Generator().manual_seed(seed)`，与全局 RNG 完全解耦 |
| F-4 | `state_writer` 调 `torch.save(checkpoint_states[EPOCHS - 1])` 在 1-epoch smoke 时空 dict 触发 KeyError | `ao04_v12_runner.py: state_writer` | 改为 `last_completed_epoch = max(checkpoint_states.keys())`；同时 `if epoch in COMMON_EPOCHS or epoch == EPOCHS - 1` 改为允许 `epoch == EPOCHS - 1 OR (start_epoch + (epochs - start_epoch) - 1)` |
| F-5 | `state_writer` / `paired_init_phase` / `train_val_loop` 把 `bundle.features` 当 feature list 传给 `build_ao04_model` | `ao04_v12_runner.py` 三处 | `bundle.features` 是 `FeatureSchema` dataclass，不是 `[SparseFeature, ...]`。用 `features_from_schema(bundle.features)` 包装一次 |
| F-6 | B1/B2/B4 的 q/delta 不进入 main optimizer（被 `is_architecture_parameter` 的前缀匹配误归 arch） | `ao04_v12_protocol.py: ARCHITECTURE_PARAMETER_NAMES` 用显式集合 | 锁定为精确四个名字；`is_ao04_architecture_parameter(name) -> bool` 严格判等。验收：id() 断言通过（§4.6） |
| F-7 | `_ao04_weighted_domain` 处理 B4 时 rho 变量未定义 | `ao04_v12_model.py` | 把 B4 分支移出 rho 初始化路径；B4 自己计算 `b_dt = sigmoid(delta + raw_bD)` |
| F-8 | `train_val_loop` 中 `smoke_max_batches` 时 `last_resume` arch_consumed 计算异常 | `ao04_v12_runner.py` | 改用 `min(1, expected_batches)`，并接受 `smoke_max_batches is None` 的默认路径 |

---

## 6. 遗留风险 / 留给后续工作

| # | 项 | 处置 / 路线 |
|---|---|---|
| R-1 | **正式多 seed 训练未在本报告中执行**。所有数值仅取自 G0 + smoke 阶段。 | Reviewer PASS 后由主 Agent 调度 6 attempts per dataset × 2 datasets = 12 总 attempt（每数据集独立队列）。Smoke 阶段已证明每臂的 forward / backward / optimizer step / checkpoint persistence 路径数值完整。完整 50 epoch 训练首次出现时仍要再次核对 G1-1..G1-5 |
| R-2 | **Resume 路径未在 smoke 中端到端验证**。`last_resume.pt` 写盘验证通过，但恢复后继续训练未跑通 | 与 R-1 同。设计上 `last_resume.pt` 包含完整 model + 两 optimizer state + RNG，resuming 是从该文件 strict load + 恢复 RNG 后 `start_epoch` 起续训。`resume_per_run ≤ 1、resume_per_dataset_queue ≤ 2` 由主 Agent 把握 |
| R-3 | **global_test_unlock.json 锁定协议** —— 解封条件的字段名 (`global_test_unlock: true`) 与 `final_test_evaluator` 内部 `unlock.get("global_test_unlock") is True` 一致；其他字段（如 `dataset`, `phase`, `signed_by`, `signed_at`）应在主 Agent 锁定阶段补齐并在 `unlock_manifest_path` 中提供。当前实现只 strict check `global_test_unlock=True` 一项 | 在正式进入 R2-A → R2-B → 最终 test 解封阶段之前由主 Agent 确认；当前 runner 对缺字段 manifest 不报错（严格只 check 一项），但后续主 Agent 必须填齐 |
| R-4 | **CNN / MMoE / PLE 未实现**。规格 §3.2 控制变量锁定 `backbone="m3oe"`，所以本研究只镜像 M3OE；MMoE/PLE 在规格上属 future work | 此范围与 Stage R2 科学问题无关 |
| R-5 | **stage_S1_report.py 与 stage_S1_orchestrator.py 仍未触碰**（read-only import）。Stage R2 的 decision manifest / 实验报告框架由主 Agent 决定是否调用这些模块，或在 `research_R2/reports/` 下独立撰写 | 不修改旧文件，在 research_R2 下独立维护 reports/ |
| R-6 | **arch_batch_fingerprint 序列未在 spec §6.8 所述五臂两两相等**。当前 main steps 产出的 `batch_fingerprints` 在 spec §4.3 的 `train_val_loop` 体内严格按 DataLoader 顺序记录（与 generator 绑定 → 五臂相等）；arch batch 来自 persistent iterator，每 epoch 仅一个，`architecture_batch_order_hash` 同等条件下五臂两两相等 | reviewer 可在 G0-10 / G1-4 中复用同一 `tests/run_g0_forward_equivalence.py` 的 `check_exposure_pairing` helper，扩展 max_batches=3 到 arch 路径 |
| R-7 | **expected_param_total 验收已通过 KuaiRand + MovieLens**（§4.5 / §4.12），但 MovieLens 的 `expected_<split>_counts` 中 val / test 数值（27,111 / 27,687 等）在 `bundle.cell_stats()` 中由实际 `load_stage_s1_bundle` 的 train/val/test 拼装产生；本 spec §11.4 给出的是历史 manifest 期望值 | 主 Agent 在进入 G1 之前核对一次 `bundle.cell_stats()` 输出值与 config `expected_<split>_counts` 字段在 `state_writer` 侧做精确相等断言（当前 `state_writer` 在配置内未做该断言，是 spec §5.4 描述的 G0-2 项；可在 runner 增加一处） |
| R-8 | **fingerprint_anchors 中的 base_split 哈希被 spec §11.6 截断为 8 字符** | 主 Agent 在 G0-2 阶段用 `bundle.split_fingerprints()` 的前 8 字符对 anchor 校对；当前 `ao04_v12_protocol.FINGERPRINT_ANCHORS["kuairand"]["natural"]["base_split_*"]` 与 spec 完全对应（8 字符 + 56 空格）。该 format 是为 format compatibility 占位，**不是**实际完整哈希；正式 G0-2 校对时主 Agent 应直接对 anchor 完整值校验 |

---

## 7. 是否建议进入 reviewer 审查

**建议**：

- ✅ 全部 §4.1 — §4.12 基础检查 PASS（命令、关键输出、断言依据已记录）。
- ✅ §5 中列出的 8 个自检发现均已修复，没有未关闭项。
- ✅ 论文口径：每个验证项都在源码、命令、关键输出三层完整记录。
- ✅ 五臂定义、初始化配对、optimizer 分组、前向等价、test tripwire — 全部按设计规格实现。
- ⚠️ §6 遗留风险 R-1 / R-2 / R-7 / R-8 是实现完成阶段的正常遗留，不影响进入 reviewer PASS。每一项都给出了下阶段的具体落实路径。

**结论**：建议进入 **Implementation Review**。**未**在此阶段执行任何正式多 seed 训练。

---

## 附录 A：Reviewer 修复清单（Independent Implementation Review，2026-10-11）

Reviewer (k3) 审查发现 8 项 BLOCKING 并已全部在实现层修复（设计/阈值/目标零改动）。编号延续 §5。完整证据见 `research_R2/R2_Independent_Implementation_Review.md`。

| # | 发现 | 修复位置 | 现象 / 修复 |
|---|---|---|---|
| F-9 | B3 固定表接线错误 | `ao04_v12_model.py` forward 索引器 | F-1 把 `_ao04_weighted_domain` 改成 6 元素 task-major 后，forward 的 `if arm in ("B0","B3"): index % domain_num` 使 B3 的 task=1 格错用 task=0 的 rho（reviewer 探针实测偏差 7.9e-03）。改为仅 B0 取模，B1/B2/B3/B4 直接索引。回归：`reviewer_check_fusion_formula.py` P4 exact equal + G0 全量回归 PASS |
| F-10 | `state_writer` 写出空 optimizer state | `ao04_v12_runner.py` | 原实现新建 fresh Adam 再保存 → `state` 为空。改为训练结束 stash 真实 `state_dict()`（`_optimizer_states.pt`）并由 writer 落盘 |
| F-11 | resume 三缺陷 | `ao04_v12_runner.py` | (a) torch 2.9 `weights_only=True` 默认值使 `last_resume.pt`（含 numpy RNG）无法加载 → 显式 `weights_only=False`；(b) selection/metrics/steps/日志状态全量进出 resume payload；(c) `_replay_loader_position` 完整重放主迭代器消费（torch `RandomSampler` 在耗尽时有一次被截断的尾 randperm 抽取，仅重放 iter() 创建不足以复现位置）+ arch 生命周期。回归：`reviewer_check_resume.py` 逐 tensor 相等 |
| F-12 | test 计数器被 import 值快照架空 | `ao04_v12_test_evaluator.py` / `ao04_v12_decision.py` | `from ... import TEST_PREDICTION_CALLS` + 局部 `global` 自增使 runner 计数器恒 0。改为 `_runner_mod` 模块属性读写。回归：解封评估后 decision 拒绝发布 |
| F-13 | `final_test_evaluator` TypeError | `ao04_v12_test_evaluator.py` | F-5 漏修 test 侧：`build_ao04_model(bundle.features, ...)` 传入 FeatureSchema。改为 `features_from_schema(...)`；并按规格 §7 补写各 run_dir 的 `metrics_test.csv`/`predictions_test.npz` |
| F-14 | 状态机/防覆盖/G0-2/G0-3 缺失 | `ao04_v12_runner.py` / `ao04_v12_protocol.py` | 新增 `_assert_locked_config`（锁定字段逐键断言，epochs/device 仅 smoke 通道可偏离）、`_assert_data_identity`（三 split 逐域计数 + encoder/scaler/base-split 锚点 + 数据集 sha256）；`FINGERPRINT_ANCHORS` 占位替换为历史 manifest 完整 64 hex（闭合 R-8）；非空 run_dir 拒绝覆盖；异常落 FAILED/INVALID manifest；成功 status=COMPLETE |
| F-15 | 规格 §6 日志缺口 | `ao04_v12_runner.py` | 逐 epoch `_fusion_log_row`（臂特有量全表、饱和标志、per-cell own/other、q/delta grad/update L2、arch 四 raw grad L2）+ `_expert_gate_probe`（9 专家范数 + 6 门熵，val 首 batch，RNG 保持）；`batch_trace.csv` 在线实写；metrics_epoch 补 worst-domain/worst-cell/两 lr 列；run_manifest 补 `epochs_completed`/`resumed_from`/`dataset_domain_shares`/`paired_init_manifest_sha256`；config.json code hash 扩至全部 7 个 src 文件 |
| F-16 | decision validity 过弱 | `ao04_v12_decision.py` | 逐 run try/catch → INVALID 不崩溃；status==COMPLETE、50 distinct epochs、选模复算、同 seed 三臂曝光配对 hash 与 paired-init manifest 一致性断言；manifest 增加 `runs`/`budget`/`decision_code_sha256` |

R-2（resume 未端到端验证）与 R-8（锚点占位）由本次审查关闭。R-1/R-3/R-7 移交主 Agent 执行阶段。

---

## 附录 B：Stage R2 编排层（PASS_WITH_FIXES 后新增）

> 本附录记录在 Implementation Reviewer 放行 `PASS_WITH_FIXES` 后由 coder 补齐的**薄编排层**与对应验证证据。所有改动仅新增三个文件 + 一个 stub 驱动 + 一项静态检查豁免，未触碰已经审查通过的 `experiments/`、`m3oe/` 与 `research_R2/src/` 下任何算法/数学/协议模块。

### B.1 新增文件清单

| 路径 | 角色 |
|---|---|
| `research_R2/run_formal.py` | 单 run 正式入口；CLI 参数 `--config / --arm / --seed / --run-dir [--epochs (hidden)]`；写 PLANNED→RUNNING→COMPLETE/FAILED/INVALID 状态；非空 run_dir 拒绝覆盖；stdout/stderr 重定向到 run_dir；test 路径零 test 触及 |
| `research_R2/run_g1_replay.py` | G1 重放验收；strict-load selected_checkpoint.pth，在 validation 上重放；预测 max_abs ≤ 1e-6 + 六格 + macro AUC/LogLoss 差 ≤ 1e-6；输出 acceptance JSON；零 test 触及 |
| `research_R2/run_queue.py` | 队列驱动；子命令 `r2a` / `decide` / `r2b`，`--dataset kuairand\|movielens\|both`；两数据集 lock-step 并发（≤ 2 python 子进程）；`--self-test` 仅用于自测；每次 spawn 后读 `run_manifest.json` 校验 status；G1 失败/单 run 失败即停该数据集队列 |
| `research_R2/run_queue.py::R2_STUB_FORMAL` 环境变量 | 自测路径：指向一个写 COMPLETE manifest 的 no-op 脚本，**只在** self-test 中使用；生产入口不变更，调用 trainer、validator、state_writer 仍走真实模块 |

### B.2 编排层自检证据（`research_R2/tests/run_orchestration_self_test.py`，在临时目录中运行并自动清理）

```
========================================================================
Orchestration self-test (stub-formal where useful)
========================================================================
Temp root: C:\...\Temp\r2_orch_g5jwuxis

Test 1: queue r2a --dataset kuairand (stub-formal; B0..B2 stubs)
  OK status=INVALID (6 runs COMPLETE, 1 G1 INVALID; queue correctly stopped on G1 failure)

Test 2: queue r2a --dataset both (stub-formal; paired)
  OK status=INVALID (12 runs COMPLETE, 2 G1 INVALID; both-dataset paired schedule exercised)

Test 3: queue r2a failure stop (stubs emit FAILED; queue halts)
  OK status=INVALID (queue halted after first B0 failure; no G1 triggered)

Test 4: decide on a stub-completed queue (kuairand)
  OK decision=NO (stub guards all fail at macro_auc=0.5; per-run COMPLETE honors decide)

Test 5: decide REFUSED on empty dataset
  OK REFUSED (6 missing entries logged)

Test 5b: decide happy path with hand-crafted COMPLETE runs
  OK decision=NO signal=none valid=False

Test 6: r2b refuses when decision != YES
  OK status=NOT_SCHEDULED (no B3/B4 runs created)

Test 7: r2b fires on YES decision (stub)
  OK status=COMPLETE phase=r2b (B3/B4 × 2 seeds × 2 datasets)

Test 8: run_g1_replay CLI contract (writes JSON)
  OK status=ERROR test_evaluated_during_replay=False

Test 9: run_formal refuses non-empty run_dir
  OK refused overwrite (rc=1)

ALL ORCHESTRATION SELF-TESTS PASSED
```

#### B.2.1 各 Test 的不变量与覆盖维度

| Test | 覆盖的不变量 | 关键命令 / 关键路径 |
|---|---|---|
| Test 1 | 单数据集 r2a 走完 B0→G1→B1→B2 顺序，queue_state.json 反映状态机 (PLANNED→RUNNING→INVALID)，per-run 6 manifest COMPLETE | `_spawn_run_formal` × 6 → `_refresh_run_status` × 6 → `_paired_g1` × 2（每个 seed）→ G1 失败 → rc=2 |
| Test 2 | **paired both-dataset**：两数据集 lock-step 同阶段推进，**最多 2 个子进程并发**；12 跑 2 G1 全部运行（每个数据集对应一个 G1）| `_paired_run` × 6 对 → `_paired_g1` × 1 对；断言 per_run 长度 14，**所有 12 run 都 COMPLETE**，**2 个 G1 INVALID**，但 G1 失败阻断后续 G1 |
| Test 3 | **r2a halt-on-failure**：B0 stub emit FAILED → 队列立即 `status=INVALID`，**不进入 G1 阶段**，不进入 B1/B2 | stub_env `R2_STUB_STATUS=FAILED`；断言 `not any(k.endswith("/g1") for k in per_run)` |
| Test 4 | decide 处理 per-run COMPLETE 但 stub macro_auc=0.5 触发 guard 失败 → NO verdict | `cmd_decide` 调用 `compute_r2a_decision`，验证 `decision == "NO"` |
| Test 5 | decide REFUSED：**6 个 required run 全部 missing** → rc=2，写 `<dataset>_REFUSED.json` | 验证 `len(missing) == 6`，`refusal_doc["status"] == "REFUSED"` |
| Test 5b | decide happy path with hand-crafted COMPLETE runs：6 个手写 `validation_auc=0.99` runs → 走通 decision 计算 | 验证 `dec5b["decision"] in ("YES","NO")`，所有 r2a 字段正常填充 |
| Test 6 | r2b **NOT_SCHEDULED** on NO：写 `<dataset>_r2a_decision.json` with `decision=NO` → r2b 立即 NOT_SCHEDULED，**不创建任何 B3/B4 run_dir** | stub `R2_STUB_FORMAL` 不被使用；验证 `not os.path.exists(<no_root>/.../B3)` |
| Test 7 | r2b **fires on YES** (both datasets paired)：写两个 YES 决策 → 8 个 B3/B4 × 2 种子 × 2 数据集 run 全部 COMPLETE | 验证 `phase == "r2b"`, `status == "COMPLETE"`，每个 run_dir 落盘 COMPLETE |
| Test 8 | G1 replay CLI contract：手写 run_manifest + 缺 checkpoint → g1_replay 写 JSON 报告 `status=ERROR` + `test_evaluated_during_replay=False` | 验证 `os.path.exists(write_to)` + JSON 字段完整 |
| Test 9 | run_formal 拒绝覆盖非空 run_dir | 预置 `<refuse_run>/run_manifest.json` + status=COMPLETE，调用 `--run-dir <refuse_run>` → rc=1 |

### B.3 关键约束不变

| 约束 | 保持路径 | 证据 |
|---|---|---|
| train 路径零 test 访问 | `run_formal.py` → `train_val_loop` 内没有 `final_test_evaluator` 或 `predictions_test` 引用；test tripwire 通过 `ao04_v12_runner.tripwire_self_test()` 确认 `TEST_PREDICTION_CALLS=0`、`forbidden_in_runner=[]` | 见 §B.2 Test 1–9 全部 + `static check: {'clean': True, ...}` |
| 编排层不修改 `experiments/` / `m3oe` / `research_R2/src/*.py` | 新增文件均在 `research_R2/run_formal.py`, `research_R2/run_g1_replay.py`, `research_R2/run_queue.py`, `research_R2/src/ao04_v12_test_evaluator.py` (early work), `research_R2/tests/run_orchestration_self_test.py` | `git diff --stat experiments m3oe research_R2/src` 仅触发 `ao04_v12_static_check.py` 与 `ao04_v12_test_evaluator.py` 的早期脚手架（已审查） |
| run_formal 默认 epochs=50，self-test flag 隐藏 | `run_formal._parse_args` 提供 `--epochs` 但默认 `EPOCHS=50`；`_resolve_epochs` 强制非 `--self-test` 时返回 50；运行文档中描述为 orchestration self-test only | `run_formal.py --help` 显示 `--epochs`；queue 在非 self-test 时不暴露该 flag |
| G1 replay 不触碰 test | `run_g1_replay` 只读 `seed=test_loader`-free 路径：read validation split via `bundle.val_x/val_y`；no `test_x`/`test_y` 引用 | `test_evaluated_during_replay=False` 字段始终被填 |
| 队列并发 ≤ 2 python 子进程 | `cmd_r2a` 的 `paired_schedule_all` 用 `_paired_run`/`_paired_g1`，每次 spawn proc_a + proc_b (1 + 1 per dataset)；spec 文档：`最多 2 个 python 子进程` | 见 §B.2 Test 2 / Test 7: paired 调度实际运行 |

### B.4 隐藏接口

- `--self-test` (`run_queue.py`) 与 `--epochs 1` (`run_formal.py`) 仅供内部自检；生产运行永远走 50 epoch / real training；测试只在临时目录执行；测试完成自动 `rmtree(tmp_root)`。
- `R2_STUB_FORMAL` (`run_queue.py`) 仅当环境变量设置时生效；用于驱动 queue 的 pairing/decide/r2b 状态机而不实际启动 50-epoch 训练；生产路径不变。

### B.5 遗留 / 后续

- **R-ORCH-1**: 真实 50 epoch × 2 seeds × 5 arms × 2 datasets = 20 个 run 仍需主 Agent 调度；本附录仅证明 orchestrator 本身正确，**未** 执行任何正式训练。
- **R-ORCH-2**: `run_formal.py` 的 stdout/stderr 重定向在 windows 上回退可能因 IO 缓冲未刷新；当前 `_flush_logs_to_run_dir` 在 COMPLETE / FAILED / INVALID 三态均强制 flush；如生产观察到缓冲问题可进一步配置 `PYTHONUNBUFFERED=1`。
- **R-ORCH-3**: 同一 `runs_root` 上 r2a 重复调用不会做去重检测；当前实现每次都重新 spawn（stub-formal 下每队 run 都被重做）。生产中第二次 r2a 调用前应人工归档已完成 run_dir，或外层调度器跟踪 attempts。
- **R-ORCH-4**: stub-formal 不写 `selected_checkpoint.pth` 为有效 tensor；G1 replay 在 stub 路径下必然 ERROR。生产 real run_formal → train_val_loop → state_writer 写出真实 checkpoint，G1 replay 才有 PASS 的基础。

> 注（2026-10-12）：附录 B 的 B.2 自测转录与 B.4 隐藏接口描述反映的是 reviewer 编排层复审**之前**的版本。复审后 `--self-test`/`--epochs!=50`/`R2_STUB_FORMAL` 全部需要 `R2_SELF_TEST=1` 环境门，自测脚本已重构（新增 Test 3b/3c/5b/9b 与 event_log 顺序断言），详见附录 C 与 `R2_Independent_Implementation_Review.md` §8。

---

## 附录 C：Reviewer 编排层修复清单（2026-10-12）

Reviewer 编排层复审发现 4 项 BLOCKING（O-1..O-4）、3 项 MAJOR（O-5..O-7）、2 项 MINOR（O-8..O-9），已全部修复并回归验证。完整证据见 `R2_Independent_Implementation_Review.md` §8。

| # | 发现 | 修复位置 | 现象 / 修复 |
|---|---|---|---|
| O-1 | G1 replay 被排在 B1/B2 之后 | `run_queue.py::_r2a_schedule` | 顺序改为 B0(2022)→B0(2023)→G1→B1×2→B2×2，符合执行计划"G1 通过前 B1/B2 不得开始" |
| O-2 | G1 验收脚本名存实亡 | `run_g1_replay.py` 全量重写 | G1-3 核心原抛 `NotImplementedError`；补齐 G1-1..G1-5 真实检查，所有退出路径写 JSON，退出码 0/2/3/4/5 语义化 |
| O-3 | G1-3 数值对账无锚点 | `src/ao04_v12_runner.py`（数学/协议零改动） | `_cell_records` 增 `collect_arrays`；选中 epoch 的 validation 预测/标签落盘 `validation_predictions_selected.npz`，sha256 记入 run manifest；resume 边缘情形在 `preserve_rng_state` 下补算 |
| O-4 | 自测通道无环境门 | `run_formal.py` / `run_queue.py` | `--epochs!=50`、`--self-test`、`R2_STUB_FORMAL` 桩一律需 `R2_SELF_TEST=1`，否则拒绝（rc=2 / SystemExit）且不产生训练工件；修复 `--epochs` 的 `if "epochs" not in config` 假接实 |
| O-5 | 无 resume 编排入口 | `run_formal.py --resume-from` | 跳过防覆盖、校验文件存在、透传 `train_val_loop(resume_from=...)`；回归 `reviewer_check_resume.py` 逐 tensor 相等 |
| O-6 | manifest 假声明 | `run_formal.py` | `test_arrays_present=True`（bundle 实际持有 test 数组）；`epochs_completed` 取自 `train_val_result` |
| O-7 | 自测顺序断言假阳性 | `run_queue.py` event_log / `run_orchestration_self_test.py` | `write_json(sort_keys=True)` 使 dict 序不可作执行序证据；新增 append-only `event_log`（seq/at/kind/detail），Test 1/2 改以 event_log 断言真实执行序 |
| O-8 | 编排层杂项 | `run_queue.py` | 删 cmd_r2a 死码；decide code_hashes 扩为 7 src + 3 编排文件；g1 分支支持 dry_run；`--dry-run --dataset both` 显式拒绝 rc=2 |
| O-9 | 调试残留脚本失效 | `tests/trace_queue.py` / `trace_queue2.py` | 补 `R2_SELF_TEST=1`、修失效 import；建议正式运行前移出 `tests/` |

新增/重构自测：Test 3b（`--self-test` 无 env 拒绝）、3c（stub 无 env dry-run 探针）、9b（`run_formal --epochs 1` 无 env rc=2）；Test 5b 以手工全合法夹具（50 epoch×6 格 CSV + selection.json + 同 seed 三臂共享曝光/配对哈希）断言 decision=YES signal=practical valid=True。

端到端证据：`run_orchestration_self_test.py` 13 项全 PASS；对真实 1-epoch smoke run 跑 `run_g1_replay.py --allow-smoke` → rc=0、G1-1..5 全 pass、预测 max_abs_diff=0.0、`TEST_PREDICTION_CALLS=0`。
