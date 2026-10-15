# R2 Independent Implementation Review

- **Reviewer**: Independent Implementation Reviewer (k3)
- **Date**: 2026-10-11
- **Scope**: `research_R2/` — Stage R2 (AO-04 v1.2) 五臂融合实验的实现审查
- **Contract documents**: `research_R1_R3/R1_Experiment_Charter_v1.2.md`、`research_R1_R3/R2_Execution_Plan.md`、`research_R2/R2_Design_Specification.md`、`research_R2/R2_Implementation_Report.md`
- **Verdict**: **PASS_WITH_FIXES**（全部 BLOCKING 已修复并回归验证；未执行正式训练，符合 reviewer 权限边界）

---

## 1. 审查范围与方法

1. 通读四份合同文档与 `research_R2/src/`（7 个模块）、`research_R2/configs/`（2 个配置）、`research_R2/tests/`（coder 的 4 个脚本）。
2. 与 Stage S1 只读来源逐段对照（`experiments/stage_S1_run.py` / `stage_S1_models.py` / `stage_S1_data.py` / `stage_S1_metrics.py` / `stage_E1_instrumentation.py`，只读，未修改任何 `experiments/`、`m3oe/` 文件）。
3. 重跑 coder 全部验证脚本；另自建 4 个 reviewer 探针（见 §4 命令清单），覆盖 coder 探针判别力不足的路径（B1 对称盲区、B3 接线、resume 等价、test 计数器管路）。
4. 发现 BLOCKING 后在 reviewer 权限内直接修复实现（不改设计/阈值/目标），修复项同步追加到 `R2_Implementation_Report.md` 附录。

环境：`venv/Scripts/python.exe`（Python 3.13.5，torch 2.9.0+cu126，CUDA 可用，numpy 2.1.3），与规格锁定值一致。两个数据集 CSV 的 SHA256 已核对（KuaiRand `a1872911…c681f`、MovieLens `f060ecb8…b5e4`）。B3 表再生核验通过（`RandomState(10404).uniform(0.1,0.9,6)`，json SHA256 `182edcbe…9957`）。

---

## 2. BLOCKING 发现与修复

### B-1：B3 固定表接线错误（模型 forward）

- **现象**：`ao04_v12_model.py` 的 `forward` 中索引器写为 `if self.arm in ("B0", "B3"): wd_indexer = weighted_domain[index % domain_num]`。B0 的 `weighted_domain` 是 3 元素共享表，`% 3` 正确；但 B3 的 `_ao04_weighted_domain` 自 coder F-1 修复后返回 **6 元素 task-major 列表**，`% 3` 使 task=1 的三格（cell 3/4/5）错用 task=0 的 rho 表值（rho[0..2] 替代 rho[3..5]）。B1/B2/B4 在 F-1 中已改为直接索引，唯独 B3 被遗漏——而 B3 又被规格排除在 G0-5 前向等价门之外、coder 的非对称探针只测 B1，因此漏网。
- **证据**：reviewer 公式探针（`tests/reviewer_check_fusion_formula.py` P4）以非均匀表独立重算 R_dt，修复前 task=1 格输出偏差最大 7.9e-03（远超 1e-6 容差）；修复后 exact equal。j/k 交换对照实验证明探针有判别力。
- **修复**：索引器改为仅 B0 用 `% domain_num`，B1/B2/B3/B4 一律直接索引。
- **回归**：`reviewer_check_fusion_formula.py` OVERALL PASS；`run_g0_forward_equivalence.py` 全部 PASS（§3）。

### B-2：`state_writer` 写出空 optimizer state

- **现象**：`state_writer` 用 `build_optimizers_for_arm(model)` **新建** Adam 再 `torch.save(opt.state_dict())` → `optimizer_state.pth` / `architecture_optimizer_state.pth` 的 `state` 为空 dict。规格 §7 要求"训练结束时 main optimizer"状态；空 state 使该工件失去审计价值且误导 resume。
- **修复**：`train_val_loop` 结束时把真实 `main_opt/arch_opt.state_dict()` stash 到 `_optimizer_states.pt`；`state_writer` 读取该 stash 写出两份正式工件。
- **回归**：1-epoch smoke 内容断言 `optimizer_state_nonempty=True`、`arch_optimizer_state_nonempty=True`（§3 命令 5）。

### B-3：resume 路径完全不可用（三处缺陷）

- **B-3a**：torch 2.9 起 `torch.load` 默认 `weights_only=True`，`last_resume.pt` 合法包含 numpy RNG state（tuple+ndarray）→ resume 必抛 `UnpicklingError`。**修复**：resume 加载显式 `weights_only=False`（文件由本模块自写自读）。
- **B-3b**：resume 只恢复 model/optimizer/epoch，丢失 selection 状态、逐 epoch 指标、步数计数、日志行 → 续训后选模与日志全错。**修复**：全量状态进 `last_resume.pt` 并在 resume 时恢复。
- **B-3c**：train loader 的显式 `torch.Generator` 位置与 persistent arch 迭代器未恢复。**这是最深的一处**：torch 的 `RandomSampler.__iter__` 在 `num_samples % n == 0` 时仍会多抽一次被截断为空的 `randperm`，且该抽取发生在迭代器**耗尽时**而非创建时（reviewer 微实验实测：每 epoch 完整迭代的 generator 状态变化发生在 iter()/batch1/batch5 三处）。因此"每 epoch 重放一次 iter() 创建"的重放方案（reviewer 的第一版实现）无法复现位置。**修复**：`_replay_loader_position` 按真实循环**完整消费**主迭代器（含 smoke 截断语义）+ 逐 epoch 重放 arch 生命周期，逐 batch 镜像真实调用序列。
- **回归**：`tests/reviewer_check_resume.py`（MovieLens B1 seed 2022，CPU 决定性环境，1 epoch 中断 + resume vs 一次性 2 epoch 参照）——`batch_order_hash`、`architecture_batch_order_hash`、`macro_auc_by_epoch`、selected_epoch、末态全部 tensor `torch.equal`、q_rho 的 Adam `exp_avg` 逐项相等，**全部 OK**。common-epoch checkpoint 改为捕获时立即落盘（`common_epoch_{e}.pth`），resume 时从磁盘装回。

### B-4：test tripwire 计数器被 import 语义架空

- **现象**：`ao04_v12_test_evaluator.py` 用 `from ao04_v12_runner import TEST_PREDICTION_CALLS` 导入**值快照**，再用自己的 `global` 自增 → `ao04_v12_runner.TEST_PREDICTION_CALLS` 永远为 0 → decision 模块的"决策前计数器必须为 0"断言形同虚设。
- **修复**：evaluator 与 decision 均改为 `import ao04_v12_runner as _runner_mod` 并读写模块属性；删除 evaluator 内的影子 `_bump_test_counter`。
- **回归**：`tests/reviewer_check_test_evaluator.py`——解封评估后 `TEST_PREDICTION_CALLS=2`，decision 随即 `RuntimeError` 拒绝发布（§3 命令 7）。

### B-5：解封评估器根本无法运行

- **现象**：`final_test_evaluator` 调 `build_ao04_model(bundle.features, ...)` 传入 `FeatureSchema` dataclass 而非 feature list → 必抛 `TypeError: 'FeatureSchema' object is not iterable`（coder 的 F-5 修了三处，漏了 test 侧这处）。
- **修复**：改用 `features_from_schema(bundle.features)`。
- **回归**：同上检查脚本——带临时解封 manifest 端到端跑通，聚合与 per-run 工件均写出（规格 §7 要求的 run 内 `metrics_test.csv` / `predictions_test.npz` 已补）。

### B-6：状态机与防覆盖缺失

- **现象**：`run_manifest.json` 的 `status` 恒为 `"OK"`（规格 §7 要求 PLANNED/RUNNING/COMPLETE/FAILED/INVALID/INCOMPLETE 状态机）；训练异常不留痕；非空 run_dir 被静默覆盖；规格 G0-2（数据身份断言）与 G0-3（锁定配置断言）在 runner 内完全缺失。
- **修复**：
  - `train_val_loop` 外壳：非空 run_dir 拒绝写入（resume 除外）；异常时写 FAILED/INVALID manifest 后再抛出；成功路径 status=`COMPLETE`。
  - 新增 `_assert_locked_config`（G0-3）：config 中任何出现的锁定字段必须等于协议常量；`epochs`/`device` 仅在 smoke 通道（`smoke_max_batches` 显式传入）允许偏离。
  - 新增 `_assert_data_identity`（G0-2）：三 split 逐域行数 == `EXPECTED_COUNTS`；encoder/scaler hash 与 base split fingerprint 对照历史锚点；config 携带 `dataset_sha256` 时对原始文件做字节级 SHA256。锚点 8 字符占位已替换为从历史 manifest 取出的完整 64 hex（两数据集 × encoder/scaler/train/val/test 共 10 值，逐一核对，闭合 R-8）。
- **回归**：1-epoch smoke 与 resume 测试均在新断言下通过（锚点断言真实生效而非空转）。

### B-7：规格 §6 日志大面积缺失

- **现象**：`fusion_log.csv` 只在 run 末写 1 行（8 个全局系数），无逐 epoch 轨迹、无臂特有量（q/rho/delta/b_dt）、无饱和标志、无 per-cell own/other 快照；§6.6 专家范数/门熵 probe 完全未实现；§6.7 的 q/delta 梯度与 update 范数、arch step 四 raw 梯度范数未记录；`batch_trace.csv` 恒为空（`_build_batch_trace_rows` 返回 `[]`）；`metrics_epoch.csv` 缺 `worst_domain_validation_auc`/`worst_cell_validation_auc`/`learning_rate`/`architecture_learning_rate`。
- **修复**：训练循环在线捕获——每 epoch `_fusion_log_row`（raw/sigmoid 四全局量 + B1 q/rho 全表与均值 + B2 q_d/rho_d 与共享标记 + B3 固定表 + B4 delta/均值/中心化 delta/有效 b_dt + per-cell own/other JSON + 饱和标志（<0.01 或 >0.99 只标记不 clamp）+ 末批 q/delta grad/update L2 + arch 四 raw grad L2）与 `_expert_gate_probe`（val 首个 batch、eval+no_grad+preserve_rng_state，9 专家输出批量均 L2 + 6 门 softmax 熵均值）；逐 batch `batch_rows`（fingerprint/domain_counts/objective，train 与 arch 分开）。`state_writer` 按 epoch 合并写 `fusion_log.csv`、直接写 `batch_trace.csv`；metrics_epoch 补齐四列；run_manifest 增加 `epochs_completed`、`resumed_from`、`dataset_domain_shares`、`paired_init_manifest_sha256`；config.json 的 code hash 扩到全部 7 个 src 文件。
- **回归**：1-epoch smoke 内容断言 `fusion_rows=1`（==epoch 数）、`batch_rows=50/6`（train+arch）、`probe_cols=15`（9+6）。

### B-8：decision 模块 validity 过弱

- **现象**：`compute_r2a_decision` 只数 run 个数；不查 run 状态机、不核 50 epoch 覆盖、不复算选模、不核同 seed 三臂曝光配对、不核 paired-init 一致性；任一 run 文件缺失直接抛异常而非判 INVALID。
- **修复**：逐 run try/catch 收集 + status==COMPLETE 断言 + metrics_epoch 恰好 50 个 distinct epoch + 用 `selection.json` 的 `macro_auc_by_epoch` 以 `select_checkpoint_epoch` 复算==selected_epoch + 同 seed 三臂 `batch_order_hash`/`architecture_batch_order_hash`/`paired_init_manifest_sha256` 两两相等；decision manifest 增加 `runs` 明细（formal_run_key/manifest sha256/selected epoch 全精度 AUC）、`budget` 块（BUDGET_PER_DATASET + attempts/epochs/resumes 实际用量）、`decision_code_sha256` 自填。
- **回归**：`tests/reviewer_check_decision.py`——空目录/坏目录均优雅 INVALID（不崩溃），计数器门生效。

### B-9（reviewer 自引入并已修复）：`_reconstruct_arm` allow-list 逻辑错误

reviewer 新增的配对断言初版把 B1/B2/B4 的新参数误判为"意外缺失键"（先查 buffer 后查 allow-list 的顺序错误）。`reviewer_check_resume.py` 首轮即捕获，已修为单次 allow-list 判定；该路径此前从未被 coder 测试触达（coder 的 smoke 只跑 B0）。

---

## 3. 验证命令与关键输出（全部可复跑）

| # | 命令（项目根，`venv/Scripts/python.exe`） | 结果 |
|---|---|---|
| 1 | `python -m py_compile research_R2/src/*.py research_R2/tests/*.py` | 通过 |
| 2 | `research_R2/tests/reviewer_check_fusion_formula.py`（reviewer 新增） | OVERALL PASS：B0==父类 `torch.equal`；B1/B2/B4 非均匀 q/delta 独立重算 exact/max 6e-8；B3 修复后 task-major 表 exact equal；j/k 交换对照证明探针有判别力 |
| 3 | `research_R2/tests/run_g0_forward_equivalence.py`（242s） | 全 PASS：G0-5 synthetic+real 前向等价（max_abs ≤1.2e-7）；G0-8 参数总数两数据集×五臂精确匹配；G0-7 optimizer 分组（kuairand 134+4=138 等）；B1 非对称探针；曝光配对五臂两两相等；tripwire clean |
| 4 | `research_R2/tests/run_smoke.py` | 两数据集×五臂 loss/梯度有限，q/delta 与四 raw 均实际更新 |
| 5 | `research_R2/tests/run_smoke_1epoch.py`（已适配 smoke 通道与内容断言） | 两数据集 B0 端到端 PASS；11 类工件齐全；optimizer state 非空；fusion/batch/probe 行数正确；manifest status=COMPLETE 且无 test 字段 |
| 6 | `research_R2/tests/reviewer_check_resume.py`（reviewer 新增） | 全 OK：resume 后与一次性参照**逐 tensor 相等**（见 B-3） |
| 7 | `research_R2/tests/reviewer_check_test_evaluator.py <run_dir>`（reviewer 新增） | 解封门 FileNotFoundError；端到端评估通过；per-run 工件写出；计数器=2 且 decision 拒绝 |
| 8 | `research_R2/tests/reviewer_check_decision.py`（reviewer 新增） | 缺 run/坏 run → INVALID 不崩溃；计数器门生效 |

数值复核：KuaiRand 1-epoch smoke macro_auc=0.5621（49 main steps + 1 arch step）；MovieLens 0.6359（5+1）；与 coder 报告 §4.11 的量级一致。

---

## 4. 已确认无问题的部分（抽查不再列证据）

- S1 对照：loss/优化顺序、arch 更新协议（persistent iterator + preserve_rng_state）、选模 strict>+earliest、`_cell_records`、`_domain_weighted`（÷TASK_NUM = W 定义）、`setup_seed`、LOGLOSS_EPS=1e-12 一致。
- 数据装载：KuaiRand split-first + stored labels、encoder 仅用 base train、OOV=0、train counts 内建断言；MovieLens 旋转 `RandomState(seed*10+domain)`、7 sparse+1 dense、构造标签。
- configs 两文件与规格 §5.1 逐字段一致；`fusion_coefficients` 的 B2 task-major 展开正确；decision 的 G-a..G-f 方向、D/T 阈值与 YES 布尔条件正确（实现层 bug 已单列修复）。
- B3 buffer 不入任何 optimizer（named_parameters 不含 buffer），且在 `_reconstruct_arm` 中与固定表逐元素断言。

---

## 5. MINOR / 记录项

- M-1：test 聚合工件写在 `runs/<arm>/` 层级的 `_final_test_evaluated/`（由 run_dir 推导），per-run 工件已按规格 §7 补齐进各 run_dir；聚合目录位置属展示层选择，不阻塞。
- M-2：coder 的 B1 探针只查梯度非零（q=0 时对称，看不见 j/k 交换），判别力不足；reviewer 公式探针已替代其职能，coder 探针保留无害。
- M-3：`run_smoke_1epoch.py` 原以 `epochs=1` 裸跑，在新增 G0-3 锁定断言下必须走 smoke 通道（`smoke_max_batches=10**9`，不截断 batch）；已适配并注明。

## 6. 剩余风险（不阻塞，移交主 Agent）

- R-R1：**正式 50 epoch × 2 seed × 5 臂 × 2 数据集未执行**（reviewer 权限禁止）。G1 门槛仍需在正式 B1/B2 训练前按规格 §9 执行。
- R-R2：resume 逐 bit 等价在 CPU 验证；CUDA 下 embedding backward 的 atomicAdd 非决定性意味着 CUDA 上 resume 只能保证结构性一致（同 S1 既有限制）。正式预算允许每 run ≤1 次 resume，决策侧已有 pairing/selection 复算作最后防线。
- R-R3：decision 的 VALID 全通路（6 个真实 COMPLETE run）只能等正式训练后演练；当前已对缺/坏/状态异常路径充分覆盖。
- R-R4：`global_test_unlock.json` 的完整字段（signed_by 等）仍待主 Agent 在锁定阶段填齐（沿 R-3）。

## 7. Verdict

**PASS_WITH_FIXES**。

理由：发现的 8 项 BLOCKING（B-1..B-8，另加 reviewer 自引入立即修复的 B-9）全部在实现层修复完成，且每一项都有可复跑的回归证据（§3 命令 1–8 全部 PASS）。设计、阈值、目标、数据、旧阶段文件零改动；未执行正式训练；未创建任何基线真值文件。修复明细已追加至 `R2_Implementation_Report.md` 附录 A（F-9..F-16）。实现现已满足进入正式 R2-A 队列的代码条件，剩余事项均为执行阶段事项（§6）。

---

## 8. 编排层复审（2026-10-12 追加）

> coder 在 PASS_WITH_FIXES 后补齐薄编排层。本轮按主 Agent 复审清单逐条核验；发现的问题已在 reviewer 权限内修复并回归验证。本节结构：Reviewed files / Problems found / Fixes applied / Remaining risks / Ready for runner。

### 8.1 Reviewed files

- `research_R2/run_formal.py`（单 run 正式入口）
- `research_R2/run_g1_replay.py`（G1 reload-replay 验收）
- `research_R2/run_queue.py`（队列驱动：r2a / decide / r2b，两数据集 paired ≤2 进程）
- `research_R2/tests/run_orchestration_self_test.py`、`trace_queue.py`、`trace_queue2.py`
- `research_R2/R2_Implementation_Report.md` 附录 B
- 联动核验：`research_R2/src/ao04_v12_runner.py`（G1-3 预测留痕依赖）、`src/ao04_v12_decision.py`（§2 B-8）、`src/ao04_v12_test_evaluator.py`（§2 B-4/B-5）

### 8.2 Problems found

- **O-1（BLOCKING）G1 调度顺序错误**：原 `_r2a_schedule` 为 B0×2→B1×2→B2×2→G1，G1 replay 被排在 B1/B2 **之后**，直接违反执行计划 §5/§6"G1 通过前 B1/B2 不得开始"。若 B0 checkpoint 不可重放，B1/B2 已消耗的 4 个 attempt 预算将被污染。
- **O-2（BLOCKING）G1 验收脚本名存实亡**：`run_g1_replay.py` 的 G1-3 重放核心抛 `NotImplementedError`（任何真实 run 上 G1 必崩）；`initial_hash_replay_match` 系伪断言；G1-1（完整性+选模复算）、G1-4（paired-init 对账）、G1-5（test tripwire 计数）缺真实实现；异常退出路径不写 JSON 报告，队列读不到 `status=PASS` 之外的结构化证据。
- **O-3（BLOCKING）G1-3 数值对账无锚点**：runner 从未留存选中 epoch 的 validation 原始预测数组，"strict load + 重放对账 ≤1e-6"没有可对账的工件——即使 O-2 修好也只能空转。
- **O-4（BLOCKING）自测通道无环境门**：`run_formal --epochs` 用 `if "epochs" not in config` 合并（config 恒含 `epochs=50`）→ 旗标假接实、静默无效；队列的 `R2_STUB_FORMAL` 桩只看变量是否存在；`--self-test` 无门槛。缩短训练/桩替换可在正式 shell 中无声发生。
- **O-5（MAJOR）无 resume 编排入口**：执行计划 §9 允许每 run ≤1 次精确 resume，`run_formal.py` 无 `--resume-from`，中断后只能整 run 重跑（违反预算语义）。
- **O-6（MAJOR）manifest 假声明**：`state_writer(..., test_arrays_present=False)` 与事实不符（bundle 实际持有 test 数组，封存语义是"不评价"而非"不存在"）；`epochs_completed` 未取自 `train_val_result`。
- **O-7（MAJOR）自测顺序断言假阳性**：`queue_state.json` 经 `write_json(sort_keys=True)` 序列化，自测以 per_run dict 顺序断言执行顺序——该断言永真，无法捕获 O-1 类回归。
- **O-8（MINOR）**：`cmd_r2a` 残留死码（`paired_schedules`/`_split`/`paired_run_schedule`/`paired_gate_schedule`）；`decide` 的 code_hashes 只覆盖 7 个 src，未含编排层三文件；`--dry-run --dataset both` 静默走 paired 实跑路径；单数据集 g1 分支无 dry_run 处理。
- **O-9（MINOR）**：`trace_queue.py`/`trace_queue2.py` 为 coder 调试残留，环境门引入后因缺 `R2_SELF_TEST` 失效（后者另有失效 import）。

### 8.3 Fixes applied（逐项回归验证）

- **`run_queue.py`**：`_r2a_schedule` 改为 B0(2022)→B0(2023)→G1→B1×2→B2×2（O-1）；`_spawn_run_formal` 的 stub 仅在 `R2_SELF_TEST` 设置时生效（O-4）；`_resolve_epochs` 要求 `--self-test` 必须配 `R2_SELF_TEST=1`，否则 `SystemExit`（O-4）；删除 cmd_r2a 死码、decide 的 code_hashes 扩为 7 src + 3 编排文件（O-8）；新增 append-only `event_log`（seq/at/kind/detail）作为执行顺序的唯一权威证据（O-7）；g1 分支支持 dry_run；`--dry-run --dataset both` 显式拒绝 rc=2（O-8）。
- **`run_g1_replay.py` 全量重写**（O-2/O-3）：G1-1（status==COMPLETE、`epochs_completed`==计划值、selection.json 用 `select_checkpoint_epoch` 离线复算==记录值、metrics_epoch.csv 行数==epochs×6）；G1-2（12 类工件 + common_epoch_{9,19,29,39,49} 齐全、逐件 SHA256、npz sha 与 manifest `selected_validation_predictions_sha256` 对账）；G1-3（strict load `selected_checkpoint.pth`、按 run config 记录的设备同设备重放、预测 max_abs≤1e-6 对 npz、六格与 macro AUC/LogLoss |diff|≤1e-6 对 metrics_epoch.csv、labels 完全相等）；G1-4（init 三件套存在性 + `paired_init_manifest_sha256` 对账 + B0 的 `initial_parameter_state_hash`==init manifest 的 `initial_state_hash` + 曝光四字段 + 兄弟 seed B0 COMPLETE 且 epoch 数一致）；G1-5（运行期读 `ao04_v12_runner.TEST_PREDICTION_CALLS`==0 + manifest 两个 test 标志==False）。所有退出路径经 `_emit` 写 JSON；退出码 0=PASS/2=参数/3=ERROR/4=INVALID/5=FAIL；隐藏旗 `--allow-smoke` 仅供自测（队列从不传递）；严格模式限 arm==B0 且 seed 匹配 manifest。
- **`run_formal.py`**（O-4/O-5/O-6）：新增 `--resume-from`（跳过防覆盖、校验文件存在、透传 `train_val_loop(resume_from=...)`）；`--epochs != 50` 需 `R2_SELF_TEST=1` 否则 rc=2 且不产生训练工件，smoke 通道打印 WARNING 并以 `config_for_runner["epochs"]=epochs` 无条件覆盖（修复假接实）；`test_arrays_present=True`；`epochs_completed` 取自 `train_val_result`。
- **`src/ao04_v12_runner.py`**（O-3，编排层需求驱动的 runner 补齐，数学/协议零改动）：`_cell_records` 增 `collect_arrays` 可选返回拼接预测/标签数组；验证调用改三元解包，`improved` 分支留存当 epoch 数组；循环末若数组缺失（resume 后无改善的边缘情形）在 `preserve_rng_state()` 下对已恢复的 selected state 补算一次；`np.savez_compressed` 暂存 `_val_pred_selected.npz`，`state_writer` 原子改名 `validation_predictions_selected.npz` 并把 sha256 记入 manifest。新标识符经 `ao04_v12_static_check` 禁用词扫描确认不引入 test 耦合。
- **`tests/run_orchestration_self_test.py` 重构**（O-7 及新增门回归）：stub 环境统一加 `R2_SELF_TEST=1`；Test 1/2 改以 `event_log` 断言真实执行序（`run_start/run_end/g1_start/g1_end`）并断言 B1/B2 目录不存在；新增 Test 3b（`--self-test` 无 env 拒绝）、3c（stub 无 env 的 dry-run 探针）、9b（`run_formal --epochs 1` 无 env rc=2 且无工件）；Test 4 直接调 stub×6 后 decide（缺 metrics/selection → decision=NO valid=False）；新增 Test 5b 以 `_write_decision_grade_run` 造全合法夹具（50 epoch×6 格 CSV + selection.json + 同 seed 三臂共享 boh/aboh/pim 哈希），断言 decision=YES signal=practical valid=True。`trace_queue.py`/`trace_queue2.py` 补 `R2_SELF_TEST=1` 并修失效 import（O-9）。

### 8.4 验证证据（本会话全部实跑，可复跑）

| # | 命令（项目根，`venv/Scripts/python.exe`） | 结果 |
|---|---|---|
| O-v1 | `research_R2/tests/run_orchestration_self_test.py` | **ALL ORCHESTRATION SELF-TESTS PASSED**（13 项，含 3b/3c/5b/9b 新门与 event_log 顺序断言） |
| O-v2 | `research_R2/tests/reviewer_check_resume.py` | RESUME CHECK: PASS（8 项逐 tensor 相等，含 `--resume-from` 链路） |
| O-v3 | `research_R2/tests/reviewer_check_decision.py` | DECISION CHECK: PASS（缺/坏目录 INVALID 不崩溃；计数器门生效） |
| O-v4 | `research_R2/tests/reviewer_check_fusion_formula.py` | OVERALL PASS（runner 改动未触碰数学路径） |
| O-v5 | `research_R2/tests/reviewer_check_test_evaluator.py <run_dir>` | TEST EVALUATOR CHECK: PASS（解封门、per-run 工件、计数器=2 后 decision 拒绝） |
| O-v6 | `research_R2/tests/run_smoke_1epoch.py` | 两数据集 PASS；`validation_predictions_selected.npz` 落盘，manifest sha256 与文件字节级一致 |
| O-v7 | `research_R2/tests/run_smoke.py` | 两数据集×五臂 10/10 OK（loss/梯度有限，q/delta 与四 raw 实际更新） |
| O-v8 | `research_R2/run_g1_replay.py --config configs/kuairand_natural_m3oe.json --run-dir <smoke B0_seed2022> --seed 2022 --allow-smoke` | rc=0，status=**PASS**：G1-1..G1-5 全 pass，`max_prediction_abs_diff=0.0`、labels_equal=True、macro/cell diff 全 0、`TEST_PREDICTION_CALLS=0`（真实数值重放的端到端证据） |
| O-v9 | `python -m py_compile` 全部改动文件 | 通过 |

### 8.5 Remaining risks（不阻塞，移交主 Agent）

- **R-O1**：G1-3 的 1e-6 容差按 run config 记录的设备做同设备重放（CUDA 跨进程实测 diff=0.0，仅 smoke 规模）；正式 50 epoch 后首次 G1 仍需按执行计划 §5 复核。
- **R-O2**：单数据集 r2a 的 G1 在某 seed 失败时不补跑另一 seed 的 G1（fail-closed，符合规格停队列语义）；`r2b` 路径未打 event_log（执行序证据目前只覆盖 r2a）。
- **R-O3**：`r2b` 在 spawn 前未把当前 code hashes 与 decision manifest 对账（执行计划 §7"再次确认实现未按 A 结果改动"目前依赖 decide 时记录 + 文件静止）；建议主 Agent 在正式 B 阶段启动前人工核对一次。
- **R-O4**：`trace_queue.py`/`trace_queue2.py` 为开发调试残留，建议正式运行前移出 `tests/` 或删除。
- **R-O5**：正式 50 epoch × 6 run × 2 数据集未执行（reviewer 权限边界，沿 §6 R-R1）。

### 8.6 编排层 Verdict

**Ready for runner: YES**（编排层维度）。

理由：O-1..O-9 全部修复并各有可复跑回归（§8.4 O-v1..O-v9）；编排层不含算法/数学/协议逻辑的平行实现（run_formal 仅调 `paired_init_phase`/`train_val_loop`/`state_writer`；run_g1_replay 的重放循环是验收对账所需的独立重算，指标函数复用 `stage_S1_metrics`）；test 封存在编排层保持（train/decide 路径 TEST_PREDICTION_CALLS=0 有 O-v1/O-v8 双重证据，唯一 test 入口仍需 `global_test_unlock.json`）；队列顺序、并发上限、decide fail-closed、r2b YES 门与 NOT_SCHEDULED 语义均有自测断言。正式训练仍需人类另行授权与 G0 数据/runtime 核验，不在本判断范围内。
