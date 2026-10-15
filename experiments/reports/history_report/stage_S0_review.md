# REVIEW REPORT — Review 1/2 (Archived)

## 1. Verdict: FAIL

这是第 1 次审核（最多允许 2 次）。当前实现**不得进入正式 84-run**。

核心数学路径总体符合 Stage 0 设计，84 个任务矩阵也正确；但当前形式化运行器仍允许若干协议偏差在返回成功状态后被接纳，包括关闭 instrumentation、改变关键超参数、训练后出现 NaN/Inf，以及汇总时静默丢弃非有限指标。架构参数更新所使用的 batch 也没有进入配对证据链。以上问题可导致“脚本成功结束，但实验并未按设计执行”，属于 blocking。

本审核没有运行 84-run，也没有分析任何研究结论。

## 2. Requirement-by-Requirement Verification

| # | 设计要求 | 结论 | 独立证据 |
|---|---|---|---|
| 1 | 恰好 84 个唯一 joint tasks，每 arm 12 个 | PASS | `stage_S0_orchestrator.py:81-96` 构造 `7×3×4` 并检查唯一键；独立 dry-run 显示 84。 |
| 2 | A/B 均为 fresh runs | PASS（代码层） | A/B 都由同一新 S0 runner 和新 `stage_S0_raw_results/<arm>/...` 目录生成，manifest 写入 `fresh_run=true`（`stage_S0_run.py:510-514`）；未读取历史 A/B 作为正式 paired gate。正式产物尚未运行，因此只能确认实现路径。 |
| 3 | B/C 除 gamma vector 外路径一致 | PASS | 四个 EMA arm 共用 `_train_epoch`、`DomainEMA`、shared Adam β1=0；arm 分支只在 `gamma_vector()` 中改变 gamma（`stage_S0_run.py:59-67,121-129,298-308,360-378`）。 |
| 4 | A/D 除 shared β1/必要分组外一致 | PASS | 三个 no-EMA arm 共用同一路径，唯一 arm 参数是 shared β1（`stage_S0_run.py:59-67,298-305,360-378`）。 |
| 5 | β1=0.9 的 split-group Adam 与 baseline 更新等价 | PASS | reviewer 在 PyTorch 2.0.0/CUDA 环境独立执行 3-step tensor test，参数逐元素相等；现有测试见 `test_stage_S0_momentum.py:42-62`。 |
| 6 | 所有 arm 使用同一 sample objective | PASS | 系数为 `p_d / mean(p)=3p_d` 且 task-major tile（`stage_S0_run.py:249-256`）；main、per-domain raw gradient 和 architecture update 均消费同一加权六-cell BCE（`329-335,352-368`）。 |
| 7 | examples/steps/batches/init 完全配对 | **FAIL** | 主训练 batch、初始化和计数进入 `PAIR_FIELDS`，但每 epoch 的 architecture-update batch 没有写入 `batch_trace.csv`、没有 hash，也没有 block audit（`stage_S0_run.py:319-339,409-414,520-524`; `stage_S0_orchestrator.py:40-43`）。architecture batch 会实际更新模型，不能只靠 step count 证明配对。另见 Critical C3。 |
| 8 | 固定 50 epochs，不被 early stop 截断 | PASS（实现路径） | `fit()` 无 early break，shadow state只限制 best 更新（`stage_S0_run.py:448-486`）；validator 检查 epoch=50（`stage_S0_orchestrator.py:131-135`）。 |
| 9 | checkpoint/control-family 选择不接触 test | PASS | selected epoch 只读 validation macro-AUC，严格 `>` 实现 first argmax 且停留在 shadow-visible window（`stage_S0_run.py:465-475`）；C/D family 只调用 `validation_score()`（`stage_S0_report.py:97-99,129-136`）。 |
| 10 | momentum buffers、Adam states、raw metrics 实际落盘 | **FAIL** | main optimizer 和 EMA state 会保存，但 architecture Adam state没有保存（`stage_S0_run.py:480-484`）；validator 不检查 `momentum_step.csv` 行数/finite/content，也不要求 formal `instrument=1`（`stage_S0_orchestrator.py:115-164`），因此空 instrumentation 文件可被正式接纳。 |
| 11 | oracle fingerprints 合法 | PASS（静态路径） | 复用 E3 严格 validator，要求 dataset/device、seed/setting/role count、formal protocol、per-domain remapped train/val/test row fingerprint 和 finite selected metrics（`stage_S0_run.py:263-273`; `stage_E3_weighting.py:641-787`）。formal run 默认 fail-closed。 |
| 12 | instrumentation OFF/ON 短程参数和指标一致 | PASS（reviewer 补充验证） | reviewer 用真实 `S0Runner`、`sample_adam`、2 epochs、tiny role sizes 分别运行 ON/OFF：initial hash、final hash、batch hash完全相等，`metrics_epoch.csv` 零差异，两个 final checkpoint SHA256 相同；这是非正式短测，不进入研究结果。现有仓库测试本身只覆盖 toy linear observer，覆盖度仍应补强。 |
| 13 | 无 Curriculum/后续 stage/新 momentum 变体 | PASS | S0 三个实现文件中没有 Curriculum、controller、stacked momentum、bias-corrected comp1 或后续-stage 分支；arm registry 对七臂闭合。 |

### 设计中其他必需输出

- `selected/fixed/common` 原始 pair rows 已实现（`stage_S0_report.py:163-193`）。
- seed 是主重复单位、rotation 在 seed 内平均，当前 selected-view AUC 聚合实现正确（`stage_S0_report.py:232-249`）。
- role B-vs-A selected gains 已实现（`stage_S0_report.py:194-204`）。
- **未完整实现** task-level paired gain、physical-domain paired gain、历史 A/B 对 fresh A/B 的描述性复现审计。原始 cell rows虽可人工重算，但设计要求的正式汇总证据没有生成。

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

- `w_d=3p_d`、task-major索引、六个 cell mean BCE 的实现正确。
- comp1 使用每域加权 shared gradient，EMA 为 `m_d←γ_dm_d+(1-γ_d)g_d`，最终 shared gradient 为三域 buffer 之和；uniform arms 使用完全相同路径。
- gamma 数值正确：正式份额下 B 为 `[0.9, 0.977, 0.99]`，C1 为 `0.9431818`，C2 为 `0.980453`。
- D arms 不经过外层 EMA，普通 Adam bias correction保持启用；shared/private参数组 β1 隔离正确。
- **Blocking:** `DomainEMA.update()` 把 shared gradient 的 `None` 静默替换为零（`stage_S0_run.py:175-178`）。本实验的 shared partition 应参与每域 loss；若未来参数断图，此处会把严重实现错误变成成功运行。应对 expected-shared `None` 直接失败，而不是零填充。

### Gradient flow & Loss participation

- B/C 的 raw shared gradient来自同一个 forward graph；随后完整 objective backward 为 private参数提供梯度，再仅替换 shared `.grad`，计算图参与关系正确（`stage_S0_run.py:360-378`）。
- architecture optimizer 每 epoch 使用 sample objective，且与 main optimizer 参数分离（`stage_S0_run.py:319-339`）。
- **Blocking:** 训练 loss、cell terms、raw/full gradients、EMA buffers、Adam moments、模型参数和 eval输出均无系统性 `isfinite` 断言。若早期有过 finite best、后续发生 NaN，`fit()` 仍可能保存早期 checkpoint并写 `status=OK`（`stage_S0_run.py:448-486,510-540`）。

### Controlled variables & Baseline preservation

- S0实现局部新增，没有修改 core MDMTRec/Curriculum 路径。工作区无可用 Git repository metadata，故 reviewer 无法用版本差异独立证明所有历史文件从未改变；只能确认 S0代码通过只读 import复用 legacy模块。
- main initialization hash、split fingerprints、selected training rows、main batch hash、examples/steps及 optimizer/architecture step counts进入 block pairing audit。
- **Blocking:** architecture-update batch未记录/配对；它每 epoch真实更新 architecture参数，因此属于训练轨迹的一部分。
- **Blocking:** formal CLI 暴露并接受 `role_sizes/batch_size/lr/darts_lr/weight_decay/expert_num/shadow_patience/instrument`，而 `validate_run()` 只锁了 epoch、beta和objective字符串（`stage_S0_orchestrator.py:218-240,115-164`）。例如 `--instrument 0` 或非 40960 batch 可产生 validator 认可的 `OK` run。默认值正确不等于协议被锁定。

### Fail-fast enforcement

当前**不满足** fail-fast：

1. 无训练/梯度/状态/指标 nonfinite gate；report 的 `_mean()` 反而静默删除 NaN/Inf（`stage_S0_report.py:72-83`）。
2. shared `None` gradient静默零填充（`stage_S0_run.py:175-178`）。
3. orchestration 中某 task 失败后只打印并继续后续任务（`stage_S0_orchestrator.py:277-295`），不符合“异常立即停止”。
4. `validate_run()` 不验证 formal critical config、metric有限性、每 epoch六个唯一 cell、raw momentum行数/内容、batch trace行数、checkpoint state内容。
5. report 可在非有限 cell存在时用剩余 cell计算 macro；`_descriptive()` 同样静默缩小 n（`stage_S0_report.py:220-229`）。由于 validator没有 finite gate，最终 `complete=true` 仍有可能出现。

## 4. Over-Engineering & Code Bloat Audit

未发现明显 enterprise 式框架、factory/registry膨胀或与 Stage 0 无关的抽象。新增模块职责清晰，复用 E1/E3 的 fingerprint/oracle helper是合理的局部复用。

轻微问题：`ARM_SPECS` 在 runner 与 orchestrator 各维护一份（`stage_S0_run.py:59-67`; `stage_S0_orchestrator.py:24-32`），存在实验臂定义漂移风险。无需引入复杂 registry；只需一个小型、单一来源常量模块或启动时交叉校验即可。

## 5. Bugs / Deviations Found

### CRITICAL（均为 blocking）

**C1 — 非有限值和断图可静默通过。**

- 证据：`stage_S0_run.py:175-178,341-415,448-486`；`stage_S0_report.py:72-83,220-229`。
- 影响：训练可能已经失效但 manifest/report仍为成功；汇总可能改变样本数并给出看似正常的 macro。
- 最小修复：在每个 step 对 loss/cell terms、expected gradients、EMA/aggregate grads、optimizer states、step后参数做 finite断言；eval六 cell全部 finite；shared `None` gradient直接报错；report遇到缺失/nonfinite直接失败，禁止过滤。

**C2 — formal protocol 未锁定，instrumentation证据可缺失仍被接纳。**

- 证据：`stage_S0_orchestrator.py:115-164,218-240`。validator不核对 batch=40960、role sizes、两种 lr、weight decay、expert number、patience、instrument=1等；`momentum_step.csv`只检查文件存在。
- 影响：非设计实验可完整执行并进入 `VALID/complete`，直接破坏 controlled variables。
- 最小修复：定义唯一 `FORMAL_PROTOCOL`，launcher启动前和 `validate_run()`产物后都逐字段严格比较；formal必须 `instrument=1`、`require_oracles=1`；验证 momentum rows=`3×optimizer_steps`、batch rows=`optimizer_steps`（加入 architecture rows后按明确 schema检查）、所有要求字段 finite/identity完整。

**C3 — architecture batch不在配对证据链。**

- 证据：architecture batch在 `stage_S0_run.py:319-339` 被消费并更新参数，但 batch trace只在 main loop `409-414` 记录；`PAIR_FIELDS` 没有 architecture batch hash（`stage_S0_orchestrator.py:40-43`）。
- 影响：七臂即使 architecture数据顺序不同，block audit仍可通过，无法确认严格配对。
- 最小修复：记录每次 architecture batch fingerprint、样本数和顺序 hash；manifest写入 architecture batch count/hash/examples；加入 `PAIR_FIELDS` 和行数核验。

**C4 — 汇总缺失/非有限数据会静默缩约，且缺少预注册诊断。**

- 证据：`stage_S0_report.py:72-83,129-136,220-249`；没有 task/physical-domain paired summary 或 historical-vs-fresh A/B表。
- 影响：C_val/D_val和macro可能基于不完整 cell；主分析无法按预注册输出直接核对单 task/physical domain与历史复现解释。
- 最小修复：严格要求每 block每 arm每 epoch六个唯一 finite cell；只对通过 `audit_block` 的 block汇总；增加 task、physical-domain、role、selected/final/common以及历史-vs-fresh A/B的描述性表，不读取 test做任何选择。

### MAJOR（blocking）

**M1 — resume/failure逻辑不安全。**

- `successful_keys` 可使当前 artifact缺失/损坏的任务永久跳过（`stage_S0_orchestrator.py:247-257`）。
- FAILED任务在未指定 rerun 时从 pending移除，并被 queue manifest 的 `completed_count=84-len(pending)` 误算为完成（`258-264`）。
- 子任务失败后不立即停止，仍继续运行矩阵（`277-295`）。
- 最小修复：是否完成只能由当前 artifact `validate_run()`决定，master CSV不能替代；FAILED/invalid必须明确使 preflight失败或在显式 rerun后归档；第一项失败立即终止并保留append-only attempt row。

**M2 — 原始优化器状态不完整。**

- 只保存 main optimizer state，architecture optimizer state没有落盘（`stage_S0_run.py:480-484`）。
- 最小修复：保存 architecture optimizer state（若存在），manifest列出并由 validator要求；至少核对 shared/private param-group beta和step state与manifest一致。

**M3 — 自动化测试不足以覆盖正式验收。**

- 当前 pytest：`5 passed, 1 skipped`；skip来自默认 Python 无 torch。
- reviewer发现 `C:\Users\86199\.conda\envs\EDDAenv\python.exe` 有 PyTorch 2.0.0/CUDA，但该环境无 pytest；已手动验证核心tensor测试和真实 sample-arm ON/OFF等价。
- 缺少：formal config拒绝、nonfinite失败、architecture batch mismatch、resume stale-master/stop-on-first-failure、checkpoint shadow窗口、oracle mismatch、完整 raw schema、真实 EMA-arm instrumentation等价测试。

### MINOR（non-blocking）

**N1 — arm registry重复。** 见 over-engineering audit；建议用最小方式消除漂移。

**N2 — `log_path` 未使用。** `stage_S0_run.py:568` 创建变量但实际 stdout重定向由 orchestrator完成；不影响实验逻辑。

## 6. Corrections Made

- 未修改研究算法或 runner/report实现；blocking问题数量与范围适合返回 coder做一次集中、最小修复，保留第 2 次也是最后一次 reviewer审核。
- 新增本审核报告 `experiments/reports/stage_S0_review.md`。
- reviewer执行了以下非正式检查：
  - base Python：`5 passed, 1 skipped`；
  - PyTorch 2.0.0/CUDA：DomainEMA数值与 split-Adam等价手工测试通过；
  - 真实 `S0Runner` tiny sample arm 2-epoch instrumentation ON/OFF：最终参数 hash、checkpoint SHA256、batch hash和全部 metrics一致；
  - syntax compile与 84-task dry-run通过。

## 7. Remaining Risks

1. 尚未在正式 150000/34500/3000、batch 40960、50 epochs配置上执行任何训练；这符合本次 REVIEW权限边界。
2. 尚未对真实 EMA arm做端到端短程运行；静态数学路径和手工 DomainEMA测试通过，但修复后应补一项真实 EMA tiny test。
3. oracle validator本身严格，但本次未实例化全部 12 个 `(rotation,seed)` 的 formal oracle coverage；RUN前应做只读 preflight并要求 72/72 cell coverage（12 blocks×6 cells）。
4. 无 Git metadata，无法审计历史文件修改范围；在当前工作区只能依据S0文件边界与代码引用检查。
5. 正式 launcher默认 `sys.executable` 当前无 torch；runner必须显式使用已验证的 EDDAenv Python，或在启动前做依赖/CUDA preflight。不能让 84 个任务逐个失败。

## 8. Whether Experiments Are Safe to Run

**否。当前不安全，禁止启动正式 84-run。**

只有在 C1-C4、M1-M3 的最小修复完成，并在第 2 次（最后一次）review中确认以下条件后才可进入 luna_runner：formal protocol严格锁定、所有finite/fail-fast gate有效、architecture batch完整配对、resume首错即停、raw optimizer/instrumentation证据完整、报告不静默过滤且预注册诊断齐全。

---

# REVIEW REPORT — Review 2/2 (Final)

## 1. Verdict: PASS WITH FIXES

这是第 2/2 次且最后一次审核。

第 1 次审核的全部 blocking 项已经在实际代码中解决。最终审核又发现 C1 的两个残余边角：A/D instrumentation-only raw shared gradients 与 private gradients未逐张量即时检查，以及 optimizer audit未要求 state覆盖全部参数。Reviewer 已作最小局部修复并重新验证。**当前没有剩余 blocking issue，允许进入 RUN。**

本次仍未运行正式 84-run，也未分析研究结论。

## 2. Requirement-by-Requirement Verification

| # | 设计要求 | 最终结论 | 证据 |
|---|---|---|---|
| 1 | 84-run矩阵完整、唯一、每臂12个 | PASS | 单一 arm registry 位于 `stage_S0_protocol.py:9-17`；`build_tasks()` 在 `stage_S0_orchestrator.py:101-116` 强制 84 个唯一键；dry-run再次打印 `7×3×4=84`。 |
| 2 | A/B fresh | PASS | A/B都由同一 S0 runner创建新 run目录，manifest写 `fresh_run=true`；旧结果只由 report 的 descriptive audit读取，永不进入 formal gate。 |
| 3 | B/C 仅 gamma vector 不同 | PASS | 四个 EMA臂共享同一 forward、per-domain raw gradient、`DomainEMA`、full-objective private backward及 β1=0 shared Adam；arm差异仅来自 protocol registry 中的 gamma配置和 `gamma_vector()`。 |
| 4 | A/D 仅 shared β1/必要param grouping不同 | PASS | 三个 no-EMA臂共享路径，shared β1分别为0.9/0.9431818/0.980453；private β1固定0.9。validator核对实际 optimizer group audit（`stage_S0_orchestrator.py:254-272`）。 |
| 5 | split-group Adam默认β1更新等价 | PASS | PyTorch direct 3-step等价测试通过；测试实现位于 `test_stage_S0_momentum.py`。 |
| 6 | 所有参数路径使用同一 sample objective | PASS | `w_d=3p_d` task-major系数同时用于main、per-domain raw shared gradient和architecture update；真实 patched EMA tiny run通过全部shared/private/architecture gradient gate。 |
| 7 | init/data/main batch/architecture batch/examples/steps严格配对 | PASS | `PAIR_FIELDS` 包含init、split、main batch hash、architecture batch hash/count/examples和所有step预算（`stage_S0_orchestrator.py:34-38`）；block audit逐字段比较七臂（`285-315`）。 |
| 8 | fixed 50 epochs，shadow early-stop不截断训练 | PASS | formal protocol锁定50；runner无early break，shadow只冻结selection window；validator核对epoch、step和example exact budget。 |
| 9 | validation-only checkpoint和C/D family selection | PASS | checkpoint严格使用validation macro-AUC与first-argmax；C_val/D_val selector只读取validation字段；对抗性test-ranking单测通过。 |
| 10 | momentum/Adam/raw metrics真实落盘 | PASS | main Adam、architecture Adam、EMA state、300 cell metrics、`3×optimizer_steps` momentum rows、main+architecture batch trace均保存并验证非空/cardinality/finite。Reviewer追加state coverage校验，要求Adam state parameter count等于全部group parameter count。 |
| 11 | oracle fingerprint合法 | PASS | runner使用E3 formal oracle validator并要求seed/rotation/role/dataset/device/remapped train-val-test row fingerprints及六cell覆盖；`require_oracles=1`被formal protocol锁定，缺失即在训练前失败。 |
| 12 | instrumentation ON/OFF等价 | PASS | Review 1的真实 sample-arm 2-epoch ON/OFF test中final hash、checkpoint SHA256、batch hash和metrics完全相同；新增finite读取不改变tensor值。Review 2真实patched EMA tiny run也完整通过。 |
| 13 | 禁止Curriculum/后续stage/新momentum | PASS | S0 registry仅含七个预注册arms；源码扫描无Curriculum/controller/stacked/bias-corrected comp1/gamma schedule分支。 |

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

- sample objective实现为 `D p_d` 乘每个 domain-task mean BCE，task-major索引正确。
- B的正式 gamma为约 `[0.9,0.977,0.99]`；C0/C1/C2使用统一 `[0.9,0.9431818,0.980453]`；D1/D2只改变普通Adam shared β1。
- EMA公式、domain buffer更新频率、buffer求和与 β1=0 Adam位置均与设计一致。
- common epochs、fixed final和shadow-visible selected checkpoint均被保存。

### Gradient flow & Loss participation

- B/C的domain raw gradients实际参与EMA，EMA aggregate实际写回shared `.grad` 后进入optimizer step。
- full weighted objective同时为private参数反传；architecture参数由同objective的独立optimizer更新。
- runner现在对每个cell loss、combined loss、per-domain raw shared gradient、full shared/private gradient、architecture gradient、EMA buffer、Adam moments、post-step参数和eval输出执行立即finite/connected检查。
- reviewer修正后的关键行：`stage_S0_run.py:390,448-460`。真实 tiny comp1 run显示 main Adam state覆盖107/107参数、architecture Adam覆盖4/4参数。

### Controlled variables & Baseline preservation

- `stage_S0_protocol.py:19-31` 是formal critical variables唯一来源；launcher在启动任何child前拒绝偏差，产物validator再次逐字段核对。
- main与architecture batch均有ordered fingerprint；初始化、split、code、examples和steps进入七臂block pairing。
- 新实现局部限于S0文件及tests/reports；未修改core MDMTRec、Curriculum、历史raw results或E1-E7路径。工作区仍无Git metadata，因此无法通过commit diff证明更早历史状态，但当前S0作用边界清晰。
- historical A/B audit固定 `included_in_formal_gate=0`。仓库中没有可验证的旧per-run A/B artifacts时会明确输出MISSING，不会从叙述性markdown或不匹配结果中伪造/替代。

### Fail-fast enforcement

确认满足：

1. runner遇到NaN/Inf、expected gradient断连、空domain cell、oracle不完整或CUDA不可用立即抛错；顶层仅写FAILED manifest后重新抛出。
2. orchestrator严格锁formal protocol，并在首个child/validation失败后立即保存attempt row、写fail-fast queue manifest并返回1。
3. resume只相信当前artifact的完整validator结果；stale master `OK`不能跳过任务。已有invalid/failed目录若无显式 `--rerun_failed` 会阻断；显式重跑前旧目录被时间戳归档。
4. report只汇总通过完整七臂block audit的block；numeric helper遇到missing/NaN/Inf直接失败，不再缩小n。

## 4. Over-Engineering & Code Bloat Audit

PASS。第一次指出的重复arm registry已由小型 `stage_S0_protocol.py` 消除。没有引入factory、schema framework、复杂兼容层或与Stage 0无关的防御逻辑。新增检查均直接防止错误实验被当作成功证据，符合研究代码的最小fail-fast原则。

## 5. Bugs / Deviations Found

### CRITICAL

- **已修复 C1:** nonfinite/断图静默通过。最终审核补齐A/D raw shared与所有private gradient检查，并要求optimizer state覆盖全部参数。
- **已修复 C2:** formal protocol未锁定、instrumentation可关闭。launcher与artifact validator双重拒绝偏差。
- **已修复 C3:** architecture batch未配对。现已记录batch rows/hash/count/examples并加入`PAIR_FIELDS`。
- **已修复 C4:** report静默过滤或缺少预注册诊断。现已strict numeric、block-gated，并输出role/task/physical-domain subgroup及24-row historical audit。

### MAJOR

- **已修复 M1:** stale master、FAILED误计完成、失败后继续。当前artifact为唯一真值且首失败停止。
- **已修复 M2:** architecture optimizer state缺失。现已保存、列入required artifacts并核对step/state coverage。
- **已修复 M3:** 回归覆盖不足。base suite现为11 passed/1 torch-file skipped；PyTorch/CUDA direct tests与两轮真实tiny runner检查补充覆盖。

### MINOR

- `stage_S0_run.py` 顶层保留广义 `except Exception`，但它只负责写FAILED manifest随后原样re-raise，不抑制异常，因此不构成silent fallback。
- base Python的pytest cache因目录权限产生warning；不影响测试结果。torch tests在EDDAenv以direct checks执行，因为该环境未安装pytest。

## 6. Corrections Made

Reviewer在最终审核中仅作以下最小修复：

1. `stage_S0_run.py`
   - architecture batch必须覆盖六个非空cell；
   - 所有arm的per-domain raw shared gradients立即检查connected/finite；
   - 所有private gradients在optimizer step前立即检查connected/finite。
2. `stage_S0_orchestrator.py`
   - main与architecture optimizer audit都要求state parameter count精确覆盖其全部param groups。

未修改算法、arm、gamma、β1、预算、checkpoint规则或研究假设。

验证结果：

- `py_compile`：PASS。
- base pytest：`11 passed, 1 skipped`；skip仅因base Python无torch。
- manual PyTorch 2.0.0/CUDA fail-fast测试：PASS。
- patched真实 `comp1_domain` tiny run：PASS；1 main step、1 architecture step、EMA counts `[1,1,1]`、3 momentum rows、2 batch rows、main Adam 107/107 state params、architecture Adam 4/4 state params；正确标记为nonformal。
- formal dry-run：84 tasks；非40960 batch在child启动前被拒绝。
- 未运行正式84-run。

## 7. Remaining Risks

1. 尚未执行formal-scale训练；运行时资源、墙钟时间与完整84-run结果仍未知，这属于RUN阶段而非实现blocking。
2. 历史per-run A/B artifacts当前不可验证，historical audit将明确标记MISSING；这不影响fresh paired gate，也不得用markdown中的聚合数字替代。
3. 正式运行必须显式使用已验证的 `C:\Users\86199\.conda\envs\EDDAenv\python.exe`。默认base Python无torch，preflight会在任何child启动前阻断。
4. RUN完成后仍必须要求queue manifest `valid_run_count=84`、12个block全部`VALID`、report manifest `complete=true`，否则不得进入main analysis。

## 8. Whether Experiments Are Safe to Run

**是。当前实现允许进入 luna_runner 的正式 RUN 阶段。**

PASS WITH FIXES 中的fixes已由reviewer完成并验证；**没有必须在RUN前继续修复的blocking issue**。Runner应使用EDDAenv Python、保持formal protocol默认值，不得使用`--max_runs`缩减正式矩阵，并在首个异常时保留证据后停止。
