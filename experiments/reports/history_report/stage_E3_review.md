# REVIEW REPORT

## 1. Verdict: PASS

这是 Stage E3 第 3 次且最终独立复审。第 2 次复审遗留的唯一 blocking——oracle reuse 未绑定本次 external device 与当前 joint 实际 per-domain rows——已被正确、完整地定点修复。复核未发现算法 gate 回归或剩余 blocking；Stage E3 可以进入 RUN。

本次没有读取任何 `promt` 文件或 `group3.md`，没有修改实现代码，没有运行正式 36-oracle/60-joint 矩阵，没有执行 GPU smoke，也没有分析科研结果。

## 2. Requirement-by-Requirement Verification

| Requirement | Status | Evidence |
|---|---|---|
| `c_d(alpha)=D*p_d^alpha/sum(p^alpha)` | PASS | 五点 alpha 使用唯一公式实现并从实际 domain share 计算。 |
| 六-cell 总系数 6、alpha 端点、task 等权 | PASS | alpha=0 为六个 1；alpha=1 为 `[2.4,0.552,0.048]` task-major 重复；无 task 二次 normalization。 |
| Joint loss/optimization participation | PASS | 六个 masked-mean cell BCE 的显式 coefficient 点积进入一次真实 backward 和一次 main optimizer step。 |
| 唯一训练自变量 | PASS | 五臂只改变 alpha/coefficient/output identity；sampling、model、optimizer、architecture cadence、checkpoint rule 固定，无禁用机制。 |
| 60 joint + 必要时 36 oracle | PASS | 固定 queue 为 5×3×4=60；当前 artifact audit 缺 oracle 时为 3×3×4=36，oracle 完成后才允许 joint launch。 |
| Oracle 两 task objective | PASS | `oracle_main_loss` 为两个 per-task BCE means 之和，与冻结 single-domain/P0-4 语义一致。 |
| Oracle formal/reference gate | PASS | 要求 formal 50-epoch protocol、合法 early stop、完整 finite checkpoint、模型文件、dataset/device/split identity；1-epoch smoke 不可复用。 |
| External expected device | PASS | `args.device` 从 `audit_oracles`、`_finalize_oracle`、joint resolver 传至 `validate_oracle_artifact`；artifact device 必须等于本次 requested device。 |
| Current joint row identity | PASS | 当前 MovieLens arrays 按 setting/seed 做相同 role sampling，逐 physical domain 筛选并把 domain column remap 为 0，再对 train/validation/test 的 ordered feature+label rows 计算 fingerprints。该 external expected map 贯穿 audit/resolve/validator。 |
| Wrong-device / wrong-row rejection | PASS | Focused fixture 对内部自洽但 external device 不同或 validation fingerprint 不同的 formal oracle 均返回 invalid；resolver coverage 保持 0。 |
| Joint resume/batch pairing | PASS | dataset/device、split row fingerprints、initial hash 与每个共同 epoch 的完整 train/architecture support 均跨五臂核对。 |
| Shared-only gradient diagnostics | PASS | 复用 E1 strict shared partition；observer 使用 `autograd.grad`，不写 `.grad` 或额外 step；重建、raw/weighted norm、signed contribution、mismatch、alignment 均记录。 |
| Checkpoint/test semantics | PASS | Joint 严格使用完整六-cell validation macro AUC；oracle 使用完整两-task macro；non-finite/missing cell 立即失败；test 不参与选择且 RNG 恢复。 |
| Summary/logging | PASS | Seed/cell/domain/role/task、AUC/LogLoss、domain size/share、oracle/transfer、shared metrics、正确 common-support/censored 字段齐全，支持设计规定的后续分析。 |

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

- 权重公式、正归一化、task-major index 与端点数值符合设计。
- 每 task 使用同一个 domain coefficient，六项总 coefficient scale 固定为 6。
- Raw shared cell gradients 经 coefficient 加权后的和与真实 backward shared gradient 使用相同 computational graph；实现未出现广播、自动 reshape 或错位索引。
- Oracle 对 train/validation/test 的 external fingerprints 精确复刻 oracle loader 的 physical-domain selection 与 local-domain-0 view，因此比较的是实际相同 rows，而非仅比较路径或 counts。

### Gradient flow & Loss participation

- 新 weighting component 实际参与 forward 后的 loss、backward 与 optimizer update，不是断开的诊断分支。
- Shared observer 只读同一图；`.detach()` 仅用于记录/统计，不切断真实 loss path。
- `weighted_gradient_norm = coefficient × raw_gradient_norm`、signed contribution sum 与 reconstruction 均有 focused/smoke 证据覆盖。

### Controlled variables & Baseline preservation

- 同一 setting/seed 的五 alpha 初始参数、数据 rows、完整共同 batch prefix、split/schema 与运行 device 被锁定。
- 只有显式 early-stop/censored suffix 可以造成尾长差异；共同 epochs 内 support shape 和 fingerprints 必须一致。
- Frozen oracle 与当前 requested dataset、device、setting、seed、physical domain 及 remapped rows 绑定，不会复用旧 device 或 stale split reference。
- E1/core model/data/trainer 未被修改，变更局限于 E3 模块与 focused tests。

### Fail-fast enforcement

- 非注册 alpha、错误 role counts/coefficient、missing formal artifacts、short oracle、错误超参/device/rows、非连续 support、缺失或 non-finite checkpoint cells 都会显式失败。
- 没有 equal endpoint、Markdown aggregate、旧 oracle 或 finite-cell subset 的 silent fallback。
- 正式 orchestrator 在 oracle audit 不完整时不会启动 joint matrix。

## 4. Over-Engineering & Code Bloat Audit

PASS。最终修复复用了已有 fingerprint/validator 路径，只增加当前 dataset 的 deterministic external expected map 和参数传递，没有引入 factory、registry、controller、production monitoring 或范围外实验机制。相关检查直接保护 transfer baseline 与 controlled variables，符合科研代码的最小必要 fail-fast 原则。

## 5. Bugs / Deviations Found

### CRITICAL

无。

### MAJOR

无。前两轮报告的 blocking 均已修复。

### MINOR

- Pytest 仍报告工作区 `.pytest_cache` permission warning；不影响测试收集、算法执行或实验产物。

## 6. Corrections Made

Reviewer 未修改实现代码；仅更新本最终 review 报告。

独立验证：

- Focused tests：`11 passed`。
- Dry-run 使用当前 MovieLens 路径实际加载数据并生成 external oracle row identities；输出为 `36 expected, missing/invalid: 36` 与固定 `60` joint、`60` pending，未启动任何子进程。
- Dry-run 临时目录已从已验证的精确路径 `D:\CODE\phase0-3\experiments\review_e3_final_tmp` 删除，不影响正式 raw-results 路径。
- 未重复完整 GPU smoke；最终轮任务仅要求 external identity 与算法 gate 回归，现有 focused fixtures 已直接覆盖该修复。

## 7. Remaining Risks

- 当前没有实现层面的 blocking。正式运行仍可能受硬件中断、运行时长或四 seed 统计不确定性影响，这些应由 runner manifest/status 与 main analysis 的 seed-cluster protocol处理。
- Oracle、joint response curve 或 adaptive/static 假设是否成立完全未知；PASS 只说明实验按设计执行，不预设科研结论。
- Runner 应保存全部原始 artifact，并在任何 child failure、oracle coverage 不完整或 block invalid 时停止/标记失败，不得分析部分矩阵为完整证据。

## 8. Whether Experiments Are Safe to Run

**YES。** Stage E3 实现已通过第 3 次最终 REVIEW，可以按 AGENTS.md 进入 `luna-runner` 阶段。Runner 可执行必要的 36 个 frozen oracle references 和固定 60 个 joint runs，只记录原始结果与完整性状态；不得在 RUN 阶段解释科研结论。
