# REVIEW REPORT

## 1. Verdict: PASS

这是 Stage E4 第 3 次、也是最终独立复审。前两轮全部blocking已修复，未发现新的blocking issue。Stage E4可进入RUN阶段。

本轮严格只复核`selected_support` scope修复、真实非空summary的`PENDING → VALID`回填，以及既有算法gate是否无回归；没有读取任何prompt/group3，没有修改实现，没有启动正式12+12矩阵，也没有分析科研结论。

## 2. Requirement-by-Requirement Verification

| Requirement | Final verification | Status |
|---|---|---|
| Equal/Sqrt/Sample使用train shares，domain系数和3、六cell和6、task-major复制 | 公式与原focused tests未变，acceptance coefficients与gradient test继续通过 | PASS |
| 六个masked-mean BCE显式组成唯一受操纵main loss并进入真实backward | joint训练路径未回归；actual coefficients仍直接参与`loss.backward()`与main Adam step | PASS |
| chronological split-first/filter-second oracle | loader/oracle代码未回归；synthetic ordered split test继续通过 | PASS |
| shared-only observer不改变训练路径 | 同一forward graph上`autograd.grad(..., retain_graph=True)`后执行一次真实weighted backward/step，未出现额外step | PASS |
| exact 12 joint + 12 oracle matrix | builders仍精确枚举3×4 joint与3×4 oracle | PASS |
| 首/第二weighting arm保持PENDING并继续queue | `_finalize_joint`只对完整block返回的`INVALID`停止；真实非空summary fixture证明前两臂成功回填PENDING且queue继续 | PASS |
| 三臂齐备后判VALID/INVALID | comparator仅在三臂required artifacts齐备后检查formal identity与common-support trace；第三臂fixture将同seed三臂统一更新为VALID | PASS |
| report/summary严格传播block validity | `VALID`才可成为`FORMAL_VALID`；`PENDING→BLOCK_PENDING`，`INVALID→INVALID_BLOCK`；report focused tests均通过 | PASS |
| selected checkpoint/common epoch/common trace真实可分析 | selected epoch来自manifest best epoch；common epochs/traces来自三臂实际trace交集；summary保存selected flag、selected-in-common-support、epoch count与trace count | PASS |
| oracle coverage与transfer方向 | joint只接受formal matched oracle；transfer仍为joint-oracle，缺coverage不伪造 | PASS |
| checkpoint/test/censoring | 六joint cells或两oracle tasks必须finite；test eval保存/恢复CPU与全部CUDA RNG；early-stop尾部显式censored | PASS |
| summary支持AUC/loss/domain/task/seed/size/difficulty/transfer/shift分析 | per-cell selected metrics、macro/pooled字段、split counts/rates、oracle/transfer、shared metrics与common-support字段齐全 | PASS |

## 3. Scientific & Algorithmic Integrity Audit

### Mathematical & Tensor correctness

- Index保持`idx=t*D+d`，与MDMTRec cell输出顺序一致。
- Equal/Sqrt/Sample系数仍从审计后的train proportions动态计算，同一domain在两个tasks复制相同系数，无label-prior二次加权。
- Joint loss是六个domain/task masked sample-mean BCE与系数的显式点积。
- Oracle仍在全量编码与global chronological 80/10/10 split之后逐split过滤，并remap local domain id为0。

### Gradient flow & Loss participation

- Weighting coefficient实际参与优化图，不是仅记录到日志。
- Architecture parameters与main parameters互斥；weighting不改变architecture update、sampling或optimizer config。
- Shared observer只读取raw shared gradients，随后真实weighted loss执行单次backward/optimizer step；reconstruction与weighted-norm测试无回归。

### Controlled variables & Baseline preservation

- 同seed三臂比较检查dataset/split、initial model hash、partition/model/optimizer identity，以及common-support上的train/architecture batch fingerprint、size与domain/task counts。
- 合法early-stop尾长允许不同，但共同支持不填充censored epochs。
- Matched oracle保留original physical domain、global/filtered split fingerprints与相同seed/protocol。

### Fail-fast enforcement

- incomplete block保持显式`PENDING`，不会被误当成可分析结果，也不会阻止后续arms生成。
- 完整block的identity或batch mismatch返回`INVALID`并阻断；report不会把PENDING/INVALID标为FORMAL_VALID。
- 数据审计、系数、checkpoint和oracle coverage异常均保持显式失败或invalid状态；未发现silent fallback。

## 4. Over-Engineering & Code Bloat Audit

最终修复保持局部：仅消除了nested scope变量遮蔽，并让既有queue lifecycle test经过真实非空summary回填。没有增加新算法、通用框架或与实验无关的防御逻辑。

现有artifact checks主要用于防止split/oracle/三臂身份错误后实验仍成功结束，符合科研fail-fast需求，不要求进一步重构。

## 5. Bugs / Deviations Found

### CRITICAL

无。

### MAJOR

无。

### MINOR

无新增问题。

前两轮问题的最终状态：

1. `PENDING`被错误转为`INVALID_BLOCK`：已修复。
2. Report未传播block validity：已修复。
3. `common_support`恒定为0：已修复，未知时为空，三臂齐备后由真实block comparison回填。
4. `selected_support` nested scope导致`UnboundLocalError`：已修复为只读外层mapping的局部`support_map`。
5. Queue test空summary漏测真实回填：已改为写入非空summary row，并断言PENDING与VALID两个阶段。

## 6. Corrections Made

Reviewer未修改实现代码。本轮只更新最终review report。

## 7. Remaining Risks

- n=4 seeds的统计能力、KuaiRand temporal/domain-mixture shift及自然domain/task heterogeneity属于预注册科研解释风险，不是实现blocking；应由main analysis按设计独立评估。
- Reviewer没有运行正式矩阵，因此formal artifacts的实际coverage、运行时数值与GPU稳定性仍须由runner记录并验证。
- 正式RUN若出现任何artifact invalid、oracle coverage不足、non-finite或block mismatch，应按现有fail-fast状态停止，不得进入main analysis。

## 8. Whether Experiments Are Safe to Run

**YES。Stage E4实现已通过最终review，可以进入RUN。**

验证证据：

- Focused suite：`8 passed in 1.77s`。
- Syntax：`stage_E4_weighting.py`、`stage_E4_run.py`、`stage_E4_oracle_run.py`、`stage_E4_orchestrator.py`、`stage_E4_report.py`全部通过`py_compile`。
- 非空summary lifecycle fixture确认：首/第二arm为PENDING且common-support为空；第三臂完成后同seed三臂全部为VALID，common-support epoch/trace与selected flag正确回填。

批准范围仅为设计规定的12 matched oracle + 12 joint正式矩阵。Runner只记录raw results，不得越权分析科研结论。
