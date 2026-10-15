# Stage E7 DESIGN GATE AUDIT — STOP

## 0. Scope and prompt-isolation declaration

本次 gate audit 严格只读取了提示词文件 `promt/验证性补全和实验方向寻找/group3.md`。没有读取、搜索、打开或委托其他代理读取其他 prompt 文件。

本文件审计 group3 内部下一顺序 E7 是否具有合法 DESIGN 入口。它不是 E7 experiment specification，不授权 IMPLEMENT、REVIEW 或 RUN，也不创建 generalization matrix。

## 1. Research question

当前需要回答的不是新的模型研究问题，而是阶段前置条件：

> E6 是否通过了 Prompt 7 明确要求的 GO gate，从而允许冻结一个成功 adaptive method 并进入 E7 generalization？

Prompt 7 的必要条件是：`只有 E6 通过 GO gate 才执行本阶段。`

## 2. Hypothesis being tested

### Gate hypothesis G-E7

仅当 E6 正式 verdict 为 `GO`，且存在已冻结、相对 sample weighting 通过预注册 gate 的 adaptive contribution calibration 方法时，E7 才可设计和执行。

### Current evidence

E6 authoritative reports 给出：

- `STOP — Method too complex for gain`；
- Adaptive-NoEMA 与 Adaptive-EMA 均未通过 GO；
- 两者均未通过 CONDITIONAL；
- Adaptive-NoEMA 相对 Sample overall AUC mean gain 约 `+0.000795`，低于 `0.001` margin；
- Adaptive-EMA 相对 Sample overall AUC mean gain约 `+0.001584`、3/4 seeds为正，但 Medium transfer 约 `-0.001363`，违反预注册 remaining-domain safety gate；
- exact NoEMA calibration 将 mismatch 降至约0，却未产生合格的 performance gain；
- `EMA not justified because the adaptive method failed`；
- `Curriculum unnecessary / dropped by E5`。

因此 G-E7 为 **FALSE**。当前没有可冻结并推广的 adaptive 主方法。

## 3. Variables changed

无。E7 experiment 不获授权：

- 不改变 imbalance ratio；
- 不运行 role reversal；
- 不增加 backbone；
- 不增加 domain count；
- 不增加 dataset；
- 不选择或修改 adaptive method；
- 不恢复 Curriculum 或 EMA。

正式 E7 新运行矩阵规模：**0 runs**。

## 4. Variables held constant

本审计保持以下 authoritative state 不变：

- E6 的预注册 GO/CONDITIONAL/STOP 判据；
- E6 24/24 formal runs 与 seed 2022、2023、2024、2025 的正式分析；
- E6 STOP verdict；
- EMA not justified；
- E5 Curriculum DROP；
- Sample weighting 是最关键、仍未被无 trade-off 稳定超越的 baseline；
- 现有 E1–E6 原始结果、报告、代码与模型均不修改。

不得通过更换 primary metric、放宽 safety margin、忽略 Medium-domain decline、把 last-common gain 替代 preregistered selected-checkpoint gate，或将 CONDITIONAL 误写为 GO 来改变结论。

## 5. Mathematical / algorithmic gate definition

定义：

\[
\operatorname{E7Allowed}=
\mathbb{1}[\operatorname{E6Verdict}=\mathrm{GO}].
\]

当前：

\[
\operatorname{E6Verdict}=\mathrm{STOP}
\quad\Longrightarrow\quad
\operatorname{E7Allowed}=0.
\]

即使 E6 某个 secondary metric 为正，也不能替代 GO。Prompt 7 没有提供“独立 generalization purpose”例外，也没有允许以 E6 CONDITIONAL 进入；当前实际结果甚至未通过 CONDITIONAL。

把 `sample_static` 重新命名为“最终 adaptive method”不满足条件，因为它没有 adaptive controller。把失败的 Adaptive-EMA 保留为主方法也不满足条件，因为它违反 E6 safety gate。两种做法都会改变 E7 所定义的 Method → Generalization 逻辑。

## 6. Exact affected modules/files

本阶段唯一新增文件：

- `experiments/reports/stage_E7_gate_stop.md`（本 gate audit）。

不得新增或修改：

- `experiments/stage_E7_*.py`；
- `experiments/tests/test_stage_E7_*.py`；
- `experiments/stage_E7_raw_results/`；
- E6 calibration、runner、orchestrator 或分析文件；
- core model、trainer、data pipeline；
- E5 Curriculum 实现；
- E1–E6 raw artifacts。

## 7. Implementation requirements

无 E7 implementation requirements，因为 gate 未通过。

工作流在 DESIGN gate 处终止：

1. 不派发 coder；
2. 不创建 E7 algorithm/config；
3. 不派发 reviewer 审查不存在的实现；
4. 不派发 runner；
5. 不执行 imbalance × role × backbone × domain-count × dataset 矩阵；
6. 不产生或分析任何 E7 新结果。

若未来要研究 static sample 的跨设置泛化，必须由用户提出一个与 Prompt 7 不同、具有独立研究问题和新 gate 的实验阶段；不能冒充当前 E7。

## 8. Ablations required

无。E7 未获授权，因而不存在合法的 E7 baseline、ablation 或 comparison。

特别禁止：

- 把 E6 Adaptive-EMA 作为“best method”进行大规模扩展；
- 只挑 E6 的 positive seeds/metrics 后进入 E7；
- 把 EMA 平滑度当作方法成功；
- 把已 DROP 的 Curriculum 放回主表；
- 为了“顶会 evidence matrix”继续增加实验数量。

## 9. Expected logging requirements

无 E7 training/evaluation logs。

本 gate audit 自身必须明确记录：

- authoritative prompt condition；
- E6 GO/CONDITIONAL/STOP verdict；
- Adaptive-NoEMA、Adaptive-EMA 的关键失败证据；
- EMA 与 Curriculum 结论；
- E7 matrix size=0；
- no IMPLEMENT / no RUN。

不得创建空的 raw-results 目录或伪 summary 来暗示 E7 已启动。

## 10. Acceptance criteria for reviewer / orchestrator

本 gate-stop 判定仅在以下均成立时视为正确：

1. Prompt 7 的原文必要条件被识别为 E6 `GO`，不是 GO-or-CONDITIONAL，也不是“只要 E6 完成”；
2. E6 authoritative verdict 被准确读取为 STOP；
3. 没有用 secondary positive metric 推翻预注册 gate；
4. 没有隐藏 Adaptive-EMA 的 Medium transfer 退化或 NoEMA 的 sub-margin gain；
5. EMA 被记为 not justified，而非 necessary；
6. Curriculum 保持 dropped；
7. 未设计 E7 matrix、未指定新算法、未修改代码、未运行实验；
8. 明确 matrix size为0且无需后续 IMPLEMENT/REVIEW/RUN/main-analysis chain。

若任何代理在此 gate 后开始实现或运行 E7，即为 blocking workflow violation，而不是实验进展。

## 11. Conditions that would falsify the gate-stop conclusion

只有以下 authoritative state change 才会推翻本 STOP：

1. E6 的原始数据或正式分析被发现存在足以使当前 verdict 无效的实质错误；
2. 该错误经过既定 reviewer 流程修正后，E6 依据原预注册判据正式改判为 `GO`；
3. 新的 authoritative `stage_E6_go_no_go.md` 明确记录 GO，并冻结一个通过 gate 的 adaptive method。

单纯希望补充 generalization、某个 E6 secondary metric 为正、Adaptive-EMA 均值略高于Sample、或 sample weighting 本身很强，都不能推翻 STOP。

## 12. Gate verdict and workflow termination

**E7 GATE: STOP — NOT AUTHORIZED.**

理由：E6 未通过 Prompt 7 要求的 GO gate，而是正式 STOP；NoEMA/EMA 均未通过 GO 或 CONDITIONAL。E7 没有独立于该必要条件的合法入口。

因此：

- E7 DESIGN experiment specification：不创建；
- E7 IMPLEMENT：不需要且禁止；
- E7 REVIEW：无实现可审；
- E7 RUN：不需要且禁止；
- E7 main analysis：无新结果可分析；
- 新实验数：0。

科学记录应保留当前负结论：Sample weighting 很强；exact adaptive calibration 未无 trade-off 地超过它；EMA 未被证明必要；Curriculum 已被 DROP。不得为了形成完整论文叙事而继续 E7。
