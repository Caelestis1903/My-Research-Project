# 00｜Codex Execution Runbook

## 0. 运行授权与禁止事项

本目录处于 DESIGN_ONLY。Codex 获准完成本文件定义的来源审计、实现、独立审查、训练、分析和报告；不获准修改历史实验源码、历史结果、既有上下文文件或其他研究分支。

新代码仅放入：
- `experiments/mdmt_shared_private_coordination_20261123/`
- `research_reports/mdmt_shared_private_coordination_20261123/`
- `artifacts/mdmt_shared_private_coordination_20261123/`

本规划文件在 `research_plans/11-23/`。所有写入只新增本轮目录文件。

禁止：
1. 改写或覆盖 M3oE 源实现、数据文件、历史报告、历史结果和历史模型基线。
2. 改变数据人口、划分、标签、batch、数值精度、模型结构、fixed1 融合、Adam/EMA 设置以方便得到正结果。
3. 用 Test 选 scope、学习率、更新顺序、checkpoint 或决定是否继续。
4. 将参数组名称直接视为真实计算图支持；必须执行参数和梯度支持审计。
5. 将一步局部收益写成长期 AUC 收益。
6. 在 Stage 1 未通过时继续 Stage 2；在 Stage 2 未通过时开放 Test。
7. 以小域、样本少或梯度噪声作为困难度代理。
8. 启动未登记的后台 Codex 会话、独立 runner 或无限自动循环。

## 1. 阶段总览

| 阶段 | 内容 | 训练量 | Test | 进入条件 |
|---|---|---:|---|---|
| S0 | 来源/数据/模型/资源审计，参数分组与梯度支持验收 | 仅工程 smoke | 禁止 | 无 |
| S1 | 标准 M3oE 五种子轨迹 + 共享/专有更新顺序的受控诊断 | 10 条 × 50 epoch 基线；诊断只在克隆状态执行 | 禁止 | S0 PASS |
| S2 | 固定协调策略的完整五种子训练比较 | 仅在 S1 选出一个 scope 后运行 | 冻结前禁止 | S1 GO |
| S3 | 最终冻结、Test 评价、统计和交付 | 不新增训练 | 一次性开放 | S2 GO 且 freeze PASS |

本研究不是要求证明每个假设。若关键机制不存在或无法转化为更新收益，按规定结束并报告。

## 2. S0：来源审计与模型结构验收

Codex 首先读取：
- 仓库 `main` 的 `EXP_AI_CONTEXT.md`、`experiments/Experimental_Global_Summary.md`、`experiments/Experimental_Stable_Findings.md`、`experiments/Experimental_Open_Questions.md`、`research_context/experiments/` 最新对应上下文、`MDMT_ALGORITHM_METRICS_LEDGER.md`。
- 最新分支 `26-11-23` 的条件历史与共享更新报告、FINAL_REVIEW、A/B 报告、S5 数表。
- `m3oe/MDMTRec/models/multi_domain/model.py`、当前 fixed1 canonical adapter、当前训练/评价协议。
- 本目录 01–09 全部文件。

建立 `SOURCE_MANIFEST.json`：记录 Git commit、每个源文件 Git blob/SHA256、数据 raw/split/encoder/scaler/row-order hash、Python/PyTorch/CUDA、GPU、关键依赖。不得仅写“与历史一致”。

### S0 必须通过的检查

1. 真实 M3oE fixed1 forward 与当前 canonical runner 输出一致。
2. 3 域 × 2 任务输出索引与 `gate[3*t+d]`、`tower[3*t+d]` 一致。
3. 每个参数恰好属于 U、P_CELL、P_AXIS 或 FROZEN/OTHER 中一个集合；无重叠、无遗漏。
4. 每个参数组在每个 cell 的真实梯度支持与前向依赖均有记录。特别检查 gate 输入 detach、domain/task expert 的跨 cell 支持、STAR slot 的域内双任务支持。
5. fixed1 的四个历史融合权重是否冻结、是否参与优化器必须从实际构造器确认；如仍可训练则 BLOCKED，不得自行冻结或移除。
6. 同一 batch 的完整联合 Adam 更新在新 adapter 与原 canonical runner 中达到参数、optimizer state、EMA 的数值一致性。
7. U-only/P-only 子步的 `grad=None` 与显式零梯度语义经过测试；没有梯度的参数不增加 Adam step，不执行 weight decay。
8. 诊断 clone 不改变 baseline model、optimizer、EMA、RNG、DataLoader 顺序和训练计数。
9. FP32/TF32/AMP 状态、自然尾批和六格指标实现通过验收。

若 S0 有结构或来源问题，标记 TECHNICAL_BLOCKED；不得把工程阻塞写成科学阴性。

## 3. S1：受控机制诊断

运行两数据 × 五 seed 的标准 Clean M3oE fixed1 50 epoch 轨迹。Seeds 仅 2022–2026。该轨迹同时提供 E10、E25、E40 结束状态及完整 Adam/EMA/RNG/loader 状态。

在每个锚点，从下一 epoch 的前两个完整自然训练 batch 构造 A/B。对每个锚点建立独立 RAM clone，执行 03 中定义的：
- JOINT-2（AB、BA）
- JOINT-ACC
- U-ONLY、P_CELL-ONLY、P_AXIS-ONLY
- U→P_CELL / P_CELL→U
- U→P_AXIS / P_AXIS→U
- U→P_FULL / P_FULL→U

AB 与 BA 均执行，避免 batch 顺序与更新顺序混杂。每个 fork 完成后在冻结的 V_A/V_B 诊断面板和完整 Validation 上评价；不得评价 Test。

S1 训练轨迹本身不得因诊断而改变。所有诊断使用虚拟/克隆状态，结束后销毁。

### S1 scope 选择

按 07 的配对规则，先用 V_A 选择 scope，再用独立 V_B 确认。候选 scope 仅 P_CELL、P_AXIS、P_FULL。选择时以跨数据集等权的 selected-like validation Macro AUC 增量为主，比较 U→P 与其反向顺序、JOINT-ACC 和 U-ONLY；并报告六格、阶段、online/EMA。

并列时按复杂度优先：P_CELL → P_AXIS → P_FULL。若所有 scope 在两个数据集的 V_B 上均无正向平均增量，或 U→P 的优势被反向顺序/JOINT-ACC 完全解释，则 S1=NO_GO，完成报告后终止。置信区间跨零不单独作为否决条件；均值、逐 seed、方向和 cell 代价全部保留。

若只在一个数据集出现可重复的正向信号，报告适用范围并由 07 规定的决策规则判断是否进入 S2；不得自动声称跨数据集机制成立。

## 4. S2：完整训练验证

仅将 S1 选出的一个 scope 固定为候选，不继续搜索其他 scope。

完整训练臂：
- J_ORIG：原始每 batch 联合 Adam + 固定 EMA。
- J_ACC：两批梯度在同一 pre-state 累积后一次联合 Adam 更新；与 block schedule 匹配每个参数组的更新次数。
- P→U：先更新选定专有 scope，再更新 U。
- U→P：先更新 U，再在更新后的表示上更新选定专有 scope；这是唯一候选算法。

每个数据集、每个 seed、每个 arm 完整 50 epochs。S1 的 J_ORIG baseline 只有在 SOURCE_MANIFEST、训练配置、初始化、row order、代码和全部状态身份完全一致时才允许复用；否则重新训练，并单列成本。

每个 epoch 完整 Validation。Online/EMA 各自按六格 Macro AUC 最早严格最大值选点。选点、scope、实现和统计脚本在 Test 前冻结。

S2 不进行新的算法结构搜索，不新增 loss 权重、不搜索 EMA rho、不叠加 Gaussian 或 PAIR。若 U→P 没有超过 J_ACC 和 P→U 的 validation 增量，或收益完全由训练更新次数/批次顺序解释，S2=NO_GO，不开放 Test。

## 5. S3：最终冻结与 Test

仅当 S2 GO：
1. 冻结源代码、配置、scope、optimizer、EMA、所有 checkpoint epoch/hash、统计代码和正式比较对象。
2. 确认每个 seed 的 selected 模型状态可读且 hash 正确。
3. Test 评价仅执行冻结名单中的必要 arms；不得按 Test 结果追加训练、换 scope 或重新选点。
4. 对历史 Test 暴露情况作明确标注；新 Test 结果不是独立数据集泛化证据。
5. 生成 FINAL_REPORT、FINAL_REVIEW、DECISION、完整 CSV 和成本表。

## 6. 终止状态

只允许以下终态：
- `COMPLETE_GO`：完成 S3，报告候选资格与适用边界。
- `COMPLETE_NO_GO_S1`：共享—专有适配机制未形成可行动信号。
- `COMPLETE_NO_GO_S2`：机制诊断存在，但完整训练无独立收益。
- `TECHNICAL_BLOCKED`：来源、代码、资源或状态完整性阻塞。
- `PAUSED_RESOURCE`：资源暂不可用，保留状态，等待当前会话恢复。

不得将 TECHNICAL_BLOCKED/PAUSED 记为算法失败；不得在终态后自动启动新研究循环。
