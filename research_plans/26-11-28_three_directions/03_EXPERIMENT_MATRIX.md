# 03 — 实验矩阵、阶段依赖与触发条件

## 1. 全局固定协议

- Dataset：MovieLens-1M、KuaiRand-Pure；objective 固定为 Sample/natural objective，禁止切换回 Equal objective。
- Seeds：2022、2023、2024、2025、2026。
- Backbone：M3oE fixed1。
- 数据切分和自然样本分布保持项目 canonical identity。
- 训练预算：50 epoch；batch size 40960；FP32；复用对应 dataset 的 canonical learning rate、weight decay、optimizer epsilon/betas 和 EMA rho。S0 必须从当前 main protocol manifest 读取确切值并写入 frozen config；若某值与历史规划不同，报告具体差异后暂停相关 job，不自行猜值。
- 每个 seed 的 online 与 fixed EMA 分别独立按 Validation 六格 Macro AUC 选点；精确并列选最早 epoch。
- Validation 决策完成并冻结后，才执行 Test。
- 记录 Macro、六格、Traffic、Pooled（若该协议提供）、LogLoss、selected epoch、SD、95% paired CI、正向 seed 数和计算成本。

## 2. S0：来源审计

| 审计项 | 结果必须包含 |
|---|---|
| main HEAD / file SHA | 本方案来源身份、当前代码 HEAD、关键文件 blob SHA |
| 数据 | 数据文件 SHA、split row-id hash、Train/Val/Test population、标签支持 |
| 训练协议 | M3oE fixed1、batch、LR、WD、Adam、EMA rho、epoch、seed |
| Gaussian | GA31-R3 源实现、噪声键、字段顺序、D0 coverage、row identity |
| PAIR | A21 sampler、cell、w/q、pair event clock、lambda、scope |
| 结果 | EXP069、EXP080/081 和 EXP082 用户补充证据身份分开 |
| Test | 当前 Test 授权状态和历史暴露身份 |

S0 不训练。任何身份 mismatch 阻塞受影响方向，不能用近似配置继续冒充原方法。

## 3. A1：最小矩阵

每数据集 4 arms × 5 seeds = 20 trajectories；两数据集共 40 条，每条 50 epoch。

| Arm | BCE view | Pairwise view | EMA | 父方法 | 机制类别 |
|---|---|---|---|---|---|
| B0 | clean | none | fixed | M3oE fixed1 | baseline |
| P0 | clean | clean | fixed | B0 | new objective component |
| G0 | GA31-R3 D0 | none | fixed | B0 / G-D0 | established enhancement |
| GP0 | GA31-R3 D0 | Gaussian view | fixed | G0 | improved mechanism |

主判断：
- P0−B0 ≥+0.002 才能将 clean pairwise 作为新机制候选。
- GP0−G0 ≥+0.001 才能将 Gaussian pairwise 作为 G-D0 的改进候选。
- GP0−P0 判断 Gaussian 对 pairwise 的额外贡献；不要求此项单独通过门槛，但必须报告。
- 若 GP0 不超过 G0 +0.001，关闭 Gaussian Pairwise。
- 所有比较都先看 selected Validation，再看对应 Test；Test 不反向选候选。

## 4. A2：条件触发消融

只有 A1 的 GP0 相对 G0 通过 Validation 门且六格保护线通过才触发。
最多 2 个候选：
- GP-shared：pair 内同字段共享 noise；
- GP-independent：pair 正负样本独立 noise（A1 默认）。
不再搜索 κ；不加入额外 loss family。每个候选两数据×5 seeds×50 epoch。若共享/独立差异低于 +0.001，保留成本更低者；若均未超过 GP0 父方法，关闭 A。

## 5. B1：历史梯度新鲜度校正

每数据集最多 3 arms × 5 seeds = 15 logical trajectories；若 A1 P0 可严格复用，B1 阶段只新增 B1/B2 共 20 条训练轨迹。
| Arm | PAIR source | Freshness correction | 父方法 |
|---|---|---|---|
| B0 | ordinary PAIR Adam | none | 复用 A1 的 P0（仅当完整 config/source hash 一致） |
| B1 | A21 collapsed history | none | A21；若 EXP082 轨迹不可从 main 合法读取则重训 |
| B2 | A21 collapsed history | γ=0.5 same-batch shared-state correction | B1 |

B1 的历史 Test 信号必须标记 EXP082_HISTORICAL_TEST；如当前可用训练轨迹可重建 A21，则新训练用于正常 Validation 选点，不能复制旧 Test 选择身份。
- B2−B1 是主增量，改进门 +0.001。
- B1−B0 是历史辅助梯度整体增量。
- B2−B0 是整体方法相对普通 PAIR Adam 的收益。
- 任何结果都分别对照 HyperGate/AutoIFS/M3oE。
- 若 B2 未通过 Validation 改进门或六格保护，关闭 B2，不做 γ scan。

## 6. B2 条件确认

只有 B2−B1 的 Validation ΔAUC≥+0.001、六格保护通过且 Test 对照未出现实质反向时，才用 γ=0.25 做一次有限确认。此确认用于判断 γ=0.5 是否过校正，不进行更广网格。γ=0.25 的父方法为 B1，属于改进机制，门槛仍 +0.001。

## 7. C0：历史关系离线审计

0 条新训练。使用可用 selected checkpoint 和 Train-only pair：
- 先核实 checkpoint、row IDs、预测缓存及 SHA；
- 评估历史 pair flip rate 和 margin；
- 量化与当前 pair sampler 重合；
- 统计固定记忆预算下的可用关系数量和曝光次数。

触发 C1 的条件：
1. selected checkpoint 之前存在稳定且非微小的历史正确 pair flip；
2. 关系 flip 在至少两个数据集或一个数据集多个 seed 上重复；
3. 历史正确关系与 current pair sampler 不高度重复，或历史 margin 提供了可测增量；
4. 可合法重建记忆 pair，且 Train-only 无泄漏。

不满足任一关键条件则 C 关闭，不训练。

## 8. C1：历史关系记忆训练

最多 4 arms × 2 datasets × 5 seeds = 40 logical trajectories，每条 50 epoch。C0 clean BCE 与 C1 current pairwise 可分别复用 A1 B0/P0（仅当完整 config/source hash 一致）；若复用成立，C 阶段最多新增 C2/C3/C4 共 30 条训练轨迹。
| Arm | Objective | Memory |
|---|---|---|
| C0 | BCE | none |
| C1 | BCE + current pairwise | current batch pairs |
| C2 | BCE + pairwise memory | random historical Train pairs + historical margin, no correctness filter |
| C3 | BCE + pairwise memory | historical correct Train pairs + historical margin |\n| C4 | BCE + pairwise memory | same historical correct pair identities + hard pairwise labels, no margin distillation |

所有 arm 使用相同 pair count、每 epoch 曝光预算、lambda_pair=0.05（沿用 A21）、batch 和 optimizer。C3−C2 为主比较，检验筛选历史正确关系的价值；C3−C1 检验历史记忆是否超出当前 pairwise 的价值。C4 hard-label control 只在 C3 通过改进门后触发，且每数据×5 seed。

## 9. S4：机制组合

仅当两个单独方向分别通过各自门槛，且其机制证据支持互补时允许一个组合：
- A + B：A 产生更直接的 pairwise view signal，B 校正 PAIR 历史梯度在共享参数空间的当前适配。
- A + C：增强 view 与历史 relation memory 是否提供互补的排序约束。
- B + C：历史梯度信息和历史函数关系信息是否互补。

一次只选一个组合；父方法为两单独通过的最佳方法，改进门 +0.001。禁止全组合搜索。若没有两个独立机制通过，不执行 S4。

## 10. 阶段转移表

| 当前阶段 | 结果 | 决策 |
|---|---|---|
| S0 | 数据/代码/协议身份未核实 | PAUSE，修复来源，不运行科学训练 |
| A1 | GP0−G0 <+0.001 | 关闭 A-Gaussian pairwise；保留其他单独 pairwise 结论 |
| A1 | P0−B0 ≥+0.002 且六格通过 | 保留 clean pairwise，按 A 结果决定是否做有限后续 |
| A1 | GP0−G0 ≥+0.001 且六格通过 | 触发 A2 |
| B1 | B2−B1 <+0.001 或六格失败 | 关闭 freshness correction |
| B1 | B2−B1 ≥+0.001 且六格通过 | 触发 γ=0.25 有限确认 |
| C0 | 无 selected 前关系遗忘/关系冗余 | 关闭 C |
| C0 | 历史关系存在且有独立信息 | 触发 C1 |
| 单方向 | 新机制总门 ≥+0.002 且六格通过 | 作为完整候选保留 |
| 单方向 | 仅相对直接父方法 ≥+0.001 | 作为改进组件保留，完整方法仍按外部基线裁定 |
| A/B/C | 至少两个独立方向合格且有互补证据 | 触发一个组合 |
| S4 | 组合不超过单方法最佳 | 不保留组合，独立方法继续 |
| 全部 | 配置冻结 | 依据权限执行 Test、报告、上下文更新 |

