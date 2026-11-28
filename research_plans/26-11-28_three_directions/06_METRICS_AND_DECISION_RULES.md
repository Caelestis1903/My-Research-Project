# 06 — 指标、统计、选点和停止规则

## 1. 主要评价顺序

### 第一优先级：selected checkpoint
对每个 seed、每个 arm、每个 readout 独立选择：
\[
e^*=\arg\max_{e\in\{1,\dots,50\}} AUC_{Val}^{Macro,6cells}(e).
\]
严格 tie 取最早 epoch。报告：
- selected epoch；
- Validation Macro AUC；
- 对应 Test Macro AUC；
- 六格 AUC；
- 同 checkpoint Macro LogLoss；
- Traffic/Pooled（若协议提供）；
- 相对直接父方法的配对增量。

LogLoss 不另选 epoch。不得使用 E50 替代 selected checkpoint。

### 第二优先级：形成 selected 的过程
- 最佳点首次达到时间；
- selected 之前的训练损失、梯度范数、pair margin、梯度 cosine 和模型预测变化；
- A/B 的梯度校正/噪声/Pair exposure；
- C 的历史 pair formation 和 forgetting。

### 第三优先级：selected 之后
- 后期 AUC 退化；
- 过拟合；
- 梯度/参数稳定性。
这些指标用于机制解释，不取代 selected 性能。

## 2. AUC 准入门槛

| 类别 | 对照 | 门槛 |
|---|---|---:|
| 新机制 | 原始 M3oE fixed1 | Δ selected Test Macro AUC ≥ +0.002 |
| 改进机制 | 直接父方法 | Δ selected Test Macro AUC ≥ +0.001 |

A 方向：
- P0 是新增 Pairwise objective，相对 B0 作为新机制，门槛 +0.002。
- GP0 相对 G0 是改进已有 Gaussian 增强，门槛 +0.001。
- GP0 的整体收益相对 M3oE 单独列示，不能将 G0 已有收益计入 GP0 的新增贡献。

B 方向：
- A21 相对 ordinary PAIR Adam 属于历史辅助机制候选；B2 相对 A21 是改进机制，门槛 +0.001。
- 不把 EXP082 A21 的历史 Test 增量替代新一轮正常 Validation 选点证据。

C 方向：
- C3 相对 C1/C2 的关系记忆增量属于改进机制，门槛 +0.001。
- 若作为新算法整体与 M3oE 比较，仍报告 +0.002 新机制门槛。

## 3. 六格保护与 LogLoss

- 六格 Macro AUC 平均保护线：候选−父方法 ≥−0.001。
- 逐格 AUC 全部列出，不以 Macro 均值隐藏局部大幅退化。
- LogLoss 单独报告，候选−父方法 <0 为改善，>0 为恶化。
- AUC 与 LogLoss 不同向时按双指标 trade-off 报告，不把 LogLoss 改善称为排序收益。
- 如果 AUC 通过门但 LL 明显恶化，报告该代价及六格分布，不自动否决；由研究目标和现有阈值决定是否保留。

## 4. 配对统计

训练 seed 2022–2026 是配对单位。对每 seed 计算候选减父方法：
- mean paired Δ；
- paired SD；
- 95% paired CI；
- positive seeds / 5；
- 六格 paired Δ；
- seed-wise selected epochs；
- 成本比。

置信区间使用 t 区间或项目已冻结的同一 paired bootstrap 方案；全项目统一，不允许为候选选择更有利的统计方法。报告样本量 n=5。方向数只作描述，不作为额外淘汰门槛。

## 5. 超参数

- A：τ=1 固定；κ 使用原 GA31-R3 数据集值；pair cap=256/cell/batch；λp 用 Train-only seed2022 calibration，标定后冻结。不得对 κ/τ/λp 做广泛搜索。
- B：β=0.99、λ_PAIR 与 A21 原值固定；γ=0.5 主候选；γ=0.25 只按 B2 Validation 触发。
- C：T=1，margin clip [-5,5]，memory capacity 512/cell，构建 snapshot epoch 5/10，冻结于 epoch10。不得根据 Validation/Test 筛 memory。
- 所有参数均需将最终值、来源、标定方式和 config hash 写入 manifest。

## 6. 停止条件

- A：GP0 相对 G0 未达到 +0.001 或六格保护失败，关闭 Gaussian Pairwise；不继续换噪声类型。
- B：B2 相对 A21 未达到 +0.001 或六格保护失败，关闭 freshness correction；不继续调 β/λ 或重复 C1/C2。
- C：selected 前历史正确 pair flip 不稳定、与 current sampler 冗余、无法合法取得 Train-only checkpoint 或 C3 不超过 C2/C1，关闭 C。
- 任何方向若仅 LogLoss 改善、AUC 不达门槛，保留为概率校准结果，不晋级为排序方法。
- 任一新机制若相对 M3oE Test ΔAUC 未达 +0.002，不作为独立新机制保留；但若作为已有机制改进超过 +0.001，可作为组件增量保留。
- 组合实验仅在两个独立机制分别通过时触发。组合未超过单方法最佳，不保留组合。
- 不因单个数据集失败而否定另一个数据集已通过的结果；按 dataset-specific scope 报告。

## 7. 报告模板字段

| Dataset | Arm | Parent | Readout | Selected epoch | Val Macro AUC | Test Macro AUC | Test ΔAUC | Test Macro LL | Test ΔLL | Six-cell AUC Δ | CI95 | Positive seeds | Cost |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|

每个候选还需单独给出：
- 新机制/改进机制身份；
- 直接父方法；
- 最强发表基线；
- Validation 选择是否通过；
- Test 授权和历史暴露标记；
- selected 前机制证据；
- selected 后辅助诊断；
- 保留、关闭或待授权决策。
