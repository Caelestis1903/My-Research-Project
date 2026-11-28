# C0 独立研究评审

**裁决 PASS。MovieLens-1M 和 KuaiRand-Pure 均满足 C1 的科学触发条件，且容量裁定已完成。** 复核覆盖 20 条自然采样 M3oE fixed1 轨迹的 60 个 FP32 fixed EMA 快照、10 份 Train 候选和 20 份 margin 档案。独立重算 60 个窗口的逐条件翻转率与精确 pair 重合，复现 80 个采样边界的配对哈希。从 1,000 份 A1 逐轮记录重新确定 20 个 Validation 选点并对齐 AUC 与 LogLoss。当前 10 项停止控制均未触发，C0 状态记录的 Test 访问为 0。[执行设计](C0_EXECUTION_DESIGN.md) · [运行报告](C0_RUNNER_REPORT.md) · [分析](C0_ANALYSIS.md) · [状态](../../artifacts/three_directions_20261128/C0/STATUS.json)

## 翻转幅度与训练阶段

**STRONG。** E10 严格正确的 Train 配对到选点时发生严格翻转，在两个数据集、两组各五个种子上复现。严格翻转采用 E10 margin > 0、选点 margin < 0。稳健指标沿用相同分母，要求 E10 margin ≥ 0.001、选点 margin ≤ −0.001。E10→选点未出现历史正确配对在选点打平。以下比率先在每个种子内按 macro6 汇总，再跨种子取均值和样本 SD。[配对明细](../../artifacts/three_directions_20261128/C0/PAIR_METRICS.json) · [种子汇总](../../artifacts/three_directions_20261128/C0/SUMMARY.json)

| 数据集 / 组 | E10 严格率，均值 ± SD（%） | 五种子范围（%） | E10 稳健率，均值（%） | E5 严格率，均值（%） |
|---|---:|---:|---:|---:|
| MovieLens B0 | 2.947 ± 0.232 | 2.645–3.206 | 2.931 | 3.479 |
| MovieLens P0 | 2.788 ± 0.162 | 2.532–2.922 | 2.775 | 3.523 |
| KuaiRand B0 | 11.246 ± 0.314 | 10.797–11.648 | 11.177 | 14.837 |
| KuaiRand P0 | 8.334 ± 0.460 | 7.739–8.831 | 8.284 | 12.672 |

**MEDIUM。** KuaiRand 的 P0−B0 翻转率在 E10→各自选点时五个种子均为负，均值 −2.912 个百分点，95% 配对 t 区间 [−3.431, −2.393]。两个组的窗口分别为 11–19 和 23–40 轮。MovieLens 的配对差为 −0.159 个百分点，区间 [−0.518, +0.199]。20 条轨迹上 E5→选点的严格翻转均高于 E10→选点，起点和时长同时变化。共同五轮的 E5→E10 比较在 KuaiRand 得到 +0.280 个百分点 [−0.612, +1.173]，在 MovieLens 得到 +0.119 个百分点 [+0.013, +0.225]。E5→E10 描述早期形成。估计 E10 后共同轮距的保持效应需要对齐 E10 后的终点，且需区分同一候选池与各组自身的历史正确子集。[种子汇总](../../artifacts/three_directions_20261128/C0/SUMMARY.json)

## 精确重合与置信度

**STRONG｜有限窗口内的精确重合。** 以条件和有序正负 Train 行 ID 识别同一配对。独立重放的命中抽数保留重复，覆盖数按成员去重。均匀参照条件于实际抽数，合法配对空间由同一 Train 标签池重建。MovieLens B0/P0 对翻转子集的实际命中均为 0，期望分别为 0.053 和 0.042。KuaiRand B0/P0 分别命中 626 和 217 次，期望为 631.93 和 210.74。KuaiRand 的翻转子集 macro6 覆盖每百万关系实测为 152.84 和 176.56，参照为 267.80 和 97.08。局部覆盖更高，B0 seed 2023 的 d2t0 条件 (2,0,2,1) 实测覆盖 14.20% 的历史正确配对，参照 13.92%，该条件的 macro6 权重约为 0.000081。结果支持有限窗口中精确重合稀少。采样器对合法配对具有完全支持，端点、标签与相似样本带来的监督可替代性需由训练对照判定。[采样重合](../../artifacts/three_directions_20261128/C0/OVERLAP.json)

**MEDIUM｜历史 margin 的描述性区分。** 四组各五个种子均随 E10 正 margin 档位升高呈现翻转率下降。最低档 (0, 0.1] 的合并原始翻转率为 27.77%–30.76%，最高档 (2, 5] 为 0.002%–0.085%。95.5%–98.0% 的翻转来自历史 margin ≤ 1 的配对。这个单调关系与到排序反转边界的距离一致，支持用置信度描述稳定性。软 margin 对训练的增量价值由 C4 的同配对硬标签对照检验。[置信度分箱](../../artifacts/three_directions_20261128/C0/decisions.json) · [配对明细](../../artifacts/three_directions_20261128/C0/PAIR_METRICS.json)

## Validation 关联与 C1 触发

**STRONG｜所见轨迹，WEAK｜训练收益。** 四组在所有种子上均取得比 E10 更高的选点 Macro AUC，选点定义使用该指标。E10→选点的 Macro LogLoss 在 MovieLens 两组全部上升，在 KuaiRand 两组全部下降。两任务五种子均值呈现相同的数据集方向，MovieLens P0 的任务 0 有一个种子的 LogLoss 小幅下降。较高选点 AUC 与 Train 翻转并存，强制保留历史排序对 AUC 和 LogLoss 的方向待测。[Validation 对齐](../../artifacts/three_directions_20261128/C0/VALIDATION.json)

**MEDIUM｜域规模与局部权衡。** KuaiRand B0 翻转最多的是规模居中的 d0，两格 E10 AUC 最低。最大域 d1 翻转最少，最小域 d2 的两格翻转居中。MovieLens 最高翻转格 d2t0 位于最大域。KuaiRand 的 P0 选点 AUC 相对 B0 在 d0 两任务均升、d1 两任务均降，在最小域 d2 一升一降。这个局部权衡属于 A1 当前配对损失比较。域训练行数、基线 AUC 和选点时距各自提供不同信息。[Validation 对齐](../../artifacts/three_directions_20261128/C0/VALIDATION.json) · [A1 分析](A1_ANALYSIS.md)

两个数据集各自通过计划 03 §7 的四项条件。选点前翻转幅度在五个种子上稳定复现，有限窗口的精确重合率低且 margin 提供可测区分，历史配对可通过已核验的 Train 快照重建。用户批准每个 active condition 最多存储 2,048 个唯一配对，每次有放回抽取 2,048 对。在现有 A1 P0 档案中，MovieLens 的 60 个种子×条件单元均有至少 2,048 个 E10 正确候选，KuaiRand 的 100 个单元中有 89 个达到该数，最少有 1,538 个。C1 设计应以实际 C3 bank 大小匹配同条件 C2，保留固定事件剂量。该可用数量基于现有 A1 轨迹，实际 bank 和对照复用资格由 C1 的配置、源与快照身份核验确定。[实验矩阵 §7](../../research_plans/26-11-28_three_directions/03_EXPERIMENT_MATRIX.md#7-c0历史关系离线审计) · [容量裁定](../../artifacts/three_directions_20261128/C0/decisions.json) · [算法设计 §3](../../research_plans/26-11-28_three_directions/02_ALGORITHM_DESIGN.md#3-c历史正确排序关系记忆)

**C1 进入设计与审批。** C3−C2 在相同候选池、容量、条件与曝光下检验历史正确性筛选，C3−C1 检验历史 bank 相对当前配对的增量。两项比较的改进门和六格保护共同判定 C3 的去留。达到改进门后，C4 保持相同 pair 身份并改用硬标签，以识别软 margin 的贡献。C0 判定支持运行这组证伪对照，Train 翻转仅提供机制诊断。[实验矩阵 §8](../../research_plans/26-11-28_three_directions/03_EXPERIMENT_MATRIX.md#8-c1历史关系记忆训练) · [决策规则](../../research_plans/26-11-28_three_directions/06_METRICS_AND_DECISION_RULES.md)
