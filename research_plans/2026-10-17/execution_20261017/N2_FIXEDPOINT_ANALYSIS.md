# N2 固定点结果：域计数分解与固定状态 Adam 映射

状态：**固定点 16/16 COMPLETE_VALIDATED；N2 全程 quota 训练仍待完成**。这不是本轮最终报告。新模型 test 未读取。设计、实现、审查和运行授权见 N2_DESIGN_ADDENDUM、N2_READOUT_ADDENDUM、N2_N3_REVIEW 及验证重试文件；按 start.md §11，审查由主 agent 完成，`independent_review=false`。

## 人口、重复单位与可追溯性

固定 KuaiRand 训练人口 N=2,006,132；Adam η=.01，B=40,960。四个训练 seed 2032–2035，各取 SN/SB 原生路径 E10/E40，共 16 锚；每锚四臂（IID/quota × SN/SB），每臂 K=512。路径、阶段和 MC draw 都不是新增独立训练 seed。所估计的是固定 θ、原生 m/v/step 下的条件分布。

`N2_fixedpoint_validation.json` 核验完整 draw、计数、有限量、检查点与配置身份。`N2_fixedpoint_summary/receipt.json` 保存每个原始 JSON/CSV/本地 NPZ 的 SHA；大坐标矩文件保留本地，不上传 Git。正式 readout 使用 `N2_fixedpoints_resolved_manifest.json`：原成功 14 锚，加 2 个验证修正重试。原来的两个 FAILED 状态、日志及配置均保留，不伪装成首次成功。

重试只更正身份验证：原跨 batch 形状 forward 的 CUDA float32 断言受到数值差异影响；同图梯度代数身份、float64 对照和逐字估计器等价证明见 `N2_IDENTITY_FAILURE_DIAGNOSIS.md` / `N2_validation_only_equivalence.json`。同 seed、同 draw stream、同 checkpoint、同 K 和同估计量。新记录的首 draw 跨 forward 残差在有定义子块至多约为梯度均值 MC RMS 的 4.2%；这只是该 draw 的量级比较，不是所有 draw 的统一误差界。调度协调与外层状态元数据恢复的两次机械问题也在诊断文件记录，未改变科学数据。

## 1. 固定点 trace：稳定的 q 差异与很小的域计数项

以下为每路径/阶段四训练 seed 的**比值均值**，不是先均值 trace 后求比值。

| 原生锚路径 | 阶段 | IID SB/SN | quota SB/SN | SN quota/IID | SB quota/IID |
|---|---:|---:|---:|---:|---:|
| SN | E10 | 2.505010 | 2.498725 | 1.003812 | 1.001487 |
| SN | E40 | 2.378294 | 2.374368 | 0.995438 | 0.993817 |
| SB | E10 | 2.478118 | 2.449725 | 1.014960 | 1.003729 |
| SB | E40 | 2.123074 | 2.101472 | 1.002827 | 0.993100 |

全参数 ALL 的 IID SB/SN trace 比在全部 16 锚均为 2.008914–2.584112，quota 为 1.994669–2.550850。IID 预测分解中的域间计数项只占总 trace **0.001297%–0.105936%**。预测总 trace 与经验 trace 的比对范围为经验/预测 .991556–1.002505；分解的域内协方差也是同一 draw bank 的估计，不能当作独立 oracle 的精度验证。

**STRONG（限于这四 seed、两路径、两阶段、大 batch 固定点）**：q 间全参数方差差异在固定域计数后保持，域计数项不构成这些锚点方差的主要来源。这直接反驳“观察到的 trace 差主要由域计数波动产生”的局部解释。并不排除较小 batch、别的模型或阶段出现不同结果，也不能据此取消已经触发的长期 quota 训练。

**MEDIUM（跨块异质性）**：C 块 IID SB/SN 为 2.038371–2.574374，所有锚均大于 2；routing R 为 .211109–2.332441，部分小于 1。故“SB 给每一模块统一乘以同一个方差因子”不符合数据。所有已报告块有定义的 IID 计数份额最大值仅 .362906%（C_shared_STAR），但零梯度块的比值未定义，不能填零。SB 权重二阶矩 3∑p²=2.295181 只是采样权重属性，不能等同梯度方差或有效样本数。

## 2. Adam 映射：未识别出超出 MC 尺度的 q 均值漂移

32 组锚×采样变体的 ALL 块中，`||mean U_SB − mean U_SN|| / (MC_RMS_SB + MC_RMS_SN)` 为 **.479189–.624439**。分母是边际 MC RMS 之和，既不是置信区间，也不是正式显著性阈值。固定滞后分母后的均值对比范数 / 原生 Adam 对比范数为 **.910502–1.054373**，没有观察到明确的“固定分母后差异消失”。

**WEAK / INCONCLUSIVE**：当前 K=512 的全参数读数不足以确认 q 引起的总体平均更新漂移，更不足以把长期性能差归因于即时非线性。有限 bank 的 Jensen 残差可以非零；该量不等于已确认的总体 q 均值差。C_star_mlp 的个别对比/MC 尺度可至 1.531915，P 可至 1.105636，R 可至 1.120741；这些独立挑出的范数也不形成统计确认。不追加抽样搜索阳性，不据此声称历史累积机制必要。

`SGD_affine_mapping_control.csv` 将同一 bank 的梯度矩显式变换为单位学习率 SGD 均值对照：两 q 的更新均值差范数等于梯度均值差范数（同 θ 的 coupled L2 相消），MC RMS 尺度也相同。`E[U(g)]=U(E[g])` 是此仿射映射在精确算术下的恒等式；不把它伪装成另一次数值测量恰为零。理论总体 q 均值相等，有限 bank 仍有抽样差与浮点精度边界；没有保存逐draw跨q协方差，边际RMS之和也不是配对CI。此项只复用既有矩，不增加draw或补造显著性。

ALL 的这项有限 bank SGD 对比/边际 MC RMS 之和实际为 .360744–1.251960；因此即使仿射映射下也可得到非零范数，且个别范数超过 RMS 尺度。RMS 是二阶量级而非置信上界，不能把比值 1 当显著性门槛，也不能将这项对照误称为从 SGD 训练路径得到的锚点。

## 3. 阶段与模块反例

`zero_sampled_gradient_blocks.csv` 记录：E40 的 SN 路径 seed2032/2033/2034 和 SB 路径 seed2032，C_shared_STAR 与 C_star_mlp 在四臂采样中均呈零数据梯度均值、零更新 MC 变异，但原生 Adam 更新均值非零。E10 的八个路径锚没有对应的零均值现象。

**MEDIUM（描述性重复现象）**：同一模型不同训练阶段和路径的模块状态不同。非零更新与已有 m/v 历史及 coupled L2 正则兼容；这些读数不能单独确定具体激活失效原因、证明每一人口行梯度均零、证明模块无作用或证明历史效应是必要条件。N3 必须依据实际更新范数判断方向是否定义，并报告完整替换和 norm/direction 结果，不能把零方差比值解释为收益。

## 4. 尚待区分的解释与必要后续

1. 域内样本梯度的分布/方向异质性可产生较大的 q 方差差异，当前分解支持其量级；它是否导致长期 AUC/LL 差异仍未确定。
2. q 的批内依赖、高阶矩及累计路径可能影响下游；quota 同时改变这些量，不能视为纯 trace 开关。
3. 固定状态瞬时均值效应目前分辨率不足；不能从未检出反推不存在，更不能推导历史必要性。
4. 域大小与梯度分布、目标权重、基准难度相关，但本实验不单独操纵域大小或难度。没有证据把小域当作困难域，也没有从 trace 推导小域获益/大域受损。

最能区分当前解释的**已规定实验**是完整 quota 四臂的 q×quota 性能交互及每个 q 的绝对变化，以及 N3 双路径、独立 seed 的局部方向确认。两者正在既定流程中执行；不因本节机制读数追加新矩阵。只有全部规定/触发训练和选择闭合后才统一解封 test，最终结合八 seed AUC、LL、六 cell、训练阶段和人口基准分析。

## 产物与复算

- `N2_fixedpoint_summary/raw_moments.csv` / `trace_contrasts.csv`：每 seed、路径、阶段、模块四臂。
- `N2_fixedpoint_summary/mean_mapping_contrasts.csv`：Adam 原生/固定分母/Jensen 与 MC 尺度。
- `N2_fixedpoint_summary/block_context.csv` / `ALL_path_stage_context.csv`：本文数值来源。
- `N2_fixedpoint_summary/mapping_context.csv` / `zero_sampled_gradient_blocks.csv`：阶段零梯度反例。
- `N2_figures/fixed_point_overview.png` 与 PDF：已视觉核对；点重用四训练 seed，图中 RMS 线不是 CI。

主 readout 入口 `experiments/n17_geometry_readout.py`，使用 resolved manifest 及 `--validation-equivalence`；描述上下文入口 `experiments/n17_geometry_context.py --summary <N2_fixedpoint_summary>`；图入口 `experiments/n17_geometry_figures.py --summary <N2_fixedpoint_summary> --out <新的图目录>`。上下文脚本只读正式产物，无新增训练、抽样或 test 访问；主 agent 已核对比值方向、百分比单位、重复单位、零分母和源 SHA。
