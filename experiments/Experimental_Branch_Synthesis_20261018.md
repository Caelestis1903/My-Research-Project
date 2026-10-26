# 多分支实验结论与遗留问题综合报告（截至 2026-10-18）


> **历史快照/已被后续实验扩展（2026-10-26修订说明）。** 本文只覆盖至N17的2026-10-18状态；其中“当前”“最新一轮”“未合入实验文件”“下一轮优先q×LR”等是当时语境，不是2026-10-26的研究决策。随后已经完成N18/N19、CM19、10-21复现、10-24 I1、10-25 RHD/LPS与10-26 DFD；10-20自然MovieLens重评完成部分子项，但扩展矩阵尚未闭环。实验原件已按分支存入 [history_experiments](../history_experiments/README.md)。完整更新与早期结论修订请以 **[2026-10-26跨阶段总报告](Experimental_Branch_Synthesis_20261026.md)** 为准；该新文件不覆盖本文原始历史数值。


> **历史定位：本文曾是main在2026-10-18时的跨轮结论入口。** 按提交时间和真实实验依赖，整合 2026-10-15—2026-10-18 未直接合入 main 的实验执行记录。仅抽取科研事实、方案状态、反例、论断演化及未解决问题，不 cherry-pick 实验代码、checkpoint、原始预测或历史阶段进度快照。2026-10-09 的 E001–E032、SF001–SF012、OQ001–OQ010 等仍保留为原始历史记录；与本报告不一致时以**同一估计对象中较晚且最终核验的结果**为准。不同训练协议/种子/数据人口不可直接合并效应量。

## 1. 分支与时间序列——先判断是新证据还是重复祖先

| 时间（UTC，按阶段完成） | 分支及报告入口 | 处理方式 |
| --- | --- | --- |
| 10-15 至 10-16 05:36 | [execute-2026-10-15：B1、C2 与范围审计](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-15/research_plans/2026-10-15/execution_20261015/B1_reconstruction_20261016/scientific_analysis.md) | 新 B1 重建及数值审计；不冒充 S6 历史精确复现 |
| 10-16 至 09:23 | [execute-2026-10-16：Stage2D 最终结果](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16/research_plans/2026-10-16/stage2d_execution_20261016/experiment_results.md) | 该分支继承 B1；早期 Stage2+ 的“0 训练/阻塞”是已被 16/16 Stage2D 覆盖的**过程快照** |
| 10-16 至 12:00 | [execute-2026-10-16-v2：P1 与机制中期分析](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16-v2/research_plans/2026-10-16_v2/execution_20261016/mechanism_analysis.md) | 继承 Stage2D，但 v2 改为固定 fusion/单优化器；最新文件仍标 INTERIM，不能宣布整个 v2 已终局 |
| 10-16 至 10-17 05:19 | [execute-2026-10-16-v3：完整终局报告](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-16-v3/research_plans/2026-10-16_v3/execution_20261016/FINAL_REPORT.md) | 从较早祖先独立开展、与 v2 **历史分叉**；必须并列保留不同固定 fusion / 校准 LR / optimizer /时钟协议 |
| 10-17 至 10-18 07:00 | [execute-2026-10-17-run：N17 最终研究报告](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/FINAL_ANALYSIS.md) | 继承 v3；追加独立优化随机种子、固定 quota、ML 固定人口和局部反事实；当前最新正式终局 |
| 10-16—10-17 方案分支 | 2026-10-16-v2-research-plan、research-plan-2026-10-16-v3、add-research-plans-2026-10-17 | 方案正文已在 main 的 research_plans/2026-10-16_v2、2026-10-16_v3、2026-10-17。**不计新增实验证据**，不重复全量复制 |

前三个实验分支在 Git DAG 上连续，v3 与 v2 发散，N17 连续继承 v3。方案分支共有 3 条，实验执行分支 5 条，总计盘点 8 条。源分支未做 Git merge，仍保持独立可复核。

## 2. 按时间先后提取的科研发现、限制与历史修订

### 2.1 10-09 之前的既有主干：P0–S8 与 S6-R2（不是本轮重训）

- **Equal / Sample 是不同优化目标**；Equal 的特定 selected 损伤不能直接视为 Sample 下的实用负迁移。既有 S1/S2 对已测 Sample/自然比例未检出稳定实用 negative transfer，**不等于 Sample 任何情况下都最优**。
- S3–S5 的 KuaiRand/Equal/d1 shared-access 与持久 Adam-history carrier 影响，在特定 checkpoint 和模型中为**有边界的局部现象**。S5 的完整 history 效果不识别一阶矩 m、二阶矩 v 或共享/architecture 哪一部分独立必要；不应扩大成 Adam 对一切 MDMT 现象的必要性。
- S6 Sample selected 的 Equal 类 target rescue 不再达到原预定实用门；但后期 MovieLens 过拟合缓和效应存在。S8 未使正常 validation 选择的最佳模型普遍优于原 F best，因此“late rescue = 最佳泛化提升”这条方法化思路被否定。
- 原 S6-R2 同一组 KuaiRand R0 checkpoint：test DH−F 非 target AUC +0.006757（4/4 正），validation −0.003851（1/4 正），MovieLens 接近零。属于**历史保存工件的局部观察**，不是已经识别了可部署因果机制；不能以 test 筛选规则。
- **“小域就是困难域”从来不是本项目应采纳的结论。** 域数据量、域特征和标签、任务、曝光及评价人口共同影响指标。后续未出现“规模决定困难度”的因果证据；针对该断言的反例应作为边界事实，不应被不断当成新的核心假设反复证伪。

### 2.2 10-15 执行分支：B1 历史现象的独立重建与 C2.1

**新增** KR natural、Sample/M3OE、F/DH×4 seed×50 epoch＝8/8 新训练；400 个完整状态、800 个预测工件数值核验通过。由于历史 init/lineage 不一致，身份为 **RECONSTRUCTION_NOT_HISTORICAL_EXACT**。R0（只按 validation macro AUC 选）test DH−F：macro +0.000363（2/4 正）、non-target +0.000579（2/4 正）、target −0.000069；target LogLoss +0.004007（恶化）。没有复现旧 S6-R2 的 +0.006757/4 正组合，**但不是证明旧报告造假或错误**。R1/R4 次选点的 test non-target +0.011584（4/4 正），两规则选择完全同一批 epoch、不计两次复制，且 target loss 代价存在。选点不是全部反号的解释：seed2022 两臂同轮仍有 val/test 差异。

**零训练** C2.1 两数据集四seed/8 case：同 alpha 改 q 的期望梯度相同，协方差不同；观察到“Balanced 必降噪”随 alpha 和所测参数分区改变的反例。仅固定 theta / 有限 pool，不能宣称训练提升或历史 Adam causality 已识别。S5/S6 原始 checkpoint、完整 history 和同θ m/v 追踪未找到；保留身份不可恢复缺口，不再当作新训练统一禁令。

### 2.3 10-16 Stage 2+ 与 Stage2D：把“未跑”的历史状态更正为“已跑但候选失败”

本分支早期 Stage2+ 文档有“0 新正式训练/依赖阻塞”的时间快照；同分支随后完成 **16/16 Stage2D**：KR/M3OE，四 seed，Sample/Equal × Natural/Balanced，**learned fusion、双 Adam 优化器**，配对 IID，50 epoch。最终效应：validation selected 宏 AUC **SB−SN = −0.010406（4/4 负）**；**EB−EN = −0.006547（3/4 负）**。Test（已用于历史开发，仅法证）分别 −0.012674 / −0.002714。Sample 同目标改变采样有明显**负向总体干预效果**，所以“Sample 下 optimizer/sampling 效应全部消失”被修订；但 q 同时改变 batch 梯度分布、重复曝光与后续优化路径，不能锁定 Adam m/v 为性能因果。两种 Balanced 候选都未通过原定实用收益/无害性门，**本配置方法化 NO_GO**；不否认某些 cell/LL 方向不同。

固定 theta 的 Stage 2B/2C 共8 case×4096 replay×64步证实 SGD/Adam 局部递推区别，**完整 C trace 与投影坐标 trace 可显示相反的 Balanced/Natural 比值**；这对“单 scalar variance 解释全模型”构成限制，而不是选中模型收益证据。旧 S5/C×A 与 m/v 专项因缺原状态而未运行；“未运行”不等于“机制无效”。

### 2.4 10-16 v2：无动量 SGD 反例，固定 fusion/单优化器的中期结论

v2 改为固定 fusion / 单一 optimizer；与 Stage2D **不是**只改 Adam/SGD 的严格历史配对。已核验 P1 Sample q 差：
- KR validation-selected Adam SB−SN **−0.005556**，3/4 负，seed 区间 [−0.012286,+0.005721]；不能写成稳健普遍负效应。
- 无动量 SGD SB−SN **−0.002104**，4/4 负，区间 [−0.002478,−0.001680]（按此 v2 协议）；说明 **“Adam 是采样损伤的通用必要条件”已被反例证伪**。Adam 与 SGD 的 q-gap 差区间跨 0，不能据两均值判定 Adam 额外中介必要。
- SGD calibration .1 是所测 [.1,1,10] 网格下界，不是全局调参充分；P3 参数距离、P4 域任务异质性、P5 selection、P6 投影几何只支持对应描述范围。v2 最终执行/结果/问题注册仍标 RUNNING，P2/P8 不宣称已最终完成。此中期结果与后续采用其他 LR / seed 的 v3、N17 不构成一组同估计对象的直接矛盾。

### 2.5 10-16 v3（17日完成）：条件效应成立、唯一长期中介未成立

完成 v3 规定及触发实验与冻结 test 法证；总分类 **BOUNDARY-CONFIRMED**，方法 **NO_GO**。同 Sample J / 同 routing-detach 计算图，同目标 q 变动可改变实际有限 batch 协方差和轨迹，冻结全坐标固定θ协方差比约2.3155–2.7522；该平均梯度不变/协方差改变原理不是创新本身。

四训练 seed 的**不同配置、不同评价估计量**，不得混成一种“Sample 比 Equal 好/坏”：
| 家族 | 冻结 test selected 宏 AUC SB−SN | 主要限制 |
| --- | ---: | --- |
| fixed Adam .01 | −0.007069（2/4 负） | 区间跨0、依赖选点 |
| fixed SGD .1 | −0.002518（4/4 负） | 不是所有无动量 SGD 配置 |
| calibrated Adam .003 | −0.008698（4/4 负） | 配置/LR 作用 |
| calibrated SGD .3 | −0.000972（4/4 负） | 同共同早期37轮却 +0.001614 四正；阶段反号 |
| 第二批 SGD .3 | −0.000253（3/4 负） | 平均接近0，不是严格等效 |
| MovieLens SGD .3 | +0.000539（1/4 正） | 不跨环境普遍复现 |
| AdaTask .003（组件） | −0.018917（4/4 负） | 不代表完整 AdaTask 最优方案失败 |

**状态与机制边界：** V11 的原生/影子 m/v/history 与破坏性消融存在健康 floor（beta2=0 最终 AUC≈.5）；同θ局部状态映射作用不证明长期唯一必要性；校准 Adam 与 SGD 有有条件交互，SGD 仍给 Adam 通用必要性的反例。V7 在早期和中期的共享 C 局部 AUC 筛查 4/4 达到衰减条件，但 loss/J 联合门未通过，**长期 C 私有化训练未触发**；不能说“C 全阴性”，也不能说“共享 C 已识别为必要中介”。真实共享 C/E、私有 R gates/towers、混合 P 的参数支持已核对。

**曝光/时间：** 同 multiset 的 ORDER 在固定50轮可呈有条件正向 q 交互，但最终预算和绝对 SN 健康反向；ORDER 还移动短尾 batch，不能叫纯课程收益。重复 R=E−U、coverage 与曝光在许多配额下代数耦合；N/2x alias 复制不增加独立新样本，不能由此识别自然域数据量 N 的因果作用。谱 shape vs scalar trace 缺同 trace 的独立干预，继续 **NOT_IDENTIFIABLE** 而非无限复做固定点谱图。

**适用边界：** MovieLens 早期共同16轮 q 差正，自选又混合；V8 strong mask 的最终 d2 q gap 表面变好但两臂绝对模型均损伤；少域/大域不能定义难度。D3、SAF 往往改变 J，不与本研究固定 J 的 q 操作同构；AdaTask 是相关状态机制先例；重要性采样、方差、shuffle-memory 与选点风险有前人工作，故目前没有已证实的独特 MDMT 必要机制或“新方法四seed主终点稳定改善且全 cell 无害”。

### 2.6 10-17—10-18 N17：独立 seed、统计主门和新的反例（目前最后一轮）

N17 **120 条新的 H100 训练轨迹（48 tuning +72 formal）全部完成**，其中 72 条正式轨迹×100 个 epoch＝7200 个已冻结模型/预测读取；N2 的16有效固定点、N3独立局部确认另行计数，**不能把沿同轨迹重复选点当独立训练**。Test 虽统一冻结读数，但曾在历史开发使用，**非 pristine holdout**。若引用本节主结果，以8个独立配对训练seed及预定 Holm 家族判定；二级阶段/cell区间只能有条件解读。

| 对比：SB−SN（固定test，selection选点） | 宏AUC差 | 8 seed t95 | 解释 |
| --- | ---: | --- | --- |
| KR Adam/common LR | **−0.005203** | [−0.012006,+0.001600]，7/8负，Holm p=.171875 | **主统计未确认**；bootstrap 区间负不能事后替换预定 t/符号检验 |
| KR SGD/common LR | **+0.000837** | [−0.004309,+0.005984]，5/8正，Holm p=.6953125 | 既不能确认为正，也不能证明 ±.003 实用等效 |
| KR Adam/fixed quota | −0.009191 | [−0.015795,−0.002588]，6/8负 | **次级 pointwise**；quota×q交互 −0.003988，区间跨0 |
| ML Adam/common LR、固定 subset | **−0.010392** | [−0.014599,−0.006185]，8/8负 | 固定 MovieLens 人口/配置的次级条件现象 |
| ML Adam/each-arm own LR、固定 subset | **−0.010875** | [−0.014447,−0.007303]，8/8负 | 与同组 **E100 +0.021145**、8/8正同时成立；SN .001 与 SB .0003 跨LR，不能叫纯 q 长期收益 |

**新机制缩窄而非全面识别：**
1. KR Adam 早期 AULC1..20 −0.007480，8/8负，t95 [−0.010031,−0.004930]；晚期同组 AULC1..100 −0.005324。故效应并非全由过拟合之后或选点伪象造成，但也非所有阶段、所有优化器都负。
2. **N2 固定θ四臂计数/方差分解**：16 锚的 SB/SN trace 比 IID 2.009–2.584、quota 1.995–2.551；域计数波动仅占所算 trace 约0.001297%–0.105936%。**“固定点协方差差异主要是域计数随机波动”在此可否定**；域内梯度结构仍贡献，但 quota 同时变批内依赖/高阶矩，长期交互未定，不证明纯 trace 中介。
3. **N3 共享路径独立四seed确认**：预先选择 embedding/E10，两个 full 局部路径 4/4正但效应仅约 +0.00003，方向门和实际阈值失败。长期 N4 topology 实验按预冻结条件 **NOT_TRIGGERED**；不等于“共享参数没作用”。
4. **评价权重可以翻转同一预测的数值符号**：历史 calAdam E50 macro −0.009074、训练 p 加权 cell AUC +0.001016、test-p 加权 +0.003108；新 ML own selected macro −0.010875，train-p +0.000744、test-p −0.013586。皆为**各 cell AUC 加权平均而非 pooled AUC**。不同宏/部署目标不可交换比较。
5. **ML own E100 正向不能升级方法**：宏 q gap +0.021145（8/8正，点区间正），但 E100 的较小 LR 与更长进度/selection 不同；较小域两任务的非劣保护置信下界 −0.005968、−0.009874，未达到 −0.003 门。保护未通过表示证据不足，**并非断言均值有损**。所有预定方法门仍 NO_GO。
6. **审计更正**：N2 原2项 float32 身份检查失败经同 seed、同 checkpoint、原估计器不变的验证脚本修复才纳入16有效锚；原失败保留。N17 主要 Q1–Q46 历史字段不改写，新增 Q47–Q53；N5-C 固定 ML subset_seed=2022，再用2032–39独立优化seed消除了 v3“seed同时改变subset”的部分混杂，**未**操纵第二个subset。
7. **论文方法门**：无已执行候选同时满足宏收益与七项保护；可说 **GO-EFFECT（指定设置/阶段）**、**GO-MECHANISM（仅固定点域计数贡献与有限局部作用）**、**NO_GO-METHOD（本轮）**，不可以从后者推出所有优化器/动量/MDMT研究无意义。

## 3. 跨轮潜在冲突与最终裁决

| 历史说法/表面冲突 | 现在如何登记 |
| --- | --- |
| Stage2+ 曾写 0正式训练；Stage2D 说16/16 | **解决——阶段性执行状态覆盖**：同分支 16/16 成功，旧报告保留历史；不可将其计成“未实施”。 |
| 旧 S6-R2 KR test 非 target +.006757 四正；新 B1 +.000579 两正 | **真实差异，原因未识别**：初始化、完整 lineage、selected checkpoint 身份不同；不是同一模型的相互否定。 |
| Stage2D 强负向、v2 Adam CI跨0、N17 Adam主Holm未过、v3多条件不同方向 | **不同协议/估计对象 + 证据强度修订**：learned/fixed fusion、双/单 optimizer、LR、population、seed数、选择/时钟和确认统计都不同；保留局部负效应，不宣称“普遍稳定 q 损伤”。 |
| S5 Equal Adam-history 局部救援；v2/v3 无动量 SGD 也有 q 效应 | **命题不同**：SGD反例只否定 Adam 对“所有 q 相关损伤”的必要性，不否定 Equal 下历史 carrier 的局部条件效应；旧S5 m/v必要性仍未唯一识别。 |
| v3 C 局部 AUC 正、N17 N3绝对约3e-5/长期未触发 | **筛查对象/阈值不同**：早期 C pathway 有局部作用；联合 loss/J 或独立实际界不满足；共享长时必要性未证明，也未被反证。 |
| 标量 trace、谱 shape、Adam 均值更新哪个解释 q 效应？ | **进一步排除计数主导，但唯一中介仍开放**：固定θ均值/cov身份可证，参数分区trace可反号；N2计数份额小，不等于排除重复/coverage/依赖或识别 m/v/谱shape。 |
| ML own E100 正 +.021145、selected −.010875；calAdam 宏负/加权正 | **不是同指标矛盾**：时钟/选点/LR或评价权重定义不同；不以事后更有利的端点改主评价。 |
| “小域=困难域”“课程/动量一定有效”反复否定 | **非前提、非核心创新**：历史已排除从数据量直接推困难度；新轮只是记录适用范围，禁止以反复证伪该稻草人代替新科学问题。 |

## 4. 问题去向：旧 OQ001–OQ010 与新 Q1–Q53 统一追踪

**机器可读完整清单：[Experimental_Question_Disposition_20261018.json](Experimental_Question_Disposition_20261018.json)**。保留53条问题的历史状态、必要原结论、最后一次具体证据、未排除解释和最小后续判别；Q1–Q46 的源历史字段仍见 N17 branch 的 [QUESTION_REGISTRY_FINAL.json](https://github.com/Caelestis1903/My-Research-Project/blob/codex/execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/QUESTION_REGISTRY_FINAL.json)。主文件**不**把“某一轮已执行完所有触发实验”误认为“所有机制问题已解决”。

| 旧主报告问题 | 截至 10-18 的裁决 | 新证据或仍缺什么 |
| --- | --- | --- |
| OQ001 Sample 极端域比例仍否有少数域损伤 | **开放** | V8 mask/物理复制与自然独立新样本量不等价；不能以无新增极端自然比例终局解决 |
| OQ002 size 独立因果，还是身份/任务/曝光代理 | **未识别；拒绝以大小直接定义难度** | N2 只排除所测固定点 count-fluctuation 主导 trace，不识别真实新增 unique N 的性能因果 |
| OQ003 S5 Equal persistent history 的 m/v/共享/架构组成 | **未识别，历史身份不足** | v3/N17 的**新 Sample q** shadow state 不是原 S5 Equal 同θ/同 lineage |
| OQ004 Equal access/history 跨模型/target/数据集范围 | **开放** | 后续主要是固定 J 的 q，未完成 Equal access 的跨 backbone 因果复制 |
| OQ005 ML late DH overfit vs optimizer-policy | **边界缩窄，唯一机制未识别** | S8 原 best 不改善；v3/N17 提供阶段/绝对健康反例，未把 old DH 的每条路径单独归因 |
| OQ006 Domain×Task gradient 结构跨设置稳定 | **部分机制描述，普适性开放** | 完整定θ谱/任务方向及局部块支持，缺稳定 performance 中介 |
| OQ007 训练/测试人口或标签漂移 vs objective | **人口权重算术已解决，域内 shift 因果开放** | N0/N17 三权重同预测反号，无法解释域内 split 的因果来源 |
| OQ008 S5-ML 不同复现来自数据/阶段/结构哪一项 | **开放** | 跨环境改变配置束，无法从 ML vs KR 自然观测剥离单因素 |
| OQ009 S6-R2 val/test 反转及精确协议再现 | **依旧未解释；重建未稳定重现** | B1 新模型/lineage 不同，已识别 selection 部分但非充分原因 |
| OQ010 S6-R2 参数置换 collapse 的定位 | **开放** | 原 swap 不能定位；V7/N3 是新状态局部控制，不是原 intact DH 分区唯一中介 |

**Q1–Q53 新增/修订的状态分层（截至最新证据）：**

- **已解决事实/审计身份（但不许扩大外推）：** Q15 同图/同目标公式及 q 校正、Q26 D3/SAF 目标边界、Q29 实际参数支持、Q31 调参/确认seed独立、Q32 固定SGD旧读数口径更正、Q33 common/own LR估计对象、Q39 同轨迹前缀身份、Q43 ML固定subset 复制、Q45 本轮全H100端点完成；Q3 的“Adam对一切效应必需”原命题为 FALSIFIED，Q34 moment telemetry 的旧修复保留。**已解决的是精确定义的局部命题，不是整个研究线全关闭。**
- **在已测设置内观察到效应、但推广或独立解释受限：** Q1–Q2、Q7–Q8、Q10–Q11、Q12–Q14、Q16–Q18、Q23–Q25、Q28、Q30、Q35–Q38、Q40–Q42、Q44、Q46–Q50、Q52。要逐条看完整清单的原始状态和最新边界，而不是按“BOUNDARY”统一叫未完成。
- **当前方案不能唯一识别（应转为明确新辨别研究，而不是反复复核同批数据）：** Q4（m/v/history必要性）、Q5（重复/曝光/覆盖/顺序分解）、Q6/Q22（共享长期必要性）、Q9（域/任务独立因果）、Q19–Q21（重叠支持与谱shape vs trace）、以及 Q51（Adam即时均值更新，**INCONCLUSIVE**）。
- **证据精度明确不足：** Q53 KR SGD共同LR selected差的t95 [−.004309,+.005984] 同时跨±.003，既不能宣告为零、也不能宣告实用提升/损伤；不要把“增seed追显著”作为机械下一步，除非先固定效应界与统计精度目标。
- **新Q47–Q53 全数保留：** Q47评价权重/人口，Q48调LR和预算，Q49计数与quota，Q50共享C局部路径，Q51 Adam瞬时漂移，Q52 EMS/MIS先例，Q53 SGD精度。N17 对它们的状态分别是 **5个 BOUNDARY、2个 INCONCLUSIVE**；“本轮执行完成”不意味着理论问题解决。

问题不能因“数量限制”自动关闭；新增问题/未识别项不设数额上限。“未触发”必须和“已实验否定”分开。任何新的机制研究应写清将从哪篇 MDMT 方法的哪项假设、迁移边界或尚未解决的矛盾切入，而不是再以早已不主张的“小域必然困难”作为研究动机。

## 5. 当前能成立的最终结论及行动边界

1. **存在条件效应：** 同一 Sample 经验目标下，改变域采样 q（配以 p/q 校正）能改变批梯度高阶统计、优化路径和某些配置/阶段的 AUC/LL；统计与泛化范围取决于 LR、optimizer、fusion、dataset、物理人口、时钟及 validation selection，**没有普遍收益/损伤定律**。
2. **已证伪通用必要性，不等于确认单一替代中介：** 无动量 SGD 可以出现 q 差，故 Adam 不是普遍必要条件；但某些 Adam 状态响应存在，不能宣布 Adam 历史无作用或 S5 Equal 效应错误。固定点域计数波动不足以解释 trace 差；谱shape/域内噪声/重复/覆盖/长期 C 必要性尚未分离。
3. **不应以数据量代替困难度：** 三自然域的规模、输入、正例率、域任务分布、split、baseline 等高度共变，大小不是独立已识别原因。历史一开始就已拒绝“小域=困难域”；不应再把反复否定它算作新创新。
4. **评价是研究结论的一部分：** 报告必须绑定 macro vs train-p/test-p weighted cell AUC（均不是pooled AUC）、AUC vs LL、validation vs test、selected vs same-clock/E100/AULC、同q不同LR与独立seed；不能只看使单一方法显得有利的一组端点。
5. **方法化暂不成立：** 已测试静态 Balanced、历史 late rescue、局部 C/state 方案均不满足本项目的可用收益和全单元非劣/证据门。既有文献覆盖许多一般采样/优化机制；目前适合主张**条件边界与可证伪问题**，而不是新 MDMT 优化算法已获验证。**NO_GO 只针对本轮所测候选和当前证据**。
6. **最少必要的下一轮辨别**（不是本次合并自动新训）：优先预注册固定 ML 人口上的 **q×LR(.0003,.001) 完整2×2、独立 optimizer seed、相同外部进度和主要评价**，针对 ML own E100 +.021145 与 selected −.010875 这个最有研究价值的冲突；保留所有六cell和 LL保护，避免事后调主终点。是否进一步独立测 S5历史状态或共享C长期机制，必须以新的明确可区分反事实与可获得的原工件为前提，不做无止境的重复防御测试。

## 6. 阅读路径和计数口径

1. 先读本文件（跨轮最终口径），再读 [Global Summary](Experimental_Global_Summary.md)、[Stable Findings](Experimental_Stable_Findings.md)、[Evidence Ledger](Experimental_Evidence_Ledger.md)、[Contradiction Register](Experimental_Contradiction_Register.md)。
2. 逐问题查 [53条问题去向 JSON](Experimental_Question_Disposition_20261018.json) 与 [Open Questions](Experimental_Open_Questions.md)，识别“现象已证实 / 机制未识别 / 缺独立场景 / 科学NO / 条件未触发”。
3. 原始分支报告链接列于第1节；精确数字、置信区间、每seed和数据来源须回查各分支的 CSV/JSON/审计收据。**main 本次只汇总文字与提取后的问题清单，不复制所有原始数据、所有训练提交、状态 checkpoint 或试验算法。**
4. 新 EXP 编号作为**研究阶段索引**而非把每个派生分析都算新独立实验。B1、Stage2D、v2、v3、N17 之间的祖先共享、读取旧模型和派生统计不得重复加总；16/16 Stage2D、v2已核验子集、v3规定训练和 N17 的120条新H100分别按**自己协议**报告。
