# B1 重建测量：完整结果与科学判读

主 agent，2026-10-16。按用户最新指示全程由主 agent 执行；self-review 不等于独立 review。所有差值均为 DH−F，AUC 越高越好，logloss 越低越好。这里的训练与完整评估已结束，科学解释不由数值 PASS 自动产生。

## 证据身份及核验

本研究为 **RECONSTRUCTION_NOT_HISTORICAL_EXACT**：KR natural / M3OE / Sample、target d1、F/DH×2022–2025、50 epochs。原历史完整 lineage/checkpoint 缺失，四 seed 的当前初始化均不匹配历史 pin。因此不能将本研究与旧 S6-R2 当作同一执行身份，也不能从差异定位历史报告错误。

8/8 运行退出码为 0；执行源码 commit `fa8e8e62789fa972121e3a551a91987cb13cc3d2`。每 seed 两臂初始化、数据身份及全部 main/arch 批次指纹一致。选点仅使用 validation，冻结 R0–R4 后评估全部 50 个 test 端点。test 已在历史开发中反复使用，本研究是法证分析，不是独立 holdout。

[数值凭据](analysis/analysis_receipt.json)记录 400 份完整状态和 800 个预测文件的审计，独立重算 AUC/float64 BCE、核验实际 row IDs/labels/domains、优化器参数覆盖及 step、50 epoch 覆盖和选点/分解恒等式。范围为 `B1_RECONSTRUCTION_FORENSIC`，状态为 `PASS_NUMERICAL_AUDIT_ONLY`。[解读数据凭据](interpretation/readout_receipt.json)验证原分析的 16 个输出 SHA；新表只作描述性汇总，不新增训练，不改变选点规则。

第一次 seed2022 F 构造期 native abort 及日志仍单独保留，根因为 **UNRESOLVED_NOT_REPRODUCED**；成功重试不证明该问题已根治。原训练定义、主/架构优化器超参数及 private FULL 未因结果修改。

## 主要比较：R0 validation macro 选点

下表为四个配对 seed 的均值与样本标准差。M=六 cell 宏平均，T=d1 两任务，N=d0/d2 四 cell；它们不是 pooled AUC。

| split / 指标 | ΔM 均值±SD | ΔN 均值±SD | ΔT 均值±SD |
|---|---:|---:|---:|
| validation AUC | +0.000637±0.004224 | +0.001134±0.006140 | −0.000356±0.001093 |
| test AUC | +0.000363±0.003997 | +0.000579±0.005486 | −0.000069±0.001159 |
| validation logloss | −0.000581±0.005595 | −0.002630±0.004735 | +0.003516±0.007319 |
| test logloss | +0.001053±0.002420 | −0.000424±0.000866 | +0.004007±0.007150 |

R0 test M/N 均仅 2/4 正向；M≥+0.003 和 N≥+0.003 均仅 1/4。T test AUC 仅 1/4 正向，没有 seed 达到 +0.003。test 宏 logloss 仅 1/4 改善。**MEDIUM：主要协议没有支持稳定实用的 selected 增益**；这不是严格的等效性证明，四 seed 和条件性区间不能排除小效应。

| seed | F/DH 选中 epoch（从0计） | val ΔM | test ΔM | test ΔN | test ΔT |
|---|---|---:|---:|---:|---:|
| 2022 | 32 / 32 | +0.004199 | −0.001639 | −0.002028 | −0.000862 |
| 2023 | 41 / 49 | +0.004107 | +0.001701 | +0.002717 | −0.000333 |
| 2024 | 41 / 48 | −0.001452 | −0.003882 | −0.005465 | −0.000718 |
| 2025 | 49 / 34 | −0.004305 | +0.005272 | +0.007090 | +0.001637 |

2022 与2025的 val/test 宏差反号，2023/2024同号。旧历史 R0 test ΔN≈+0.006757、4/4正，val≈−0.003851、1/4正；本研究没有重现该稳定组合。**STRONG（算术及身份事实）**：两组观测不同；**WEAK（历史反转原因）**：初始化和历史 lineage 差异阻止将差别单独归因于 dense saving、timing 或旧实现。

## 各域各任务、baseline 和不确定性

| test cell | F AUC | DH AUC | ΔAUC | Δlogloss |
|---|---:|---:|---:|---:|
| d0t0 | 0.627278 | 0.632260 | +0.004982 | −0.002958 |
| d0t1 | 0.655482 | 0.652703 | −0.002779 | −0.001726 |
| d1t0 | 0.698790 | 0.698471 | −0.000318 | +0.004199 |
| d1t1 | 0.726464 | 0.726644 | +0.000180 | +0.003815 |
| d2t0 | 0.654703 | 0.653147 | −0.001556 | +0.001658 |
| d2t1 | 0.673419 | 0.675086 | +0.001667 | +0.001328 |

validation 的 d2t0/t1 平均 AUC 改善 +0.008079/+0.010622（各3/4正），但 test 为 −0.001556/+0.001667（各2/4正）。validation d0t1 平均下降 −0.011932，test 为 −0.002779。test task0/task1 宏 ΔAUC 为 +0.001036/−0.000311，均只有2/4正。R0 worst-cell test均值改善+0.004982但seed范围−0.008880至+0.011527；不能由均值宣布最差cell安全。所有细项及F/DH绝对loss见[完整表](interpretation/r0_cell_means.csv)和[逐seed比较](interpretation/r0_selected_comparisons.csv)。

[配对bootstrap](analysis/selected_paired_bootstrap.csv)固定已拟合模型，按cell/标签分层抽相同行权重，1000次；四seed模型共用抽样权重，未将同一测试行当成四份独立数据。四模型平均效应的六个 test AUC 边际95%区间均含0，例如d0t0 [−0.015978,+0.023839]、d0t1 [−0.026242,+0.022244]。d0t1只有120个正例。validation d2t1区间[+0.001096,+0.020843]不含0，但这是未校正的多cell比较，且不推广为test稳定性。

d1 test logloss两任务条件性区间分别[+0.003737,+0.004680]、[+0.003418,+0.004223]；对应seed SD较大，且只有3/4 seed变差。因此 **MEDIUM：给定这四个模型的平均校准/loss代价清楚，跨训练重复的普适代价证据较弱**。这些区间没有训练seed总体、用户/时间聚类或反复选点的不确定性，不能称完整泛化置信区间；没有拼造macro CI。

## 不能删去的反证：R1/R4及训练阶段

R1=target validation AUC，R2=non-target，R3=worst-domain，R4=validation行频率加权域AUC。R0仍为主规则；以下为原先冻结的法证敏感度，不能观察test后提升为新的主要协议。

| 规则 | test ΔM | test ΔN | test ΔT | ΔN正向seed |
|---|---:|---:|---:|---:|
| R0 | +0.000363 | +0.000579 | −0.000069 | 2/4 |
| R1 | +0.007856 | +0.011584 | +0.000399 | 4/4 |
| R2 | +0.000464 | +0.001086 | −0.000781 | 2/4 |
| R3 | +0.002152 | +0.004030 | −0.001602 | 3/4 |
| R4 | +0.007856 | +0.011584 | +0.000399 | 4/4 |

R1/R4在全部8个模型上选中相同epoch，不能算两次独立确认。其test ΔN≥+0.003为3/4，seed2024仅+0.000945。validation ΔN平均+0.004800，3/4正；不是统一的val负/test正反转。R1/R4 test非目标logloss四seed均改善（均值−0.003499），但target logloss平均+0.006006，宏logloss仅−0.000331且seed范围−0.004626至+0.005416。**MEDIUM：存在选点规则依赖的非目标收益线索；WEAK：足以构成普适或安全的方法改进。** R0的阴性不能抹去这条线索，R1的阳性也不能替换主规则。

[双锚点分解](analysis/selected_timing_decomposition.csv)保留selected delta=同epoch policy+沿某臂timing两种恒等式。2022 R0两臂同epoch，val/test反号仍存在，所以timing不能解释全部反号。2023 R0 test ΔN=+0.002717，在F锚点为+0.010432加timing−0.007715；2024 R1同epoch policy为负而selected略正，说明timing可改变符号。分解是轨迹算术，不是同θ因果分离。

[分阶段统计](interpretation/stage_summary.csv)先按每seed每10epoch平均，再报告四seed分布，未把epoch当成独立重复。test同epoch ΔN在30–39为+0.008031（3/4正），40–49为+0.007009（3/4正）；R0 selected仅+0.000579。最后epoch test ΔN在2024为−0.015648，其余三seed为+0.005137至+0.008113；不能以末轮平均证明稳定收益。target test AUC在0–9四seed阶段均正，10–19四seed均负，后期仅1/4正，存在阶段边界。

[训练loss](interpretation/selected_final_training_loss.csv)最后epoch F/DH分别为2022 0.487308/0.491390、2023 0.491050/0.490585、2024 0.490364/0.492737、2025 0.489358/0.491250。DH仅1/4更低。训练目标loss、按cell宏LL和部署行频率LL的加权不同，不直接互换。

## 校准、观察窗口、规模与难度

[校准差](interpretation/r0_calibration_differences.csv)中，test d1 Brier平均增加0.001838/0.001501，10bin ECE增加0.006390/0.006274，与logloss代价方向一致；d0的Brier/ECE则平均改善。ECE依赖分箱，不是校准机制的独立证明。

[row-order四窗口](interpretation/r0_window_differences.csv)中test d0t1第0窗的平均ΔAUC为−0.056879、四seed均负，第3窗平均+0.043776、seed范围−0.047492至+0.130755，表明汇总可掩盖窗口异质性。窗口只由全局行序构造，未证明真实日期；小窗口正例不足、seed相关性和条件关系变化均可能解释差异。不能声称已识别时间漂移因果。

KR训练规模d0=172,782、d1=1,743,898、d2=89,452；validation/test d1约98.2%，R1与R4重合也与加权规则被d1主导相容。R0 test域平均ΔAUC分别+0.001101/−0.000069/+0.000056，小域间两个任务各有正负；validation最小d2收益未稳定转移至test。**WEAK：域规模是主解释变量；不存在稳定、跨split与seed成立的“小域收益/大域下降”规律。** 某些规则有非目标收益/target LL代价，但AUC与LL必须分别陈述。d0的F test AUC较低且正例少，大小和baseline难度混杂；单骨干、单比例、固定target不能分离物理域身份、label关系与规模。不能把小域等同困难域，也不能断言M3OE必然在困难域失败。

## 机制、停机与最有区分力的下一项实验

1. **STRONG（当前工件事实）**：完整配对与指标数值可核验；主R0和敏感度R1/R4呈现不同selected结论，训练阶段与seed2024构成反证。
2. **MEDIUM（局部现象）**：选点/轨迹与校准代价是有数据支持的竞争解释；小cell及窗口不确定性限制推广。
3. **WEAK（具体机制）**：本设计同步改变C/A current来源，history保持FULL，未做common-θ m/v因子干预，不能证明动量有效、m单独致因或C/A谁足够；课程学习没有被实验。
4. 按计划§7 STOP主要canonical selected收益线的自动方法开发。保留R1/R4线索，但不因回顾性test阳性启动新方法、重选主规则或扩大训练。
5. 最有区分力的后续应为**新预注册、未开发的时间后置数据**上的选择规则×policy测量：提前指定R0与target规则、相同训练与曝光、完整状态，分别报告同epoch/各自selected、target LL及non-target AUC。这样可区分选择偏差/窗口偶然性与可迁移收益。若缺少独立后置数据，该检验不成立。此处是后续建议，不是本轮已授权/已执行的新训练。

原计划全部条件节点的最终判定见[条件范围判定](../final_scope_decisions.md)。本研究不宣布新方法有效或创新成立。
