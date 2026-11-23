# MDMTRec 实验索引

以此文件作为 AI 的研究上下文入口。用实验编号、干预对象、匹配对照和主终点检查设想是否重合。**仅当新设计与既有结论矛盾，或干预与既有实验重合且需要区分具体设计及适用边界时，读取 [02_AI_EXPERIMENT_DETAILS.md](02_AI_EXPERIMENT_DETAILS.md) 的对应章节。** 核对数值时访问 [03_EXPERIMENT_RESULTS.md](03_EXPERIMENT_RESULTS.md)。禁止 AI 获取或读取同目录的 `04_HUMAN_RESEARCH_REVIEW.md`，递归搜索、批量加载与上下文注入同样排除该路径。

## 设计判据

- **小域不是困难域。** 样本支持、标签基率、基线表现、流量权重和训练阶段分别记录。禁止以域规模代理困难度，或设计验证该判断的实验与假设分析。
- 写出“问题 → 可观测现象 → 作用对象 → 干预 → 最简单匹配替代 → 能改变决策的端点”。分别核对假设对象实际存在、干预是否击中该对象、简单对照能否达到同一目的。归因优先使用冻结的同数据、同 seed、同优化预算、同监督/尺度或同状态对照。
- 性能比较的训练 seed 固定为 2022–2026，五 seed 综合裁定。各模型按验证六格 Macro AUC 选点，严格并列取最早，冻结后用对应 Test；历史替代协议保留其原标记。正式基线仅限已发表模型。常用强方法门为对最强匹配发表基线均值 ΔMacro AUC ≥ +0.003，六格均值保护线 −0.001。+0.002 候选线仅适用于明确规定该线的方案。LogLoss 独立裁定；方向数用于描述。当前研究的 Test 权限与冻结状态以各次任务和收据为准。
- 证据单位为配对训练 seed。同轨迹 online/EMA、E10/E50、同模型不同权重聚合、重放或历史复用均分别标记；诊断和未触发阶段不作性能否决。Macro、Traffic、Pooled 与 per-cell，以及 Validation、Test、selected 和 common epoch，保持各自口径。
- 自研干预分别比较普通已发表模型、强内部控制与同组件控制。固定参数 EMA 与 Adam 一阶矩作用在不同对象上。保留已观测的收益、损害和例外，同时区分局部因果、长期反馈与跨人口复现。

## 当前可据以决策的结论

1. 自然人口下固定参数 EMA 是强内部参照。年龄匹配 online 仍落后；共享参数尤其 embedding 承担主要排序收益，近期与较早历史互补。现有结果支持共享参数平滑收益，MDMT 特有的自适应优势尚未形成。方向响应、分组常驻时间尺度、未来收益预测及其动作策略已有多轮对照。
2. Gaussian＋固定 EMA 的完整方案在 MovieLens 的统一发表基线上有收益；KuaiRand 的 D0 覆盖优于 D2/D02 的 Macro 排序，仍低于 HyperGate。Gaussian 的附加增量取决于固定 EMA 参照、数据和选点。补测 Gaussian 前先核对实际噪声键与支持人口。
3. 局部梯度方向或一步排名动作真实，可以与持续训练失败并存。同状态投影、真实 Adam 锚点、固定源和伪源已比较。优先要求观测问题在正常强基线上存在，并给出可由匹配替代区分的终点。
4. 异质性增强原目标的大部分排序代价在零扰动时即可复现。修复自然权重后也未获得稳定额外排序收益。完成权重带来的 KR 概率损失改善与节点保护失败应同时保留。
5. A21 的条件历史在 KR 相对同 PAIR Adam＋EMA 的 Test 排序有正增量，当前人口下仍落后最强发表模型并触发单格保护；ML 对同 PAIR 与 clean＋EMA 均无额外排序。选择性共享线路的来源增量也需要强目标参考与同容量本地方案。

## 实验与结论

同一行保留可分别检索的实验编号，编号代表研究或审计条目。历史实验的协议与对应证据位于链接章节。

| 实验 | 实际干预或比较 | 结论与设计约束 |
| --- | --- | --- |
| [EXP001–004](02_AI_EXPERIMENT_DETAILS.md#exp001-019) | P1 梯度与选点；P2 固定融合/优化器修复；P5 两数据域比分层与单域 oracle | 早期差值受人口、总 N、实现与 checkpoint 影响；比率扫描不单调，oracle 差不能直接归因于共享。 |
| [EXP005–009](02_AI_EXPERIMENT_DETAILS.md#exp001-019) | 固定 N 轮换域比、跨 M3oE/MMoE、训练权重×选点、Equal/Sample×comp1 | objective 与选点会改变结果；固定 N 证据限制旧域角色叙事；comp1 独立增量弱于早期结论。 |
| [EXP010–017](02_AI_EXPERIMENT_DETAILS.md#exp001-019) | curriculum 前后向、Equal/Sample 与 α 扫描、KR 切分审计、动态校准与 EMA | 早晚观测未转成稳定 schedule；E5/E6 STOP，E2/E7 未运行。切分后人口和基率另行检查。 |
| [EXP018–019](02_AI_EXPERIMENT_DETAILS.md#exp001-019) | comp1/课程/梯度 EMA 消融，骨干与 D/T 扩展 | 历史正向有设置边界；后续 S0 的匹配普通 EMA 控制削弱域条件动量的独特性；STAR 复现不纳入。 |
| [EXP020–022](02_AI_EXPERIMENT_DETAILS.md#exp020-032) | S0 匹配动量；S1/S2 joint vs 单域 reference，Equal/Sample | 已测 Sample 设置未显示普遍的 practical residual transfer loss；KR Equal 下 target 与 non-target 权衡明显，自然 ML 不复现同一结构。 |
| [EXP023–028](02_AI_EXPERIMENT_DETAILS.md#exp020-032) | 共享访问、raw norm 与 step norm、Adam history/current 交叉，等权与样本加权 | KR Equal/tab1 的 target rescue 跟随持久 history，两个简单范数匹配未解释；ML 仅部分复现，Sample 不支持一般化规则。 |
| [EXP029–032](02_AI_EXPERIMENT_DETAILS.md#exp020-032) | S7 梯度探针/同 θ 短续训、S8 全状态续训、S6-R2 重选/参数交换 | 局部梯度 utility 和末轮减损未提升正常 selected；旧 KR DH 的 Test 阳性在 Validation 缺失，跨分区交换未保住收益。 |
| [EXP033–037](02_AI_EXPERIMENT_DETAILS.md#exp033-042) | B1 精确口径重建；同目标不同 q 的固定 θ replay、正式 Adam/SGD 与 V0–V13 | B1 未复现旧 DH 阳性；同目标不同 q 会改变轨迹，无动量 SGD 也可产生；LR、预算与阶段影响方向。 |
| [EXP038–042](02_AI_EXPERIMENT_DETAILS.md#exp033-042) | 同预测评价权重、KR q×LR、quota/count、共享 C 虚拟更新、固定自然 ML | Macro 与人口加权可反向。KR N17 主门未过；域计数不是所测协方差的主要解释；C 局部信号未建立独立长期必要性。 |
| [EXP043–047](02_AI_EXPERIMENT_DETAILS.md#exp043-051) | N18/N19 六方向、CM19、人工→自然 ML、原生 M3oE+D/G | 自然与人工人口分开；共享乘性路径数值中性；D 套件总收益真实，动态样本区分的独立增量欠缺；末轮减损未过终点。 |
| [EXP048–051](02_AI_EXPERIMENT_DETAILS.md#exp043-051) | I1 来源读写权限、MDMT-NEXT、RHD/LPS、DFD 后继函数 | 来源可读、需求可预测、形成后继函数分属三种证据；更强目标对照削弱来源必要性，未导出或未启动的环节保留状态。 |
| [EXP052–054](02_AI_EXPERIMENT_DETAILS.md#exp052-060) | Relation-Basis、BASP-A、HyperGate/AutoIFS 旧 LL 选点复现 | 来源低秩/多原型未稳定超目标直接适配；BASP 两 seed 未触发后续；旧基线排序受人口/选点影响。 |
| [EXP055–058](02_AI_EXPERIMENT_DETAILS.md#exp052-060) | GA31 缩放×Gaussian、L2、IID/反向配对、SAD βδ、AIG 动态门 | ML 损失缩放可有效；普通 L2 是强对照；方差降低未变成排序优势；固定 EMA 增益可复现，复杂差分/门未晋级。 |
| [EXP059–060](02_AI_EXPERIMENT_DETAILS.md#exp052-060) | GATS 难度覆盖/任务投影/时序；联合标签、教师和任务梯度视图 | KR B2 开发正、统一 Test 五 seed 接近匹配 A1；EMA02V2 分方向存在，收益未晋级。 |
| [EXP061–062](02_AI_EXPERIMENT_DETAILS.md#exp061-064) | A0 未来动作诊断；PUB 原尺度 P0/P1/P2 与同 α 控制 | A0 不提供可执行增量且后续未触发；PUB 当前迁移未胜普通/固定 EMA，同 α 退化对照不足以归因。 |
| [EXP063–064](02_AI_EXPERIMENT_DETAILS.md#exp061-064) | EMA 参数组/历史窗口/函数干预；A 反馈、B 两态、C 条件动作与 KD | 共享 S/embedding 与历史互补已定位；预测组合只覆盖部分收益。A 反馈失败；B 在 ML 主要改善 LL、KR 小幅诊断 AUC，最终未胜最强发表基线；C/KD 未建立额外收益。 |
| [EXP065](02_AI_EXPERIMENT_DETAILS.md#exp065) | PAIR/ECCM 两 seed 2×2 与 P01/P10 补充 | KR P01 相对 P00 Test +.012099、Validation −.000101，交互 Test −.012347；保持开发证据级别。 |
| [EXP066](02_AI_EXPERIMENT_DETAILS.md#exp066) | C_MIX/H_MIX/M、任务视图、真实历史与尺度对照 | KR C_MIX 单 seed Test +.009531，Validation d0t0 破保护；KR TM 成本阻塞。五 seed 延伸见 EXP067。 |
| [EXP067](02_AI_EXPERIMENT_DETAILS.md#exp067) | C/H/M/P01 补齐五 seed，拆总收益与相对 EMA 增量 | C/H/P01 多组总收益成立；KR P01 对 EMA +.004086，d1 两格破保护。 |
| [EXP068](02_AI_EXPERIMENT_DETAILS.md#exp068) | 正式 M3oE/HyperGate/AutoIFS 统一 AUC 选点；C_MIX＋P01 | ML M3oE .775190、KR HyperGate .682113 为各自最强发表模型；CP 不优于单 P01。 |
| [EXP069](02_AI_EXPERIMENT_DETAILS.md#exp069) | Gaussian D0/D2/D02，同协议五 seed | KR Macro AUC 优先 D0、LL 优先 D2，D02 降 AUC；ML D0 Test .779357，较 M3oE +.004167。 |
| [EXP070](02_AI_EXPERIMENT_DETAILS.md#exp070) | EMA 方向响应、路径分组时间尺度、漂移协方差增强 | 三类未稳定超过固定 EMA；保留局部交换与数据集条件。 |
| [EXP071](02_AI_EXPERIMENT_DETAILS.md#exp071) | EMA 年龄匹配、排序稳定性、常驻时间尺度、收益预测和 Gaussian 交互 | 超年龄匹配成立；更稳定或可预测与更高 AUC 是不同终点；常驻时间交叉仅在 KR Gaussian 条件成立。 |
| [EXP073](02_AI_EXPERIMENT_DETAILS.md#exp073) | 保持辅助质量的零扰动；自然质量自身任务视图 NDP-A | 零扰动复现主要退化；NDP-A 修正后 ML/KR selected EMA ΔAUC −.000101/−.005775；继续比较目标组成与阶段，勿重做零扰动。 |
| [EXP074](02_AI_EXPERIMENT_DETAILS.md#exp074) | TV-GRM 残差历史、shuffle、instant、γ 跨阶段 | 未形成续跑信号；ML γ=.20 保留弱正均值及一致 LL 方向；KR 优于 NDP-A 的局部改进未超过 clean。 |
| [EXP075](02_AI_EXPERIMENT_DETAILS.md#exp075) | AXIS-EVEN、IID2、匹配 isotropic 视图 | 额外轴向结构未改善 selected；KR IID2 有弱反例，均值机制与噪声路径仍可区分。 |
| [EXP076–077](02_AI_EXPERIMENT_DETAILS.md#exp076-077) | NQR/Brier/mean-gap；相同剂量 EARLY10/LATE10 | NQR 两数据 selected EMA AUC 非正；ML 早期正但选点负，关闭后 LL 改善未恢复排序。 |
| [EXP078–079](02_AI_EXPERIMENT_DETAILS.md#exp078-079) | 完成权重 W/总质量 S；移除格内类别异质性 U | W 在 KR 保留 LL 改善且 d2 AUC 正，d0t0 破保护；U 两数据 AUC 均值非正，格质量单独未产生稳定排序增量。 |
| [EXP080](02_AI_EXPERIMENT_DETAILS.md#exp080) | A20/A21 条件/简单历史、匹配剂量和定位；B 真实/伪源脉冲 | A21 Test 对匹配 PAIR ML −.000083、KR +.004062，两数据未达最终资格；B 未获持续信号。Gaussian 同身份失败，技术暂停。 |
