# Experimental Global Summary

## 2026-11-03 新增最高优先级：正式基线只允许已发表论文模型；统一算法指标台账

> **研究者强制指令，后续一切设计、实验、代码与论文报告优先执行。** 本条与紧随其后的“五seed/AUC/LogLoss”新规则共同适用；覆盖下文历史报告中将本项目自研算法、无增强+固定EMA、当前最优候选等称为“基线”的惯例。历史文件中的原称谓保留为当时的研究术语，不表示今后获准作为正式论文baseline。

1. **正式基线资格为“已有正式发表论文的模型”。** 论文基线（baseline model）只能来自**在可核验会议论文集或期刊正式发表过论文的模型**，应登记模型名称、论文名称、venue、年份、DOI/正式出版页以及对应的实现来源。未核实正式发表状态的模型暂不得列入正式baseline表。例：M3oE（SIGIR 2024）、HyperGate（WWW 2025 Companion）、AutoIFS（WSDM 2026）均可作为已发表模型基线候选；其仓库适配版本、参数配置和与原论文的一致程度须明确报告，不可伪称原论文严格复现。
2. **本项目自行设计、改造、探索的所有算法、组件、组合与变体一律不是正式基线。** 包括但不限于Gaussian/MASK变体、SAD、ANTI2、AIG-M、PAIR/ECCM、M/H/C_MIX/T、各类动态权重/共享/原型算法以及本项目自己的A0+固定EMA训练/评估变体。它们只能命名为**候选方法、研究配置、内部参考、匹配控制或消融对照**；可以用于解释组件作用、增量及组合交互，但不能列入“正式发表模型baseline”名单，也不能因是当前最强内部配置而升级为正式基线。固定EMA若是对已发表架构额外加入的仓库内方法，正式报告应明确区分“论文模型原实现/适配”与“内部EMA后处理版本”。
3. **比较协议必须对应。** 与已发表模型进行正式论文性能对比时，须使用同一数据人口/划分、评价口径、合法checkpoint选择和可比较的训练协议，并明确各模型自身有据可查的训练设置；不可将不同分支协议的绝对AUC/LL跨表排序。对自研组件做内部消融时允许相互比较，但标题、列名和结论必须标为“内部消融/控制”，不能以这种比较冒充已发表模型的baseline实验。最新的五seed配对平均AUC≥0.003、逐域任务格平均跌幅不超过0.001、五seed平均LL下降的规则同样用于相应可比的正式评价。
4. **主目录性能总台账为长期维护入口：** [MDMT_ALGORITHM_METRICS_LEDGER.md](../MDMT_ALGORITHM_METRICS_LEDGER.md)（对于 experiments/ 目录文件请使用相对链接 `../MDMT_ALGORITHM_METRICS_LEDGER.md`）。新增算法实验结束后，必须回填**每个域×任务格的绝对AUC与LogLoss、总体Macro AUC与Macro LogLoss，以及有据可查的traffic/pooled指标**；同时记录数据集、seed数量、模型/算法配置、在线/EMA状态、selected/common epoch、协议身份、来源报告和原始数表。不同协议分组存放；不得猜测缺失数字或将少于5seed的结果写为已充分否决。台账中的自研方法数据仅为**历史观测记录和内部方法分析**，不赋予其正式baseline资格。
5. **执行落地。** 后续Pro方案与Codex自动执行队列应直接继承本条及前述新五seed规则；在编写基线表、自动决策、生成论文比较表时强制检查“论文正式发表”标签。若历史任务中的旧停机门、旧基线用语或单格/seed否决门与本条冲突，以当前强制规则为准。原报告不得追溯性修改，新增研究任务也不得以旧报告为由重启已经充分确认的研究事实。

---


## 2026-11-03 当前强制实验评价规则：五seed、AUC收益与域任务保护（优先于下方历史阶段门）

> **研究者最新明确指令；自2026-11-03起对后续所有新算法/新实验生效。** 本节优先于下文旧分支的单seed/双seed D0/D1 停止门、要求4/5或5/5 seed 同向的门、旧版域任务下降0.002保护线及“未触发后续阶段=算法性能失败”等措辞。以下规则用于**新的研究规划、Codex运行配置、审查、分析和算法取舍**。既有实验的原始报告、当时按冻结协议得到的 STOP/NO_GAIN 状态及数值**不回写、不篡改**；在新规则下应重新区分“旧协议停止”与“已完成五seed的性能否决”。

1. **任何算法不得以单一seed作为最小性能测试或否决依据。** 判断一个算法/固定候选配置在指定数据和训练协议下是否合格、是否因性能不足而停止，**必须至少完成5个不同随机seed的完整有效训练**，并与相同seed、可比训练/选点协议的基线配对。1–4 seed 的验证/测试结果只能记录为**初步信号或尚未充分评价**，不能据此宣告算法无效，也不能以其平均AUC不佳、正向seed不足、某一个cell退化、95%区间跨0等理由自动阻断五seed正式性能评价。允许少量CPU/单步/单seed工程smoke检查正确性，但它们不是科学性能淘汰门；明确的实现错误、不可运行、数值故障或资源阻塞只能分别记**工程阻塞/成本阻塞**，不能伪装成五seed阴性。不同epoch、六格、online/EMA和同一轨迹复用不计为新增seed。有限调参可按事先约定进行，但**不能用少于5seed的预筛选否决整个算法方向**；最终受评配置必须满足五seed。
2. **AUC按五seed配对平均收益判断，不强求每个seed为正。** 预先指定主AUC口径，当前默认**validation选点后的test六格等权macro AUC**；记 \(\Delta AUC_s=AUC_{method,s}-AUC_{baseline,s}\)，五seed均值 \(\overline{\Delta AUC}=\frac15\sum_{s=1}^{5}\Delta AUC_s\)。**\(\overline{\Delta AUC}\ge +0.003\) 即达到总体AUC收益标准（含恰好0.003）**，不另设4/5、5/5同向、显著性或置信区间必须完全为正的硬门槛。同时执行明确的**域任务保护约束**：六个域×任务cell各自的五seed配对平均 \(\overline{\Delta AUC}_{d,t}\ge-0.001\)，即**任一单格平均AUC最多下降0.001**；无需每个seed的该格均非负。两项都满足才标记“**AUC合格且满足逐格约束**”；宏平均达标但某格低于−0.001，则标记“**宏AUC达标、逐格约束未满足**”，保留实际收益和具体代价，不抹去观察、更不据此否定整个研究方向。validation、traffic、pooled、各cell、selected/common epoch及seed方向数仍必须报告，作为解释与后续改进依据，不擅自加额外一票否决门。
3. **LogLoss同样按五seed综合表现，不要求每个seed都下降。** 统一使用真实评价人口、同一选点协议下未经训练缩放的LogLoss，记 \(\overline{\Delta LL}=\frac15\sum_s(LL_{method,s}-LL_{baseline,s})\)。**\(\overline{\Delta LL}<0\) 即可认定“综合LogLoss下降”**，不以单seed恶化、只有部分seed改善或区间跨0自动否决LL改善。AUC与LL是**分别报告、分别裁定**的目标：以AUC为主的算法不能仅因LL未下降而被取消AUC合格资格，但须完整披露LL代价；若明确宣称“同时改善AUC与LogLoss”，则两项各自满足上述标准。
4. **比较对象、组合收益与研究状态必须清楚。** 每轮预先固定主比较基线，并采用严格匹配seed/数据人口/模型/训练预算/选点的配对差；对采用固定参数EMA的算法，需分别报告“完整算法＋EMA vs 普通训练”的**总收益**及“完整算法＋EMA vs 无新增组件的固定EMA”的**组件增量**，不得混为一谈，也不得因组件增量不够大就将已有整体收益写成“没有收益”。某项收益的“合格”只对清楚标记的比较对象成立。旧报告中D0/D1因1–2 seed或旧保护门而未触发的后续阶段，今后应标注为**不足五seed、按新规则尚不能性能否决**，不是自动断言方法成功；是否优先补做由当前研究价值排序决定，不为防御性确认无限延长实验。
5. **执行约束。** 后续Pro设计、Codex计划及自动runner应把上述规则直接写入实验矩阵、阶段门和最终裁定代码；如任何旧任务模板与此冲突，以本节为准，不得沿用单seed淘汰或强制全部seed正向规则。不得仅修改报告文字而让实际派发逻辑继续在1–2 seed时关停科学候选。

---


## 2026-11-02 最新跨分支综合裁定（10-27—11-02；优先于全部历史快照）

**总状态：** main的系统性结论此前主要截至2026-10-26；此次补入9组不同已执行实验/基线（EXP052–EXP060），不把同源代码、重复训练、多个endpoint和未触发计划重复计数。报告和原始实验记录仍保留各自分支，本次仅将结论、证据与停止/开放状态纳入main。此前的历史实证与明确已接受的研究结论保留，不以新一轮防御性分析“重新归零”。

| 时间与实验 | 主要实证结果 | 当下研究决策 |
| --- | --- | --- |
| 10-27 Relation-Basis v2 | 两自然数据各4seed，48条正式适配；LOWRANK相对Q选择的TARGET_ONLY历史E宏AUC：ML−0.000623、KR−0.002727（均4/4负）；FULL也无强优势 | **停止当前来源关系基底版本**；保留局部来源读取的历史价值。[10-27 Relation-Basis](https://github.com/Caelestis1903/My-Research-Project/blob/03c338092008a71b2d360a61b06d985eac551ec7/research_reports/26-10-27_v2/FINAL_REPORT.md) |
| 10-28 BASP-A | 两自然数据各2seed、24条正式轨迹；CLUSTER_RELATION相对TARGET_MATCH的E宏AUC ML−0.000020、KR+0.000548（方向都1/2），留出μ预测不胜F常量、读取NULL权重约0.99+ | **当前原型关系/μ政策不晋级**，P2–P5未触发；不否定收益驱动原型概念。[10-28 BASP-A](https://github.com/Caelestis1903/My-Research-Project/blob/97151c74544c3651feb805d1898bc9ce1b8d12d3/research_reports/2026-10-28/FINAL_REPORT.md) |
| 10-29 HyperGate/AutoIFS基线 | 两数据×3模型×5seed共30正式训练。HyperGate相对M3OE_FIXED1宏AUC：ML+0.007379、KR+0.014412；AutoIFS：+0.003551/+0.007308，四组都5/5正。KR AutoIFS宏LL−0.004400但natural LL+0.006677（5/5变差） | **两基线已经正式运行可用**；HyperGate配置有UNVERIFIED_PROPOSAL，V选择、训练协议不同，不称原论文严格复现。[10-29基线](https://github.com/Caelestis1903/My-Research-Project/blob/d2fdd272a2cdf7da626e29b97f61555d25271f89/research_reports/mdmt_baselines_2526/FINAL_REPORT.md) |
| 10-31 GA31-v2 | 40条训练；ML无增强c=.1对c=1 selected AUC+0.003473（4/5），E10−0.004169、E50+0.026978；KR缩放selected−0.006026；NR-Gauss在ML两个尺度分别−0.000267/−0.000188 | 确认**loss scale影响轨迹/选点**，NR-Gauss本版无独立晋级收益；耦合wd与eps不能忽略。[10-31 GA31-v2](https://github.com/Caelestis1903/My-Research-Project/blob/ee6e81fdfd37284e6447e8149cb93d04736dba02/research_reports/2026-10-31/REPORT.md) |
| 11-01 GA31-R3 | 45条训练；普通L2控制可作为ML缩放的简单排序参照。KR ANTI2相对普通基线test AUC+0.001202（4/5），但ANTI2−同K IID2均值−0.001237，validation及最差格不支持全门 | **反向配对减条件方差是真的，独有方法增量未成立**；不继续堆叠SAGU/SAOS。[11-01 GA31-R3](https://github.com/Caelestis1903/My-Research-Project/blob/a6f496b54f467cdf7b814b6bb8defdbdd457bc38/research_reports/2026-10-31_v2/REPORT.md) |
| 11-01 参数EMA＋SAD | 两数据×4臂×5seed共40正式训练；ML无增强shadow−online selected AUC+0.004399（5/5），LL+0.007800（4/5恶化）；KR AUC+0.004616（4/5）。SAD βδ=.99相对普通增强Adam宏AUC ML−0.000336、KR−0.001554 | **固定EMA接受为排序有效的强简单参照**；当前差分长记忆版本停止，不将非零差分误当收益。[11-01动量/EMA](https://github.com/Caelestis1903/My-Research-Project/blob/ee4cf0512a7723624c9ee6a1ef192f0f7f17686b/research_reports/2026-11-01_momentum/REPORT.md) |
| 11-01 AIGM01 | 56条唯一完整训练；调参后ML EMA selected AUC+0.003849（5/5）、KR+0.004616（4/5）；KR Gaussian+EMA相对无增强EMA宏LL−0.001664（5/5）、宏AUC+0.001476（3/5），但最大域d1两任务AUC−0.004654（5/5）/−0.004929（4/5） | EMA稳定参照；**增强表现为明确域/任务与指标取舍**。固定SAD/AIG-M开发门没过，不虚构正式五seedAIG阴性。[11-01 AIGM01](https://github.com/Caelestis1903/My-Research-Project/blob/6f5742c2085c1e49d1767b0a7079bb9f28160eda/research_reports/2026-11-01_aig_m/REPORT.md) |
| 11-02 GATS02 | 31条新增科学轨迹50epoch；KR B2开发验证对A1宏AUC+0.008184（2/2），保留seed确认缩小，五seed统一test对A1宏AUC+0.000005（3/5）；对更强A0固定EMA+0.001481（仅2/5）。B1、B3、当前C2规则均未晋级 | **B2开发局部信号保留，但不是可晋级整体算法**；当前B1/B3/C2停止。[11-02 GATS02](https://github.com/Caelestis1903/My-Research-Project/blob/e59efa2fd954c4d2231b679a742406cfbe2a1077/research_reports/2026-11-02_gradient_aug_time_select/REPORT.md) |
| 11-02 EMA02V2 | 36/36正式轨迹，2开发seed，144/144端点；ML F1−A0+EMA test AUC+0.000359（2/2）、验证均值−0.000424；KR TG/AG−A0+EMA test宏AUC−0.017174/−0.017534，LL均约+0.0124~0.0125，两seed同向 | 当前J/F/TG/AG视图构造**不晋级**，F1只是微弱test线索，确认/W1未触发；保留无增强固定EMA。[11-02 EMA02V2](https://github.com/Caelestis1903/My-Research-Project/blob/38a53bc1aa6e0c16c0b6d2b0e24e07018f91ab27/research_reports/2026-11-02_ema_view_design_v2/REPORT.md) |

**跨轮收敛的判断。** EMA的selected AUC收益不再是待“发现”的问题；用EMA影子参数本身不是对Adam动量公式的改造。缩放产生AUC效应、普通L2是强竞争者、方差下降不蕴含排名改善、非零差分历史不蕴含实用收益、复杂视图未胜简单EMA、某一任务收益可能转移成其他任务损失，这些都有各自已执行的实证支持。10-21 D套件历史结果与新GA31不同目标/人口，不互相取代。新研究应重点考虑**域×任务特征/标签需求如何影响增强与梯度，以及这些信息怎样进入真实动量/参数时间递推**，而不是重复先前阴性配置或再给EMA取新名字。

**证据边界与下一行动：** 用户已明确认可的研究前提无需专门复验。10-29基线使用不同训练/选模口径，不能与GA31 AUC选点绝对值直接排名；GATS的开发、确认和test必须分开；ML EMA排序改善与LL代价并存；任何声称新总体收益的方法须至少胜当前强EMA和必要匹配控制。历史test/seed池多次用于研究判断，这不是独立新增用户/时间人口。未触发阶段只记NOT_TRIGGERED。后续如果有具体新增设计改变研究选择，再安排最小有区分力的实验；不为绝对确定性无限追加。

**统一索引：** [实验EXP052–EXP060](Experimental_Experiment_Index.md)、[证据E058–E069](Experimental_Evidence_Ledger.md)、[稳定事实SF029–SF037](Experimental_Stable_Findings.md)、[版本停线](Experimental_Dead_Ends.md)、[指标分歧](Experimental_Contradiction_Register.md)。

---

> **2026-10-26 总结更新：** 本文件的当前科研裁定以 [2026-10-26跨阶段总报告](Experimental_Branch_Synthesis_20261026.md) 为准；后文标为10-18/10-09的Q1–Q7、训练计数及“当前”陈述均保留为当时的历史快照，不能覆盖晚于它的证据。原研究阶段报告未修改，原始文件与固定SHA见 [history_experiments](../history_experiments/README.md)。

## 2026-10-26 当前研究结论（优先于下方历史快照）

1. **长期研究轴是样本重加权与差异化共享。** Equal/Sample是训练目标的局部口径；S5 Equal的特定Adam-history作用、v2/N17无动量SGD的采样效应并存：Adam不是全体效应的必要条件，也没有被Sample“完全消除”。研究仍要分开样本曝光、目标加权、动量状态和共享读写权限。
2. **先修正数据人口，再谈相互印证。** 10-15之后多轮MovieLens在人为固定子集187500行（34500/3000/150000）上训练；自然base-train为800167行（155949/318599/325619）。[自然人口审计](../history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/DATA_PROTOCOL_AUDIT.md)显示比例、真实支持与训练时钟一起变化。自然复核已经有部分可靠结果，但扩展分支整体仍为IN_PROGRESS，不能把N19/CM19人工结果宣称为全自然重复。
3. **N18/N19已经实质完成，不是只有“方案”或都没有尝试。** [N18](../history_experiments/codex__execute-2026-10-18/research_plans/2026-10-18/execution/FINAL_REPORT.md)进行了六方向有界诊断；[N19](../history_experiments/codex__execute-n19-2026-10-18/research_plans/2026-10-18/N19/FINAL_REPORT.md)完成328次全预算训练。方法独有性和稳定强对照收益未获支持，但局部反例、容量/通道边界和方向机会保留，不把NO_GO扩展为领域失败。
4. **共享路径的数值功能与实际泛化收益不同。** [CM19](../history_experiments/codex__conditional-paths-2026-10-19/research_plans/2026-10-19/execution/FINAL_REPORT.md)完成856次训练：KR fixed M3oE选点的输入乘性调制路径有明确数值中性，局部AdamW/耦合L2干预可以恢复响应；STAR等独立路径保持活跃，且恢复响应并未产生通用selected收益。
5. **10-21的D套件在自然MovieLens确有收益，但“动态样本区分导致收益”尚未成立。** [复现报告](../history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md)中D−B test宏AUC +0.006303、LL −0.015962（4/4），Dconst01−B AUC +0.005676（4/4）；常数权重仍带复制/噪声/总尺度，不能从这组对照确定哪个模块有效。匹配G−Gtau0调幅的宏AUC：ML −0.003466、KR −0.006484（均4/4负，但KR区间跨零）。其目标为legacy_group_sum，非其他阶段的Sample/Equal。
6. **动态权限/收益/形成新方法须通过强简单对照。** 10-24 I1_PERM在ML/KR selected未胜固定RW；10-25 v2核心48条完成但正式门未触发；10-25 v3 RHD相对BASE selected test LL +0.00074075、e50 −0.00875428，LPS特定形成未稳定优于简单读法；10-26 DFD-v4的ML需求预测有限有利，但完整DFD相对H_DIRECT/ETA_DIRECT LL更差、KR P1失败，P3/P4未触发，另缺原refitη外层μ分组。各轮**版本性结论**不等于来源知识价值或多原型研究全盘不成立。
7. **最新研究决策是RCF设计而非新实验阳性。** [2026-10-26-v2方案](../research_plans/2026-10-26-v2/00_README.md)保留目标剩余风险指导来源后继函数形成的科学问题，以来源保持、有限容量、固定读出及强目标直接头作判别；目前**DESIGN_ONLY，无新增RCF训练/评价结果**。
8. **评价主张必须绑定协议。** 宏cell AUC、人口加权cell AUC、pooled AUC、LL，selected、末轮、长期AULC、Q/validation/test与真正新封存数据互不等价。早期“小域=困难域”从来不是用户认同的前提，不再把重复推翻它算作研究成果。

**合并结论：** 现有证据支持目标/域/群体条件性收益与共享损害的研究价值，支持一些真实局部机制与反例；没有建立一个在当前完整强对照和不同自然支持下通用、安全、独有的新算法。更值得检验的是**强目标预测之外的来源功能增量、群体需求驱动形成，以及样本权重收益的真实贡献结构**。完整时间线、已完成/未触发/部分完成和历史结论更正见[跨阶段总报告](Experimental_Branch_Synthesis_20261026.md)。

---



> **2026-10-18 跨分支更新。** 下方原“整理截止 2026-10-09”以及 Q1–Q7/历史计数保持**历史快照**身份，不再代表最新所有实验的最终统计或执行状态。当前权威入口：[多分支综合报告](Experimental_Branch_Synthesis_20261018.md)；[53条问题逐项去向](Experimental_Question_Disposition_20261018.json)。证据编号 E033–E041、SF013–SF020 和新增 EXP 索引补录见相应文件。**未把代码/权重/原始数据从执行分支全量合并。**

## 2026-10-18 历史科研摘要（优先于当时10-09快照；已由10-26总报告更新）

1. **同 Sample 目标不等于同训练路径。** KR Stage2D（learned fusion、双 Adam、四seed）validation-selected SB−SN macro AUC = −0.010406、4/4负；v2（fixed fusion、单优化器）无动量SGD仍为 −0.002104、4/4负。两组不构成协议同构的直接复现，却共同说明 Adam **不是所有 q 相关效应的通用必要条件**；不能从此反驳旧 S5 Equal 下持久 history carrier 的窄结论。
2. **最新 N17 的统计强度低于简单符号多数投票。** 120条新H100轨迹全部完成；正式 KR selected test：Adam SB−SN = −0.005203，8seed t95[−0.012006,+0.001600]、Holm p=.171875；SGD = +0.000837，t95[−0.004309,+0.005984]、Holm p=.6953125。两项**预定主比较均未确认**，不能称普遍损伤或等效。KR Adam AULC1..20 −0.007480、8/8负是**阶段性次级证据**，不是取代主门。
3. **晚期、选点、LR与评价权重会真实改变方向。** 固定 ML subset 的 own-LR E100 q-gap +0.021145、8/8正，但 selected −0.010875、8/8负；两臂采用不同 LR，不能称纯 q 优势，且cell保护门未过。同一个历史 calAdam E50 预测：宏cell AUC差 −0.009074、train-p加权 +0.001016、test-p加权 +0.003108（均不是 pooled AUC）。
4. **机制从“可能”缩窄为“局部反证与仍未识别”。** N2固定点 quota 计数项只占目标梯度trace约0.001297%–0.105936%，不支持“域计数波动主导”的特定解释；quota长期效应仍存在但交互区间跨0。N3局部共享路径约 +3e−5的阳性达不到独立方向/实用门，N4长期拓扑实验**未触发**；独立 shape vs trace、m/v/history、重复/coverage与共享 C 长期中介仍未唯一识别。
5. **研究主线不再围绕反复否定“小域=困难域”。** 这原本就不是成立前提。数据量、身份、标签/特征、域任务与曝光共同变化；未做真实独立数据支持 N 的纯因果操纵。当前结论是**条件效应有证据、普适/唯一中介不足、已测试候选方法 NO_GO**；不预设动量/课程有效，不从未触发分支推断方法阴性。

**跨时间纠正：** 10-16 早期 Stage2+ “正式训练 0、后续阻塞”已经被同分支 Stage2D **16/16 COMPLETE** 的后续报告覆盖。旧 S6-R2 KR test non-target +0.006757（四正）与10-15 B1重建 +0.000579（两正）来自不同init/lineage，不是同协议相互否定。v2是中期研究（不能声称全部完成）；v3在 Git 历史上与 v2 发散；N17继承v3。三个方案分支的原计划正文已经在 main，不额外计数实验。

**问题状态：** 旧 OQ001–OQ010 均逐项再评估（一些计数/身份问题已排除，一些机制依然开放）；N17 保留 Q1–Q46历史对象并新增Q47–Q53，完整53条见链接。若将“已执行/未触发”直接等同于“已解决/已否定”，会误报；也不限制新增开放问题的数量。

---

整理截止：2026-10-09。本总结把已复核的实验事实与未验证的解释分开；证据强度和适用范围以 [Evidence Ledger](Experimental_Evidence_Ledger.md)、[Hypothesis History](Experimental_Hypothesis_History.md) 及 [Open Questions](Experimental_Open_Questions.md) 为准。

## Q1 — 目前最强的实验事实是什么？

1. **已测自然比例下，Sample joint training 没有显示稳定的实用负迁移。** S1 的 72 个已测配对单元中为 0 个 practical NT；S2 的 Sample 对照为 0/24。S6 预注册 selected-target gates 也未达到 practical rescue 阈值。这是对已测协议的结论，不覆盖未测的极端比例或目标域设置（E009、E010、E018、E019）。
2. **Equal/KuaiRand/tab1 下的 target 结果对 shared-access/history 干预有条件性响应。** S3 access 对照 4/4 seeds 同向；S5 显示该配置的 rescue 主要随完整 persistent Adam history source，而非单独 current source。此事实限于 M3OE、KuaiRand natural、Equal、d1/tab1 及相应 checkpoint；S5 的 moment gate 为 15/16，不能据此声称 m 或 v 单独致因（E011、E015、OQ003）。
3. **S6-R2 在 KuaiRand 观察到 test-side selected non-target gain，但 validation 方向相反。** R0-selected test ΔN 为 +0.0068（4/4 正）、ΔM 为 +0.0044（4/4 正），target ΔT 为 −0.0004；同一组 checkpoint 的 validation ΔN 是 −0.0039。R1/R3/R4 的同向结果只在 A-grid sensitivity 中成立，且 MovieLens negative control selected contrast 近零（E028–E030）。源报告在其 KuaiRand gate 上判为 GO；这不解决 validation/test 反转或增益机制。
4. **S8 的 MovieLens continuation 未提高冻结的原 F best。** Full-state selected-anchor 对照接近零，所有 MovieLens continuation best 均低于原 F best；late macro rescue 是 seed-skewed 的 overfit-anchor 现象，不能替代 S6-R2 的 KuaiRand test-side observation，也不能证明普遍 optimizer-policy gain（E026、E027）。

## Q2 — 哪些早期结论后来被修正？

- “小域梯度必然更噪、需要 per-domain momentum”没有得到稳定支持；comp1 的独特动量机制被 S0 的匹配 uniform-EMA 对照削弱，domain size 也没有被独立识别为主因（E001、E008、E009）。
- “Sample 消除了所有 optimizer/history effect”范围过宽。S6 Phase A 只支持 selected practical target gate 未通过；后续 Phase B 与 S8 仍观察到 MovieLens late trajectory effects，S6-R2 又在 KuaiRand 发现 test-side selected non-target gain，但 validation 方向相反。KuaiRand Phase B 的 M1 gate 失败，不能写成机制识别已全面通过（E018–E020、E028–E030、C014、C019）。
- `exp_sn.md` 是 2026-10-06 的历史快照：其“Phase B 未触发”后来已过时；该汇总遗漏了独立的 S5-ML，而 S6ML 与 S5ML 不是同一实验。S6 样本量复核显示 MovieLens d2=150,000、d0=34,500、d1=3,000，故 d1 最小、d0 居中（C015、C016、C018）。
- S7-B 报告曾将 MovieLens anchor39 step5 target AUC 写成 4/4 seeds 为负。原始 CSV 实为均值 −0.001226、3/4 为负、seed2025 为 +0.002359；LogLoss 均值 +0.011130、3/4 变差。已修正源报告与对应 Evidence/Stable Finding；“短窗内未见稳定改善”仍成立，但逐 seed 一致性证据降低（E025、C012）。

## Q3 — 哪些现象只在特定 dataset/objective/setting 下成立？

- Equal 下 KuaiRand d1/tab1 的 access/history rescue 是局部识别结果；不能直接外推到 Sample、其他域、其他 backbone 或 MovieLens。
- S5-ML 的 MovieLens Equal target/history pattern 只部分复现 KuaiRand S5；S6 Sample 下 selected gates 未达阈值，而 MovieLens late DH effects 出现在较晚训练阶段，受严重过拟合和 seed 影响。
- S7 的 Domain×Task/validation-gradient 观察只覆盖 MovieLens 与 KuaiRand 的 M3OE/Sample 设置。MovieLens late overfit 段的 U_k 与性能下降同步，KuaiRand 的 negative-U snapshot 没有同样后果；它不是跨数据集的充分性能标记。
- S6-R2 的 selected test-side gain 只见于所测 KuaiRand/Sample/M3OE F-vs-DH 对照；MovieLens selected effect 近零。KuaiRand 同一 selected checkpoint pair 的 validation/test 方向相反，且替代规则的 test 支持来自 A-grid sensitivity，并非精确重选 checkpoint。
- 旧 MovieLens ratio/role 结果描述的是特定数据生成、目标权重和训练协议，不能单独证明“大域必然受损”或“小域等于困难域”。

## Q4 — 哪些机制目前真正有证据？

- 在 S3 的 Equal/KuaiRand/tab1 配置中，shared-access 干预对 target 结果有因果影响；这证明的是干预效应，不是 gradient-conflict 根因。
- 在 S5 的 Equal/KuaiRand/d1/tab1 配置中，完整 persistent Adam history source 承载了大部分 target rescue；状态子成分 m、v、step/bias-correction 与参数路径尚未区分。S6 Sample Phase B 的 KuaiRand M1 未通过，限制跨 objective 机制主张。
- S4 已显示 raw-gradient norm matching 和 applied-step norm matching 单独不足以复制该 target rescue。
- S7-B 能改变部分 gradient utility 指标，但短程内没有稳定 validation 收益；S8 的 MovieLens endpoint 也没有 selected/best 性能收益。S6-R2 的 KuaiRand parameter swaps 在四 seeds 均退化，支持所测分区不可直接互换，但没有定位 DH gain 的来源（E032）。因此，gradient correction 或某个 optimizer component 造成 test-side selected gain 仍未被证明。

## Q5 — 哪些所谓机制其实仍然只是 hypothesis？

以下仍不是已建立机制：domain size 是退化主因；小域低 SNR 导致需要更长动量；Sample 下存在普遍 practical NT；negative U_k/gradient conflict 是充分性能原因；Equal history 机制跨 objective 保持不变；KuaiRand S6-R2 的 validation/test reversal 由某个单一机制造成；parameter swap collapse 已定位增益分区；MovieLens 与 KuaiRand 差异由某一个数据因素单独造成。KuaiRand test-side selected gain 是条件性 observation，不等同于普遍机制或部署收益。相关历史解释已在 Hypothesis History 中标为 WEAKENED、FALSIFIED、SUPERSEDED 或 OPEN，不能当作 Phase 3 前提。

## Q6 — 哪些问题仍然 unresolved？

- Sample 在更极端域比例下是否出现 selected minority-domain gap；domain size 是否有独立因果效应（OQ001–OQ002）。
- Equal/KuaiRand rescue 中 Adam history 的具体状态来源，以及 shared-access/history 效应能否跨 backbone、target 和数据集复现（OQ003–OQ004）。
- MovieLens late DH 现象中普通 overfitting 与 optimizer-policy 各自贡献；Domain×Task gradient pattern 的跨设置稳定范围（OQ005–OQ006）。
- KuaiRand 的 train/test mixture 或 label shift 如何与 objective effect 区分；S5-ML 部分复现由数据结构还是训练阶段调节（OQ007–OQ008）。
- KuaiRand S6-R2 的 selected validation/test reversal 与替代 selection rules 在精确 checkpoint 上的结果；跨分区 swaps collapse 后能否定位 intact DH test gain（OQ009–OQ010）。

## Q7 — 哪些旧研究线程不应在 Phase 3 中被重新当成有效前提？

不能预设小域更噪或更难、域大小已解释退化、M3oE 独有大域牺牲、comp1/域条件动量有独特优势、curriculum 或 adaptive calibration 已胜过 Sample、Sample 下存在普遍 practical NT、Equal 下 history 结果可推广到 Sample、target-only/HT 是安全部署方法、gradient conflict/U_k 修正会改善性能，或 DH 的 selected test gain 普遍存在/普遍不存在。S6-R2 的 KuaiRand 结果不能被当作某个机制或部署方法的既定前提；其 metric-layer reversal 和分区定位问题仍开放。

## Traceability Audit

- `reports/exp_sn.md` 已先行核对；其中 S0–S6 在索引中各登记一次，未重写重复阶段总结。S5-ML 独立补登；S6 DH trajectory analyses 重用同一批训练数据，标为非独立证据。S6-R2 在该快照之后完成，作为 evaluation-only EXP032 单独纳入；它复用既有 S6/S6ML checkpoints，不新增训练。
- 抽查 S5 聚合表、S6 Phase B audit、S6-R2 gate-summary / window / parameter-swap CSV 与 integrity report、S7-B validation 原始 CSV、S8 paired endpoints；核对了 dataset/objective/seed/checkpoint 口径和关键数值。S6-R2 复用了 16 个既有 F/DH checkpoints、没有新训练；Part D 的 manifest partitions / state-dict parity 检查通过，seed2025 retention 分母近零已限定说明。保留 S6 KuaiRand M1 失败与 MovieLens d0/d1 样本量标签更正。
- 报告中的观察值与机制解释分开登记；单数据集、单 objective、固定 checkpoint 的结果没有推广为普遍规律；S6/S7 派生分析没有计为独立复现。
- 本整理未做 literature matching，也未提出或设计新算法。

## Completion Counts and Phase 3 Readiness

| 项目 | 数量 / 判定 |
|---|---:|
| Experiment Index | 32 条：29 个既有实验组有正式运行/探测记录，2 个未运行/门控（E2、E7），另有 1 个 evaluation-only（S6-R2）；共 30 个条目有运行或评估结果 |
| Evidence Ledger | 32 条：STRONG 10、MEDIUM 19、WEAK 3 |
| Historical hypotheses | ACTIVE 2、PARTIALLY_SUPPORTED 3、WEAKENED 5、FALSIFIED 9、SUPERSEDED 1、OPEN 3 |
| Stable Findings | 12 |
| Open Questions | 10 |
| Dead Ends | 12 |
| Phase 3 条件 | **具备进入证据驱动的下一阶段梳理条件**；尚不构成任何特定算法或普遍机制已成立的依据。Phase 3 应以这些限定事实和未解决问题为输入。 |

