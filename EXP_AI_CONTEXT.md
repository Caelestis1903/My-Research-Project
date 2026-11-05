# EXP_AI_CONTEXT.md

## 2026-11-03 新增最高优先级：正式基线只允许已发表论文模型；统一算法指标台账

> **研究者强制指令，后续一切设计、实验、代码与论文报告优先执行。** 本条与紧随其后的“五seed/AUC/LogLoss”新规则共同适用；覆盖下文历史报告中将本项目自研算法、无增强+固定EMA、当前最优候选等称为“基线”的惯例。历史文件中的原称谓保留为当时的研究术语，不表示今后获准作为正式论文baseline。

1. **正式基线资格为“已有正式发表论文的模型”。** 论文基线（baseline model）只能来自**在可核验会议论文集或期刊正式发表过论文的模型**，应登记模型名称、论文名称、venue、年份、DOI/正式出版页以及对应的实现来源。未核实正式发表状态的模型暂不得列入正式baseline表。例：M3oE（SIGIR 2024）、HyperGate（WWW 2025 Companion）、AutoIFS（WSDM 2026）均可作为已发表模型基线候选；其仓库适配版本、参数配置和与原论文的一致程度须明确报告，不可伪称原论文严格复现。
2. **本项目自行设计、改造、探索的所有算法、组件、组合与变体一律不是正式基线。** 包括但不限于Gaussian/MASK变体、SAD、ANTI2、AIG-M、PAIR/ECCM、M/H/C_MIX/T、各类动态权重/共享/原型算法以及本项目自己的A0+固定EMA训练/评估变体。它们只能命名为**候选方法、研究配置、内部参考、匹配控制或消融对照**；可以用于解释组件作用、增量及组合交互，但不能列入“正式发表模型baseline”名单，也不能因是当前最强内部配置而升级为正式基线。固定EMA若是对已发表架构额外加入的仓库内方法，正式报告应明确区分“论文模型原实现/适配”与“内部EMA后处理版本”。
3. **比较协议必须对应。** 与已发表模型进行正式论文性能对比时，须使用同一数据人口/划分、评价口径、合法checkpoint选择和可比较的训练协议，并明确各模型自身有据可查的训练设置；不可将不同分支协议的绝对AUC/LL跨表排序。对自研组件做内部消融时允许相互比较，但标题、列名和结论必须标为“内部消融/控制”，不能以这种比较冒充已发表模型的baseline实验。最新的五seed配对平均AUC≥0.003、逐域任务格平均跌幅不超过0.001、五seed平均LL下降的规则同样用于相应可比的正式评价。
4. **主目录性能总台账为长期维护入口：** [MDMT_ALGORITHM_METRICS_LEDGER.md](MDMT_ALGORITHM_METRICS_LEDGER.md)（对于 experiments/ 目录文件请使用相对链接 `../MDMT_ALGORITHM_METRICS_LEDGER.md`）。新增算法实验结束后，必须回填**每个域×任务格的绝对AUC与LogLoss、总体Macro AUC与Macro LogLoss，以及有据可查的traffic/pooled指标**；同时记录数据集、seed数量、模型/算法配置、在线/EMA状态、selected/common epoch、协议身份、来源报告和原始数表。不同协议分组存放；不得猜测缺失数字或将少于5seed的结果写为已充分否决。台账中的自研方法数据仅为**历史观测记录和内部方法分析**，不赋予其正式baseline资格。
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

# 面向后续 AI 实验的最小充分科研上下文



## 2026-11-06 最新已完成：AUG_HISTORY增强响应与参数历史

[中文总报告](research_reports/2026-11-05_aug_history/REPORT.md)及[B](research_reports/2026-11-05_aug_history/B_STAGE_ANALYSIS.md)/[A](research_reports/2026-11-05_aug_history/A_STAGE_ANALYSIS.md)/[C](research_reports/2026-11-05_aug_history/C_STAGE_ANALYSIS.md)分析为本轮来源。规划固定`2645f1f724f52f5ed5dea4b479335c0dc12fabcd`，父源码`c8675ae2923c15b6965884b140b5b646690cd95e`；沿用自然80/10/10、M3oE fixed1、batch40960、50epoch，所有科学候选五seed2022–2026。每阶段独立设计/实施/审查后正式运行，选择及下一动作在本阶段test前冻结。

- **完成与计数：** B/A各两数据×五seed，C仅ML六配置×五seed；50次物理训练、2500epoch、64500 Adam、19000模型validation。B/A的500个online/order hash逐一相同，是十个父路径重放；统计n仍为每配置五seed。96条配置×数据集×selected/E10/E50记录已回填主台账。KR C为DEFERRED_NOT_RUN，B+C未运行，最终STOP_AND_REPORT。
- **B，STRONG描述性事实：** D−F test AUC为ML+0.000000104、KR−0.000004946；响应实际启用，D秩2/M秩1。当前有限方向EMA网格没有实用幅度独立增量。
- **A，MEDIUM：** ML GROUP/GLOBAL对F test AUC−0.000750/−0.000728，LL−0.000813/−0.001472，4/5 seed AUC下降；ML按validation简化选F并进入C。KR固定S_FAST的validation+0.001009861转为test−0.000217263、LL+0.000311569，内部最差格−0.001824793；按原validation动作保留profile并停止KR后续C。早期E10优势不等于selected收益。
- **C，MEDIUM：** 搜索赢家C3_WEAK(γ=.5,K20)，最终按validation成本容忍规则选C_MERGED。MERGED test AUC/LL=.778689413/.496180076，对F−0.000667190/+0.000357811，五seed AUC均负；C3/ROTATED均值也低于F。任务C*对MERGED/ROTATED AUC+0.000397735/+0.000018904，CI均跨0且LL均更高；任务分解、真实方向独立必要性证据WEAK。1375次刷新中1345次统计非零，其余30次为初始化。
- **总收益与模型：** ML完整MERGED+EMA对普通M3oE AUC+0.003499743、最差格+0.003065726，满足正式AUC及保护线；LL+0.001289159。KR A_FIXED_S_FAST对主发表参照HyperGate AUC−0.003149636、LL−0.004760317。完整方案总收益和新增组件增量分别裁定，F仍属内部控制。HG保持UNVERIFIED_PROPOSAL配置标记。
- **解释与后续：** A的KR GROUP局部出现小域改善/大域下降，固定profile反而损失最小域，C多格同降；规模主解释为WEAK。当前版本停止。最有区分力的新研究是固定train-only统计及逐步谱/强度后的真实方向对旋转方向，保留F和新封存自然人口评价；本轮未执行该建议。

运行事实：B四次必要RAM轮换，A两次必要及一次不需要的重复轮换，均从保留的RAM提交继续；额外未提交开销缺少完整计时。C无轮换。源码、标量、曲线、hash及报告归档，参数/优化器/Q/A/root/旋转矩阵及checkpoint均仅RAM。test为历史研究划分。

## 2026-11-04 最新 AI Handoff：五seed、统一发表模型与Gaussian覆盖（优先于11/02旧快照）

**本次增量范围。** 此前11/02归档的EXP052–EXP060不再重复登记；新增整合11/02–03 PAIR/ECCM补充、11/03条件信息/真实历史分支、11/03五seed补全（11/04发布）、11/04 Overnight A+B统一发表模型比较，以及11/04 Gaussian增强域覆盖和MovieLens来源补齐。仅提取科学结论和关键指标，不合并原实验分支的代码、日志或训练产物；[指标台账](MDMT_ALGORITHM_METRICS_LEDGER.md)记录六格绝对值、Macro/Traffic/Pooled及选点身份。**下方11/02“最新”“下一步”是历史快照，不得覆盖本节研究决定。**

1. **正式论文模型已统一AUC选点，不能继续把这项工作列为未执行。** FULLTRAIN_NATURAL80_10_10、train-only预处理、validation六格Macro AUC最早最大值选点，test五seed：MovieLens M3oE fixed1 / HyperGate / AutoIFS Macro AUC依次**0.775190 / 0.771412 / 0.772425**，其中M3oE最强；KuaiRand依次**0.671907 / 0.682113 / 0.678597**，其中HyperGate最强。三者是已发表论文模型的工程适配，各自保留不同的训练/优化器/批量和AutoIFS search/reuse阶段；HyperGate超参仍标记UNVERIFIED_PROPOSAL。旧10/29的V按LogLoss选点及训练人口/预处理不同，旧排序不能跨协议继承。[A+B统一对比](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md)
2. **五seed补全修正旧单seed的局部判断：总收益≠组件增量。** 相对普通M3oE online，C_MIX/H_MIX/P01＋固定EMA在MovieLens配对Macro AUC分别**+0.003253/+0.003059/+0.003225**，KuaiRand分别**+0.003720/+0.005579/+0.008702**，均达到五seed总AUC阈值且六格均值满足−0.001保护。相对*无新增组件*固定EMA，MovieLens三者的新增组件AUC增量均负；KuaiRand P01组件增量**+0.004086**，但d1t0/d1t1 **−0.001334/−0.001047**，属于“宏收益达标、单格保护未满足”。M整体增量在两数据均不足+0.003。早期条件信息实验只在seed2022执行13条新轨迹，KR C_MIX test+0.009531/LL−0.005302但validation d0t0下降；旧D0停止门、未触发阶段不再冒充五seed结论，T0/TM/H_REAL依旧只具早期信号。[五seed补全](https://github.com/Caelestis1903/My-Research-Project/blob/652ee0235fec2d0d4cfc67b4a6ef9977519619bf/research_reports/2026-11-03_mdmt_five_seed_completion/REPORT.md)；[旧条件信息报告](https://github.com/Caelestis1903/My-Research-Project/blob/f9c2a8ef7b29b899e63b1beba823df1b6ea22c72/research_reports/2026-11-03_mdmt_history_information/REPORT.md)
3. **Gaussian＋EMA在两个数据集已可同协议定位，覆盖更多领域并不保证更好的AUC。** MovieLens旧Gaussian实际增强d0（现名G-D0），Macro AUC/LL **0.779357/0.495822**；对本轮最强论文模型M3oE AUC **+0.004167**，六格保护通过，LL却**+0.000931**。KuaiRand旧Gaussian增强d2（G-D2）Macro AUC/LL **0.677999/0.434283**；本轮新增G-D0 **0.679180/0.435518**，G-D02（同时增强d0、d2）**0.674604/0.436172**。KR D0−D2 AUC **+0.001181**；D02−D0 **−0.004576**，d0t0 **−0.014949**。三种KR Gaussian均不及最强论文模型HyperGate 0.682113，但G-D0相对M3oE的完整方案总AUC **+0.007274**、六格保护通过；对固定EMA的组件AUC **+0.002657**，d1两个任务下降越过保护线。ML G-D0核验**仅复用已有五seed，新增训练0条**，并非另一个新算法。[KR覆盖报告](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/REPORT.md)；[ML别名与来源核验](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-04_gaussian_domain_coverage/MOVIELENS_SUPPLEMENT.md)
4. **C_MIX＋P01的CP组合未取得预期排序叠加。** CP fixedEMA在ML/KR Macro AUC/LL分别**0.777525/0.494212**和**0.678168/0.434505**；对普通M3oE总AUC **+0.002335/+0.006261**，对固定EMA内部组件增量**−0.001514/+0.001645**，对P01分别**−0.000890/−0.002441**。ML未达到相对最强发表M3oE的+0.003，KR落后最强HyperGate **−0.003945**。保留ML相对HG/AutoIFS的AUC优势、KR相对M3oE已达标整体AUC与LL改善，但**当前排序优先级不以CP取代P01**。当前投影同时改变梯度方向与范数，不直接将损失变化解释为独立方向机制。[CP完整2×2及A+B](https://github.com/Caelestis1903/My-Research-Project/blob/c8675ae2923c15b6965884b140b5b646690cd95e/research_reports/2026-11-03_overnight_AB/FINAL_REPORT.md)
5. **PAIR/ECCM早期两seed仅作为组件线索，后续完整P01五seed是另一个结论层次。** 11/03补充P01/P10时，KR P01−P00 test AUC **+0.012099**（n=2），但validation **−0.000101**；KR两seed2×2交互（P11−P10−P01+P00）test **−0.012347**。说明在此配置内存在非加性与V/test差异，但不证明ECCM单组件的五seed必要性。原S2/S3 NOT_TRIGGERED与KR TM的COST_BLOCKED原样保留为执行状态。[PAIR/ECCM与补充](https://github.com/Caelestis1903/My-Research-Project/blob/a06f9d3e544a8658b2d24ddc8f7d10d3f20f0288/research_reports/2026-11-02_mdmt_information_momentum_v3/REPORT.md)

**研究决策：** 正式论文Baseline仅使用已发表的M3oE、HyperGate、AutoIFS；固定EMA及C/H/P01/Gaussian/CP只能作为候选或内部消融。现有五seed足以接受若干完整配置相对普通训练的总AUC收益；不能把这部分收益完全归因于新增机制。ML G-D0已超过最强发表模型并满足六格保护；KR Gaussian未超过最强HyperGate。KR宏AUC优先G-D0、宏LL优先G-D2；当前不优先追加G-D02。未来如研究独立梯度/历史机制，只在差异可能改变下一方案选择时安排匹配控制，不重复测试已归档事实。

---

## 2026-11-02 AI Handoff：历史研究事实及不重复清单（11/04优先）

**本段仅是2026-11-02历史Handoff；以上方2026-11-04最新裁定为准，但相对下方10-26、10-18、10-09旧快照仍具有更高优先级。** 核对了截至11-02尚未进入main总台账的9组不同实验或正式基线活动，登记为[EXP052–EXP060](experiments/Experimental_Experiment_Index.md)和[E058–E069](experiments/Experimental_Evidence_Ledger.md)。只汇总结论，不复制研究分支代码、数据、报告或实验产物。已执行报告的同seed/同训练轨迹复用不当作独立验证。

**已可接受的事实：**
1. **固定参数EMA是当前有效的selected AUC基线。** 11-01的无增强online→非反馈EMA shadow，在MovieLens test宏AUC配对+0.004399（5/5），在另一次调参对照+0.003849（5/5）；KuaiRand +0.004616（4/5）。ML同期LogLoss分别+0.007800/+0.001249，均4/5恶化，**排序收益与概率损失收益不可等同**。EMA并未改写真实在线Adam动量递推。
2. **统一loss尺度已经有可重复影响AUC的实证。** GA31-v2 ML无增强c=.1对c=1 selected test AUC+0.003473（4/5）、KR−0.006026；早/晚epoch可反向。耦合Adam中loss scale同时改变相对wd与eps；R3普通L2控制在ML可形成相近的简单工作参照。旧10-21 D套件也曾阳性，但使用不同目标、复制/噪声和人群，**不能直接把新缩放数值当成旧D的因果分解**。
3. **更复杂的增强、差分与梯度视图尚未取得稳定独立方法收益。** NR-Gauss未过增量门；ANTI2确实降条件方差但未优于同预算IID2；SAD长差分记忆非零并改变更新但不胜普通增强；AIG-M创新度门、GATS的难度覆盖B1/延迟B3/CV→rho C2未晋级。GATS任务梯度投影B2在KR开发验证有正信号，但统一test五seed对A1的宏AUC只有+0.000005、对强A0+EMA仅2/5正。EMA02V2 ML F1仅有2seed测试+0.000359且验证反向；KR TG/AG相对A0+EMA的宏AUC约−0.0172/−0.0175，LL同时恶化。不要把没运行的后续确认阶段说成阴性。
4. **共享/原型历史结论仍有范围。** Relation-Basis v2与BASP-A的所测形成/收益群组方法都未稳定超过强TARGET_ONLY/TARGET_MATCH；不否定此前局部来源可读性。10-29 HyperGate、AutoIFS**已经完成**两个自然数据集各五seed正式对比，并且相对M3OE_FIXED1的macro AUC四组配对均5/5正；各方法训练/选模协议与GA31不同，不能跨表直接排名。
5. **任务/域的收益分配是有实证的。** KR Gaussian+EMA的宏LL可以5/5下降，但占流量优势的d1两个任务AUC多数seed下降；GATS B2各任务与验证/test分化。任何新算法都需要比较六格、macro/pooled/traffic、AUC/LL、selected/共同epoch，而不是只取正的宏均值。

**下一步研究选择（明确为设计目标，而非已经证明的因果机制）：** 以固定无增强EMA为强参考，思考如何利用不同**域×任务**的信息构造具有针对性的增强或梯度，并使其**实际改变动量/参数时间递推公式或状态**；不能把普通梯度转成“梯度构造视图”就视为完成梯度＋增强＋动量的联合机制。只有当新增不确定性会改变算法选择时，安排匹配强简单控制的判别实验。过往已排除/保留的特定结论不能在下一分支以改名方式重新当作新问题。

**九组源报告（均指向源分支固定提交，文件没有合入main）：** [10-27 Relation-Basis](https://github.com/Caelestis1903/My-Research-Project/blob/03c338092008a71b2d360a61b06d985eac551ec7/research_reports/26-10-27_v2/FINAL_REPORT.md)；[10-28 BASP-A](https://github.com/Caelestis1903/My-Research-Project/blob/97151c74544c3651feb805d1898bc9ce1b8d12d3/research_reports/2026-10-28/FINAL_REPORT.md)；[10-29基线](https://github.com/Caelestis1903/My-Research-Project/blob/d2fdd272a2cdf7da626e29b97f61555d25271f89/research_reports/mdmt_baselines_2526/FINAL_REPORT.md)；[10-31 GA31-v2](https://github.com/Caelestis1903/My-Research-Project/blob/ee6e81fdfd37284e6447e8149cb93d04736dba02/research_reports/2026-10-31/REPORT.md)；[11-01 GA31-R3](https://github.com/Caelestis1903/My-Research-Project/blob/a6f496b54f467cdf7b814b6bb8defdbdd457bc38/research_reports/2026-10-31_v2/REPORT.md)；[11-01动量/EMA](https://github.com/Caelestis1903/My-Research-Project/blob/ee4cf0512a7723624c9ee6a1ef192f0f7f17686b/research_reports/2026-11-01_momentum/REPORT.md)；[11-01 AIGM01](https://github.com/Caelestis1903/My-Research-Project/blob/6f5742c2085c1e49d1767b0a7079bb9f28160eda/research_reports/2026-11-01_aig_m/REPORT.md)；[11-02 GATS02](https://github.com/Caelestis1903/My-Research-Project/blob/e59efa2fd954c4d2231b679a742406cfbe2a1077/research_reports/2026-11-02_gradient_aug_time_select/REPORT.md)；[11-02 EMA02V2](https://github.com/Caelestis1903/My-Research-Project/blob/38a53bc1aa6e0c16c0b6d2b0e24e07018f91ab27/research_reports/2026-11-02_ema_view_design_v2/REPORT.md)。

---

> **重要状态更正（2026-10-26）：** 下文原先写作“当前”的S3/S4机制、S5未识别性、下一步实验建议，均属于**当时的历史handoff快照**。后续已出现多轮独立执行、新人口复核及算法决策。新AI会话的当前研究事实入口应为 [跨阶段总报告](experiments/Experimental_Branch_Synthesis_20261026.md)、[Global Summary](experiments/Experimental_Global_Summary.md)、[历史实验原件索引](history_experiments/README.md)。不得只读下方旧RQ而忽略后续实验。

## 2026-10-26 AI Handoff：当前最小充分研究状态

**研究主线：样本重加权 → 优化/梯度实际响应 → MDMT差异化共享与目标局部需求。** Equal与Sample是某些训练口径，不替代长线主题。“小域=困难域”从来不是研究者承认的前提。

1. **数据与人口先行。** 10-15以来多轮MovieLens用了187500行人工固定采样（34500/3000/150000），全自然base-train是800167行（155949/318599/325619）；两者同时改变比例、有效支持和优化时钟。10-20新自然补充有S5/S6、V2、AO04、N17等阶段性核验，但 [CURRENT_WORK](history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/CURRENT_WORK.md)仍IN_PROGRESS，N19/CM19完整自然矩阵未完成。10-25/26自然fit是base-train的内部子角色，不能与人工187500或base-train总量混同。
2. **优化与共享历史并存。** Sample同经验目标不同q可改变训练；无动量SGD亦出现q效应，故Adam不是**通用必要条件**。旧S5 Equal下特定persistent-history承载仍有效，不能被SGD反例删除。S4 raw/applied norm匹配不足以解释target rescue。N17正式KR selected两主比较未过预设Holm；不同训练预算、宏/加权AUC与LL可反号。
3. **共享路径机理的边界。** [CM19](history_experiments/codex__conditional-paths-2026-10-19/research_plans/2026-10-19/execution/FINAL_REPORT.md)完成856正式训练，在KR fixed M3oE的**输入乘性调制路径**观察selected数值中性；局部AdamW/双factor耦合L2可使路径恢复响应，STAR等独立结构路径仍活跃。**路径恢复≠泛化收益已建立，不能称所有共享参数失活。**
4. **重加权与生成的已验证部分。** [10-21报告](history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md)完成48正式训练：MovieLens D−B test宏AUC +0.006303/LL −0.015962，而Dconst01−B仍有+AUC0.005676，不能将整体收益归因动态specificity。D的权重迅速接近下界。G−Gtau0匹配路径宏AUC差ML −0.003466、KR −0.006484（KR区间跨0）；其训练目标是legacy_group_sum，不能冒充其他轮的Sample或Equal。
5. **共享收益算法的实际成绩。** N18有六方向研究，N19完成328正式训练，但完整强对照下新算法贡献尚不成立。10-24 I1未胜固定RW；10-25 v2 48核心完成而正式门未触发；v3 RHD相对BASE selected test LL +0.00074075、e50 −0.00875428，LPS部分source可读但特定形成器不稳；10-26 DFD-v4 ML P1有有限选读信号、KR P1失败，ML P2弱于H_DIRECT/ETA_DIRECT且P3/P4未触发，另缺Q/outer固定μ分组。**版本NO_GO不是来源知识或“收益驱动多原型”整个科学问题失效。**
6. **研究决策与未执行边界。** [RCF-v1设计](research_plans/2026-10-26-v2/00_README.md)已经制定，但**尚未实现或训练**。拟用真实目标群体剩余风险组织有约束的来源后继函数形成，需对照强目标直接头、固定来源读取、target-only、统一微调和容量/预算匹配。样本权重次线优先最小同loss尺度且无复制/噪声的匹配控制，不预设动态权重有益。

**执行/证据原则：** 每次比较先核查人口、优化目标、采样q、model/fusion/optimizer、更新步数、seed、合法选点、six-cell和LL、Q/validation/test与历史外露性。NO_START不是阴性训练、训练seed不是独立新数据、非零路由不是因果增量。查不到原checkpoint不捏造前向复核。讨论创新应明确强简单近邻与具体增量，不因存在先例就提前放弃方向。

---


## 0. 文档用途、权威级别与读取规则

本文件是 M3oE / 多域多任务学习项目供后续 main agent、designer、coder、reviewer、runner 和 analyst 常驻使用的精简研究状态与 handoff；不是完整实验记录，也不替代 **experiments/reports/exp.md**。完整数字、逐 seed 结果、审计过程和运行日志以归档、已签署阶段报告及对应原始 CSV 为准。

阅读顺序：先按本文件的当前裁定理解研究状态；历史解释仅用于追溯，不得覆盖后期裁定。若本文件与完整归档不一致，以完整归档及其指向的源报告核对；具体统计值以对应原始 CSV 为准；机制结论优先采用最新的正式 Reviewer / analysis 裁定。若源报告彼此冲突且没有后续正式裁定，必须标为 OPEN / 未解决，不得自行挑选。未明确支持、仅相关或被标为诊断性的内容，不得写成已证明因果事实。

文中标签：**FACT** = 在指定数据和协议下直接观察或核验；**INTERPRETATION** = 有证据支持但因果边界有限；**REJECTED / STOP / DROP** = 当前路线裁定；**OPEN QUESTION** = 尚未拆分；**LIMITATION** = 外推或口径限制。

---

## 1. 历史研究状态（截至早期S4；非2026-10-26当前裁定）

### 1.1 当前研究问题

项目从复现 M3oE 及测试“域规模驱动的逐域动量能否保护不均衡下的大域”起步。随后固定总样本量、轮换域大小角色并操纵损失目标，发现早期所谓“大域牺牲”很大部分依赖 equal objective；sample-proportional weighting 是更稳妥的目标基线。domain-conditioned momentum 的残余增益虽可在修正目标后复现，但匹配控制表明它不需要域条件机制即可解释，故该算法路线已 STOP。

S 系列进一步区分普遍负迁移与目标函数条件效应：Sample 目标下没有跨数据集、backbone 和阶段稳定的实用残余负迁移；Equal 目标下则在特定 Large 域 cell 出现稳定退化。最新 S3/S4 将证据收窄到 KuaiRand-Pure、M3OE、Equal 目标、本地 tab1（最大域）这一条件：限制非目标域访问 shared update 可显著救援目标域，但会损害非目标域；raw gradient norm 或即时 Adam applied-step magnitude 单独不足以复现救援。当前值得继续的核心问题是：该条件性救援来自每步梯度方向/组成，还是来自这些更新对 Adam moment history 的影响，且如何避免把损失转移给其他域。现有证据不支持普遍安全 MTL 方法或通用梯度冲突叙事。

### 1.2 当前已确认科学事实

**F1. M3oE 的 as-published 复现成功。** MovieLens-1M 五 seed 的早期六个 domain-task AUC 汇总为 0.7686±0.0017，论文值 0.7702。边界：对应旧训练与评估协议，不等同于后续 sample objective 或 SD-reference 实验。来源：P0。

**F2. 旧版可微 AutoML 未显示可测贡献，且存在会影响可比性的 optimizer bug。** 冻结 0.7311 与 as-published 差异不显著（+0.0007，p=0.32）；B10 使四个 architecture scalar 同时进入两个 Adam。修正后均值无显著变化，但 seed SD 从约 0.0017 降至 0.0002。边界：不据此推断其他模型的所有 architecture search 都无效；跨 bug 修复前后不得未经标注地合并方差。来源：P1/P2。

**F3. “小域梯度噪声更大”这一早期动量设计前提被否定。** MovieLens batch 40960 时小/大域 SNR 约 0.509/0.558、绝对差小；batch 4096 时域差消失并反号。共享梯度 cosine 弱正相关而非明显冲突。边界：这是特定 MovieLens、该共享参数集合与梯度统计口径下的诊断，不证明其他数据无噪声差异。来源：P1。

**F4. 早期 P5 的 comp1 大增益是真实观测，但“随不均衡度单调放大”不成立，且跨档归因受总样本量影响。** MovieLens r30/r100 相对旧 baseline 分别 +0.0166/+0.0202；六档差值为 +0.0064、+0.0034、−0.0012、+0.0053、+0.0166、+0.0202，1:1–10:1 非单调，3:1 处于待复核区。下采样同时改变 N；固定 N 的 Px_n 后证实 ratio 本身仍能驱动部分增益，但后续目标控制和 S0 又改变了机制归因。边界：这些旧数不能直接当作当前 comp1 方法的净效应。来源：ML_P5_REPORT、P0_1/2。

**F5. KuaiRand 上旧 comp1 只推动 Pareto 前沿，没有达到大域 oracle。** 本地 d1（论文大域）t0 Δoracle 从 −0.070659 收窄约 +0.0091，名义 p=0.0228；t1 收窄 +0.0171 但 p=0.369。跨 15 个检验未过 Bonferroni。边界：仅方向性证据；绝不能把两任务都写成显著改善或称为真 Pareto 外推。来源：KUAI_P5_MAIN_REPORT、FINAL_RESEARCH_REPORT。

**F6. 固定 N 与域角色轮换下，早期负迁移跟随 Large 角色而非固定物理域。** Px_n 的 MovieLens 角色旋转显示 L 角色 Δtransfer 为负，M/S 多为正；在多个 ratio 上负值随失衡加深。MMoE 也复现方向，故旧现象不是 M3oE 独有。边界：域大小与有效权重、份额、样本多样性、单样本梯度尺度和收敛共同变化；“size role 是效应位置标签”比“size 单独是因果机制”可靠。来源：P0_1/2/3。

**F7. Equal 与 sample-proportional 是不同的有效训练目标；Equal 可制造大域损伤，Sample 可大幅缓解。** 固定 N、相同曝光/初始化等条件下，P0-4 显示训练 weighting 是 Large 角色损伤的主控制因素；P0-5 2×2 表明 comp1 在 sample 下仍有 +0.0054 左右残余，但旧 equal 下增益大约 69% 在 objective 修正后消失。边界：“92%/69%”是特定统计口径下的描述性分解，不是中介比例；sample+comp1 均值无 Large 损伤不等于逐 cell 全正。来源：P0_4/5 及独立 evidence audit。

**F8. Curriculum 已在修正目标下 DROP。** P1-2 的正向、反向 architecture-fusion curriculum 均未产生稳定最终 AUC、峰值 epoch、梯度动态或 comp1 交互；课程未完整到 ρ=1，裁定只覆盖该实现与预算。E5 又测试了 loss-weighting 的前向/反向时间 schedule：具体预设 schedule 不胜静态 α=1，故再次 STOP。边界：不同机制的两个阴性结果不等于所有可能的时间变化数学上皆无效。来源：P1_2_STAGE_REVIEW、stage_E5_analysis/go_no_go。

**F9. Sample objective 下的贡献差异有机制线索，但不能据此宣称单一因果中介。** E1 instrumentation 观察到 sample 让实际 shared update alignment 向 Large 移动，equal 下 Small 对 shared optimization 的超代表性很强；约 98.5% mismatch 削减来自系数的数学效应，routing 解释未排除。E3 对权重系数做响应面干预，AUC/transfer 总体随 α 增加而改善、无稳定 interior 胜者；端点对 E1 bit-exact 复现。边界：E1 是观察性机制诊断；E2（目标系数 × sampling 因果拆分）未执行，E3 不能把响应面归因于纯 objective 而非 sampling exposure。来源：stage_E1/E3_*、exp.md §18。

**F10. Sample weighting 在 KuaiRand 有跨数据集方向性支持，但仍有残余 Large gap 和数据漂移边界。** E4 中 Equal 下 dominant d1 两任务均有负 transfer，Sample 改善 d1 与全局指标；Sample 下 d1 仍约 −0.006。训练到测试域组成 TV=0.112839，d1 click 正例率从约 0.378 降至 0.211，是重要竞争解释。边界：支持跨数据集方向，不构成时间漂移或 objective 的独立因果分解。来源：stage_E4_dataset_audit/analysis。

**F11. 动态/自适应路线未超过静态 Sample，且被正式停止。** E5 的 schedule 不过 gate；E6 的 adaptive calibration 能把目标 mismatch 压低但没有稳定胜过 Sample，EMA 不是必要机制，中间域还触发预设安全限制；E7 因 E6 STOP 而 gate STOP，正式 runs=0。边界：不能将 Sample 重新命名为新 adaptive 方法，也不能将 E7 描述为负结果实验。来源：stage_E5/E6/E7。

**F12. domain-conditioned momentum 的历史残余效应可复现，但其独立归因被 S0 否定。** Sample 口径下 comp1 相对 baseline 的约 +0.0054 新鲜复现；然而 comp1 输给验证集可选的 uniform-smoothing family（selected 差约 −0.0030，0/4 seed 为正），KEEP gate 第二条失败。S0 正式裁定 STOP Momentum：普通时序平滑足以解释，域条件机制不独有。边界：效应存在不等于原机制归因成立；S0 未证明 uniform EMA 在所有 setting 最优。来源：stage_S0_analysis/review。

**F13. 在 Sample 目标与架构匹配 SD reference 下，不存在跨条件稳定的实用 residual negative transfer。** S1 72 组均值中 63 正、9 负；0/72 达到稳定实用 NT 标准。最负候选 −0.002923，幅度被 reference 的 seed 方差覆盖；换 backbone、dataset 或共同训练阶段即不稳定。S1 因此 STOP 将“普遍 residual NT”作为主问题及 Safe-MTL 方法线。边界：不排除另一目标函数或特定任务的条件性退化；S1 头号候选在 matched common epochs 消失。来源：stage_S1_analysis/review 与台账。

**F14. Equal 目标下仍有窄而稳定的条件性负迁移，Sample 将其压到实用阈值之上。** S2 selected 视图中 8/24 组 Equal−SD ΔAUC≤−0.003，均为 Large 角色；严格组合门仅 3/24 组通过。Sample−SD 没有组≤−0.003，8 个 Equal 受损组全回到阈值上方。最强目标为 KuaiRand tab1 两任务；另一预设候选为 MovieLens d2_large rating>3。边界：这不是普遍 sharing failure；Large 角色因果解释为 MEDIUM，size 与份额/权重等共变；训练阶段可反转 MovieLens 部分结果。来源：stage_S2_analysis/review。

**F15. AUC 与 LogLoss 不可互换，selected checkpoint 也不足以单独支持机制 claim。** S2 有 7 个 Equal−SD 组 AUC 为正但 LogLoss 变差；MovieLens Large 的 selected 损伤有多项在 late/common-final 缩小或反转，而 KuaiRand tab1 的损伤随训练加深更稳定。边界：报告须并列排序与校准指标及训练视图；不能用 macro 聚合掩盖目标 cell。来源：stage_S2_analysis。

**F16. S3 识别出 KuaiRand Equal/tab1 下 shared-access 干预对目标 AUC 的大幅因果效应，但它转移了其他域的表现。** Equal 的 target-only−FULL selected AUC：click +0.038876、long-view +0.039576，4/4 seeds 同向；Sample 下同一干预约 +0.0003/+0.0001。Equal−Sample 目标差距缩小约 89%/92%，仍有残余；Equal 非目标域 AUC 平均 −0.017040（16 个 task-seed 中 14 个负）。边界：目标限 KuaiRand tab1；MovieLens replication 晚期反例；target-only 是诊断干预，不是安全方法；尚未直接测试几何梯度冲突。来源：stage_S3_analysis/review/amendment_review。

**F17. S4 排除了“只缩小梯度幅度”作为 KuaiRand d1 救援的充分解释。** 与 FULL 相比，target-only selected 增益约 +0.039；把 FULL raw gradient norm 匹配到 target-only 的 N 臂反而 −0.0040/−0.0049，按即时 Adam applied-step 匹配的 U 臂仅 −0.0006/−0.0009。d1 救援对 N/U 仍约 +0.04，跨 4 seed 与 5 个 common epochs 稳定。边界：支持“梯度组成/方向及其诱发的 Adam 历史”组合解释；不能拆成纯方向或纯 optimizer history，也不能泛化为通用梯度冲突。来源：stage_S4_A2_review/execution_report/analysis 与 A2 CSV。

**F18. S3/S4 的效应不是全局改进。** S4 target-only 对 KuaiRand d0 两任务 selected AUC 约 −0.031/−0.032；d2 则在 late checkpoint 转正。目标收益、非目标代价和阶段须同时报告。边界：S4 仅 M3OE、KuaiRand natural、Equal、tab1；size role 与该效应相关，但未独立操纵。来源：stage_S4_analysis。

**F19. S8 终裁：S6 的 DH/T late-stage benefit 仅为 late overfitting rescue，不具备方法化价值（STOP）。** 在完整 optimizer state（含 private Adam moments，修复 S7-B 限制）、严格 same-state、same-batch、2-epoch continuation 下：MovieLens late anchor 的 macro rescue 4/4 同向（final DH−F +0.0107）但 ~89% 由 2024/2025 两个 seed 承载，target-domain 口径仅 +0.0004~+0.0026（S6 的 target 口径 residual 未以同量级复现）；selected anchor 为 null（~6e-5，practical threshold 的 1/50）；任何 dataset × anchor × branch × seed 的 continuation best 都未稳定超过 F 原始 validation-selected best；KuaiRand 保持 null 对照且 T 在 non-target 有实际损害（final −0.0102）。Adam 预调节系统性吸收 raw-gradient 方向分歧（同 θ 下 effective cosine 显著高于 raw cosine），eff_align 与 validation rescue 无空间对应。GO 条件 5/6 失败；STOP-1/2/4 触发、STOP-3/5 部分触发。最终裁决 **B. LATE OVERFITTING RESCUE ONLY**，optimizer/gradient method 主线停止，禁止 S9 gradient method。边界：kill test 限 Sample objective、M3OE、ML d2_large/KR natural、2-epoch 窗口；不排除"过拟合救援"作为独立课题的价值。来源：stage_S8_analysis.md、stage_S8_outputs/。

### 1.3 当前机制解释

**已基本排除：** 小域 SNR 必然更低；普遍 residual NT；M3oE 独有的“大域牺牲”；domain-conditioned momentum 的独立优势；本项目已测的课程、线性 schedule 与 adaptive calibration 优于静态 Sample；仅用 target 或 norm 大小即可说明 S3/S4 救援；基线难度是 S2/S3 效应的充分解释。

**仍获支持但范围狭窄：** Equal 的有效目标在不均衡域份额下会产生条件性 Large-cell 退化；在 KuaiRand Equal/tab1，非目标域对 shared-update 的访问是目标损伤的重要来源；仅匹配 raw norm 或即时应用步长不足以救回目标。证据分别是目标加权干预与 shared-access 干预，不等于已确定统一的单一机制。

**尚未拆分：** 梯度方向/组成如何与 Adam moment history 共同形成救援；target rescue 与 d0 cost 的关系；domain size、有效权重、数据异质性及训练速度各自贡献。当前不应将其包装成“几何梯度冲突”或 size 单变量机制。

**证据强度标签：** Equal/Sample 目标效应在固定 exposure 的直接对照下为 **STRONG**；将该效应单因归于 domain size 仅 **MEDIUM**。S0 对 domain-conditioned 独立优势及 S1 对普遍 residual NT 的 STOP 均为各自预注册范围内 **STRONG**。S2 的 KuaiRand tab1 条件 NT 和 S3 shared-access rescue 为条件内 **STRONG**，size-role 作为因果机制为 **MEDIUM**。S4 的 norm / applied-step controls 未救回目标为条件内 **STRONG**；把残余归于方向/组成加 Adam history 为 **MEDIUM-STRONG**，纯几何冲突未证明。E4 跨数据集方向为 **STRONG**，其因果归因受 temporal shift 限制。

---

## 2. 已证伪、已停止、已降级的研究方向

| 假设 / 方法 / 研究方向 | 当前状态 | 为什么 | 决定性实验 |
|---|---|---|---|
| “小域梯度噪声必然更大”，以此解释逐域动量 | REJECTED | P1 两 batch 下 SNR 差小/消失，梯度余弦弱正而非强冲突 | P1 D0.4；P6 SNR γ 还显著差于公式 γ |
| M3oE 可微 AutoML 是主要性能来源 | NOT PROVEN / DOWNGRADED | fixed 0.7311 与旧版无显著差异；B10 导致双 Adam 更新 | P2 fixed073/fixedbug 对照 |
| “comp1 增益随不均衡度单调放大” | SUPERSEDED | 完整六档非单调；旧 r30/r100 同时缩小 N | ML_P5_REPORT Reviewer 仲裁；P0-2 固定 N |
| M3oE 固有 large-domain sacrifice / M3oE-specific 方法问题 | SUPERSEDED | 固定 N 轮换后现象随大小角色移动；MMoE 也出现方向 | P0-1/2/3 |
| 普遍的 small-large trade-off，与目标 objective 无关 | REJECTED | sample weighting 可同时改善宏观和目标；Equal/Sample 目标条件不同 | P0-4/5、S2 |
| comp1 是修复 objective mismatch 的必要方法 | SUPERSEDED | 大部分旧增益随目标修正消失；残余由 uniform smoothing 匹配解释 | P0-5、S0 |
| domain-conditioned momentum 有独立机制优势 | STOP / REJECTED | fresh residual 存在，但对 matched uniform-smoothing family 失败 | S0 KEEP gate |
| Curriculum（域条件融合权重） | DROP | Sample 目标下性能/动态/反向对照/交互均无稳定效果 | P1-2 |
| 时间顺序 contribution/loss schedule | STOP（仅限所测 schedules） | E5 前向 schedule 不胜静态 Sample，未通过 GO gate | E3 诊断后 E5 预注册反证 |
| 在线 adaptive contribution calibration + EMA | STOP | 精确匹配 contribution 不充分；不稳胜静态 Sample，存在中间域安全代价 | E6；E7 因 STOP 为 0 runs |
| 普遍 residual negative transfer / Safe-MTL 主线 | STOP | Sample + SD reference 下不满足幅度、稳定性、跨条件门槛 | S1 |
| Target-only shared update 作为可部署方案 | DIAGNOSTIC ONLY | 能救 KuaiRand Equal/tab1，但显著损害 d0/非目标域 | S3/S4 |
| 梯度 norm matching / applied-step matching 可单独解释救援 | NOT SUPPORTED | S4 N/U 未重现 d1 救援 | stage_S4 A2 |
| “梯度几何冲突”已被证明 | NOT PROVEN | 未拆开方向变化与 Adam 历史；单步 probe 也非每步一致 | S3/S4 的限制 |
| 任意 backbone、D/T 或数据集均可泛化 | NOT PROVEN | P7 多数支持限 MMoE/PLE 与 D=3→5,T=2；当前 S2–S4 核心 effect 更窄 | P7、S2–S4 |
## 3. 当前开放研究问题

### RQ1 — KuaiRand Equal/tab1 的 target rescue 究竟依赖什么？

**已知：** S3 的 shared-access 干预在 Equal 下显著救回 tab1，Sample 下该干预近零；S4 的 raw gradient norm matching 与即时 Adam step-size matching 都没有复现 rescue（F16–F17）。  
**未知：** 每步 gradient direction/composition 的作用与 Adam moments 被不同梯度序列塑造的作用仍缠在一起；方向探针支持度有限，并非每一步都同向。  
**为什么重要：** 这是当前唯一被 S 系列定位并重复的实用目标域损伤，也是判断能否形成机制贡献的关键。  
**需要区分：** 共享梯度组成/方向与 optimizer history 各自对 d1 rescue 的贡献；任何解释必须同时容纳 d0 代价及 d2 late benefit。

### RQ2 — 目标函数条件效应中，域大小角色是否为因果原因？

**已知：** S2 的 Equal−SD 负值在三轮 MovieLens rotation 中随 Large 角色翻转，6/6 role-task 组合呈 L 负、M/S 正；难度解释未获支持。  
**未知：** 大小、份额、loss coefficient、梯度尺度、样本多样性和收敛速度共变，尚未分别操纵。S3/S4 聚焦自然 KuaiRand 最大域，不识别这些变量各自贡献。  
**为什么重要：** 不得将“Large cell 发生”写成“domain size 导致”，也不能把结论推广到所有失衡数据。  
**需要区分：** size role 与物理域身份、数据难度及有效目标权重对同一负迁移条件的解释力。

### RQ3 — 目标救援能否在不牺牲其他域的条件下成立？

**已知：** target-only 对 KuaiRand tab1 有大幅 AUC/LogLoss 改善，但 Equal 下 d0 明显退化；d2 的方向依赖训练阶段。  
**未知：** 目标域改善与非目标域代价是否有可重复的共同机制；何种目标和训练视图可避免纯粹的性能搬移。  
**为什么重要：** target-only 目前是诊断 intervention，不能据此提出安全方法或部署建议。  
**需要区分：** 局部 target benefit、全域宏观效果、worst-domain 效果与 AUC/LogLoss 校准代价。

### RQ4 — 结论可外推到哪些 backbone、数据和任务？

**已知：** P7 的组件 1 在 MMoE/PLE 有正向证据，ShareBottom 是自引入适配且复现严格性较低；P7 的 D=5 只测 T=2。S2–S4 的核心负迁移机制则主要在 M3OE 上识别。  
**未知：** Equal/tab1 的 S3/S4 因果效应是否跨 backbone、task 或自然域结构复现；现有结果不能填补此缺口。  
**为什么重要：** 防止把 M3OE、KuaiRand、Equal、tab1 的局部裁定写成一般多任务学习规律。  
**需要区分：** 新条件下是否重现同一目标函数 × shared-access 交互及相同的跨域代价。

### RQ5 — AUC 与 LogLoss、selected 与 matched-stage 证据如何共同描述该效应？

**已知：** S2 出现 AUC 与 LogLoss 方向不一致；MovieLens Large 的 selected 负迁移不少在后期缩小/反转，KuaiRand tab1 更稳定。  
**未知：** 目标域 rescue 的校准变化与排序变化是否由同一更新成分带来。  
**为什么重要：** 只报一个中心汇总或一个 selected checkpoint 会隐藏实质性代价或阶段依赖。  
**需要区分：** 选点效果与固定训练阶段效果，并报告目标域、非目标域、macro/worst-domain 及两个指标。

---

## 4. 实验协议与不可违反的不变量

### 4.1 先明确比较的实验对象

- P0–P7 的旧 baseline、as-published、修复 B10 的 fixedbug、Px_n 的 equal/sample、S 系列的 FULL/SD reference 是不同条件。写报告时必须把 objective、优化器参数分组、数据、checkpoint 规则写进 arm 定义；不能只用“baseline”一词。
- 原始 M3oE 训练的六项 BCE 是等权求和；后续 Px_n/E/S 对有效目标做过明确操纵。S2 的口径为 Sample：w_d=1、有效系数 a_d=p_d；Equal：w_d=1/(3p_d)、a_d=1/3，目标总体平均权重归一。P0-4/5 报告采用其自身的总尺度归一实现。跨实验不能仅凭“sample”同名就假定逐项损失和尺度完全相同。
- S1/S2 的 single-domain reference 是保留双任务的 D=1,T=2 参考，用来隔离跨域联合训练与独立单域训练的差异；不是 P5 中所有历史 oracle 的同义词。对负迁移只用 Joint−SD 或 Equal−SD / Sample−SD 判断，Equal−Sample 差本身不是负迁移定义。
- S3 的 Equal FULL/Sample FULL 与 target-only 是共享更新访问对照；S4 另含 raw-gradient norm 匹配 N 和 applied-step 匹配 U。S3 的历史 D=1 reference 仅作 legacy context，不是严格因果臂。不要把 target-only、N 或 U 当成 proposed method。

### 4.2 数据、域、任务与目标适用范围

- MovieLens-1M 早期三域按年龄：d0={1,18}、d1={25}、d2={35,45,50,56}；任务分别为 rating>3 和 rating>4。Px_n 固定 N=187,500 并轮换 L/M/S 的物理域身份；ratio、role、域名不可混为一谈。
- KuaiRand-Pure 本地 tab0/tab1/tab4 映射 d0/d1/d2；S/E 系列任务为 is_click / long_view。全量统计占比约 7.10%/89.18%/3.72%，S 系列 train split 实测约 8.6%/86.9%/4.5%；使用哪一组必须注明 split。数据按全局 (date,time_ms) 排序作 8:1:1 时序切分，是论文未公开切分规则下的重建方案，不是官方唯一管线。
- KuaiRand 有明显 train→test domain-mixture 与 label-prior shift；结论不得忽略该竞争解释。MovieLens 人工角色轮换与 KuaiRand natural setting 回答不同问题，不能把“域大小”合并成一个共同操纵变量。
- P7 MMoE/PLE 为 FuxiCTR 层移植；ShareBottom 是本地 4-expert+mean 适配，非逐字移植；STAR 复现失败并排除。P7 组件 1 泛化只到所测结构/ML 条件；D=5 证据只覆盖 T=2。S2–S4 机制结论为 M3OE 的窄范围发现。

### 4.3 配对、统计单位与模型选择

- seed 是重复与配对单位。配对比较按同 seed 对齐；同一实验的域、任务、epoch、probe 不得冒充独立 seed 增大 n。S 系列主要 n=4；旧 P5 有 n=3/n=5 混合，不能直接比较跨档 p 值。新实验若无经审查的其他预注册种子方案，使用 seeds 2022、2023、2024、2025。
- 以验证集作 checkpoint 选择；test 不可见。Selected、fixed-final 与 common epochs 是不同 estimand，分别报告，不可择其有利视图。S1 joint 用六 cell macro val AUC、reference 用双 task macro；S2 selected 用六 cell unweighted macro AUC 和最早并列规则。复用数据/代码须核对初始化、曝光、source identity 和 block validity。
- P5/P6 的 test_auc_mean pooled AUC 与六个 per-(d,t) AUC 再平均不同；P7 pooled 结果也不是 per-cell macro。每个表明确指标、聚合顺序、样本权重、ddof 和 paired n。
- AUC 衡量排序、LogLoss 衡量概率损失；两者不同向时必须保留差异。S2/S3/S4 应并报目标和非目标 cell、全局 macro/worst-domain、LogLoss 与 AUC、selected/final/common-stage。
- 显著性、实用阈值和证据强度分开表述。许多旧比较为名义 p 值且未过多重校正；不显著不等于等效，单次小样本不显著不得当成零效应。S2 的 AUC practical threshold 为 0.003，强效应 0.005；按各阶段预注册门槛，不事后变更。

### 4.4 优化器、RNG、实现与数据记录

- B10 是真实历史 bug：Python 的位取反被误作逻辑非，旧版主 Adam 包含全部参数，architecture scalars 又由第二个 Adam 更新。比较旧结果或复现时必须注明 grouping 版本；不得将含 bug 与修正版结果无标识混合。P2 显示均值影响不显著、方差明显变化。
- 任何额外 DataLoader iterator（即使 num_workers=0）会从全局 RNG 抽取 base seed，可能改变后续 shuffle。若做额外评估/诊断，应保存并恢复 torch RNG state，且验证训练轨迹未漂移。M3oE forward 返回 CPU prediction，调用侧需转到训练 device。KuaiRand 大数据使用 num_workers=0 以避免 Windows spawn pickle 失败。
- 保留训练目标、每步 loss、optimizer state 规则、数据 exposure、初始化映射与源代码身份。因果对照除预注册变量外锁定其余项；target access 的更改不得同时偷偷改变 private 参数或 loss scaling。
- 测试集 LogLoss 没有在 P5/P6/P7 主台账留存；不得从 P0/P2 或验证集 LogLoss 推算旧主矩阵的 test LogLoss。后续报告若声称 test LogLoss，须有本阶段直接评估记录。
- 报告至少保留逐 seed 原始结果、失败/重跑行、执行窗口、对照配对、manifest 与 reviewer 核查路径；failed 行不静默删除。重跑应依项目 AGENTS.md 的 DESIGN→IMPLEMENT→REVIEW→RUN→ANALYZE 顺序；当前证据不授权跳过独立设计/审查直接开跑。

---

### 4.5 项目执行角色

Main agent 只负责 ORCHESTRATOR 与最终科研分析，不直接实现研究算法。新算法实验依次由 designer 给出 DESIGN SPECIFICATION、luna-coder 按该规格实现、reviewer 独立审查、luna-runner 只记录原始结果，最后 main agent 独立分析。Reviewer 未给出 PASS 或 PASS WITH FIXES 且 blocking issues 未修复前不得 RUN。Runner 不作科研结论分析；所有阶段不得跳过。
## 5. 关键证据链

本章仅作溯源地图；当前结论见 F1–F18，不在此重复完整结果。

### 5.1 P0–P7：复现与早期算法候选

P0/P1 建立 M3oE 复现、代码事实和梯度诊断：baseline 可复现，但“小域低 SNR”设计前提失败（F1–F3）。P2 发现 B10 optimizer grouping bug，P3 重建 KuaiRand 数据切分，P4 实现三个候选组件并验证 all-off 等价；后续使用时须保留 B10/数据拆分边界（第4节）。

P5/P6/P7 先记录了 comp1 在 MovieLens 强失衡档的显著收益、KuaiRand 大域前沿移动、课程无稳定独立增益及有限 backbone 移植；同时暴露非单调 ratio、oracle 混合口径、多重检验、test LogLoss 缺失和不同 backbone fidelity。后续 P0-1–P0-5 与 S0 已重审其主要机制解释。今天仍可用的证据价值是：旧现象确实出现过、comp1 可产生增益、KuaiRand 大域负迁移量级很大；不能继续沿用“单调 size 因果”“真 Pareto”或“domain-conditioned mechanism 已证实”。来源：**P1_diagnosis.md、P2_baselines.md、ML_P5_REPORT.md、KUAI_P5_MAIN_REPORT.md、P6_ablation.md、P7_generalization.md、FINAL_RESEARCH_REPORT.md**。

### 5.2 P x_n：固定 N、域角色、objective 的归因修正

P0-1/2 在固定 N 下操纵大小角色与 imbalance ratio，支持效应位置随 Large 角色移动；P0-3 在 MMoE 复现方向，撤销 M3oE-specific 叙事。P0-4 发现 equal loss weighting 是 Large 牺牲的主要可操纵因素；P0-5 的 2×2 显示 comp1 的大部分旧增益是对等权错配的补偿，sample 下仍有小残余。P1-2 在 sample 目标下 DROP Curriculum。独立审计纠正了逐 seed 全正、完全消失、阈值即显著等过强表述。详见 F6–F8。来源：**P0_1_STAGE_REVIEW.md、P0_2_STAGE_REVIEW.md、P0_3_STAGE_REVIEW.md、P0_4_STAGE_REVIEW.md、P0_5_STAGE_REVIEW.md、P1_2_STAGE_REVIEW.md、research_diagnosis_20260909/**。

### 5.3 E 系列：机制诊断、跨数据集和动态方法淘汰

E1 支持但不确证 shared contribution distortion；E2 未执行，故 E3 响应面缺少系数与 sampling 的因果拆分。E3 不支持稳定优于 Sample 的 interior 权重；E4 在 KuaiRand 重现 Equal/Sample 方向，但 temporal shift 是残余的强竞争解释。E5 STOP 所测时间 schedule；E6 STOP 自适应校准；E7 因 gate 未授权，0 runs。详见 F9–F11。来源：**stage_E1_analysis.md、stage_E3_analysis.md、stage_E4_dataset_audit.md、stage_E4_analysis.md、stage_E5_go_no_go.md、stage_E6_go_no_go.md、stage_E7_gate_stop.md**。

### 5.4 S 系列：当前研究前沿

- **S0**：comp1 的约 +0.0054 残余可新鲜复现，但在 matched uniform-smoothing family 下失败 KEEP gate；STOP domain-conditioned Momentum。见 F12。
- **S1**：Sample+SD reference 下 STOP 普遍 residual NT/Safe-MTL 线；analysis 中“72/72 均正”和“候选在 weaker domain”两句已被审计纠正。见 F13。
- **S2**：Equal 目标下出现 Large 角色条件性 NT，Sample 在 selected 口径缓解；size 仍是关联因素，不是已隔离因果。见 F14–F15。
- **S3**：KuaiRand Equal/tab1 shared-access intervention 大幅救援 d1，但损害其他域；MovieLens selected replication 有 late-stage 反例。见 F16、F18。
- **S4**：fresh run 复现 S3；norm 与 applied-step magnitude 对照失败救援；证据收窄到梯度组成/方向及 Adam history 的联合候选，未拆分。见 F17–F18。

优先溯源：**stage_S0_analysis.md / stage_S0_review.md；stage_S1_analysis.md / stage_S1_review.md / stage_S1_paired_aggregates.csv；stage_S2_analysis.md / stage_S2_review.md / stage_S2_raw_results.csv / stage_S2_paired_results.csv；stage_S3_analysis.md / stage_S3_amendment_review.md / stage_S3_aggregates.csv；stage_S4_analysis.md / stage_S4_A2_review.md / stage_S4_A2_execution_report.md / stage_S4_A2_outputs/**。

---

## 6. 关键锚点数字

只保留会改变研究判断的锚点。除标明外，差值为 AUC 原始比例单位；跨报告的口径、paired n、指标不能互换。

| ID | 条件 | 指标 / 对比 | 核心数字 | 科学意义与边界 | 主要来源 |
|---|---|---|---:|---|---|
| A1 | MovieLens，as-published | 复现 AUC | 0.7686±0.0017；论文 0.7702 | baseline 复现成立；旧协议 | P0_repro.csv |
| A2 | P2，freeze 0.7311 | vs 原版 | +0.0007，p=0.32 | 旧 AutoML 无可测贡献 | P2_baselines.csv |
| A3 | P2，修 B10 | 5-seed SD | 约 0.0017→0.0002 | bug 主要放大 seed 方差 | P2_baselines.csv |
| A4 | P1，MovieLens | 小域/大域 SNR | 0.509/0.558；小 batch 差消失 | 小域更噪声大假设不成立 | p1/D04_grad_stats.csv |
| A5 | ML P5，ratio 30/100 | comp1−baseline | +0.0166 (p=.0002,d=5.80) / +0.0202 (p=.0061,d=2.37) | 旧强效应真实，但该下采样同时减少总 N | results/p5/P5_raw.csv |
| A6 | ML P5，六档 | comp1 差值序列 | +.0064, +.0034, −.0012, +.0053, +.0166, +.0202 | 不单调；3:1 为待复核负凹陷 | ML_P5_REPORT.md |
| A7 | KuaiRand P5，大域 local d1 | baseline Δoracle t0/t1 | −.070659/−.099016 | 严重负迁移基线 | P5_oracle.csv、P5_raw.csv |
| A8 | KuaiRand P5，comp1 vs baseline | 大域收窄 t0/t1 | +.009102 (p=.0228) / +.017077 (p=.369) | 只有 t0 名义显著；均未追平 oracle | KUAI_P5_MAIN_REPORT.md |
| A9 | P0-1，固定 N 角色轮换 | Large-role Δtransfer | −.014557±.007439 | SD 取自 P0-2 同配置；现象随角色移动 | P0_1 raw/review |
| A10 | P0-2，N 固定 ratio=50 | Large-role Δtransfer | −.015545±.010894 | ratio 效应在固定 N 下仍存在 | P0_2 raw/review |
| A11 | P0-4，Equal→Sample | Large loss damage 修复 | 约 92% | 特定分解描述值，不是中介率 | P0_4 headline/review |
| A12 | P0-5，Sample+comp1 | 残余 gain | +.005407±.000620；p=.00041；12/12 配对正 | sample 下还有小增量；69% 是描述性拆分 | P0_5_STAGE_REVIEW、audit_statistics.json |
| A13 | P1-2，forward curriculum | B0−A0 AUC | +.000839±.001018（seed-block n=4）；p=.198；CI 跨 0 | Curriculum 在 sample 口径未通过 KEEP | P1_2_STAGE_REVIEW |
| A14 | E1/E3，Equal→Sample，ML r50 | 六 cell selected AUC（12 blocks） | +.021085±.009768；E3 端点 bit-exact | 强目标差异证据；E1 机制本身仍观察性 | stage_E1/E3_summary.csv |
| A15 | E4，KuaiRand | train-test shift | mixture TV=.112839；d1 click rate .378→.211 | temporal shift 是强竞争解释 | stage_E4_dataset_audit.md |
| A16 | E5，预设 schedule | Forward 相对静态/controls | 对静态 Sample 仅 1/4 seed 正；GO gate 9 项 3 项过 | 所测 schedule STOP，不代表所有时间机制 | stage_E5_go_no_go.md |
| A17 | S0，Sample，comp1 vs C_val | selected macro AUC | −.003033±.000815；0/4 seed 正 | residual 真实，但非 domain-conditioned 独有 | stage_S0 paired/analysis |
| A18 | S1，Sample−SD reference | 组级结果 | 63/72 均值正；0/72 stable practical NT | 普遍 residual NT 主线 STOP | stage_S1_paired_aggregates.csv |
| A19 | S2，KuaiRand tab1 click | Equal−SD selected AUC | −.045006±.000602；4/4 seed 负；5/5 common epochs 负 | 最强条件性 NT 靶点 | stage_S2_paired_results.csv |
| A20 | S2，ML d2_large rating>3 | Equal−SD selected AUC | −.018093±.004687；4/4 seed 负 | 第二靶点；后期反转限制机制外推 | stage_S2_paired_results.csv |
| A21 | S2，全部目标条件 | selected practical NT | Equal 8/24；完整严格 gate 3/24；Sample 0/24 | objective 条件性；不是普遍 NT | stage_S2_analysis/paired_results |
| A22 | S3，KuaiRand Equal tab1 | target-only−FULL AUC | click +.038876±.003950；long-view +.039576±.004107 | 4/4 同向；selected 条件 | stage_S3_aggregates.csv |
| A23 | S3，同条件非目标域 | target-only−FULL AUC | −.017040 平均；14/16 为负 | rescue 是性能搬移风险，不是全局改进 | stage_S3_aggregates.csv |
| A24 | S4，KuaiRand d1 | norm臂 N−FULL | −.004048/−.004873 | raw norm match 未救回目标 | stage_S4_A2_outputs/stage_S4_aggregates.csv |
| A25 | S4，KuaiRand d1 | applied-step 臂 U−FULL | −.000587/−.000942 | 即时 Adam 步长 match 也未救回目标 | 同 A24 |
| A26 | S4，d1 target-only−N/U | selected AUC | +.042924/+.044448 click；+.039463/+.040517 long-view | 方向/组成或 moment history 更符合，但两者未分离 | stage_S4_A2_outputs/stage_S4_aggregates.csv |
| A27 | S4，非目标域 | d0 selected / d2 final | d0 −.031386/−.032142；d2 +.019810/+.036160 | 训练阶段和 domain-specific cost 必须报告 | 同 A24 |
| A28 | P7，D=3 backbone transfer | comp1−none | MMoE +.002978；PLE +.005538；ShareBottom +.004012 | 限 ML；ShareBottom 为本地适配；非 S3/S4 泛化证据 | P7_summary.csv、P7_generalization.md |

---

## 7. 结论演化与 Superseded Findings

| 早期认识 | 后续证据 | 当前裁定 |
|---|---|---|
| 小域梯度噪声更大，因此需用 per-domain EMA 补偿小域 | P1 SNR 在两 batch 条件下无稳定域差；P6 的 SNR 反推 γ 显著差于公式 γ | **SUPERSEDED — 不得用“小域噪声更大”作为当前机制前提。** 组件可能有效的历史事实与原动机需分开。 |
| comp1 在 MovieLens 强不均衡下收益巨大，收益随 ratio 增大 | 六档序列非单调；r30/r100 同时缩小 N；固定 N 的 Px_n 确认 imbalance 有贡献；P0-5 又发现大部分增益补偿 Equal mismatch | **PARTIALLY SUPERSEDED — 强档增益现象保留；单调性和单一 size 机制归因失效。** |
| comp1 是大域保护 / Pareto 改善方法 | KuaiRand 仅名义改善大域 t0，t1 不显著，仍远低于 oracle；固定 N 与 objective 对照改变解释 | **SUPERSEDED — 旧证据为前沿移动，不是真 Pareto 外推，也不是当前保留的方法结论。** |
| M3oE 特有的 Largest-domain sacrifice | 负值随 Large role 移动；MMoE 也复现；Equal/Sample 操纵改变方向 | **SUPERSEDED — 不能称为 M3oE 固有，也不能称为不依赖目标的普遍 trade-off。** |
| comp1 在修正目标后仍是独立的 domain-conditioned 机制 | P0-5 残余 +0.0054 新鲜复现；S0 对均匀平滑 family 的 KEEP gate 失败 | **SUPERSEDED — residual effect 可是真实，但 domain-conditioning 归因已 STOP；普通时序平滑足以解释。** |
| Sample 目标下 residual NT 候选证明 Safe-MTL 仍有广泛问题 | S1 以匹配 SD reference、backbone、数据与 common stage 检验，未达到任何 stable practical NT 门 | **SUPERSEDED — 普遍 residual NT 主线 STOP。** S1 的负候选不得脱离其阈值和 reference 方差单独引用。 |
| “negative transfer 在此任务不存在”可从 S1 推出 | S1 只用 Sample；S2 用相同 exposure 下的 Equal/Sample 对比发现 Equal 特定 Large cell 退化 | **SUPERSEDED — S1 结论限 Sample；S2 证明目标条件化现象，两者不是同条件矛盾。** |
| 负迁移来自域难度，或 Large 受损是普遍“救小牺牲大” | S2 effect 随 size role 关联，难度相关弱；Sample 使多个域同升；AUC/LogLoss 与 stage 有反例 | **DOWNGRADED — size role 是定位标签，独立 size 因果未识别；“小域受益、大域付出”不成立为普遍机制。** |
| Target-only rescue 证明 shared gradient conflict，target-only 可能是方法 | S3/S4 识别 shared-access intervention；非目标域明显受损；norm 与 applied-step 控制失败但 Adam history 未拆 | **PARTIALLY SUPPORTED / DIAGNOSTIC ONLY — access effect 有因果证据；纯几何冲突未证明，target-only 不可作安全方法。** |
| E3 的 early/late α 诊断表示 schedule 会优于 Sample | E5 预注册 forward/reverse/exposure-matched 对照未通过 GO gate | **SUPERSEDED — E3 是诊断性线索，所测 schedule 被 STOP；不得把早晚选择相关直接改写为有效时间干预。** |
| adaptive calibration、EMA 或 curriculum 可补足静态 Sample | P1-2、E5、E6 均未达到预设 KEEP/GO；E7 STOP、0 runs | **STOP / DROP — 这些当前路线不能重新包装成已验证方法或新贡献。** |

### 尚无最终机制裁定的主张

没有发现最新 S3/S4 与 Reviewer 裁定之间存在需要在两种结论中择一的未裁冲突。仍开放的是**因果不可辨识**而非相互矛盾：S4 的救援可由梯度方向/组成、其诱发的 Adam moment history 或二者共同作用解释；当前数据不能单独判定。域大小与份额/有效权重/多样性共变也未拆开。E3 late-phase 选择计数因 phase 切分口径出现 10/12 与 11/12 的差异，方向相近但精确计数应回源，不应把任一数字写成普遍事实。E2 未执行使“objective 系数 vs sampling exposure”仍缺因果拆分。

---

## 8. 历史实验入口 / AI Handoff（旧阶段快照）

### 8.1 当前停止在哪里？

**2026-10-09 更新：Stage S8（Optimizer/Trajectory Final Kill Test）已完成并 STOP。** S5–S7 的 optimizer-history / gradient-dynamics 主线以裁决 **B. LATE OVERFITTING RESCUE ONLY** 终结（见 F19 与 `experiments/reports/stage_S8_analysis.md`）。按预注册纪律，下一阶段唯一允许的研究问题是：Sample objective 在 extreme domain imbalance 下是否存在 selected-checkpoint joint-training performance gap（`Gap_{d,t} = Perf_joint − Perf_single/oracle`，ratio sweep，selected checkpoint，4 seeds，先证 performance problem 再做机制诊断）；不得设计任何 S9 gradient method。注意 S1/S2 目前没有已知的 selected-checkpoint 性能缺口，新阶段必须先证明 gap 存在。新阶段仍须按 DESIGN→IMPLEMENT→REVIEW→RUN→ANALYZE 工作流。

截至 2026-09-27，S4 A2 正式实验和 analysis 完成：16/16 runs 有效、4/4 paired seed blocks 有效，独立 review PASS；S3/S4 的核心 target contrast 已复核。S 系列已完成“普遍 residual NT 检验 → Equal 条件靶点定位 → shared access 干预 → 梯度幅度控制”的链条。当前停止点是机制方向/组成与 Adam history 尚未分离。S4 文档提出的下一步只是研究问题建议，不授权自动实现或运行；新阶段仍须按 DESIGN→IMPLEMENT→REVIEW→RUN→ANALYZE 工作流。

### 8.2 当前最重要的因果缺口是什么？

在固定 KuaiRand natural、M3OE、Equal、tab1 条件下，S3/S4 已证明改变 shared-gradient access 会改变目标域表现，并排除“仅改变 raw norm”或“仅匹配即时 applied-step magnitude”作为充分解释。仍不知道优势主要来自当步共享梯度的方向/组成，还是非目标梯度序列改变了 Adam 一阶/二阶状态和后续更新。任何解释还必须说明为什么 d0 代价大且稳定、d2 会在后期改善，并在 AUC 与 LogLoss 两个指标上检查。

### 8.3 下一阶段不能重新做什么？

- 不要重新证明小域 SNR 必然更差，或以此作为 domain-conditioned momentum 的设计理由。
- 不要把 domain-conditioned EMA、uniform smoothing、Adam β1、Curriculum、已测线性 schedule、E6 adaptive calibration 改名后重启为“新方法”；其状态见第2节。
- 不要把普遍 residual NT / Safe-MTL 当作已成立问题；S1 在 Sample 口径已 STOP。若研究 Equal 的条件性 NT，必须保留 Equal 与 Sample 的目标对照和 SD reference 定义。
- 不要把 M3oE-specific sacrifice、size-alone causality、gradient conflict、target-only 全局改善、Δoracle≥0 或真 Pareto 外推写成已知事实。
- 不要重跑 E7 原提案：其 gate 为 NOT AUTHORIZED、runs=0。若主张复议，须满足 E7 文档规定的正式证据纠错路径，而不是改变名称或引用次要指标。
- 不要将 P5/P6/P7 缺失的 test LogLoss 从其他阶段推算；不要混用 pooled/macro、selected/final/common epoch、不同 N、优化器分组或不同 seed 配对。

### 8.4 下一阶段应回答什么类型的问题？

应是**辨别机制的因果拆分问题**：在不改变 S4 目标域与训练条件的前提下，判断每步 shared-gradient direction/composition 与 optimizer moment history 哪一项对 tab1 rescue 必需、各自贡献多大；同时把 d0 损失、d2 late-stage 行为、LogLoss、AUC 和 selected/common/final 视图作为共同结果。设计须有能区分竞争解释的预注册对照与可审计状态记录；本 handoff 不规定具体算法实现，也不授权运行。size 与目标权重是否可独立因果识别，是并列但次级的开放问题，不应和本机制拆分混成同一 claim。

### 8.5 当前创新点状态

- **可靠发现：** Equal 目标下存在条件性、大域 cell 负迁移；Sample 可在已测条件下消除 selected practical 阈值上的退化。KuaiRand Equal/tab1 的 shared-access rescue 在 S3/S4 重复，norm magnitude 两类控制未重现，且带明确的其他域代价。
- **机制价值但未形成安全方法：** shared-update access 与目标表现有因果联系；现有证据尚未锁定纯方向机制，也不能将 target-only 用作方法。对 optimizer history 的进一步拆分才可能形成机制贡献。
- **失效的早期方法候选：** domain-conditioned momentum 作为独立机制、domain-conditioned curriculum、已测时间 schedule、adaptive contribution calibration / EMA；均按 STOP/DROP 处理。comp1 的历史性能现象保留，原独立机制解释不保留。
- **仍需验证才能成为论文贡献：** S3/S4 局部效应的机制拆分；跨条件/模型复现；避免目标救援转移成其他域损失的证据。当前不可声称通用 Pareto、安全 MTL 或通用梯度冲突方法贡献。

---

## 9. 溯源索引

读取顺序建议：先读最新签署的 Reviewer 与 analysis，再查 paired/aggregate CSV，最后查 run-level 原始结果和日志。表中报告路径相对仓库根目录。

| 结论 / 问题 | 最权威源报告 | 原始数据 / CSV / 日志 |
|---|---|---|
| P0 复现、B10、梯度 SNR | experiments/reports/P1_diagnosis.md；experiments/reports/P2_baselines.md | experiments/results/P0_repro.csv；experiments/results/p2/P2_baselines.csv；experiments/results/p1/D04_grad_stats.csv；P0/P1/P2 run logs |
| P5 MovieLens comp1、ratio 非单调、oracle 混合口径 | experiments/reports/ML_P5_REPORT.md（Reviewer 签署） | experiments/results/p5/P5_raw.csv；experiments/results/p5/P5_oracle.csv |
| P5 KuaiRand 大域负迁移 / 前沿移动 | experiments/reports/KUAI_STAGE1_ORACLE.md；experiments/reports/KUAI_P5_MAIN_REPORT.md；experiments/reports/FINAL_RESEARCH_REPORT.md | experiments/results/p5/P5_raw.csv；experiments/results/p5/P5_oracle.csv；对应 kuairand run logs |
| Px_n 固定 N、role、weighting 归因 | experiments/reports/P0_1_STAGE_REVIEW.md 至 P0_5_STAGE_REVIEW.md；research diagnosis 独立审计 | experiments/P0_1_role_reversal/P0_1_raw_dedup.csv；experiments/P0_2_ratio_sweep/P0_2_summary_ratio.csv；experiments/P0_4_loss_weighting/；experiments/P0_5_comp1_x_weighting/；experiments/reports/research_diagnosis_20260909/audit_statistics.json |
| Curriculum 生死判定 | experiments/reports/P1_2_STAGE_REVIEW.md | experiments/P1_2_curriculum/P1_2_paired.csv、peak_epoch.csv、grad_norm.csv、rho_sanity.csv |
| E1/E3 contribution 与 response surface，E2 缺位 | experiments/reports/stage_E1_analysis.md；experiments/reports/stage_E3_analysis.md | experiments/stage_E1_* gradient_step/summary files；experiments/stage_E3_summary.csv；E1/E3 review reports |
| KuaiRand objective、temporal shift、Sample 残余 | experiments/reports/stage_E4_dataset_audit.md；experiments/reports/stage_E4_analysis.md | experiments/stage_E4_* raw/paired/summary files；dataset split and audit manifest |
| E5/E6 STOP，E7 未运行 | experiments/reports/stage_E5_go_no_go.md；stage_E6_go_no_go.md；stage_E7_gate_stop.md | experiments/stage_E5_*；experiments/stage_E6_*；E7 gate record (no run matrix) |
| Momentum matched-control 裁定 | experiments/reports/stage_S0_review.md；stage_S0_analysis.md | experiments/stage_S0_* paired/aggregates/subgroup_gains/summary tables |
| Sample 条件下普遍 residual NT STOP | experiments/reports/stage_S1_review.md；stage_S1_analysis.md | experiments/stage_S1_paired_aggregates.csv；stable_cells、reference_baseline_variation、common_epoch summaries；run manifest |
| Equal vs Sample 条件性 NT | experiments/reports/stage_S2_review.md；stage_S2_analysis.md | experiments/stage_S2_raw_results.csv；stage_S2_paired_results.csv；stage_S2_* training-stage/rotation summaries |
| Shared-access causal effect 与 source identity | experiments/reports/stage_S3_review.md；stage_S3_amendment_review.md；stage_S3_analysis.md | experiments/stage_S3_aggregates.csv；stage_S3_interactions.csv；stage_S3_paired_results.csv；summary manifest |
| S4 norm/applied-step controls 与 frontier | experiments/reports/stage_S4_A2_review.md；stage_S4_A2_execution_report.md；stage_S4_analysis.md | experiments/stage_S4_A2_outputs/stage_S4_aggregates.csv；stage_S4_paired_results.csv；stage_S4_gradient_steps.csv；stage_S4_probe.csv；stage_S4_summary_manifest.json |
| Backbone/D/T 的历史泛化边界 | experiments/reports/P7_generalization.md（含 Reviewer 裁定） | experiments/results/p7/P7_backbones.csv；experiments/results/P7_summary.csv |

> 源文件若与本摘要数字或表述冲突，应先追到对应 reviewer 裁定与 CSV；若没有正式裁定，保留 OPEN 状态并记录冲突来源。本文档不授权新实验，也不取代完整档案中的逐项审计证据。


