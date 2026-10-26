# MDMT 跨阶段实验总报告：时间线、证据修订与研究状态

> **汇总截止：2026-10-26；性质：main 中的跨阶段科研事实与结论索引。** 本文整合新归档的实验分支与最新研究方案，不修改各阶段原始报告，也不把方案写成已运行实验。原始数值、配置和源码以归档文件为准。此前 [2026-10-18 综合报告](Experimental_Branch_Synthesis_20261018.md)是截至当时的历史快照，已不是当前结论入口。

## 1. 研究主线、证据身份与解释纪律

长期主线是 **MDMT推荐中的样本重加权、优化响应与差异化共享**：研究不同域/任务/目标群体的真实知识收益、局部共享需要及负迁移。Equal/Sample是某些实验中的具体损失聚合口径，不能替代整个长期研究主题。

- **分清执行状态：** COMPLETED表示已触发的规定工作完成；NO_START/NOT_TRIGGERED表示前置条件没有通过，不是运行后的阴性；PARTIAL/IN_PROGRESS只能使用已经核验的子集；DESIGN_ONLY不能宣称算法性能。
- **分清四层证据：** 数据/代码身份、训练与预测观测、可识别的受控干预、尚待验证的解释或技术假设。非零梯度、非恒定权重、可生长原型和激活路径本身不等于独立方法增益。
- **分清估计对象：** 自然/人工训练人口、目标与采样q、参数图及优化器、训练时钟、selected与最终epoch、六格macro与按人口加权cell AUC、pooled AUC、LogLoss及测试/验证/Q角色，均不得擅自交换。
- [history_experiments](../history_experiments/README.md)保存各原分支相对归档前main不同的Git文件对象与相对路径，不是每个分支独立完整工作树；引用阶段事实时核对原提交SHA和原报告，不跨阶段相加共享训练/前缀。

## 2. 贯穿2026年9—10月的实验时间线

| 时间/轮次 | 原始报告或方案 | 已取得结果/当前状态 | 不能外推的部分 |
|---|---|---|---|
| 9月—10-09 P0–P7、E、S0–S8/S6-R2 | [历史索引](Experimental_Experiment_Index.md)、[旧证据](Experimental_Evidence_Ledger.md) | 样本聚合和共享更新权限的条件效应；Equal下S5特定Adam history承载；S8 late rescue不提高合法selected best | 不同客观目标/协议的结果不可合并成普遍“Sample优于Equal/Adam是根因” |
| 10-15—16 B1/C2与Stage2D | [B1与审计](../history_experiments/codex__execute-2026-10-16-v2/research_plans/2026-10-15/execution_20261015/README.md) | B1重建8/8完成，但与旧S6-R2初始化谱系不同；Stage2D 16/16新训练完成，覆盖早期“零正式训练”的**过程快照** | 旧S6-R2非目标增益与新B1不能当同协议互相推翻 |
| 10-16—17 v2/v3 | [v3终报](../history_experiments/codex__execute-2026-10-16-v3/research_plans/2026-10-16_v3/execution_20261016/FINAL_REPORT.md) | v2固定fusion的无动量SGD仍有采样效应，驳斥Adam普遍必要性；v3规定V0–V14及触发项完成、当前方法NO_GO | v2保留中期范围；v3与v2为不同Git实验谱系，不互称完整复现 |
| 10-17—18 N17 | [FINAL_ANALYSIS](../history_experiments/codex__execute-2026-10-17-run/research_plans/2026-10-17/execution_20261017/FINAL_ANALYSIS.md) | 120条新H100、72条唯一formal。KR Adam selected SB−SN宏AUC −0.005203（Holm p=.171875）、SGD +0.000837（p=.6953125），两个预定主比较未确认；ML ownLR E100 +0.021145而selected −0.010875 | ML固定人为subset，LR/时钟和选点不可混称纯q效果；N4未触发 |
| 10-18 N18 P1–P6 | [六方向首轮](../history_experiments/codex__execute-2026-10-18/research_plans/2026-10-18/execution/FINAL_REPORT.md) | 数学反例、数据与预算审计、四种子冻结头及有界诊断完成；尚无六方向中稳定且具独立增量的方法候选 | DATA-BLOCKED不代表科学方向已失败；非触发长期训练不得记作阴性训练 |
| 10-18—19 N19 | [N19完整终报](../history_experiments/codex__execute-n19-2026-10-18/research_plans/2026-10-18/N19/FINAL_REPORT.md) | 328条完整正式网络训练及冻结测试；报告容量路径数值弱化、通道误设、指标/阶段反转和受控机会；未支持通用新方法贡献 | MovieLens此轮主要是人工固定subset，不是自然full人口 |
| 10-19—20 CM19 | [共享路径终报](../history_experiments/codex__conditional-paths-2026-10-19/research_plans/2026-10-19/execution/FINAL_REPORT.md) | 856次全预算训练及对应冻结测试。KR fixed M3oE某个输入乘性调制路径selected 24/24 cell数值中性，双factor耦合Adam-L2或局部AdamW干预能恢复路径响应，selected泛化收益未稳定建立 | 不等于全部共享参数失活；独立STAR、PEPNet、HAMUR及MMoE/PLE相关路径仍活跃；恢复活性不等于模型必优 |
| 10-20 自然MovieLens历史复核 | [人口审计](../history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/DATA_PROTOCOL_AUDIT.md)、[已完成子项重评](../history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/RESEARCH_CONCLUSION_REASSESSMENT.md) | 审计确认多项10-15以来ML实验为人为187500行/34500:3000:150000，自然base-train为800167行/155949:318599:325619；已完成V2/S5/S6/AO04、N17和若干局部机制比较，原部分selected/late结论须限制人口 | 该分支 [最后状态](../history_experiments/codex__natural-movielens-reassessment-2026-10-20/research_plans/2026-10-20_natural_movielens_reassessment/CURRENT_WORK.md)仍IN_PROGRESS，**N19/CM19全套自然重训未完成**，不得当作完整复现 |
| 10-21 原生M3oE、数据生成/重加权、调幅 | [48次正式训练](../history_experiments/codex__m3oe-reproduction-2026-10-21/experiments/reproduction_20261021/REPORT.md) | ML D−B macro AUC +0.006303、LL −0.015962（4/4有利）；Dconst01−B AUC +0.005676（4/4有利），动态样本specificity不是获得套件收益的必要条件；匹配路径G−Gtau0 AUC ML −0.003466、KR −0.006484（均4/4负，KR区间跨0） | D含权重、复制、噪声、总尺度；原生O是受控移植；目标为legacy_group_sum，不能与标准Sample目标直接混比 |
| 10-24 I1权限 | [阶段报告](../history_experiments/10-24/research_reports/2026-10-24/README.md) | 44主pilot、4基线适配pilot、40正式训练；I1_PERM比固定RW的ML/KR selected test宏AUC −0.000334/−0.000513，未过主优势 | KR final LL次要改善不替代selected主结论 |
| 10-25 MDMT-NEXT-v2 | [v2报告](../history_experiments/26-10-25/research_reports/2026-10-25/README.md) | 48条24epoch核心轨迹完成；晋级门均未过，正式四seed/50epoch为NO_START；部分aux/基线预算停止 | 不把未触发的正式训练当作实际阴性实验 |
| 10-25 MDMT-NEXT-v3（RHD/LPS） | [v3完整报告](../history_experiments/2026-10-25_15-56-04/research_reports/2026-10-25_v3/FINAL_REPORT.md) | 69条24epoch逻辑轨迹＋20条KR正式50epoch。RHD HB−BASE：selected test LL +0.00074075（差），epoch50 −0.00875428（好）；LPS部分foreign可读取，但特定形成器未稳定超过简单控制 | late减损≠最佳selected改善；测试和旧确认Q历史已暴露 |
| 10-26 DFD-v4 | [完整执行报告](../history_experiments/26-10-26/research_reports/2026-10-26_v4/FINAL_REPORT.md) | ML P1需求选择有限优于常量（CONST−DEMAND LL +0.00108935）；KR P1未过。ML P2 DFD−H_DIRECT/+0.00163966、DFD−ETA_DIRECT/+0.00181402，完整方法不支持；184份冻结预测导出、12项推理测量完成 | KR P2、P3/P4按门NO_START；原Q/outer refitη固定μ分组缺失MAJOR，不能称所有机制收据齐全 |
| 10-26 研究重设计RCF | [RCF-v1方案](../research_plans/2026-10-26-v2/00_README.md)、[事实审查](../research_plans/2026-10-26-v2/01_实验事实与科学问题.md) | **DESIGN_ONLY**：在来源行为保留/预算约束下，让目标群体剩余风险指导来源后继残差函数形成；配强直接、固定来源读、统一微调与target-only对照 | 没有新的RCF训练或结果；不能将新方向的理论计划写成已被实验验证 |

> 各行训练数属于不同实验定义。祖先/后继分支、复用的前缀与预测、相关cell及fold均不能被当成独立新训练或独立样本数。

## 3. 早期说法应如何更正

| 早期或过强说法 | 目前可使用的最强陈述 | 状态 |
|---|---|---|
| “10-15后MovieLens都是自然比例，结论可外推原人口” | 多轮用了明确人为角色采样，旧实验仍有效但只是条件结论；自然全量实验同时改变N、支持、相对比例和时钟，不能归因单一规模变量 | **修订人口前提** |
| “Equal普遍伤大域，Sample在所有场景更好并消除共享损伤” | Equal/Sample是不同优化目标；S_n指定配置中Sample缓解大域损害是真实条件结果，不能上升为普遍目标优劣，也不能代替长期样本重加权主线 | **局部保留，普遍性撤回** |
| “Adam历史是所有采样/共享效应根因”或“Sample使Adam历史完全没用” | 无动量SGD仍出现q效应，故Adam**不具通用必要性**；历史Equal S5的特定persistent history承载仍成立，不能由SGD反例把它整体删掉 | **必要性被驳，局部中介开放** |
| “小域=困难域”或“反复推翻小域=困难域是新贡献” | 这原本就不是研究者认可的前提；规模、物理域身份、标签/特征支持、基准性能不等同，仍未识别域规模独立因果 | **避免虚构起点** |
| “仅大域更新可稳定提高小域/共享路径恢复即收益” | S6-R2阳性、B1及自然复评异向且不同谱系；CM19恢复活性并未一致改善selected指标，其他域/任务可能承担代价 | **缩限且保留冲突** |
| “匹配梯度范数就可复现共享救援” | S4 raw/applied范数匹配均不足；10-21调幅匹配路径在本协议下AUC负向，不能推广为所有梯度调节无价值 | **通用机制未支持** |
| “动态权重D获得增益，就证明样本级区分有效” | ML D、Dconst01相对B均有益，D权重很快接近下界；常数方案尚未被证明不劣于D，复制/噪声/总尺度也未独立拆分 | **性能事实保留；归因未建立** |
| “原型/历史控制非零说明有共享功能分工” | N18/N19、10-24、RHD/LPS及DFD仅对各自当前版本给出NO_GO或有限边界；foreign可读与功能需要形成是不同命题。RCF仍是待验证方案 | **版本NO_GO，不是方向否定** |
| “最终epoch相对好就说明最优可部署模型更好” | S8、N17、D/G、RHD的selected/final/AUC/LL/角色可反号，需优先合法选点和逐格保护 | **评价口径修正** |
| “10-16仍没有正式实验”或“10-20自然复核已全数闭环” | Stage2D后续16/16完成；10-20自然矩阵有大量已完成子项但总项目标记IN_PROGRESS，N19/CM19自然正式复制不可虚报 | **按最终状态分别保留** |

**数据角色补充：** 10-25 v3 / 10-26 DFD在自然base-train内另外分割fit及P_form/P_read/P_cal等角色；DFD的ML/KR fit分别为640133/1604904，不等于整个自然base-train的800167/2006132。原始M3oE的learned-fusion双Adam与fixed-Sample单Adam也不是可直接相减的模型。

## 4. 当前可信的综合科研认识

1. **样本重加权/梯度/优化：** 它们的作用确实可能随目标、population、优化器、学习率和训练阶段而变化；当前尚无证据支持一个通用的“增大小域权重”或“动量修复”的安全算法。10-21套件的正结果表明仍有可深入的尺度、增强与权重交互，但不能把套件收益直接归因某个动态权重函数。
2. **共享机制：** 部分M3oE乘性路径可在特定条件下数值中性，存在设计/优化器相关的恢复；并非所有结构都会失活，也不意味着激活恢复就能提升性能。普通foreign读取的有限价值与特定来源功能形成的必要性需分开判定。
3. **局部需求驱动共享：** ML中目标条件选读相对常量有信号，KR未过，完整DFD未超过强直接控制；需求外推及正权形成机制缺口已清楚，因而可以保留“目标真实剩余需求—来源可用函数”的问题，并用RCF的强对照检验新增价值，而不是宣称其已有效。
4. **评价与科研证据：** 逐格AUC/LL、宏与不同部署人口、selected与末态、validation与历史exposed-test必须并存。当前大部分正式测试沿用已暴露的人口，独立新时间/真实样本支持才有助于升级外部效度。

## 5. 后续最有判别力的工作（不是已执行结果）

主问题是**强目标直接头已可预测后，来源行为监督形成的函数是否仍提供有限样本/容量/预算下的独立增量**。按照 [RCF-v1规范](../research_plans/2026-10-26-v2/00_README.md)先比较完整来源读取、H_DIRECT/ETA_DIRECT、目标专有/统一微调和固定来源，再决定是否值得目标群体组织多个来源后继残差；同时设置来源保持约束并检查真实成本。RCF尚未运行。

样本权重支线保留最小对照：10-21建议固定loss尺度0.1且关闭复制与噪声，以拆分全局scale与增强的收益，再判断动态specificity是否有必要；不要用未运行的“理论有用”填实验结论。

## 6. 阅读顺序与历史保护

1. 本文件及 [Global Summary](Experimental_Global_Summary.md)是截至2026-10-26的跨轮入口；[Experiment Index](Experimental_Experiment_Index.md)标记最新阶段；[Evidence Ledger](Experimental_Evidence_Ledger.md)记录证据边界。
2. [Stable Findings](Experimental_Stable_Findings.md)、[Contradiction Register](Experimental_Contradiction_Register.md)、[Hypothesis History](Experimental_Hypothesis_History.md)、[Open Questions](Experimental_Open_Questions.md)、[Dead Ends](Experimental_Dead_Ends.md)保留历史编号并标出后期限制。
3. [2026-10-18旧综合](Experimental_Branch_Synthesis_20261018.md)及[旧Q1–Q53](Experimental_Question_Disposition_20261018.json)是日期固定的历史记录。新证据不可回填改写它们。
4. 全部阶段原文、不同实验分支的代码与表格在 [history_experiments](../history_experiments/README.md)中；归档内容和各阶段原始报告不做编辑。
