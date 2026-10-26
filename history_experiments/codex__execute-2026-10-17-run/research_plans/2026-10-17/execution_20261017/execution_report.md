# 实验执行与交付报告

**状态：COMPLETE。** 研究规划规定及已触发的训练、核验、原始读取、分析、图表、问题终态与归档已完成。最终发布commit见Git HEAD及交付消息；本报告的冻结起点为889e012。

## 入口、工作区和流程

按start.md（GB18030，另存UTF8）及research_plans/2026-10-17实际规划执行。start.md §11要求所有非机械任务由主agent完成，故各实验DESIGN→IMPLEMENT→主agent REVIEW→机械RUN→主agent ANALYZE完整保留，independent_review=false；不伪称独立review。原工作区Git无HEAD/remote且全未跟踪，保留；实际独立checkout /home/xzf/Code/phase0-3-20261017-git，分支codex/execute-2026-10-17-run；计划来自main@a56639ab，研究源底座2d04c6de。原路径的执行输出有链接。

Python /home/xzf/miniconda3/envs/phase0-3-env/bin/python，3.13.5、torch2.9+cu126、两张3090。复用原缓存及处理人口；线程OMP2/BLAS1，队列每卡总compute PID≤5、显存准入<90%，保留当时外部任务。模型/预测/原始trace在/home_new/xzf_phase0_3_n17_artifacts_20261017，Git收代码、表、图、manifest和SHA索引。未安装新环境或重下数据。

## 完成矩阵

|阶段|实际完成|独立重复与说明|
|---|---|---|
|N0|历史4068行评价分解、672校准拟合、用户簇条件读数|无新正式训练；旧四seed不变，部分E100仅n1–2|
|N6训练前|八篇直接原文公式/干预对照|有限检索，不宣称穷尽|
|N1 tuning|32×H100|2030/31，双q四LR两optimizer，未早期淘汰|
|N1 formal|32×H100|2032–39，common/own同赢家复用|
|N2固定点|16有效锚×4臂×512draw|四训练seed、双路径E10/E40；原14成功+2验证修正重试|
|N2 quota|16×H100|同8seed，IID16父复用，4臂总比较|
|N3|开发16、SGD整体C16、独立确认8，各32draw|一次冻结embedding/E10；无64补充|
|N5-C tuning|16×H100|固定subset2022、2030/31对称网格|
|N5-C formal|24×H100|8seed×SN.001/SB.001/SB.0003，去重SN|
|统一test|72 unique×100model|7200预测，43200有限cell行；48 tuning排除|

总计120条新H100训练轨迹、0训练失败；同一轨迹别名不是额外独立重复。N1每条4900steps/200613200examples，ML每条500/18750000。FINAL_TRAINING_RUN_LEDGER保存每条真实配置、选轮、GPU/PID、起止、code commit、source/row/config SHA及时间；约KR32分钟/条、ML2.6–2.7分钟/条为共享GPU条件的实际wall均值，非独占吞吐或纯算法效率比较。

## 条件分支闭合

H200：四q/optimizer平均rho均<.005，未触发；不是证明已平台。N4：独立局部实际/方向门槛不满足，未触发，不能当拓扑阴性。N3四项MCSE达标，不补64、不重选。N5-A/B/D/shape/交叉subset未承担本轮主张，test前已固定不触发并保留相应识别限制；N5-C已触发且完成。无新方法候选训练。所有依据见CONDITIONAL_BRANCH_DECISIONS_PRETEST、PRETEST_TRAINING_STAGE_CLOSURE、原gate及授权，未按test反向更改。

## 异常、最小修正和交互恢复

两原N2 SN/E40 seed2033/2034在跨batch形状CUDA float32身份检查失败。CPU/同图/float64最小复核支持数值差异；隔离n17_geometry_identity保持原估计器逐字等价，只改验证/记录，同seed同checkpoint同draw重试。原FAILED、日志、配置保留，resolved16有效；未修改健康训练源码，未重跑14个无影响锚。细节及具体误差见N2_IDENTITY_FAILURE_DIAGNOSIS和N2_validation_only_equivalence。

机械协调曾出现missing status被误作0PENDING、以及一次派生投影误写retry外层元数据；均记录并修正。后者恢复queue PID、逐task记录未变，原外层terminal写入时间不能恢复；不掩盖此限制。仅scheduler短暂暂停准入，健康计算子进程保留。后续只在真实旧PENDING0且任务集/PID核对后准入，不增加协调框架。

交互中断期间持久主队列完成quota，实际最后完成2026-10-17 17:26:14UTC；用户于10-18要求恢复后核对终态，从冻结候选接续，未重复训练。新独立N3→MLtuning→MLformal→全局冻结→test真实完成。RESUME_20261018及各launch/status/机械观察保留精确记录。

## 冻结、核验、分析与交付

训练源与原生状态在运行期间冻结。全局test前提交889e012，56个比较=7组×8seed、72唯一formal、7200纯model；冻结config/trajectory/选轮/模型与reader/analysis源码SHA。test reader逐模型验SHA、逐预测存SHA/行序索引；TEST_READOUT_VALIDATION核对72完整receipt、模型join、600cell行/轨迹、shape/dtype及人口身份，未重复全量二进制oracle。

主agent依冻结入口实际运行performance、quota四臂、两套method guards、domain context和figures，PNG均视觉查看；只做影响定义、模型、指标和结论的必要核验，没有额外可靠性矩阵。完整表/置信界/原seed/反例均归档，Q53记录精度不足，不扩seed。

最终报告：FINAL_ANALYSIS、experiment_results、mechanism_analysis、METRIC_AND_POPULATION_DECOMPOSITION、Q_LR_BUDGET_RESPONSE、DOMAIN_COUNT_COUNTERFACTUAL、C_LOCAL_AND_TOPOLOGY_ANALYSIS、LITERATURE_COVERAGE；问题终态QUESTION_REGISTRY_FINAL；数值FINAL_NUMERIC_FINDINGS及CSV，索引REPORT_INDEX与FINAL_DELIVERY_MANIFEST。原设计/阶段授权保留签署时状态，最终执行状态以本报告为准。

## 注册表字节来源说明

REGISTRY_PRESERVATION_CHECK实测固定Git blob SHA为c63bb2750fc6a15374e7fd22f93ab7a65be942d8c03dee90b10452f4a1387f19；计划提供的original_registry_sha256为efe478…，两者不符，原因/哈希约定未知。原声明保留；46个解码问题对象及全部字段与固定Git来源逐项完全相等。该元数据差异不改变科学字段、模型或结果，记录后不阻塞/重跑实验。
