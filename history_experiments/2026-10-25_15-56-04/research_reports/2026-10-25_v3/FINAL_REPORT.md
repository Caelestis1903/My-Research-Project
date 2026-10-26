# MDMT-NEXT-v3 完整实验总报告

**结论：预定完整实验已完成，当前主目标不支持将RHD-PB_HB或LPS-BF作为稳定优于强简单参照的新方法推广。RHD存在可重复的晚期相对风险减损，应与selected主目标分开保留。**

## 研究问题与评价边界

本轮检验两个可被反驳的主张：付费半epoch响应探测及当前验收能否超越简单控制；跨目标风险形成的来源块责任能否超越uniform/scalar元重权及无foreign预测修正。固定fusion M3oE为共同主架构，MovieLens-1M与KuaiRand-Pure均保持原自然比例、fit-only编码器和3域×2个Bernoulli任务。

主训练batch40960，保留末批，Adam lr=.001、betas=(.9,.999)、eps=1e-8、coupled weight_decay=1e-5；FP32、无AMP/TF32。开发seed3101/3102固定24epoch。RHD按validation macroLL、LPS及独立本地头按validation macroAUC选点，eligible≥12，精确平局最早；共同终点与次指标另列。没有从测试结果挑选checkpoint。

Q_explore是开发性评价；benchmark_test_exposed与Q_legacy_confirm_exposed均为历史已暴露样本。KR四新seed固定50epoch是锁定方法后的训练随机性复评，不是新的未见数据确认。seed是统计单位，n=2与n=4分别计算配对差、样本SD、方向数与Student t95%区间，不将task/epoch当成独立重复。

## 完成范围与证据链

本版完成89条逻辑轨迹：52核心P1、4旧aux与6外部方法补全、7条件P2，共69条完整24epoch；另20条KR新seed正式轨迹完整50epoch。12条物理共享warmup只在研究实际账计一次，独立部署补回必要前缀。L0另有8个独立MONO/FEATURE预测器，不计成新增主训练。

每个预测器均保留合法selected和共同终点：Q_explore共194行、1164个六格记录；两历史暴露角色共178个run-role任务、388个predictor-checkpoint-role结果。全部97个predictor的194个AUC/LL-best验证别名与实际状态epoch一致，旧schema缺少顶层epoch者保留null。51/51统一推理测量成功，没有未跑完的已触发轨迹。

正式锁SHA256为`697c9f3250986d7687a26a21e1f4bc57b4a8077c62e1ff5208537d91d96c934b`，在任何正式进程前冻结。全部终点核验后的最终方法冻结于2026-10-25T12:02:26.282158Z，SHA256 `8a0698db50ed4d129909b9decdd400e1ee8db89cfa142591f942282f9405e382`；随后才评分旧test/Q_confirm，配置未据其变化。未触发的ML-RHD正式、LPS正式及STATE/C14/PAIR为规定NO_START/NO_TRIGGER，并非未完成。

## 历史机制复核与实现身份

P0从48条旧核心轨迹的selected/24共96份预测重新计算六格与macro，最大指标差为0；旧源码归档逐项hash匹配。旧HIST/ridge/NN与STATIC的24个配对快照在模型、Adam、梯度和RNG上相同，拒绝链的28次控制提案均被拒绝；非零gamma不等于产生了实际控制。对这一具体旧失败模式的证据STRONG，不能外推为历史信息普遍无效。

十个可重建旧树快照都生成了六棵候选树，selected深度均0，八个快照的深度2原始P_cal风险更低。事实指向选择/收缩链可能抑制已有候选，不能称“树从未长出来”；缺失候选叶表使更细归因受限。规定的两个ML公共续训锚点恰好比较STATIC与STATIC，共32个物理更新的零差是退化对比，不能证实或否定效用反转，也不更换锚点追求阳性。

初始及条件实现都经过设计、专门编码、独立审查和机械运行；阻塞问题在对应实验前修复，没有以看过质量后的算法改动替代原计划。基础71文件源码SHA256为`3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2`，条件/正式81文件为`31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276`。全部源归档、数据身份、父状态证书和修订记录保留，报告公开范围不含权重或逐行预测。

## 开发发现与有限分支

RHD的KR两开发seed相对BASE，selected Q_explore macroLL差分别−.00224545/−.00182366，均值−.00203456；对H1均值−.00224416。ML相对BASE均值+.00073460，对STATIC两个seed均稍差，不能声称跨数据通用。只在KR进入必要PB_HISTORY归因，继而冻结五方法×四新seed的50epoch包。单seedHISTORY对照提供候选证据，正式独立增量由完整四seed比较判断。

LPS的L4对L2、L3分别达到有限开发归因门槛，但ML两个seed均未胜FEATURE，KR对L2一个seed为负；正式晋级条件不满足。六条CAL/OWNER/GEOM条件轨迹完整24epoch，不将其中赢家替换主L4。LPS50明确NO_START，STATE/C14/PAIR没有触发，不通过无限追加实验追逐阳性。

KR L4对scalar的selected Q AUC均值+.00035125，两个seed正，而LL均值+.00136978，两个seed坏，且终点AUC两个seed下降。ML L4对FEATURE AUC均值−.00044316、LL+.001903，两seed均坏。单seedCAL、OWNER基本匹配L4排序，GEOM在ML selected AUC更高，削弱“跨目标形成及特殊条件块读门必要”的主张；它们仍保留各自真实foreign路径，不证明foreign内容完全无用。上述机制结论须保留单seed证据WEAK的边界。

## 正式复评与冻结后历史基准

主要历史test的LL配对差（HB−参照）如下；负数有利，n=4。全部逐seed差、SD、median、t区间和AUC见[锁定复评](FORMAL_LOCKED_REEVALUATION.md)。

| reference | selected_LL_delta | good_seeds | epoch50_LL_delta |
| --- | --- | --- | --- |
| R0_BASE | +0.000740750 | 2 | -0.008754277 |
| R5_PB_H1 | +0.000413212 | 2 | -0.013943692 |
| R7_PB_HISTORY | -0.000559137 | 2 | -0.035783181 |
| STGU_DOMAIN_ADAM | +0.002532282 | 1 | -0.017795771 |

主选点对BASE/H1/STGU没有平均优势，对HISTORY仅2/4有利，主要LL区间全部包含0；没有稳定主目标收益证据。selected AUC对四参照平均均低，对HISTORY四seed均低。legacy selected对BASE/H1/HISTORY的LL四seed均坏，Q诊断对它们也只有1/4好，开发阳性未稳定迁移。

共同50的HB对H1/HISTORY/STGU在test、legacy和Q三个角色均4/4 LL更低；test对HISTORY均值−.03578318、t95%[−.06969527,−.00187110]。这提供MEDIUM的阶段性相对风险减损证据。所有方法的全部正式轨迹在50的LL仍高于自己的selected；不能用晚期较小恶化替换主选点目标，更不能挑有利评价角色重新晋级。

LPS在冻结后test的KR L4−L2 selected AUC为−.00590775/−.00430570，两seed均负，与Q开发有利均值矛盾；ML对FEATURE的selected test AUC均值−.00166688、LL+.00892946，legacy也两seed均坏。P2 CAL/OWNER通常匹配近似排序或更低LL，但部分终点跨角色反向；单seed证据WEAK，没有统一最优新变体。LPS50保持NO_START，原形成主张缺乏稳定独立增量。

外部六条完整24适配全部完成；AutoIFS在test selected AUC高于L4两数据，但只有一个seed。fresh STGU正式LL选择公平，其test selected LL均值.43581552，HB .43834780；共同50却HB较低。任何“全面外部领先”的主张均不成立。

## 域、任务、阶段及竞争解释

KR自然fit规模为138225/1395118/71561，ML为124759/254879/260495。域规模没有独立干预，与基率、用户/物品、特征支持、基线难度混杂。KR最低基线AUC域不是最小域，且其LL最低，不能将小域等同困难域，不能用绝对LL排出统一难度。

开发中存在具体的小域排序改善/大域排序下降，例如ML L4对FEATURE在两seed的最小域AUC均正、最大域均负；但最小域LL同时变差。这是局部描述性取舍（MEDIUM），规模作为主要因果变量证据WEAK。不能从某个比较外推“复杂方法必然帮助小域”或“M3oE必然失败于困难域”。正式域任务结果另结合全部六格、基线和50epoch阶段判断。

ML validation用户OOV约76.16%、test约19.32%，Q_explore几乎0；不同评价角色的特征支持差异提供竞争解释，但没有因果识别，原划分与fit-only编码保持不变。当前反馈、跨度、计算增量、选点、校准、后期过拟合、普通连续读取和额外监督必须共同评估，不能仅凭非零动作或超梯度宣称机制有效。

## 成本与实际价值

已记录研究leaf分量18715.766543秒；规范训练进程wall相加19226.915540秒、活跃区间并集9972.533860秒、事件跨度13112.449982秒，定义不同且不能相加或称GPU小时。研究实际、独立部署与开发搜索三账重叠；部分历史前缀及cached objective访问未知，完整总量不伪造为精确。

正式50的独立部署已记录组件均值HB992.683秒、BASE322.388秒、H1为367.477秒、STGU541.409秒；HISTORY为部分已知、总量UNKNOWN。HB约为BASE3.08倍、H1的2.70倍的记录分量，没有得到主selected收益支持。KR L2/L4部署均1634942参数、P50约5.51/5.52毫秒；元形成增加训练信息与计算，没有自动带来推理额外模块。CAL实际min-L2路径P50约6.05毫秒，FEATURE约4.00毫秒。

所有51测量、完整方法汇总、已知/未知项及[注册检查点质量—成本观察](APPENDIX_QUALITY_COST_OBSERVATIONS.md)均保留；短顺序推理重复不是独立seed，预算不是实际耗时。

## 证据强度与下一项最小工作

STRONG：源码/角色/检查点身份、审查后机制执行及完整有限矩阵有直接可核验工件；非零动作/超梯度不等于有效性。MEDIUM：RHD晚期相对风险减损跨四seed/多个旧角色重复；LPS对FEATURE的ML反例及KR对uniform的test反例重复，但样本固定、seed有限。WEAK：RHD主selected优势、LPS跨目标形成必要性、域规模主因、所有单seedP2及外部推广结论。

主要竞争解释是验证选点和评价分布、后期过拟合/校准、保守BASE回退频率、更多shadow计算、普通连续读取及额外条件预测监督。最小后续优先在新封存自然窗口上比较预先固定动作预算的保守HISTORY与当前验收HB，保留BASE及原selected主目标；如果不用当前标签也匹配晚期收益，就直接削弱当前响应信息的独有必要性。本版不追加状态网络、模型组合或事后新seed搜寻。

详细研究边界见[尚未解决的问题](OPEN_QUESTIONS.md)，全部原始逐seed数值及六格见[数值附录](APPENDIX_INDEX.md)。本版不再追加未登记训练。

## 复核与交付位置

[完整报告索引](REPORT_INDEX.md)连接全部主报告、设计/审查、Q与两个历史角色完整逐seed/六格数值、检查点与父状态身份、成本和推理附录。原始数据、逐行预测、checkpoint、操作日志和完整源码归档保留服务器`/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/`。远端仅发布报告、聚合数值、来源收据与图；[发布范围](PUBLICATION_SCOPE.md)记录固定父提交和时间分支。
