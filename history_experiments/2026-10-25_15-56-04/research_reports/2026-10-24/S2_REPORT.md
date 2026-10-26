# S2 研究判断

> 本次按“仅上传实验报告”发布。配套CSV/JSON、配置、日志、源码、原始数据、预测和模型权重均未纳入本次提交；正文中的运行工件路径指原服务器归档。数值、统计口径和研究结论保持原报告内容。
> 原始报告SHA256：`be6236b771bda22689797768cdfafd6e3c138073891a845fe9da65e5093bff79`。

## 1. 本阶段回答对象

自然 ML-1M/KuaiRand、严格 fit-only、探索 seeds2022/2023。44 条主 pilot 均完成固定20epoch；8规则×5epoch静态校准各数据集完整保留。176份 best/final×validation/Q_explore 评价齐备，无缺失。主张为 ranking，六格macro AUC为主要质量指标，LL及自然人口指标并列；尚未访问Q_confirm/test。

## 2. 主要比较与完整原始表

| 数据 | seed | 候选−对照 | validation ΔAUC | Q_explore ΔAUC | Q_explore ΔLL |
|---|---:|---|---:|---:|---:|
|movielens|2022|B_CONT−DIRECT_CONT|-0.000034307|-0.003178719|+0.002116086|
|movielens|2022|B_CONT−GLOBAL_CONT|-0.000020453|-0.001681297|+0.001462881|
|movielens|2022|C_CTRL−C_STATIC|-0.000206817|-0.000077067|+0.000396161|
|movielens|2022|C_CTRL−NATIVE_FIXED1|-0.000004649|-0.000000703|+0.000000676|
|movielens|2022|I1_PERM−I1_RW|-0.000235487|-0.000104255|-0.001600114|
|movielens|2023|B_CONT−DIRECT_CONT|+0.000141445|+0.000029450|+0.000258144|
|movielens|2023|B_CONT−GLOBAL_CONT|+0.000046859|+0.000057935|-0.000071125|
|movielens|2023|C_CTRL−C_STATIC|-0.000005951|-0.000020493|+0.000652237|
|movielens|2023|C_CTRL−NATIVE_FIXED1|+0.000045239|+0.001811268|-0.001753500|
|movielens|2023|I1_PERM−I1_RW|+0.000065153|+0.000108913|-0.001089411|
|kuairand|2022|B_CONT−DIRECT_CONT|-0.000942533|-0.000925926|+0.000118129|
|kuairand|2022|B_CONT−GLOBAL_CONT|-0.000531580|+0.000400951|-0.000286568|
|kuairand|2022|C_CTRL−C_STATIC|-0.000156879|+0.002098163|-0.000227402|
|kuairand|2022|C_CTRL−NATIVE_FIXED1|-0.000166267|+0.002842324|-0.000594247|
|kuairand|2022|I1_PERM−I1_RW|+0.000207821|+0.002459136|+0.001668186|
|kuairand|2023|B_CONT−DIRECT_CONT|-0.000204025|-0.003412663|-0.001021298|
|kuairand|2023|B_CONT−GLOBAL_CONT|+0.001239527|+0.000632798|-0.001528491|
|kuairand|2023|C_CTRL−C_STATIC|-0.000055869|-0.000724048|-0.000088121|
|kuairand|2023|C_CTRL−NATIVE_FIXED1|+0.000225530|-0.002465523|+0.001302123|
|kuairand|2023|I1_PERM−I1_RW|+0.000440090|-0.001630627|-0.000708636|

完整全arm、逐seed、best/final、六格与域/任务汇总见 raw_summary.csv（`raw_summary.csv`，服务器留存）、raw_cells.csv（`raw_cells.csv`，服务器留存）、raw_groups.csv（`raw_groups.csv`，服务器留存）。所有配对差见 paired_differences.csv（`paired_differences.csv`，服务器留存）、paired_cell_differences.csv（`paired_cell_differences.csv`，服务器留存），成本与工件身份见 raw_costs.csv（`raw_costs.csv`，服务器留存）、identities.csv（`identities.csv`，服务器留存）。没有用final替代selected，也没有把不同人口的得分混作配对。

## 3. 机制链及阶段限制

B_CONT在KR相对GLOBAL的Q_explore AUC两个seed均正，但均低于+.001实用带，且对同信息DIRECT两seed均负；不能把粗策略对比的收益归于收益监督。ML相对DIRECT从−.003179翻为+.000029。BANK_OFF相对native的ML query容量收益两seed为正、KR两seed为负，不能归于动态共享。

C_CTRL相对最强静态规则的validation AUC在两个数据集、两个seed均负；KR query从+.002098翻为−.000724。ML seed2022 selected为epoch11、控制16步、仅1锚点，critic尚未ready，实际部署BASE；final有8锚点且已ready。不能把早期选点的弱效果写成充分测试了任意状态控制器。

I1在KR validation AUC相对固定RW两seed为+.000208/+.000440，LL为−.007793/−.002973；query AUC却为+.002459/−.001631。ML selected均在epoch14、控制64步、只作过1次边决策，2023的selected全部RO；final才作过5次边决策。KR selected在epoch20、17次决策。完整可部署模式/锚点/阶段状态见 POLICY_STATES.json（`POLICY_STATES.json`，服务器留存）。因此仍需区分权限测量、决策覆盖、选点和外层泛化。

## 4. 强简单对照与最近方法

C_STATIC由seed2022的八规则validation选定：ML SHARED_EQUAL，KR TRANSFER_2_TO_1。I1最佳固定对照由seed2022 validation选定为RW（两数据集），2023未重选。B_CONT包含同信息DIRECT与凸BCE域级GLOBAL。最近方法已完成STEM_TASK与STEM_DOMAIN_TASK两候选的真实数据固定20epoch pilot，只按两数据集validation最佳AUC等权均值选择全局STEM_DOMAIN_TASK进入正式。保留官方核心RO语义和默认网络，每行仅监督本域两个行为标签；TikTok八共享专家大结构不冒称已复现。

## 5. 异常、成本与证据强度

**WEAK**：两个seed仍属探索，n不按六格/epoch扩张。稳定的有限现象包括KR的B_CONT未胜DIRECT、C_CTRL未胜validation最佳静态、I1的LL改善；矛盾包括query/validation、AUC/LL及两个seed的方向不一致。

44主训练和44评价均完成；主训练无OOM/NaN。扩展执行发生两次机械脚本错误（错误导入、glob局部遮蔽），均未修改研究算法。ML重放虽无捕获退出码但完整收据存在；KR部分工件保留并完整重跑。详细日志/incident收据不可删除，不将unknown退出码写作0。

I1实际额外probe更新为ML120/KR180步/轨迹，主更新为320/800。实现walltime受并发和其他用户负载影响，不能从KR PERM/RO实测小于1推导算法更快；ML2023实测比1.923超过1.8警戒，正式阶段会同时保存并审计预算内选点。详细定时、F/B、复制与controller开销原样保留。

## 6. 研究决定

B_CONT与C_CTRL：当前默认版本不晋级，缺少可重复的主要比较收益；这是当前协议/表示/动作的版本判断，不是理论类别不可能。

I1：**PROMISING_UNCERTAIN**，不称已证实有效。KR validation微小ranking方向两seed一致且LL改善，但query反例明确。进入已定义的四个新seed确认，主指标保持ranking，不将LL改名为主要成功标准。正式包保留两数据集×OFF/RO/RW/PERM×2024–2027，再加匹配STEM_DOMAIN_TASK，共40轨迹；结果可直接否证可重复实用ranking优势。

## 7. 域大小、竞争解释与下一有限任务

当前不能将域大小认定为主要因果变量。域大小与自然梯度质量、稀少正例、标签任务、风险等权、行序人口和输入表示混杂。ML小域并非天然更困难；两个任务须分别检查。逐格交换已经完整列出，不能从macro或某个小域获益推出普遍的小域改善/大域下降规律。

最小判别是固定I1规则与配置的四seed新证据；如果近零且区间宽，应报告证据不足，不声称等效。若LL改善而AUC不改善，按冻结ranking标准报告，并保留校准/目标交换解释。B_UNIT/I2/I4等未运行joint包保持OPEN/PENDING，不借此宣布失败，不要求将reserve用满。

## 8. 可复现收据

当前算法源码777ad7131d913268e5a51b708b012263059d557aaccde10d8589d85ab59f974d；source578到777仅运行设备政策迁移已审计。严格数据hash、config和实际checkpoint身份均见identities与POLICY_STATES。GPU按每卡显存<=85%且功耗<85%独立准入、无固定并发数；Q_confirm/test尚未解锁。正式训练前将单独写不可变MATRIX_LOCK，所有主训练和STEM基线完成后才统一访问test。
