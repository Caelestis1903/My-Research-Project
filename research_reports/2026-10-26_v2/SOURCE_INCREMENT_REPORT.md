# 固定来源内容与读取：四种子证据

状态：S1及全部触发后继已完成，统一冻结后已导出并复算test/legacy。每个实际比较条件均有5201–5204四训练seed；本文将Q、test、legacy分别解释，不以外层更改选择。

## 证据与主判断

Q_explore上PILOT_PROB 在两个数据集都稳定优于 H_DIRECT，但这个比较不能单独证明追加的外域来源监督是收益原因。KuaiRand 的 NATIVE/LOCAL 反例尤其重要：H_DIRECT 在该协议下已经损失相当一部分原生预测能力，恢复这部分能力与超越原生基线是不同主张。MovieLens 的概率读出在四个种子上同时改善 H_DIRECT 与 NATIVE 的 LL/AUC，属于值得保留的简单方法结果，证据强度 MEDIUM；暴露基准和四个训练种子限制其外推。

本轮分析先核过 26 个数据集×方法条件各四个训练种子，共 104 条完成收据，再撰写本报告。重复重放、两折 OOF、任务、epoch 和 bootstrap 均未增加 seed 数。见 [S1_FOUR_SEED_ANALYSIS_RECEIPT.json](S1_FOUR_SEED_ANALYSIS_RECEIPT.json)。

## Q_explore 的四种子均值

主指标是六个 domain×task 单元等权 Macro LL；AUC 同样对六格等权。此处没有混用 pooled AUC。逐 seed、SD、配对 CI、六格、域、任务、群体见表格文件。

| 方法 | ML Macro LL | ML Macro AUC | KR Macro LL | KR Macro AUC |
|---|---:|---:|---:|---:|
| NATIVE | 0.47618634 | 0.80778138 | 0.46659406 | 0.69812867 |
| LOCAL | 0.47761243 | 0.80743150 | 0.46604456 | 0.69935183 |
| B_REFIT | 0.47655122 | 0.80643808 | 0.48061013 | 0.67141990 |
| H_DIRECT | 0.47533167 | 0.80771574 | 0.47322343 | 0.67506338 |
| ETA_DIRECT_MATCHED | 0.47524008 | 0.80776411 | 0.47339907 | 0.67380950 |
| H_DIRECT_EFFORT | 0.47824998 | 0.80538042 | 0.50922842 | 0.64716530 |
| GROUP_CAL | 0.47590390 | 0.80694888 | 0.47093368 | 0.68451379 |
| PILOT_PROB | 0.47447740 | 0.80858434 | 0.46653321 | 0.69480475 |
| PILOT_LOGIT | 0.47564804 | 0.80722907 | 0.47066502 | 0.68516355 |
| FULL_SOURCE | 0.47655122 | 0.80643808 | 0.47262788 | 0.68118506 |
| FULL_SOURCE_ALL | 0.47584576 | 0.80683887 | 0.47998674 | 0.67067426 |
| FULL_TARGET | 0.47655122 | 0.80643808 | 0.47216975 | 0.68271997 |
| RANDOM_FEATURE | 0.47655122 | 0.80643808 | 0.47528534 | 0.67935848 |

## 稳定结果与反例

PILOT_PROB−H_DIRECT 的四种子平均 ΔLL：ML −0.00085427（描述性配对 t95 [−0.00116501, −0.00054352]），KR −0.00669022（[−0.00919760, −0.00418284]）；两数据均四个种子 LL 改善，AUC 也四个种子改善。两任务和三域的平均 LL/AUC 方向均支持这一比较。该观察强于原两种子开发判断，但不是新增未见数据的确认。

反例一：KR 的 PILOT_PROB−NATIVE ΔLL 只有 −0.00006085，两个种子改善、两个恶化；ΔAUC 为 −0.00332392，四个种子均下降。LOCAL−NATIVE 平均 ΔAUC 为 +0.00122316，四个种子同向，但平均 ΔLL −0.00054950 的方向只有 2/4。不能把赢过重拟合小头等同于赢过现有部署候选，也不能用 AUC 单项代替联合风险判断。

反例二：ML 的 LOCAL−NATIVE 平均 ΔLL +0.00142609、ΔAUC −0.00034988，两项四个种子均变差；同样的追加本地 pilot 并非跨数据集稳定获益。

ML 的 FULL_SOURCE、FULL_TARGET、RANDOM_FEATURE 都在四个种子的 P_cal 合法选择中取精确 OFF，因此与 B_REFIT 的最终预测相同。FULL_SOURCE_ALL 四个种子均选择 λ=0.1 的 READER，仍在 Q 上四个种子劣于 H_DIRECT。不能把 OFF 的零增量解释为未执行该条件，或据此声称来源函数空间完全没有内容。KR 的四种完整特征方法均选择 λ=0.1 的 READER；FULL_SOURCE 对 H_DIRECT 的 AUC 四种子改善，但 LL 只有 3/4 同向。

KR 的 FULL_SOURCE−FULL_TARGET 平均 ΔLL +0.00045813、ΔAUC −0.00153492；与 RANDOM_FEATURE 比则为 ΔLL −0.00265746、ΔAUC +0.00182658。学得来源特征相对未训练随机特征可用，不代表其已经超过目标专有特征或原生网络。证据强度 MEDIUM，具体同向数和区间以完整配对表为准。

固定 60 次 P_form 遍历的 H_DIRECT_EFFORT 未改善这次合法选点的 Q 表现。实现使用同 unit=2 初始化、归一化和前 20 次行计划；H_DIRECT 从 B_REFIT 的 20 次训练和 Adam 状态继续到 P_read，EFFORT 则先完成 60 次。这个有界剂量控制反对“只需增加该训练剂量”的简单解释，不能推出所有更长训练或其他选点都无效。

## 不同读取器不是相同表达类

PILOT_PROB 的候选是 B_refit 与两个外域的 Z+f_s，经 [h,z,候选概率] 的目标 gate 做概率凸组合。FULL_SOURCE 则在 B_refit 背景上读取来源隐藏特征并加 bias/temperature。后者对固定 φ 的线性谱/RRR空间具有表达包含关系，却不包含前者整条原生 Z 路径和非线性 gate。因此不能将这张表解释成纯粹“扩大线性读出仍失败”或“来源内容本身已被否定”。

B_REFIT 的 OOF 仅用于后续需求代理，不是全链 OOF。FULL_SOURCE_ALL 额外使用 P_form+P_read 监督，其他最终线性 reader 用 P_read；P_cal 只整组六头政策选择。所有行预算和活跃参数会在成本报告中单列，不以相同唯一行集合冒称监督剂量相同。

## 规模、难度与权衡

PILOT_PROB−H_DIRECT 的平均 LL 在 ML/KR 的三个域都改善，没有出现这一比较下“小域改善、大域下降”的统一权衡。与 NATIVE 相比，KR 的域0 LL 恶化约 +0.00166879，而样本更少的域2与样本最多的域1分别改善约 −0.00103297、−0.00081837；三个域 AUC 都下降。比较参照、任务和校准会改变叙述，域规模不能单独解释这些方向。域规模为主要因果变量的证据 WEAK；应结合自然数量、标签比例、支持/OOV及原生基线风险，不将小域预设为难域。

## 当前边界与下一判别

当前结论只覆盖已审查 fixed-fusion M3oE 的冻结 h/z、指定标签角色和预算。来源函数是 h 的确定函数，不创造给定 h/z 后的新观察信息；可检验的是有限标签、容量和预算下的可访问函数先验。额外目标拟合、原生路径绕过差基准、校准及来源隐藏先验仍是竞争解释。

若继续简单 reader 路线，最能区分解释的是保留所有臂相同的原生 Z 候选和 gate 预算，再对外域残差作预先锁定的真实/置零或标签置乱控制。本轮不因这个开放问题增加未登记矩阵；先完成已触发 RCF 形成和外层交付。

所有区间以 seed 为单位、n=4，使用 df=3 的配对 t 区间作描述；小样本分布假设、历史测试暴露和 KR 缺可信 user-ID 的限制仍保留。非显著不等于等效，门槛未过不等于所有来源共享无用。

## 数据表

- [逐 seed](tables/S1_Q_four_seed_per_seed.csv)、[六格](tables/S1_Q_four_seed_six_cells.csv)、[域与任务](tables/S1_Q_four_seed_domain_task.csv)、[群体](tables/S1_Q_four_seed_groups.csv)。
- [均值与 SD](tables/S1_Q_four_seed_summary.csv)、[配对区间](tables/S1_Q_four_seed_paired_summary.csv)、[逐 seed 配对差](tables/S1_Q_four_seed_paired_differences.csv)。
- 旧两种子表仅属 SUPERSEDED_TWO_SEED_PROVISIONAL，不参与这些数值。

## 开发角色的域与任务反例（四seed）

PILOT_PROB−NATIVE在ML域0/1/2的平均ΔAUC为+0.00065411/+0.00058481/+0.00116997，LL三个域均4/4改善；域0的AUC只有3/4改善，不能把宏观4/4写成每个格都一致。两个任务的AUC/LL则各四seed同向改善，task0/1平均ΔAUC为+0.00091097/+0.00069495。最大域2平均排序收益最大，规模没有提供单调小域获益解释。

KR反例集中但不限于小域：域0的概率读取相对NATIVE，AUC −0.00902224、LL +0.00166879，两者4/4恶化；域1/2的LL平均改善却各只有3/4，AUC平均也下降且方向不稳。任务0/1的AUC均4/4下降，平均−0.00191383/−0.00473401；task1 LL只有1/4改善。这里低基率、目标头/旁路及原生基线风险是竞争解释，不能从“最小/最大域”单独推因果。

[四seed域任务配对完整表](tables/Q_four_seed_main_contrasts_domain_task.csv)同时保留均值、SD、区间与种子方向；没有按这些诊断重新选模型或群体。

## 冻结后的test/legacy：不保留单向叙事

| 数据/角色 | 对比，候选−参照 | ΔMacro LL | ΔMacro AUC | LL/AUC改善seed |
|---|---|---:|---:|---|
| ML test | PILOT_PROB−H_DIRECT | +0.00278641 | −0.00005596 | 0/4，2/4 |
| ML legacy | PILOT_PROB−H_DIRECT | −0.00076706 | +0.00076562 | 4/4，4/4 |
| ML test | PILOT_PROB−NATIVE | −0.00436016 | +0.00255431 | 4/4，4/4 |
| ML legacy | PILOT_PROB−NATIVE | −0.00193477 | +0.00111521 | 4/4，4/4 |
| KR test | PILOT_PROB−H_DIRECT | −0.00506889 | +0.01216050 | 4/4，4/4 |
| KR legacy | PILOT_PROB−H_DIRECT | −0.00779688 | +0.02250493 | 4/4，4/4 |
| KR test | PILOT_PROB−NATIVE | +0.00253926 | −0.00210808 | 0/4，0/4 |
| KR legacy | PILOT_PROB−NATIVE | +0.00019486 | −0.00307111 | 2/4，0/4 |

ML概率读取对NATIVE跨三个暴露角色的两指标均四seed改善，值得保留；但相对H的test LL四seed反向，t95[+.00138142,+.00419141]，而legacy LL仍改善且区间[−.00136240,−.00017171]。因此动作是保留简单方法候选，不能宣称PILOT_PROB跨角色支配直接头。域占比、标签基率、支持与校准机会变化是竞争解释，不是已隔离的唯一原因。证据MEDIUM，测试暴露和四seed限制仍在。

KR对弱H的收益重复，但对NATIVE的test两指标全四seed恶化；LL区间[+.00000829,+.00507023]，AUC区间仍跨0但方向4/4。legacy又出现LL2/4、AUC4/4恶化，说明改善重拟合头不等于来源净增量。LOCAL对NATIVE的test LL在两数据均4/4恶化，不能只保留KR Q/legacy的局部AUC改善。

线性与概率读取也有指标冲突。ML的PILOT_LOGIT−PILOT_PROB在test LL −.00403440而AUC −.00046076，两者方向都4/4；legacy则LL +.00137691、AUC −.00147976，均4/4恶化。KR的LOGIT test AUC −.00604224、legacy −.00968409，均4/4，无法用“线性类更大就一定更好”解释。ML三个FULL_SOURCE/TARGET/RANDOM角色仍精确OFF；KR FULL_SOURCE在legacy胜随机LL/AUC均4/4，但test LL仅3/4、AUC3/4，且不胜FULL_TARGET。固定空间有可读取差异不等于真实来源内容具有独立稳定优势。

## 支持、任务与难度

ML test概率读取相对NATIVE在三个域和两个任务的平均AUC/LL均改善，域/任务均四seed同向；最小域0平均ΔAUC+.00427499、最大域2+.00095955，与Q的最大域增量最大不同。KR test域0 AUC −.00756675、LL +.00101850均4/4恶化，域2 AUC +.00199074为4/4改善，域1 LL +.00812253为4/4恶化。跨任务的LL也都4/4恶化，不能把个别小域正结果代替完整结论。

按原始ML user是否在fit出现分层，PILOT_PROB−NATIVE的test六格Macro LL在seen/unseen分别−.00525517/−.00075968，AUC+.00278610/+.00133046，均四seed改善。这排除了“收益只存在于见过的user”这一过强说法，但不是来源监督因果隔离。KR test编码OOV只有42行且都在域1，域0/2为0；OOV指标缺失或极小支持不能当作原始user/item重叠，也不足以解释整域退化。

完整外层逐seed、SD、配对t95、六格、域任务、最差格、固定logit组及支持均保留在CORE/SUPPORT表。ML按user、KR按row的1000次配对bootstrap覆盖RCF对H/FULL_ALL/POOLED/LOCAL_ACTIVE四个主比较；本节其他对比的区间为四训练seed配对t，不冒称新增bootstrap结果。

证据：[CORE核算](CORE_FOUR_SEED_ANALYSIS_RECEIPT.json)、[支持核算](SUPPORT_FOUR_SEED_ANALYSIS_RECEIPT.json)、[所有逐seed指标](tables/CORE_four_seed_per_seed_metrics.csv.gz)、[配对差/区间](tables/CORE_four_seed_paired_summary.csv.gz)、[支持逐seed](tables/SUPPORT_four_seed_per_seed.csv.gz)、[ML user分层配对](tables/SUPPORT_ML_user_macro_contrasts.csv)。
