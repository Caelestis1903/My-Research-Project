from pathlib import Path
import pandas as pd,numpy as np,json
P=Path(__file__).resolve().parent

def table(df):
 cols=list(df.columns)
 def fmt(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
 return '| '+' | '.join(map(str,cols))+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in r)+' |' for r in df.itertuples(index=False,name=None))+'\n'
recalc=pd.read_csv(P/'ao04_test_metric_recalc.csv');s6=pd.read_csv(P/'s6r2_summary.csv');paired=pd.read_csv(P/'ao04_selected_paired_l2.csv');boot=pd.read_csv(P/'ao04_paired_bootstrap.csv');s5=pd.read_csv(P/'s5_recomputed_delta_auc.csv')
rows=s6[(s6.rule=='R0')&s6.metric.isin(['M','T','N'])][['dataset','split','metric','count','mean','std','positive_seeds','min','max']]
r='''# 实验结果：历史产物独立算术复核

本轮新增训练为0。下列数字来自可访问的历史工件，不能计为新的训练复制。证据等级：L2=从保存CSV重算；L3限定为保存预测与当前hash匹配原CSV、row IDs、labels、域映射验证后的指标重算；L4/L5未达到。统计分析为retrospective，不是未使用test上的确认。

## AO04 R2（不等于 S6-R2）

KuaiRand B0/2022、B0/2023、B1/2022 的7个checkpoint、6个cell共126行，逐cell AUC/LogLoss最大绝对残差分别为 %.3g / %.3g。原样本哈希与配置一致，保存test row IDs唯一且在界内，labels与原数据逐行一致。四份可用完整epoch CSV重新选点均匹配manifest：KR B0/2022=32、B0/2023=36、B1/2022=33；ML B0/2022=7。AO04源码28项config hash核验一致。全链执行复现仍未达到，不能把这些核验叫作整个矩阵已完成。

保存validation数组缺row IDs；其NPZ SHA匹配manifest，但不能把labels相等当成唯一行身份。因此validation配对差值保留L2，未提升为L3。MovieLens缺失五个完成run的原预测/manifest；可用aggregate CSV仍可算术复核。

'''%(recalc.auc_residual.abs().max(),recalc.logloss_residual.abs().max())
r+=table(paired[['dataset','seed','split','contrast','delta_macro_auc','delta_macro_logloss']])
r+='\nMovieLens两seed均值：B1−B0 validation macro ΔAUC=%.9f；B1−B2=%.9f。只支持该设定无实用量级优势，不能证明B1普遍无效。\n'%tuple(paired[(paired.dataset=='movielens')&(paired.split=='validation')&(paired.contrast==c)].delta_macro_auc.mean() for c in ['B1-B0','B1-B2'])
r+='''
### KuaiRand单seed配对不确定性

selected B1−B0，seed2022；同一row的两臂配对，在每cell内分别按正/负标签重抽，固定analysis RNG=20261015、1000次，百分位95% marginal CI。此为固定训练seed的行抽样变异，不是训练seed置信区间。未做familywise校正；时间相关样本不满足独立同分布时，这些CI不能代表全部时间泛化不确定性。没有把不同任务独立bootstrap结果拼成macro CI。

'''
r+=table(boot[['domain','task','n','n_pos','delta_auc','ci95_low','ci95_high']])
r+='''
六个cell的区间均覆盖0；这表示数据不足以在该检验下区分零，不能判为两臂等价，也不能消除已观测的点估计下降。d0两任务仅228/120正例，宏观不确定性不能只用训练seed方差替代。原epsilon=1e-12，另有1e-7诊断列，未混合定义。

每run selected/final/common的macro、eval-row-weighted（按当前evaluation样本频率加权cell AUC，不等于pooled AUC）、worst-cell、worst-domain、分任务/分域AUC和LL见 `ao04_test_aggregates_recalc.csv`；逐cell见 `ao04_test_metric_recalc.csv`。历史aggregate中的W不自动视为当前evaluation频率权重。

## S6-R2 F/DH selected（L2）

Δ=DH−F；M=six-cell macro，T=target两任务均值，N=其余四cell均值。KuaiRand target=d1，MovieLens target=d2，不能交换物理域身份。四个seed均为2022–2025。

'''+table(rows)
r+='''
KuaiRand test N均值正且4/4同向，但seed2025约+0.000011，4/4同向不等于4/4达到+0.003。validation N仅1/4正，均值负。MovieLens test M/N近0且符号混合，构成外推限制。当前支持“保存CSV内存在反转”，不支持“反转机制已证明”。

所有domain×task、per-seed、选中epoch与R0–R4差值在 `s6r2_paired_recomputed.csv`，汇总在 `s6r2_summary.csv`。R0为原selected端点；其他规则只有test_available=1时才计算，缺值不补齐。A-grid替代选点test结果仍为grid sensitivity；不能以同方向升级为exact-selected robustness。S6原NPZ/checkpoint/manifest缺失，无法对这批F/DH做原预测复算、paired row bootstrap、分数校准或精确同epoch/timing分解。AO04的bootstrap不替代S6-R2的bootstrap。

## S5 Equal四臂（L2）

selected ΔAUC，每cell4seeds；模型M3OE、KuaiRand natural，d1为target。下表从保存paired CSV重算，不仅抄写原报告。

'''+table(s5[(s5.checkpoint=='selected')&s5.contrast.isin(['T-F','HT-F','DH-F'])][['contrast','domain','task','count','mean','std','min','max']])
r+='''
LL同步见 `s5_recomputed_delta_logloss.csv`；完整selected/final/common contrasts见 `s5_recomputed_delta_auc.csv`，四臂绝对AUC/LL及validation指标见 `s5_absolute_recomputed.csv`。原始CSV来源均在inventory可追踪；没有完整optimizer原状态，暂不将此L2重算升级为状态机制复现。

## 图表和数据

![Forensic evidence](forensic_evidence.png)

图左为AO04单seed selected test配对区间，图右为S6-R2四seed selected非目标反转；两图不是同一个实验。PDF版 `forensic_evidence.pdf`。原数据类别计数、split rowhash、特征schema分别见 `split_cell_counts.csv` 和 `dataset_preflight_*.json`。
'''
(P/'experiment_results.md').write_text(r)
(P/'mechanism_analysis.md').write_text('''# 机制分析与证伪边界

状态：仅完成可访问历史证据的法证复核；无新训练、无新方法有效性或创新性认定。证据强度STRONG/MEDIUM/WEAK描述支持范围，不能与L0–L5来源等级混为一谈。

## 稳定现象与矛盾

**STRONG（限定为当前CSV算术事实）**：S5 Equal/KR/d1/M3OE的selected target rescue约T−F +0.0414/+0.0423，HT−F +0.0411/+0.0405，四seed两任务均正；DH−F约+0.0009/+0.0008。S6-R2 KR selected test ΔN=+0.006757（4/4正），但同组validation ΔN=−0.003851（仅1/4正）。本次独立重算确认这些数字；没有独立重跑执行链。

**MEDIUM（历史状态整体解释，非本轮因果复现）**：S5 pattern与persistent history总体因子相关，但C与A_arch同步干预、m/v/step/路径未分离，原moment gate只15/16。不能把它解释成m单独致因，或把shared expert的名字当作梯度隔离证明。selected选点时序本身也跟history改变，必须做同epoch与独立选点的两种比较。

**MEDIUM（现有预测点估计；泛化解释WEAK）**：AO04 KR B1/2022 validation macro升0.001494，test降0.008385，两个d0 cell分别降0.022434/0.030011。原test逐cell指标可以从预测重算至浮点精度，但本次六个paired bootstrap CI均含0、只有一完整paired seed，故反转的普遍性和统计稳定性证据弱。KuaiRand B1/2023是执行INCOMPLETE，不是科学NO。MovieLens B1两seed平均validation效应约+0.000005765，也不足以主张结构性增益。

重要反例：S6-R2 MovieLens selected test effect近0；S5 seed2024 HT在d0的selected代价明显且T late target collapse；历史S8 selected continuation没有稳定超过原F best，late阶段局部rescue不能替代selected性能改善。S4 norm controls、S5历史pattern和S8阴性边界来自不同协议，不可简单合并为一个机制已被跨设定验证。当前S4/S8原始状态也缺失，此处历史描述不冒充新复算。

## 规模、难度和域/任务代价

规模作为主解释变量的证据为 **WEAK**。KR train d1=1,743,898，d0=172,782，d2=89,452；目标d1反而是最大域。S5改善这个最大域时d0下降，两任务T−F约−0.0273/−0.0374；d2 selected接近零或稍差，HT却同时改善d1与最小d2。因此现象不是统一的“小域获益、大域下降”规律；确有特定干预下的目标/非目标trade-off，但方向取决于物理域、objective、arm、seed与training stage。

域难度不能由n定义，也不能把不同任务的绝对AUC直接当作纯难度排序。应比较同域同任务的F baseline、matched reference、label结构和time window。这里缺失matched reference原预测/状态，不能给域规模因果归因。MovieLens任务为嵌套rating阈值，与KR两个行为任务并不等价；跨数据集效应差异无法单独识别任务相关性原因。

## 竞争解释

1. **选点/训练时序**（MEDIUM，可检验）：S5与S6 selected各自选epoch，不等于同θ/同epoch干预。S6旧test grid稀疏，无法恢复未保存epoch的exact alternative test。最优先的判别是B1 dense replication，但必须先恢复/明确原协议身份和完整状态约束。
2. **有限正例与时间窗口**（MEDIUM事实、WEAK因果）：KR test d0 long_view只有120正例，AO04配对区间宽；时间窗口的特征/条件关系或排名噪声可产生不同泛化方向。当前bootstrap按行分层未建模时间依赖；无法凭该CI排除时间相关性。
3. **domain/label marginal shift**（STRONG描述、WEAK反转解释）：原数据重新计数确认train d1占86.93%，val/test约98.2%，d1 click正率0.3778降至约0.210；但val/test域比例极近，因此粗域mix变化不足以单独解释val/test AUC反号。AUC、LL和校准应分开看，label prior变化不直接等于排序损伤原因。
4. **C/A耦合和optimizer history**（WEAK具体成分归因）：历史history bundle效应不说明共享C足够，也不说明architecture A足够。先做B2 C×A正交隔离，再决定C1 common-θ ghost m/v probes；禁止跨轨迹状态硬插后称分量因果贡献。
5. **融合不可辨识或低敏感度**（WEAK）：AO04现有rho均未接近0/1饱和，scalar derivative系数约0.048–0.049；这只否定“rho饱和必然解释一切”的简单说法。没有E_j−E_k方向与logit Jacobian、冻结θ预测干预，就不能从q变化/不变化断言可辨识或有效学习。gate entropy与branch norm仅日志描述，不是branch collapse证明。
6. **普通过拟合/隐式正则**（MEDIUM历史线索）：S8 late rescue而selected无收益支持这种竞争解释，但不能自动解释S6-R2 KR已选test反转。两条问题应继续分开。

## 最有区分力的后续工作与停机

先恢复S5/S6原始manifest、source/data/init/batch hashes、selected/common/final预测与完整optimizer，关闭A0链条缺口。若原工件不可恢复，需要明确承认不能exact复现，再确定新协议，不将重训结果无标签混入旧结果。

B1限定KR Sample F/DH×4seeds×50epochs，每epoch full C/P/A state、RNG/iterator，冻结validation R0–R4选择；用same-epoch Δpolicy与各自selected Δ并列、cell/label paired bootstrap及时间窗口分析区分timing和噪声。若差异消失或CI不足，停止该线，不开发补救方法。B2才用于Equal的C/A定位，不能与B1混成支持某个先验结论的总证据。

C2先证明固定目标alpha与不同q的权重oracle等价，再谈Adam噪声交互；本次尚未运行该oracle。D1/E1仍未触发。被多次研究过的test只能作forensic，任何新方法性主张必须用未开发的后置/外部holdout。

本次未进行文献全文或创新性查新；规划§16仅是已知背景，novelty保持OPEN。没有把课程学习、动量、小域困难、M3OE困难域失败作为分析前提。
''')
(P/'r2_prediction_metric_recalc.md').write_text('# AO04 prediction audit\n\n见 [结果报告](experiment_results.md) 的AO04部分和 `ao04_test_metric_recalc.csv`。126 test cell/checkpoint指标复算；validation缺row ID，ML覆盖不足，整项A1为PARTIAL。\n')
(P/'r2_rho_identifiability.md').write_text('# AO04 sensitivity limitations\n\n见 `ao04_rho_log_diagnostics.csv` 与 [机制分析](mechanism_analysis.md)。仅日志描述，无冻结模型Jacobian，identifiability未判定。\n')
(P/'s6r2_forensic_review.md').write_text('# S6-R2 forensic review\n\n见 [结果报告](experiment_results.md) 的S6-R2部分。selected CSV反转已重算；原checkpoint/NPZ/optimizer缺失，exact替代选点和分层bootstrap均INCOMPLETE，A-grid仅为grid sensitivity。\n')
(P/'domain_task_bootstrap_evidence.md').write_text('# Bootstrap identity boundary\n\n`ao04_paired_bootstrap.csv`是AO04 KR B1−B0 seed2022的6个cell结果；不是S6 F/DH的结果。S6原预测缺失，要求的S6 paired stratified bootstrap为INCOMPLETE。两个问题不得混写。\n')
(P/'split_shift_evidence.md').write_text('# Split evidence\n\n原CSV哈希、split类别计数和rowhash已由 `dataset_preflight_*.json`、`split_cell_counts.csv` 与 `dataset_prediction_binding.json` 实测。保留global row-order80/10/10，未读取更早原始事件bytes重新证明时间排序。train→val域/label shift明显，val→test域mix近似一致；粗域mix不足以解释AUC反号。\n')
