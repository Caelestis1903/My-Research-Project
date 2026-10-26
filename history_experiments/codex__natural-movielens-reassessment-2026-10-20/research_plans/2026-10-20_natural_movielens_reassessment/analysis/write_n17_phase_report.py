"""Render the primary-authored N17 analysis with frozen numerical tables."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent;P=ROOT/'analysis/n17'
f=pd.read_csv(P/'paired_summary.csv');f=f[f.task==-1]
def row(comp,readout,metric):
 x=f[(f.comparison==comp)&(f.readout==readout)&(f.metric==metric)];assert len(x)==1;return x.iloc[0]
def delta(r):return f'{r["mean"]:+.6f}（{int(r.positive)}/{int(r.negative)}）'
def ci(r):return f'[{r.ci95_low:+.6f},{r.ci95_high:+.6f}]'
qnames=['formal_sameLR0.0003_SB-SN','formal_sameLR0.001_SB-SN','formal_sameLR0.01_SB-SN','matched_budget_sameLR0.0003_SB-SN','matched_budget_sameLR0.001_SB-SN','matched_n_sameLR0.0003_SB-SN','matched_n_sameLR0.001_SB-SN','quota_sameLR0.01_SB-SN','H200_sameLR0.01_SB-SN']
lines=['# N17自然MovieLens完整阶段分析','', '''## 1. 完成范围、协议与证据单位

本阶段规定训练和触发追加均完成：16pilot、64正式H100、8条件H200；N2/N3基础48项及精度追加14项完成；72正式配置全部冻结后执行零更新test，pilot不进入正式test。模型真实bytes SHA冻结、配置/选模绑定和receipt见[冻结目录](analysis/n17_freeze/release_receipt.json)。8000个epoch预测（含H200共享前缀）及48000个记录cell已保存；主代理独立复算selected/final共864个预测cell，残差<1e−12。没有宣称8000个预测都由第二套指标实现重算。[分析凭证](analysis/n17/analysis_receipt.json)、[全部测试工件](analysis/n17_test_artifacts/index_status.json)。

模型为原N17固定融合结构、单Adam及耦合WD1e−5；本轮未另拟合ML校准器。任务为rating>3/>4，Sample目标α=p，IID SN q=p/SB q=uniform；quota按原整数配额归一化。自然base人口800167，三域155949/318599/325619，优化seed固定人口；正式四seed2022–25，pilot2030/31。LR对称grid由pilot冻结共同/两侧own赢家均.01，处于grid上界，不能称全局最优。common/own指向同组模型，不算独立重复。

H100完整人口2000更新、80,016,700曝光；matched-budget与matched-N各500更新、18,750,000曝光，同为100个macro epoch。matched-N187500保留自然比例；matched-budget使用完整支持。控制只注册.0003/.001，不能用赢家.01对控制做纯人口归因。H200复制且保持H100前缀，新增100轮/2000步/80,016,700曝光，累计200轮/4000步；[实际新增预算](analysis/n17_training_accounting/accounting_status.json)避免重复计算前缀。

每cell独立算AUC，再等权或指定p加权，不是pooled AUC。区间为固定人口下四优化seed的点态描述性t95，没有多重比较校正；相关模型、bank、cell与H200前缀都不是新的独立总体重复。历史test已反复暴露，本轮为探索性复现。

## 2. 全部同LR q比较

差值SB−SN；括号AUC为正/负seed、LL为变差/改善seed。下列18个AUC点态区间全部跨0，不能等同于严格等效或不存在任何轨迹效应。''','', '|条件|阶段|SN AUC|SB AUC|ΔAUC（正/负）|t95|ΔLL（差/好）|','|---|---|---:|---:|---|---|---|']
for name in qnames:
 for label in ['selected','final']:
  a=row(name,label,'auc');l=row(name,label,'logloss');assert a.ci95_low<0<a.ci95_high
  lines.append(f'|{name}|{label}|{a.reference_mean:.6f}|{a.candidate_mean:.6f}|{delta(a)}|{ci(a)}|{delta(l)}|')
lines+=['','''共同LR .01选中SN .767539、SB .767283，Δ−.000256（一正三负）；test宏LL .510430/.512205，Δ+.001775（二差二好）。H100末轮q差AUC−.001187（一正三负）、LL+.016585（三差一好）。开发selection相同选中模型q差约+.000481（三正一负），末轮约+.002420（四正）；test对应为负，不能把开发正号推广为独立总体收益，也不能仅据不显著把它们视为相等。

.001末轮AUC四负（−.001286），但区间仍跨0；.0003选中LL四seed改善（−.001656），区间也跨0。不同指标、LR与阶段均保留。旧人工ML共同.001 selected−.010392、8/8负，末轮+.002114、5/8正且LL更差；新自然对应selected+.000127二正二负、末轮−.001286四负。旧KR Adam selected−.005203（一正七负）亦不是本轮−.000256的等价环境。旧formal ML种子2032–39，未与新2022–25伪造跨人口配对；旧人工主步/支持及pilot赢家也不同，不能唯一归因于p。

## 3. 预设时间窗口与quota

AULC严格使用已完成epoch指标的算术均值，未事后更换窗口。''','', '|条件|窗口|ΔAUC（正/负）|t95|ΔLL（差/好）|','|---|---|---|---|---|']
for name in qnames:
 if name.startswith(('formal_','quota_')):windows=['AULC1_20','AULC1_100']
 elif name.startswith('H200_'):windows=['AULC1_200']
 else:continue
 for label in windows:
  a=row(name,label,'auc');l=row(name,label,'logloss');lines.append(f'|{name}|{label}|{delta(a)}|{ci(a)}|{delta(l)}|')
lines+=['','''.0003早期1–20 AUC+.000468四正，但区间约[−.000008,+.000944]仍含0；共同.01早期+.000066二正二负，1–100−.000832一正三负。没有跨LR普遍早期收益。H200的1–100与H100完全同源，不当新重复。

quota不是通用改进：SN quota−IID selected AUC+.000119（二正二负），final−.003612（一正三负）；SB selected−.000529、final−.001409（均一正三负）。SN的1–100 AUC差−.001316四负、LL+.004692四差，后者点态区间[+.000607,+.008778]；早期1–20 AUC却+.000275四正且LL+.001781四差。quota×q交互1–100 AUC+.001060四正、区间约[−.000006,+.002126]，LL−.003185四改善、区间跨0。计数项局部小不能推出quota无任何时间效应，更不能把消除计数波动等同于性能改善。[交互全表](analysis/n17/quota_by_q_interaction_summary.csv)。

## 4. 更新/曝光与支持控制

下面均保持相同LR、同q配对；matched-N−matched-budget同时改变支持和单样本重复频率，不是单独改变抽象N。formal−matched-budget保持完整支持，但增加更新/曝光并改变选模轮次对应的优化时钟。''','', '|对照|阶段|ΔAUC（正/负）|t95|ΔLL（差/好）|','|---|---|---|---|---|']
controls=[x for x in f.comparison.unique() if '-minus-matched_budget_' in x]
for name in controls:
 for label in ['selected','final']:
  a=row(name,label,'auc');l=row(name,label,'logloss');lines.append(f'|{name}|{label}|{delta(a)}|{ci(a)}|{delta(l)}|')
lines+=['','''相同500步/18.75M曝光下，matched-N的selected宏AUC比完整支持低约.0181–.0195、所有四seed负；final低约.0460–.0735。完整支持formal加到2000步后，selected变化小且区间跨0，final则低约.0196–.0423、各四负。由此完整支持与较少训练可同时优于N匹配或更长训练条件；总N、每行重复、ID支持/标签条件与优化时钟均是竞争变量。由于两种自然人口比例近乎相同，比例变化不能解释这些控制内部差异；但不能把支持差等同于仅由N造成的因果效应。

旧人工ML共同.001 selected q差−.010392（八负），而同N187500、同500步、同.001的新自然匹配N条件为−.000123（二正二负）。因此匹配总量/步数后旧强负差也没有恢复；但训练支持、标签/特征与旧seed同时抽子集的协议仍不同，不是p的独立因果识别。

## 5. H200：训练继续优化与末轮风险恶化

预注册rhoJ平台门控已触发，两q四seed额外100轮均完整执行；选中模型仍为原12–14轮。训练J各四seed下降约.041，test selected保持同一模型、差值精确0。''','', '|E200−E100|Δtest AUC（正/负）|t95|Δtest LL（差/好）|t95|','|---|---|---|---|---|']
for arm in ['SN','SB']:
 a=row('H200-minus-H100_'+arm,'final','auc');l=row('H200-minus-H100_'+arm,'final','logloss');lines.append(f'|{arm}|{delta(a)}|{ci(a)}|{delta(l)}|{ci(l)}|')
lines+=['','''所有domain均值的续训AUC各四负、LL各四差；任务均值LL同样两q各四差。AUC任务边界：SB task1为三负一正，其余三个q×task均四负，不能写为所有任务都四负。47/48 seed×cell AUC下降，47/48 LL升高：SN2022 d0/task1 LL反而−.028988（AUC−.000695）；SB2023 d2/task1 AUC+.005686但LL+.032913。

开发阶段LL 48/48升高，故当时逐配置任意非负cell权重的恶化身份成立；test仅47/48，不能把该“逐配置任意权重”命题直接搬过来。四seed均值层面的12个q×cell LL差仍全正（最小+.059420），所以固定四seed平均风险在任意非负归一化cell权重下仍升高；这个有限算术结论与每个seed的结论不同。**STRONG**有限权重身份，**MEDIUM**固定人口的阶段恶化，**WEAK**唯一校准/过拟合原因。[逐cell计数](analysis/n17/H200_test_continuation_cell_status.json)。

![H200冻结测试曲线](analysis/n17/H200_test_trajectories.png)

图为四优化seed均值±seed SD，虚线为E100共享前缀边界；不是独立总体置信区间，也未按test重选模型。

## 6. 物理域、任务、基线与取舍

共同.01的q差按两个任务等权汇总如下；d0最小，d2最大，d1/d2规模接近，大小不是难度定义。''','', '|读数|d0 ΔAUC/ΔLL|d1 ΔAUC/ΔLL|d2 ΔAUC/ΔLL|','|---|---|---|---|']
m=pd.read_csv(P/'four_seed_domain_task_summary.csv')
for name,label in [('formal_commonLR_SB-SN','selected'),('formal_commonLR_SB-SN','final'),('H200_sameLR0.01_SB-SN','final')]:
 g=m[(m.comparison==name)&(m.readout==label)&(m.axis=='domain')];v=[]
 for d in range(3):
  a=g[(g.domain==d)&(g.metric=='auc')].iloc[0];l=g[(g.domain==d)&(g.metric=='logloss')].iloc[0];v.append(f'{a["mean"]:+.6f}/{l["mean"]:+.6f}')
 lines.append('|'+name+'/'+label+'|'+'|'.join(v)+'|')
lines+=['','''H100末轮最小域d0 AUC−.007803、四负，LL+.079300四差；d1/d2 AUC均值反而正，各三正一负。严格“两较小域改善且最大域下降”共同LR selected/final均0/4；反向的“d0下降、d1和d2都上升”末轮2/4。H200末轮q差仍是d0四负、d1/d2各三正；不能统一描述为小域受益/大域受损。selected两个task的q差分别−.000565（三负一正）/+.000052（二正二负），final−.001614（二正二负）/−.000760（一正三负），宏符号不能代替任务结论。

selected六cell中d1/task1 AUC+.000877四正，d2/task0+.000582四正，其他cell方向混合；末轮d0两个task各四负，d2/task1 LL四改善。这些是相关且未经多重校正的局部边界，不能挑出两格写成总体收益。[全部cell区间](analysis/n17/four_seed_cell_summary.csv)、[所有配方大小取舍](analysis/n17/domain_size_tradeoffs.csv)。

按原SN选中模型的逐域/任务绝对AUC与LL保留于all_test_cell_metrics，结合paired表baseline_auc判断难度；本轮最大域没有因规模大就保证最佳基线，也不把四seed误当独立域样本。更完整支持显著提高绝对性能而q差仍小且阶段依赖，使域大小/支持成为绝对性能的合理解释变量之一，但不足以作为采样效果的唯一主解释。

## 7. 指标权重可改变方向，不能据此选结果

共同.01 H100 final宏cell AUC q差−.001187，固定train-p加权cell AUC却+.000181（二正二负）；eval-p仍−.000560。各区间跨0，权重改变的有限算术事实不建立算法普遍收益。该final宏LL与train-p LL都为正（+.016585/+.003513），不能沿用开发阶段formal末轮的LL翻转。

另一个预先规定条件quota final的SB−SN宏LL+.003246（三差一好），train-p LL−.009701、四改善、点态区间[−.016955,−.002447]；eval-p−.002130、两差两好。共同IID的1–100宏LL+.006265、train-p−.000525也异号。必须一并报告，不能结果后挑权重；训练p、evaluation p及宏cell是不同目标，不是互相纠错的同一指标。[全权重原表](analysis/n17/paired_summary.csv)。

## 8. 机制重评、证据与后续辨别

N2/N3已在[N17机制报告](N17_MECHANISM_ANALYSIS.md)及[逐域任务补表](N17_CELL_MECHANISM_SUPPLEMENT.md)完整分析：自然人口仍有梯度trace的q差，计数项不是B40960 trace主量；固定previous-v线性映射已经反转P块更新方差排序，不能说当前分母非线性必需；P仅占总更新trace约.11%–.20%。N3 C_full一步AUC微小且区间跨0，追加到64/bank仍有精度不足，不能建立长期C/history必要性。旧N2是KR，不能把旧新trace差唯一归为ML人口调整。

**稳定重复现象 / MEDIUM**：同人口完整支持与匹配N的绝对性能差，以及同q继续训练末轮AUC下降/LL升高；q效应依赖LR、阶段、评价权重与cell。**矛盾 / 保留**：旧人工selected强负未在自然共同/同LR条件复现，开发/测试q均值方向也不同；早期、选中、末轮及quota并不统一。**STRONG**：同目标重要性恒等式、共享前缀身份、固定预测上的权重算术。**WEAK**：p为唯一因果变量、Adam历史必要性、严格等效或全局算法胜负。

竞争解释包括支持与重复频率、标签/特征分布、更新/曝光及选择时钟、坐标预条件与局部风险几何。当前控制已反驳“只由p/q标量或单个trace解释全部差异”，未唯一分离所有因素。若另设下一轮，优先在固定支持/同状态下预注册风险投影和坐标预条件对照，以可反证的端点检验机制；不要追逐本轮最优子块或延长到出现正号。本轮N4未自动触发新拓扑，冻结范围均已完成。

本报告完成N17阶段闭环；整项任务仍须core机制、AO04 test和N19/CM19完整实验，不能以本阶段替代全研究完成。''']
# Absolute baseline cells accompany differences; population/phase are fixed.
cells=pd.read_csv(P/('all_test_cell_metrics.csv' if (P/'all_test_cell_metrics.csv').exists() else 'all_test_cell_metrics.csv.gz'))
cells=cells[(cells.group=='formal')&(cells.lr==.01)&cells.selected.astype(bool)]
baselines=cells.groupby(['arm','domain','task'])[['auc','logloss']].mean()
appendix=['', '## 附表：共同LR选中模型test绝对基线', '', '|cell|SN AUC/LL|SB AUC/LL|', '|---|---|---|']
for d in range(3):
 for task in range(2):
  a=baselines.loc[('SN',d,task)];b=baselines.loc[('SB',d,task)]
  appendix.append(f'|d{d}/task{task}|{a.auc:.6f}/{a.logloss:.6f}|{b.auc:.6f}/{b.logloss:.6f}|')
lines+=appendix
(ROOT/'N17_PHASE_ANALYSIS.md').write_text('\n'.join(lines)+'\n')
print('N17 phase report written')
