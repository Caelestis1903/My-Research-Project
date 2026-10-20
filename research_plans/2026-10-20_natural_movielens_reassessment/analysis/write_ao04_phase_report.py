"""Primary complete AO04 interpretation; no new training or model selection."""
from pathlib import Path
import json
import pandas as pd
from write_core_mechanism_report import table

REPORT = Path(__file__).resolve().parents[1]
OUT = REPORT/'analysis/ao04'


def write():
    status = json.loads((OUT/'analysis_status.json').read_text())
    assert status['complete'] and status['full_prediction_cells_recomputed'] == 504
    summary = pd.read_csv(OUT/'paired_summary.csv')
    endpoints = summary[(summary.task == -1) & summary.checkpoint.isin(['selected','final']) &
        summary.metric.isin(['macro_auc','macro_logloss'])]
    macro = pd.read_csv(OUT/'seed_macro_task_metrics.csv')
    macro = macro[(macro.task == -1) & macro.checkpoint.isin(['selected','final'])].groupby(
        ['checkpoint','arm'],as_index=False)[['macro_auc','macro_logloss']].mean()
    domain_task = pd.read_csv(OUT/'four_seed_cell_domain_task_summary.csv')
    domain_task = domain_task[(domain_task.axis != 'cell') & domain_task.checkpoint.isin(['selected','final'])]
    absolute = pd.read_csv(OUT/'four_seed_absolute_cell_means.csv')
    absolute = absolute[(absolute.arm == 'B0') & absolute.checkpoint.isin(['selected','final'])]
    weights = pd.read_csv(OUT/'four_seed_fixed_weight_summary.csv')
    weights = weights[weights.checkpoint.isin(['selected','final'])]
    tradeoff = pd.read_csv(OUT/'seed_domain_tradeoffs.csv')
    tradeoff = tradeoff[tradeoff.checkpoint.isin(['selected','final'])].groupby(
        ['contrast','checkpoint'],as_index=False)[['two_smaller_AUC_up_largest_down',
        'smallest_AUC_down_two_larger_up','all_domain_AUC_up','all_domain_LL_improved']].sum()
    s = '''# AO04完整阶段分析：自然人口的融合自由度与局部敏感性

## 1. 完成范围与结论

B0/B1/B2×四seed共12配置，全部完成原50轮、1000main/50arch更新及40,008,350 main曝光，architecture实际曝光2,009,934。原selected规则为全validation六cell宏AUC严格最大值的最早轮次，影子停止仅记录、不提前终止。每seed三臂共用配对初始状态SHA。B0对称融合，B1每domain×task独立q共6个，B2每domain跨任务共用q共3个；未新增B3/B4或结果后调参。

12训练、12局部诊断及配置/选模冻结均已闭环。一个原始root test动作覆盖全部12模型，receipt记录optimization_steps=0，冻结SHA和所有模型/配置/selection身份一致。test保存selected、final及E10/E20/E30/E40/E50七个读数，504个逐域逐任务AUC/LL均已从NPZ独立复算，残差<1e−12，并逐数组核对row-ID、labels和共同test人口。E50与final是同训练阶段的重复命名，不能当额外独立证据。

**MEDIUM：完整自然人口没有复现清晰、稳定的B1/B2整体性能收益。** B1−B0 selected宏AUC−.0000555（二正二负），final−.0007756（一正三负）；B2−B0为−.0001392/−.0004123，各二正二负。所有12个端点宏AUC/LL点态四seed区间都跨0，不建立优越、劣效或严格等效。

**MEDIUM有限敏感性、WEAK全局识别或收益：** 36个固定开发探针点的q Jacobian均非零，因此不能简单把本次缺少完整收益归为该有限探针完全不敏感；但Jacobian非零不证明参数全局唯一可识别，也不保证增加自由度会改善测试风险。该判断与历史阴性结果并存。

本次测试复用历史已使用人口，属于探索性复现；四优化seed不是四个独立数据总体。B0 seed2023首次attempt exit−11、无有效receipt，完整同seed重跑已成功，失败attempt保留，不计科研阴性或额外seed；不能把空stdout当作已知零训练。

## 2. 宏指标、四seed及阶段

### 绝对性能

__ABS_MACRO__

### 候选−参照

__ENDPOINTS__

LL负值表示改善。B1 selected相对B0平均LL−.001328但二好二差，末轮+.005364三差一好；B2 selected+.001134二差二好、final+.002158三差一好。B1−B2 selected AUC+.0000843但二正二负，LL−.002462三好一差，final AUC−.0003634一正三负且LL+.003206二好二差。排序/概率损失与阶段不能合并成一个赢家。

共同点也保留反例：E20 B1−B0 AUC−.000929四负、点态t95[−.001712,−.000146]，LL+.001531四差但区间跨0；E40 AUC+.000475三正一负、LL−.002861三好一差，两个区间跨0；末轮又负。E10 B2−B0 AUC四负但区间仍跨0。所有检查点已在结果前固定，不按最有利点选择结论；这些相关点及多重比较未经全族校正。

各臂选中模型明显强于自身末轮，但这种共同后期退化不能单独识别B1/B2相对差异的原因。开发selected B1−B0 AUC−.000334一正三负，test变为−.0000555二正二负；B1−B2由开发−.000300变成test+.0000843。保留这种跨split/指标差异，不用开发最大值替代测试。

## 3. 逐域、逐任务与基线难度

自然三个物理域训练N=[155949,318599,325619]；test N=[27687,41434,30900]。B0六cell基线如下，auc/logloss是四seed均值。

__BASELINE__

d2最大却在两任务selected AUC都最低；d0最小却非最低。LL的排序又不完全相同，例如task0 d0风险高于d2，不能用大小、AUC或标签熵中的单个量给域定义统一难度。

下表按域平均两任务、按任务平均三域；每行仍四seed配对。完整六cell比较另存CSV。

__DOMAIN_TASK__

selected B1−B0 d0/d1/d2 AUC−.000026/+.000448/−.000589，只有d1均值正且三正一负；final三个域均值均负。selected d1/task0 AUC+.000329、四正、点态区间[+.000022,+.000635]，是唯一端点cell AUC/LL区间不含0的读数；它未经多重比较校正，也没有形成宏收益。B2−B0 final最小d0 AUC−.002975四负、按域点态区间[−.004805,−.001146]；d2均值+.001800却只有二正二负。不能用少数正cell给方案背书，也不能因宏均值接近0忽略最小域的条件损害。

两个任务selected B1−B0 AUC约+.000050/−.000161，各二正二负；final−.001159/−.000392，各一正三负。B2 selected两任务AUC均二正二负，final task0一正三负/task1二正二负；没有稳定跨任务收益。

## 4. 大小取舍与评价权重

严格取舍要求同一个seed下d0、d1域平均AUC都上升、最大d2下降，不由宏均值判断。下表是四seed中满足该条件的次数。

__TRADEOFF__

B1−B0 selected/final各1/4，B2−B0均0/4；B1−B2 selected1/4、final3/4。后者末轮宏AUC更差，且四seed平均d1差为负（其中一个负差幅度较大），不能把3/4改写为所有均值或所有seed的规律。参照更换就改变取舍状态；本轮没有单独操纵域大小，规模作为唯一解释仍WEAK。

__WEIGHTS__

train-p与eval-p均为预定cell AUC/LL加权，不是pooled AUC。selected B1−B0宏/训练p/test p的AUC约−.0000555/−.0000662/−.0000034，B2−B0约−.0001392/−.0001750/−.0002167，所有相应区间跨0。权重改变小数值和效用定义，没有产生明确整体增益；不得结果后挑权重给候选排名。

## 5. 局部几何与完整性能是不同证据

每run在initial/selected/final、每域原validation前8行（24行）诊断专家差、a_D(1−b_D)、logit对q/rho Jacobian及q±.1扰动风险，零训练更新、零test读取。B0载入同theta的B1 q=0反事实，不是重新训练的B1。单类探针AUC保持未定义，不填.5。

36点q Jacobian RMS非零。B1 selected .005227–.017806、final .018269–.033461；B2 selected .008757–.022449、final .029844–.050672。a_D(1−b_D)初始.196612，selected约.194654–.196030、final .190259–.193173，专家差RMS亦非零。相同初始函数B2的raw q Jacobian RMS约为B1的sqrt(2)倍来自坐标数归一化，不是更好可识别性或性能。

raw q扰动既不是相同rho幅度也不是相同函数距离，不能以小探针扰动比较给两种参数化作全局优劣排名。当前数据只排除“这些探针点的融合自由度完全没有一阶功能影响”这一强解释；不能排除正则/有效位移、有限样本噪声、选择准则或表达冗余解释，也不证明某个q是长期必要中介。完整几何与开发阶段见[AO04_DEVELOPMENT_ANALYSIS.md](AO04_DEVELOPMENT_ANALYSIS.md)。

## 6. 历史对应、证据强度与后续问题

历史MovieLens AO04实际是人工 **d0_large**，不是其它主线常用d2_large；只有2022/23两个seed。旧B1−B0 selected test AUC+.000015592（一正一负）、LL+.00062045（两差），B1−B2 AUC−.000012657（一正一负）。旧自然KR只有一个完整B1 seed，validation AUC+.001494而test−.008385。它们均不支持普遍的融合自由度收益，原有限/阴性结论保留；不能与新四seed池化构造统计确认。旧预测缺row-ID，旧模型重加载或全局Jacobian身份未在本次冒称完成。

- **STRONG有限事实**：12训练完整预算、固定配置/选模、零更新test、504cell预测复算及归档SHA。
- **MEDIUM条件现象**：局部敏感性非零而完整收益不稳定；阶段、域任务与参照会改变效果方向；最小d0的B2末轮条件损害。
- **WEAK**：严格等效、全局参数识别、域大小主因、某个融合自由度的长期必要性或跨总体收益。没有据小均值形成新算法主张。

最有辨别力的后续候选应先用同初始函数/固定theta、等函数距离或等实际位移的参数化对照，检验是否仍出现局部敏感性与风险效用脱节；若仍无效，则反驳“只需释放该自由度即可提高性能”。只有预先指定完整候选/选择规则后才用新总体确认。当前没有理由结果后扩B3/B4、挑cell或追加seed寻找正号；这些问题不自动新增本次实验。

## 7. 证据索引与整体状态

- [完整状态及504cell计数](analysis/ao04/analysis_status.json)、[预测复算](analysis/ao04/prediction_recalculation.csv)、[训练/测试/探针来源](analysis/ao04/source_index.csv)。
- [逐seed宏/任务](analysis/ao04/seed_macro_task_metrics.csv)、[逐cell差](analysis/ao04/paired_cell_deltas.csv)、[四seed域任务/六cell](analysis/ao04/four_seed_cell_domain_task_summary.csv)、[权重](analysis/ao04/four_seed_fixed_weight_summary.csv)。
- [局部敏感性](analysis/ao04/local_sensitivity.csv)、[扰动风险](analysis/ao04/local_probe_perturbation_risks.csv)、[旧记录](analysis/ao04_history/historical_selected_summary.csv)。
- [冻结及release](analysis/ao04_freeze/)、[实际751文件SHA/113小JSON快照](analysis/family_artifacts/AO04/all/)。大工件位于服务器/home_new/xzf_natural_movielens_reassessment/AO04，672,123,849 bytes。

AO04本阶段已完成；N19/CM19及最终总报告仍待闭环，不能以此宣称整项任务完成。
'''
    replacements = {
        '__ABS_MACRO__': table(macro),
        '__ENDPOINTS__': table(endpoints[['checkpoint','contrast','metric','reference_mean','candidate_mean',
            'mean','ci95_low','ci95_high','positive','negative']]),
        '__BASELINE__': table(absolute),
        '__DOMAIN_TASK__': table(domain_task[['contrast','checkpoint','axis','domain','task','metric',
            'mean','ci95_low','ci95_high','positive','negative']].fillna('')),
        '__TRADEOFF__': table(tradeoff),
        '__WEIGHTS__': table(weights[['checkpoint','weighting','contrast','metric','mean',
            'ci95_low','ci95_high','positive','negative']]),
    }
    for token, value in replacements.items():
        s = s.replace(token, value)
    (REPORT/'AO04_PHASE_ANALYSIS.md').write_text(s)
    print('Wrote complete AO04_PHASE_ANALYSIS.md.')


if __name__ == '__main__':
    write()
