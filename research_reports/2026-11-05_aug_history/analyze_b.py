"""Rebuild the B stage analysis from frozen scalar endpoints."""
import csv
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'artifacts/aug_history_20261105/attempt1/B'
rows = list(csv.DictReader((DATA / 'ENDPOINTS.csv').open()))
hpo = json.loads((DATA / 'HPO_SELECTION.json').read_text())
decision = json.loads((DATA / 'DECISION.json').read_text())

def points(dataset, candidate, endpoint='selected', split='test'):
    result = {int(r['seed']): r for r in rows if r['dataset'] == dataset
              and r['candidate_id'] == candidate and r['checkpoint'] == endpoint
              and r['split'] == split}
    assert set(result) == set(range(2022, 2027))
    return result

def average(data, key):
    return st.mean(float(r[key]) for r in data.values())

def paired(a, b, key):
    return [float(a[s][key]) - float(b[s][key]) for s in sorted(a)]

def interval(values):
    mean = st.mean(values)
    half = 2.7764451051977987 * st.stdev(values) / math.sqrt(5)
    return f'[{mean-half:+.9f}, {mean+half:+.9f}]'

lines = ['# B阶段实验分析：增强响应定向参数EMA', '',
    'B完成两数据集各五seed、每条50epoch和15个非反馈影子。test的D−F平均Macro AUC为MovieLens +0.000000104、KuaiRand −0.000004946。本轮未建立具有实用幅度的独立方向增量；针对当前模型、网格和预算的证据强度为 **STRONG**。两数据按冻结的validation规则进入A_FULL，test未改变下一路线。', '',
    '## 身份与完整性', '',
    '- PLAN_REF `2645f1f724f52f5ed5dea4b479335c0dc12fabcd`；父源码 `c8675ae2923c15b6965884b140b5b646690cd95e`；B实施提交 `79690772dedefafa24b80d1817aca8c61907be5b`。',
    '- B source hash `9baac157d4574ccc2ec02a9db42cd365ec918346c32b1b3ace2dcc6f65893f2a`；selection hash `' + decision['selection_hash'] + '`。',
    '- ENDPOINTS.csv SHA-256 `6d9b9cb7b1c188c732018b416b946386170834aa664b19f3ed5180976bc2b935`。',
    '- 10条独立online、500个连续epoch、17250次真实Adam；7500次影子validation与500次online validation。每候选每数据完整五seed，2022–2026。',
    '- FULLTRAIN_NATURAL80_10_10、train-only预处理、batch40960、50epoch、fixed1、原Gaussian d0、普通coupled Adam；validation六格Macro AUC最早严格最大值选点。模型/Adam/Q/checkpoint仅RAM。', '',
    '- test沿用历史研究划分；本轮在配置与选点冻结后统一评价，阶段动作于test前冻结。', '',
    '![B完整方案与组件比较](figures/B_COMPARISON.png)', '',
    '左图为五seed均值±样本SD；右图为D与各控制的配对差均值、seed级95%CI及逐seed点，单位为微AUC（10⁻⁶）。', '',
    '## 家族选择', '',
    '各家族同预算独立调参；数值为五seed selected均值。全部候选见HPO_SUMMARY与VALIDATION_CURVES。', '',
    '| 数据集 | 家族/配置 | validation AUC | test AUC均值±SD | test LL均值±SD |',
    '|---|---|---:|---:|---:|']
for dataset, selected in hpo['datasets'].items():
    for family, candidate in [('F', 'F')] + [(f, v['candidate_id']) for f, v in selected['winners'].items()]:
        a = points(dataset, candidate)
        v = points(dataset, candidate, split='validation')
        auc_sd = st.stdev(float(r['macro_auc']) for r in a.values())
        ll_sd = st.stdev(float(r['macro_logloss']) for r in a.values())
        lines.append(f'| {dataset} | {family}/{candidate} | {average(v,"macro_auc"):.9f} | {average(a,"macro_auc"):.9f}±{auc_sd:.9f} | {average(a,"macro_logloss"):.9f}±{ll_sd:.9f} |')
lines += ['', '## 逐seed D与F', '',
    '| 数据集 | seed | epoch D/F | D AUC | F AUC | ΔAUC | D LL | F LL | ΔLL |',
    '|---|---:|---|---:|---:|---:|---:|---:|---:|']
for dataset, selected in hpo['datasets'].items():
    a = points(dataset, selected['winners']['D']['candidate_id'])
    b = points(dataset, 'F')
    for seed in a:
        values = [float(a[seed]['macro_auc']), float(b[seed]['macro_auc'])]
        values += [values[0]-values[1], float(a[seed]['macro_logloss']), float(b[seed]['macro_logloss'])]
        values += [values[3]-values[4]]
        lines.append(f'| {dataset} | {seed} | {a[seed]["epoch"]}/{b[seed]["epoch"]} | ' + ' | '.join(f'{v:.9f}' for v in values) + ' |')
lines += ['', '## 总收益与组件增量', '',
    '单位为seed，SD为配对差样本SD，95%区间为Student-t(df4)。区间与同向数用于描述。正式AUC均差≥+.003、六格均差≥−.001、LL均差<0分别判断；内部控制报告实际组件增量，不增加+.003门。ML主发表参照为M3oE，KR为HyperGate；三发表模型全部保留，各自训练设置沿用源报告，HyperGate保留UNVERIFIED_PROPOSAL超参标记。', '',
    '| 数据集 | D比较对象 | ΔAUC | SD | 95% CI | ΔLL | AUC正向seed | 最差格ΔAUC | 正式AUC资格/格保护/LL下降 |',
    '|---|---|---:|---:|---|---:|---:|---:|---|']
for dataset, selected in hpo['datasets'].items():
    a = points(dataset, selected['winners']['D']['candidate_id'])
    controls = ['F'] + [x['candidate_id'] for f, x in selected['winners'].items() if f != 'D']
    controls += list(selected['matched_at_D'].values()) + ['M3oE_fixed1', 'HyperGate', 'AutoIFS']
    for candidate in dict.fromkeys(controls):
        b = points(dataset, candidate)
        delta = paired(a, b, 'macro_auc'); gain = st.mean(delta)
        loss = st.mean(paired(a, b, 'macro_logloss'))
        worst = min(st.mean(paired(a, b, f'd{d}t{t}_auc')) for d in range(3) for t in range(2))
        qualification = str(gain >= .003) if candidate in ['M3oE_fixed1','HyperGate','AutoIFS'] else '内部增量'
        lines.append(f'| {dataset} | {candidate} | {gain:+.9f} | {st.stdev(delta):.9f} | {interval(delta)} | {loss:+.9f} | {sum(v>0 for v in delta)}/5 | {worst:+.9f} | {qualification}/{worst>=-.001}/{loss<0} |')
lines += ['', '## 六格、任务与训练阶段', '',
    '| 数据集 | cell | D AUC | D LL | ΔAUC对F | ΔLL对F |', '|---|---|---:|---:|---:|---:|']
for dataset, selected in hpo['datasets'].items():
    a = points(dataset, selected['winners']['D']['candidate_id']); b = points(dataset, 'F')
    for d in range(3):
        for t in range(2):
            cell = f'd{d}t{t}'
            lines.append(f'| {dataset} | {cell} | {average(a,cell+"_auc"):.9f} | {average(a,cell+"_logloss"):.9f} | {st.mean(paired(a,b,cell+"_auc")):+.9f} | {st.mean(paired(a,b,cell+"_logloss")):+.9f} |')
lines += ['', '| 数据集 | endpoint | D AUC | ΔAUC对F | ΔLL对F | Δtask0 AUC | Δtask1 AUC | ΔTraffic AUC | ΔPooled AUC |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for dataset, selected in hpo['datasets'].items():
    for endpoint in ['selected', 'E10', 'E50']:
        a = points(dataset, selected['winners']['D']['candidate_id'], endpoint); b = points(dataset, 'F', endpoint)
        values = [average(a, 'macro_auc')] + [st.mean(paired(a,b,k)) for k in ['macro_auc','macro_logloss','task0_macro_auc','task1_macro_auc','sample_weighted_cell_auc','pooled_task_mean_auc']]
        lines.append('| '+dataset+' | '+endpoint+' | '+' | '.join(f'{v:+.9f}' for v in values)+' |')
lines += ['', '## 证据、竞争解释和后续', '',
    '1. **STRONG：D/M/S与F几乎重合。** ML selected D的AUC有3/5 seed正向，KR为1/5；共同E10/E50亦未出现有实用幅度的方向收益。固定F重现历史均值，完整Gaussian+EMA相对普通M3oE的总收益继续保留。',
    '2. **STRONG：响应分支实际启用。** ML500次、KR1225次刷新全部D秩2/M秩1，反事实前后state hash全部相同；最大FP32正交误差5.25e−8/3.54e−8。没有Q始终为空或分支未运行的现象。',
    '3. **MEDIUM：验证、test和阶段方向有分化。** KR D−F validation +9.98e−6、test selected −4.95e−6；E10 +4.77e−5、E50 −1.29e−5。ML selected LL微降，E10/E50微升。微小差异对阶段与选点敏感。',
    '4. **WEAK：域大小是本组件效果的主解释。** ML域人口155949/318599/325619，KR为172782/1743898/89452。D−F六格没有明显小域改善/大域下降的分配，两数据d0t0均微降。ML最大d2普通M3oE AUC低于较小d0；KR最小d2亦不总是基准AUC最低的域。域大小、基准排序难度和任务正例率分别考察。',
    '5. **竞争解释。** 全局时间尺度已足够；当前低秩投影对历史修正幅度小；当前Adam响应方向没有额外有效信息。日志支持响应非零且满秩；innovation投影能量尚无充分日志区分后两者，解释强度为WEAK。',
    '6. **区分实验为已定义A。** 两数据匹配B_SIMPLE_SUFFICIENT，继续同预算分组自适应/全局自适应/固定profile/统一EMA比较，不扩大B网格、不按test追加候选。', '',
    '## 成本与修订', '',
    '累计worker时间20563.108秒：训练8206.223、响应1117.640、完整验证11239.245秒；并发阶段墙钟由RUNNER_REPORT起止时间另记。GPU allocated峰值ML/KR约1.329/1.544GB，reserved约1.669/1.873GB；nvidia-smi总占用包含其他共享作业。',
    '首轮独立审查修复resume入口、存活owner保护、冲突事务RAM回滚三个问题。运行中因内存占用增长轮换四个worker，分别从KR2023 E27、2022 E35、2024 E39、2025 E40恢复；最终各seed恰好50个正式提交epoch。未提交部分物理成本见RUNNER_REPORT。配置、batch、候选和数学规则保持冻结。',
    '全部原值、逐seed六格、源/数据/模型hash和选点身份见 `artifacts/aug_history_20261105/attempt1/B/ENDPOINTS.csv`；全部候选曲线、HPO、配对与成本见同目录CSV，原始epoch标量见attempt1/runs。B模型RAM已按授权在评价后释放。A/C及完整任务结论由最终报告汇总。', '']
(Path(__file__).parent / 'B_STAGE_ANALYSIS.md').write_text('\n'.join(lines))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
labels = {'F':'Gaussian + fixed EMA', 'M3oE_fixed1':'M3oE fixed1', 'HyperGate':'HyperGate', 'AutoIFS':'AutoIFS'}
for index, (dataset, selected) in enumerate(hpo['datasets'].items()):
    directed = selected['winners']['D']['candidate_id']
    candidates = ['M3oE_fixed1', 'HyperGate', 'AutoIFS', 'F', directed]
    ax = axes[index, 0]
    for position, candidate in enumerate(candidates):
        data = points(dataset, candidate)
        values = [float(r['macro_auc']) for r in data.values()]
        ax.errorbar(position, st.mean(values), yerr=st.stdev(values), fmt='o', capsize=4,
                    color='#1167b1' if candidate == directed else '#666666')
    ax.set_xticks(range(5), [labels.get(c, 'B directed') for c in candidates], rotation=20, ha='right')
    ax.set_ylabel('Test macro AUC (mean +/- SD)')
    ax.set_title(dataset.title())
    ax.grid(axis='y', alpha=.2)
    ax = axes[index, 1]
    controls = [('F','F')] + [(f+'*', selected['winners'][f]['candidate_id']) for f in ['U','M','S']]
    controls += [(f+'@D', c) for f, c in selected['matched_at_D'].items()]
    unique = {}
    for label, candidate in controls:
        unique[candidate] = unique.get(candidate, '') + ('/' if candidate in unique else '') + label
    a = points(dataset, directed)
    for position, (candidate, label) in enumerate(unique.items()):
        values = paired(a, points(dataset, candidate), 'macro_auc')
        center = st.mean(values); half = 2.7764451051977987 * st.stdev(values) / math.sqrt(5)
        ax.errorbar(position, center*1e6, yerr=half*1e6, fmt='o', color='#1167b1', capsize=4)
        ax.scatter([position+(i-2)*.035 for i in range(5)], [v*1e6 for v in values],
                   color='#777777', alpha=.55, s=12)
    ax.axhline(0, color='#333333', linewidth=.8)
    ax.set_xticks(range(len(unique)), list(unique.values()))
    ax.set_ylabel('D - control AUC (x 1e-6)')
    ax.set_title('Paired component changes, n=5 (95% CI)')
    ax.grid(axis='y', alpha=.2)
directory = Path(__file__).parent / 'figures'
directory.mkdir(exist_ok=True)
fig.savefig(directory/'B_COMPARISON.png', dpi=180)
fig.savefig(directory/'B_COMPARISON.svg')
plt.close(fig)
print('B_STAGE_ANALYSIS.md generated')
