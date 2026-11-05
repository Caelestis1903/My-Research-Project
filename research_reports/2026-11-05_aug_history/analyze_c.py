"""Build scientific tables from the completed, frozen C scalar records."""
import csv
import hashlib
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).parent
ATTEMPT = ROOT / 'artifacts/aug_history_20261105/attempt_C1'
DATA = ATTEMPT / 'C'
rows = list(csv.DictReader((DATA / 'ENDPOINTS.csv').open()))
hpo = json.loads((DATA / 'HPO_SELECTION.json').read_text())
decision = json.loads((DATA / 'DECISION.json').read_text())['datasets']['movielens']
selected = hpo['datasets']['movielens']['winners']['C']['candidate_id']
methods = [selected, 'C_MERGED', 'C_ROTATED']


def points(candidate, endpoint='selected', split='test'):
    values = {int(r['seed']): r for r in rows if r['candidate_id'] == candidate
              and r['checkpoint'] == endpoint and r['split'] == split}
    assert set(values) == set(range(2022, 2027)), (candidate, endpoint, split)
    return values


def mean(values, key):
    return st.mean(float(r[key]) for r in values.values())


def differences(a, b, key):
    return [float(a[s][key]) - float(b[s][key]) for s in sorted(a)]


def interval(values):
    m = st.mean(values)
    h = 2.7764451051977987 * st.stdev(values) / math.sqrt(5)
    return f'[{m-h:+.9f}, {m+h:+.9f}]'


configs = hpo['datasets']['movielens']['candidates']
jobs = []
for stage in ('I', 'II', 'III', 'IV'):
    jobs += json.loads((DATA / f'movielens/STAGE_{stage}_RESOLVED.json').read_text())['jobs']
assert len(jobs) == 30
assert len({j['job_id'] for j in jobs}) == 30
audit = defaultdict(lambda: dict(epochs=0, steps=0, validations=0, refreshes=0,
                               nonzero_refreshes=0, nonzero_drift_refreshes=0,
                               maximum_drift=0.0, maximum_statistic_trace=0.0))
order = {}; noise = {}; actual_counts = defaultdict(set)
for job in jobs:
    candidate = job['candidates'][0]['id']
    actual_counts[candidate].add(job['seed'])
    for epoch in range(1, 51):
        record = json.loads((ATTEMPT / 'runs' / job['job_id'] / 'epochs' / f'E{epoch:03d}.json').read_text())
        info = audit[candidate]
        info['epochs'] += 1
        info['steps'] += record['cost']['real_adam_steps']
        info['validations'] += record['cost']['validations']
        key = (job['seed'], epoch)
        assert order.setdefault(key, record['order_hash']) == record['order_hash']
        for batch in record['batch_trace']:
            response = batch['response']; step = batch['global_step']
            assert response['teacher_preupdate_step'] == step-1
            assert response['online_preupdate_step'] == step-1
            key = (job['seed'], step)
            base_hash = response['views']['base_noise_hash']
            assert noise.setdefault(key, base_hash) == base_hash
            if response['refreshed']:
                p = response['probe']; trace = p['mean_statistic_trace']; drift = p['mean_drift_norm']
                assert p['reason'] == 'OK'
                assert p['teacher_step'] == step-1 and p['online_step'] == step-1
                info['refreshes'] += 1
                info['nonzero_refreshes'] += trace > 0
                info['nonzero_drift_refreshes'] += drift > 0
                info['maximum_drift'] = max(info['maximum_drift'], drift)
                info['maximum_statistic_trace'] = max(info['maximum_statistic_trace'], trace)
assert all(seeds == set(range(2022, 2027)) for seeds in actual_counts.values())
assert sum(v['epochs'] for v in audit.values()) == 1500
assert sum(v['steps'] for v in audit.values()) == 30000
assert sum(v['validations'] for v in audit.values()) == 3000
(REPORT / 'C_SCALAR_AUDIT.json').write_text(json.dumps(dict(
    trajectories=30, epochs=1500, adam_steps=30000, validations=3000,
    matched_order_identities=len(order), matched_base_noise_identities=len(noise),
    candidates=dict(audit)), indent=2) + '\n')

lines = ['# C阶段实验分析：历史漂移条件增强', '',
    'MovieLens C_FULL四阶段全部完成，六配置各五seed、每条50epoch。下表严格采用冻结的validation选择及test端点；KuaiRand C为DEFERRED_NOT_RUN。', '',
    '## 身份和选择', '',
    '- 实施提交 `933cb2f02790722af2fa4ea704f993b79df29afa`；source hash `d78a938330e5eb8334213c7f0beed66926956c35e1cff90c180260d12fd331e2`。',
    '- selection hash `' + decision['selection_hash'] + '`；ENDPOINTS SHA-256 `' + hashlib.sha256((DATA/'ENDPOINTS.csv').read_bytes()).hexdigest() + '`。',
    f'- 搜索赢家 `{selected}`；最终validation保留 `{decision["selected_candidate"]}`；下一动作为 `{decision["next_action"]}`。',
    '- 新增30条独立online/Adam/teacher轨迹、1500epoch、30000次Adam和3000次模型validation；γ0父F/online仅复用五seed冻结端点。每条轨迹的online与fixedEMA是同一训练的两种估计器。',
    '- 四阶段依次完成后冻结W12、W123、C_WINNER；仅最终C*及同γ/K的MERGED、ROTATED进入test，未按test重选符号、强度或刷新频率。', '',
    '![C匹配控制比较](figures/C_COMPARISON.png)', '',
    '左为test selected对F的五seed配对均差、95%CI与逐seed点，单位10⁻³ AUC；右为相同方案在selected/E10/E50的增量。', '',
    '## 有限搜索与validation选择', '',
    '| 配置 | γ | K | V AUC均值±SD | 对F最差格ΔAUC | 六格保护 | selected epochs | 核心worker秒 |',
    '|---|---:|---:|---:|---:|---|---|---:|']
for r in hpo['datasets']['movielens']['hpo_rows']:
    worst = min(r[f'd{i}t{j}_delta_F'] for i in range(3) for j in range(2))
    lines.append(f'| {r["candidate_id"]} | {r["gamma"]:g} | {r["refresh_steps"]} | {r["score"]:.9f}±{r["score_sd"]:.9f} | {worst:+.9f} | {r["cell_protected"]} | {r["selected_epochs"]} | {r["measured_cost_seconds"]:.3f} |')
lines += ['', '## 冻结test与总收益/组件增量', '',
    '置信区间为seed配对Student-t(df4)，同向数仅描述。正式模型比较独立报告平均AUC≥+.003、每格均差≥−.001及平均LL下降；内部组件比较不另设+.003门。', '',
    '| 方法 | 比较对象 | test AUC均值±SD | ΔAUC | 配对95%CI | test LL | ΔLL | AUC正向seed | 最差格ΔAUC | AUC资格/格保护/LL下降 |',
    '|---|---|---:|---:|---|---:|---:|---:|---:|---|']
for method in methods:
    a = points(method)
    controls = ['F', 'M3oE_fixed1', 'HyperGate', 'AutoIFS']
    if method == selected:
        controls += ['C_MERGED', 'C_ROTATED']
    for control in controls:
        b = points(control); delta = differences(a, b, 'macro_auc'); loss = differences(a, b, 'macro_logloss')
        worst = min(st.mean(differences(a, b, f'd{i}t{j}_auc')) for i in range(3) for j in range(2))
        qualification = str(st.mean(delta) >= .003) if control in ['M3oE_fixed1', 'HyperGate', 'AutoIFS'] else '内部增量'
        sd = st.stdev(float(r['macro_auc']) for r in a.values())
        lines.append(f'| {method} | {control} | {mean(a,"macro_auc"):.9f}±{sd:.9f} | {st.mean(delta):+.9f} | {interval(delta)} | {mean(a,"macro_logloss"):.9f} | {st.mean(loss):+.9f} | {sum(x>0 for x in delta)}/5 | {worst:+.9f} | {qualification}/{worst>=-.001}/{st.mean(loss)<0} |')
lines += ['', '## 逐seed端点', '',
    '| 方法 | seed | epoch | AUC | LL | ΔAUC对F | ΔLL对F |',
    '|---|---:|---:|---:|---:|---:|---:|']
for method in methods:
    a = points(method); b = points('F')
    for s in sorted(a):
        lines.append(f'| {method} | {s} | {a[s]["epoch"]} | {float(a[s]["macro_auc"]):.9f} | {float(a[s]["macro_logloss"]):.9f} | {float(a[s]["macro_auc"])-float(b[s]["macro_auc"]):+.9f} | {float(a[s]["macro_logloss"])-float(b[s]["macro_logloss"]):+.9f} |')
lines += ['', '## 六格与任务/训练阶段', '',
    '| 方法 | cell | AUC | LL | ΔAUC对F | ΔLL对F |',
    '|---|---|---:|---:|---:|---:|']
for method in methods:
    a = points(method); b = points('F')
    for i in range(3):
        for j in range(2):
            cell = f'd{i}t{j}'
            lines.append(f'| {method} | {cell} | {mean(a,cell+"_auc"):.9f} | {mean(a,cell+"_logloss"):.9f} | {st.mean(differences(a,b,cell+"_auc")):+.9f} | {st.mean(differences(a,b,cell+"_logloss")):+.9f} |')
lines += ['', '| 方法 | endpoint | ΔMacro AUC | ΔMacro LL | Δtask0 AUC | Δtask1 AUC | ΔTraffic AUC | ΔPooled AUC |',
    '|---|---|---:|---:|---:|---:|---:|---:|']
for method in methods:
    for endpoint in ['selected', 'E10', 'E50']:
        a = points(method, endpoint); b = points('F', endpoint)
        values = [st.mean(differences(a,b,k)) for k in ['macro_auc','macro_logloss','task0_macro_auc','task1_macro_auc','sample_weighted_cell_auc','pooled_task_mean_auc']]
        lines.append('| '+method+' | '+endpoint+' | '+' | '.join(f'{v:+.9f}' for v in values)+' |')
lines += ['', '## online与固定EMA', '',
    '各估计器分别按自身validation曲线选点；下表的online比较改变训练轨迹，EMA−online同时含估计与选点差异。', '',
    '| 方法online | test AUC | test LL | ΔAUC对父online | ΔLL对父online | fixedEMA−online AUC | fixedEMA−online LL |',
    '|---|---:|---:|---:|---:|---:|---:|']
for method in methods:
    a = points(method+'_online'); b = points('PARENT_online'); ema = points(method)
    values = [mean(a,'macro_auc'),mean(a,'macro_logloss'),st.mean(differences(a,b,'macro_auc')),st.mean(differences(a,b,'macro_logloss')),st.mean(differences(ema,a,'macro_auc')),st.mean(differences(ema,a,'macro_logloss'))]
    lines.append('| '+method+'_online | '+' | '.join(f'{v:+.9f}' for v in values)+' |')
lines += ['', '## 实际机制与成本', '',
    '| 配置 | epoch | Adam | V次数 | 刷新 | 非零统计刷新 | 非零漂移刷新 | 最大mean drift norm | 最大mean statistic trace |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for candidate, a in audit.items():
    lines.append(f'| {candidate} | {a["epochs"]} | {a["steps"]} | {a["validations"]} | {a["refreshes"]} | {a["nonzero_refreshes"]} | {a["nonzero_drift_refreshes"]} | {a["maximum_drift"]:.9g} | {a["maximum_statistic_trace"]:.9g} |')
costs = list(csv.DictReader((DATA/'COSTS.csv').open()))
lines += ['', '| 配置 | training秒 | probe秒 | validation秒 | 核心worker计时合计秒 |',
    '|---|---:|---:|---:|---:|']
for config in configs:
    values = [r for r in costs if r['candidate_id'] == config['id']]
    assert len(values) == 5
    sums = [sum(float(r[k]) for r in values) for k in ['training_seconds','probe_seconds','validation_seconds','gpu_seconds']]
    lines.append('| '+config['id']+' | '+' | '.join(f'{v:.3f}' for v in sums)+' |')
lines += ['',
    '30条轨迹对应seed的250个epoch数据顺序、5000个step的基础Gaussian noise hash跨六配置一致。teacher/online preupdate时钟在全部30000步均为当前step−1。各轨迹用自身θ/φ/statistic，不能从共享噪声或相同γ/K推断不同轨迹实际协方差谱相同。ROTATED只对自身统计做固定正交旋转；MERGED独立重算合并任务统计。', '',
    '成本为各进程记录的training+probe+validation时间之和，包含并发等待，不是整机elapsed或CUDA kernel独占时间；源字段gpu_seconds/wall_seconds均使用该合计。未额外加总sampling子项。RAM-only参数保留至冻结评价完成，原始逐seed、全曲线、配对差与运行事件见本attempt。', '',
    '## 科学判断', '',
    '- **STRONG，描述性事实：** C3_WEAK、MERGED、ROTATED的test selected AUC均值均低于F，LL均值均高于F；全部1375次刷新中1345次统计和漂移非零，其余30次对应各轨迹初始化。条件分支实际运行。',
    '- **MEDIUM：** 冻结最终方案MERGED对F的AUC均差−0.000667190，五seed均负，95%CI[−0.001039767,−0.000294612]，LL+0.000357811。当前C方案未建立对固定EMA的稳定组件增量；完整方案对普通M3oE的AUC+0.003499743、六格保护通过、LL+0.001289159，整体排序收益仍成立。',
    '- **WEAK，任务分解必要性：** 任务C*对MERGED AUC+0.000397735、4/5正向，但95%CI跨0且LL+0.000237514；对ROTATED仅+0.000018904、2/5正向、CI跨0，LL+0.000101474。两组必要性比较没有一致支持。',
    '- **MEDIUM，阶段与指标分化：** C3_WEAK对F的validation+0.000082331转为test selected−0.000269455；MERGED在E10 AUC+0.000059686、selected−0.000667190、E50−0.000237074。ROTATED的E50 LL−0.001232843，而selected LL+0.000493851。单阶段改善不能代替所定义的selected目标。',
    '- **WEAK，竞争解释：** 漂移方向可能没有提供超出固定EMA的排序信息；各向异性可能主要改变平滑/正则强度；闭环统计随自身训练变化可能放大选点差异。三个online均值相对父Gaussian online下降，说明变化已进入训练轨迹，不能全部归因于EMA读出。',
    '- MERGED及ROTATED六格AUC对F均下降，C3_WEAK仅d0t1略升；没有稳定的小域改善/大域下降模式。最能分辨方向信息与各向异性强度的下一实验，是固定train-only参考统计与每步谱后配对真实方向/固定旋转方向，并保留F；新一轮需预注册、五seed及封存自然人口评价。本轮按冻结规则STOP_AND_REPORT。', '']
(REPORT/'C_STAGE_ANALYSIS.md').write_text('\n'.join(lines))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
for n, method in enumerate(methods):
    delta = differences(points(method), points('F'), 'macro_auc')
    m = st.mean(delta); h = 2.7764451051977987*st.stdev(delta)/math.sqrt(5)
    axes[0].errorbar(n, m*1e3, yerr=h*1e3, fmt='o', capsize=4, color='#1167b1')
    axes[0].scatter([n+(s-2)*.035 for s in range(5)], [d*1e3 for d in delta], s=12, color='#777777', alpha=.6)
    values = [st.mean(differences(points(method,e),points('F',e),'macro_auc'))*1e3 for e in ['selected','E10','E50']]
    axes[1].plot(range(3), values, marker='o', label=method)
axes[0].set_xticks(range(3), ['Task C*', 'Merged', 'Rotated'])
axes[0].set_title('MovieLens test selected (paired 95% CI, n=5)')
axes[1].set_xticks(range(3), ['Selected', 'E10', 'E50'])
axes[1].set_title('Training stage and checkpoint selection'); axes[1].legend()
for ax in axes:
    ax.axhline(0, color='#333333', lw=.8); ax.grid(axis='y', alpha=.2)
    ax.set_ylabel('Test AUC change vs F (x 1e-3)')
directory=REPORT/'figures'; directory.mkdir(exist_ok=True)
fig.savefig(directory/'C_COMPARISON.png', dpi=180); fig.savefig(directory/'C_COMPARISON.svg'); plt.close(fig)
print('C analysis generated; 30 trajectories / 1500 epochs / 30000 Adam / 3000 V audited')
