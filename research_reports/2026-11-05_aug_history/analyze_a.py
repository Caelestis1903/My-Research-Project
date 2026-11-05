"""Rebuild route A tables and plots from immutable scalar outputs."""
import csv
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATTEMPT = ROOT / 'artifacts/aug_history_20261105/attempt_A1'
DATA = ATTEMPT / 'A'
rows = list(csv.DictReader((DATA / 'ENDPOINTS.csv').open()))
hpo = json.loads((DATA / 'HPO_SELECTION.json').read_text())
decision = json.loads((DATA / 'DECISION.json').read_text())
jobs = json.loads((ATTEMPT / 'PREPARED.json').read_text())['jobs']
groups = ['S','D0','D1','D2','T0','T1','DT00','DT01','DT10','DT11','DT20','DT21']

def points(dataset, candidate, endpoint='selected', split='test'):
    result = {int(r['seed']):r for r in rows if r['dataset']==dataset
              and r['candidate_id']==candidate and r['checkpoint']==endpoint and r['split']==split}
    assert set(result)==set(range(2022,2027))
    return result

def average(data, key):
    return st.mean(float(r[key]) for r in data.values())

def paired(a,b,key):
    return [float(a[s][key])-float(b[s][key]) for s in sorted(a)]

def ci(values):
    m=st.mean(values); h=2.7764451051977987*st.stdev(values)/math.sqrt(5)
    return f'[{m-h:+.9f}, {m+h:+.9f}]'

q_final={}; controller_counts={}; probe_counts={}; differing_online=[]
bpath=ROOT/'artifacts/aug_history_20261105/attempt1'
bjobs={(j['dataset'],j['seed']):j for j in json.loads((bpath/'PREPARED.json').read_text())['jobs']}
for job in jobs:
    dataset,seed=job['dataset'],job['seed']; counts=Counter(); probes=0; last={}
    old=bjobs[dataset,seed]
    for epoch in range(1,51):
        name=f'E{epoch:03d}.json'
        record=json.loads((ATTEMPT/'runs'/job['job_id']/'epochs'/name).read_text())
        parent=json.loads((bpath/'runs'/old['job_id']/'epochs'/name).read_text())
        if record['order_hash']!=parent['order_hash'] or record['model_hash']['online']!=parent['model_hash']['online']:
            differing_online.append((dataset,seed,epoch))
        for batch in record['batch_trace']:
            response=batch['response']; controllers=response['controllers']
            if controllers:
                probes+=1
                assert len({c['probe_hash'] for c in controllers.values()})==1
            for candidate,info in controllers.items():
                assert info['q_before']==last.get(candidate,[0.0]*len(info['q_before'])), (dataset,seed,epoch,candidate)
                counts[candidate]+=1; last[candidate]=info['q_after']
                assert info['valid_cells']==6
    controller_counts[dataset,seed]=dict(counts); probe_counts[dataset,seed]=probes
    winner=hpo['datasets'][dataset]['winners']['A_GROUP']['candidate_id']
    assert len(last[winner])==12
    q_final[dataset,seed]=last[winner]
assert not differing_online, differing_online[:5]

lines=['# A阶段实验分析：实际路径分组EMA','',
    'A完成两自然数据集各五seed、每条50epoch及15个影子。新增分组/自适应家族的validation赢家在test selected的平均AUC均低于F；ML统一家族选择F。MovieLens选择F继续C；KuaiRand按validation保留固定profile，但test增量未复现。B/A对应500个epoch的online参数hash与数据顺序hash逐一完全相同，新增差异来自参数估计和合法选点。','',
    '## 身份和矩阵','',
    '- 实施提交 `f00b5ce6b9e6476ca689b781667a5d698742aa70`；source hash `afecc8dcef7f1687d61383af0ca69176eb7b7ec26d0700eb43836a827b5e2ff9`。',
    '- selection hash `'+decision['selection_hash']+'`；ENDPOINTS SHA-256 `0c8d113e88214c60e3555b92e809c238d8d8bd274e8bd872b7052f38d377e226`。',
    '- 10次父online完整重放、500个提交epoch、17250次真实Adam；每条15影子，包括统一3、固定分组4、GROUP4、GLOBAL4；五seed2022–2026。',
    '- selected由validation六格Macro AUC最早严格最大值确定。每数据全五seed共用同一候选，六格保护优先；ML主发表参照M3oE，KR为HyperGate。',
    '- test沿用历史研究划分，本轮在HPO/选点/下一动作冻结后访问。决定仅用validation，后续动作不按test改派。','',
    '![A组件与控制器](figures/A_COMPARISON.png)','',
    '左：各家族validation赢家对F的test配对均差、95%CI及逐seed点，单位10⁻³ AUC；右：GROUP赢家共同E50的q均值±SD，q=log₂(半衰期倍率)，虚线为F的q0。','',
    '## 家族结果','',
    '| 数据集 | 家族/配置 | V Macro AUC | test AUC均值±SD | test LL均值±SD | ΔAUC对F | 配对95%CI | ΔLL对F | AUC正向seed |',
    '|---|---|---:|---:|---:|---:|---|---:|---:|']
for dataset,selection in hpo['datasets'].items():
    f=points(dataset,'F')
    for family,info in selection['winners'].items():
        candidate=info['candidate_id'];a=points(dataset,candidate);v=points(dataset,candidate,split='validation')
        delta=paired(a,f,'macro_auc');loss=paired(a,f,'macro_logloss')
        sd=st.stdev(float(x['macro_auc']) for x in a.values()); lsd=st.stdev(float(x['macro_logloss']) for x in a.values())
        lines.append(f'| {dataset} | {family}/{candidate} | {average(v,"macro_auc"):.9f} | {average(a,"macro_auc"):.9f}±{sd:.9f} | {average(a,"macro_logloss"):.9f}±{lsd:.9f} | {st.mean(delta):+.9f} | {ci(delta)} | {st.mean(loss):+.9f} | {sum(x>0 for x in delta)}/5 |')
lines+=['','## 冻结选择和逐seed结果','',
    'ML的最佳家族V分数来自A_FIXED_T_FAST，比F高0.000041901；F成本更低且差在0.0002内，按规则选择F，增量为0，触发C。KR的A_FIXED_S_FAST为V最佳、对F +0.001009861且六格保护通过，按规则停止KR的后续训练。该profile半衰期为S×0.5、D/T/DT×2。','',
    '| 数据集 | 被选方案 | seed | epoch | test AUC | test LL | ΔAUC对F | ΔLL对F |','|---|---|---:|---:|---:|---:|---:|---:|']
for dataset,d in decision['datasets'].items():
    candidate=d['selected_candidate'];a=points(dataset,candidate);f=points(dataset,'F')
    for seed in sorted(a):
        lines.append(f'| {dataset} | {candidate} | {seed} | {a[seed]["epoch"]} | {float(a[seed]["macro_auc"]):.9f} | {float(a[seed]["macro_logloss"]):.9f} | {float(a[seed]["macro_auc"])-float(f[seed]["macro_auc"]):+.9f} | {float(a[seed]["macro_logloss"])-float(f[seed]["macro_logloss"]):+.9f} |')
lines+=['','## 总收益、内部对照与六格','',
    '正式AUC收益≥+.003、六格均差≥−.001、LL均差<0分别判断；内部组件不新增+.003门。置信区间和同向数为描述统计，单位为seed。发表模型沿用各自冻结源配置，HyperGate保留UNVERIFIED_PROPOSAL超参标记。','',
    '| 数据集 | 方案 | 比较对象 | ΔAUC | 95%CI | ΔLL | 最差格ΔAUC | 正式AUC资格/六格保护/LL下降 |','|---|---|---|---:|---|---:|---:|---|']
for dataset,d in decision['datasets'].items():
    selection=hpo['datasets'][dataset];group=selection['winners']['A_GROUP']['candidate_id']
    for candidate in dict.fromkeys([d['selected_candidate'],group]):
        a=points(dataset,candidate)
        controls=['F','M3oE_fixed1','HyperGate','AutoIFS'] if candidate==d['selected_candidate'] else ['F',selection['winners']['A_GLOBAL']['candidate_id'],selection['winners']['A_FIXED']['candidate_id'],selection['winners']['U']['candidate_id'],selection['matched_at_GROUP']['A_GLOBAL']]
        for control in dict.fromkeys(controls):
            b=points(dataset,control);delta=paired(a,b,'macro_auc');gain=st.mean(delta);loss=st.mean(paired(a,b,'macro_logloss'))
            worst=min(st.mean(paired(a,b,f'd{i}t{j}_auc')) for i in range(3) for j in range(2))
            qualification=str(gain>=.003) if control in ['M3oE_fixed1','HyperGate','AutoIFS'] else '内部增量'
            lines.append(f'| {dataset} | {candidate} | {control} | {gain:+.9f} | {ci(delta)} | {loss:+.9f} | {worst:+.9f} | {qualification}/{worst>=-.001}/{loss<0} |')
lines+=['','| 数据集 | 方案 | cell | AUC | LL | ΔAUC对F | ΔLL对F |','|---|---|---|---:|---:|---:|---:|']
for dataset,d in decision['datasets'].items():
    f=points(dataset,'F')
    for candidate in dict.fromkeys([d['selected_candidate'],hpo['datasets'][dataset]['winners']['A_GROUP']['candidate_id']]):
        a=points(dataset,candidate)
        for i in range(3):
            for j in range(2):
                k=f'd{i}t{j}';lines.append(f'| {dataset} | {candidate} | {k} | {average(a,k+"_auc"):.9f} | {average(a,k+"_logloss"):.9f} | {st.mean(paired(a,f,k+"_auc")):+.9f} | {st.mean(paired(a,f,k+"_logloss")):+.9f} |')
lines+=['','## 任务和共同阶段','',
    '| 数据集 | 方案 | endpoint | ΔMacro AUC对F | ΔMacro LL | Δtask0 AUC | Δtask1 AUC | ΔTraffic AUC | ΔPooled AUC |','|---|---|---|---:|---:|---:|---:|---:|---:|']
for dataset,d in decision['datasets'].items():
    for candidate in dict.fromkeys([d['selected_candidate'],hpo['datasets'][dataset]['winners']['A_GROUP']['candidate_id']]):
        for endpoint in ['selected','E10','E50']:
            a=points(dataset,candidate,endpoint);f=points(dataset,'F',endpoint)
            values=[st.mean(paired(a,f,k)) for k in ['macro_auc','macro_logloss','task0_macro_auc','task1_macro_auc','sample_weighted_cell_auc','pooled_task_mean_auc']]
            lines.append('| '+dataset+' | '+candidate+' | '+endpoint+' | '+' | '.join(f'{x:+.9f}' for x in values)+' |')
lines+=['','## 控制器实际运行与成本','',
    '| 数据集 | GROUP赢家 | 共享probe次数 | 全部控制器调用数 | E50平均q（S,D0,D1,D2,T0,T1,DT00,DT01,DT10,DT11,DT20,DT21） |','|---|---|---:|---:|---|']
for dataset,selection in hpo['datasets'].items():
    qs=[q_final[dataset,seed] for seed in range(2022,2027)]
    means=[st.mean(q[i] for q in qs) for i in range(12)]
    lines.append(f'| {dataset} | {selection["winners"]["A_GROUP"]["candidate_id"]} | {sum(probe_counts[dataset,s] for s in range(2022,2027))} | {sum(sum(controller_counts[dataset,s].values()) for s in range(2022,2027))} | '+', '.join(f'{x:+.4f}' for x in means)+' |')
lines+=['','| 数据集 | training秒 | probe秒（含sampling） | sampling子项秒 | validation秒 | 核心worker计时合计秒 |','|---|---:|---:|---:|---:|---:|']
for dataset,d in decision['datasets'].items():
    c=d['cost'];lines.append(f'| {dataset} | {c["training_seconds"]:.3f} | {c["probe_seconds"]:.3f} | {c["sampling_seconds"]:.3f} | {c["validation_seconds"]:.3f} | {c["gpu_seconds"]:.3f} |')
lines+=['','## 证据与竞争解释','',
    '- **STRONG，描述性事实：** 新增分组/自适应家族validation赢家的test selected均值低于F；两个数据集的GROUP均未超过独立调参GLOBAL/固定profile/统一EMA的全部关键控制。500个对应epoch的online/order hash完全一致，隔离了online训练变化。',
    '- **MEDIUM：** ML GROUP和GLOBAL对F的selected AUC分别约−0.000750/−0.000728，LL分别下降约0.000813/0.001472；4/5 seed AUC下降，排序与概率损失出现取舍。KR固定profile V增量+0.001010转为test−0.000217、LL+0.000312，3/5 seed AUC正向、区间跨0，增量稳定性证据弱。',
    '- **WEAK，机制归因：** train排序探针可能偏好与holdout不同的时间尺度；路径差异可能有作用但当前控制器未提取稳定收益；部分细小差异来自选点。probe实际触发、q更新且源online不变，不能把结果归因于未运行控制器。固定profile的V胜出没有同步转成test胜出。',
    '- 域人口没有因A改变，分组依据实际模型路径而非大小。六格表需结合基准AUC、标签基率与阶段考察；当前结果不支持用小域等同困难域解释。存在的单格改变量和保护违例保留于上表及PAIRED_SUMMARY，宏均值不替代它们。',
    '- 下一项区分实验为已定义的MovieLens C：让历史漂移进入增强协方差，比较任务条件、合并及固定旋转控制。KR按validation停止后续C，保留V/test分化事实；不按test追加搜索。','',
    '运行中两次必要RAM轮换及一次不需要的重复轮换均保留已提交状态；全部控制器日志的q_before逐次等于上次q_after，含恢复边界。额外未提交开销无完整计时。原始记录见RUNNER_REPORT_A及ADMISSIONS/RETRIES；最终每seed恰好50个提交epoch。参数与checkpoint仅RAM，冻结评价后已释放。完整六格逐seed、epoch/model/source/data/hash见ENDPOINTS、HPO、曲线和配对CSV。','']
(Path(__file__).parent/'A_STAGE_ANALYSIS.md').write_text('\n'.join(lines))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for i,(dataset,selection) in enumerate(hpo['datasets'].items()):
    f=points(dataset,'F');ax=axes[i,0]
    for n,(family,info) in enumerate(selection['winners'].items()):
        values=paired(points(dataset,info['candidate_id']),f,'macro_auc');m=st.mean(values);h=2.7764451051977987*st.stdev(values)/math.sqrt(5)
        ax.errorbar(n,m*1e3,yerr=h*1e3,fmt='o',capsize=4,color='#1167b1')
        ax.scatter([n+(j-2)*.035 for j in range(5)],[v*1e3 for v in values],s=12,color='#777777',alpha=.6)
    ax.axhline(0,color='#333333',lw=.8);ax.set_xticks(range(4),['Group*','Global*','Fixed*','Uniform*'])
    ax.set_ylabel('Test AUC change vs F (x 1e-3)');ax.set_title(dataset.title()+' (paired 95% CI, n=5)');ax.grid(axis='y',alpha=.2)
    ax=axes[i,1];qs=[q_final[dataset,s] for s in range(2022,2027)]
    means=[st.mean(q[g] for q in qs) for g in range(12)];errors=[st.stdev(q[g] for q in qs) for g in range(12)]
    ax.errorbar(range(12),means,yerr=errors,fmt='o',capsize=3,color='#1167b1');ax.axhline(0,color='#333333',lw=.8,linestyle='--')
    ax.set_xticks(range(12),groups,rotation=45,ha='right');ax.set_ylim(-1.15,1.15);ax.set_ylabel('q = log2(half-life ratio), mean +/- SD')
    ax.set_title('Group winner controller at common E50');ax.grid(axis='y',alpha=.2)
directory=Path(__file__).parent/'figures';directory.mkdir(exist_ok=True)
fig.savefig(directory/'A_COMPARISON.png',dpi=180);fig.savefig(directory/'A_COMPARISON.svg');plt.close(fig)
print('A_STAGE_ANALYSIS.md generated; 500 B/A online and order hashes match')
