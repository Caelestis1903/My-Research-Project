"""Post-freeze descriptive analysis of saved R2 metrics; never trains models."""
from pathlib import Path
import csv
import json
import math
import statistics as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'analysis'

def load(path):
    return json.loads(path.read_text())

def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def save_csv(name, values):
    if not values:
        return
    with (OUT / name).open('w', newline='') as f:
        fields = sorted(set().union(*(v.keys() for v in values)))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(values)

def table(headers, records):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in rec)+' |' for rec in records])

def fmt(v):
    return f'{v:.9f}' if isinstance(v, float) else str(v)

def summarize(values):
    return {'n':len(values),'mean':st.mean(values),'sd_ddof1':st.stdev(values) if len(values)>1 else None,
            'min':min(values),'max':max(values),'positive_seeds':sum(v>0 for v in values)}

def rho_display(row):
    if row['arm'] == 'B2':
        return ','.join(fmt(float(row[f'rho_d_{d}'])) for t in range(2) for d in range(3))+'（两任务共享）'
    if row['arm'] == 'B0':
        return 'B0固定均匀（0.5）'
    return ','.join(fmt(float(row[f'rho_dt_{i}'])) for i in range(6))

def metrics(cells, prefix, shares):
    auc = {i:float(r[prefix+'auc']) for i,r in cells.items()}
    loss = {i:float(r[prefix+'logloss']) for i,r in cells.items()}
    values = {'macro_auc':st.mean(auc.values()),'macro_logloss':st.mean(loss.values()),
              'W_auc':sum(shares[d]*auc[t*3+d] for t in range(2) for d in range(3))/2,
              'W_logloss':sum(shares[d]*loss[t*3+d] for t in range(2) for d in range(3))/2}
    for t in range(2):
        values[f'task{t}_auc'] = st.mean(auc[t*3+d] for d in range(3))
        values[f'task{t}_logloss'] = st.mean(loss[t*3+d] for d in range(3))
    for d in range(3):
        values[f'domain{d}_auc'] = st.mean(auc[t*3+d] for t in range(2))
        values[f'domain{d}_logloss'] = st.mean(loss[t*3+d] for t in range(2))
    for i in range(6):
        values[f'cell{i}_auc'] = auc[i]
        values[f'cell{i}_logloss'] = loss[i]
    values['worst_cell_auc'] = min(auc.values())
    return values

def main():
    freeze = load(ROOT / 'locks/global_selection_freeze.json')
    assert load(ROOT / 'locks/global_test_unlock.json')['global_test_unlock']
    OUT.mkdir(exist_ok=True)
    allruns = {}
    aggregates, cell_export, epoch_export, fusion_export, stage_export = [], [], [], [], []
    for frozen in freeze['runs']:
        run = Path(frozen['run_dir'])
        m = load(run/'run_manifest.json')
        test = rows(run/'metrics_test.csv')
        assert len(test)==42 and {r['checkpoint'] for r in test} == {'selected','final',*(f'common_epoch_{e}' for e in (9,19,29,39,49))}
        val = rows(run/'metrics_epoch.csv')
        shares = {int(d):float(v) for d,v in m['dataset_domain_shares'].items()}
        epoch_cells = {e:{int(r['cell']):r for r in val if int(r['epoch'])==e} for e in range(50)}
        selected = epoch_cells[m['selected_epoch']]
        test_cells = {int(r['cell']):r for r in test if r['checkpoint']=='selected'}
        info = {'dataset':m['dataset'],'arm':m['arm'],'seed':m['seed'],'selected_epoch':m['selected_epoch']}
        splitmetrics = {'validation':metrics(selected,'validation_',shares),'test':metrics(test_cells,'test_',shares)}
        for split,v in splitmetrics.items():
            aggregates.append({**info,'split':split,**v})
        for i in range(6):
            v, t = selected[i], test_cells[i]
            cell_export.append({**info,'task':i//3,'domain':i%3,'physical_domain':v['physical_domain'],
                                **{k:v[k] for k in ('train_sample_count','train_positive_rate','validation_sample_count','validation_positive_rate','validation_auc','validation_logloss')},
                                **{k:t[k] for k in ('test_sample_count','test_positive_rate','test_auc','test_logloss')}})
        batch = rows(run/'batch_trace.csv')
        for e in range(50):
            train = [r for r in batch if r['kind']=='train' and int(r['epoch'])==e]
            bm = st.mean(float(r['train_objective']) for r in train)
            wm = sum(float(r['train_objective'])*int(r['batch_size']) for r in train)/sum(int(r['batch_size']) for r in train)
            assert math.isclose(wm,float(epoch_cells[e][0]['train_objective']),abs_tol=1e-12)
            ev = metrics(epoch_cells[e],'validation_',shares)
            epoch_export.append({**info,'epoch':e,'train_batch_mean_bce':bm,'train_sample_weighted_bce':wm,
                                 'train_macro_eval_logloss':st.mean(float(r['train_cell_logloss']) for r in epoch_cells[e].values()),**ev})
        fusion = rows(run/'fusion_log.csv')
        for r in fusion:
            if int(r['epoch']) in (m['selected_epoch'],49):
                fusion_export.append({**info,'endpoint':'selected' if int(r['epoch'])==m['selected_epoch'] else 'final',**r})
        for ck in sorted({r['checkpoint'] for r in test}):
            cells = {int(r['cell']):r for r in test if r['checkpoint']==ck}
            stage_export.append({**info,'checkpoint':ck,**metrics(cells,'test_',shares)})
        allruns[(m['dataset'],m['arm'],m['seed'])] = {'manifest':m,'metrics':splitmetrics,'val':epoch_cells,'fusion':fusion,'path':str(run)}
    paired, stats = [], []
    for dataset in ('kuairand','movielens'):
        for candidate,baseline in (('B1','B0'),('B2','B0'),('B1','B2')):
            seeds = [s for s in (2022,2023) if (dataset,candidate,s) in allruns and (dataset,baseline,s) in allruns]
            if not seeds:
                continue
            for split in ('validation','test'):
                metric_names = allruns[(dataset,candidate,seeds[0])]['metrics'][split].keys()
                for metric in metric_names:
                    diffs = []
                    for s in seeds:
                        c = allruns[(dataset,candidate,s)]['metrics'][split][metric]
                        b = allruns[(dataset,baseline,s)]['metrics'][split][metric]
                        diffs.append(c-b)
                        paired.append({'dataset':dataset,'comparison':candidate+'-'+baseline,'seed':s,'split':split,'metric':metric,'candidate':c,'baseline':b,'delta':c-b})
                    stats.append({'dataset':dataset,'comparison':candidate+'-'+baseline,'split':split,'metric':metric,**summarize(diffs)})
    guards = []
    for dataset in ('kuairand','movielens'):
        for arm in ('B1','B2'):
            seeds = [s for s in (2022,2023) if (dataset,arm,s) in allruns]
            if not seeds:
                continue
            for split in ('validation','test'):
                delta = {s:{k:v-allruns[(dataset,'B0',s)]['metrics'][split][k] for k,v in allruns[(dataset,arm,s)]['metrics'][split].items()} for s in seeds}
                values = {'G_a_task0':(st.mean(delta[s]['task0_auc'] for s in seeds),-0.001,'>='),
                          'G_a_task1':(st.mean(delta[s]['task1_auc'] for s in seeds),-0.001,'>='),
                          'G_b_W':(st.mean(delta[s]['W_auc'] for s in seeds),-0.001,'>='),
                          'G_f_macroLL':(st.mean(delta[s]['macro_logloss'] for s in seeds),0.005,'<=')}
                for i in range(6):
                    values[f'G_c_cell{i}'] = (st.mean(delta[s][f'cell{i}_auc'] for s in seeds),-0.003,'>=')
                    values[f'G_e_cell{i}'] = (st.mean(delta[s][f'cell{i}_logloss'] for s in seeds),0.010,'<=')
                    for s in seeds:
                        values[f'G_d_cell{i}_seed{s}'] = (delta[s][f'cell{i}_auc'],-0.005,'>=')
                for name,(v,line,op) in values.items():
                    guards.append({'dataset':dataset,'candidate':arm,'split':split,'n':len(seeds),'guard':name,'value':v,'threshold':line,'operator':op,'pass':v>=line if op=='>=' else v<=line,'eligible_two_seed_guard':len(seeds)==2})
    for name,data in [('selected_aggregate_metrics.csv',aggregates),('selected_cell_metrics.csv',cell_export),('paired_metric_deltas.csv',paired),('paired_metric_uncertainty.csv',stats),('guard_values.csv',guards),('training_trajectories.csv',epoch_export),('fusion_selected_final.csv',fusion_export),('fixed_test_endpoints.csv',stage_export)]:
        save_csv(name,data)
    (OUT/'summary.json').write_text(json.dumps({'aggregates':aggregates,'paired_statistics':stats,'guards':guards},indent=2,ensure_ascii=False,allow_nan=False))
    report = ['# R2 实验分析（合同受限的最终报告）',
              '日期：2026-10-12。Stage R2 的合法剩余执行与报告已完成；总体状态 **INCOMPLETE**。KuaiRand B1/2023 原生进程失败且唯一精确恢复失败，不能完成预定双数据集主分析。本报告仅对冻结的有效完成 run 作限定描述；不将缺失当科学阴性。',
              '## 执行与有效性',
              'DESIGN、IMPLEMENT、独立 REVIEW 已有工件；G0 复核与四份 B0 G1 均通过。MovieLens A 六份各完成50 epochs，初始化、train/architecture exposure及selected earliest-max验收通过；KuaiRand三份完成、一份停在保存状态epoch=38（下一epoch位置，已保存38个完整epoch），B2未执行。总10 attempts、9 COMPLETE、1 INCOMPLETE、488个已记录run-epochs、1次resume；未执行B3/B4与Hpost。',
              '数据文件仅物理路径适配且字节hash相同；新增 evaluator 可选 dataset_path 经 Coder/Reviewer 闭环，不改变数学或选模。MovieLens B2只剩两个合法run，按独立审核记录两进程例外；没有凑额外实验。',
              '全部selection/config/selected/final/common/checkpoint身份与终止/decision先冻结，再执行统一test。全局锁见 `locks/global_selection_freeze.json` 与 `locks/global_test_unlock.json`。失败部分selected不进入test。旧test已被历史研究使用，此处不称全新holdout。',
              '## 指标定义与完整主端点',
              'Macro AUC为六格等权平均；任务宏为三域等权平均。W为原训练域份额加权逐格AUC（两任务平均），不是pooled AUC。LogLoss沿实际S1实现clip epsilon=1e-12，与Charter文字1e-7的差异已在设计/实现审核中登记并冻结。SD使用paired差值、ddof=1；n=1的SD不可估，不将格或epoch当独立seed。两个seed不足以声称统计显著、等效或稳定普遍收益。',
              table(['数据集','臂','seed','selected epoch(0-based)','split','Macro AUC','Macro LL','W AUC','t0 AUC','t1 AUC'],
                    [[r['dataset'],r['arm'],r['seed'],r['selected_epoch'],r['split'],*[fmt(r[k]) for k in ('macro_auc','macro_logloss','W_auc','task0_auc','task1_auc')]] for r in aggregates]),
              '## 六格 AUC 与 LogLoss（全部有效完成run）',
              table(['数据集','臂','seed','task/domain','Val AUC','Val LL','Test AUC','Test LL'],
                    [[r['dataset'],r['arm'],r['seed'],f"t{r['task']}d{r['domain']}",*[fmt(float(r[k])) for k in ('validation_auc','validation_logloss','test_auc','test_logloss')]] for r in cell_export]),
              'KuaiRand B1/2023、B2两seed均无合格主端点；B3/B4为未执行，不能填0或引用历史CSV。selected完整全精度表见 `analysis/selected_cell_metrics.csv`；final/common固定描述端点见 `analysis/fixed_test_endpoints.csv` 和各run的完整42行 `metrics_test.csv`。',
              '## 配对差值、均值与不确定性',
              table(['数据集','比较','split','metric','n','mean Δ','SD(ddof=1)','范围','正向seed数'],
                    [[r['dataset'],r['comparison'],r['split'],r['metric'],r['n'],fmt(r['mean']),fmt(r['sd_ddof1']) if r['sd_ddof1'] is not None else '不可估',f"[{fmt(r['min'])}, {fmt(r['max'])}]",r['positive_seeds']] for r in stats if r['metric'] in ('macro_auc','macro_logloss','W_auc')]),
              '逐seed、逐任务、逐域、逐格的原始差值和全部不确定性见 `analysis/paired_metric_deltas.csv` 与 `analysis/paired_metric_uncertainty.csv`；未提供由两个seed推导的不可靠总体置信承诺。',
              '## 阶段决定与保护线',
              'MovieLens：有效A完整、validation decision **NO**。B1−B0的D_2022=-0.00006167093048625727，D_2023=+0.00007320015415501757，mean(D)=+0.000005764611834380151；未达0.001探索线且符号不一致。B1−B2的T分别=-0.00000886271742106537、-0.000004821702253021698，mean(T)=-0.000006842209837043534；未达额外任务粒度门。B1−B0 validation全部guards通过。故B3/B4 **NOT_SCHEDULED**，不是漏跑。',
              'KuaiRand：资格不完整，阶段判决 **NOT_COMPUTED**，不是科学NO；停止状态已锁，余额不挪给replacement或新实验。B1−B0仅seed2022，不能过两seed性能/保护线。',
              table(['数据集','候选-B0','split','n','全部数值线','两seed资格'],
                    [[ds,arm,sp,len({r['seed'] for r in aggregates if r['dataset']==ds and r['arm']==arm}),all(r['pass'] for r in guards if r['dataset']==ds and r['candidate']==arm and r['split']==sp),'满足' if ds=='movielens' else '不满足'] for ds,arm in [('movielens','B1'),('movielens','B2'),('kuairand','B1')] for sp in ('validation','test')]),
              '保护条件逐项全精度value/threshold/pass和资格见 `analysis/guard_values.csv`。B2为预注册次比较，不能升级主方法。最终test不重开B或重新选epoch。',
              '## Validation 与 Test 一致性']
    for ds,cmp in [('movielens','B1-B0'),('movielens','B2-B0'),('movielens','B1-B2'),('kuairand','B1-B0')]:
        selected_stats = [r for r in stats if r['dataset']==ds and r['comparison']==cmp and r['metric']=='macro_auc']
        if selected_stats:
            report.append(f"{ds} {cmp}："+'；'.join(f"{r['split']} mean Δ={fmt(r['mean'])}（正向{r['positive_seeds']}/{r['n']}seed）" for r in selected_stats)+'。仅在相同已完成seed范围比较，不跨数据集混合门。')
    report += ['MovieLens B1−B0 test 的两seed差分别为−0.000076175446和+0.000107359064，均值+0.000015591809，SD=0.000129778497；B2−B0均值+0.000028248305，同样仅1/2 seed为正。两臂test保护线均通过，但均未达到方向重复或0.001探索线；B1没有显示超过更简单B2的实用额外收益。B1/B2 test Macro LogLoss平均分别增加0.000620449656/0.000447182330，两seed均增加（loss越低越好），AUC微小均值不能概括为一致质量改善。',
               'KuaiRand唯一完整配对seed2022的B1−B0 Val Macro AUC约为+0.001494010，不能称已通过探索性筛查：t1d0 Val ΔAUC=−0.019951010925，违反任seed单格−0.005保护线，且两seed资格缺失。Test Macro AUC转为−0.008385198449；t0d0/t1d0分别退化−0.022433704807/−0.030010696617，任务宏差和W差也低于数值线。单seed均值护栏只作数值描述，不补成双seed门。Test Macro LogLoss反而下降0.000542742031，说明排序与概率损失的方向可不同；其W AUC下降0.001529004992。该结果是有效完整run中的观察退化，失败的B1/2023仍是INCOMPLETE而非科学阴性。',
               '## 逐seed逐格收益与最大退化',
               '下表按冻结selected端点比较；改善格定义为ΔAUC严格大于0。格数不是独立重复数，负差最小格用于定位退化。']
    cell_patterns = []
    for ds,arm in [('kuairand','B1'),('movielens','B1'),('movielens','B2')]:
        for seed in (2022,2023):
            if (ds,arm,seed) not in allruns:
                continue
            for split in ('validation','test'):
                delta = [allruns[(ds,arm,seed)]['metrics'][split][f'cell{i}_auc']-allruns[(ds,'B0',seed)]['metrics'][split][f'cell{i}_auc'] for i in range(6)]
                worst = min(range(6),key=lambda i:delta[i])
                cell_patterns.append([ds,arm+'-B0',seed,split,sum(v>0 for v in delta),f't{worst//3}d{worst%3}',fmt(delta[worst])])
    report.append(table(['数据集','比较','seed','split','改善格数/6','最小Δ格','最小ΔAUC'],cell_patterns))
    report += ['## 域大小、基线难度与小域/大域关系',
               'KuaiRand训练域大小为d0=172782、d1=1743898、d2=89452；MovieLens为d0=150000、d1=34500、d2=3000。大小与域内容、标签基率、训练曝光、时间划分同时变化，不能识别大小为主因。MovieLens小域validation/test样本约3万量级，不能把训练小域与评价样本少混同。',
               table(['数据集','domain','train n','Val n','Test n','B0 Val AUC(两任务两seed平均)','B0 Test AUC','B1-B0 Test Δ(可用seed)'],
                     [[ds,d,next(r['train_sample_count'] for r in cell_export if r['dataset']==ds and r['domain']==d),next(r['validation_sample_count'] for r in cell_export if r['dataset']==ds and r['domain']==d),next(r['test_sample_count'] for r in cell_export if r['dataset']==ds and r['domain']==d),fmt(st.mean(allruns[(ds,'B0',s)]['metrics']['validation'][f'domain{d}_auc'] for s in (2022,2023))),fmt(st.mean(allruns[(ds,'B0',s)]['metrics']['test'][f'domain{d}_auc'] for s in (2022,2023))),fmt(st.mean(allruns[(ds,'B1',s)]['metrics']['test'][f'domain{d}_auc']-allruns[(ds,'B0',s)]['metrics']['test'][f'domain{d}_auc'] for s in (2022,2023) if (ds,'B1',s) in allruns))] for ds in ('kuairand','movielens') for d in range(3)]),
               '难度以B0 AUC/LogLoss为操作性描述，同时受标签噪声与可预测性影响；不把低AUC自动归因模型失败。域标签基率与train→validation/test的mixture和prior shift全表见 `analysis/selected_cell_metrics.csv`（同一域两任务分别列出）。',
               '## 融合参数、训练阶段与轨迹',
               table(['数据集','臂','seed','epoch','raw aD/aT/bD/bT','rho task-major','均值gate entropy'],
                     [[r['dataset'],r['arm'],r['seed'],r['epoch'],','.join(fmt(float(r[k])) for k in ('raw_aD','raw_aT','raw_bD','raw_bT')),rho_display(r),fmt(st.mean(float(r[f'gate_entropy_{i}']) for i in range(6)))] for r in fusion_export if r['endpoint']=='selected']),
               table(['数据集','臂','seed','selected epoch','Val AUC selected','Val AUC final','train BCE epoch0','train BCE final'],
                     [[ds,a,s,r['manifest']['selected_epoch'],fmt(r['metrics']['validation']['macro_auc']),fmt(metrics(r['val'][49],'validation_',{int(k):float(v) for k,v in r['manifest']['dataset_domain_shares'].items()})['macro_auc']),fmt(float(r['val'][0][0]['train_objective'])),fmt(float(r['val'][49][0]['train_objective']))] for (ds,a,s),r in allruns.items()]),
               '训练loss同时导出batch均值与按样本数加权BCE（从已保存batch_trace复算，无重新训练），全50epoch验证轨迹见 `analysis/training_trajectories.csv`。selected/final的global raw/sigmoid、rho、梯度/update norm、专家输出范数、门熵及饱和标记见 `analysis/fusion_selected_final.csv` 与各run原 `fusion_log.csv`。B4未执行，无delta/b_dt结果。权重变化不识别真实亲和；own/other global系数学后可跨臂变化。',
               '## 证据解释与下一步边界',
               '1. 稳定重复现象：MovieLens B0/B1/B2两个seed的selected epoch均分别为7/6；已保存轨迹显示同seed各臂选模与曝光匹配。B1−B2 validation两个seed均为极小负差；这种低功效观察不能证明等效或B2充分。',
               '2. 矛盾现象：MovieLens B1−B0 validation两个seed符号相反。KuaiRand只有一个有效B1seed，不能确认或反驳跨seed重复，也不能拿不同数据集单点混成重复成功。Validation/test固定比较见上表；任何不一致均保留，不改队列。',
               '3. 证据强度：按协议，MovieLens两seed性能证据最多 **WEAK**；KuaiRand单seed更加有限。已确认的恢复预算耗尽与缺失资格为 **STRONG** 的执行事实，不能转作科学阴性。没有满足 **MEDIUM** 或 **STRONG** 性能因果证据的实验。',
               '4. 竞争解释：小参数的有效更新幅度/优化几何、专家与门塔共同适配、global系数漂移、训练后期过拟合、标签基率和时间shift、随机训练波动均未被此预算穷尽。B3固定偏置与B4 own/other机制未测；不能称已排除简单解释。',
               '5. 域大小为主解释变量的证据不足。下面给出固定test域差符号所对应的trade-off描述；它不证明域大小因果。']
    for ds in ('kuairand','movielens'):
        seeds = [s for s in (2022,2023) if (ds,'B1',s) in allruns]
        small,large = (2,1) if ds=='kuairand' else (2,0)
        def domain_delta(d):
            return st.mean(allruns[(ds,'B1',s)]['metrics']['test'][f'domain{d}_auc']-allruns[(ds,'B0',s)]['metrics']['test'][f'domain{d}_auc'] for s in seeds)
        small_delta,large_delta = domain_delta(small),domain_delta(large)
        report.append(f"6. {ds}：小域d{small} test Δ={fmt(small_delta)}，大域d{large} Δ={fmt(large_delta)}；该可用seed均值{'呈现' if small_delta>0 and large_delta<0 else '未呈现'}‘小域改善/大域退化’符号组合（n={len(seeds)}）。还须按六格/seed表看矛盾，不声称稳定trade-off。")
    report += ['7. 最能区分竞争解释的未来实验应以证伪为目标：另行设计并审核后，在保持域身份/标签任务不变的条件下操纵训练样本量，同时独立增加seed评估B1−B0的方向和实用门，并设置已冻结简单对照以检验优化/共同适配解释。本R2不执行该建议，不用剩余预算临时补实验。',
               '## 四层结论与R3建议',
               '- 描述性现象：报告了融合参数、训练轨迹、阶段与域指标；学得非均匀不等于真实亲和。',
               '- 初步性能信号：MovieLens validation不满足方向重复及探索线；KuaiRand因缺失seed未检验两seed信号。固定test结果已公开，不能替代原追加门。',
               '- 实用性能通过：MovieLens validation与test均未通过方向重复及实用量级要求，虽保护线通过；KuaiRand双seed资格不足，且唯一完整seed已出现禁止单格退化。两数据集都不能声明实用性能通过。',
               '- 剩余算法研究空间：未执行B3/B4，层级D未检验；B1相对B2的极小差异不能支撑任务粒度必要性，也不证明等效。',
               '**R3建议：CONDITIONAL（当前不进入R3）**。MovieLens本固定pilot没有达到追加投入信号，停止其预定B；KuaiRand是基础设施不完整，不是算法阴性。完整双数据集主分析要求尚未满足，如要获得该证据必须另行明确设计/预算并审查故障，不在本合同中自动重训。所有合法可执行步骤已完成并交付；不自动进入R3/R4。']
    (ROOT/'R2_Experimental_Analysis.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    macro = [r for r in stats if r['metric']=='macro_auc' and r['comparison']=='B1-B0']
    conclusion = ['# R2 全局结论（最终，受合同停止规则限制）',
                  'Stage R2 的合法剩余训练、判决、冻结、最终测试和报告均已完成；总体实验状态为 **INCOMPLETE**，完整双数据集主分析前提未达到。',
                  'MovieLens：B0/B1/B2×两seed完整有效；validation追加判决 **NO**，B3/B4按合同 **NOT_SCHEDULED**。KuaiRand：B0两seed与B1/2022完整；B1/2023原生段错误、唯一精确恢复SIGFPE失败，锁INCOMPLETE，不得再resume/replacement或继续该队列B2。',
                  table(['数据集','split','B1-B0 mean Macro AUC Δ','n','正向seed数'],[[r['dataset'],r['split'],fmt(r['mean']),r['n'],r['positive_seeds']] for r in macro]),
                  '总10attempt：9 COMPLETE、1 INCOMPLETE；488个已记录run-epochs、1次resume。九个有效完成run的selected/final/common共63个固定端点已在全局冻结后评价；失败部分checkpoint未评价。测试不会重开训练或追加B。',
                  '性能证据最多 **WEAK**；MovieLens未达预注册validation探索/实用线，KuaiRand双seed性能未检验。B3/B4竞争解释未排除，不能推导AO-04普遍无效、B2等效或真实亲和。',
                  '固定test同样没有实用通过：MovieLens B1−B0两seed方向不一致、均值远低于0.001；KuaiRand唯一完整seed由validation正差转为test退化，d0两任务的AUC下降超过0.005单格保护线。KuaiRand缺失配对仍不作为科学阴性。',
                  '**R3建议：CONDITIONAL；本任务到R2停止，不自动进入R3/R4。** 本建议表示完整研究证据仍缺失；不是授权新增预算。',
                  '完整六格AUC/LogLoss、逐seed配对差与SD/范围、domain/task/size/difficulty、融合参数、训练阶段和竞争解释见 [R2_Experimental_Analysis.md](R2_Experimental_Analysis.md)。原始结果与可复算分析表在 `analysis/`；执行报告见 `R2_Experiment_Execution_Report.md`，决定见 `R2_A_Decision_Report.md`，冻结身份见 `locks/`。']
    (ROOT/'R2_Global_Conclusion.md').write_text('\n\n'.join(conclusion)+'\n',encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2,2,figsize=(12,7))
        colors = {'B0':'black','B1':'tab:blue','B2':'tab:orange'}
        for row,ds in enumerate(('kuairand','movielens')):
            for (dataset,arm,seed),run in allruns.items():
                if dataset!=ds:
                    continue
                trajectory = [r for r in epoch_export if r['dataset']==ds and r['arm']==arm and r['seed']==seed]
                style = '-' if seed==2022 else '--'
                for col,key in ((0,'macro_auc'),(1,'train_sample_weighted_bce')):
                    axes[row,col].plot([r['epoch'] for r in trajectory],[r[key] for r in trajectory],style,color=colors[arm],label=f'{arm}/{seed}')
                    axes[row,col].set(title=f'{ds}: {key}',xlabel='epoch (0-based)')
                    axes[row,col].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT/'training_trajectories.png',dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps({'complete_runs':len(allruns),'selected_cell_rows':len(cell_export),'fixed_test_endpoints':len(stage_export),'report':str(ROOT/'R2_Experimental_Analysis.md')}))

if __name__=='__main__':
    main()
