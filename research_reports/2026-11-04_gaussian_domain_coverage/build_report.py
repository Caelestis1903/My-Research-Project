"""Render the research report and ledger increment from frozen scalar exports."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ('G-D0', 'G-D02', 'G-D2', 'FIXED_EMA', 'M3OE', 'HYPERGATE', 'AUTOIFS')
CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))
LABELS = dict(FIXED_EMA='固定EMA（内部控制）', M3OE='M3oE fixed1', HYPERGATE='HyperGate', AUTOIFS='AutoIFS')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                     '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                    ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    j = json.loads((HERE/'ANALYSIS_VALUES.json').read_text())
    decisions = json.loads((HERE/'DECISIONS.json').read_text())
    run = json.loads((HERE/'RUN_MANIFEST.json').read_text())
    assert run['progress']['status'] == 'FINAL_COMPLETE' and run['progress']['committed_epochs'] == 500
    absolute, paired = j['absolute']['selected']['test'], j['paired']['selected']['test']
    records = list(csv.DictReader((HERE/'ENDPOINTS.csv').open()))
    costs = list(csv.DictReader((HERE/'COSTS.csv').open()))
    label = lambda arm: LABELS.get(arm, arm)
    metric = lambda arm, name: absolute[arm]['metrics'][name]
    ms = lambda arm, name: f"{metric(arm,name)['mean']:.6f} ± {metric(arm,name)['sd']:.6f}"
    contrast = lambda name, m: paired[name][m]
    interval = lambda value: f"{value['mean']:+.9f} [{value['ci95'][0]:+.9f}, {value['ci95'][1]:+.9f}]"
    guard = {r['comparison_id']:r for r in decisions['comparisons']}
    blocks = [
        '# KuaiRand Gaussian领域覆盖：五seed完整研究报告',
        '完成日期：2026-11-04。状态 FINAL_COMPLETE：新增10/10条轨迹、500/500epoch、24500次Adam及EMA更新；全部最终评价完成。主结果为固定EMA的validation六格Macro AUC最高、精确并列最早checkpoint的test指标。seeds为2022–2026。',
        '**当前选择：排序优先保留G-D0，概率损失优先保留G-D2；不优先追加G-D02实验。** '
        'G-D0的Macro AUC为0.679180，比G-D2高0.001181；G-D02为0.674604，比G-D0低0.004576。'
        '三种Gaussian均低于HyperGate的0.682113，均未达到本轮候选算法或强算法标准。'
        'G-D0对M3oE的总AUC收益为+0.007274，六格保护满足，保留这项完整方案收益。',
        '## 1. 统一协议与完成范围',
        '仅改变Gaussian增强覆盖集合：[0]、[0,2]；[2]直接复用历史五seed。原fixed1 M3oE、'
        '1614254参数、embedding16、4共享专家、FCN[512,256,256,64]；Gaussian κ=.1、λ=.5、'
        '固定EMA ρ=.995、普通Adam lr=.01、coupled wd=1e-5、betas=(.9,.999)、eps=1e-8、'
        'batch40960、FP32、50epoch保持冻结。原GA31-R3局部Generator和逐字段RMS detach规则保留；'
        '单域直接调用原函数，双域分别产生视图并统一计算 (ΣLreal + .5Σselected(Laug−Lreal))/(2B)。'
        '每batch一次backward、Adam和EMA更新。',
        '数据协议FULLTRAIN_NATURAL80_10_10：原行序80/10/10、train-only编码和缩放，先切分再筛tab0/1/4，'
        '完整自然人口，不重采样。train共2006132行，validation/test各250767行，每epoch49步。'
        '任务0=is_click，任务1=long_view。',
    ]
    population = []
    for d, (tab, train) in enumerate(((0,172782),(1,1743898),(4,89452))):
        sample = {split:next(r for r in records if r['arm']=='M3OE' and r['checkpoint']=='selected'
                            and r['split']==split and int(r['seed'])==2022) for split in ('validation','test')}
        population.append([f'd{d}',tab,train,int(float(sample['validation'][f'd{d}t0_count'])),
                           int(float(sample['test'][f'd{d}t0_count'])),
                           '/'.join(str(int(float(sample['validation'][f'd{d}t{t}_positive']))) for t in (0,1)),
                           '/'.join(str(int(float(sample['test'][f'd{d}t{t}_positive']))) for t in (0,1))])
    blocks += [table(['域','原tab','train N','validation N','test N','V正例 t0/t1','test正例 t0/t1'],population),
        '复用150个端点（G-D2、固定EMA、M3oE、HyperGate、AutoIFS各五seed×三个checkpoint×V/Test），'
        '新增120个端点（两臂×五seed×online/EMA×三个checkpoint×V/Test），合计270。'
        'A0 online与固定EMA共享原训练轨迹；本轮复用20条历史轨迹的5个配置视图，新增10条轨迹。'
        '正式模型结果来自11月4日[A+B统一协议](../2026-11-03_overnight_AB/FINAL_REPORT.md)。'
        '固定EMA属于内部控制。全部数据、配置及来源文件checksum见[RUN_MANIFEST.json](RUN_MANIFEST.json)。',
        '## 2. 五seed主结果与性能裁定',
        '均值±样本SD，n=5。Macro为六格等权平均，Traffic为按test人口加权的cell指标，'
        'Pooled为各任务合并域后计算的指标再平均。所有指标的均值、SD、区间见[ABSOLUTE_SUMMARY.csv](ABSOLUTE_SUMMARY.csv)。']
    blocks.append(table(['配置/模型','Macro AUC','Macro LogLoss','Traffic AUC','Traffic LL','Pooled AUC','Pooled LL'],
        [[label(a)]+[ms(a,m) for m in ('macro_auc','macro_logloss','sample_weighted_cell_auc',
          'sample_weighted_cell_logloss','pooled_task_mean_auc','pooled_logloss')] for a in ARMS]))
    blocks += ['性能等级以五seed平均Test Macro AUC相对最强发表模型HyperGate裁定：增益≥.003为强算法，'
        '≥.002为候选算法，>0且<.002为正向胜出。六格平均ΔAUC≥−.001独立裁定；LogLoss以平均差<0独立裁定。'
        'seed方向数和置信区间用于描述重复性，分类使用完整精度均值。',
        table(['Gaussian配置','对HyperGate ΔAUC','性能等级','六格保护','对HyperGate ΔLL'],
            [[r['arm'],f"{r['delta_macro_auc']:+.9f}",'未超过最强发表基线',
              '满足' if r['cell_protection_pass'] else '不满足',
              f"{guard[r['arm']+'_vs_HYPERGATE']['macro_logloss_delta']:+.9f}"]
             for r in decisions['classification']]),
        'G-D0相对AutoIFS的平均AUC高0.000583，对HyperGate仍差0.002932。'
        'G-D02对HyperGate、AutoIFS的AUC分别低0.007509、0.003993，均为5/5 seed下降。'
        '三种Gaussian相对HyperGate的平均LogLoss均下降；相对AutoIFS则均上升。'
        '完整分类及各比较对象的六格裁定见[DECISIONS.json](DECISIONS.json)。',
        '## 3. 每seed绝对值与validation选中epoch',
        '以下为主估计器；三种Gaussian/固定EMA使用fixed_ema，三个发表模型使用online。'
        '每seed的全部六格AUC/LL、域/任务、Traffic/Pooled、epoch、model hash及V/Test同状态身份完整保存在[ENDPOINTS.csv](ENDPOINTS.csv)。']
    blocks.append(table(['配置/模型','seed','selected epoch','Test Macro AUC','Test Macro LL'],
        [[label(a),seed,absolute[a]['epochs'][i],f"{metric(a,'macro_auc')['values'][i]:.9f}",
          f"{metric(a,'macro_logloss')['values'][i]:.9f}"] for a in ARMS for i,seed in enumerate(range(2022,2027))]))
    blocks += ['## 4. 配对收益：覆盖增量、总收益和组件增量',
        '配对单位为seed；t95%区间使用df=4的2.776445系数，SD采用ddof=1。'
        '下表ΔAUC>0、ΔLL<0有利；保护列列出低于−.001的平均AUC格。'
        '全部逐seed配对原值和差见[PAIRED_SEED.csv](PAIRED_SEED.csv)，均值/SD/区间见[PAIRED_SUMMARY.csv](PAIRED_SUMMARY.csv)。']
    comparisons = [('G-D0','G-D2'),('G-D02','G-D2'),('G-D02','G-D0')] + [
        (a,b) for a in ('G-D0','G-D02','G-D2') for b in ('M3OE','FIXED_EMA','HYPERGATE','AUTOIFS')]
    blocks.append(table(['比较','ΔAUC [t95%]','AUC正seed','ΔLL [t95%]','LL降seed','保护越线格'],
        [[f'{a}−{label(b)}',interval(contrast(a+'_vs_'+b,'macro_auc')),
          f"{contrast(a+'_vs_'+b,'macro_auc')['positive']}/5",
          interval(contrast(a+'_vs_'+b,'macro_logloss')),
          f"{contrast(a+'_vs_'+b,'macro_logloss')['negative']}/5",
          ','.join(c for c,v in guard[a+'_vs_'+b]['cell_auc_deltas'].items() if v<-.001) or '无']
         for a,b in comparisons]))
    blocks.append(table(['seed','D0−D2 ΔAUC / ΔLL','D02−D2 ΔAUC / ΔLL','D02−D0 ΔAUC / ΔLL'],
        [[seed]+[f"{contrast(k,'macro_auc')['values'][i]:+.9f} / {contrast(k,'macro_logloss')['values'][i]:+.9f}"
                 for k in ('G-D0_vs_G-D2','G-D02_vs_G-D2','G-D02_vs_G-D0')]
         for i,seed in enumerate(range(2022,2027))]))
    blocks += ['**STRONG：G-D0相对M3oE的整体排序收益。** AUC均差+0.007273579，5/5正，'
        't95%[+0.002351596,+0.012195561]，六格平均全部上升；平均LL下降0.001654150，4/5下降。'
        'G-D2对M3oE的既有平均AUC+0.006092690、LL−0.002888935仍保留，d0t0低于−.001保护线。'
        'G-D02对M3oE的AUC+0.002697370，d0t0下降0.006645151。',
        '**MEDIUM：双域相对D0的宏排序下降。** D02−D0为−0.004576209，4/5下降；'
        '平均LL+0.000654603。区间[−0.009799153,+0.000646736]保留seed异质性。'
        '双域相对D2为AUC−0.003395320、LL+0.001889388，未取得Macro指标的覆盖扩展收益。',
        '**WEAK：D0替代D2的稳定宏增量。** D0−D2的AUC+0.001180888、3/5正，'
        'LL+0.001234785、4/5恶化；共同E50的AUC差缩至+0.000124049。'
        '当前AUC均值最优为D0，D2保留更低的Macro LL。',
        '## 5. 六个域任务、最差格与性能取舍',
        '下表为五seed均值，每格格式AUC / LogLoss；逐格SD在ABSOLUTE_SUMMARY.csv。']
    blocks.append(table(['配置/模型']+list(CELLS),[[label(a)]+[
        f"{metric(a,c+'_auc')['mean']:.6f} / {metric(a,c+'_logloss')['mean']:.6f}" for c in CELLS] for a in ARMS]))
    blocks.append(table(['比较']+list(CELLS),[[a+'−'+b]+[
        f"{contrast(a+'_vs_'+b,c+'_auc')['mean']:+.6f} / {contrast(a+'_vs_'+b,c+'_logloss')['mean']:+.6f}" for c in CELLS]
        for a,b in (('G-D0','G-D2'),('G-D02','G-D2'),('G-D02','G-D0'),('G-D0','HYPERGATE'),('G-D02','HYPERGATE'))]))
    blocks += ['**STRONG：双域对D0的d0t0损失重复出现。** D02−D0的d0t0均差−0.014949244、5/5下降，'
        't95%[−0.029018980,−0.000879508]；d0t1为−0.014183380、4/5下降。'
        'D02相对D2的d2t1也为5/5下降（均差−0.006436994）。目标域被直接增强后，排序收益仍随域任务变化。',
        '增强d0相对G-D2的d0t0/d0t1平均改善+0.009575380/+0.000898182，分别4/5、2/5 seed为正；'
        '但相对HyperGate仍落后−0.015592887/−0.007939898。'
        '双域这两格相对HyperGate的缺口扩大到−0.030542131/−0.022123279。',
        '未直接增强的最大域d1也发生变化。相对G-D2，D0的两任务AUC为+0.001799451/+0.002262097，'
        'D02为+0.003514683/+0.004016655。D02相对D0进一步提高d1约0.0017，却主要损失d0。'
        '其Traffic AUC比D0高0.001545409、Traffic LL低0.001225706；六格Macro AUC却低0.004576209，Macro LL高0.000654603。'
        'test中d1占98.2123%，解释了两种汇总权重下的方向差异。',
        '**MEDIUM：相对固定EMA的小域收益/大域代价。** G-D0相对固定EMA的d0两格+0.006930376/+0.012862348，'
        'd1两格−0.002854581/−0.002666910，后两格均5/5下降。G-D0的Traffic AUC比固定EMA低0.002602037。'
        '相对G-D2，双域呈现d1改善、d0及d2多数格下降。取舍方向随比较对象变化。',
        table(['配置','任务0 Macro AUC','任务1 Macro AUC','任务0 Macro LL','任务1 Macro LL','每seed最差格AUC的均值'],
            [[a]+[f"{metric(a,m)['mean']:.6f}" for m in ('task0_macro_auc','task1_macro_auc',
              'task0_macro_logloss','task1_macro_logloss','worst_cell_auc')] for a in ('G-D0','G-D02','G-D2')]),
        'G-D0相对D2的任务0 AUC+0.002710107、任务1−0.000348330；D02相对D0两任务均下降（−0.004265568/−0.004886849）。'
        '三种Gaussian的平均最差格均为d0t0，D0提高至0.643921，D02降至0.628972。',
        '**WEAK：域规模作为主要因果解释。** d0训练人口约为d2的1.93倍，M3oE的d0两任务AUC却更低；'
        'HyperGate下最小平均格转为d2t0。域身份、标签分布、特征支持、增强曝光量与N共同变化。'
        'd0t1的validation/test正例仅101/120；本轮保持人口固定，直接识别覆盖集合的影响，未分离N的独立贡献。',
        '## 6. 训练阶段、验证/测试与在线模型',
        'E10和E50为冻结阶段诊断，主裁定使用validation-selected。完整阶段配对见[STAGE_EFFECTS.csv](STAGE_EFFECTS.csv)，'
        '所有种子及在线对照见[ANALYSIS_VALUES.json](ANALYSIS_VALUES.json)。']
    blocks.append(table(['阶段','配置','Test Macro AUC','Test Macro LL'],
        [[cp,a,f"{j['absolute'][cp]['test'][a]['metrics']['macro_auc']['mean']:.6f}",
          f"{j['absolute'][cp]['test'][a]['metrics']['macro_logloss']['mean']:.6f}"]
         for cp in ('E10','selected','E50') for a in ('G-D0','G-D02','G-D2')]))
    blocks += ['**MEDIUM：阶段均值方向发生反转。** D02−D0在E10为AUC+0.001159284，到selected为−0.004576209、'
        'E50为−0.005048548。D02−D2在E10为+0.003370022，E50转为−0.004924499。'
        'D0−D2在selected的正均值，在E50接近零；选点与训练阶段对该微小增量的贡献需要共同解释。',
        'validation-selected：D0−D2的V均差+0.002976529、test+0.001180888；'
        'D02−D2的V均差+0.000736884、test−0.003395320。'
        'D02−D0则V/test均负（−0.002239645/−0.004576209），均4/5下降。seed2022的D02−D0 test为正，保留为宏趋势的反例。']
    blocks.append(table(['估计器/阶段','D02−D0 Test ΔAUC','Test ΔLL','V ΔAUC','V ΔLL'],
        [[f'online/{cp}']+[f"{j['online'][cp][split]['G_D02_minus_G_D0'][m]['mean']:+.9f}"
          for split,m in (('test','macro_auc'),('test','macro_logloss'),('validation','macro_auc'),('validation','macro_logloss'))]
         for cp in ('selected','E50')]))
    blocks += ['**MEDIUM：双域代价在在线模型共同末轮已经出现。** E50 online的D02−D0为AUC−0.007889939、'
        'LL+0.004916048；LL为5/5恶化。固定EMA与各自选点可以改变幅度，训练轨迹本身也包含差异。',
        table(['配置','E10优化loss / clean BCE','E50优化loss / clean BCE'],
            [['G-D0','0.519679 / 0.519678','0.489461 / 0.489444'],
             ['G-D02','0.518888 / 0.518880','0.488713 / 0.488661']]),
        '双域训练目标和clean BCE更低，同时selected与E50的Macro排序及概率损失劣于D0。'
        '竞争解释包括共享表示/Adam优化轨迹的非加性交互、EMA与选点、有限域任务评估支持和自然train/test分布差异。'
        '共享梯度方向、表示变化及噪声耦合的独立机制证据为WEAK，本轮日志保留损失/视图/状态身份，未新增机制干预。',
        '## 7. 研究问题裁定与下一步',
        '1. 同时增强d0+d2相对仅d2：Macro AUC下降0.003395、Macro LL上升0.001889，未提高本轮Macro主指标。\n'
        '2. 双域相对仅d0：Macro AUC下降0.004576、Macro LL上升0.000655；优先D0。\n'
        '3. D0相对D2改善d0、d1排序，损失d2两格；D02相对D2改善d1，损失d0与d2四格。\n'
        '4. D0缓解d0相对HyperGate的差距，d0t0/t1仍落后0.015593/0.007940；D02扩大缺口。\n'
        '5. d1虽未直接增强，两个任务随覆盖集合改变；相对D2两新配置均改善，相对固定EMA则仍下降。\n'
        '6. 最强发表模型仍是HyperGate。D0超过AutoIFS的AUC均值0.000583，LL上升0.002526；D02低于HyperGate与AutoIFS。\n'
        '7. 当前单域配置足以取得更好的Gaussian结果。AUC优先选D0，Macro LL优先选D2；不优先继续双域扩展。',
        '最有区分力的下一项研究是固定G-D0/G-D2，在新增的完整自然留出批次进行同一五seed配对，'
        '判别D0的局部排序改善能否跨自然人口保留，同时检验其d2和LogLoss代价。'
        '它直接影响两个单域方案的取舍；本轮已完成规定矩阵，不追加新训练配置或超参数搜索。',
        '## 8. 工程检查、修订与实际成本',
        'DESIGN→独立CODER→独立REVIEW→RUN→ANALYZE完成。14项定向CPU检查、两臂batch40960 GPU smoke及RAM往返通过。'
        '审查结论PASS WITH FIXES。正式训练失败0、OOM/状态丢失重放0；'
        '513次派发中13次资源/owner门禁拒发，随后自动重试，500次epoch提交完整。'
        'guardian正常SHUTDOWN，RAM状态在评价后释放。',
        '正式运行前修订两项恢复问题：RAM整体丢失重放采用追加账本中的最新连续提交链并保留旧物理记录；'
        '已封存DONE先校验checksum再释放残余RAM。源码、数据、算法公式、训练矩阵和选点规则保持冻结；'
        '记录见[CODER_IMPLEMENTATION_REPORT.md](CODER_IMPLEMENTATION_REPORT.md)、[REVIEW_REPORT.md](REVIEW_REPORT.md)及[RUNNER_REPORT.md](RUNNER_REPORT.md)。']
    cost_rows = []
    for arm in ('G-D0','G-D02'):
        values = [r for r in costs if r['arm']==arm and r['cost_origin']=='new_training']
        assert len(values) == 5
        total = lambda name: sum(float(r[name]) for r in values)
        cost_rows.append([arm,5,f"{total('training_seconds'):.3f}",f"{total('validation_seconds'):.3f}",
            f"{total('worker_seconds'):.3f}",int(total('optimizer_steps')),int(total('extra_views')),
            f"{max(float(r['peak_cuda_reserved_bytes']) for r in values)/(1<<20):.1f}"])
    blocks += [table(['配置','n','训练秒合计','验证秒合计','worker秒合计','Adam/EMA步数','增强行曝光','峰值reserved MiB'],cost_rows),
        '首次派发至完成墙钟3761.7秒（62.70分钟）；逻辑worker时间13961.696秒，'
        '500次epoch物理worker时间16529.491秒。物理worker计时覆盖注册后的训练、RAM IPC与该launch评价；'
        '协调器墙钟覆盖完整队列。正式原始行曝光1003066000、标签曝光2006132000、增强视图108754000；'
        '每个臂Adam/EMA各12250步。历史复用成本标记为historical_reuse，新增训练时间为0。'
        '两张RTX3090与既有任务共享，逐卡按显存和功耗85%及峰值/RAM余量派发。'
        '环境Python3.13.5、PyTorch2.9.0+cu126、FP32；完整成本见[COSTS.csv](COSTS.csv)。',
        '## 9. 源码、原始记录及复算',
        '原CSV SHA256：a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f。'
        '冻结实现SHA256：c8c76da2cb644559ea1cdf69d56e75fef00ae1fb70e214529708a023884b096d。'
        '基于A+B提交9e682f3e3f608c7d63c2e9306b8657b16d822541；同一实现的源码归档提交'
        'de7947e01c49aaecbd9d438c5ef662de46cca8f9。初始训练基准HEAD与源码归档提交在[运行报告](RUNNER_REPORT.md)区分记录。',
        '实验分支：[research/26-11-04-gaussian-domain-coverage](https://github.com/Caelestis1903/My-Research-Project/tree/research/26-11-04-gaussian-domain-coverage)。'
        '实现见[实验包](../../experiments/gaussian_domain_coverage_20261104/)，'
        '数学和接口合同见[DESIGN_SPECIFICATION.md](DESIGN_SPECIFICATION.md)，'
        '完整数据/源码/配置/选点身份见[RUN_MANIFEST.json](RUN_MANIFEST.json)。',
        '原始标量归档含1581个文件，见[RAW_ARTIFACTS.md](RAW_ARTIFACTS.md)及[清单](RAW_ARTIFACTS_MANIFEST.json)。'
        'checkpoint身份保留epoch、model hash、冻结validation与test端点；模型/optimizer/EMA状态仅驻RAM，评价后释放。'
        '运行本目录analyze.py可从已发布ENDPOINTS.csv复算阶段、在线/EMA、域任务及逐seed对比；'
        'build_report.py从这些冻结数表重建本报告与台账片段。'
    ]
    (HERE/'REPORT.md').write_text('\n\n'.join(blocks)+'\n')
    ledger = [
        '### 11/04 KuaiRand：Gaussian增强领域覆盖（五seed）',
        'FULLTRAIN_NATURAL80_10_10 / AUC_SELECTED_V1；fixed1 M3oE、自然Sample、batch40960、50epoch、'
        'Adam lr=.01/wd1e-5、κ=.1/λ=.5/EMAρ=.995。G-D0=[0]、G-D02=[0,2]各新增五seed，'
        'G-D2=[2]复用既有五seed；新增10条/500epoch。以下均为fixedEMA、validation六格Macro AUC严格最大最早selected的test均值，n=5。',
        '[完整报告](research_reports/2026-11-04_gaussian_domain_coverage/REPORT.md) · '
        '[逐seed/身份](research_reports/2026-11-04_gaussian_domain_coverage/ENDPOINTS.csv) · '
        '[裁定](research_reports/2026-11-04_gaussian_domain_coverage/DECISIONS.json)',
        table(['配置','n']+list(CELLS)+['Macro AUC / LL','Traffic AUC / LL','Pooled AUC / LL'],
              [[a,5]+[f"{metric(a,c+'_auc')['mean']:.6f} / {metric(a,c+'_logloss')['mean']:.6f}" for c in CELLS]+[
                  f"{metric(a,m)['mean']:.6f} / {metric(a,ll)['mean']:.6f}" for m,ll in
                  (('macro_auc','macro_logloss'),('sample_weighted_cell_auc','sample_weighted_cell_logloss'),
                   ('pooled_task_mean_auc','pooled_logloss'))] for a in ('G-D0','G-D02','G-D2')]),
        'G-D0−G-D2：ΔAUC+0.001180888、ΔLL+0.001234785，d2两格越过−.001保护线。'
        'G-D02−G-D0：ΔAUC−0.004576209、ΔLL+0.000654603，d0t0为5/5下降。'
        '最强发表模型仍是HyperGate（0.682112551）；D0/D02/D2相对HG均差为−0.002932373/−0.007508582/−0.004113262，'
        '均未超过最强基线，六格保护均未满足，平均LL均下降。'
        'D0相对M3oE的总ΔAUC+0.007273579、ΔLL−0.001654150，六格满足；'
        '相对固定EMA增量+0.002657179，但d1两格5/5下降。当前AUC优先D0、Macro LL优先D2，不优先双域扩展。',
    ]
    (HERE/'LEDGER_APPEND.md').write_text('\n\n'.join(ledger)+'\n')
    print(json.dumps(dict(report_bytes=(HERE/'REPORT.md').stat().st_size,
                          ledger_bytes=(HERE/'LEDGER_APPEND.md').stat().st_size)))


if __name__ == '__main__':
    main()
