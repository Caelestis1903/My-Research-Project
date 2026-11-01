"""Rebuild the study report from frozen raw tables and the orchestrator's analysis."""
import argparse
import json
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent
NAMES = {'movielens': 'ML', 'kuairand': 'KR'}
ARMS = ['A0', 'A0_EMA', 'A1_EMA']


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
        ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]) + '\n\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    completion = json.loads((run / 'completion.json').read_text())
    assert completion['status'] == 'COMPLETE' and completion['unique_training_runs'] == 56
    test = pd.read_csv(run / 'test_metrics.csv')
    val = pd.read_csv(run / 'validation_metrics.csv')
    effects = pd.read_csv(run / 'effect_summary.csv')
    pairs = pd.read_csv(run / 'paired_effects.csv')
    costs = pd.read_csv(run / 'costs.csv')
    manifest = json.loads((run / 'data_manifest.json').read_text())
    selected = test[test.checkpoint == 'selected']
    assert len(test) == 90 and len(selected) == 30
    def effect(dataset, name, metric, checkpoint='selected', split='test'):
        rows = effects[(effects.dataset == dataset) & (effects.effect == name) &
            (effects.metric == metric) & (effects.checkpoint == checkpoint) & (effects.split == split)]
        assert len(rows) == 1
        return rows.iloc[0]
    def diffs(dataset, name, metric):
        rows = pairs[(pairs.dataset == dataset) & (pairs.effect == name) &
            (pairs.metric == metric) & (pairs.checkpoint == 'selected') & (pairs.split == 'test')]
        assert len(rows) == 5
        return '[' + ', '.join(f'{x:+.6f}' for x in rows.sort_values('seed').delta) + ']'

    text = '''# AIGM01：优化器校准、增强差分与创新量门控完整实验

完成日期：2026-11-01。**56条唯一完整训练、2800 epoch、96600次参数更新，训练失败/OOM均为0；90条正式selected/E10/E50测试记录齐全。** 名义64条中8条以完整身份一致的RAM轨迹复用，实际56条；设计、实现、独立审查PASS、真实CUDA Stage0、开发搜索、正式五seed、统一测试和数值汇总均完成。

**结论：EMA是本轮可靠的排序参考；固定SAD与AIG-M没有建立推进所需的独立增量。普通增强在ML未晋级，在KR出现明确的域/任务交换。AIG-M停止在开发阶段，正式比较保留A0、A0+EMA和A1+EMA。**

| 研究对象 | 当前证据 | 决策 |
|---|---|---|
| ML EMA，H2 | selected test AUC +.003849，5/5正；LL +.001249，4/5恶化，均值未达+.002保护线 | GO；排序STRONG，损失收益未建立 |
| KR EMA，ρ=.995 | AUC +.004616，4/5正；最大域两任务各5/5改善，保护项为空 | GO；整体排序MEDIUM |
| ML普通增强+EMA | 相对A0+EMA AUC +.000318，2/5正；validation均值+.000010 | NO_PROMOTION，WEAK |
| KR普通增强+EMA | macro AUC +.001476，3/5正；LL −.001664，5/5改善；最大域两任务稳定下降 | TRADEOFF；域损害STRONG，macro增量WEAK |
| 固定SAD与AIG-M | 两数据集有限开发搜索均未通过+.001且2/2正向的增量门 | STOP；未建立候选推进证据 |

## 1. 来源、数据和评价身份

入口为start.md和[规划00–06](../../research_plans/26-11-1_v2/06_Codex完整执行提示词.md)。实际规划main为`0973fa5f1a804aec34f88ccb2ffdd8c0231b602b`；GAM01发布HEAD为`ee4cf0512a7723624c9ee6a1ef192f0f7f17686b`。本轮已审执行代码commit为 **`2cec25ab5b32411855a13ab3143678a5e980ea3f`**，训练源码manifest SHA256为`d297bea130cae8fedaeef47ae35239fd9af4f4d7f3076416eaf50670bcdae02c`，完成后再次核对一致。

计划、GAM01/GA31-R3依赖和GAM01报告90份文件逐blob匹配；PROJECT_ARCHITECTURE另行核验。原始CSV、六数组、split/行号、encoder/scaler与固定来源一致，见[SOURCE_FETCH_RECEIPT](SOURCE_FETCH_RECEIPT.json)、[LOCAL_SOURCE_VERIFICATION](LOCAL_SOURCE_VERIFICATION.json)、[RAW_DATA_VERIFICATION](RAW_DATA_VERIFICATION.json)和[实际数据manifest](raw_attempt_tables/data_manifest.json)。保留旧代码/报告，历史勘误只在新[证据审查](EVIDENCE_REVIEW.md)中记录。

| 数据 | train / validation / test | train域d0/d1/d2 | 增强目标 | 任务 |
|---|---|---|---|---|
| ML | 800167 / 100021 / 100021 | 155949 / 318599 / 325619 | d0 | rating>3、rating>4 |
| KR | 2006132 / 250767 / 250767 | 172782 / 1743898 / 89452 | d2 | is_click、long_view |

ML域为ages1/18、age25、ages35/45/50/56；KR域为tab0/1/4。自然原行序80/10/10，KR先全局划分再过滤，train拟合编码器/scaler，OOV0。CSV路径分别为`/home_new/xzf/datasets/ml-1m/ml-1m.csv`和`/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv`。不改变人口、标签、采样比例、原Sample权重预算或网络结构。

模型fixed1 MDMTRec/M3oE，3域×2任务、4 shared experts、embedding16、fcn_dims=[512,256,256,64]，四个融合系数为1。batch40960、50epoch、FP32、loss_scale1、自然shuffle，无早停/AMP/TF32。环境Python3.13.5、PyTorch2.9.0+cu126、CUDA12.6、两张RTX3090；[依赖锁](../../requirements-phase0-3.lock)和[环境记录](raw_attempt_tables/environment.json)保留实际版本。GPU0既有外部训练保留，逐卡按显存和功耗85%门并发。

两数据集均由开发搜索选择Adam lr=.01、betas(.9,.999)、eps1e−8，coupled wd分别1e−4/1e−5；wd同时进入m/v。ML正式EMA半衰期H2，对应ρ=.982820598545；KR保留ρ=.995，约2.82epoch半衰期。ML增强参考κ=.05、λ=.5；KR κ=.1、λ=.5。Gaussian K1仅扰动目标域非路由sparse embedding，RMS尺度stopgrad；其他数据字段与标签保持原值。

online与各shadow独立按完整validation六格macro AUC严格提升选epoch，平局最早，E0排除。S5前冻结方法、ρ、超参、代码/数据身份；全体训练后冻结90个端点，再统一test。[配置冻结](raw_attempt_tables/formal_configuration_frozen.json)、[选点冻结](raw_attempt_tables/selection_frozen.json)、[逐seed身份](raw_attempt_tables/selected_metrics.csv)提供epoch、step、model/optimizer hash。

macro为六格等权；pooled先按任务跨域计算AUC再平均；traffic为域人口加权cell AUC。LL为真实输入概率LogLoss，不乘loss_scale。seed为重复单位，sample SD和df4配对t95%区间由五个seed计算。2022/2024参与开发选参并在正式阶段复用，五seed不称独立确认；历史test已参与既往研究判断。区间描述固定划分上的训练随机性，不增加task/epoch为样本。

## 2. 有限搜索、算法和阶段去留

完整原值、配对区间、保护项、阈值与各shadow网格见[SEARCH_REPORT](SEARCH_REPORT.md)、[开发checkpoint原值](development_checkpoints.csv)和[开发汇总](development_summary.csv)。

| 阶段 | 完整训练 | 选择/结果 |
|---|---:|---|
| S1 Adam四联合lr/wd点×两seed×两数据 | 16 | 两数据均C；ML H2，KR .995（近似平局顺序） |
| S2 Gaussian三κ/λ点 | 12 | ML G1/H2，KR G2/.995；开发增量门均未通过，仍保留正式参考 |
| S3 固定SAD βδ=.95/.99、λ=.5 | 8 | ML最优观察与KR最优观察各只有1/2 seed正向，停止固定记忆晋级 |
| S4 AIG-M两个组合 | 8 | 两数据均无晋级候选 |
| S5 A0/A1两online×两数据×五seed | 20个online输出 | 8条严格RAM缓存、12条新训练；30个逻辑模型-seed输出 |

SAD真实通道累计g+wd·θ，q只累计增强差分d；v使用当前(g+wd·θ+λd)²，分子用各自偏差修正的m+λq。d来自同θ、同batch的target-aug减target-clean，分母为原B×2。AIG-M固定βδ=.95，按预注册E/P参数组计算u=||d−q_old||/(||q_old||+||g||+1e−12)，λ=λmax·sigmoid(4(u−τ))；U组严格Adam，无q。首版仅动态λ，未实施动态β或AIG-SWA，完整公式与退化关系见[设计](DESIGN_SPECIFICATION.md)。

M1=(训练q50,λmax.25)、M2=(q75,λmax.5)。τ来自S95的E1、两个开发seed、E/P两组共276条训练统计，未用validation/test拟合阈值数值。开发门为相对所选增强+EMA mean ΔAUC≥.001、2/2正且保护为空；正式主门为≥.001、至少4/5正及validation支持。保护为LL mean恶化≥.002，或worst/最大域任一任务AUC mean下降≥.002，并达到开发2/2或正式4/5不利seed。

ML最好的AIG观察M1/H2：两个ΔAUC为[+.000512915,+.000006961]，mean+.000259938，未达门；KR M2/.995为[+.006292613,−.006321216]，mean−.000014301，方向不稳定。停止条件已满足，未为被拒AIG增加五seed测试或额外搜索。基础lr/wd只做共享底座校准，没有逐方法独立lr/wd搜索；各方法计算成本另列。

![创新门训练曲线](figures/innovation_gate_curves.png)

曲线是两个开发seed的epoch平均gate，阴影为sample SD；E为embedding、P为目标路径。gate多数在.48–.61，ML E组从约.51上升到.59，其他组上升较缓。平均λ为M1约.13、M2约.26，q注入非零；门改变了更新，但没有达到收益门。实际剂量、训练时钟与创新信号效用没有被分离：没有与平均剂量/时程匹配的恒定或回放控制。故“创新度包含有效额外信息”的证据WEAK。逐组分阶段值见[aig_mechanism_summary.csv](aig_mechanism_summary.csv)。

## 3. 正式selected结果

AUC/LL为五seed均值±sample SD，其他列为均值。A0与A0_EMA共享online轨迹。

'''
    rows = []
    for d in NAMES:
        for arm in ARMS:
            x = selected[(selected.dataset == d) & (selected.arm == arm)]
            v = val[(val.dataset == d) & (val.arm == arm) & (val.checkpoint == 'selected')]
            assert len(x) == len(v) == 5
            rows.append([NAMES[d], arm, f'{v.macro_auc.mean():.6f}',
                f'{x.macro_auc.mean():.6f} ± {x.macro_auc.std():.6f}',
                f'{x.macro_logloss.mean():.6f} ± {x.macro_logloss.std():.6f}',
                *[f'{x[k].mean():.6f}' for k in ['pooled_task_mean_auc', 'sample_weighted_cell_auc', 'worst_cell_auc']]])
    text += table(['数据', '臂', 'val AUC', 'test macro AUC', 'test macro LL', 'pooled AUC', 'traffic AUC', 'worst AUC'], rows)
    text += '### 五个配对差与区间\n\nAUC正值有利，LL负值有利；seed顺序2022–2026。\n\n'
    rows = []
    for d in NAMES:
        for name in ['EMA_A0', 'augmentation']:
            for metric in ['macro_auc', 'macro_logloss']:
                s = effect(d, name, metric)
                rows.append([NAMES[d], name, metric, diffs(d, name, metric),
                    f"{s['mean']:+.6f} / {s.sample_sd:.6f}", f'[{s.ci95_low:+.6f},{s.ci95_high:+.6f}]',
                    f'{int(s.positive if metric == "macro_auc" else s.negative)}/5'])
    text += table(['数据', '对照', '指标', '五个原始差', 'mean / sample SD', 'df4 95%配对区间', '有利seed'], rows)
    text += '''![selected五seed配对效应](figures/selected_seed_effects.png)

ML EMA的selected val AUC也5/5正、mean+.002847；test五差很集中，排序证据STRONG。LL mean+.001249、4/5恶化，区间跨0；GO来自排序门和预设保护，不代表LL已经改善。KR EMA val5/5、test4/5正，AUC区间正；pooled和traffic均5/5改善，最大域两任务均5/5改善，当前协议排序证据MEDIUM，稀有域与LL一致性更弱。

ML增强相对A0_EMA的test AUC只有2/5正，均值+.000318、区间跨0，未达门。KR增强test macro+.001476但3/5正、区间跨0；validation mean−.000629、2/5正。test LL五seed都改善，mean−.001664、区间全负；排序与损失、validation与test的方向分裂必须保留，不能用macro均值单独推进。

### 逐seed绝对值及选点\n\n下表每个AUC/LL单元格对应同一validation-selected模型；完整model/optimizer hash与所有共同epoch在selected_metrics.csv和selection_frozen.json。

'''
    rows = []
    for d in NAMES:
        for seed in range(2022, 2027):
            x = selected[(selected.dataset == d) & (selected.seed == seed)].set_index('arm')
            rows.append([NAMES[d], seed, '/'.join(str(int(x.loc[a, 'epoch'])) for a in ARMS)] +
                [f'{x.loc[a, "macro_auc"]:.6f} / {x.loc[a, "macro_logloss"]:.6f}' for a in ARMS])
    text += table(['数据', 'seed', 'epoch A0/EMA/A1', 'A0 AUC/LL', 'A0_EMA AUC/LL', 'A1_EMA AUC/LL'], rows)
    text += '## 4. 所有域任务、规模、难度与交换\n\n下表均为selected test五seed均值；ΔEMA=A0_EMA−A0，ΔAug=A1_EMA−A0_EMA；括号为AUC正向seed数。\n\n'
    rows = []
    for d in NAMES:
        base = selected[(selected.dataset == d) & (selected.arm == 'A0')]
        for domain in range(3):
            for task in range(2):
                k = f'd{domain}t{task}'
                a = effect(d, 'EMA_A0', k + '_auc'); b = effect(d, 'augmentation', k + '_auc')
                la = effect(d, 'EMA_A0', k + '_logloss'); lb = effect(d, 'augmentation', k + '_logloss')
                stats = manifest[d]['cell_stats']['test'][k]
                rows.append([NAMES[d] + ' ' + k, manifest[d]['train_domain_counts'][str(domain)],
                    f"{stats['count']}/{stats['positive_count']}", f'{base[k+"_auc"].mean():.6f}',
                    f"{a['mean']:+.6f} ({int(a.positive)})", f"{b['mean']:+.6f} ({int(b.positive)})",
                    f'{base[k+"_logloss"].mean():.6f}', f"{la['mean']:+.6f}", f"{lb['mean']:+.6f}"])
    text += table(['cell', 'train域n', 'test n/正例', 'A0 AUC', 'ΔEMA AUC (+seed)', 'ΔAug AUC (+seed)', 'A0 LL', 'ΔEMA LL', 'ΔAug LL'], rows)
    text += '''**KR存在小域收益/大域损害交换。** 增强相对A0_EMA：d1t0下降.004654、5/5，d1t1下降.004929、4/5，validation/test都触发最大域保护。目标最小域d2两任务均值提高，同时d0t1提高约.011964；d0t0下降约.002645，说明小域也不是统一受益。test最大域占约98.21%，pooled−.004494、traffic−.004623，各4/5下降；macro+.001476不能覆盖这项损害。域损害证据STRONG，最小域每任务稳定程度见上表。

EMA没有同样的AUC交易：KR最大域两任务各5/5改善，ML worst-cell也5/5改善。KR EMA worst-cell只有3/5改善，稀有域效果仍有seed差异。ML增强的所有cell均差较小，未建立稳定的域级独立收益。

**域规模作为主解释变量的证据WEAK。** ML最大的d2 baseline AUC低于最小d0；KR最低AUC的d0比最小d2更大。任务难度、正例率、增强目标、共享路径和自然train/test人口变化都与规模混杂。KR d0t1测试只有120正例，d1流量极高；LL也受标签基率影响。本轮没有独立操纵规模，不能把小域等同难域，或用规模统一解释效果。

按任务三域等权汇总如下；完整逐seed域/任务AUC和LL在原始指标CSV。

'''
    rows = []
    for d in NAMES:
        for task in range(2):
            for name in ['EMA_A0', 'augmentation']:
                a = effect(d, name, f'task{task}_macro_auc'); b = effect(d, name, f'task{task}_macro_logloss')
                rows.append([NAMES[d], task, name, f"{a['mean']:+.6f}", f'{int(a.positive)}/5', f"{b['mean']:+.6f}"])
    text += table(['数据', 'task', '对照', 'ΔAUC', 'AUC正向', 'ΔLL'], rows)
    text += '''KR增强收益集中task1：mean ΔAUC+.004328、5/5正；task0为−.001376，只有2/5正。目标域、标签语义与任务共享是比单一规模变量更具体的竞争解释。

## 5. 训练阶段、时间尺度与历史对照

共同epoch与各自selected分开报告，以下为test配对均值。

'''
    rows = []
    for d in NAMES:
        for name in ['EMA_A0', 'augmentation']:
            for checkpoint in ['E10', 'selected', 'E50']:
                a = effect(d, name, 'macro_auc', checkpoint); b = effect(d, name, 'macro_logloss', checkpoint)
                rows.append([NAMES[d], name, checkpoint, f"{a['mean']:+.6f}", f'{int(a.positive)}/5', f"{b['mean']:+.6f}"])
    text += table(['数据', '对照', '端点', 'ΔAUC', 'AUC正向', 'ΔLL'], rows)
    text += '''KR .995 EMA在E10仍5/5排序下降，E50则5/5排序和LL改善；平均的启动和训练时钟是明确因素。ML H2在E10排序已5/5改善，E50排序也5/5改善，但E50 LL mean+.003301、4/5恶化；selected LL代价不能仅归因于选点。

本轮A0 online的两数据×五seed×三个端点共30条model hash和AUC/LL与GAM01完全相同。KR保留历史C/.995及G2/.995配置，历史与本轮不合并增加n；复现证明来源/非反馈语义一致，仍是固定历史划分的五个seed。

ML H2相对历史A0 .995 selected：test AUC mean−.000550201，只有1/5正；LL mean−.006550953。开发两seed的validation却倾向H2，说明时间尺度在validation/test上的排序不同。原.995半衰期约6.91epoch，H2更快，E10和E50的损失代价都明显缩小；当前选择保留排序收益并降低LL代价，但仍有4/5 seed的selected LL轻微恶化。没有根据新test改回.995或追加温度拟合。

历史Polyak的旧协议结果只作研究动机，不与当前数值合并。可核验旧摘要，原始v2_momentum日志不在工作区；该缺口已在EVIDENCE_REVIEW注明。GAM01报告中ML A1−A0、A2−A1的LL改善seed数均为2/5；原表3/5在新审查中勘误，历史文件不改。规划EMA半衰期69/28epoch的算术误差也只在新规格修正为6.91/2.82。

## 6. 证据、竞争解释与下一判别实验

| 判断 | 强度 | 证据与剩余解释 |
|---|---|---|
| ML EMA提高当前协议排序 | STRONG | val/test均5/5正、AUC区间全正、worst5/5改善；LL收益未建立 |
| KR EMA在当前协议可用 | MEDIUM | val5/5、test4/5正，最大域两任务5/5改善；稀有域与损失较不一致 |
| KR增强损害最大域 | STRONG | d1t0五seed下降、d1t1四seed下降，val/test均触发保护 |
| 普通增强提供稳定独立macro增量 | WEAK | ML2/5正，KR3/5且val/test反向；KR的LL改善与域损害并存 |
| 注册AIG-M值得推进 | WEAK | ML增量太小，KR种子方向抵消；预定停止门成立 |
| 创新度信号本身有额外效用 | WEAK | 门有动态和非零q，但剂量/训练日程未匹配控制，未达实用收益门 |
| 规模统一解释域效果 | WEAK | baseline难度排序与规模不一致，任务/目标/人口混杂 |

稳定现象是EMA排序收益、KR普通增强的最大域代价，以及有限差分候选未建立推进增量。矛盾包括KR增强validation/test反向、ML时间尺度validation/test排序反向、ML小幅AUC收益与LL代价，以及SAD短记忆在两数据上没有统一优势。新规则的非零注入说明实现改变了更新；q、正cos或创新量大都不是效用指标。历史A2的实数等价/长期FP32分离也要求区分数值轨迹差异与机制收益。

竞争解释按当前证据保留：EMA的参数轨迹平均及启动滞后；AUC选点与概率校准的差别；目标增强的标签/特征语义及共享参数引起的域交换；差分历史的滤波滞后与参数漂移；AIG较低平均剂量和近似epoch日程，而非创新信号；小测试cell和自然人口变化引起的指标方向差异。共享lr/wd校准没有识别每个新规则的独立最优lr/wd，本轮结论对应注册有限设置。

**最有区分力的下一实验**：保持当前公式、自然人口、κ/β/ρ和总注入剂量，把AIG的u驱动门替换为开发训练曲线的逐epoch/参数组平均λ回放日程，与u门及简单EMA参考并列。先冻结全部规则，再在未参与方法选择的新自然留出评价，并先核验行号、时间和人口身份。若u门不能超过剂量/时钟匹配的回放，拒绝“创新信号提供独立信息”的解释。该实验区分当前主要竞争解释，不为现有失败候选扩大搜索；本轮到预注册停止点结束。

## 7. 成本、验收与实际交付

'''
    keys = [('原始自然样本曝光次数', 'original_exposures'), ('目标增强视图曝光次数', 'extra_views'),
        ('训练forward', 'training_forwards'), ('backward/autograd遍历', 'gradient_calls'),
        ('optimizer更新', 'optimizer_steps'), ('四shadow累计更新', 'shadow_updates'),
        ('validation forward（online+四shadow）', 'validation_forwards'), ('选点比较次数', 'selection_checks')]
    text += table(['成本', '实际合计'], [[name, f'{int(costs[key].sum()):,}'] for name, key in keys])
    text += f'''累计训练worker {costs.training_seconds.sum():.2f}秒；all_stages至统一test/数表完成wall {completion['wall_seconds']:.2f}秒（{completion['wall_seconds']/60:.2f}分钟），不含前期设计、Stage0、数据预处理和本报告。训练CUDA-event累计{costs.training_gpu_event_ms.sum()/1000:.2f}秒，validation forward event累计{costs.validation_gpu_event_ms.sum()/1000:.2f}秒；这些是各进程stream elapsed合计，训练区间含host/并发等待，不能当作独占kernel GPU小时或算法加速结论。test去重后422次forward，worker wall29.87秒、forward event22.43秒。峰值allocated {costs.peak_memory_bytes.max()/1024**3:.3f}GiB、reserved {costs.peak_reserved_bytes.max()/1024**3:.3f}GiB；RAM快照tensor合计{completion['snapshot_tensor_bytes']:,}bytes，结束后已释放。

开发44条、正式20个online输出及8次缓存的身份/原因分别在[trial ledger](raw_attempt_tables/trial_ledger.csv)、[launch ledger](raw_attempt_tables/launch_ledger.csv)、[costs](raw_attempt_tables/costs.csv)。失败账本为空表，失败/OOM0；没有删seed、缩batch、缩epoch或训练期提前剪枝。

必要验收：coder CPU测试3 passed、独立reviewer复核3 passed、最终PASS；真实CUDA Stage0每数据集19次原batch更新，源预测/BCE/梯度/Adam一步误差0，U差分精确零，四shadow非干扰和完整RAM恢复下一步PASS。完成后核对56条×50epoch、90条测试身份、成本计数与冻结源码一致。没有因本轮实现缺陷重训的记录。公式歧义、β0哨兵与历史勘误在[设计](DESIGN_SPECIFICATION.md)、[实施](IMPLEMENTATION_REPORT.md)、[独立审查](REVIEW_REPORT.md)、[Stage0](STAGE0_REPORT.md)中保留修订依据。

所有模型、optimizer/q、shadow与RNG快照仅RAM；[state_release](raw_attempt_tables/state_release.json)确认测试和数表后释放。磁盘保留数值、配置、hash、代码与日志，模型复评需按冻结入口重训。

## 8. 复算与重训

全套原始数值/日志归档为`raw_attempts_2026-11-01_aig_m.tar.gz`，含check和all两个attempt；逐文件SHA256为`raw_attempts_2026-11-01_aig_m_SHA256SUMS.txt`。直接可读顶层数表在[raw_attempt_tables](raw_attempt_tables/)，详细工件说明见[README](README.md)。不随包上传原始CSV或权重。

在仓库根目录解包恢复原attempt后，可复算数表、开发表、图与本文：

```bash
tar -xzf research_reports/2026-11-01_aig_m/raw_attempts_2026-11-01_aig_m.tar.gz -C .
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m experiments.gradient_aug_aig_m_20261101.analyze --run-dir artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-01_aig_m/development_tables.py --run-dir artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-01_aig_m/make_figures.py --run-dir artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python research_reports/2026-11-01_aig_m/build_report.py --run-dir artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720
```

完整重新训练使用新的空attempt目录，按[代码运行入口](../../experiments/gradient_aug_aig_m_20261101/README.md)先执行Stage0，再传入同一已审commit及PASS receipt运行all；完整配置、数据hash、阈值原值和候选清单均在本交付中。当前结论与研究去留是：保留EMA基线，KR增强按TRADEOFF保留描述，停止本轮固定SAD/AIG-M推进。
'''
    text = text.replace('raw_attempt_tables/', 'raw_attempt_tables/attempt_20261101_1720/')
    text = text.replace('ML最优观察与KR最优观察', 'ML所选观察与KR所选观察')
    text = text.replace('selected LL轻微恶化', 'selected LL恶化')
    text = text.replace('tar -xzf research_reports/2026-11-01_aig_m/raw_attempts_2026-11-01_aig_m.tar.gz -C .',
        'mkdir -p artifacts/gradient_aug_aig_m_20261101\n'
        'tar -xzf research_reports/2026-11-01_aig_m/raw_attempts_2026-11-01_aig_m.tar.gz -C artifacts/gradient_aug_aig_m_20261101')
    (OUT / 'REPORT.md').write_text(text)
    print(f'Report written: {len(text)} characters')


if __name__ == '__main__':
    main()
