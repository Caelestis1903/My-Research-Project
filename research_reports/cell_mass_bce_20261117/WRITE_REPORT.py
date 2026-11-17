"""Foreground parent report assembled from accepted numeric evidence."""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
RAW = Path('/tmp/phase0-3-c05-01a1e9a9/attempt1')


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def table(headers, values):
    return ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)] + ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in values] + ['']


def number(value):
    return f'{float(value):+.9f}'


def main():
    decision = json.loads((OUT / 'DECISION.json').read_text())
    assert decision['decision'] == 'NO_CONTINUATION_SIGNAL'
    summary = rows(RAW / 'tables/ALL_PAIRED_SUMMARY.csv')
    pairs = rows(RAW / 'tables/ALL_PAIRED_SEEDS.csv')
    means = rows(OUT / 'ENDPOINT_MEANS.csv')
    index = {(r['dataset'], r['comparison'], r['endpoint'], r['metric']): r for r in summary}
    report = ['# C05 CELL-MASS-BCE-025：完整研究报告', '',
        '结论：NO_CONTINUATION_SIGNAL。按预先固定的 selected fixed_ema 主终点，U−C 在 MovieLens 和 KuaiRand 的五 seed 平均 Macro AUC 都非正，均未达到 +0.003。ML 六 cell 保护门槛通过，KR d0t0 未通过 −0.001 保护线。两套数据的平均 validation LL 也均未下降。此为当前协议的筛选结果，不能解释为已经证明总体效应严格为零。', '',
        '本轮已完成 DESIGN→IMPLEMENT→独立 REVIEW→RUN→主智能体 ANALYZE：10条新 U 轨迹各50 epochs、500次已确认提交、17250次有用 Adam/EMA 更新。复用 C/W/S 数值控制30条，共40条唯一轨迹；统计单位始终是五个配对 seed。原始前台父智能体直接 get 验收20个选中模型，实际张量95508960字节。Test 未开启，未选择 champion。本轮按用户最终指令交付后停止，不启动 C06。', '',
        '完整定义与固定协议见 [METHODS.md](METHODS.md)，全部逐 seed / stage / cell 绝对值见 [NUMERICAL_APPENDIX.md](NUMERICAL_APPENDIX.md)，工程失败、恢复与来源修正见 [EXECUTION_HISTORY.md](EXECUTION_HISTORY.md)。所有区间为 n=5、ddof1 的描述性 t4 区间，相关探索比较未作多重校正。', '',
        '## 1. 主终点与逐 seed 结果', '']
    primary = []
    for dataset in ('movielens', 'kuairand'):
        for metric in ('macro_auc', 'macro_logloss'):
            r = index[dataset, 'U_fixed_ema-C_fixed_ema', 'selected', metric]
            primary.append([dataset, metric, number(r['mean']), number(r['sd']), '[' + number(r['ci95_low']) + ', ' + number(r['ci95_high']) + ']', r['positive'] + '/' + r['negative']])
    report += table(['dataset', 'U−C selected EMA', 'mean', 'SD', '95% CI', '+ / − seeds'], primary)
    report += table(['dataset', 'seed', 'U AUC', 'C AUC', '差值', 'U选中epoch', 'C选中epoch'],
                    [[r[k] for k in ('dataset', 'seed', 'left', 'right', 'delta', 'left_epoch', 'right_epoch')]
                     for r in pairs if r['comparison'] == 'U_fixed_ema-C_fixed_ema' and r['endpoint'] == 'selected' and r['metric'] == 'macro_auc'])
    report += ['ML 为1正4负；KR 为4正1负，但 seed2024 的差值−0.009762547使总体均值转负。该 seed 的 U/C 选点分别为epoch30/32，均遵循独立最早严格最大 AUC；不删除该点、不改用中位数替换主均值，也不以其他 seed 的方向投票重写主判定。KR 区间宽且跨零，疗效方向证据为 WEAK；ML 区间同样跨零，不能把小负均值直接写成确定伤害。未达到预设门槛这一运行内事实为 STRONG。', '', '## 2. 绝对表现与模型 stream', '']
    absolute = []
    for dataset in ('movielens', 'kuairand'):
        for arm in ('CLEAN', 'MASS-SCALE-025', 'COMPLETION-BCE-025', 'CELL-MASS-BCE-025'):
            for kind in ('online', 'fixed_ema'):
                selected = {r['metric']: r for r in means if (r['dataset'], r['arm'], r['estimator'], r['endpoint']) == (dataset, arm, kind, 'selected')}
                absolute.append([dataset, arm, kind, selected['macro_auc']['mean'], selected['macro_auc']['sd'], selected['macro_logloss']['mean'], selected['macro_logloss']['sd'], selected['epoch']['mean']])
    report += table(['dataset', 'arm', 'stream', 'AUC mean', 'AUC SD', 'LL mean', 'LL SD', '平均选中epoch'], absolute)
    report += ['U_ema−C_online 的 ML AUC 为+0.002775289，五 seed 均正且描述性区间全正；但 matched EMA 主比较为负。固定 EMA 在 CLEAN 自身也提供收益，跨 stream 优势不能归因于 U 或替代主终点。U_online−C_online 的 ML/KR AUC 分别−0.000043691/+0.000004129，区间均跨零；online 也未提供稳定救援。', '']
    report += ['## 3. 六个 cell、domain 与 task', '']
    regional = []
    metrics = [f'd{d}t{t}_{m}' for d in range(3) for t in range(2) for m in ('auc', 'logloss')]
    metrics += [f'{scope}{i}_macro_{m}' for scope, n in (('domain', 3), ('task', 2)) for i in range(n) for m in ('auc', 'logloss')]
    for dataset in ('movielens', 'kuairand'):
        for metric in metrics:
            r = index[dataset, 'U_fixed_ema-C_fixed_ema', 'selected', metric]
            regional.append([dataset, metric, number(r['mean']), '[' + number(r['ci95_low']) + ', ' + number(r['ci95_high']) + ']', r['favorable'] + '/5'])
    report += table(['dataset', 'U−C selected EMA 指标', 'mean', '95% CI', '有利seed'], regional)
    report += ['ML 的 d1t0 AUC 五 seed 均下降，描述性区间全负；d2t1 五 seed 均上升但区间跨零，不能只凭同号宣称确定收益。KR 最差保护 cell d0t0 平均−0.007526250、1正4负；d0t1 均值正但2正3负且区间极宽。cell、task 和 domain 的聚合回答不同问题，不用一个有利 cell 替代六 cell 主指标。', '',
               '## 4. C/S/W/U 归因：F2 与 F3', '']
    attribution = []
    for dataset in ('movielens', 'kuairand'):
        for comparison in ('U_fixed_ema-W_fixed_ema', 'U_fixed_ema-S_fixed_ema', 'W_fixed_ema-S_fixed_ema', 'S_fixed_ema-C_fixed_ema', 'W_fixed_ema-C_fixed_ema'):
            for metric in ('macro_auc', 'macro_logloss', 'task1_macro_logloss'):
                r = index[dataset, comparison, 'selected', metric]
                attribution.append([dataset, comparison, metric, number(r['mean']), '[' + number(r['ci95_low']) + ', ' + number(r['ci95_high']) + ']', r['favorable'] + '/5'])
    report += table(['dataset', 'selected EMA 比较', '指标', 'mean', '95% CI', '有利seed'], attribution)
    report += ['F2 未复现两套数据都下降的方向：U−S 的 task1 LL 为 ML−0.000733901（3/5下降，区间跨零），KR+0.000712949（1/5下降，区间跨零）。相比之下，复用 W−S 的 task1 LL 为 ML−0.002988932、KR−0.005124581，均5/5下降且区间全负。因而在当前协议下，单保留 cell/task 质量分配不足以重复 W 的跨数据校准收益。此归因的证据为 MEDIUM；不能升格为“类内异权在任何模型都必要”。', '',
               'U−W 的 task1 LL 在两套数据都5/5变差：ML+0.002255031（区间跨零），KR+0.005837530（区间全正）；KR Macro LL +0.004518754，同样5/5变差、区间全正。这是当前配对实验内对移除类内类别政策所付校准代价的 STRONG 证据，ML 部分的幅度判断为 MEDIUM。W/U 的长期差异包含该政策导致的 Adam、EMA 和轨迹历史；不计算“中介贡献百分比”。', '',
               'F3 也未满足 AUC 与 LL 同时改善：KR d0t0 的 U−W AUC−0.004384878（1/5改善，区间跨零），LL−0.001922553（4/5改善，区间跨零）。LL 方向改善没有救回 AUC，且 U−C 的该 cell 均值仍低于保护线。不能把单项校准改善写成小域救援成功。', '',
               '## 5. 训练阶段、选点与 EMA', '', '![预设阶段 AUC 差及描述性区间](STAGE_AUC.png)', '']
    stage = []
    for dataset in ('movielens', 'kuairand'):
        for endpoint in ('E1', 'E10', 'E30', 'E50', 'selected'):
            for metric in ('macro_auc', 'macro_logloss'):
                r = index[dataset, 'U_fixed_ema-C_fixed_ema', endpoint, metric]
                stage.append([dataset, endpoint, metric, number(r['mean']), '[' + number(r['ci95_low']) + ', ' + number(r['ci95_high']) + ']', r['favorable'] + '/5'])
    report += table(['dataset', '阶段', 'U−C EMA', 'mean', '95% CI', '有利seed'], stage)
    report += ['E1 两套数据的 AUC 均5/5下降且区间全负，均值分别−0.009865868/−0.010638697；这是 STRONG 的早期重复现象。E50 AUC 则分别+0.000701257/+0.000880189，均4/5为正但区间跨零，证据 WEAK；它不满足+0.003，也不能替换 selected 主终点。早期损害、末期点均值改善与 selected 不通过并存，说明训练阶段和选点必须分开报告；本轮没有直接检验 Curriculum 的有效性。', '',
               '![selected EMA 各 cell 的 AUC 差](CELL_AUC.png)', '', '## 6. domain 大小、难度与权衡', '']
    baseline = rows(OUT / 'BASELINE_DOMAIN_CONTEXT.csv')
    report += table(['dataset', 'domain', 'train rows', 'CLEAN EMA AUC mean', 'CLEAN EMA LL mean'],
                    [[r[k] for k in ('dataset', 'domain', 'train_rows', 'clean_selected_ema_auc_mean', 'clean_selected_ema_logloss_mean')] for r in baseline])
    report += ['ML 最小的 d0 有155949训练行，baseline AUC 约0.767958，反而高于大域 d1/d2 的0.756878/0.754468；小域不等同于困难域。KR 最小 d2 的baseline AUC较低，但另一个较小域d0与d2在U−C中的点均值方向相反。类别率与任务定义也影响 LL 水平，不能仅凭原始 LL 跨域大小定义内在难度。完整类别计数和绝对 baseline 见 BASELINE_CELL_CONTEXT.csv。', '',
               'U−C 的 ML 三个domain AUC均值均为负；KR 的 d2点均值上升、主导大域d1略降，局部存在“小域收益/大域下降”的点估计权衡，但小域d0同时下降，各KR domain区间均跨零。没有稳定跨域、跨数据的普遍权衡；domain规模作为主要解释变量的证据为 WEAK。C04中KR d0/d2方向相反的反例继续成立，不能选择性只看d2。', '']
    report += ['## 7. 质量账本与共同状态诊断', '']
    mass = rows(RAW / 'tables/MASS_SUMMARIES.csv')
    mass_values = []
    for dataset in ('movielens', 'kuairand'):
        selected = [r for r in mass if r['dataset'] == dataset and r['scope'] == 'total' and r['mass_kind'] == 'bce_mass_scale']
        average = sum(float(r['row_weighted_sum']) for r in selected) / sum(int(r['rows']) for r in selected)
        mass_values.append([dataset, len(selected), average, 1 - average])
    report += table(['dataset', 'seed×epoch记录', '按训练行加权平均m', '1−m'], mass_values)
    report += ['U/W 的 cell/task/总 BCE 系数质量按设计相同；本轮没有恢复被移除的排序图，也没有额外类别平衡或重归一化。账本中空类和未计算统计仍保留 null 原因；不能把工程 λ0 的 H=null 当成正式 H=0。40个固定阶段的 epoch-order 分组在 C/S/W/U 间全部一致。', '']
    diag = rows(RAW / 'tables/DIAGNOSTIC_MEANS.csv')
    di = {(r['dataset'], r['epoch'], r['parameter_group'], r['objective_or_pair'], r['metric']): r for r in diag}
    diagnostic = []
    for dataset in ('movielens', 'kuairand'):
        for epoch in ('1', '50'):
            for pair in ('U-W', 'U-S', 'S-C'):
                vals = [di[dataset, epoch, 'ALL', pair, metric] for metric in ('data_gradient.cosine', 'parameter_update.cosine', 'parameter_update.difference_norm')]
                diagnostic.append([dataset, epoch, pair, *[r['mean'] for r in vals], vals[0]['valid_seed_count']])
    report += table(['dataset', 'epoch', '同一U状态反事实比较', 'data grad cosine mean', 'source Adam update cosine mean', 'update差范数mean', 'seed n'], diagnostic)
    report += ['S−C 的数据梯度 cosine 几乎为1，但源 Adam 虚拟更新差范数非零，说明在当前 Adam状态、耦合WD和eps约定下不能把全局缩放视作完全相同更新。此为 STRONG 的共同状态数值事实；没有独立消融时，不能断言某一个项就是唯一原因。', '',
               'E50 的 U−W 数据梯度 cosine 为 ML0.909131、KR0.573357，而对应更新 cosine 为0.997781/0.971004；U−S 的梯度与更新方向则更接近1。自适应状态改变了数据梯度差异映射到参数更新的方式，支持优化轨迹作为竞争解释。诊断仅在真实U轨迹状态上比较反事实C/S/W，不能当作其他三条独立轨迹的状态，也不能证明 momentum机制普遍有效。', '',
               '实际40诊断点、120额外sweep、160虚拟Adam均已对账；真实U/虚拟U、诊断中立性、prefix与恢复suffix通过精确性检查。deltaEMA=(1−rho)deltaTheta 是共同前态的代数恒等式，不能再算作额外独立支持样本。按参数组的全部活动、梯度、private step、null及阶段信息保留在原始COMMON_STATE_DIAGNOSTICS/DIAGNOSTIC_MEANS中。', '',
               '## 8. 重复现象、历史反例与证据强度', '',
               '| 判断 | 证据等级 | 支持与边界 |', '|---|---|---|',
               '| 当前主筛选不通过 | STRONG | 两套selected EMA均值非正、KR保护线失败；不等于总体无效的显著性证明。 |',
               '| U早期AUC落后 | STRONG | E1两数据均5/5、描述区间全负；后期与selected不能混用。 |',
               '| 移除类内政策损失校准收益 | STRONG（KR），MEDIUM（ML幅度） | U−W task1 LL两数据均5/5变差，KR Macro/task1区间全正；ML区间跨零。 |',
               '| cell/task质量单独复现W−S校准收益 | 不支持，MEDIUM | U−S task1方向跨数据不一致；W−S原有5/5、区间全负结果仍在。 |',
               '| U末期AUC改善可替代selected | 不支持，WEAK | E50均值正、区间跨零，不符合预设主终点。 |',
               '| 小domain规模是主要原因 | WEAK | ML最小域baseline最好；KR两个小域方向相反，规模/类别率/任务/难度混杂。 |',
               '| 普遍小域改善/大域下降 | WEAK | 仅部分KR点估计符合，ML不符合；C04也有小域反向反例。 |', '',
               '历史反例保持原结论：C04 W−C 为 ML−0.000154621、KR+0.002555086（均未获主资格），KR d2收益与d0损害同时存在；C03 EARLY 的ML selected AUC−0.000340045、五seed负，KR点均值−0.000068613且区间跨零；LATE在ML选中epoch位于干预前，精确等于CLEAN不能称为收益。C02 NQR两套selected均值负，但其相对BRIER的LL多数seed下降；KR最大d1的NQR LL改善−0.006196973、五seed下降，反驳“大域必然受损”。C01 AXIS和原TV没有形成稳定双数据资格；TV ML gamma.20微小正信号区间跨零。修正质量账本后的自然NDP仍未救回主要结果，不能把后续失败全部解释成旧质量bug。上述历史只作反例和背景，不增加本轮配对n。', '',
               '竞争解释包括类内政策对校准的影响、cell/task相对质量、Adam状态与耦合正则的尺度敏感性、EMA/选点和种子轨迹分叉。当前数据最清楚地区分了校准和排序，尚不能唯一识别哪个优化子机制主导；KR seed2024 的分叉需要新的独立检验，不能事后排除。', '']
    cost = json.loads((OUT / 'COST_ACCOUNTING.json').read_text())
    report += ['## 9. 预算、实际成本与运行偏差', '',
               '有用正式预算为500 epochs / 17250次真实forward、backward、Adam、EMA，701574750训练行、1403149500任务项；validation共5000 chunks / 175394000行（双stream）。实际返回正式调用17333次，比有用预算多83次，额外3399680行源于暂停后的未确认后缀重放；它们只记成本，不增加样本量。正式诊断实际40/120/160，未增加诊断疗效预算。', '',
               '工程规定32调用/8诊断/24sweep/32虚拟Adam，实际33/9/27/36，包含一个额外未确认FULL。全期按PID去重有25个worker进程；父状态观察最大10个同时RUNNING，两个GPU各观察到5个worker重叠，实际GPU驻留另由allocation ACK与原始遥测佐证。所有任务在本会话监督；没有重启退役driver。', '',
               f'已测worker seconds合计{cost["measured_worker_seconds"]:.6f}，有{cost["unknown_worker_duration_count"]}个缺失terminal的进程时长未知；supervisor process seconds已测{cost["measured_supervisor_seconds_overlapping"]:.6f}，与前者重叠，不能相加为总用时。COST_COMPONENTS.csv按组件计数，组件也不能跨scope相加。历史PID2982386的未返回工作与时长保留null；即使handoff对有terminal的pending-call求和为0，也不代表所有历史未知工作为0。', '',
               '本轮经历参数字典首次物化竞争、不完整profile分类、可追加journal被整文件绑定，以及父进度文件非原子更新。原始失败、来源身份、额外工作和恢复凭据均保留。最后一次仅修改父写入为temp+os.replace；在同owner、同55文件源码下从精确最后ACK继续，未重跑已经完成的ML轨迹。完整过程见EXECUTION_HISTORY.md。', '',
               '## 10. 可复核交付与停止', '',
               '冻结科学源码55文件SHA256：f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d。TABLE_RECEIPT SHA713245d77a304e9c9c62462de498236c063d13088e4eac8abaebf7dc07857548；RUNNER_HANDOFF SHAa3abe972f7e3fabff6f7489e2bd736fccf82289d3b5549fcffad342a3c6e0453。父验收直接核验20模型/500ACK/17250有用更新/95508960字节，所有LL绑定同AUC选点。', '',
               '原始表含400终点、2400终点cell、4000曲线、24000曲线cell、28900逐seed配对、5780配对汇总、103500权重行和500阶段计数。archive/index/parts及当前RAM可用性见DATA_AVAILABILITY.md；来源、原始失败和数值控制一起保留。C04模型当前不可用的披露保持不变，本轮只复用其数值，不能从C05的RAM可用性推出C04模型恢复。', '',
               '最佳后续鉴别实验是预注册、同domain内独立操纵类别比例和样本量，并固定预算、配对seed和难度分层；它能区分规模、类别政策与优化历史。还应以新的独立seed检验KR轨迹分叉，不能仅重跑或删除2024。此为未来研究建议，本次不执行，不追加lambda搜索或确认性循环。', '',
               '最终处置：保留NO_CONTINUATION_SIGNAL及全部相反证据；完成报告、原始归档、独立最终证据复审和实际发布后按用户指令停止，不启动C06，不自动开启Test或选champion。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')


if __name__ == '__main__':
    main()
