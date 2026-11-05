"""Insert the frozen experiment tables into the long-term metrics ledger."""
import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).parent
sources = [('B', 'attempt1'), ('A', 'attempt_A1'), ('C', 'attempt_C1')]
formal = {'M3oE_fixed1', 'HyperGate', 'AutoIFS'}
lines = ['### 11/05 AUG_HISTORY：定向EMA、路径分组EMA及历史漂移条件增强（五seed）', '',
    '**协议：** FULLTRAIN_NATURAL80_10_10 / validation六格macro AUC最早严格最大值；MovieLens/K uaiRand自然人口、train-only预处理，候选模型均为M3oE fixed1，50epoch、batch40960、FP32、普通coupled Adam；seed2022–2026、n=5。以下各表全部为test，单元格为AUC / 未缩放LogLoss；Traffic为人口加权cell指标，Pooled为跨域池化任务指标。正式模型参照沿用11/04已冻结M3oE/HyperGate/AutoIFS端点及各自预算，本节候选均为内部研究方法。'.replace('K uaiRand','KuaiRand'), '',
    '**计数：** B/A各10条物理训练，分别500epoch/17250 Adam/8000 V；C为ML六配置×五seed共30条、1500epoch/30000 Adam/3000 V。总计50条物理训练、2500epoch、64500 Adam、19000 V；B/A相同seed的online/order hash逐epoch相同，是同一父路径的精确重放，不增加独立seed。一个seed中的影子、online/EMA、selected/E10/E50也不增加n。C中的F/父online沿用B冻结端点，未新增训练。', '',
    '[中文总报告](research_reports/2026-11-05_aug_history/REPORT.md) · [B分析](research_reports/2026-11-05_aug_history/B_STAGE_ANALYSIS.md) · [A分析](research_reports/2026-11-05_aug_history/A_STAGE_ANALYSIS.md) · [C分析](research_reports/2026-11-05_aug_history/C_STAGE_ANALYSIS.md)。全部有限HPO候选只在validation比较；下列test配置为冻结家族赢家、必要匹配控制及online，未进入test的搜索臂只见HPO/全V曲线。', '']
count = 0
for route, attempt in sources:
    path = ROOT / f'artifacts/aug_history_20261105/{attempt}/{route}/ENDPOINTS.csv'
    rows = list(csv.DictReader(path.open()))
    lines += [f'**{route} 原始数表：** [ENDPOINTS.csv](artifacts/aug_history_20261105/{attempt}/{route}/ENDPOINTS.csv)。', '']
    for dataset in sorted({r['dataset'] for r in rows}):
        candidates = sorted({r['candidate_id'] for r in rows if r['dataset']==dataset}-formal)
        if route == 'C':
            candidates = [c for c in candidates if c not in {'F','PARENT_online'}]
        for endpoint in ('selected', 'E10', 'E50'):
            lines += [f'**{route} / {dataset} / {endpoint}**', '',
                '| 研究配置/估计器 | n | d0t0 | d0t1 | d1t0 | d1t1 | d2t0 | d2t1 | Macro AUC / LL | Traffic AUC / LL | Pooled AUC / LL |',
                '|---|---:|---|---|---|---|---|---|---|---|---|']
            for candidate in candidates:
                group = [r for r in rows if r['dataset']==dataset and r['checkpoint']==endpoint
                         and r['candidate_id']==candidate and r['split']=='test']
                assert sorted(int(r['seed']) for r in group) == list(range(2022,2027))
                def pair(a, b):
                    return f'{st.mean(float(r[a]) for r in group):.6f} / {st.mean(float(r[b]) for r in group):.6f}'
                cells = [pair(f'd{i}t{j}_auc',f'd{i}t{j}_logloss') for i in range(3) for j in range(2)]
                cells += [pair('macro_auc','macro_logloss'),pair('sample_weighted_cell_auc','sample_weighted_cell_logloss'),pair('pooled_task_mean_auc','pooled_logloss')]
                lines.append('| '+candidate+' | 5 | '+' | '.join(cells)+' |')
                count += 1
            lines.append('')
lines += [f'本节共{count}条配置×数据集×端点记录；含B/A重放的F/online及U_FAST，不将重复展示计为新增科学轨迹。', '']
ledger = ROOT/'MDMT_ALGORITHM_METRICS_LEDGER.md'
current = ledger.read_text()
if lines[0] not in current:
    marker = '## 已回填性能矩阵\n\n'
    assert current.count(marker) == 1
    current = current.replace(marker, marker+'\n'.join(lines)+'\n', 1)
    current = current.replace('> 更新：2026-11-04。',
        '> 更新：2026-11-06。新增AUG_HISTORY的96条配置×数据集×端点记录（B/A同源重放不增加独立seed）；', 1)
    ledger.write_text(current)
print(f'Ledger contains {count} AUG_HISTORY configuration/endpoint rows')
