"""Export complete scalar evidence and figures after frozen test evaluation."""
import json
import gzip
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_tables import tables

REPORT = Path(__file__).resolve().parent
ATTEMPT = REPORT.parents[1]/'artifacts/conditional_information_momentum_20261102_v3/attempt1'


def md(frame):
    return '\n'.join(['| '+' | '.join(map(str, frame.columns))+' |', '| '+' | '.join(['---']*len(frame.columns))+' |'] + ['| '+' | '.join(map(str, row))+' |' for row in frame.itertuples(index=False, name=None)])


def main():
    frame, stats = tables(ATTEMPT, REPORT/'tables')
    assert len(frame) == 288 and (frame.role == 'test').sum() == 144
    for name in ('paired_seed.csv', 'paired_summary.csv'):
        path = REPORT/'tables'/name
        path.with_suffix('.csv.gz').write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    results = [json.loads(p.read_text()) for p in (ATTEMPT/'runs').glob('*/result.json')]
    costs = []
    for r in results:
        j = r['job']; c = j['config']; counts = r['counts']
        expected = 1000 if j['dataset'] == 'movielens' else 2450
        assert r['epochs'] == 50 and r['global_step'] == counts['optimizer_steps'] == counts['ema_updates'] == expected
        costs.append(dict(dataset=j['dataset'],arm=c['arm'],lambda_pair=c['lambda_pair'],seed=j['seed'],
            epochs=50,steps=expected,worker_seconds=r['worker_seconds'],peak_allocated_gib=r['peak_memory_bytes']/2**30,
            peak_reserved_gib=r['peak_reserved_bytes']/2**30,**counts))
    costs = pd.DataFrame(costs); costs.to_csv(REPORT/'tables/costs.csv',index=False)
    selected = frame[(frame.role=='test') & (frame.checkpoint=='selected') & (frame.estimator=='fixed_ema')].copy()
    selected['method'] = selected.apply(lambda r: r.arm if r.arm in ('A0','A1') else f'{r.arm} λ={r.lambda_pair:g}',axis=1)
    raw = selected[['dataset','method','seed','epoch','macro_auc','macro_logloss','worst_cell_auc','sample_weighted_cell_auc','pooled_task_mean_auc']].sort_values(['dataset','method','seed'])
    raw.to_csv(REPORT/'tables/selected_test_by_seed.csv',index=False)
    appendix = ['# 完整数值附录','', 'seed为统计单位。AUC/LogLoss均来自未扰动真实测试输入；选点只使用validation。','', '## EMA selected逐seed','',md(raw.round(7)), '', '## EMA selected六格均值与样本SD','']
    cells=[]
    for (dataset, method), group in selected.groupby(['dataset','method']):
        for d in range(3):
            for task in range(2):
                cells.append(dict(dataset=dataset,method=method,cell=f'd{d}t{task}',n=len(group),
                    auc_mean=group[f'd{d}t{task}_auc'].mean(),auc_sd=group[f'd{d}t{task}_auc'].std(),
                    logloss_mean=group[f'd{d}t{task}_logloss'].mean(),logloss_sd=group[f'd{d}t{task}_logloss'].std()))
    cells=pd.DataFrame(cells);cells.to_csv(REPORT/'tables/selected_test_cells.csv',index=False)
    appendix += [md(cells.round(7)), '', '完整online/EMA、selected/E10/E50、逐域逐任务及原始计数见tables/endpoints.csv；全部方法两两配对、逐seed差和95% t区间见tables/paired_seed.csv.gz与paired_summary.csv.gz。n=2的t区间自由度为1。', '', '## 运行成本', '', md(costs[['dataset','arm','lambda_pair','seed','steps','worker_seconds','peak_allocated_gib','peak_reserved_gib']].sort_values(['dataset','arm','lambda_pair','seed']).round(3))]
    (REPORT/'NUMERICAL_APPENDIX.md').write_text('\n'.join(appendix)+'\n')
    (REPORT/'figures').mkdir(exist_ok=True)
    pairs=pd.read_csv(REPORT/'tables/paired_seed.csv')
    p=pairs[(pairs.checkpoint=='selected')&(pairs.estimator=='fixed_ema')&(pairs.control=='A0')]
    methods=[('A1',0.),('P00',.05),('P00',.2),('P11',.05),('P11',.2)]
    labels=['A1','P00 .05','P00 .20','P11 .05','P11 .20']
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for i,dataset in enumerate(('movielens','kuairand')):
        for j,metric in enumerate(('macro_auc','macro_logloss')):
            ax=axes[i,j]
            for role,offset,color in [('validation',-.12,'tab:blue'),('test',.12,'tab:orange')]:
                for x,(arm,lam) in enumerate(methods):
                    v=p[(p.dataset==dataset)&(p.role==role)&(p.metric==metric)&(p.arm==arm)&(p.lambda_pair==lam)].difference.to_numpy()
                    assert len(v)==2
                    ax.scatter(np.full(2,x+offset),v,color=color,alpha=.6,s=25)
                    ax.scatter(x+offset,v.mean(),marker='_',s=250,color=color,label=role if x==0 else None)
            ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(5),labels,rotation=20)
            ax.set_title(dataset+' / '+metric+' difference vs A0');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(REPORT/'figures/selected_paired.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(trajectories=len(results),epochs=int(costs.epochs.sum()),steps=int(costs.steps.sum()),
        summed_worker_seconds=float(costs.worker_seconds.sum()),test_endpoints=144)))


if __name__=='__main__': main()
