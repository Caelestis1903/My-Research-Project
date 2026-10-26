"""Same-model frozen module neutralization; separate from retrained ablations."""
from pathlib import Path
import gzip
import json
import numpy as np
import pandas as pd
from analyze_core import interval

ROOT=Path(__file__).resolve().parent/'transfer'
OUT=Path(__file__).resolve().parent/'neutral_interventions'

def analyze():
    f=pd.read_csv(ROOT/'all_cell_metrics.csv');f=f[f.study=='CM19']
    configs=pd.read_csv(ROOT/'run_config_index.csv').set_index('run_id')
    cells=[];seeds=[]
    for (run_id,checkpoint),g in f.groupby(['run_id','checkpoint']):
        native=g[g.intervention=='none'].set_index(['domain','task'])
        for mode,gg in g[g.intervention!='none'].groupby('intervention'):
            for row in gg.itertuples():
                ref=native.loc[(row.domain,row.task)]
                cells.append(dict(run_id=run_id,checkpoint=checkpoint,intervention=mode,seed=row.seed,
                    domain=row.domain,task=row.task,auc=row.auc-ref.auc,logloss=row.logloss-ref.logloss,
                    baseline_auc=ref.auc,baseline_logloss=ref.logloss,delta_p=row.delta_p,
                    nonzero=int(row.nonzero),n=int(row.n),max_delta_p=row.max_delta))
            for task in (-1,*sorted(gg.task.unique())):
                selected=gg if task==-1 else gg[gg.task==task]
                ref=native.loc[list(zip(selected.domain,selected.task))]
                meta=configs.loc[run_id].to_dict()
                seeds.append(dict(meta,run_id=run_id,checkpoint=checkpoint,intervention=mode,task=task,
                    delta_auc=float(selected.auc.mean()-ref.auc.mean()),
                    delta_logloss=float(selected.logloss.mean()-ref.logloss.mean()),
                    baseline_auc=ref.auc.mean(),baseline_logloss=ref.logloss.mean(),
                    exactly_equal_on_full_test=bool(selected.nonzero.eq(0).all()),
                    nonzero=int(selected.nonzero.sum()),n_row_tasks=int(selected.n.sum()),
                    weighted_delta_p=np.average(selected.delta_p,weights=selected.n),max_delta_p=selected.max_delta.max()))
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(cells).to_csv(OUT/'seed_cell_intervention_deltas.csv',index=False)
    seeds=pd.DataFrame(seeds);seeds.to_csv(OUT/'seed_task_intervention_deltas.csv',index=False)
    recipe=['population','scope','dataset','optimizer','method','suite','variant','decay_mode','schedule','fraction',
        'backbone','noise_interaction','stage_group','budget','lr','checkpoint','intervention','task']
    seeds[recipe]=seeds[recipe].fillna('UNSPECIFIED')
    summary=[];incomplete=[]
    for key,g in seeds.groupby(recipe):
        meta=dict(zip(recipe,key))
        if set(g.seed)!={2022,2023,2024,2025} or len(g)!=4:
            incomplete.append(dict(meta,available_seed_count=len(g)));continue
        for metric in ('delta_auc','delta_logloss','weighted_delta_p','max_delta_p'):
            summary.append(dict(meta,metric=metric,reference_auc=g.baseline_auc.mean(),reference_logloss=g.baseline_logloss.mean(),
                exact_identity_seeds=int(g.exactly_equal_on_full_test.sum()),**interval(g[metric])))
    pd.DataFrame(summary).to_csv(OUT/'four_seed_summary.csv',index=False)
    (OUT/'scope.json').write_text(json.dumps(dict(complete_recipe_summaries=len(summary),incomplete=incomplete,
        intervention='implemented frozen-checkpoint neutral action; actual task subset retained',
        not_a_retrained_ablation=True,cross_model_intervention_scales_not_equivalent=True,
        historical_heavy_tensors_reevaluated=False),indent=2)+'\n')
    for path in OUT.glob('*.csv'):
        if path.stat().st_size>5_000_000:
            path.with_suffix(path.suffix+'.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    print('Frozen neutral interventions:',len(cells),'paired cells;',len(summary),'four-seed metric summaries')

if __name__=='__main__':analyze()
