"""Per-cell matched-task contrasts; actual support size roles and trade-offs."""
from pathlib import Path
import gzip
import json
import numpy as np
import pandas as pd
from analyze_core import interval

OUT=Path(__file__).resolve().parent/'transfer_cells'
INPUT=Path(__file__).resolve().parent/'transfer'

def analyze():
    allcells=pd.read_csv(INPUT/'all_cell_metrics.csv');allcells=allcells[allcells.intervention=='none']
    index=pd.read_csv(INPUT/'run_config_index.csv').set_index('run_id')
    macro=pd.read_csv(INPUT/'paired_seed_deltas.csv');macro=macro[macro.task==-1]
    cache={};metrics={}
    def counts(run_id):
        if run_id not in cache:
            c=json.loads(Path(index.loc[run_id,'config_path']).read_text())
            if 'main_domain_counts' in c:cache[run_id]=np.asarray(c['main_domain_counts'],dtype=float)
            else:cache[run_id]=np.asarray(c['p'],dtype=float)*c['train_rows']
            assert np.allclose(cache[run_id],np.rint(cache[run_id]),atol=1e-6,rtol=0)
            cache[run_id]=np.rint(cache[run_id]).astype(np.int64)
        return cache[run_id]
    def cells(run_id,checkpoint):
        key=(run_id,checkpoint)
        if key not in metrics:
            metrics[key]=allcells[(allcells.run_id==run_id)&(allcells.checkpoint==checkpoint)].set_index(['domain','task'])
        return metrics[key]
    rows=[];roles=[];meta=[]
    for record in macro.itertuples():
        a=cells(record.candidate_run_id,record.checkpoint);b=cells(record.reference_run_id,record.checkpoint)
        # A single-task candidate is compared only with that task in a joint
        # reference; never with its unrelated second task or joint macro.
        assert set(a.index)<=set(b.index)
        b=b.loc[a.index]
        assert abs((a.auc-b.auc).mean()-record.macro_auc)<1e-12
        assert abs((a.logloss-b.logloss).mean()-record.macro_logloss)<1e-12
        candidate_counts=counts(record.candidate_run_id);reference_counts=counts(record.reference_run_id)
        candidate_order=np.argsort(candidate_counts,kind='stable');reference_order=np.argsort(reference_counts,kind='stable')
        delta_by_domain=(a.auc-b.auc).groupby(level='domain').mean()
        base=dict(comparison=record.comparison,seed=record.seed,checkpoint=record.checkpoint,
            candidate_run_id=record.candidate_run_id,reference_run_id=record.reference_run_id)
        for (d,t),row in a.iterrows():
            ref=b.loc[(d,t)]
            rows.append(dict(base,domain=d,task=t,auc=row.auc-ref.auc,logloss=row.logloss-ref.logloss,
                baseline_auc=ref.auc,baseline_logloss=ref.logloss,
                candidate_unique_main_rows=candidate_counts[d],reference_unique_main_rows=reference_counts[d],
                candidate_size_rank=int(np.flatnonzero(candidate_order==d)[0])+1,
                reference_size_rank=int(np.flatnonzero(reference_order==d)[0])+1,
                test_rows=int(row.n),test_positives=int(row.positives)))
        roles.append(dict(base,candidate_smallest=int(candidate_order[0]),candidate_largest=int(candidate_order[-1]),
            reference_smallest=int(reference_order[0]),reference_largest=int(reference_order[-1]),
            same_size_order=bool(np.array_equal(candidate_order,reference_order)),
            reference_small_gain_large_decline=bool(delta_by_domain[reference_order[0]]>0 and delta_by_domain[reference_order[-1]]<0),
            reference_both_smaller_gain_largest_decline=bool(all(delta_by_domain[d]>0 for d in reference_order[:2]) and delta_by_domain[reference_order[-1]]<0),
            candidate_small_gain_large_decline=bool(delta_by_domain[candidate_order[0]]>0 and delta_by_domain[candidate_order[-1]]<0),
            candidate_both_smaller_gain_largest_decline=bool(all(delta_by_domain[d]>0 for d in candidate_order[:2]) and delta_by_domain[candidate_order[-1]]<0),
            interpretation='rank actual unique main support, not replay exposure or domain difficulty; reference ordering primary when intervention changes roles'))
        c=index.loc[record.candidate_run_id];r=index.loc[record.reference_run_id]
        meta.append(dict(comparison=record.comparison,candidate_population=c.population,reference_population=r.population,
            candidate_scope=c.scope,reference_scope=r.scope,dataset=c.dataset,method=c.method,model=c.model,
            optimizer=c.optimizer,stage_group=c.stage_group,variant=c.variant,decay_mode=c.decay_mode,
            schedule=c.schedule,fraction=c.fraction))
    OUT.mkdir(exist_ok=True)
    cell=pd.DataFrame(rows);cell.to_csv(OUT/'paired_seed_cell_deltas.csv',index=False)
    pd.DataFrame(roles).to_csv(OUT/'actual_size_roles_and_tradeoffs.csv',index=False)
    pd.DataFrame(meta).drop_duplicates().to_csv(OUT/'comparison_metadata.csv',index=False)
    summaries=[]
    for key,g in cell.groupby(['comparison','checkpoint','domain','task']):
        assert set(g.seed)=={2022,2023,2024,2025} and len(g)==4
        for metric in ('auc','logloss'):
            summaries.append(dict(zip(['comparison','checkpoint','domain','task'],key),metric=metric,
                reference_auc=g.baseline_auc.mean(),reference_logloss=g.baseline_logloss.mean(),**interval(g[metric])))
    pd.DataFrame(summaries).to_csv(OUT/'four_seed_cell_summary.csv',index=False)
    for level in ('domain','task'):
        table=cell.groupby(['comparison','seed','checkpoint',level])[['auc','logloss','baseline_auc','baseline_logloss']].mean().reset_index()
        table.to_csv(OUT/('paired_'+level+'_deltas.csv'),index=False)
    for path in OUT.glob('*.csv'):
        path.with_suffix(path.suffix+'.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    (OUT/'analysis_status.json').write_text(json.dumps(dict(complete_comparisons=macro.comparison.nunique(),
        paired_cell_rows=len(cell),same_task_macro_consistency_recomputed=True,
        size_defined_by='actual unique main support, candidate/reference roles both retained',
        domain_size_not_difficulty=True,intervals='pointwise descriptive four-seed t95, no familywise/population confirmation'),indent=2)+'\n')
    print('Transfer cell contrasts',len(cell),'rows;',macro.comparison.nunique(),'complete comparisons')

if __name__=='__main__':analyze()
