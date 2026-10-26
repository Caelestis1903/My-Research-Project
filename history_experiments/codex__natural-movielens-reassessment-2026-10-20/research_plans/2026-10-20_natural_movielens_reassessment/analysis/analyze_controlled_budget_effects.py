"""Paired natural S5/S6 budget effects and control-by-budget interactions."""
from pathlib import Path
import json
import pandas as pd
from analyze_core import interval

ROOT=Path(__file__).resolve().parent

def analyze():
    for stage in ('AC-S5','AC-S6'):
        out=ROOT/'core'/stage
        if not (out/'seed_macro_metrics.csv').exists():continue
        f=pd.read_csv(out/'seed_macro_metrics.csv')
        if set(f.budget)!={'full50','matched250'}:continue
        provenance=pd.read_csv(out/'source_index.csv');provenance['seed']=provenance.run_id.str.rsplit('__',n=1).str[-1].astype(int)
        identity=provenance.groupby('seed').agg(init_hashes=('init_sha256','nunique'),runs=('run_id','size')).reset_index()
        assert identity.init_hashes.eq(1).all()
        identity.to_csv(out/'cross_budget_initialization_identity.csv',index=False)
        effect=pd.read_csv(out/'population_budget_seed_deltas.csv')
        rows=[]
        for key,g in effect.groupby(['arm','checkpoint']):
            assert set(g.seed)=={2022,2023,2024,2025}
            for metric in ('macro_auc','macro_logloss','train_p_cell_auc','eval_p_cell_auc'):
                rows.append(dict(arm=key[0],checkpoint=key[1],metric=metric,
                    comparison='full50 minus matched250; same population and initial model, main/arch ratio changes',**interval(g[metric])))
        pd.DataFrame(rows).to_csv(out/'paired_budget_effect_summary.csv',index=False)
        p=pd.read_csv(out/'paired_seed_deltas.csv')
        a=p[p.budget=='full50'].set_index(['seed','checkpoint','contrast'])
        b=p[p.budget=='matched250'].set_index(['seed','checkpoint','contrast']);assert set(a.index)==set(b.index)
        rows=[]
        for ix in a.index:
            rows.append(dict(seed=ix[0],checkpoint=ix[1],contrast=ix[2],
                **{metric:a.loc[ix,metric]-b.loc[ix,metric] for metric in ('macro_auc','macro_logloss','train_p_cell_auc','eval_p_cell_auc')}))
        q=pd.DataFrame(rows);q.to_csv(out/'control_by_budget_interaction_seed.csv',index=False)
        summary=[]
        for key,g in q.groupby(['checkpoint','contrast']):
            for metric in ('macro_auc','macro_logloss','train_p_cell_auc','eval_p_cell_auc'):
                summary.append(dict(checkpoint=key[0],contrast=key[1],metric=metric,
                    interaction='control contrast at full50 minus same contrast at matched250',**interval(g[metric])))
        pd.DataFrame(summary).to_csv(out/'control_by_budget_interaction_summary.csv',index=False)
        print(stage,'budget pairs',len(effect),'interaction rows',len(q))

if __name__=='__main__':analyze()
