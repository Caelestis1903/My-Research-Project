"""Preserve AO04 historical selected contrasts and compare only available seed sets."""
from pathlib import Path
import hashlib
import json
import pandas as pd

OLD=Path('/home/xzf/Code/phase0-3-git/research_plans/2026-10-15/execution_20261015')
OUT=Path(__file__).resolve().parent/'ao04_history'
NEW=Path(__file__).resolve().parent/'ao04'

def analyze():
    OUT.mkdir(exist_ok=True)
    names=['ao04_selected_paired_l2.csv','ao04_test_aggregates_recalc.csv',
           'ao04_rho_log_diagnostics.csv','ao04_validation_identity.json','ao04_pairing_claims.json']
    sources=[]
    for name in names:
        source=OLD/name;target=OUT/('historical_'+name);target.write_bytes(source.read_bytes())
        sources.append(dict(source=str(source),copy=str(target),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            evidence='historical recorded evidence, no old checkpoint reload claimed'))
    f=pd.read_csv(OLD/'ao04_selected_paired_l2.csv');rows=[]
    for (dataset,split,contrast),g in f.groupby(['dataset','split','contrast']):
        assert not g.seed.duplicated().any()
        for column in g.columns:
            if column.startswith('delta_'):
                v=g[column]
                rows.append(dict(population='historical_artificial_d0_large' if dataset=='movielens' else 'historical_natural_kuairand',
                    checkpoint='selected',split=split,contrast=contrast,metric=column[6:],n_seeds=len(g),
                    seeds=','.join(map(str,sorted(g.seed))),mean=v.mean(),positive=int((v>0).sum()),negative=int((v<0).sum()),
                    interpretation='old ML only 2 seeds; KR only 1 complete B1 seed; no four-seed inference'))
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'historical_selected_summary.csv',index=False)
    if (NEW/'analysis_status.json').is_file():
        assert json.loads((NEW/'analysis_status.json').read_text())['complete']
        n=pd.read_csv(NEW/'paired_summary.csv');n=n[(n.checkpoint=='selected')&(n.task==-1)]
        n.to_csv(OUT/'new_natural_selected_summary.csv',index=False)
        new_available=True
    else:new_available=False
    (OUT/'source_index.json').write_text(json.dumps(sources,indent=2)+'\n')
    (OUT/'status.json').write_text(json.dumps(dict(historical_complete=True,new_natural_available=new_available,
        no_new_bootstrap_or_training=True,old_ML_population='d0_large, not d2_large',
        row_identity_limit='historical validation NPZ lack row IDs; identity incomplete'),indent=2)+'\n')
    print(summary[summary.metric.isin(['macro_auc','macro_logloss'])].to_string(index=False))

if __name__=='__main__':analyze()
