"""Compare already completed Equal/Sample F/DH trajectories; no training or new evaluation."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from analyze_core import interval

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'controlled_objectives'
OUT.mkdir(exist_ok=True)
frames={};sources=[]
for stage in ('AC-S5','AC-S6'):
    path=ROOT/'core'/stage/'seed_macro_metrics.csv'
    frames[stage]=pd.read_csv(path).query('arm in ["F","DH"]')
    sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
manifest=json.loads(Path('/home_new/xzf_natural_movielens_reassessment/core_manifest.json').read_text())['runs']
configs={(c['stage'],c['budget'],c['arm'],c['seed']):c for c in manifest}
identity=[]
for budget in ('full50','matched250'):
    for arm in ('F','DH'):
        for seed in (2022,2023,2024,2025):
            dirs=[Path(configs[(stage,budget,arm,seed)]['run_dir']) for stage in ('AC-S5','AC-S6')]
            init=[json.loads((d/'config.json').read_text())['init_sha'] for d in dirs]
            assert init[0]==init[1]
            batch=[pd.read_csv(d/'batch_trace.csv') for d in dirs]
            columns=['epoch','batch','kind','n','d0','d1','d2','row_ids_sha256']
            assert batch[0][columns].equals(batch[1][columns])
            identity.append(dict(budget=budget,arm=arm,seed=seed,init_sha256=init[0],
                paired_batches=len(batch[0]),equal_batch_sha256=hashlib.sha256((dirs[0]/'batch_trace.csv').read_bytes()).hexdigest(),
                sample_batch_sha256=hashlib.sha256((dirs[1]/'batch_trace.csv').read_bytes()).hexdigest(),
                identity_scope='initial parameter identity and row/size/domain batch sequence; objectives and resulting states differ'))
pd.DataFrame(identity).to_csv(OUT/'cross_objective_identity.csv',index=False)
keys=['budget','arm','seed','checkpoint'];metrics=['macro_auc','macro_logloss','train_p_cell_auc','eval_p_cell_auc']
a=frames['AC-S5'].set_index(keys);b=frames['AC-S6'].set_index(keys);assert set(a.index)==set(b.index)
rows=[]
for ix in a.index:
    rows.append(dict(zip(keys,ix),**{m:a.loc[ix,m]-b.loc[ix,m] for m in metrics}))
effect=pd.DataFrame(rows);effect.to_csv(OUT/'equal_minus_sample_seed.csv',index=False)
summary=[]
for key,g in effect.groupby(['budget','arm','checkpoint']):
    assert set(g.seed)=={2022,2023,2024,2025}
    for m in metrics:
        summary.append(dict(budget=key[0],arm=key[1],checkpoint=key[2],metric=m,contrast='Equal minus Sample',**interval(g[m])))
pd.DataFrame(summary).to_csv(OUT/'equal_minus_sample_summary.csv',index=False)
a=effect[effect.arm=='DH'].set_index(['budget','seed','checkpoint']);b=effect[effect.arm=='F'].set_index(['budget','seed','checkpoint'])
inter=[]
for ix in a.index:
    inter.append(dict(budget=ix[0],seed=ix[1],checkpoint=ix[2],**{m:a.loc[ix,m]-b.loc[ix,m] for m in metrics}))
inter=pd.DataFrame(inter);inter.to_csv(OUT/'objective_by_control_interaction_seed.csv',index=False)
summary=[]
for key,g in inter.groupby(['budget','checkpoint']):
    for m in metrics:
        summary.append(dict(budget=key[0],checkpoint=key[1],metric=m,
            contrast='(DH-F) Equal minus (DH-F) Sample',**interval(g[m])))
summary=pd.DataFrame(summary);summary.to_csv(OUT/'objective_by_control_interaction_summary.csv',index=False)
(OUT/'source_files.json').write_text(json.dumps(sources,indent=2)+'\n')
print(summary[summary.checkpoint.isin(['selected','final'])&summary.metric.isin(['macro_auc','macro_logloss'])].round(7).to_string(index=False))
