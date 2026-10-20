"""Primary descriptive N3 cell analysis; frozen paired draws and optimizer seeds."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze_core import interval
P=Path(__file__).resolve().parent/'n17_mechanisms'
f=pd.read_csv(P/'N3_all_cell_readouts.csv');f=f[(f.draw>=0)&(~f.direction_undefined.astype(bool))].copy()
keys=['seed','anchor_arm','epoch','bank','draw','split','domain','task']
assert not f.duplicated(keys+['case']).any()
ref=f[f.case=='native_SB'][keys+['auc','logloss']].rename(columns={'auc':'reference_auc','logloss':'reference_logloss'})
f=f.merge(ref,on=keys,validate='many_to_one')
for metric in ['auc','logloss']:f['delta_'+metric]=f[metric]-f['reference_'+metric]
seedkeys=['seed','anchor_arm','epoch','split','case','block','domain','task']
s=f.groupby(seedkeys,as_index=False)[['delta_auc','delta_logloss']].mean()
s.to_csv(P/'N3_seed_cell_deltas.csv',index=False)
rows=[]
for key,g in s.groupby(seedkeys[1:]):
 if set(g.seed)!={2022,2023,2024,2025}:continue
 for metric in ['delta_auc','delta_logloss']:
  rows.append(dict(zip(seedkeys[1:],key),metric=metric,**interval(g[metric])))
pd.DataFrame(rows).to_csv(P/'N3_four_seed_cell_summary.csv',index=False)
b=pd.read_csv(P/'N3_bank_MC_summary.csv')
b=b[(b.split=='diagnostic')&b.case.isin(['C_full','C_norm','C_direction'])&(b.metric=='auc_minus_SB')]
b.to_csv(P/'N3_final_precision.csv',index=False)
status=dict(cell_summaries=len(rows),undefined_directions=int(pd.read_csv(P/'N3_all_cell_readouts.csv').direction_undefined.sum()),
 precision_cases=len(b),above_threshold_at_cap=int((b.mcse>2.5e-5).sum()),maximum_final_mcse=float(b.mcse.max()),test_read=False,
 precision_rule='no extension beyond64/bank; remaining imprecision retained')
(P/'cell_analysis_status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2))
