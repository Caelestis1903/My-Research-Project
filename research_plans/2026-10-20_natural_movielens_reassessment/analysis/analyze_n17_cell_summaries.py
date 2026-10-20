"""Primary summaries of already completed N17 frozen test cells and interactions."""
from pathlib import Path
import json
import pandas as pd
from analyze_core import interval
P=Path(__file__).resolve().parent/'n17'
f=pd.read_csv(P/'paired_cell_deltas.csv');rows=[]
for key,g in f.groupby(['comparison','readout','domain','task']):
 assert set(g.seed)=={2022,2023,2024,2025}
 for metric in ['auc','logloss']:rows.append(dict(zip(['comparison','readout','domain','task'],key),metric=metric,reference_mean=float(g.baseline_auc.mean()) if metric=='auc' else None,**interval(g[metric])))
pd.DataFrame(rows).to_csv(P/'four_seed_cell_summary.csv',index=False)
rows=[]
for axis in ['domain','task']:
 k=['comparison','seed','readout',axis];m=f.groupby(k,as_index=False)[['auc','logloss','baseline_auc']].mean()
 for key,g in m.groupby(['comparison','readout',axis]):
  assert set(g.seed)=={2022,2023,2024,2025}
  for metric in ['auc','logloss']:rows.append(dict(zip(['comparison','readout',axis],key),axis=axis,metric=metric,baseline_auc=float(g.baseline_auc.mean()),**interval(g[metric])))
pd.DataFrame(rows).to_csv(P/'four_seed_domain_task_summary.csv',index=False)
q=pd.read_csv(P/'quota_by_q_interaction_seed.csv');rows=[]
metrics=['auc','logloss','train_p_cell_auc','eval_p_cell_auc','train_p_cell_logloss','eval_p_cell_logloss']
for key,g in q.groupby(['readout','task']):
 assert set(g.seed)=={2022,2023,2024,2025}
 for metric in metrics:rows.append(dict(zip(['readout','task'],key),metric=metric,**interval(g[metric])))
pd.DataFrame(rows).to_csv(P/'quota_by_q_interaction_summary.csv',index=False)
h=f[(f.readout=='final')&f.comparison.isin(['H200-minus-H100_SN','H200-minus-H100_SB'])]
status=dict(H200_minus_H100_final_cells=len(h),AUC_negative=int((h.auc<0).sum()),LL_positive=int((h.logloss>0).sum()),LL_min=float(h.logloss.min()),LL_max=float(h.logloss.max()),
 statement='any nonnegative normalized cell weights preserve worse risk per configuration iff all relevant cell LL deltas are positive')
(P/'H200_test_continuation_cell_status.json').write_text(json.dumps(status,indent=2)+'\n');print(status)

perconfig=[]
for key,g in h.groupby(['comparison','seed']):
 perconfig.append(dict(comparison=key[0],seed=int(key[1]),all_cell_LL_worse=bool((g.logloss>0).all())))
status['individual_configuration_weight_dominance']=perconfig
status['four_seed_mean_LL_positive_cells']=int((h.groupby(['comparison','domain','task']).logloss.mean()>0).sum())
(P/'H200_test_continuation_cell_status.json').write_text(json.dumps(status,indent=2)+'\n')
