from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/home/xzf/Code/phase0-3-20261016-v3-git/research_plans/2026-10-16_v3/execution_20261016');D=O/'analysis/V11_shadow_statistics';S=pd.DataFrame(json.loads((D/'contrast_summary.json').read_text()));F=O/'figures';F.mkdir(exist_ok=True)
conditions=['native','swap_m','swap_v','full_shadow','beta1_zero_v_only','beta2_zero','m_only'];colors=plt.get_cmap('tab10').colors
fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
for row,norm in enumerate(['raw','match_native_global_norm']):
 for col,epoch in enumerate([1,10,40]):
  ax=axes[row,col]
  for i,c in enumerate(conditions):
   x=S[(S.epoch==epoch)&(S.split=='diagnostic')&(S.variable=='macro_auc')&(S.condition==c)&(S.norm_control==norm)&(S.contrast=='current_q_gap')].iloc[0]
   ax.errorbar(i,x['mean'],yerr=[[x['mean']-x.ci_low],[x.ci_high-x['mean']]],fmt='o',color=colors[i],capsize=3)
  ax.axhline(0,color='gray',lw=.7);ax.set_title(f'Epoch {epoch}, '+('raw' if row==0 else 'global norm matched'));ax.set_xticks(range(7),conditions,rotation=55,ha='right');ax.grid(axis='y',alpha=.2);ax.set_ylabel('One-step diagnostic AUC: SB - SN')
fig.suptitle('V11 local q gaps at common native SN theta (four training seeds; pointwise 95% seed bootstrap)');fig.tight_layout();fig.savefig(F/'V11_shadow_local_q_gaps.png',dpi=180);fig.savefig(F/'V11_shadow_local_q_gaps.pdf');plt.close(fig)
S[S.variable.isin(['macro_auc','macro_logloss','train_empirical_objective'])].to_csv(D/'macro_and_train_contrast_summary.csv',index=False)
S[(S.contrast=='history_by_current_q_interaction')&S.variable.str.match(r'd[0-2]t[0-1]_(auc|logloss)$|d[0-2]_(auc|logloss)$|t[0-1]_(auc|logloss)$')].to_csv(D/'domain_task_history_interactions.csv',index=False)
# Separate derived artifact receipt; preserve completed statistics receipt and hashes.
import hashlib,time
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(O/'V11_shadow_statistics_presentation_receipt.json').write_text(json.dumps(dict(status='COMPLETE',time=time.time(),independent_review=False,statistics_receipt_sha=sha(D/'receipt.json'),primary_verification_sha=sha(O/'V11_shadow_statistics_actual_primary_verification.json'),hashes={str(p.relative_to(O)):sha(p) for p in [D/'macro_and_train_contrast_summary.csv',D/'domain_task_history_interactions.csv',F/'V11_shadow_local_q_gaps.png',F/'V11_shadow_local_q_gaps.pdf']},test_access=False),indent=2)+'\n')
print('actual local statistics plot and compact tables complete')
