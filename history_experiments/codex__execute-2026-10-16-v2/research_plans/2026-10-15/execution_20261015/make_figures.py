from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
d=pd.read_csv(p/'ao04_paired_bootstrap.csv');x=np.arange(len(d));ax[0].errorbar(x,d.delta_auc,yerr=np.vstack([d.delta_auc-d.ci95_low,d.ci95_high-d.delta_auc]),fmt='o',capsize=3)
ax[0].set_xticks(x,[f'd{r.domain}t{r.task}' for r in d.itertuples()]);ax[0].set_title('AO04 KR B1-B0, seed 2022\nPaired cell/label bootstrap 95% CI');ax[0].set_ylabel('Selected test AUC difference')
s=pd.read_csv(p/'s6r2_paired_recomputed.csv');s=s[(s.dataset=='kuairand')&(s.rule=='R0')&(s.metric=='N')]
for split,marker in [('validation','o'),('test','s')]:
 v=s[s.split==split].sort_values('seed');ax[1].plot(v.seed,v.delta,marker=marker,label=split)
ax[1].set_xticks([2022,2023,2024,2025]);ax[1].set_title('S6-R2 DH-F non-target selected AUC\nHistorical CSV recalculation (L2)');ax[1].set_ylabel('AUC difference');ax[1].legend()
for a in ax:a.axhline(0,color='gray',ls='--',lw=.8);a.grid(alpha=.2)
fig.savefig(p/'forensic_evidence.png',dpi=180);fig.savefig(p/'forensic_evidence.pdf')
