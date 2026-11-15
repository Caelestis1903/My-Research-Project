# Hyperparameters and Statistics

Fixed: seeds 2022–2026 only; 50 epochs; batch 40960; FP32; natural sampling; inherited split/preprocessing; Adam lr=.01, betas(.9,.999), eps=1e-8, coupled WD ML=1e-4/KR=1e-5; EMA rho ML=.9828205985452511/KR=.995, updated once after parameter update.

Only new scalar is gamma; {.05,.10,.20} all run with five seeds on both datasets. Alpha=.03 inherited from NDP-A. No auxiliary BCE mass/lambda_aug.

Report six-cell AUC/LL, Macro, Traffic, Pooled, domain/task, worst cell, selected epoch and E10/E30/E50, each seed, paired mean/sample SD/95% t CI/positive count, runtime/memory. Online and EMA independently select earliest strict validation six-cell Macro AUC maximum.

AUC primary; LL separate. Report total candidate+EMA vs CLEAN-online and component increment vs CLEAN-fixed-EMA. Formal Test gate: Macro AUC gain >=.003 vs strongest applicable published model and every cell mean delta >=-.001. Not all seeds must be positive. Reused 2022–2026 is the same seed pool, not independent confirmation.
