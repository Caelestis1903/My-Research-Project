# Hyperparameters and Statistics

Seeds2022–2026 only; 50epochs; batch40960 with tail; FP32; natural population/inherited80/10/10 and train-only encoding. Single coupled Adam lr=.01, betas(.9,.999), eps1e-8, WD ML1e-4/KR1e-5; foreach=False,fused=False,amsgrad=False. Fixed EMA rho ML=.9828205985452511/KR=.995, once after complete update.

Only gamma {.05,.10,.20} is searched, each complete five seeds/both datasets. View alpha=.03; no auxiliary mixture lambda in TV-GRM. INSTANT and SHUFFLE at .10 are mandatory.

Online/EMA independently select earliest strict validation six-cell Macro AUC maximum. Report selected and E10/E30/E50, all six AUC/unscaled LL, Macro/Traffic/Pooled/domain/task/worst cell, epoch/hash, each seed and paired mean/sample SD/95%CI=mean±2.776445*SD/sqrt(5), positive count, actual runtime/memory. Seed is the statistical unit.

Report component vs CLEAN EMA, total vs CLEAN online, direct NDP-A at corresponding readout, .10 vs SHUFFLE and INSTANT. AUC primary, LL separately. No added CI exclusion/all-seed-positive gate.

Test sealed in ACTIVE/PAUSED and all autonomous cycles. Only USER_STOPPED + final freeze binding candidate/comparator/source/data/config/selection + eligible live selected RAM for both datasets/five seeds permits final Test. No eligible candidate => NOT_TRIGGERED. Publication gate: paired mean Macro AUC>=.003 vs strongest applicable formally published model; every cell mean>=-.001. Mean LL<0 judged separately. CLEAN EMA and NDP-A are internal controls, not formal published model baselines. Final Test does not feed HPO or restart exploration.
