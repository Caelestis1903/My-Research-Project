# Research Context

Can task-conditioned augmented views provide useful information beyond clean fixed EMA when their gradient residual uses a separate bounded first-moment channel?

Accepted history: fixed EMA's measured benefit is mainly carried by shared parameters, especially EMBED. Tested adaptive EMA/PUB/A_MATCH did not provide stable incremental value; A_MATCH's single-step signal did not survive the tested six-round feedback horizon. PAIR/PCGrad/ECCM/GATS and stopped DT-PGNA/TGA formulations are already covered. No renamed repetition.

NDP-A is complete: natural all-2B row-task support, objective .75Lc+.25Lv, alpha=.03, ordinary combined-gradient Adam. ML selected validation fixed-EMA Macro AUC delta vs CLEAN=-0.00010125623089003443 (1/5 positive); KR=-0.005775319390310641 (0/5). Both CIs cross zero. ML online +0.000359628 did not add to fixed EMA. KR d2 gains coexist with larger d0 losses; AUC and loss diverge. Natural-weight correction was executed. Read the FULL report and all evidence pointers in 02 before any fresh Astra; do not attribute failure solely to old auxiliary mass.

TV-GRM changes both overall routing and residual history. Direct historical NDP-A comparisons test overall routing, not an isolated causal effect of clean v. INSTANT-.10 removes residual history while keeping clean Adam; SHUFFLE-.10 removes structured direction at matched actual residual-update norm. These are predeclared controls.

Current context starts at source main dda38ec8212d6b0c239b8db5ccf0e6ec8f207f11. Canonical current question path: research_context/experiments/Experimental_Open_Questions.md; read active experiments/Experimental_* too. Source snapshot experiments/Experimental_Open_Questions.md is the pinned provenance path. Original control snapshots are mappings, not a replacement for current main.

Small domain is not difficult domain. Population counts describe protocol; difficulty needs direct evidence and is not inferred from size. Keep AUC, loss, cell/task, seed, readout and training stage separate.
