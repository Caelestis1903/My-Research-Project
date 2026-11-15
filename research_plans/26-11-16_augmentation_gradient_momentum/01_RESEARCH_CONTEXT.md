# Research Context

## Current question
Can task-conditioned augmented views provide useful gradient information when the view-minus-clean gradient is routed through a separate bounded first-moment channel, while Adam's second moment remains anchored to the clean objective?

## Accepted history
- Fixed parameter EMA has stable AUC gains in tested M3oE fixed1 settings; shared parameters, especially EMBED, carry most measured benefit. Do not claim private-parameter EMA advantage.
- Tested adaptive EMA, parameter grouping, gradient drift/gain prediction, PUB migration and A_MATCH feedback did not provide stable incremental value; A_MATCH feedback harmed F in its tested six-round horizon.
- PAIR, PCGrad, ECCM, GATS and gradient balancing already cover related sample/gradient processing directions. Do not relabel them.
- DT-PGNA/TGA-Pair/TGA-Example at alpha=.03, K=128, lambda=.25 failed the validation continuation gate.
- Stage1 auxiliary-objective attribution found support, label ratio and loss normalization materially changed effective objective mass.
- Stage2 NDP-A used natural row-task weights, all rows and task-specific views in a convex clean/view objective. Selected validation fixed-EMA Macro AUC delta vs CLEAN: ML -0.000101256 (1/5 positive); KR -0.005775319 (0/5 positive). HPO/Test not triggered.
- Stage2 zero-delta counterpart passed exact model/Adam/EMA state-hash equality with CLEAN on both datasets. Reuse only after exact identity audit.

## Boundary
**Small domain is not difficult domain.** Do not propose any test, analysis, proxy or interpretation to prove/disprove it. Domain counts may appear only as descriptive protocol metadata.

## Why TV-GRM is distinct
NDP-A used one ordinary Adam gradient for a convex clean/view loss, placing view gradient in both Adam moments. TV-GRM stores (g_v-g_c) separately and preconditions it with clean (v_t). This changes the optimizer recurrence, not merely loss weight or EMA readout.

Every design must identify nonexistent assumptions, distinguish them from unverified hypotheses, compare simpler/stronger alternatives, and define a stopping result. Task labels alone do not establish MDMT-specific novelty.
