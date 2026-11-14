# Literature and Novelty Review

## 1. Closest MDMT / recommendation work

1. **SAF — Stable and Adaptive Fusion for Multi-domain Multi-task Recommendation**, AAAI 2026, DOI 10.1609/aaai.v40i17.38491. SAF uses weighted HSIC to decorrelate irrelevant features, learns sample weights, and uses adaptive feature/expert gating; its sample weights also reweight MDMT losses. It is a very close current MDMT reference for stable feature learning and sample-level training weighting. DT-PGNA differs by generating an input-level adversarial view using target-logit sensitivity and constraining non-target logits, without a new gate/fusion architecture or learned sample-weight estimator. This difference must be substantiated, not asserted as a broad first.
2. **M3oE**, SIGIR 2024, DOI 10.1145/3626772.3657686. This is the fixed architecture platform, not the proposed contribution.
3. **MMFI — Multi-Scenario and Multi-Task Aware Feature Interaction**, ACM TKDD 2024, DOI 10.1145/3651312. It conditions feature-interaction importance on scenario-task pairs, but changes the model feature-interaction module. It is adjacent to the domain-task conditioning motivation, not the same training intervention.
4. **PMTRec — Personalized Multi-task Training for Recommender System**, arXiv:2407.21364 (2024 preprint). It uses personalized task weights and gradient magnitude/focusing modules. Its status and final publication venue must be verified before treating it as a formal published baseline.

## 2. Closest general MTL / feature perturbation work

1. **DTME-MTL — Resolving Token-Space Gradient Conflicts**, ICCV 2025, DOI 10.1109/ICCV51701.2025.00277. It detects gradient conflicts in token space and modifies token representations. This is the closest conceptual neighbor to feature-space gradient manipulation. DT-PGNA's proposed distinction is per-domain-task positive/negative ranking-pair hardening with explicit non-target-logit preservation and target-only BCE, not token conflict classification or token expansion.
2. **CROSSGRAD**, 2018, uses domain-guided cross-gradient input perturbation. It predates the current period and establishes that gradient-guided input augmentation itself is not new. The contribution cannot be gradient perturbation for generalization; it must be the MDMT domain-task interaction constraint and ranking-pair construction.
3. **SALARec**, Expert Systems with Applications 2026, DOI 10.1016/j.eswa.2025.130070, uses preference-aware adversarial augmentation in sequential recommendation. This confirms adversarial augmentation is already used in recommendation. It is not MDMT, but it reduces broad novelty claims.
4. **Task Weighting through Gradient Projection for Multitask Learning**, 2024, DOI 10.1007/978-981-96-6954-7_22, adapts PCGrad for conflict-dependent task prioritization. This is parameter-gradient/task-weighting work, not the proposed input perturbation.

## 3. Novelty boundary

Potential contribution:
- a pair-directed latent perturbation that explicitly preserves non-target task logits locally;
- domain-task-conditioned generation without changing the architecture;
- a controlled demonstration that output-preserving task-specific hard views improve ranking and are not equivalent to Gaussian or unprojected perturbation.

Not a contribution:
- generic gradient-guided augmentation;
- generic adversarial feature perturbation;
- pairwise ranking loss;
- label-conditioned pair selection alone;
- generic PCGrad or gradient balancing;
- merely using domain/task IDs to choose six independent hyperparameters.

## 4. Literature decision

The current search identifies relevant adjacent work but does not establish exhaustive novelty or open-source status. Before paper submission, verify full PDFs, code availability, and whether SAF or DTME contains a mathematically equivalent output-preserving projection. If an equivalent method exists, narrow the contribution to recommendation-specific pair construction and empirical mechanism, or stop claiming method-level novelty. Similar work is not an automatic reason to discard the route; actual mechanism overlap determines the decision.
