# Literature and Novelty Review

## 1. Closest MDMT / recommendation work

1. **SAF — Stable and Adaptive Fusion for Multi-domain Multi-task Recommendation**, AAAI 2026, DOI 10.1609/aaai.v40i17.38491. SAF introduces weighted HSIC to decorrelate irrelevant features, learns sample weights, uses adaptive feature/expert gates, and reweights the MDMT loss. It is a very close MDMT reference for stable feature learning and sample-level training weighting. The paper links a public extended implementation: https://github.com/q1179897215/SAF. DT-PGNA differs by generating an input-level adversarial view from target-logit sensitivity and constraining non-target logits, without a new gate/fusion architecture or learned sample-weight estimator. This difference must be substantiated, not asserted as a broad first.
2. **M3oE**, SIGIR 2024, DOI 10.1145/3626772.3657686. This is the fixed architecture platform, not the proposed contribution.
3. **MMFI — Multi-Scenario and Multi-Task Aware Feature Interaction**, ACM TKDD 2024, DOI 10.1145/3651312. It conditions feature-interaction importance on scenario-task pairs, but changes the model feature-interaction module. A public implementation exists: https://github.com/EnnengYang/MMFI. It is adjacent to domain-task conditioning, not the same training intervention.
4. **PMTRec — Personalized Multi-task Training for Recommender System**, arXiv:2407.21364 (2024 preprint). It uses personalized task weights and gradient magnitude/focusing modules. Its final publication venue and official code status were not confirmed in this search; do not treat it as a verified published baseline without checking.

## 2. Closest general MTL / feature perturbation work

1. **DTME-MTL — Resolving Token-Space Gradient Conflicts**, ICCV 2025, DOI 10.1109/ICCV51701.2025.00277. It detects gradient conflicts in token space and applies token modulation/expansion. This is the closest conceptual neighbor to feature-space gradient manipulation. An official public implementation exists: https://github.com/wooseong97/DTME-MTL. DT-PGNA's proposed distinction is per-domain-task positive/negative ranking-pair hardening with explicit non-target-logit preservation and target-only BCE, not token conflict classification or token expansion. More specifically, DTME's range/null spaces are derived from an SVD of the shared-token covariance, whereas DT-PGNA projects each sample's target-logit sensitivity into the nullspace of the other-task logit Jacobian. This is a concrete mathematical distinction, but the broader idea of task-gradient projection in representation space is already established; compare full code and experiments rather than claiming projection itself as new.
2. **CROSSGRAD — Generalizing Across Domains via Cross-Gradient Training**, ICLR 2018, arXiv:1804.10745. It uses domain-guided cross-gradient input perturbations and trains on augmented inputs. An official public code repository is linked by the paper: https://github.com/vihari/crossgrad. This establishes that gradient-guided input augmentation itself is not new. The contribution cannot be generic gradient perturbation for generalization; it must be the MDMT task-output constraint and ranking-pair construction.
3. **SALARec — Dual-Alignment Contrastive Learning with Preference-Aware Adversarial Augmentation for Sequential Recommendation**, Expert Systems with Applications, published 1 March 2026, DOI 10.1016/j.eswa.2025.130070. It uses preference-aware adversarial perturbations of item embeddings to create semantically meaningful sequential views. This is a close recommendation-side augmentation neighbor. An official public code repository was not confirmed in this search. DT-PGNA differs in its multi-domain multi-task setup, explicit preservation of non-target task logits, and target-only supervised BCE rather than sequential contrastive alignment.
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

The current search identifies relevant adjacent work but does not establish exhaustive novelty. SAF, DTME-MTL, MMFI and CROSSGRAD have public code links; SALARec and PMTRec code/publication status remain unverified here. Before paper submission, compare full PDFs and source implementations, especially SAF, DTME-MTL and SALARec, and verify whether any contains a mathematically equivalent output-preserving projection. If an equivalent method exists, narrow the contribution to recommendation-specific pair construction and empirical mechanism, or stop claiming method-level novelty. Similar work is not an automatic reason to discard the route; actual mechanism overlap determines the decision.


## 5. Main-task literature refresh (2026-11-15)

The following primary sources sharpen the novelty boundary. They do not establish that DT-PGNA is equivalent to any one of them.

- **D3: A Methodological Exploration of Domain Division, Modeling, and Balance in Multi-Domain Recommendations**, AAAI 2024. It explicitly studies domain modeling and domain balance, including optimization-level loss adaptation. It is relevant prior art for domain-aware training/optimization; DT-PGNA must not claim domain-aware optimization as a broad first.
- **Measure Domain's Gap: A Similar Domain Selection Principle for Multi-Domain Recommendation**, KDD 2025. It studies domain similarity and transfer selection. It is relevant to domain-conditional recommendation but focuses on cross-domain transfer selection, not per-example target-logit nullspace perturbation.
- **Stable and Adaptive Fusion for Multi-domain Multi-task Recommendation (SAF)**, AAAI 2026. It uses weighted HSIC, learned sample weights and adaptive feature/expert gating. This is the closest recent MDMT paper in the current search for stable feature learning and sample reweighting. DT-PGNA differs in the training-time adversarial view and explicit non-target-logit Jacobian constraint; do not overstate the gap without full-paper comparison.
- **MMFI**, ACM TKDD 2024, conditions feature-interaction importance on scenario-task pairs. It changes feature-interaction modeling, not the proposed input perturbation.
- **DTME-MTL**, ICCV 2025, manipulates token representations using task-conflict information. This establishes active representation-space MTL optimization prior art. DT-PGNA's possible distinction is the exact per-sample logit-Jacobian nullspace operation and target-only BCE training view.
- **Rep-MTL**, ICCV 2025, uses representation-level task saliency and cross-task saliency alignment. It is a close MTL representation-level neighbor and must be compared against the full mathematical definition.
- **SALARec**, Expert Systems with Applications (2026), uses preference-aware adversarial perturbation of item embeddings for sequential recommendation. It establishes recommendation-side adversarial embedding augmentation as prior art; the candidate distinction is multi-domain multi-task output preservation, not adversarial embeddings alone.
- **CROSSGRAD**, ICLR 2018, uses domain-guided cross-gradient input perturbation. Generic gradient-guided input augmentation is therefore not novel.
- **Virtual Adversarial Training**, TPAMI 2019, provides a general local-smoothness/adversarial perturbation precedent. DT-PGNA should be framed as supervised, target-task-specific augmentation with an explicit other-task output constraint, not as a new general adversarial-training principle.

The search found relevant primary sources but is not an exhaustive novelty search. Before a paper claim, compare the complete SAF, DTME-MTL, Rep-MTL and SALARec papers and code where available. If an equivalent per-sample cross-task logit-nullspace perturbation already exists, narrow the contribution to the specific recommendation task construction and empirical evidence or stop claiming method-level novelty. Similarity alone is not a reason to discard the route.
