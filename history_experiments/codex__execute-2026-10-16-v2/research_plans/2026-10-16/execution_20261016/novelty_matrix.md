# Novelty matrix — 15 primary sources

Each row’s full protocol extraction is in novelty_matrix.json / .csv (all 14 required scientific fields). NOT_VERIFIED is a limitation, not a claim of absence. No paper was executed as a baseline in this review.

| ID / primary source | Mechanism already covered | Remaining testable gap |
|---|---|---|
| [L01 M3oE (SIGIR 2024)](https://arxiv.org/html/2404.18465v1) | Adaptive fusion already established | Estimator/state/selection factorial study could be distinct; efficacy remains unproved |
| [L02 AdaTask (AAAI 2023)](https://arxiv.org/html/2211.15055v2) | Task-specific optimizer histories are prior art | Objective-contingent source intervention must differ beyond renamed task/domain states |
| [L03 PCGrad (NeurIPS 2020)](https://arxiv.org/html/2001.06782v3) | Generic conflict projection is occupied | Hold objective fixed and isolate sampler/state without gradient projection |
| [L04 CAGrad (NeurIPS 2021)](https://arxiv.org/html/2110.14048v2) | Conflict-aware gradient aggregation is prior art | Estimator mean-preserving interventions must be separate from CAGrad gradient changes |
| [L05 FAMO (NeurIPS 2023)](https://arxiv.org/html/2306.03792v3) | Dynamic loss weighting and history already occupied | Need distinguish sampler variance control from changing alpha |
| [L06 Adam (ICLR 2015)](https://arxiv.org/pdf/1412.6980) | State propagation is intrinsic optimizer behavior | Empirical MDMT bottleneck and causal downstream link remain to be established |
| [L07 AdamW (ICLR 2019)](https://arxiv.org/html/1711.05101v3) | Decay-state confounding already known | Keep coupled/decoupled decay explicit in fixed-theta intervention |
| [L08 Zhao & Zhang (ICML 2015)](https://proceedings.mlr.press/v37/zhaoa15.pdf) | Same objective/different sampler is classical | Need practical adaptive optimizer and domain structure evidence |
| [L09 Katharopoulos & Fleuret (ICML 2018)](https://proceedings.mlr.press/v80/katharopoulos18a/katharopoulos18a.pdf) | Variance-aware importance sampling is prior art | Adam-specific selected-performance effect is not implied |
| [L10 Lahire (2023)](https://arxiv.org/html/2303.16529v1) | Direct prior art on sampling × adaptive optimizer | A new MDMT contribution needs mechanism/holdout evidence beyond this known interaction |
| [L11 SAF (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/38491/42453) | Sample weighting + adaptive state + MDMT fusion already combined | Need fixed-objective causal separation, not generic combination |
| [L12 PEPNet (KDD 2023)](https://arxiv.org/html/2302.01115v3) | Adaptive feature/parameter sharing is established | Do not relabel personalized gating as estimator-aware mechanism |
| [L13 Cawley & Talbot (JMLR 2010)](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf) | Selection bias itself is prior art | Need paired policy×selection data and unused holdout for a distinct empirical claim |
| [L14 DomainBed (ICLR 2021)](https://arxiv.org/pdf/2007.01434) | Algorithm incomplete without model-selection rule | Selection-aware MDMT interpretation remains setting-specific, not wholly new principle |
| [L15 DoReMi (NeurIPS 2023)](https://arxiv.org/html/2305.10429v2) | Objective and sampling can explicitly be separated in prior work | Do not call all data-mixture tuning same-objective variance control |
