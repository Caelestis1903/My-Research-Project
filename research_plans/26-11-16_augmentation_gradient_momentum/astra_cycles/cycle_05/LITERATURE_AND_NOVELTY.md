# Primary literature and novelty assessment

Primary author/publisher sources were searched and read on2026-11-17. The search covered proper/class-weighted binary losses, prior adjustment, virtual adversarial prediction smoothing, gradient centralization, optimistic updates, and source Adam. No literature result supplies evidence of effectiveness on the present natural ML/KR protocol. Contribution is a falsification-oriented attribution experiment; global algorithmic novelty is unestablished.

## U: conditional averaging of completion coefficients

[Reid and Williamson, Composite Binary Losses, JMLR11:2387–2422,2010](https://www.jmlr.org/papers/v11/reid10a.html) develops proper composite binary losses and their relation to probability estimation and classification calibration. It supports distinguishing these targets; it does not show that the proposed weighting improves AUC. The fixed-weight BCE optimum written in ALGORITHM_DESIGNS is our elementary differentiation, with the shared-network/batch-weight limitations stated there.

[Menon et al., Long-tail learning via logit adjustment, ICLR2021, author institution publication](https://research.google/pubs/long-tail-learning-via-logit-adjustment/) and [authors' paper](https://arxiv.org/abs/2007.07314) explicitly use label-prior logit adjustment, during training or after it. U instead applies one positive scalar to both labels in a domain/task cell, preserving C04's cell mass. No logit adjustment, inverse-frequency balance, rare-class margin or posthoc predictor correction is applied. Prior-dependent weighting and probability shifts have extensive precedent; this plan makes no priority claim for them.

[Yong et al., Gradient Centralization, ECCV2020, authors' paper](https://arxiv.org/abs/2004.01461) centralizes parameter-gradient vectors. U averages a **coefficient field within observed cells before differentiating**; it does not center coordinates, project gradients or impose a zero-mean parameter-gradient constraint. The distinct algebra is the exact removal of `λΣ_c w_c Cov_pi(pi_a,mean_BCE_a)` while retaining every cell's coefficient mass.

Repository contribution: C04 controlled global mass but jointly changed class/cell/task allocation. U supplies the previously missing cell-mass-preserving comparison. W−U identifies the complete class-policy intervention conditional on fixed cell masses; U−S identifies cell/task allocation conditional on fixed global mass. This does not isolate class priors as a population causal factor, identify an optimizer mediator, or establish cell-versus-task attribution separately.

## V: label-free predictive-KL views

[Miyato et al., Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning, authors' paper,2017/revised2018](https://arxiv.org/abs/1704.03976) defines local robustness of the predictive distribution using perturbation directions that do not need observed labels. V deliberately instantiates that established idea at the model's embedding boundary, with source gate detach and an explicit random-direction KL control. It is an adaptation of VAT, not a newly named invention.

NDP-A's direction uses supervised BCE ascent and trains supervised view BCE; C01 averages positive/negative supervised views on an inherited task axis. V uses detached current predictions and KL curvature, not observed-label loss, for view geometry and consistency. Historical EMA02V2 teacher constraints introduce teacher/acceptance semantics absent here. These are mathematical differences, but prior teacher/view failures and VAT's existing status lower V's priority. The clipped probability KL, single power step, source-buffer handling and fixed radius would need exact formal design if ever selected; V is not part of C05 execution.

## G: two-step temporal filtering

[Daskalakis et al., Training GANs with Optimism, ICLR2018, authors' paper](https://arxiv.org/abs/1711.00141) develops optimistic dynamics and an Optimistic Adam variant for game training. G's LEAD/LAG control is a simple finite impulse filter on raw data gradients **before** unmodified source Adam, not a reproduction of the paper's adaptive update formula. Their game-theoretic results do not transfer as convergence or generalization guarantees for shared M3oE BCE.

[Kingma and Ba, Adam: A Method for Stochastic Optimization, ICLR2015](https://arxiv.org/abs/1412.6980) supplies the first/second-moment framework. Both moments in G see the transformed gradient; U simply changes the scalar objective feeding the same source operator. Finite epsilon, coupled decay and time-varying gradient mass prevent inferring exact scale invariance here.

[Zhuang et al., AdaBelief Optimizer, NeurIPS2020, authors' paper](https://arxiv.org/abs/2010.07468) uses gradient deviation from momentum to adapt stepsizes. It is a stronger optimizer alternative, but it replaces the second-moment rule and is outside frozen source Adam. G does not implement AdaBelief, variable EMA, residual second moments or a parameter-space correction.

## Non-equivalence to stopped repository routes

| Prior route | Difference in the proposed candidates | Consequence |
|---|---|---|
| NDP-A / DT-PGNA / TGA | U has no view, teacher or gradient-derived perturbation. V uses predictive KL rather than label-BCE ascent. G filters ordinary W gradients. | No radius/sign rescue of the stopped supervised view version. |
| AXIS-EVEN / IID2 / ISOTROPIC | U is deterministic coefficient averaging; V is a KL-sensitive single view, not even/odd supervised BCE averaging; G has no embedding view. | C01's weak IID2 signal remains counterevidence, not a selected replacement. |
| PAIR / ECCM / NQR | U has no pair risk, ranking term, acceptance subset or conflict memory. It preserves only the observed completion cell masses. | The natural quadratic ranking experiment remains complete; no relabeling as new ranking theory. |
| PCGrad / task projections / PUB | No parameter-gradient projection, utility optimization or solved task weights in U; V changes input geometry; G uses a fixed time filter. | No conflict-removal efficacy assumption. |
| A_MATCH / TV residual momentum / SAD | U has no extra history. G stores the preceding raw W gradient and transforms the input to both Adam moments, rather than injecting a preconditioned/norm-matched view-minus-clean residual after Adam. | Different recurrence, still WEAK efficacy hypothesis; not selected. |
| Adaptive EMA / teacher history | Every candidate retains the same fixed rho and independent selection. | No new readout or temporal teacher introduced to rescue an endpoint. |
| C04 W or S | U preserves W's six masses but replaces class coefficients by their conditional mean; S preserves only global m. | Precisely the new intervention chosen for C05. |

The closest simpler explanations already have full five-seed controls C/S/W. A targeted search did not establish priority for this exact attribution chain; absence from search is not evidence of worldwide novelty. A useful result would be a new empirical separation under a pinned protocol. A null or contradictory result would still rule against specific class-allocation predictions and complete the authorized cycle.
