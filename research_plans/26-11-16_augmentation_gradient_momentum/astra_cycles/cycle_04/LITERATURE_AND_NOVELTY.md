# Primary literature and bounded novelty assessment

Technical references below were checked through primary publisher/author-paper pages during this research turn. Access date is 2026-11-17 Asia/Shanghai. OpenReview's direct AdamW PDF presented a browser challenge; the authors' arXiv record was used instead. No search snippet is used to assert a new theorem or priority claim.

| Primary reference | Verified relevance | What it does not establish here |
|---|---|---|
| Wei Gao, Rong Jin, Shenghuo Zhu, Zhi-Hua Zhou, *One-Pass AUC Optimization*, ICML 2013, PMLR 28(3):906–914. [Publisher](https://proceedings.mlr.press/v28/gao13.html), [paper](https://proceedings.mlr.press/v28/gao13.pdf). | Squared pair-surrogate AUC optimization and first/second-order sufficient statistics predate these experiments. The paper's linear-score algorithm is not our probability-space M3oE training rule. | Neither the pair-moment identity nor use of quadratic ranking is novel here. Its results do not predict efficacy under this batch, architecture or composite BCE risk. |
| Mark D. Reid, Robert C. Williamson, *Composite Binary Losses*, JMLR 11(83):2387–2422, 2010. [Publisher and paper](https://www.jmlr.org/papers/v11/reid10a.html). | Proper composite losses distinguish probability estimation, links and surrogate properties. This motivates keeping loss/calibration and ranking conclusions separate. | The fixed-class-weight Bayes calculation below is our elementary derivation, not a quoted theorem establishing finite-network ranking invariance. |
| Diederik P. Kingma, Jimmy Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015. [Authors' paper](https://arxiv.org/abs/1412.6980). | Adaptive first/second moments explain why raw-gradient and parameter-update diagnostics differ. | Ideal gradient-rescaling statements do not imply exact invariance for this installed implementation with finite epsilon, coupled L2, time-varying batch mass and different trajectory histories. |
| Ilya Loshchilov, Frank Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019. [Authors' paper](https://arxiv.org/abs/1711.05101). | Distinguishes coupled L2 from decoupled weight decay in adaptive optimizers. This makes the S=mC control scientifically relevant. | It does not authorize switching the frozen source Adam to AdamW or retuning regularization. The present experiment preserves source coupled decay. |
| Arvind Neelakantan et al., *Adding Gradient Noise Improves Learning for Very Deep Networks*, 2015. [Authors' paper](https://arxiv.org/abs/1511.06807). | Direct gradient noise is established prior art and is a mechanism distinct from changing the mean risk. | Their architectures and noise experiments do not establish a benefit for these recommendation tasks, paired-view noise or five-seed selected EMA. |
| Tianhe Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020. [Official proceedings](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html). | Gradient projection to address task interference is established prior art. | The unselected projection of a ranking residual against a pointwise gradient is not claimed identical to PCGrad or globally novel, and interference has not been shown to explain these results. |

## Algebra that motivates, but does not prejudge, the test

For positive and negative scalar scores drawn independently within a cell,

`E[.5(1-p_plus+p_minus)^2] = .5[(1-mu1+mu0)^2+var1+var0]`.

This is elementary expansion; finite full pair enumeration equals the population-variance expression computed from those finite class subsets. C2's frozen source uses probabilities, variance divided by n, and no pair sampling. The completion component `pi1^2 mean_BCE1 + pi0^2 mean_BCE0` assigns a row of class a a multiplier proportional to its observed class prevalence pi_a. It is not inverse-frequency balancing.

For fixed positive class weights q1,q0 and unrestricted prediction at conditional positive probability eta, weighted BCE is minimized at

`p*(eta)=q1 eta / [q1 eta + q0(1-eta)]`.

Its derivative is `q1 q0/[q0+(q1-q0)eta]^2 > 0`. Thus fixed within-cell class weighting preserves the ideal population ordering while changing the probability scale. Here weights vary by batch, the model shares parameters across tasks/domains, Adam includes coupled decay and training is finite. None of those conditions is removed by this derivation. An observed rank change remains possible and must be measured.

## Actual claim of contribution

The prospective contribution is a controlled attribution experiment on an already documented failure pattern: remove NQR's ranking term while holding its completion weights, then control their global BCE mass. The contrast `C -> mC -> W -> W+R` is explicit and falsifiable under fixed source dynamics. It may discover a useful candidate, or exclude a tempting ranking-only explanation. It is not a new AUC theory, new optimizer, new class-balancing method or demonstrated recommendation improvement.

The literature check is targeted, not an exhaustive priority review. **Global novelty is unestablished and is not claimed.** Repository novelty is limited to the presently unrun completion-only and matched-mass interventions plus common-state source-Adam diagnostic decomposition. Prior scale experiments and gradient manipulation studies are retained as counterevidence, not renamed as new algorithms.

No literature result substitutes for the registered primary comparison, six-cell protection, all five paired seeds, full 50 epochs or independent engineering review. Claims must remain scoped to the fixed1 adaptation and source populations until a separately authorized experiment tests transfer.
