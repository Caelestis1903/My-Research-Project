# N19 primary-source literature update

Consulted2026-10-18; builds on N18's audit rather than replacing it. These are
coverage judgments about candidate ideas, not claims of reproducing every paper.

| Direction | Closest primary evidence | Implication for N19 |
|---|---|---|
|P1|[SAF, AAAI2026](https://ojs.aaai.org/index.php/AAAI/article/view/38491): feature decorrelation, sample reweighting and adaptive feature/expert fusion in MDMT.|Spurious-correlation robustness and adaptive fusion are covered; N19 must isolate the cost of applying an inappropriate invariance rule to another task. We do not label our counterfactual controls a SAF reproduction.|
|P1|[Domain Faithfulness through Counterfactually Robust Learning, CLeaR2026](https://proceedings.mlr.press/v323/balashankar26a.html).|Counterfactual robustness and domain-sensitive rules are established neighbors. Controlled task compatibility may be useful evidence, but consistency/DRO alone is not a new principle.|
|P1|[V-REx, ICML2021](https://proceedings.mlr.press/v139/krueger21a.html).|Environment-risk variance is a conventional comparator. Our task0-only variance penalty is a declared task-aware variant.|
|P2|[Patrini et al., loss correction](https://arxiv.org/abs/1609.03683).|Transition-based forward correction is standard. Axis sharing and nested co-observation must outperform fair anchored controls; denoising itself is not novel.|
|P2|[Learned Task Relationships in Multi-task Models, Google Research/RecSys2026](https://research.google/pubs/learned-task-relationships-in-multi-task-models/).|Joint-label modeling and cross-task transfer are already used in recommendation. N19's possible distinction is audited cross-domain measurement channels, not merely a joint label likelihood. The author page supports abstract-level coverage only.|
|P3|[AdaLoRA](https://arxiv.org/abs/2303.10512).|Adaptive allocation of low-rank budget and singular components is established. N19 cannot call task-sensitive rank pruning a new family.|
|P3|[Not All Ranks Are Equal](https://arxiv.org/html/2609.22237).|The paper scores spectral components and globally allocates a merging budget from data-free utility/interference. N19 instead prunes trainable STAR modulation using cell risks, but global spectral allocation itself is covered; any distinction must carry empirical value.|
|P3|[Multi-Task Dense Prediction via Mixture of Low-Rank Experts, CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Multi-Task_Dense_Prediction_via_Mixture_of_Low-Rank_Experts_CVPR_2024_paper.pdf).|Sharing low-rank experts across tasks has precedent; cell dictionaries need an explicit advantage beyond efficient factorization.|
|P3|[Factorized Tensor Networks](https://arxiv.org/abs/2310.06124).|Task/domain low-rank factors in shared models are established; its frozen-backbone setting differs from N19's fully trained network. That setting difference alone does not prove originality.|
|P4|[AutoIFS](https://arxiv.org/html/2512.13396v1), plus original[ M3oE](https://arxiv.org/html/2404.18465v3).|Learning cross-domain/task sharing is established. N19 isolates a centered logit interaction subspace and replicated finite-risk gating; an ANOVA decomposition or a confidence name alone is insufficient.|
|P5|[When Are Experts Misrouted?](https://arxiv.org/html/2605.07260).|Finite alternative-routing losses and regret-weighted router optimization have direct prior coverage. Its language-model, frozen-expert setting differs from full MDMT training, but N19 must demonstrate a meaningful decision/cost advantage rather than repackage the interpretation.|
|P5|[Equifinality in Mixture of Experts](https://arxiv.org/html/2604.14419).|Routing topology alone is insufficient evidence of model quality. Capacity and measured compute matter; utility and ordinary-routing controls are mandatory.|
|P5|[VAO](https://arxiv.org/html/2510.07760v2).|Validation-utility allocation is a close conventional explanation. Our utility_router is a specified train-meta gradient control, not a full VAO reproduction.|
|P6|[ARMS](https://arxiv.org/abs/2105.14141),[ CARMS](https://arxiv.org/abs/2110.14002).|Copula/antithetic variance reduction predates this study, although these papers concern discrete stochastic gradient estimators rather than exactly this supervised task-observation field.|
|P6|[Cost-Effective Multitask Active Learning in Wearable Sensor Systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902699/).|Multitask label-budget selection is also not new. N19 evaluates equal-marginal joint coupling and prediction control variates under a declared per-visit simulator, with limited annotation interpretation.|

Searches also located2026 topology/low-rank recommendation papers. They reinforce
that efficient conditional parameterization is a crowded area; an abstract-only
match does not justify a claim of exact equivalence or a fabricated reproduction.
No contribution is accepted on terminology. Performance, mechanism discrimination,
cost and scope must all be reported after the complete training matrix.
