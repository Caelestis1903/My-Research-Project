# N19 observed research questions and discriminating future studies

These are questions exposed by development results, not claims of algorithmic
novelty, and not additional unexecuted requirements of the declared N19 matrix.
Final test consistency belongs in FINAL_REPORT. Independent contribution still
requires new evidence and a defensible distinction from the cited literature.

|Observed question|Evidence and competing explanation|Best discriminating next study|
|---|---|---|
|Can a capacity allocator be evaluated after its modulation path becomes inactive?|KR Adam finite slot removal is exactly prediction-invariant by epoch5, even in dense baseline; AdamW restores activity. Skip-path dominance, coupled decay and training-step count compete.|Vary slot/shared decay only while fixing non-slot updates; independently cross domain N with update count and use trainable no-slot controls. Do not infer a size effect from native domains.|
|Does task-conditioned storage affect ranking and calibration differently over training?|KR AdamW task dictionary improves final AUC over domain dictionary but selected/loss evidence differs; ML original Adam effect is weak. Two task STAR paths also change compute.|Use the completed dataset/optimizer transfer first. A new study would prespecify deployment stage and calibration utility, with compute-matched independently retrained alternatives.|
|Can audit sharing retain efficiency without assuming an exactly additive corruption law?|Fixed-draw additive axis calibration improves loss; non-additive truth reverses it. Audit class support and channel misspecification vary jointly.|Independent audit draws, crossed class-specific audit support and non-additivity; compare conventional shrinkage with an identifiable estimator. Four optimization seeds are not audit replication.|
|Can stability rules be learned without a known task-nuisance map?|Task-aware robustness beats ERM in the controlled shift, but direct known-rule masking explains most gain and task1 can decline.|Prespecified independent interventions with hidden task compatibility, fresh shift populations and a rule-discovery baseline. Do not claim natural causal invariance from known Z.|
|Does replicated interaction evidence do more than alter optimization geometry?|Positive PCP confidence has the same feasible interaction family; ML selected benefit is tiny and KR harms strong controls.|First require a useful endpoint benefit; only then compare heterogeneous confidence to global matched shrinkage using the same probes. Current negative gating does not justify a repair grid.|
|What decision does finite expert credit actually improve?|Finite single-source choices differ from a local coordinate preference, but a chosen source can lose to the current nonlinear mixture. Targets often approach uniform.|Compare actual matched finite mixture interventions and the full utility update on independent meta populations. Avoid equating a coordinate probe with the optimizer or a better one-hot choice with mixture utility.|
|When is a repeated label-observation budget a real cost constraint?|N19 masks spend about1.1 labels per row/visit but labels are static/cacheable and computation remains dense; residual adaptivity mostly chooses the static antithetic endpoint.|A prospective persistent-observation/caching protocol with actual annotation and execution costs, full-parameter residual covariance and appropriate static controls.|
|Why can selected and final ranking gains disagree with loss and weighting?|Several candidate/control contrasts reverse by stage or metric; native populations and cell difficulties differ.|Prespecify real deployment utility and update budget on a fresh population, report calibration and cell risks, and independently manipulate N. No post-test choice of the most favorable checkpoint or metric.|

No N19 result establishes that small domains are difficult, that large domains
necessarily lose, that curriculum/momentum is useful, or that M3OE necessarily
fails on hard domains. The native three-domain comparisons are descriptive.
