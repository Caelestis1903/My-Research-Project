# N19 mathematical claims, counterclaims and interpretation limits

These statements precede outcome analysis. They define what the mechanisms can
test, rather than assert success from an attractive formula.

## P1: compatible invariance

Write C for original covariates and Z for the controlled nuisance. If a universal
representation discards Z, the best achievable task1 log risk is H(Y1|C), whereas
retaining it permits H(Y1|C,Z). Their difference I(Y1;Z|C)>=0 is an information
cost whenever Z retains predictive information after conditioning on C. It need
not be large: C may itself predict Z. This is a population statement, not a proof
that a particular shared network reaches either optimum. Task-conditioned
invariance only requires p0(C,Z)=p0(C,flipZ); p1 may still depend on Z. Known-rule
feature masking achieves this directly, so regularization beating ERM is
insufficient for an independent algorithmic contribution. The controlled shift
defines a causal compatibility problem; it does not discover business causes.

## P2: measurement anchors and co-observation

For a cell, eta(x)=a+(1-a-b)p(x). Without audit or restrictions, affine changes
in latent p and compensating channel changes can preserve eta. Thus observed
calibration alone does not identify clean risk. Random paired audits anchor a
and b, subject to finite class-specific sample sizes. Cross-axis additive logits
share statistical strength under a declared channel model, but can bias a
misspecified channel. The nested likelihood sums over states00,10,11:
P(o0,o1|x)=sum_s q_s(x) product_t P(o_t|s_t,d,t).
Here q=(1-p0,p0-p1,p1), p1=p0*r1 ensures support. An audit contributes clean-state
likelihood and observed-given-clean likelihood. Joint likelihood may exploit
information discarded by marginal composite likelihood; it does not prove
generic nonparametric identification. Compare nested_ind to joint_axis to expose
the combination of channel sharing and co-observation, and axis_channel to
ind_channel for sharing alone. These contrasts do NOT fully factorially isolate
every term, so a positive joint-specific claim may require nested axis composite
control before test freeze. audit_fixed checks ordinary anchored correction.

## P3: storage allocation, not guaranteed risk protection

Slot S_d=sum_r m_dr u_dr v_dr^T. Deleting atom r has first-order risk change
-dR_dt/dm_dr; max_t[0,-dR_dt/dm_dr] is a conservative local score for the two
tasks of domain d. It ignores second-order interactions and all post-pruning
optimization. It therefore has no finite-deletion safety guarantee. Canonical
spectral factors before BOTH allocations remove arbitrary scale/rotation of
factor coordinates. Equal rank budget48 means equal active-factor storage, not
equal FLOPs or equal quality. Selection after pruning prevents warmup capacity
from contaminating deployment comparisons. Effective STAR matrices are S_d
Hadamard W_shared; a low-rank SLOT does not force this product to have low matrix
rank. The claim is compressed modulation/storage, not a neural bottleneck theorem.

Codebook S_dt=U diag(c_dt) V reuses dictionary atoms across cells. Its task
dependence is before STAR's nonlinearity and therefore does not generally fold
into a linear output head. This defeats N18's specific head-equivalence objection,
but shared dictionary factorization and conditional parameter generation have
extensive prior art. Independent value must survive domain_codebook and cost.

## P4: identifiable logit interaction subspace

P(v)_dtk=v_dtk-mean_d(v)_tk-mean_t(v)_dk+mean_dt(v)_k.
For each source k this has zero destination domain/task means. Raw18 parameters
represent only6 centered degrees of freedom. Softmax is invariant to a common
source shift, leaving at most4 effective mixture degrees of freedom. This is
not a probability-space zero-margin statement; softmax can create probability
main effects. Full cell mixture has12 effective degrees, domain-only6. Confidence
must be applied as P(c*P(v)) to stay in the same interaction space. Replicated
finite slopes support a heuristic evidence gate; they do not calibrate uncertainty
or establish that the interaction is causally transferable across data regimes.

## P5: finite alternatives and ordinary utility

For finite source functions f_k through the SAME cell tower, q_k proportional
exp(-R(f_k)/tau). KL(q||w) pushes cell-level mixture w toward lower measured
source risk, with detached q. Main risk still updates all functions. This is
batch/cell aggregate credit, not input-specific optimal gating, nor a guarantee
that the mixture improves when each expert improves. Leave-one-out costs instead
score R(f_without_k); high loss under removal favors k. This can detect conditional
contribution but is not a Shapley value and depends on current mixture weights.
Permutation of functions AND source coefficients preserves outputs and targets;
permuting expert modules while keeping domain-anchored prior weights fixed is a
different model, so that would be an invalid invariance test. A new interpretation
alone is insufficient: compare ordinary learned mixtures and train-meta utility.

## P6: covariance design and prediction control variates

At a fixed row and current parameter state let g0,g1 be task gradients. With
marginals pi and joint inclusion r, the unbiased estimator for mean-task risk is
G=(m0*g0+m1*g1)/(2*pi). Its conditional variance trace includes the cross term
(r/pi^2-1)<g0,g1>/2. Thus minimizing this term chooses r=.1 for positive alignment
and r=.55 for negative alignment, at pi=.55. Independent r=.3025 is generally
not optimal for this local criterion. The implementation estimates inner products
at the shared STAR representation, NOT the full parameter gradient; this proxy
need not give the full-gradient optimum. It estimates jointly observed inner
products using inclusion correction, with predictable current masks determined
from previous-step EMA state. Adaptivity therefore does not change the current
conditional mean field. Updating an EMA after observing the batch affects only
subsequent masks.

For label-free predictable teacher b_t=BCE(p_t,pEMA_t), the estimator
b_t+(m_t/pi)(loss_t-b_t) has conditional expectation loss_t, including its student
gradient when teacher predictions are detached. This is a standard prediction
control variate/augmentation principle, not new by itself. Teacher extra forward
cost and dense student execution are measured. The fixed per-visit label budget
does not establish real annotation savings when native labels can be reused.

## Decision discipline

No theorem here establishes empirical improvement, novelty or top-conference
adequacy. Four seeds, two fixed dataset populations and six correlated cells
support descriptive paired uncertainty, not population-wide causal attribution
to domain size. Analyze N, baseline difficulty, task, model, stage and metric
together. A smallest-domain gain with largest-domain decline is a trade-off only
when it actually repeats; do not infer it from a macro/population gap alone.

## Positive replicated confidence changes geometry, not the feasible family

For P4 let P be the orthogonal destination double-centering projection and
C=diag(c), with .25<=c_i<=1 under the specified EMA. On S=range(P),
<v,PCPv>=sum_i c_i v_i^2 >=.25||v||^2. Thus PCP restricted to S is invertible
and its image is S. Frozen positive confidence therefore retains exactly the
same effective interaction-logit family as confidence-off; softmax gauge removal
also leaves the same4 functional degrees. It is a data-dependent optimization
parameterization/regularization mechanism, not a hard interaction-selection rule
or a new larger/smaller sharing hypothesis class. Network weights may compensate
by scaling raw logits. Under fixed C and plain SGD the effective-logit update
contains (PCP)^2; Adam adds coordinate adaptation, so that exact SGD expression
is not asserted for the actual optimizer.

If replicated gating has a meaningful empirical advantage, generic uniform
shrinkage/optimization geometry remains a required competing explanation. No
new control is triggered by this identity alone; all preplanned confidence runs
still complete. The identity limits interpretation even if performance improves.
