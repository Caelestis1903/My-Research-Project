# N18 DESIGN SPECIFICATION — frozen before implementation

Base: main 9deed97122a5b380eafa9783b98d23db527b16f5. Historical code:
`codex/execute-2026-10-17-run@5550afbc57f76e0b94c5783db16f4c1796933541`.
Execution branch: `codex/execute-2026-10-18`. Seeds: 2022–2025.

## Authority and stages

The latest user expressly invokes `/home/xzf/Code/phase0-3/start.md`.
Its principle 11 prohibits non-training subagents and requires `luna_worker`
for mechanical training execution. Therefore DESIGN, IMPLEMENT, REVIEW and
ANALYZE are distinct primary-agent stages; REVIEW is explicitly self-review,
not independent review. This supersedes the earlier delegated-stage workflow.
Only the training runner may be delegated. No algorithm implementation precedes
this specification. No substantive run precedes implementation report/review.

## Fixed scientific conditions

Reuse V4final KR Adam SN E10 checkpoints for all four seeds, chosen by ID/epoch
before new measurements. These are fixed-stage diagnostics, not newly selected
models. Population seed 2022 is fixed. Forward uses StageS1M3OEAdapter, fixed
sigmoid(1) fusion, detached routing inputs, no DBO, balancing, augmentation or
DARTS. There is no trunk optimizer in this study. Parent training used one Adam
and Sample objective J=E_row mean_task BCE. No new claim about M3oE-original.

Existing validation is split deterministically per domain into disjoint
selection/diagnostic halves using a new fixed stream 181018. No historical
test set is accessed by new code. These validation records are reused historical
data, so readouts are exploratory, never untouched confirmation. Publish cell
AUC/logloss, macro and train-population-weighted cell metrics, sample sizes,
parameters, measured seconds. Weighted cell AUC is not pooled AUC.

## P1 — support, preservation and common weights

Compute exact finite-support 2-domain × 2-task moment feasibility. Compare
pooled cancellation, per-domain moment balance, selective preservation, and
common versus separate task weights; include diagonal support and a full-support
control. A positive residual with exact support is structural for this declared
moment family. It is not a theorem about general HSIC training or SAF performance.
Audit raw schemas for legitimate environments/nuisances before any GPU training.
A: genuinely new joint restriction and valid environment -> consider stable pilot;
B: ordinary LP/selective conditioning already solves it, or environment absent ->
stop new algorithm/current application. SAF/DVD are literature comparators, not
faithfully implemented training baselines. No paper-failure claim without them.

## P2 — identification and controlled missingness

Construct connected 2×2 same-observation worlds with reversed missing-cell
decisions. Compute exact binary-label nested bounds and same-P paired BCE risk
difference bounds with LP. On ML train labels, generate one MAR and one MNAR
mask for each seed (MAR .4; MNAR .7 positive/.2 negative). Oracle labels are
excluded from bounds. Compare unconstrained bounds, observed label logic,
ordinary LP using identical information, and oracle truth. Compare equal-size
random versus task0-stratified audit labels, design-unbiased estimates and SE;
do not treat sampled audit moments as exact population constraints. Logic can
infer unobserved Y1=0 where Y0=0 without buying extra labels; quantify this gain.
A: additional identifiable quantity beyond same-information ordinary fusion ->
advance; B: ordinary fusion gives identical bounds or natural bridge absent ->
stop simple novelty claim, preserve explicit data blocker. No MAR-to-MNAR claim.

## P3 — budget feasibility and frozen-head pilot

First verify heterogeneous 2×2 interval feasibility against exact LP, and a
small-grid difference-constraint/negative-cycle certificate. Standard mathematics
is labeled standard. Then inspect real module parameter costs and fit compressed
linear logit heads on frozen 64-dimensional M3oE tower-input features (bias makes
K=65). Trunk, all routing and experts remain identical across arms. Teacher is
the same frozen E10 checkpoint; no cherry-picked reference ensemble.

Use 2048 distinct train rows/domain (total 6144), 512 selection and 512 diagnostic
rows/domain. Labels only evaluate; teacher logits fit distillation. Minimize
equal-cell mean squared logit error with ridge 1e-6 on standardized features,
including bias. Models: flat full heads (6K), additive row+column heads (4K),
additive plus one cell residual (5K), and the algebraically equivalent centered
interaction repair (5K). Choose residual cell using training distillation loss
only. Solve both parameterizations in an identical orthonormal cell-span basis,
so regularization does not manufacture a difference. Include rank-4 SVD of full
heads (4(6+K)=284 stored scalars, less than 5K=325) as a generic compression
control, explicitly not globally optimal reduced-rank regression for heterogeneous
features. Report actual stored coefficients (orthonormal basis + reduced weights)
and common trunk costs, not just nominal degrees of freedom. SVD stores factors.

The repair span equals ordinary extra-cell-head span: this is an especially strong
falsifier of a claimed distinctive algorithm. Train/fitting is an actual new pilot,
but only frozen-head distillation, not end-to-end M3oE retraining. Four fixed seeds;
no epoch search. Protection screen: each diagnostic cell AUC loss ≤.003 and
logloss increase ≤.003 relative to frozen teacher. Screen only, not population
guarantee. A: repair exceeds equal-cost conventional head and meets protection
on every seed with nontrivial whole-model saving -> advance; B: algebraic
equivalence, protection failure or negligible saving -> stop this repair method.
A failed fit does not certify neural model-class infeasibility.

## P4 — interaction measurement

For each fixed checkpoint measure six full-parameter BCE gradients on two disjoint
train batches of 256 rows/domain and a third diagnostic batch of 256/domain.
Use unweighted cell means, cell ordering (d,t). Decompose into mean, row, column,
zero-marginal interaction. Predict C from mean(A,B), or from its additive projection.
Compare error and sign of inner products of source update directions with each
heldout cell gradient; abstain when |cosine|<.01. Compare all coordinates and shared
C block. Record wall time and number of gradient calls. D=3,T=2 has interaction
dimension 2, and full cell matrix rank at most 2, so rank-2 recovery is no compression.
A: stable interaction changes heldout decisions AND a lower-cost structural
estimator saves ≥25% total measurement cost -> advance; B: no stability or no
cost advantage -> stop current-scale measurement method, not all interaction theory.

## P5 — proxy symmetry and decision consequence

Numerically implement D3 published product fusion with swapped softmax outputs,
including bias, and compute prediction equality, specificity-weight change and
change in the weighted-loss gradient. This tests published mathematics, not
unavailable official implementation. Separately use fixed-checkpoint source/target
gradients from P4 to measure potential sign disagreements in source actions;
constructed gates must never be called trained D3 gates. Same-action validation
utility is a conventional comparison. A: real persistent decision regret beyond
existing utility methods -> advance; B: symmetry only/standard utility resolves/
no faithful D3 artifacts -> retain narrow audit, stop standalone method.

## P6 — fixed-point joint sampling

All records have both labels and same execution graph. Check whether any actual
label/execution constraints distinguish this data from classical vector IS.
On each seed use two disjoint train samples of 48 rows/domain (design and heldout).
Compute individual-row mean-task full gradients. Save gradient squared norms,
six projections onto normalized heldout cell gradients, and logit-residual squared
norm proxies. Proposals over domain: natural p; norm IS p sqrt(E||g||²);
logit-gradient EMAIS-style proxy p sqrt(E||p-y||²/T²); generalized direction IS
p sqrt(E mean_c(h_c·g)²); MIS mixture of natural/norm/two task-residual proposals
with simplex weights fitted on design trace. Minimum mixture .02*p for all learned
proposals. Heldout population is the finite stratified gradient population, and
all estimators have mean sum p_d mean_d g; covariance trace and six projected
variances use exact finite-population second moments for one IID draw, scaled
to B=40960 for illustration. Mean correction is mandatory. Report ratios and
proposal fitting/scoring costs. This is not a full online EMAIS/OMIS reproduction.
A: ≥10% mean projected variance benefit vs norm IS on ≥3/4 seeds, trace penalty
≤10%, ≥25% estimated net budget benefit AND a real nonclassical joint constraint ->
one short sampler training design; B: otherwise stop new MDMT sampler. Even a
positive generalized IS oracle is not novel and does not imply AUC improvement.

## Scope and completion

No tuning on heldout outcomes; no seeds/epochs added to chase success. A change
that can alter conclusions requires a recorded amendment before rerun. Six
directions each receive derivation, raw measurements, nearest-paper differences,
negative/conflicting evidence, STRONG/MEDIUM/WEAK strength scoped to exact claim,
GO/CONDITIONAL/NO-GO/DATA-BLOCKED and a minimal next discriminating experiment.
Final report must discuss size vs difficulty, task/domain/baseline/stage/seed
dependence and possible small-domain gain/large-domain loss without presuming it.
Complete all triggered work, commit/push execution branch and open PR for main.
