# N19 DESIGN SPECIFICATION — before implementation/training

Authority: revised `/home/xzf/Code/phase0-3/start.md` (N19). Base N18
14cf09a3865303f5e451636dc6407fd0e2cc1b8c. New branch
`codex/execute-n19-2026-10-18`. Non-training work and review are primary-only;
only mechanical execution is delegated to luna_worker. Review is self-review,
not independent review. Sequence DESIGN → IMPLEMENT → REVIEW → RUN → ANALYZE
is retained. This plan replaces N18's proxy-based cancellation gates.

## Scope, baselines and common protocol

All six directions get trainable candidates, full end-to-end training, explicit
closest conventional controls, mechanism readings and an alternative mechanism.
Names below denote research hypotheses, not established novel algorithms. We must
not claim novelty by renaming standard rank pruning, counterfactual learning,
noise correction, routing or multivariate sampling. Final literature coverage can
invalidate novelty even if performance improves. A first failed candidate does
not terminate its broader direction.

Use the complete fixed cached KR population (2,006,132 train) and ML seed2022
population (187,500 train), with original preprocessing and original val/test.
Hold out 5% of train stratified by domain as train-meta for rank/routing decisions;
every compared arm excludes these records from ordinary training. Random seeds
2022/2023/2024/2025. Main train is one reproducible shuffled full pass/epoch.
KR: 50 epochs, lr=.003. ML and ML-based controlled tasks: 100 epochs, lr=.001.
Batch 40960, Adam(.9,.999), eps1e-8, weight decay1e-5, no early stopping.
These are complete from-scratch networks, not frozen-head studies. Epoch1/5/10,
20/30/40/50/100 and selected/final dynamics are retained.

Validation splits per domain into selection/development halves with stream191019.
Earliest strict maximum six-cell macro selection AUC chooses checkpoint. Inspect
development only for method changes; do not tune on test. After first and
alternative candidates are evaluated on development, freeze all method choices
and test reads. Report selected and final test once for all finished arms. Test
has historical reuse: exploratory evidence, not untouched confirmation.
Cell AUC/BCE, macro, train/test-population weighted cell metrics, examples,
steps, parameters/deployable budget, peak memory and time are mandatory.

Reference architectures: fixed-fusion M3oE, MMoE, PLE, learned-fusion M3oE under
the SAME Sample objective (original_sample), plus literal upstream-protocol
M3oE reference (original): learned fusion, Equal sum, original two-Adam behavior
including main optimizer overlap caused by source `~bool`, and epoch extra
training-batch architecture update. Do not silently repair that source behavior.
The original-protocol result is not a controlled estimate of fusion alone.
Fixed fusion remains sigmoid(1), never implicitly restored. All switches are
explicit independent arms. Relevant baseline papers remain attributed faithfully;
we do not call a partial method full SAF/AutoIFS/VAO/EMAIS.

## P1: task-compatible stability under a conflicting nuisance

Controlled ML-derived population: Y0 is native rating>3. Add categorical Z,
P(Z=Y0|train E=0)=.95, E=1=.75; E is independent row randomization fixed across
model seeds. Domain modifies neither its meaning nor task identity. Set synthetic
Y1~Bernoulli(.2+.6Z). Thus Z is an unstable proxy for task0 but a stable causal
predictor for task1. Validation/development uses P(Z=Y0)=.1, test=.05, with
independent fixed population randomness. Clearly label synthetic task1 and shift;
this is not native business evidence. Keep full support, preserving both task
labels and domain sizes. No inference uses hidden target labels to create Z
except the declared data-generating simulator.

Candidate A task_cf: flip Z only in task0's loss-compatible intervention set,
add lambda=1 probability consistency (p0(X)−p0(flipZ X))² to ordinary BCE.
Candidate B task_dro: optimize .5[max(BCE0(X),BCE0(flipZ X))+BCE1(X)] per row;
this changes the risk envelope, not a renamed consistency coefficient.
Controls: ERM; common flip augmentation (destroys task1 relation, negative control);
task-conditional feature masking (task0 Z=OOV, task1 original), a strong known-rule
solution; task0 environment risk variance (V-REx style); original_sample.
Train all seven arms × four seeds to full budget. Compare worst-domain/task OOD
risk and preserved task1 information. If task masking/ordinary robustness matches
or beats candidates, reject method novelty, not the existence of the problem.

## P2: cross-axis measurement-channel learning

ML native nested labels are latent clean truth. Observed labels follow independent
class-dependent flips per domain/task, a_dt=.35 sigmoid(A_d+B_t),
b_dt=.35 sigmoid(C_d+F_t), with A=(-1,.2,.8), B=(-.3,.5),
C=(.5,-.5,.2), F=(.4,-.4). A fixed random 2% training-row audit reveals both
clean labels. All arms have the same noisy labels, audit and validation labels.
Unobserved clean train labels are forbidden in optimization outside the audit.
This is a controlled measurement experiment, not evidence of natural MNAR.

Candidate A axis_channel learns clean posteriors and additive-logit channel rates
across both axes using noisy BCE plus audit clean BCE. Candidate B joint_axis
uses a three-state nested clean-label distribution (00,10,11), and exact JOINT
noisy-pair likelihood, sharing channel parameters; it exploits co-observation
information rather than merely changing scalar weights. Controls: naive observed
training (audit overwrites noisy entries), audit-only, conventional independent
cell forward correction, nested independent-cell correction, clean-label oracle
(extra-information upper reference), and original_sample on observed data.
Train all eight × four seeds. Read clean-cell AUC/loss and learned vs true rates;
compare same-information independent and joint likelihood to distinguish ordinary
measurement denoising, parameter-sharing variance reduction and genuine new
identification. If needed, non-additive channel perturbation is the discriminating
extension; it tests the assumption, not additional positive-result hunting.

## P3: protect cell risk by allocating dominant STAR capacity

Replace the three dominant full slot matrices by U_d diag(m_d) V_d, maximum
rank32. First ten epochs all ranks train. Then total active rank is exactly48,
with >=4/domain. Uniform baseline takes16/domain; protected allocation scores
rank removal with signed mask derivatives for both task risks on train-meta,
and greedily preserves atoms with worst-cell deletion cost. Task gradients are
kept separate; do not identify risk with domain size. Same supernet/warmup/data
and inference parameter budget for both. No teacher or test labels are required.
Report physically exportable active factors, not all masked supernet parameters
as a fictional deployment saving. Low rank acts before STAR nonlinearities on
dominant parameters, so N18's extra-linear-cell-head equivalence does not apply.

Alternative mechanism cell_codebook: use a global rank47 input/output dictionary
with distinct (d,t) coefficients, generating task-dependent slot matrices before
the shared nonlinear STAR map. This reuses capacity across cells rather than
redistributing independent domain ranks. It fits below the uniform active-factor
budget (47*(in+out+6) <= 48*(in+out)); it has two task STAR evaluations and must
report actual compute. Include a domain-only shared dictionary control to isolate
the task interaction. Train primary/controls on BOTH KR/ML, alternatives after
development analysis. Do not conclude neural infeasibility from optimization.

## P4: interactions that change actual expert training

Candidate A interaction: domain-source mixture logits for each (d,t,k) equal
log baseline mixing + a tensor double-centered over destination d,t. Parameters
therefore encode only zero-marginal destination interactions, not row/column
changes. Learn them jointly with the ENTIRE M3oE via BCE. Compare fixed baseline,
domain-only source mixture and unconstrained full cell source mixture. Report
parameter/cell behavior and time, not only a frozen gradient tensor.
Alternative reliability_interaction updates an interaction confidence from
independent train-meta row halves: retain/shrink a source interaction only when
its finite coefficient perturbation has reproducible risk direction; include
confidence-off ablation. Implement before running that alternative, with a
separate design/review addendum. Two mechanism families (static representation
versus evidence-gated optimization) are assessed on actual training, not cancelled
by N18's 3×2 fixed-point noise.

## P5: finite function regret versus coordinate proxy/validation utility

Candidate A regret_router: each source domain expert defines a finite candidate
function through the same current cell tower; score its actual BCE with the
queried training label, and distill temperature-.1 softmin regret into the cell
source router (KL weight .1, detached regret targets). Main BCE trains all network
weights. This targets finite nonlinear output differences rather than gate
names or infinitesimal validation cosine. Source permutations permute outputs
and router targets together. Contemporary counterfactual routing already covers
much of this principle: no independent novelty claim without an MDMT difference.
Strong control utility_router: same full source mixing architecture, epoch-wise
train-meta risk utility updates of router parameters; ordinary full-mix router
is the capacity control. Compare all on both datasets.
Alternative coalition_router compares finite source LEAVE-ONE-OUT mixture
functions and assigns regret targets to conditional marginal contributions;
it differs from individual-source competition and tests synergistic sources.
Both are trained fully; source contribution/routing diagnostics and topological
symmetry do not substitute for selected model performance.

## P6: nonseparable task-observation decisions, not static row IS

ML controlled stochastic label availability/execution: mean label budget1.1 per
row/visit; both marginal inclusion probabilities pi=.55. Joint two-task masks
have r=P(both) in [.1,.55], so same marginals/mean field can have different
covariance and shared-row execution patterns. This is a per-visit label/execution
simulator, not real annotation savings (repeated native labels could be cached).
Use exact mask/pi risk correction. Unknown labels are zero-filled BEFORE loss
and never used to score proposals; joint covariance uses only jointly revealed
labels. All policies share row batches and fixed objective.

Candidate A copula chooses each domain's r endpoint using an EMA of observed
same-row task derivative inner products at the shared representation (IPW of
joint observations); this changes joint dependence while row q and task marginals
stay fixed. Controls: independent masks r=pi²; positive coupling r=.55;
antithetic lower bound r=.1; all-label reference (larger label budget).
Candidate B copula_cv adds an EMA prediction control variate:
Ltilde=sum_t [BCE(p_t,pEMA_t)+m_t/pi*(BCE(p_t,y_t)−BCE(p_t,pEMA_t))]/2.
Its conditional mean is the full risk gradient; it changes estimator values,
not just IS probability. Include independent_cv to isolate this conventional
control-variate component. EMA=.99, no train-label leakage in teacher.
Train six policies plus full reference × four seeds. Explicitly compare known
multivariate design/control-variate literature. If the entire mechanism reduces
to classical joint design/DR, report that coverage; do not call it a novel MDMT
sampler merely because training is positive.

## Full execution and improvement rules

Baseline/primary matrix is fixed before results. All four seeds complete their
entire budgets irrespective of proxy thresholds or a single negative seed.
First analyze DEVELOPMENT results; implement each stated structurally different
alternative and relevant ablation before stopping a still-viable direction.
Add discriminating cross-dataset/optimizer/assumption extensions for actual
positive methods or unresolved scope, without unlimited grids/seeds. Literature
coverage and exact theoretical impossibility may terminate a SPECIFIC proposal,
but all six directions must have actual end-to-end attempts and scoped records.
Final decisions use paired seed differences and intervals plus cell protection,
cost, N, baseline, task and stage, with STRONG/MEDIUM/WEAK evidence. Four seeds
do not provide precise distribution-free significance; do not chase p<.05.

Outputs in `research_plans/2026-10-18/N19/`; large full states in independent
`/home_new/xzf_n19_artifacts`. Preserve N18 and all negative attempts. Finish
all queued/triggered work, code/results/reports push and main PR; do not mark
COMPLETE at plan, proxy, partial-run or first-candidate stage.

## Chronological design amendments

Numbered addenda preserve reasons and timing:01 spectral canonicalization for
paired rank controls;02 replicated finite-risk confidence;03 conventional audited
fixed channel;04 rank selection after deployment-budget pruning;05 nested-axis
composite likelihood isolation;06 axis-additive routing control;07 P1 MMoE
architecture falsification after complete primary development results;08 frozen
audit-profile calibration;09 residual-field coupling and fixed augmented control;
10 non-additive channel assumption stress;11 active-capacity AdamW discrimination
after numerical branch attenuation was measured;12 fixed-uniform-target routing
after regret routes became nearly uniform. These
amendments do not cancel any original run. Actual commands and tags in the
reloadable queue manifest are the execution authority for this amended design.
