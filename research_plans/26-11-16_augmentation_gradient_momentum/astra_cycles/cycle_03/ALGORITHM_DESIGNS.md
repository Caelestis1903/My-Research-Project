# Cycle03 mathematical designs

Only Mechanism A is selected for a fresh formal design. B and C are comparative research options, not authorized arms. No global invention claim is made. All formulas below use the ordinary detached training-gate path; no full-gate probe, feature perturbation, architecture, sampling or label change is selected.

## Shared notation and frozen NQR operator

One natural batch has actual size B (including tails), source layout `x[F,B]`, integral domains `d=x[-1]` in {0,1,2}, and binary labels/probabilities `y,p[B,2]`. Runtime uses FP32. `b_it=BCE(p_it,y_it)` is source probability BCE, with source library behavior at boundaries; no new clamp, logit loss or epsilon. Let `C=sum_it b_it/(2B)`.

For cell c=(d,t), n is its domain row count, n_a its class count, pi_a=n_a/n, and w_c=n/(2B). For each present class a:

`mu_a=mean_{i in c,y=a} p_it`

`v_a=mean_{i in c,y=a}(p_it-mu_a)^2` (population divisor n_a; singleton variance zero)

`bbar_a=mean_{i in c,y=a} b_it`

For two-class cells:

`S_c=pi_1^2*bbar_1 + pi_0^2*bbar_0`

`Q_R,c = 0.5*((1-mu_1+mu_0)^2 + v_1+v_0)`

`A_c=S_c + 2*pi_1*pi_0*Q_R,c`; `A=sum_c w_c*A_c`.

This QR is exactly half the mean squared unit-margin difference over all within-cell positive/negative pairs. No explicit pair matrix is needed. Means and centered variances retain autograd; only numeric logs detach. Counts/labels have no gradient. A one-class cell uses its ordinary mean BCE; an empty cell contributes zero, absent statistics are null. For all-single-class batches the frozen implementation delegates to source CLEAN. These branches avoid invented opposite-class examples.

The mixed loss is `L_lambda=(1-lambda)*C+lambda*A`. On active batches lambda=.25. Coefficient ledger: each present row-task explicit BCE coefficient is `(1-lambda+lambda*pi_y)/(2B)`; opposite-class completion mass is `lambda*w_c*2*pi_1*pi_0`. Same-label BCE plus opposite completion mass sums to w_c, total mass 1, task mass .5 each. This is a coefficient budget, not a guarantee of equal gradient norms or Adam steps. At all scores .5, natural completion has the source BCE derivative with respect to each score; do not infer equality elsewhere. At lambda=0, the frozen source CLEAN branch is required, not a recomputed algebraic approximation. Its logged C=A=L and null moment fields are a branch convention; they are not measurements of the inactive NQR risk.

The C2 probability-margin 1, variance convention, support, reduction order and scalar backward are retained exactly through the frozen C2 core. No new degree of freedom is introduced into this operator.

## A — temporal exposure (selected)

Use integer source epochs e=1,...,50 and one of two immutable schedules:

`lambda_E(e)=.25 * 1[1 <= e <= 10]`

`lambda_L(e)=.25 * 1[41 <= e <= 50]`.

Candidate `NQR-EARLY10` uses lambda_E; control `NQR-LATE10` uses lambda_L. Historical constant `NQR-025` uses .25 throughout; historical CLEAN uses zero. Every new run starts from the source seed initialization. No state is forked from a historical owner and no old IPC is accessed.

For each batch: determine lambda from immutable arm and epoch; create a local inner config with `arm='NQR-025'` and effective lambda; call the frozen `natural_quadratic_ranking_20261116.core.train_step` once. External job/arm identity always remains the temporal arm. Never mutate a shared config, old module global, or old protocol root. One ordinary forward, one scalar backward, one source Adam step and one EMA update per batch, on or off. No additional algorithm RNG, state tensor, teacher, pair sampler or probe. Diagnostics only observe an already-required update.

Source Adam remains lr .01, betas (.9,.999), eps 1e-8, coupled weight decay ML1e-4/KR1e-5, amsgrad/foreach/fused false. Preserve source parameter order and activity. `grad=None` receives no update/weight decay/moment/step increment; an explicit zero gradient is active under source Adam semantics. No moment reset at E10→11 or E40→41. EMA starts equal to online, rho ML .9828205985452511 / KR .995, updates once after every Adam step, source buffers copied, no schedule-boundary reset.

| Dataset | Steps/epoch | Tail B | Total steps per trajectory | Active steps | Off steps | Sum of lambda over updates |
|---|---:|---:|---:|---:|---:|---:|
| MovieLens | 20 | 21927 | 1000 | 200 | 800 | 50 |
| KuaiRand | 49 | 40052 | 2450 | 490 | 1960 | 122.5 |

Every training row has exactly ten active and forty off exposures; no `drop_last`, class resampling, domain balancing, early termination or phase-local loader reset. The private CPU shuffle generator advances continuously, source seed+11000, one randperm per epoch. Its later-epoch batches are not replaced with early batches to manufacture equal opportunities. Log actual class opportunity and active updates so scheduled exposure cannot be confused with realized influence.

Required identities: EARLY10 matches frozen constant NQR numerical states, optimizer hashes and order hashes through E10; LATE10 matches CLEAN through E40. A separate source lambda-zero check starts from a populated Adam/EMA state and must preserve source gradients including None, state, counters and RNG. Diagnosed and undiagnosed updates must be identical. Job/continuation hashes differ by configuration and are not compared as state hashes. Failure is technical PAUSE, never scientific underperformance.

This adds a fixed schedule to the C2 operator. It is locally different from constant NQR only in epochs where lambda differs, and identical on the stated prefixes. It is not a new square-ranking objective. Unlike C2 BRIER, it retains `QR=.5*(gap^2+v0+v1)` rather than `QB=(1-mu1)^2+mu0^2+v0+v1`; unlike MEANGAP it retains both variances. Those ingredient controls are read as evidence, not rerun in Cycle03.

## B — opportunity flattening (deferred)

Consider `C_c=pi_1*bbar_1+pi_0*bbar_0` and, in two-class cells, `A_flat,c=.5*C_c+.5*Q_R,c`; otherwise `A_flat,c=C_c`. Use `L_flat=.75*C+.25*sum_c w_c*A_flat,c`. This fixed support-dependent completion has total coefficient mass 1 and natural domain weights, but opposite-class mass no longer depends on prevalence. At pi0=pi1=.5 it equals natural NQR; away from balance it does not. For two-class cells the explicit BCE coefficient becomes .875/(2B), while each class's pair influence is conditioned on its own count.

This is not C2 BRIER or MEANGAP: the QR function stays intact while its natural pair opportunity and same-class completion are replaced. It is not mere coefficient renaming: the ratio of positive/negative pair contribution to natural BCE changes with prevalence. It loses the source derivative identity at neutral predictions for imbalanced cells. Conditional square AUC risk and its moment representation are established ingredients; no novelty or consistency theorem is asserted here. See [Ying, Wen and Lyu, 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/c52f1bd66cc19d05628bd8bf27af3ad6-Paper.pdf).

This could test prevalence/opportunity effects, but changing class influence and gradient scale together makes the present 20-run timing contrast more discriminating for the new E10 observation. No B arm or compensating scale sweep is requested.

## C — parameter-neighborhood minimization (deferred)

Known SAM objective: `min_theta max_{||epsilon||_2<=r} C(theta+epsilon)`. Its first-order update estimates `g=grad C(theta)`, uses `epsilon=r*g/||g||_2` (epsilon=0 if g=0), computes `g_SAM=grad C(theta+stopgrad(epsilon))`, restores theta, then performs one source Adam step and one fixed EMA update. Weight decay remains the source optimizer operation at base theta. This is a parameter-space perturbation with two training passes, not feature-view BCE, probability moment regularization or a temporal coefficient. This description follows [Foret et al., ICLR 2021](https://arxiv.org/pdf/2010.01412); it is not a new optimizer invention.

Radius r is deliberately not selected because C is deferred, not an implementable fallback arm. C2 had already considered SAM without running it. The new evidence concerns timing and task/calibration separation, not measured neighborhood sharpness; silently fixing or searching a radius would create another unresolved question. A future explicit design would have to bind parameter support and perturbation/null semantics before implementation.

## Exact distinctions from failed local neighbors

The C1 completeness addendum is authoritative for historical DT-PGNA/TGA. Its pilot used up to min(128,npos,nneg) opposite-label pairs per cell; DT-PGNA projected a feature direction `q=g-(h·g/||h||^2)h` with margin/backtracking/drop rules, whereas TGA used example BCE ascent on a subset. The pilot objective `C+.25*(2/M)*sum(view BCE)` had total mass 1.5. Natural repaired NDP used `.75C+.25Lplus`. AXIS-EVEN used `.75C+.125Lplus+.125Lminus=C+.25E`; one-sided retained an odd component, and IID2 changed finite-view sampling. TV controlled a residual gradient update. None is equivalent to the differentiable within-cell QR score moment operator used here. We do not claim those negative results exhaust all perturbation methods, nor disguise one of them under a new name.

Chronological windows here do not rank domains or examples by presumed difficulty. They do not alter source momentum, task weights, architecture gates or sampling, and do not establish that curriculum or momentum is effective. Their purpose is one prespecified falsification of persistent early benefit under unchanged training machinery.
